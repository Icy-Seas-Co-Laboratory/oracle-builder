"""Bounded process-group termination while the parent retains lease control."""
from __future__ import annotations
import gzip
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

from oracle_builder.worker.executor_child import EVENT_PREFIX


_FAILURE_LOG_TAIL_BYTES = 4 * 1024 * 1024
_FAILURE_LOG_LIMIT = 20


def _retain_failure_log(scratch: Path) -> Path | None:
    """Keep a bounded diagnostic tail outside disposable execution scratch."""
    source = scratch / 'executor.log'
    if not source.is_file():
        return None
    failures = scratch.parent / 'failure-logs'
    target = failures / f'{scratch.name}.log.gz'
    temporary = failures / f'.{scratch.name}.{os.getpid()}.tmp'
    try:
        failures.mkdir(mode=0o700, exist_ok=True)
        os.chmod(failures, 0o700)
        with source.open('rb') as log, temporary.open('xb') as output:
            os.chmod(temporary, 0o600)
            log.seek(max(0, source.stat().st_size - _FAILURE_LOG_TAIL_BYTES))
            with gzip.GzipFile(fileobj=output, mode='wb') as compressed:
                compressed.write(log.read(_FAILURE_LOG_TAIL_BYTES))
        os.replace(temporary, target)
    except OSError:
        temporary.unlink(missing_ok=True)
        return None
    try:
        for old in sorted(failures.glob('*.log.gz'), key=lambda path: path.stat().st_mtime, reverse=True)[_FAILURE_LOG_LIMIT:]:
            old.unlink(missing_ok=True)
    except OSError:
        pass
    return target


def _process_group_exists(process_group_id: int) -> bool:
    """Return whether a process group still has a live member."""
    try:
        os.killpg(process_group_id, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def terminate_process_group(process, *, grace_seconds=2, process_group_id: int | None = None):
    """Terminate a session leader and every descendant in its process group.

    ``Popen.poll()`` only describes the leader. A multiprocessing child can
    outlive that leader, so signal the process group even after the direct
    child has exited.
    """
    process_group_id = process.pid if process_group_id is None else process_group_id
    if not _process_group_exists(process_group_id):
        if process.poll() is None:
            # Restricted hosts can deny process-group signals. Preserve the
            # old best-effort leader cleanup there; production workers run on
            # Linux where the group is available.
            try:
                process.terminate()
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except ProcessLookupError:
                    return
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        return
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + max(0.0, float(grace_seconds))
    while time.monotonic() < deadline:
        # Reap a dead leader promptly. A dead leader may remain as a zombie
        # while descendants still need the rest of the grace period.
        process.poll()
        if not _process_group_exists(process_group_id):
            break
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    if _process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def execute_supervised(*, work_unit, inputs, configuration, staging, emit_event, control):
    scratch = staging.parent
    request_path, result_path = scratch / 'executor-request.json', scratch / 'executor-result.json'
    request = {'work_unit': work_unit, 'inputs': {key: str(value) for key,value in inputs.items()},
        'configuration': str(configuration) if configuration else None, 'staging': str(staging),
        'control_file': str(control.control_file), 'result_path': str(result_path)}
    request_path.write_text(json.dumps(request, allow_nan=False))
    os.chmod(request_path, 0o600)
    environment = dict(os.environ)
    # Deployment secrets belong to the parent; execution receives only local
    # materializations. Remove the known worker/operator credentials.
    for key in list(environment):
        if 'TOKEN' in key or 'SECRET' in key or 'PASSWORD' in key:
            environment.pop(key)
    # Descendants of this executor must remain in its process group. The
    # executor uses this marker to avoid creating an isolated calibration
    # session that would escape worker cancellation.
    environment['ORACLE_SUPERVISED_EXECUTION'] = '1'
    process = subprocess.Popen([sys.executable, '-m', 'oracle_builder.worker.executor_child', str(request_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True, env=environment)
    events = queue.Queue(maxsize=256)
    def collect():
        with (scratch / 'executor.log').open('w') as log:
            for line in process.stdout:
                log.write(line)
                if line.startswith(EVENT_PREFIX):
                    try:
                        event = json.loads(line[len(EVENT_PREFIX):])
                        control.progress(event['event'], event.get('data'))
                        events.put_nowait(event)
                    except (ValueError, KeyError, queue.Full):
                        pass
    reader = threading.Thread(target=collect, daemon=True)
    reader.start()
    telemetry_done = threading.Event()
    def send_events():
        while not telemetry_done.is_set():
            try:
                event = events.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                emit_event(event['event'], event['message'], event.get('data'))
            except Exception:
                pass
    sender = threading.Thread(target=send_events, daemon=True)
    sender.start()
    try:
        while process.poll() is None:
            control.check()
            time.sleep(0.1)
        control.check()
        reader.join(timeout=1)
        if process.returncode:
            saved_log = _retain_failure_log(scratch)
            detail = ''
            if result_path.is_file():
                try:
                    failure = json.loads(result_path.read_text()).get('execution_error', {})
                    detail = f"{failure.get('type', '')}: {failure.get('message', '')}".strip(': ')
                except (OSError, ValueError, AttributeError):
                    pass
            if not detail:
                log_path = scratch / 'executor.log'
                if log_path.is_file():
                    with log_path.open('rb') as log:
                        log.seek(max(0, log_path.stat().st_size - 1200))
                        detail = log.read(1200).decode(errors='replace')
            suffix = f' [diagnostic log: {saved_log}]' if saved_log else ''
            raise RuntimeError(f'Execution child exited with status {process.returncode}: {detail[:1400]}{suffix}')
        if not result_path.is_file():
            raise RuntimeError('Execution child did not write its result')
        return json.loads(result_path.read_text())
    finally:
        telemetry_done.set()
        terminate_process_group(process)
        if process.stdout:
            process.stdout.close()
