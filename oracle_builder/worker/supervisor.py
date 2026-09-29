"""Bounded process-group termination while the parent retains lease control."""
from __future__ import annotations
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


def terminate_process_group(process, *, grace_seconds=2):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


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
            tail = (scratch / 'executor.log').read_text(errors='replace')[-4000:]
            raise RuntimeError(f'Execution child exited with status {process.returncode}: {tail}')
        if not result_path.is_file():
            raise RuntimeError('Execution child did not write its result')
        return json.loads(result_path.read_text())
    finally:
        telemetry_done.set()
        terminate_process_group(process)
        if process.stdout:
            process.stdout.close()
