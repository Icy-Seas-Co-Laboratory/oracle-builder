from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from oracle_builder.worker.pull import TrainingExecutor, _run_batch_calibrator
from oracle_builder.worker.supervisor import terminate_process_group as _terminate_process_group


def _process_state(pid: int) -> str | None:
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True, text=True, check=False,
    )
    state = result.stdout.strip()
    return state or None


def _supports_process_group_signals() -> bool:
    probe = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        os.killpg(probe.pid, 0)
    except PermissionError:
        return False
    finally:
        probe.terminate()
        probe.wait(timeout=2)
    return True


def test_calibration_timeout_kills_its_multiprocessing_descendant(tmp_path, monkeypatch):
    if not _supports_process_group_signals():
        pytest.skip("host does not permit process-group signalling")
    child_pid = tmp_path / "probe.pid"
    code = (
        "import os, pathlib, time\n"
        "pid = os.fork()\n"
        "if pid:\n"
        f"    pathlib.Path({str(child_pid)!r}).write_text(str(pid))\n"
        "    time.sleep(60)\n"
        "else:\n"
        "    time.sleep(60)\n"
    )
    observed = []

    def terminate(process, **kwargs):
        child = int(child_pid.read_text())
        state = _process_state(child)
        observed.append(state)
        assert state is not None and not state.startswith("Z")
        _terminate_process_group(process, **kwargs)

    monkeypatch.setattr("oracle_builder.worker.pull.terminate_process_group", terminate)
    with pytest.raises(subprocess.TimeoutExpired):
        _run_batch_calibrator([sys.executable, "-c", code], timeout=0.1)
    assert child_pid.exists()
    child = int(child_pid.read_text())
    assert observed
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        state = _process_state(child)
        if state is None or state.startswith("Z"):
            break
        time.sleep(0.02)
    state = _process_state(child)
    assert state is None or state.startswith("Z")


def test_supervised_calibration_timeout_leaves_group_cleanup_to_executor(monkeypatch):
    class TimedOutProcess:
        pid = 4321

        def __init__(self):
            self.terminated = False

        def communicate(self, *, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(["tuner"], timeout)
            raise AssertionError("supervised timeout must not wait on descendants")

        def terminate(self):
            self.terminated = True

        def wait(self, *, timeout=None):
            return 1

    captured = {}
    process = TimedOutProcess()
    monkeypatch.setenv("ORACLE_SUPERVISED_EXECUTION", "1")
    monkeypatch.setattr(
        "oracle_builder.worker.pull.subprocess.Popen",
        lambda *_args, **kwargs: captured.setdefault("start_new_session", kwargs["start_new_session"]) or process,
    )

    with pytest.raises(subprocess.TimeoutExpired):
        _run_batch_calibrator(["tuner"], timeout=0.01)

    assert captured["start_new_session"] is False
    assert process.terminated


def test_training_executor_calibrates_on_worker_and_uses_the_resolved_config(tmp_path, monkeypatch):
    configuration = tmp_path / "configuration"
    configuration.mkdir()
    (configuration / "payload").write_text("[data]\nbatch_size = 4\n", encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "payload").write_bytes(b"sqlite")
    staging = tmp_path / "staging"
    staging.mkdir()
    events: list[tuple[str, dict[str, object]]] = []

    def tuned_process(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="ORACLE_BATCH_TUNE_RESULT=" + json.dumps({
                "ready": True, "recommended_batch_size": 24,
                "largest_verified_batch_size": 32, "tuning_strategy": "oom_boundary_fallback",
                "probe_kind": "representative_forward_backward",
            }) + "\n",
        )

    monkeypatch.setattr("oracle_builder.worker.pull._run_batch_calibrator", tuned_process)
    captured: dict[str, str] = {}

    def run_training(request):
        captured["config"] = request.config
        captured["display_name"] = request.display_name
        captured["output"] = request.output
        run_dir = Path(request.runs_dir) / request.output
        run_dir.mkdir()
        (run_dir / "artifact.json").write_text("{}\n", encoding="utf-8")
        return 0

    monkeypatch.setattr("oracle_builder.training.workflow.run_training", run_training)
    result = TrainingExecutor().execute(
        work_unit={"action": "train", "parameters": {"run_name": "North slope baseline", "queue_execution": {
            "mode": "auto", "minimum_batch_size": 1, "maximum_batch_size": 64,
            "target_vram_fraction": {"minimum": 0.3, "maximum": 0.8},
        }}},
        inputs={"input": dataset}, configuration=configuration, staging=staging,
        emit_event=lambda event_type, _message, data=None: events.append((event_type, dict(data or {}))),
    )

    assert result == {"artifact_root": "."}
    assert Path(captured["config"]).parent == tmp_path
    assert captured["display_name"] == "North slope baseline"
    assert captured["output"] == "run"
    assert "batch_size = 24" in Path(captured["config"]).read_text(encoding="utf-8")
    assert ("batch_calibration_started", {"minimum_batch_size": 1, "maximum_batch_size": 64}) in events
    assert any(event == "batch_size_resolved" and data["batch_size"] == 24 for event, data in events)


def test_verification_probes_but_never_enters_training(tmp_path, monkeypatch):
    configuration, dataset, staging = [tmp_path / name for name in ('configuration', 'dataset', 'staging')]
    for path in (configuration, dataset, staging): path.mkdir()
    (configuration / 'payload').write_text('[data]\nbatch_size = 4\n')
    (dataset / 'payload').write_bytes(b'sqlite')
    calls = []
    def probe(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='ORACLE_BATCH_TUNE_RESULT=' + json.dumps({'ready': True, 'recommended_batch_size': 4}))
    monkeypatch.setattr('oracle_builder.worker.pull._run_batch_calibrator', probe)
    monkeypatch.setattr('oracle_builder.config.resolve_config', lambda *args: {'classification': {}})
    monkeypatch.setattr('oracle_data_contracts.artifacts.splits.create_split_manifest', lambda *args: {'assignments': [], 'counts': {}})
    monkeypatch.setattr('oracle_builder.training.workflow.run_training', lambda request: (_ for _ in ()).throw(AssertionError('Training must not run')))
    TrainingExecutor().execute(work_unit={'work_unit_id': 'verify-job', 'parameters': {'queue_execution': {'mode': 'manual', 'phase': 'verify', 'batch_size': 4}}},
                               inputs={'input': dataset}, configuration=configuration, staging=staging, emit_event=lambda *args: None)
    assert calls[0][calls[0].index('--minimum') + 1] == '4'
    assert calls[0][calls[0].index('--maximum') + 1] == '4'
    assert json.loads((staging / 'verification.json').read_text())['batch_size'] == 4
    assert not (staging / 'run').exists()


def test_verification_reports_sparse_split_warning_and_remains_ready(tmp_path, monkeypatch):
    configuration, dataset, staging = [tmp_path / name for name in ('configuration', 'dataset', 'staging')]
    for path in (configuration, dataset, staging): path.mkdir()
    (configuration / 'payload').write_text('[data]\nbatch_size = 4\n')
    (dataset / 'payload').write_bytes(b'sqlite')
    monkeypatch.setattr('oracle_builder.worker.pull._run_batch_calibrator', lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=0, stdout='ORACLE_BATCH_TUNE_RESULT=' + json.dumps({'ready': True, 'recommended_batch_size': 4})))
    monkeypatch.setattr('oracle_builder.config.resolve_config', lambda *args: {'classification': {}})
    warning = 'Class 56 has no test example; train and validation are preserved.'
    monkeypatch.setattr('oracle_data_contracts.artifacts.splits.create_split_manifest',
                        lambda *args: {'assignments': [], 'coverage': {'valid': False, 'warnings': [warning]}})
    events = []
    TrainingExecutor().execute(
        work_unit={'work_unit_id': 'verify-job', 'parameters': {'queue_execution': {'mode': 'manual', 'phase': 'verify', 'batch_size': 4}}},
        inputs={'input': dataset}, configuration=configuration, staging=staging,
        emit_event=lambda event, _message, data=None: events.append((event, data)),
    )
    report = json.loads((staging / 'verification.json').read_text())
    assert report['ready'] is True and report['warnings'] == [warning]
    assert any(event == 'split_coverage_warning' and data['warning_count'] == 1 for event, data in events)


def test_verified_training_applies_batch_without_recalibrating(tmp_path, monkeypatch):
    configuration, dataset, staging = [tmp_path / name for name in ('configuration', 'dataset', 'staging')]
    for path in (configuration, dataset, staging): path.mkdir()
    (configuration / 'payload').write_text('[data]\nbatch_size = 4\n')
    (dataset / 'payload').write_bytes(b'sqlite')
    monkeypatch.setattr('oracle_builder.worker.pull._run_batch_calibrator', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('Must not recalibrate')))
    def train(request):
        assert 'batch_size = 24' in Path(request.config).read_text()
        run = Path(request.runs_dir) / request.output
        run.mkdir()
        (run / 'artifact.json').write_text('{}')
        return 0
    monkeypatch.setattr('oracle_builder.training.workflow.run_training', train)
    TrainingExecutor().execute(work_unit={'parameters': {'queue_execution': {'mode': 'verified', 'batch_size': 24}}},
                               inputs={'input': dataset}, configuration=configuration, staging=staging, emit_event=lambda *args: None)
    assert (configuration / 'payload').read_text() == '[data]\nbatch_size = 4\n'


def test_training_output_is_resealed_after_runtime_path_sanitization(tmp_path, monkeypatch):
    import uuid
    from oracle_builder.artifacts import create_run_artifact, update_run_artifact, seal_run_artifact, validate_run_artifact
    configuration, dataset, staging = [tmp_path / name for name in ('configuration', 'dataset', 'staging')]
    for path in (configuration, dataset, staging): path.mkdir()
    (configuration / 'payload').write_text('[data]\nbatch_size = 4\n')
    (dataset / 'payload').write_bytes(b'sqlite')
    before = {}
    def train(request):
        run_dir = Path(request.runs_dir) / request.output
        config = {'run': {'task': 'classification', 'model': 'simple_cnn'}, 'data': {'input_shape': [16,16,1], 'num_classes': 3},
                  'dataset': {'dataset_id': str(uuid.uuid4()), 'lifecycle': 'frozen'},
                  'paths': {'input_path': '/worker-private/dataset.sqlite', 'run_dir': str(run_dir)}}
        create_run_artifact(run_dir, run_id=str(uuid.uuid4()), name='packaging fixture', config=config, source_config=configuration / 'payload')
        # An interrupted artifact needs no model files, but exercises the same
        # real manifest, provenance inventory, and checksum contract.
        update_run_artifact(run_dir, status='interrupted')
        before.update(seal_run_artifact(run_dir))
        assert validate_run_artifact(run_dir)['valid']
        return 0
    monkeypatch.setattr('oracle_builder.training.workflow.run_training', train)
    TrainingExecutor().execute(work_unit={'parameters': {}}, inputs={'input': dataset}, configuration=configuration, staging=staging, emit_event=lambda *args: None)
    report = validate_run_artifact(staging)
    assert report['valid'], report
    assert report['fingerprint_sha256'] != before['fingerprint_sha256']
    runtime = json.loads((staging / 'provenance/runtime.json').read_text())
    assert runtime['paths'] == {'input_artifact': 'materialized'}
