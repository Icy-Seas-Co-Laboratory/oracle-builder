from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from oracle_builder.worker.pull import TrainingExecutor


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

    monkeypatch.setattr("oracle_builder.worker.pull.subprocess.run", tuned_process)
    captured: dict[str, str] = {}

    def run_training(request):
        captured["config"] = request.config
        run_dir = Path(request.runs_dir) / request.output
        run_dir.mkdir()
        (run_dir / "artifact.json").write_text("{}\n", encoding="utf-8")
        return 0

    monkeypatch.setattr("oracle_builder.training.workflow.run_training", run_training)
    result = TrainingExecutor().execute(
        work_unit={"action": "train", "parameters": {"queue_execution": {
            "mode": "auto", "minimum_batch_size": 1, "maximum_batch_size": 64,
            "target_vram_fraction": {"minimum": 0.3, "maximum": 0.8},
        }}},
        inputs={"input": dataset}, configuration=configuration, staging=staging,
        emit_event=lambda event_type, _message, data=None: events.append((event_type, dict(data or {}))),
    )

    assert result == {"artifact_root": "."}
    assert Path(captured["config"]).parent == tmp_path
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
    monkeypatch.setattr('oracle_builder.worker.pull.subprocess.run', probe)
    monkeypatch.setattr('oracle_builder.config.resolve_config', lambda *args: {'classification': {}})
    monkeypatch.setattr('oracle_data_contracts.artifacts.splits.create_split_manifest', lambda *args: {'assignments': [], 'counts': {}})
    monkeypatch.setattr('oracle_builder.training.workflow.run_training', lambda request: (_ for _ in ()).throw(AssertionError('Training must not run')))
    TrainingExecutor().execute(work_unit={'work_unit_id': 'verify-job', 'parameters': {'queue_execution': {'mode': 'manual', 'phase': 'verify', 'batch_size': 4}}},
                               inputs={'input': dataset}, configuration=configuration, staging=staging, emit_event=lambda *args: None)
    assert calls[0][calls[0].index('--minimum') + 1] == '4'
    assert calls[0][calls[0].index('--maximum') + 1] == '4'
    assert json.loads((staging / 'verification.json').read_text())['batch_size'] == 4
    assert not (staging / 'run').exists()


def test_verified_training_applies_batch_without_recalibrating(tmp_path, monkeypatch):
    configuration, dataset, staging = [tmp_path / name for name in ('configuration', 'dataset', 'staging')]
    for path in (configuration, dataset, staging): path.mkdir()
    (configuration / 'payload').write_text('[data]\nbatch_size = 4\n')
    (dataset / 'payload').write_bytes(b'sqlite')
    monkeypatch.setattr('oracle_builder.worker.pull.subprocess.run', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('Must not recalibrate')))
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
