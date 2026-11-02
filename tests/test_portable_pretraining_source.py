from __future__ import annotations

import numpy as np
import pytest

from oracle_builder.training import recovery


def test_portable_source_checks_identity_configuration_and_loads_weights(tmp_path, monkeypatch):
    from oracle_builder import artifacts
    from oracle_builder.saving import load_test

    config = {"run": {"task": "classification", "model": "simple"}, "data": {"input_shape": [2], "batch_size": 2}, "model": {}}
    manifest = {"artifact_id": "source", "run_id": "run"}
    state = {"completed_epoch": 0, "pretraining_completed": True,
             "pretraining_handoff": {"boundary": "whole_phase"},
             "model_path": "model/recovery/model.keras", "model_sha256": "digest"}
    monkeypatch.setattr(artifacts, "validate_run_artifact", lambda _: {"valid": True, "lifecycle": "sealed", "errors": [], "fingerprint_sha256": "fp"})
    monkeypatch.setattr(artifacts, "read_run_manifest", lambda _: manifest)
    monkeypatch.setattr(artifacts, "read_run_config", lambda _: config)
    monkeypatch.setattr(recovery, "validate_recovery_state", lambda *a, **kw: state)

    class Model:
        layers = []
        def __init__(self, value): self.value = value
        def get_weights(self): return [np.array([self.value])]
        def set_weights(self, weights): self.value = weights[0][0]

    monkeypatch.setattr(load_test, "_load_keras_model", lambda *a, **kw: Model(7))
    target = Model(1)
    provenance = recovery.load_portable_pretraining(target, tmp_path, config, "source")
    assert target.value == 7
    assert provenance["source_model_sha256"] == "digest"
    with pytest.raises(ValueError, match="identity"):
        recovery.load_portable_pretraining(target, tmp_path, config, "other")
    recovery.load_portable_pretraining(target, tmp_path, {**config, "data": {"input_shape": [2], "batch_size": 16}}, "source")
    with pytest.raises(ValueError, match="data input_shape"):
        recovery.load_portable_pretraining(target, tmp_path, {**config, "data": {}}, "source")
    state["completed_epoch"] = 1
    with pytest.raises(ValueError, match="epoch-zero"):
        recovery.load_portable_pretraining(target, tmp_path, config, "source")

@pytest.mark.parametrize("mode", ["fine_tune", "transfer_encoder", "resume"])
def test_worker_rejects_unsupported_initialization(tmp_path, mode):
    from oracle_builder.worker.pull import PullProtocolError, TrainingExecutor

    dataset = tmp_path / "dataset"
    config = tmp_path / "config"
    dataset.mkdir()
    config.mkdir()
    (dataset / "payload").write_bytes(b"dataset")
    (config / "payload").write_bytes(b"configuration")
    with pytest.raises(PullProtocolError, match="Unsupported training initialization"):
        TrainingExecutor().execute(
            work_unit={"parameters": {"initialization": {"mode": mode}}},
            inputs={"input": dataset}, configuration=config, staging=tmp_path,
            emit_event=lambda *args: None,
        )

@pytest.mark.parametrize("handoff", [
    None,
    {},
    {"boundary": "whole_phase", "supervised_epoch": 1, "ssl_continuation": "not_portable"},
    {"boundary": "whole_phase", "supervised_epoch": 0, "ssl_continuation": "portable"},
    {"boundary": "unknown", "supervised_epoch": 0, "ssl_continuation": "not_portable"},
    {"boundary": "portable_pretraining", "supervised_epoch": 0, "ssl_continuation": "not_portable"},
])
def test_recovery_rejects_incomplete_pretraining_handoff(monkeypatch, handoff):
    config = {}
    state = {"artifact_id": "a", "run_id": "r", "phase": "supervised",
             "config_sha256": recovery.recovery_config_hash(config),
             "completed_epoch": 1, "pretraining_completed": True,
             "pretraining_handoff": handoff}
    monkeypatch.setattr(recovery, "read_recovery_state", lambda _: state)
    with pytest.raises(ValueError, match="pretraining"):
        recovery.validate_recovery_state("unused", config, artifact_id="a", run_id="r")


def test_recovery_accepts_complete_handoffs_and_rejects_unmarked(monkeypatch):
    config = {}
    state = {"artifact_id": "a", "run_id": "r", "phase": "supervised",
             "config_sha256": recovery.recovery_config_hash(config), "completed_epoch": 1,
             "pretraining_completed": True,
             "pretraining_handoff": {"boundary": "whole_phase", "supervised_epoch": 0,
                                      "ssl_continuation": "not_portable"}}
    monkeypatch.setattr(recovery, "read_recovery_state", lambda _: state)
    assert recovery.validate_recovery_state("unused", config, artifact_id="a", run_id="r") is state
    state["pretraining_handoff"].update(boundary="portable_pretraining", source_artifact_id="a",
                                        source_run_id="r", source_fingerprint_sha256="a" * 64,
                                        source_model_sha256="b" * 64)
    assert recovery.validate_recovery_state("unused", config, artifact_id="a", run_id="r") is state
    state["pretraining_handoff"]["source_model_sha256"] = "bad"
    with pytest.raises(ValueError, match="provenance"):
        recovery.validate_recovery_state("unused", config, artifact_id="a", run_id="r")
    del state["pretraining_completed"]
    with pytest.raises(ValueError, match="without completion"):
        recovery.validate_recovery_state("unused", config, artifact_id="a", run_id="r")
