from __future__ import annotations

import json
import uuid

import pytest

from oracle_builder.inference.workflow import InferenceRequest, _seal_result
from oracle_builder.worker.pull import InferenceExecutor, PullProtocolError
from oracle_data_contracts.artifacts import ArtifactRef
from oracle_data_contracts.work_units import WorkUnit


def _unit(*, parameters=None) -> dict:
    return WorkUnit(
        work_unit_id=str(uuid.uuid4()), attempt_id=str(uuid.uuid4()), specification_id=str(uuid.uuid4()),
        action="infer",
        inputs={"model": ArtifactRef("model_run", "model-1"), "input": ArtifactRef("dataset", "data-1")},
        configuration=None, staging=ArtifactRef("staging", "result-1"), resources={}, parameters=parameters or {},
    ).to_dict()


def test_infer_work_unit_is_versioned_and_keeps_bounded_parameters():
    encoded = _unit(parameters={"split": "validation", "prediction_set": "demo"})
    parsed = WorkUnit.from_dict(encoded)
    assert parsed.action == "infer"
    assert parsed.parameters == {"split": "validation", "prediction_set": "demo"}


def test_inference_executor_only_accepts_materialized_model_and_dataset(tmp_path, monkeypatch):
    model, dataset, staging = tmp_path / "model", tmp_path / "dataset", tmp_path / "staging"
    model.mkdir(); dataset.mkdir(); staging.mkdir()
    (model / "artifact.json").write_text("{}")
    (dataset / "payload").write_bytes(b"sqlite")
    received = {}
    def fake_run(request, *, progress):
        received["request"] = request
        progress("inference_processing", "working", {"records": 1})
        return {"artifact_id": "result-id", "outputs": {"records": 1}}
    monkeypatch.setattr("oracle_builder.inference.workflow.run_inference", fake_run)
    events = []
    result = InferenceExecutor().execute(
        work_unit=_unit(parameters={"split": "all"}), inputs={"model": model, "input": dataset},
        configuration=None, staging=staging, emit_event=lambda *event: events.append(event),
    )
    assert result == {"artifact_root": ".", "artifact_id": "result-id", "records": 1}
    assert received["request"].model_run == str(model)
    assert received["request"].input == str(dataset / "payload")
    assert any(event[0] == "inference_complete" for event in events)


def test_inference_executor_rejects_unsafe_parameters(tmp_path):
    model, dataset, staging = tmp_path / "model", tmp_path / "dataset", tmp_path / "staging"
    model.mkdir(); dataset.mkdir(); staging.mkdir()
    (model / "artifact.json").write_text("{}")
    (dataset / "payload").write_bytes(b"sqlite")
    with pytest.raises(PullProtocolError, match="unsupported parameters"):
        InferenceExecutor().execute(
            work_unit=_unit(parameters={"command": "bad"}), inputs={"model": model, "input": dataset},
            configuration=None, staging=staging, emit_event=lambda *_: None,
        )


def test_inference_result_is_sealed_with_portable_lineage(tmp_path):
    output = tmp_path / "result"; output.mkdir()
    (output / "predictions.sqlite").write_bytes(b"prediction-db")
    manifest = _seal_result(
        output=output,
        model_manifest={"artifact_id": "model-id", "fingerprint_sha256": "model-sha"},
        request=InferenceRequest(model_run="model", input="input", output_dir=str(output), artifact_id=str(uuid.uuid4()),
                                 model_reference={"kind": "model_run", "id": "model-id"}, input_reference={"kind": "dataset", "id": "data-id"}),
        written=1, model_ref={"kind": "model_run", "id": "model-id"}, input_ref={"kind": "dataset", "id": "data-id"},
    )
    assert manifest["artifact_type"] == "inference_result"
    assert manifest["lifecycle"] == "sealed"
    assert (output / "checksums.sha256").is_file()
    assert json.loads((output / "artifact.json").read_text())["fingerprint_sha256"] == manifest["fingerprint_sha256"]
