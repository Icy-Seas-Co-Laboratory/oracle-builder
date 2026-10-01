from __future__ import annotations

import shutil
import json
import uuid
from pathlib import Path

import numpy as np
import pytest
import httpx
from fastapi.testclient import TestClient

from oracle_builder.inference.contracts import ArrayPayload, InferenceItem, InferenceResult, InferenceResultSet, ModelReference
from oracle_builder.inference.resident import ResidentModel
from oracle_builder.inference.server_v2 import create_inference_runtime_app
from oracle_builder.inference.transport_v2 import (
    V2_NPZ_MEDIA_TYPE,
    V2InferenceRequest,
    V2ResponseOptions,
    decode_v2_inference_result_set,
    encode_v2_inference_request,
)
from oracle_builder.orchestration.api import create_app


FINGERPRINT = "a" * 64
ARTIFACT_ID = str(uuid.uuid4())
RUN_ID = str(uuid.uuid4())


class _Store:
    def __init__(self, source: Path):
        self.source = source
        self.materializations = 0

    def materialize(self, _ref: object, target: Path) -> Path:
        self.materializations += 1
        shutil.copytree(self.source, target)
        return target


class _Orchestrator:
    def __init__(self, source: Path, task: str = "classification"):
        self.database = source.parent / "oracle.sqlite"
        self.artifact_store = _Store(source)
        self.row = {
            "artifact_id": ARTIFACT_ID,
            "artifact_type": "model_run",
            "lifecycle": "sealed",
            "status": "complete",
            "fingerprint_sha256": FINGERPRINT,
            "task": task,
        }

    def artifact(self, artifact_id: str):
        return self.row if artifact_id == ARTIFACT_ID else None

    def artifacts(self):
        return [self.row]

    def reconcile_startup(self):
        return {}

    def reconcile_worker_deployments(self):
        return {}

    def start_operation_runner(self):
        pass

    def stop_operation_runner(self):
        pass

    def _model_artifact_ref_for_inference(self, artifact: dict[str, object]) -> object:
        assert artifact is self.row
        return object()


class _Runtime:
    def __init__(self, task: str = "classification"):
        self.model: ResidentModel | None = None
        self.warm_count = 0
        self.register_count = 0
        self.predict_count = 0
        self.task = task

    def register(self, run_dir: Path, *, artifact_fingerprint: str | None = None) -> ResidentModel:
        self.register_count += 1
        assert artifact_fingerprint == FINGERPRINT
        candidate = ResidentModel(FINGERPRINT, ARTIFACT_ID, RUN_ID, self.task, "test", run_dir)
        if self.model is None:
            self.model = candidate
        assert self.model.run_dir == run_dir
        return self.model

    def warm(self, selector: str):
        assert selector == FINGERPRINT
        self.warm_count += 1
        return {"state": "ready"}

    def predict(self, selector: str, items: list[InferenceItem]) -> InferenceResultSet:
        assert selector == FINGERPRINT
        self.predict_count += 1
        reference = ModelReference(
            artifact_id=ARTIFACT_ID, run_id=RUN_ID, task=self.task,
            architecture="test", artifact_fingerprint=FINGERPRINT,
        )
        results = InferenceResultSet(model=reference)
        for sequence, item in enumerate(items):
            if item.metadata.get("fail"):
                results.append(InferenceResult(
                    request_id=item.request_id, item_id=item.item_id, model=reference,
                    output=None, status="failed", error={"type": "SyntheticFailure", "message": "bad input"},
                    input_sha256=item.input_sha256, result_set_id=results.result_set_id,
                    sequence_number=sequence,
                ))
                continue
            if self.task == "embedding":
                output = {"type": "embedding", "embedding": ArrayPayload(np.array([0.6, 0.8], dtype="float32")), "embedding_normalized": True}
            elif self.task == "clustering":
                output = {
                    "type": "clustering", "embedding": ArrayPayload(np.array([0.6, 0.8], dtype="float32")),
                    "embedding_normalized": True,
                    "evidence": {
                        "schema_version": "1.0.0", "metric": "cosine_similarity",
                        "decision": {"cluster_id": "cluster-0001", "novel": False},
                        "clusters": [{"cluster_id": "cluster-0001", "similarity": 0.9}],
                        "nearest_neighbors": [{"uuid": "reference-1", "cluster_id": "cluster-0001", "similarity": 0.92}],
                    },
                }
            elif self.task == "segmentation":
                output = {
                    "type": "mask_refinement", "target_mode": "candidate_delta",
                    "mask": ArrayPayload(np.zeros((4, 5), dtype="uint8")),
                    "reconstructed_mask": ArrayPayload(np.ones((4, 5), dtype="uint8")),
                    "probability_map": ArrayPayload(np.zeros((4, 5), dtype="float32")),
                    "reconstructed_probability_map": ArrayPayload(np.full((4, 5), 0.8, dtype="float32")),
                }
            else:
                output = {"type": "classification", "decision": {"label": "fish"},
                          "probabilities": ArrayPayload(np.array([0.2, 0.8], dtype="float32"))}
            results.append(InferenceResult(
                request_id=item.request_id,
                item_id=item.item_id,
                model=reference,
                output=output,
                input_sha256=item.input_sha256,
                result_set_id=results.result_set_id,
                sequence_number=sequence,
            ))
        return results.complete()

    def describe(self):
        return []


def _app(tmp_path: Path, task: str = "classification") -> tuple[object, _Orchestrator, _Runtime]:
    source = tmp_path / "source"
    source.mkdir()
    (source / "artifact.json").write_text("{}", encoding="utf-8")
    (source / "config").mkdir()
    (source / "config" / "resolved.json").write_text(json.dumps({
        "run": {"task": task, "model": "test"},
        "data": {"input_shape": [4, 5, 2] if task == "segmentation" else [4, 4, 1]},
        "training": {"segmentation_target": "candidate_delta" if task == "segmentation" else "validated_mask"},
    }), encoding="utf-8")
    if task == "clustering":
        (source / "model" / "clustering_evidence").mkdir(parents=True)
    orchestrator = _Orchestrator(source, task)
    runtime = _Runtime(task)
    return create_inference_runtime_app(
        orchestrator, runtime, "internal-secret", materialized_root=tmp_path / "materialized"
    ), orchestrator, runtime


def _headers() -> dict[str, str]:
    return {
        "Authorization": "Bearer internal-secret",
        "X-Artifact-Fingerprint": FINGERPRINT,
    }


def test_sidecar_warms_catalog_pinned_model_and_reuses_materialization(tmp_path: Path):
    app, orchestrator, runtime = _app(tmp_path)
    with TestClient(app) as client:
        first = client.post(f"/internal/v2/models/{ARTIFACT_ID}:warm", headers=_headers())
        second = client.post(f"/internal/v2/models/{ARTIFACT_ID}:warm", headers=_headers())

    assert first.status_code == 200
    assert first.json()["model"]["artifact_fingerprint"] == FINGERPRINT
    assert second.status_code == 200
    assert runtime.warm_count == 2
    assert runtime.register_count == 1
    assert orchestrator.artifact_store.materializations == 1


def test_sidecar_predicts_v2_binary_payload_and_enforces_catalog_fingerprint(tmp_path: Path):
    app, _, _ = _app(tmp_path)
    request_id = str(uuid.uuid4())
    item = InferenceItem.from_array(
        np.zeros((4, 4), dtype="uint8"), request_id=request_id
    )
    payload = encode_v2_inference_request(V2InferenceRequest(request_id, "classification", [item]))
    with TestClient(app) as client:
        unauthorized = client.post(f"/internal/v2/models/{ARTIFACT_ID}:predict", content=payload)
        mismatch = client.post(
            f"/internal/v2/models/{ARTIFACT_ID}:predict", content=payload,
            headers={**_headers(), "X-Artifact-Fingerprint": "b" * 64, "Content-Type": V2_NPZ_MEDIA_TYPE},
        )
        response = client.post(
            f"/internal/v2/models/{ARTIFACT_ID}:predict", content=payload,
            headers={**_headers(), "Content-Type": V2_NPZ_MEDIA_TYPE},
        )

    assert unauthorized.status_code == 401
    assert mismatch.status_code == 409
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(V2_NPZ_MEDIA_TYPE)
    decoded = decode_v2_inference_result_set(response.content)
    assert decoded["counts"]["succeeded"] == 1
    output = decoded["results"][0]["output"]
    assert output["schema_name"] == "oracle_builder.inference.v2.image_output"
    assert set(output) >= {"class_probabilities", "primary_decision", "embedding", "embedding_normalized", "knn", "prototype_similarity", "secondary_heads", "cluster_evidence"}
    assert [row["probability"] for row in output["class_probabilities"]["classes"]] == pytest.approx([0.2, 0.8])
    assert output["embedding"] is None


def test_sidecar_rejects_wrong_content_type_and_oversized_content_length(tmp_path: Path):
    app, _, _ = _app(tmp_path)
    with TestClient(app) as client:
        wrong_type = client.post(
            f"/internal/v2/models/{ARTIFACT_ID}:predict", content=b"x", headers=_headers()
        )
        too_large = client.post(
            f"/internal/v2/models/{ARTIFACT_ID}:predict", content=b"x",
            headers={**_headers(), "Content-Type": V2_NPZ_MEDIA_TYPE, "Content-Length": str(999)},
        )

    assert wrong_type.status_code == 415
    # The default is intentionally large; malformed body gets transport validation.
    assert too_large.status_code == 422


@pytest.mark.parametrize("model_task,request_task", [
    ("classification", "classification"),
    ("embedding", "classification"),
    ("clustering", "classification"),
    ("segmentation", "mask_refinement"),
])
def test_v2_end_to_end_contract_for_four_workflows(tmp_path: Path, model_task: str, request_task: str):
    app, _, _ = _app(tmp_path, model_task)
    request_id = str(uuid.uuid4())
    image = np.zeros((4, 5), dtype="uint8") if model_task == "segmentation" else np.zeros((4, 4), dtype="uint8")
    mask = np.zeros((4, 5), dtype="uint8") if model_task == "segmentation" else None
    first = InferenceItem.from_array(image, candidate_mask=mask, request_id=request_id)
    failed = InferenceItem.from_array(image, candidate_mask=mask, request_id=request_id, metadata={"fail": True})
    payload = encode_v2_inference_request(V2InferenceRequest(request_id, request_task, [first, failed]))
    with TestClient(app) as client:
        described = client.get(f"/internal/v2/models/{ARTIFACT_ID}:describe", headers=_headers())
        assert described.status_code == 200
        capabilities = described.json()["capabilities"]
        assert capabilities["model_task"] == model_task
        assert capabilities["request_task"] == request_task
        response = client.post(
            f"/internal/v2/models/{ARTIFACT_ID}:predict", content=payload,
            headers={**_headers(), "Content-Type": V2_NPZ_MEDIA_TYPE},
        )
    assert response.status_code == 200, response.text
    decoded = decode_v2_inference_result_set(response.content)
    assert decoded["counts"] == {"requested": 2, "succeeded": 1, "rejected": 0, "failed": 1}
    good, bad = decoded["results"]
    assert (good["request_id"], good["item_id"]) == (request_id, first.item_id)
    assert (bad["request_id"], bad["item_id"]) == (request_id, failed.item_id)
    assert bad["output"] is None and bad["error"]["type"] == "SyntheticFailure"
    output = good["output"]
    if request_task == "classification":
        assert output["schema_name"] == "oracle_builder.inference.v2.image_output"
        assert output["schema_version"] == "1.0.0"
        assert {"class_probabilities", "primary_decision", "embedding", "embedding_normalized", "knn", "prototype_similarity", "secondary_heads", "cluster_evidence"} <= set(output)
        assert output["secondary_heads"] is None
        if model_task == "classification":
            assert output["class_probabilities"]["classes"][1]["class_index"] == 1
            assert output["embedding"] is None and output["knn"] is None
        else:
            assert output["class_probabilities"] is None and output["primary_decision"] is None
            assert output["embedding"]["dtype"] == "float32"
            assert output["embedding"]["dimension"] == 2
            assert output["embedding"]["normalized"] is True
            np.testing.assert_allclose(output["embedding"]["values"], [0.6, 0.8])
            if model_task == "clustering":
                assert output["knn"]["source_artifact_id"] == ARTIFACT_ID
                assert output["knn"]["metric"] == "cosine_similarity"
                assert output["cluster_evidence"]["decision"]["cluster_id"] == "cluster-0001"
            else:
                assert output["knn"] is None and output["cluster_evidence"] is None
    else:
        np.testing.assert_array_equal(output["mask"], np.ones((4, 5), dtype="uint8"))


def test_v2_selected_unavailable_field_rejected_and_unrequested_fields_are_null(tmp_path: Path):
    app, _, _ = _app(tmp_path, "embedding")
    request_id = str(uuid.uuid4())
    item = InferenceItem.from_array(np.zeros((4, 4), dtype="uint8"), request_id=request_id)
    request = lambda outputs: encode_v2_inference_request(V2InferenceRequest(
        request_id, "classification", [item], V2ResponseOptions(frozenset(outputs)),
    ))
    with TestClient(app) as client:
        unsupported = client.post(
            f"/internal/v2/models/{ARTIFACT_ID}:predict", content=request({"knn"}),
            headers={**_headers(), "Content-Type": V2_NPZ_MEDIA_TYPE},
        )
        selected = client.post(
            f"/internal/v2/models/{ARTIFACT_ID}:predict", content=request({"embedding"}),
            headers={**_headers(), "Content-Type": V2_NPZ_MEDIA_TYPE},
        )
    assert unsupported.status_code == 422
    assert selected.status_code == 200
    output = decode_v2_inference_result_set(selected.content)["results"][0]["output"]
    assert output["embedding"] is not None
    assert output["embedding_normalized"] is None
    assert output["knn"] is None
    assert output["secondary_heads"] is None


def test_v2_advertised_output_missing_is_server_contract_error(tmp_path: Path):
    app, _, runtime = _app(tmp_path, "embedding")
    original_predict = runtime.predict

    def broken_predict(selector, items):
        result = original_predict(selector, items)
        del result.results[0].output["embedding"]
        return result

    runtime.predict = broken_predict
    request_id = str(uuid.uuid4())
    item = InferenceItem.from_array(np.zeros((4, 4), dtype="uint8"), request_id=request_id)
    payload = encode_v2_inference_request(V2InferenceRequest(request_id, "classification", [item]))
    with TestClient(app) as client:
        response = client.post(
            f"/internal/v2/models/{ARTIFACT_ID}:predict", content=payload,
            headers={**_headers(), "Content-Type": V2_NPZ_MEDIA_TYPE},
        )
    assert response.status_code == 500
    assert response.json()["detail"] == "Model output violated its serving contract"


def test_v2_repeated_request_id_is_correlation_without_server_deduplication(tmp_path: Path):
    app, _, runtime = _app(tmp_path, "embedding")
    request_id = str(uuid.uuid4())
    item = InferenceItem.from_array(np.zeros((4, 4), dtype="uint8"), request_id=request_id)
    payload = encode_v2_inference_request(V2InferenceRequest(request_id, "classification", [item]))
    with TestClient(app) as client:
        replies = [client.post(
            f"/internal/v2/models/{ARTIFACT_ID}:predict", content=payload,
            headers={**_headers(), "Content-Type": V2_NPZ_MEDIA_TYPE},
        ) for _ in range(2)]
    assert [reply.status_code for reply in replies] == [200, 200]
    first, second = [decode_v2_inference_result_set(reply.content)["results"][0] for reply in replies]
    assert first["request_id"] == second["request_id"] == request_id
    assert first["item_id"] == second["item_id"] == item.item_id
    assert first["result_id"] != second["result_id"]
    assert runtime.predict_count == 2


@pytest.mark.parametrize("model_task,request_task", [
    ("classification", "classification"), ("embedding", "classification"),
    ("clustering", "classification"), ("segmentation", "mask_refinement"),
])
def test_public_gateway_to_resident_sidecar_contract(tmp_path: Path, model_task: str, request_task: str):
    sidecar, orchestrator, _ = _app(tmp_path, model_task)
    gateway = create_app(
        orchestrator, inference_runtime_url="http://127.0.0.1:8111",
        inference_runtime_token="internal-secret",
    )
    gateway.state.inference_http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=sidecar), base_url="http://127.0.0.1:8111",
    )
    request_id = str(uuid.uuid4())
    image = np.zeros((4, 5), dtype="uint8") if model_task == "segmentation" else np.zeros((4, 4), dtype="uint8")
    item = InferenceItem.from_array(
        image, request_id=request_id,
        candidate_mask=np.zeros((4, 5), dtype="uint8") if model_task == "segmentation" else None,
    )
    payload = encode_v2_inference_request(V2InferenceRequest(request_id, request_task, [item]))
    with TestClient(gateway) as client:
        detail = client.get(f"/v2/inference/models/{ARTIFACT_ID}")
        assert detail.status_code == 200
        assert detail.json()["capabilities"]["model_task"] == model_task
        assert detail.json()["capabilities"]["request_task"] == request_task
        result = client.post(
            f"/v2/inference/models/{ARTIFACT_ID}:predict", content=payload,
            headers={"X-Artifact-Fingerprint": FINGERPRINT, "Content-Type": V2_NPZ_MEDIA_TYPE},
        )
    assert result.status_code == 200, result.text
    decoded = decode_v2_inference_result_set(result.content)
    assert decoded["results"][0]["request_id"] == request_id
    assert decoded["results"][0]["item_id"] == item.item_id
    if request_task == "classification":
        assert decoded["results"][0]["output"]["model_task"] == model_task
    else:
        np.testing.assert_array_equal(decoded["results"][0]["output"]["mask"], np.ones((4, 5), dtype="uint8"))
