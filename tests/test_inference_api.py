from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np

from oracle_builder.api.app import create_app
from oracle_builder.api.registry import InferenceModelRegistry
from oracle_builder.classification.evidence import IdentityEvidenceIndex
from oracle_builder.inference.contracts import (
    ArrayPayload,
    InferenceItem,
    InferenceResult,
    InferenceResultSet,
    ModelReference,
)
from oracle_builder.inference.transport import (
    NPZ_MEDIA_TYPE,
    decode_inference_request,
    decode_inference_result,
    encode_inference_request,
)


def _reference() -> ModelReference:
    return ModelReference(
        artifact_id=str(uuid.uuid4()),
        run_id=str(uuid.uuid4()),
        task="segmentation",
        architecture="test",
        artifact_fingerprint="abc123",
    )


class FakeBundle:
    def __init__(self):
        self.model_reference = _reference()

    def predict_batch(self, items):
        result_set = InferenceResultSet(model=self.model_reference)
        for sequence, item in enumerate(items):
            mask = np.asarray(item.inputs["candidate_mask"].values) > 0
            result_set.append(
                InferenceResult(
                    request_id=item.request_id,
                    item_id=item.item_id,
                    model=self.model_reference,
                    output={
                        "type": "mask_refinement",
                        "mask": ArrayPayload(mask.astype("uint8")),
                    },
                    input_sha256=item.input_sha256,
                    result_set_id=result_set.result_set_id,
                    sequence_number=sequence,
                )
            )
        return result_set.complete()


def test_npz_transport_round_trip_preserves_arrays_and_hashes():
    item = InferenceItem.from_array(
        np.arange(16, dtype="uint8").reshape(4, 4),
        candidate_mask=np.eye(4, dtype="uint8"),
    )
    payload = encode_inference_request(str(uuid.uuid4()), [item])

    decoded = decode_inference_request(payload)

    assert len(decoded.items) == 1
    assert decoded.items[0].input_sha256 == item.input_sha256
    np.testing.assert_array_equal(decoded.items[0].inputs["image"].values, item.inputs["image"].values)


def test_inference_api_lists_models_and_returns_npz(tmp_path: Path):
    from fastapi.testclient import TestClient

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bundle = FakeBundle()
    registry = InferenceModelRegistry(loader=lambda _: bundle)
    registry.register("test-refiner", run_dir)
    app = create_app(registry, auth_token="secret", preload=True)
    item = InferenceItem.from_array(
        np.arange(16, dtype="uint8").reshape(4, 4),
        candidate_mask=np.eye(4, dtype="uint8"),
    )

    with TestClient(app) as client:
        catalog = client.get(
            "/v1/models",
            params={"task": "segmentation"},
            headers={"Authorization": "Bearer secret"},
        )
        response = client.post(
            "/v1/models/test-refiner:predict",
            content=encode_inference_request(str(uuid.uuid4()), [item]),
            headers={
                "Authorization": "Bearer secret",
                "Content-Type": NPZ_MEDIA_TYPE,
            },
        )

    assert catalog.status_code == 200
    assert catalog.json()["models"][0]["model"]["artifact_id"] == bundle.model_reference.artifact_id
    assert response.status_code == 200
    decoded = decode_inference_result(response.content)
    assert decoded["counts"]["succeeded"] == 1
    np.testing.assert_array_equal(decoded["results"][0]["output"]["mask"], np.eye(4, dtype="uint8"))


def test_classification_catalog_describes_evidence_and_exemplar_metadata(tmp_path: Path):
    from fastapi.testclient import TestClient

    run_dir = tmp_path / "classification-run"
    run_dir.mkdir()
    bundle = FakeBundle()
    bundle.model_reference = ModelReference(
        artifact_id=str(uuid.uuid4()),
        run_id=str(uuid.uuid4()),
        task="classification",
        architecture="classifier",
    )
    label_id = str(uuid.uuid4())
    bundle.config = {
        "dataset": {"labels": [{"class_index": 0, "label_id": label_id, "name": "copepod"}]},
        "model": {"embedding_dim": 2, "normalize_embeddings": True},
        "evidence": {"knn_k": 3},
    }
    bundle.evidence_index = IdentityEvidenceIndex.build(
        np.asarray([[1.0, 0.0]], dtype="float32"),
        np.asarray([0]),
        ["exemplar-1"],
    )
    registry = InferenceModelRegistry(loader=lambda _: bundle)
    registry.register("classifier", run_dir)
    app = create_app(registry, preload=True)

    with TestClient(app) as client:
        catalog = client.get("/v1/models", params={"task": "classification"})
        exemplar = client.get("/v1/models/classifier/evidence/exemplar-1")

    capabilities = catalog.json()["models"][0]["capabilities"]
    assert capabilities["labels"] == [
        {"class_index": 0, "label_id": label_id, "name": "copepod"}
    ]
    assert capabilities["embedding"] == {
        "available": True,
        "dimension": 2,
        "normalized": True,
    }
    assert capabilities["evidence"]["knn"] is True
    assert capabilities["evidence"]["visual_exemplars"] is False
    assert exemplar.json() == {
        "item_id": "exemplar-1",
        "class_index": 0,
        "label": {"class_index": 0, "label_id": label_id, "name": "copepod"},
        "image_available": False,
    }


def test_registry_discovers_sealed_models_under_a_root(tmp_path: Path):
    root = tmp_path / "models"

    def artifact(relative: str, *, name: str, artifact_id: str, lifecycle="sealed"):
        run_dir = root / relative
        run_dir.mkdir(parents=True)
        (run_dir / "artifact.json").write_text(
            json.dumps(
                {
                    "artifact_schema": {"name": "oracle_builder_model_run", "version": "1.0.0"},
                    "artifact_type": "model_run",
                    "artifact_id": artifact_id,
                    "name": name,
                    "lifecycle": lifecycle,
                }
            ),
            encoding="utf-8",
        )
        return run_dir

    first = artifact("plankton-v1", name="Plankton v1", artifact_id="a" * 32)
    second = artifact("nested/refiner", name="Plankton v1", artifact_id="b" * 32)
    artifact("working", name="Working", artifact_id="c" * 32, lifecycle="working")

    registry = InferenceModelRegistry(loader=lambda _: FakeBundle())
    report = registry.register_root(root)

    assert report["registered"] == [
        {"alias": "plankton-v1", "path": str(second)},
        {"alias": "plankton-v1-aaaaaaaa", "path": str(first)},
    ]
    assert report["skipped"] == [
        {"path": str(root / "working"), "reason": "artifact is not sealed"}
    ]
    assert registry.registered_count == 2


def test_failed_model_remains_visible_in_its_task_catalog(tmp_path: Path):
    run_dir = tmp_path / "broken-classifier"
    run_dir.mkdir()
    (run_dir / "artifact.json").write_text(
        json.dumps(
            {
                "artifact_schema": {"name": "oracle_builder_model_run", "version": "1.0.0"},
                "artifact_type": "model_run",
                "artifact_id": "artifact-1",
                "run_id": "run-1",
                "lifecycle": "sealed",
                "model": {
                    "task": "classification",
                    "architecture": "resnet",
                    "outputs": {"class_count": 1, "labels": [{"class_index": 0, "name": "copepod"}]},
                },
            }
        ),
        encoding="utf-8",
    )

    def fail_to_load(_):
        raise RuntimeError("integrity validation failed")

    registry = InferenceModelRegistry(loader=fail_to_load)
    registry.register("broken", run_dir)
    registry.preload()

    rows = registry.describe(task="classification")
    assert len(rows) == 1
    assert rows[0]["available"] is False
    assert rows[0]["task"] == "classification"
    assert rows[0]["model"]["artifact_id"] == "artifact-1"
    assert rows[0]["load_error"] == "RuntimeError: integrity validation failed"
