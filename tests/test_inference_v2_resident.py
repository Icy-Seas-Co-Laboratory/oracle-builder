from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

import numpy as np
import pytest

from oracle_builder.inference.contracts import (
    InferenceItem,
    InferenceResult,
    InferenceResultSet,
    ModelReference,
)
from oracle_builder.inference.resident import (
    ResidentArtifactChangedError,
    ResidentInferenceRuntime,
)


def _identity(task: str = "classification") -> tuple[str, str, str]:
    return str(uuid.uuid4()), str(uuid.uuid4()), "a" * 64


def _artifact(
    root: Path, *, artifact_id: str, run_id: str, fingerprint: str, task: str = "classification"
) -> Path:
    root.mkdir()
    (root / "artifact.json").write_text(
        json.dumps(
            {
                "artifact_id": artifact_id,
                "run_id": run_id,
                "fingerprint_sha256": fingerprint,
                "model": {"task": task, "architecture": "test-model"},
            }
        ),
        encoding="utf-8",
    )
    return root


class _FakeBundle:
    def __init__(self, *, artifact_id: str, run_id: str, fingerprint: str, task: str = "classification"):
        self.model_reference = ModelReference(
            artifact_id=artifact_id,
            run_id=run_id,
            task=task,
            architecture="test-model",
            artifact_fingerprint=fingerprint,
        )
        self.execution_diagnostics = {"accelerator": "cpu"}
        self.batch_sizes: list[int] = []
        self.config = {"data": {"input_shape": [4, 4, 1]}, "training": {}}

    def warm_for_serving(self, batch_sizes):
        return getattr(self, "warm_report", {"resolved_max_batch_size": max(batch_sizes), "warmup": []})

    def predict_batch(self, items):
        self.batch_sizes.append(len(items))
        result_set = InferenceResultSet(model=self.model_reference)
        for sequence, item in enumerate(items):
            result_set.append(
                InferenceResult(
                    request_id=item.request_id,
                    item_id=item.item_id,
                    model=self.model_reference,
                    output={"type": self.model_reference.task},
                    input_sha256=item.input_sha256,
                    result_set_id=result_set.result_set_id,
                    sequence_number=sequence,
                )
            )
        return result_set.complete()

    def predict(self, item):
        result_set = self.predict_batch([item])
        return result_set.results[0]


def test_resident_runtime_warms_and_resolves_immutable_aliases(tmp_path: Path):
    artifact_id, run_id, fingerprint = _identity()
    path = _artifact(tmp_path / "model", artifact_id=artifact_id, run_id=run_id, fingerprint=fingerprint)
    bundle = _FakeBundle(artifact_id=artifact_id, run_id=run_id, fingerprint=fingerprint)

    def validator(_):
        return {"valid": True, "lifecycle": "sealed", "status": "complete", "fingerprint_sha256": fingerprint}

    runtime = ResidentInferenceRuntime(loader=lambda _: bundle, validator=validator)
    try:
        model = runtime.register(path)
        assert model.artifact_fingerprint == fingerprint
        warmed = runtime.warm(fingerprint)
        assert warmed["state"] == "ready"
        result = runtime.predict(artifact_id, [InferenceItem.from_array(np.ones((2, 2), dtype="uint8"))])
        assert result.counts["succeeded"] == 1
        row = runtime.describe()[0]
        assert row["task"] == "classification"
        assert row["runtime"]["mode"] == "classification_microbatch"
    finally:
        runtime.close()


def test_resident_runtime_uses_calibrated_batch_limit(tmp_path: Path):
    artifact_id, run_id, fingerprint = _identity()
    path = _artifact(tmp_path / "calibrated", artifact_id=artifact_id, run_id=run_id, fingerprint=fingerprint)
    bundle = _FakeBundle(artifact_id=artifact_id, run_id=run_id, fingerprint=fingerprint)
    bundle.warm_report = {
        "strata": {"small": {"resolved_max_batch_size": 16}, "large": {"resolved_max_batch_size": 8}}
    }
    runtime = ResidentInferenceRuntime(
        loader=lambda _: bundle,
        validator=lambda _: {"valid": True, "lifecycle": "sealed", "status": "complete", "fingerprint_sha256": fingerprint},
        classification_max_batch_size=64,
    )
    try:
        runtime.register(path)
        runtime.warm(fingerprint)
        assert runtime.describe()[0]["runtime"]["max_batch_size"] == 8
        with pytest.raises(ValueError, match="8-item limit"):
            runtime.predict(fingerprint, [InferenceItem.from_array(np.ones((2, 2), dtype="uint8")) for _ in range(9)])
    finally:
        runtime.close()


def test_resident_runtime_microbatches_concurrent_classification_requests(tmp_path: Path):
    artifact_id, run_id, fingerprint = _identity()
    path = _artifact(tmp_path / "model", artifact_id=artifact_id, run_id=run_id, fingerprint=fingerprint)
    bundle = _FakeBundle(artifact_id=artifact_id, run_id=run_id, fingerprint=fingerprint)
    runtime = ResidentInferenceRuntime(
        loader=lambda _: bundle,
        validator=lambda _: {"valid": True, "lifecycle": "sealed", "status": "complete", "fingerprint_sha256": fingerprint},
        classification_max_batch_size=4,
        classification_max_wait_ms=50,
    )
    runtime.register(path)
    barrier = threading.Barrier(3)
    results = []

    def predict(value: int):
        barrier.wait()
        results.append(runtime.predict(fingerprint, [InferenceItem.from_array(np.full((2, 2), value, dtype="uint8"))]))

    first = threading.Thread(target=predict, args=(1,))
    second = threading.Thread(target=predict, args=(2,))
    first.start()
    second.start()
    barrier.wait()
    first.join(timeout=2)
    second.join(timeout=2)
    try:
        assert bundle.batch_sizes == [2]
        assert len(results) == 2
        assert all(value.counts["succeeded"] == 1 for value in results)
    finally:
        runtime.close()


def test_resident_runtime_revalidates_artifact_before_loading(tmp_path: Path):
    artifact_id, run_id, fingerprint = _identity()
    path = _artifact(tmp_path / "model", artifact_id=artifact_id, run_id=run_id, fingerprint=fingerprint)
    valid = {"value": True}

    def validator(_):
        return {
            "valid": valid["value"],
            "lifecycle": "sealed",
            "status": "complete",
            "fingerprint_sha256": fingerprint,
            "errors": ["contents changed"] if not valid["value"] else [],
        }

    runtime = ResidentInferenceRuntime(
        loader=lambda _: _FakeBundle(artifact_id=artifact_id, run_id=run_id, fingerprint=fingerprint),
        validator=validator,
    )
    try:
        runtime.register(path)
        valid["value"] = False
        with pytest.raises(ResidentArtifactChangedError, match="verification failed"):
            runtime.warm(fingerprint)
    finally:
        runtime.close()


def test_resident_runtime_keeps_mask_refinement_requests_item_oriented(tmp_path: Path):
    artifact_id, run_id, fingerprint = _identity("segmentation")
    path = _artifact(
        tmp_path / "model", artifact_id=artifact_id, run_id=run_id,
        fingerprint=fingerprint, task="segmentation",
    )
    bundle = _FakeBundle(
        artifact_id=artifact_id, run_id=run_id, fingerprint=fingerprint, task="segmentation"
    )
    runtime = ResidentInferenceRuntime(
        loader=lambda _: bundle,
        validator=lambda _: {"valid": True, "lifecycle": "sealed", "status": "complete", "fingerprint_sha256": fingerprint},
        classification_max_wait_ms=50,
    )
    runtime.register(path)
    first = threading.Thread(
        target=lambda: runtime.predict(fingerprint, [InferenceItem.from_array(np.ones((2, 2)))])
    )
    second = threading.Thread(
        target=lambda: runtime.predict(fingerprint, [InferenceItem.from_array(np.ones((2, 2)))])
    )
    first.start()
    second.start()
    first.join(timeout=2)
    second.join(timeout=2)
    try:
        # One invocation is warmup.  The two requests must remain separate,
        # preserving the bundle's own variable-size tile expansion semantics.
        assert bundle.batch_sizes == [1, 1, 1]
        row = runtime.describe()[0]
        assert row["runtime"]["mode"] == "segmentation_item"
    finally:
        runtime.close()


def test_resident_runtime_warms_candidate_distance_model_with_mask(tmp_path: Path):
    artifact_id, run_id, fingerprint = _identity("segmentation")
    path = _artifact(
        tmp_path / "distance", artifact_id=artifact_id, run_id=run_id,
        fingerprint=fingerprint, task="segmentation",
    )
    bundle = _FakeBundle(
        artifact_id=artifact_id, run_id=run_id, fingerprint=fingerprint, task="segmentation"
    )
    bundle.config = {
        "data": {"input_shape": [4, 4, 3], "candidate_distance": "geodesic"},
        "evaluation": {"segmentation_target_mode": "validated_mask"},
    }
    observed = []
    original_predict = bundle.predict

    def predict(item):
        observed.append(item)
        return original_predict(item)

    bundle.predict = predict
    runtime = ResidentInferenceRuntime(
        loader=lambda _: bundle,
        validator=lambda _: {"valid": True, "lifecycle": "sealed", "status": "complete", "fingerprint_sha256": fingerprint},
    )
    try:
        runtime.register(path)
        warmed = runtime.warm(fingerprint)
        assert "candidate_mask" in observed[0].inputs
        assert warmed["capabilities"]["required_inputs"] == ["image", "candidate_mask"]
        assert warmed["capabilities"]["target_mode"] == "validated_mask"
    finally:
        runtime.close()


def test_resident_runtime_warms_single_channel_segmentation_without_candidate(tmp_path: Path):
    artifact_id, run_id, fingerprint = _identity("segmentation")
    path = _artifact(
        tmp_path / "single_channel", artifact_id=artifact_id, run_id=run_id,
        fingerprint=fingerprint, task="segmentation",
    )
    bundle = _FakeBundle(
        artifact_id=artifact_id, run_id=run_id, fingerprint=fingerprint, task="segmentation"
    )
    bundle.config["data"]["candidate_distance"] = "none"
    observed = []
    original_predict = bundle.predict

    def predict(item):
        observed.append(item)
        return original_predict(item)

    bundle.predict = predict
    runtime = ResidentInferenceRuntime(
        loader=lambda _: bundle,
        validator=lambda _: {"valid": True, "lifecycle": "sealed", "status": "complete", "fingerprint_sha256": fingerprint},
    )
    try:
        runtime.register(path)
        warmed = runtime.warm(fingerprint)
        assert set(observed[0].inputs) == {"image"}
        assert warmed["capabilities"]["required_inputs"] == ["image"]
    finally:
        runtime.close()
