from __future__ import annotations

import io
import json
import uuid

import numpy as np
import pytest

from oracle_builder.inference.contracts import (
    ArrayPayload,
    InferenceItem,
    InferenceResult,
    InferenceResultSet,
    ModelReference,
)
from oracle_builder.inference.transport_v2 import (
    V2InferenceOutputLimitError,
    V2InferenceRequest,
    V2InferenceTransportError,
    V2ResponseOptions,
    decode_v2_inference_request,
    decode_v2_inference_result_set,
    encode_v2_inference_request,
    encode_v2_inference_result_set,
    project_v2_result_set,
)


def _reference(task: str) -> ModelReference:
    return ModelReference(
        artifact_id=str(uuid.uuid4()), run_id=str(uuid.uuid4()), task=task,
        architecture="test", artifact_fingerprint="f" * 64,
    )


def _item(request_id: str, *, mask: bool = False) -> InferenceItem:
    return InferenceItem.from_array(
        np.arange(16, dtype="uint8").reshape(4, 4),
        candidate_mask=np.eye(4, dtype="uint8") if mask else None,
        request_id=request_id,
        metadata={"source": "test", "score": 0.25},
    )


def test_v2_classification_round_trip_preserves_correlation_and_hashes():
    request_id = str(uuid.uuid4())
    item = _item(request_id)
    encoded = encode_v2_inference_request(
        V2InferenceRequest(request_id, "classification", [item])
    )

    decoded = decode_v2_inference_request(encoded)

    assert decoded.request_id == request_id
    assert decoded.task == "classification"
    assert decoded.items[0].item_id == item.item_id
    assert decoded.items[0].request_id == request_id
    assert decoded.items[0].input_sha256 == item.input_sha256
    assert decoded.response_options.outputs is None
    assert {"class_probabilities", "primary_decision", "embedding", "knn"} <= decoded.response_options.resolved_outputs("classification")
    np.testing.assert_array_equal(decoded.items[0].inputs["image"].values, item.inputs["image"].values)


def test_v2_mask_refinement_round_trip_keeps_candidate_mask_and_requested_maps():
    request_id = str(uuid.uuid4())
    item = _item(request_id, mask=True)
    encoded = encode_v2_inference_request(V2InferenceRequest(
        request_id, "mask_refinement", [item],
        V2ResponseOptions(frozenset({"mask", "probability_map"}), max_output_bytes=4096),
    ))

    decoded = decode_v2_inference_request(encoded)

    assert decoded.response_options.resolved_outputs("mask_refinement") == {"mask", "probability_map"}
    np.testing.assert_array_equal(
        decoded.items[0].inputs["candidate_mask"].values,
        item.inputs["candidate_mask"].values,
    )


def test_v2_rejects_hash_tampering_after_manifest_is_edited():
    request_id = str(uuid.uuid4())
    encoded = encode_v2_inference_request(V2InferenceRequest(request_id, "classification", [_item(request_id)]))
    with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    arrays["item_0_image"] = np.zeros((4, 4), dtype="uint8")
    rebuilt = io.BytesIO()
    np.savez_compressed(rebuilt, **arrays)

    with pytest.raises(V2InferenceTransportError, match="identity verification"):
        decode_v2_inference_request(rebuilt.getvalue())


def test_v2_rejects_unreferenced_and_oversized_decompressed_arrays():
    request_id = str(uuid.uuid4())
    encoded = encode_v2_inference_request(V2InferenceRequest(request_id, "classification", [_item(request_id)]))
    with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    arrays["surplus"] = np.zeros((8,), dtype="uint8")
    rebuilt = io.BytesIO()
    np.savez_compressed(rebuilt, **arrays)
    with pytest.raises(V2InferenceTransportError, match="unreferenced"):
        decode_v2_inference_request(rebuilt.getvalue())

    with pytest.raises(V2InferenceTransportError, match="decompressed byte limit"):
        decode_v2_inference_request(encoded, max_total_array_bytes=8)


def test_v2_request_requires_one_task_and_one_correlation_id():
    request_id = str(uuid.uuid4())
    with pytest.raises(V2InferenceTransportError, match="item.request_id"):
        V2InferenceRequest(request_id, "classification", [_item(str(uuid.uuid4()))])
    with pytest.raises(V2InferenceTransportError, match="Unsupported classification inputs"):
        V2InferenceRequest(request_id, "classification", [_item(request_id, mask=True)])
    with pytest.raises(V2InferenceTransportError, match="Unsupported mask_refinement outputs"):
        V2InferenceRequest(
            request_id, "mask_refinement", [_item(request_id, mask=True)],
            V2ResponseOptions(frozenset({"decision"})),
        )
    with pytest.raises(V2InferenceTransportError, match="max_output_bytes"):
        V2InferenceRequest(
            request_id, "classification", [_item(request_id)],
            V2ResponseOptions(max_output_bytes=33 * 1024 * 1024),
        )


def test_v2_result_projection_omits_large_maps_unless_requested_and_round_trips():
    request_id = str(uuid.uuid4())
    item = _item(request_id, mask=True)
    reference = _reference("segmentation")
    result_set = InferenceResultSet(model=reference)
    result_set.append(InferenceResult(
        request_id=request_id, item_id=item.item_id, model=reference,
        result_set_id=result_set.result_set_id, input_sha256=item.input_sha256,
        output={
            "type": "mask_refinement", "target_mode": "binary", "threshold": {"value": 0.5},
            "mask": ArrayPayload(np.eye(4, dtype="uint8")),
            "probability_map": ArrayPayload(np.full((4, 4), 0.75, dtype="float32")),
            "logits": ArrayPayload(np.ones((4, 4), dtype="float32")),
        },
    ))
    projected = project_v2_result_set(
        result_set, V2ResponseOptions(frozenset({"mask"})), "mask_refinement"
    )
    output = projected.results[0].output
    assert output is not None
    assert set(output) == {"type", "target_mode", "threshold", "mask"}

    decoded = decode_v2_inference_result_set(encode_v2_inference_result_set(projected))
    assert "probability_map" not in decoded["results"][0]["output"]
    np.testing.assert_array_equal(decoded["results"][0]["output"]["mask"], np.eye(4, dtype="uint8"))


def test_v2_candidate_delta_defaults_to_reconstructed_mask_and_maps_probability():
    request_id = str(uuid.uuid4())
    item = _item(request_id, mask=True)
    reference = _reference("segmentation")
    result_set = InferenceResultSet(model=reference)
    result_set.append(InferenceResult(
        request_id=request_id, item_id=item.item_id, model=reference,
        result_set_id=result_set.result_set_id, input_sha256=item.input_sha256,
        output={
            "type": "mask_refinement", "target_mode": "candidate_delta",
            "mask": ArrayPayload(np.zeros((4, 4), dtype="uint8")),
            "reconstructed_mask": ArrayPayload(np.ones((4, 4), dtype="uint8")),
            "probability_map": ArrayPayload(np.zeros((4, 4), dtype="float32")),
            "reconstructed_probability_map": ArrayPayload(np.ones((4, 4), dtype="float32")),
        },
    ))
    default = project_v2_result_set(result_set, V2ResponseOptions(), "mask_refinement")
    np.testing.assert_array_equal(default.results[0].output["mask"].values, np.ones((4, 4)))
    selected = project_v2_result_set(
        result_set, V2ResponseOptions(frozenset({"delta_mask", "probability_map"})),
        "mask_refinement",
    )
    np.testing.assert_array_equal(selected.results[0].output["delta_mask"].values, np.zeros((4, 4)))
    np.testing.assert_array_equal(selected.results[0].output["probability_map"].values, np.ones((4, 4)))


def test_v2_result_projection_enforces_per_item_binary_budget():
    request_id = str(uuid.uuid4())
    item = _item(request_id, mask=True)
    reference = _reference("segmentation")
    result_set = InferenceResultSet(model=reference)
    result_set.append(InferenceResult(
        request_id=request_id, item_id=item.item_id, model=reference,
        result_set_id=result_set.result_set_id, input_sha256=item.input_sha256,
        output={"type": "mask_refinement", "mask": ArrayPayload(np.ones((64, 64), dtype="uint8"))},
    ))

    with pytest.raises(V2InferenceOutputLimitError, match="max_output_bytes=1024"):
        project_v2_result_set(
            result_set, V2ResponseOptions(frozenset({"mask"}), max_output_bytes=1024),
            "mask_refinement",
        )


def test_v2_image_output_preserves_evidence_identity_units_and_logits():
    request_id = str(uuid.uuid4())
    item = _item(request_id)
    reference = _reference("classification")
    result_set = InferenceResultSet(model=reference)
    result_set.append(InferenceResult(
        request_id=request_id, item_id=item.item_id, model=reference,
        result_set_id=result_set.result_set_id, input_sha256=item.input_sha256,
        output={
            "type": "classification",
            "probabilities": [
                {"class_index": 0, "label_id": "salmon", "label_name": "Salmon", "probability": 0.9},
                {"class_index": 1, "label_id": "halibut", "label_name": "Halibut", "probability": 0.1},
            ],
            "decision": {"class_index": 0, "label_id": "salmon"},
            "logits": [2.0, -0.2], "logits_source": "model",
            "embedding": ArrayPayload(np.array([0.6, 0.8], dtype="float32")),
            "embedding_normalized": True,
            "evidence": {
                "schema_version": "1.0.0",
                "knn": {"k_used": 1, "neighbors": [{"uuid": "neighbor-1", "label": 0, "similarity": 0.8}]},
                "prototype": {"similarities": {"0": 0.9, "1": 0.1}},
            },
        },
    ))
    supported = frozenset({
        "class_probabilities", "primary_decision", "embedding", "embedding_normalized",
        "knn", "prototype_similarity", "diagnostics",
    })
    projected = project_v2_result_set(
        result_set, V2ResponseOptions(), "classification",
        supported_outputs=supported, default_outputs=supported,
    )
    output = decode_v2_inference_result_set(encode_v2_inference_result_set(projected))["results"][0]["output"]
    assert output["schema_version"] == "1.0.0"
    assert output["class_probabilities"]["unit"] == "probability"
    assert output["class_probabilities"]["classes"][0]["label_id"] == "salmon"
    assert output["embedding"]["dtype"] == "float32" and output["embedding"]["dimension"] == 2
    assert output["embedding"]["normalized"] is True
    assert output["knn"]["metric"] == "cosine_similarity"
    assert output["knn"]["source_artifact_id"] == reference.artifact_id
    assert output["prototype_similarity"]["similarities"]["0"] == 0.9
    assert output["prototype_similarity"]["source_artifact_id"] == reference.artifact_id
    assert output["diagnostics"]["class_logits"] == [2.0, -0.2]
    assert output["cluster_evidence"] is None

    reduced = project_v2_result_set(
        result_set, V2ResponseOptions(frozenset({"class_probabilities"})), "classification",
        supported_outputs=supported, default_outputs=supported,
    ).results[0].output
    assert reduced["class_probabilities"] is not None
    assert reduced["embedding"] is None and reduced["knn"] is None
    assert reduced["secondary_heads"] is None and reduced["diagnostics"] is None

    inferred = project_v2_result_set(result_set, V2ResponseOptions(), "classification").results[0].output
    assert inferred["knn"] is not None and inferred["prototype_similarity"] is not None


def test_v2_decoder_rejects_missing_required_image_field():
    request_id = str(uuid.uuid4())
    item = _item(request_id)
    reference = _reference("embedding")
    result_set = InferenceResultSet(model=reference)
    result_set.append(InferenceResult(
        request_id=request_id, item_id=item.item_id, model=reference,
        result_set_id=result_set.result_set_id, input_sha256=item.input_sha256,
        output={"type": "embedding", "embedding": ArrayPayload(np.array([1.0, 0.0], dtype="float32")), "embedding_normalized": True},
    ))
    encoded = encode_v2_inference_result_set(project_v2_result_set(result_set, V2ResponseOptions(), "classification"))
    with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    manifest = json.loads(arrays["manifest"].tobytes().decode("utf-8"))
    del manifest["results"][0]["output"]["knn"]
    arrays["manifest"] = np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype="uint8")
    rebuilt = io.BytesIO()
    np.savez_compressed(rebuilt, **arrays)
    with pytest.raises(V2InferenceTransportError, match="missing required fields"):
        decode_v2_inference_result_set(rebuilt.getvalue())
