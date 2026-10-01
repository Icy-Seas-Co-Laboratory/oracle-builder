"""Client-facing V2 transport tests with no Oracle ML-runtime import."""
from __future__ import annotations

import uuid

import numpy as np

from oracle_data_contracts.inference_v2 import (
    InferenceItem,
    V2InferenceRequest,
    V2ResponseOptions,
    decode_v2_inference_request,
    decode_v2_inference_result_set,
    encode_v2_inference_request,
)


def test_shared_request_codec_round_trips_image_and_candidate_mask():
    request_id = str(uuid.uuid4())
    image_item = InferenceItem.from_array(
        np.arange(16, dtype=np.uint8).reshape(4, 4),
        item_id=str(uuid.uuid4()), request_id=request_id,
        metadata={"pelagia_roi_id": "roi-1"},
    )
    decoded = decode_v2_inference_request(encode_v2_inference_request(
        V2InferenceRequest(request_id, "classification", [image_item])
    ))
    assert decoded.items[0].input_sha256 == image_item.input_sha256
    np.testing.assert_array_equal(decoded.items[0].inputs["image"].values, image_item.inputs["image"].values)

    mask_item = InferenceItem.from_array(
        np.ones((4, 4), dtype=np.uint8), candidate_mask=np.eye(4, dtype=np.uint8),
        request_id=request_id,
    )
    mask_request = V2InferenceRequest(
        request_id, "mask_refinement", [mask_item],
        V2ResponseOptions(frozenset({"mask", "probability_map"}), max_output_bytes=4096),
    )
    decoded_mask = decode_v2_inference_request(encode_v2_inference_request(mask_request))
    np.testing.assert_array_equal(decoded_mask.items[0].inputs["candidate_mask"].values, mask_item.inputs["candidate_mask"].values)


def test_shared_decoder_reads_builder_encoded_image_result():
    # Importing this encoder is test setup only.  The public decoder above
    # remains usable with oracle-data-contracts and NumPy alone.
    from oracle_builder.inference.contracts import ArrayPayload, InferenceResult, InferenceResultSet, ModelReference
    from oracle_builder.inference.transport_v2 import V2ResponseOptions as BuilderOptions, encode_v2_inference_result_set, project_v2_result_set

    request_id = str(uuid.uuid4())
    model = ModelReference(str(uuid.uuid4()), str(uuid.uuid4()), "embedding", "test", "a" * 64)
    results = InferenceResultSet(model=model)
    results.append(InferenceResult(
        request_id=request_id, item_id=str(uuid.uuid4()), model=model,
        result_set_id=results.result_set_id, input_sha256="b" * 64,
        output={"type": "embedding", "embedding": ArrayPayload(np.array([0.6, 0.8], dtype=np.float32)), "embedding_normalized": True},
    ))
    payload = encode_v2_inference_result_set(project_v2_result_set(results, BuilderOptions(), "classification"))
    decoded = decode_v2_inference_result_set(payload)
    output = decoded["results"][0]["output"]
    assert output["model_task"] == "embedding"
    assert output["class_probabilities"] is None
    np.testing.assert_allclose(output["embedding"]["values"], [0.6, 0.8])
