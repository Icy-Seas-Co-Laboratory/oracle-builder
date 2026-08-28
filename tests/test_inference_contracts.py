from __future__ import annotations

import json
import uuid

import numpy as np
import pytest

from oracle_builder.inference import (
    ArrayPayload,
    InferenceBundle,
    InferenceItem,
    InferenceResultSet,
    ModelReference,
    SourceReference,
    run_connector,
)


def model_reference(task: str = "classification") -> ModelReference:
    return ModelReference(
        artifact_id=str(uuid.uuid4()),
        run_id=str(uuid.uuid4()),
        task=task,
        architecture="simple_cnn" if task == "classification" else "unet",
        artifact_fingerprint="a" * 64,
    )


def test_inference_item_and_array_assets_have_uuid_and_hash_identity():
    source = SourceReference("pelagia", "roi", "pelagia-roi-42")
    first = ArrayPayload(np.arange(9, dtype="uint8").reshape(3, 3))
    second = ArrayPayload(np.arange(9, dtype="uint8").reshape(3, 3))
    item = InferenceItem(inputs={"image": first}, source=source)

    assert uuid.UUID(first.asset_id)
    assert first.sha256 == second.sha256
    assert len(item.input_sha256) == 64
    encoded = item.to_dict(include_data=False)
    assert encoded["schema_name"] == "oracle_builder.inference_item"
    assert encoded["source"]["resource_id"] == "pelagia-roi-42"
    assert "data_base64" not in encoded["inputs"]["image"]


def test_result_set_json_lines_is_a_streamable_envelope():
    execution = {
        "gpu_accelerated": False,
        "accelerator": "cpu",
        "device_type": "CPU",
        "device_name": "/device:CPU:0",
    }
    result_set = InferenceResultSet(
        model=model_reference(),
        execution=execution,
    ).complete()
    events = [json.loads(value) for value in result_set.to_json_lines()]

    assert events[0]["event"] == "result_set_start"
    assert events[0]["execution"] == execution
    assert events[-1]["event"] == "result_set_complete"
    assert events[-1]["counts"]["requested"] == 0


def test_connector_is_in_memory_by_default(tmp_path):
    class EchoBundle:
        model_reference = model_reference()

        def predict(
            self, item, *, result_set_id=None, sequence_number=None
        ):
            from oracle_builder.inference.contracts import InferenceResult

            return InferenceResult(
                request_id=item.request_id,
                item_id=item.item_id,
                model=self.model_reference,
                output={
                    "type": "classification",
                    "logits": [0.0],
                    "probabilities": [{"class_index": 0, "probability": 1.0}],
                },
                input_sha256=item.input_sha256,
                result_set_id=result_set_id,
                sequence_number=sequence_number,
            )

    result_set = run_connector(
        EchoBundle(),
        [InferenceItem.from_array(np.zeros((2, 2), dtype="uint8"))],
    )

    assert result_set.counts["succeeded"] == 1
    assert list(tmp_path.iterdir()) == []


def test_classification_bundle_uses_one_model_call_for_a_supplied_batch():
    class BatchRecordingModel:
        def __init__(self):
            self.batch_sizes = []

        def predict_outputs(self, values, verbose=0):
            self.batch_sizes.append(int(values.shape[0]))
            batch = int(values.shape[0])
            return {
                "logits": np.tile([[1.0, 2.0]], (batch, 1)),
                "probabilities": np.tile([[0.25, 0.75]], (batch, 1)),
                "features": np.tile([[0.6, 0.8]], (batch, 1)),
            }

    config = {
        "run": {"task": "classification", "model": "test"},
        "data": {"input_shape": [4, 4, 1]},
        "model": {"normalize_embeddings": True},
        "preprocessing": {
            "resize_mode": "fit_pad",
            "normalization": "dtype",
            "rescale": True,
            "channel_mode": "grayscale",
            "interpolation": "bilinear",
        },
    }
    model = BatchRecordingModel()
    bundle = InferenceBundle(model, config, model_reference())
    items = [
        InferenceItem.from_array(np.full((3, 5), value, dtype="uint8"))
        for value in (20, 80, 160)
    ]

    result_set = bundle.predict_batch(items)

    assert model.batch_sizes == [3]
    assert [result.request_id for result in result_set.results] == [
        item.request_id for item in items
    ]
    assert all(result.status == "ok" for result in result_set.results)
    assert result_set.execution["accelerator"] in {"gpu", "cpu"}
    assert isinstance(result_set.execution["gpu_accelerated"], bool)
    assert all(result.output["decision"]["class_index"] == 1 for result in result_set.results)


def test_classification_bundle_returns_logits_probabilities_and_embedding():
    tf = pytest.importorskip("tensorflow")
    from oracle_builder.registry import get_model_builder

    config = {
        "run": {"task": "classification", "model": "simple_cnn"},
        "data": {"input_shape": [8, 8, 1], "num_classes": 2},
        "dataset": {
            "labels": [
                {"label_id": str(uuid.uuid4()), "class_index": 0, "name": "a"},
                {"label_id": str(uuid.uuid4()), "class_index": 1, "name": "b"},
            ]
        },
        "model": {
            "base_filters": 2,
            "dropout": 0.0,
            "embedding_dim": 4,
            "normalize_embeddings": True,
        },
        "preprocessing": {
            "resize_mode": "fit_pad",
            "normalization": "dtype",
            "rescale": True,
            "channel_mode": "auto",
            "interpolation": "bilinear",
        },
    }
    model = get_model_builder("simple_cnn")(config)
    bundle = InferenceBundle(model, config, model_reference())
    item = InferenceItem.from_array(np.full((5, 7), 128, dtype="uint8"))

    result = bundle.predict(item)

    assert result.status == "ok"
    assert result.output["type"] == "classification"
    assert result.output["logits_source"] == "model"
    assert len(result.output["logits"]) == 2
    probabilities = [
        row["probability"] for row in result.output["probabilities"]
    ]
    assert np.isclose(sum(probabilities), 1.0)
    assert result.output["embedding"].values.shape == (4,)
    assert uuid.UUID(result.result_id)
    assert len(result.input_sha256) == 64


def test_segmentation_bundle_returns_logits_and_hashed_array_outputs():
    pytest.importorskip("tensorflow")
    from oracle_builder.registry import get_model_builder

    config = {
        "run": {"task": "segmentation", "model": "unet"},
        "data": {"input_shape": [8, 8, 1], "output_shape": [8, 8, 1]},
        "model": {
            "base_filters": 2,
            "depth": 1,
            "dropout": 0.0,
            "final_activation": "sigmoid",
        },
        "training": {"segmentation_target": "validated_mask"},
        "evaluation": {"segmentation_threshold": 0.5},
        "tiling": {"enabled": False},
    }
    model = get_model_builder("unet")(config)
    bundle = InferenceBundle(model, config, model_reference("segmentation"))

    result = bundle.predict(
        InferenceItem.from_array(np.zeros((8, 8, 1), dtype="float32"))
    )

    assert result.status == "ok"
    assert result.output["logits_source"] == "model"
    assert result.output["logits"].values.shape == (8, 8, 1)
    assert result.output["probability_map"].values.shape == (8, 8, 1)
    assert len(result.output["mask"].sha256) == 64
