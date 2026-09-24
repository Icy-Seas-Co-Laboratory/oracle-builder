from __future__ import annotations

import numpy as np
import tensorflow as tf

from oracle_builder.artifacts.representations import write_representation_artifacts
from oracle_builder.classification.features import build_feature_model
from oracle_builder.models.resnet import build_model
from oracle_builder.models.efficientnet_v2 import build_model as build_efficientnet_v2
from oracle_builder.posthoc import open_embedding_cache


def _config():
    return {
        "architecture": {"version": 2},
        "run": {"task": "classification", "model": "resnet", "run_id": "run-v2"},
        "artifact": {"artifact_id": "artifact-v2"},
        "dataset": {"dataset_id": "dataset-v2", "fingerprint_sha256": "abc"},
        "data": {"input_shape": [16, 16, 1], "num_classes": 3},
        "model": {
            "base_filters": 4,
            "block_counts": [1, 1, 1, 1],
            "auxiliary_features": [{"name": "area", "source": "metadata.area"}],
        },
        "pooling": {"type": "avg_max"},
        "image_embedding": {"projection": {"type": "linear", "output_dim": 8}},
        "metadata": {"encoder": {"type": "mlp", "hidden_units": [4], "output_dim": 3}},
        "fusion": {"type": "projected", "output_dim": 6},
        "classifier": {"type": "cosine"},
        "normalization": {"type": "group", "groups": "auto"},
        "classification": {"stratification": {}},
        "output": {"intermediate_artifacts": {"enabled": True, "representations": ["image_embedding", "fused_embedding"], "batch_size": 2}},
    }


def test_v2_classifier_exposes_distinct_embedding_stages():
    model = build_model(_config())
    outputs = build_feature_model(model)(
        {"image": np.ones((2, 16, 16, 1), dtype="float32"), "metadata": np.ones((2, 1), dtype="float32")}
    )
    assert {"feature_map", "image_embedding", "metadata_embedding", "fused_embedding", "projection_embedding", "logits", "probabilities"}.issubset(outputs)
    assert outputs["image_embedding"].shape[-1] == 8
    assert outputs["metadata_embedding"].shape[-1] == 3
    assert outputs["fused_embedding"].shape[-1] == 6


def test_v2_prototype_head_and_efficientnet_v2_encoder_build():
    config = _config()
    config["classifier"] = {
        "type": "prototype", "prototypes_per_class": 2, "prototype_metric": "euclidean",
    }
    config["run"]["model"] = "efficientnet_v2_b0"
    config["model"] = {
        "width_coefficient": 0.25, "depth_coefficient": 0.2,
        "stem_filters": 8, "top_filters": 16,
    }
    config["data"]["input_shape"] = [16, 16, 1]
    model = build_efficientnet_v2(config)
    assert model.name == "efficientnet_v2_b0"
    assert model.get_layer("logits").prototypes.shape == (3, 2, 6)
    values = model(np.ones((2, 16, 16, 1), dtype="float32"))
    assert values.shape == (2, 3)


class _Ref:
    def __init__(self, index: int):
        self.index = index

    def record(self):
        return {"uuid": f"sample-{self.index}", "class_index": self.index % 3, "split": "test", "original_shape": [10, 12], "metadata": {"area": self.index}}


class _Index:
    split = "test"
    refs = [_Ref(0), _Ref(1), _Ref(2)]


def test_representation_artifacts_stream_to_mmap_cache(tmp_path):
    config = _config()
    config["model"].pop("auxiliary_features")
    model = build_model(config)
    values = np.ones((3, 16, 16, 1), dtype="float32")
    dataset = tf.data.Dataset.from_tensor_slices((values, np.arange(3, dtype="int64"))).batch(2)
    report = write_representation_artifacts(model, dataset, _Index(), config, tmp_path)
    assert {item["representation"] for item in report["caches"]} == {"image_embedding", "fused_embedding"}
    cache = open_embedding_cache(tmp_path / "intermediates" / "representations" / "test" / "image_embedding")
    assert cache.shape == (3, 8)
    assert cache.records()[0]["sample_id"] == "sample-0"
