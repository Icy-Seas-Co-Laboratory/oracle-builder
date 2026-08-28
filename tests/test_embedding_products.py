from __future__ import annotations

import json

import numpy as np
from tensorflow import keras

from oracle_builder.classification.features import build_embedding_model
from oracle_builder.inference.bundle import InferenceBundle
from oracle_builder.inference.contracts import InferenceItem, ModelReference
from oracle_builder.registry import get_model_builder
from oracle_builder.saving.load_test import load_model_for_run, run_load_tests
from oracle_builder.saving.save_model import save_model_artifacts


def _config():
    return {
        "run": {"task": "embedding", "model": "simple_cnn", "run_id": "run-1"},
        "data": {"input_shape": [8, 8, 1], "num_classes": 2},
        "model": {"base_filters": 2, "embedding_dim": 4, "normalize_embeddings": True},
        "training": {"loss": "sparse_categorical_crossentropy"},
        "preprocessing": {"resize_mode": "fit_pad", "normalization": "dtype", "rescale": True},
        "output": {"export_savedmodel": True},
        "inference": {},
    }


def test_embedding_product_has_only_embedding_contract_and_round_trips(tmp_path):
    config = _config()
    classifier = get_model_builder("simple_cnn")(config)
    embedding_model = build_embedding_model(classifier)
    save_model_artifacts(embedding_model, tmp_path, config)

    manifest = json.loads((tmp_path / "model" / "model_manifest.json").read_text())
    assert manifest["task"] == "embedding"
    assert manifest["outputs"] == {
        "primary": "embedding",
        "embedding": True,
        "embedding_dimension": 4,
        "embedding_normalized": True,
    }
    report = run_load_tests(tmp_path, config)
    assert report["prediction_test_passed"] is True
    loaded = load_model_for_run(tmp_path, config, prefer_savedmodel=True)
    values = loaded.predict_embedding(np.zeros((2, 8, 8, 1), dtype="float32"))
    assert values.shape == (2, 4)


def test_embedding_bundle_returns_embedding_without_cluster_state():
    config = _config()
    classifier = get_model_builder("simple_cnn")(config)
    embedding_model = build_embedding_model(classifier)
    bundle = InferenceBundle(
        embedding_model,
        config,
        ModelReference("artifact", "run", "embedding", "simple_cnn"),
    )
    result = bundle.predict(
        InferenceItem.from_array(np.zeros((8, 8, 1), dtype="uint8"))
    )
    assert result.status == "ok"
    assert result.output["type"] == "embedding"
    assert result.output["embedding"].values.shape == (4,)
    assert "clustering_evidence" not in result.output
    result_set = bundle.predict_batch(
        [
            InferenceItem.from_array(np.zeros((8, 8, 1), dtype="uint8")),
            InferenceItem.from_array(np.ones((8, 8, 1), dtype="uint8")),
        ]
    )
    assert [item.status for item in result_set.results] == ["ok", "ok"]
    assert all(item.output["type"] == "embedding" for item in result_set.results)
