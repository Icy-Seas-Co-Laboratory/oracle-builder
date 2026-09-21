from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from oracle_builder.classification.stratification import validate
from oracle_builder.classification.stratified_data import add_stratum_dimension_input
from oracle_builder.models.resnet import build_model


def _config():
    return {
        "run": {"task": "classification", "model": "resnet"},
        "data": {"input_shape": [16, 16, 1], "num_classes": 3},
        "model": {"base_filters": 4, "block_counts": [1, 1, 1, 1]},
        "classification": {"stratification": {
            "enabled": True, "dimensions": [32, 64, 128],
            "normalization": "group", "group_norm_groups": 8,
            "conditioning": {"enabled": True, "embedding_dim": 4},
        }},
    }


def test_group_normalized_conditioned_resnet_accepts_runtime_dimension():
    model = build_model(_config())
    assert model.get_layer("stem_conv_bn").__class__.__name__ == "GroupNormalization"
    result = model({
        "image": np.ones((2, 16, 16, 1), dtype="float32"),
        "stratum_dimension": np.array([[32], [64]], dtype="int32"),
    })
    assert result.shape == (2, 3)
    assert model.get_layer("stratum_adapter").kernel_initializer.__class__.__name__.lower().startswith("zeros")


def test_conditioning_dataset_decorator_adds_constant_dimension():
    config = _config()
    dataset = tf.data.Dataset.from_tensor_slices((
        np.ones((2, 16, 16, 1), dtype="float32"), np.array([0, 1], dtype="int32"),
    )).batch(2)
    features, labels = next(iter(add_stratum_dimension_input(dataset, 64, config)))
    assert set(features) == {"image", "stratum_dimension"}
    np.testing.assert_array_equal(features["stratum_dimension"].numpy(), [[64], [64]])
    np.testing.assert_array_equal(labels.numpy(), [0, 1])


def test_conditioning_validation_rejects_invalid_embedding_size():
    config = _config()
    config["classification"]["stratification"]["conditioning"]["embedding_dim"] = 0
    with pytest.raises(ValueError, match="conditioning.embedding_dim"):
        validate(config)
