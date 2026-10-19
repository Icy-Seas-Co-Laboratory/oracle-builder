from __future__ import annotations

import pytest
from tensorflow import keras

from oracle_builder.models.simple_cnn import build_model


def _config(model: dict | None = None) -> dict:
    return {
        "run": {"task": "classification", "model": "simple_cnn"},
        "data": {"input_shape": [32, 32, 1], "num_classes": 2},
        "model": model or {},
    }


def _convolutions(model):
    return [layer for layer in model.layers if isinstance(layer, keras.layers.Conv2D)]


def test_simple_cnn_default_topology_is_preserved():
    model = build_model(_config({"base_filters": 4}))

    assert [layer.filters for layer in _convolutions(model)] == [4, 8, 16]
    assert len([layer for layer in model.layers if isinstance(layer, keras.layers.MaxPooling2D)]) == 2


def test_simple_cnn_supports_configurable_depth_and_geometric_width():
    model = build_model(
        _config({"base_filters": 4, "conv_layers": 4, "filter_multiplier": 1.5})
    )

    assert [layer.filters for layer in _convolutions(model)] == [4, 6, 9, 14]
    assert len([layer for layer in model.layers if isinstance(layer, keras.layers.MaxPooling2D)]) == 3


@pytest.mark.parametrize(
    ("model_config", "message"),
    [
        ({"conv_layers": 0}, "conv_layers"),
        ({"base_filters": 0}, "base_filters"),
        ({"filter_multiplier": 0}, "filter_multiplier"),
    ],
)
def test_simple_cnn_rejects_invalid_topology(model_config, message):
    with pytest.raises(ValueError, match=message):
        build_model(_config(model_config))
