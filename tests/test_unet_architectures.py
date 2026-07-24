from __future__ import annotations

import numpy as np
import pytest
from tensorflow import keras

from oracle_builder.registry import get_model_builder
from oracle_builder.training.losses import BinaryCrossentropySoftDice


def architecture_config(model_name: str, deep_supervision: bool = False):
    return {
        "run": {"task": "segmentation", "model": model_name, "seed": 123},
        "data": {"input_shape": [32, 32, 2], "output_shape": [32, 32, 1]},
        "model": {
            "base_filters": 4,
            "depth": 2,
            "dropout": 0.0,
            "activation": "relu",
            "final_activation": "sigmoid",
            "deep_supervision": deep_supervision,
        },
    }


@pytest.mark.parametrize(
    ("registry_name", "model_name"),
    [
        ("residual_unet", "residual_unet"),
        ("resunet", "residual_unet"),
        ("unet_plus_plus", "unet_plus_plus"),
        ("unetpp", "unet_plus_plus"),
    ],
)
def test_new_unet_architectures_build_expected_output(registry_name, model_name):
    model = get_model_builder(registry_name)(architecture_config(registry_name))

    assert model.name == model_name
    assert model.output_shape == (None, 32, 32, 1)
    prediction = model.predict(np.zeros((1, 32, 32, 2), dtype="float32"), verbose=0)
    assert prediction.shape == (1, 32, 32, 1)


def test_residual_unet_contains_projection_residual_blocks():
    model = get_model_builder("residual_unet")(architecture_config("residual_unet"))

    assert any(isinstance(layer, keras.layers.Add) for layer in model.layers)
    assert model.get_layer("encoder_0_projection") is not None


def test_unet_plus_plus_contains_nested_dense_skip_paths():
    model = get_model_builder("unet_plus_plus")(architecture_config("unet_plus_plus"))

    nested_skip = model.get_layer("nested_skip_0_2")
    assert len(nested_skip.input) == 3


def test_unet_plus_plus_deep_supervision_keeps_single_segmentation_output():
    model = get_model_builder("unet_plus_plus")(
        architecture_config("unet_plus_plus", deep_supervision=True)
    )
    model.compile(optimizer="adam", loss=BinaryCrossentropySoftDice())

    assert model.output_shape == (None, 32, 32, 1)
    assert model.get_layer("segmentation") is not None
    loss = model.train_on_batch(
        np.zeros((1, 32, 32, 2), dtype="float32"),
        np.zeros((1, 32, 32, 1), dtype="float32"),
    )
    assert np.isfinite(loss)
