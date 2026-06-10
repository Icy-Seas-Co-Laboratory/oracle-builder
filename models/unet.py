from __future__ import annotations

from typing import Any

from tensorflow import keras
from tensorflow.keras import layers


def _conv_block(x, filters: int, activation: str, dropout: float):
    x = layers.Conv2D(filters, 3, padding="same", activation=activation)(x)
    x = layers.Conv2D(filters, 3, padding="same", activation=activation)(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    return x


def build_model(config: dict[str, Any]):
    model_config = config.get("model", {})
    input_shape = tuple(config["data"]["input_shape"])
    output_channels = int(config["data"].get("output_shape", [None, None, 1])[-1])
    base = int(model_config.get("base_filters", 32))
    depth = int(model_config.get("depth", 4))
    dropout = float(model_config.get("dropout", 0.1))
    activation = model_config.get("activation", "relu")
    final_activation = model_config.get("final_activation", "sigmoid")

    inputs = keras.Input(shape=input_shape)
    x = inputs
    skips = []
    for level in range(depth):
        x = _conv_block(x, base * (2**level), activation, dropout if level else 0.0)
        skips.append(x)
        x = layers.MaxPooling2D()(x)

    x = _conv_block(x, base * (2**depth), activation, dropout)

    for level in reversed(range(depth)):
        x = layers.UpSampling2D()(x)
        x = layers.Concatenate()([x, skips[level]])
        x = _conv_block(x, base * (2**level), activation, dropout if level else 0.0)

    outputs = layers.Conv2D(output_channels, 1, activation=final_activation)(x)
    return keras.Model(inputs, outputs, name="unet")

