from __future__ import annotations

from typing import Any

from tensorflow import keras
from tensorflow.keras import layers


def _residual_block(x, filters: int, activation: str, dropout: float, name: str):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding="same", activation=activation, name=f"{name}_conv1")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(x)
    if int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", name=f"{name}_projection")(shortcut)
    x = layers.Add(name=f"{name}_residual")([x, shortcut])
    x = layers.Activation(activation, name=f"{name}_activation")(x)
    if dropout:
        x = layers.Dropout(dropout, name=f"{name}_dropout")(x)
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
        x = _residual_block(x, base * (2**level), activation, dropout if level else 0.0, f"encoder_{level}")
        skips.append(x)
        x = layers.MaxPooling2D(name=f"pool_{level}")(x)

    x = _residual_block(x, base * (2**depth), activation, dropout, "bottleneck")
    for level in reversed(range(depth)):
        x = layers.UpSampling2D(name=f"upsample_{level}")(x)
        x = layers.Concatenate(name=f"skip_{level}")([x, skips[level]])
        x = _residual_block(x, base * (2**level), activation, dropout if level else 0.0, f"decoder_{level}")

    outputs = layers.Conv2D(output_channels, 1, activation=final_activation, name="segmentation")(x)
    return keras.Model(inputs, outputs, name="residual_unet")
