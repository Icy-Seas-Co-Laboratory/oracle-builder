from __future__ import annotations

from typing import Any

from tensorflow import keras
from tensorflow.keras import layers


def _conv_block(x, filters: int, activation: str, dropout: float, name: str):
    x = layers.Conv2D(filters, 3, padding="same", activation=activation, name=f"{name}_conv1")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation=activation, name=f"{name}_conv2")(x)
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
    deep_supervision = bool(model_config.get("deep_supervision", False))

    inputs = keras.Input(shape=input_shape)
    nodes: dict[tuple[int, int], Any] = {}
    nodes[(0, 0)] = _conv_block(inputs, base, activation, 0.0, "node_0_0")
    for level in range(1, depth + 1):
        pooled = layers.MaxPooling2D(name=f"pool_{level - 1}")(nodes[(level - 1, 0)])
        nodes[(level, 0)] = _conv_block(
            pooled, base * (2**level), activation, dropout, f"node_{level}_0"
        )

    for stage in range(1, depth + 1):
        for level in range(depth - stage + 1):
            upsampled = layers.UpSampling2D(name=f"upsample_{level}_{stage}")(
                nodes[(level + 1, stage - 1)]
            )
            dense_skips = [nodes[(level, prior)] for prior in range(stage)]
            merged = layers.Concatenate(name=f"nested_skip_{level}_{stage}")(
                [*dense_skips, upsampled]
            )
            nodes[(level, stage)] = _conv_block(
                merged,
                base * (2**level),
                activation,
                dropout if level else 0.0,
                f"node_{level}_{stage}",
            )

    if deep_supervision:
        heads = [
            layers.Conv2D(
                output_channels,
                1,
                activation=final_activation,
                name=f"supervision_{stage}",
            )(nodes[(0, stage)])
            for stage in range(1, depth + 1)
        ]
        outputs = heads[0] if len(heads) == 1 else layers.Average(name="segmentation")(heads)
    else:
        outputs = layers.Conv2D(
            output_channels, 1, activation=final_activation, name="segmentation"
        )(nodes[(0, depth)])
    return keras.Model(inputs, outputs, name="unet_plus_plus")
