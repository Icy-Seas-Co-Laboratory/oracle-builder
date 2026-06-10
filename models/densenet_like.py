from __future__ import annotations

from typing import Any

from tensorflow import keras
from tensorflow.keras import layers


def _dense_block(x, growth_rate: int, layers_count: int):
    features = [x]
    for _ in range(layers_count):
        y = layers.Concatenate()(features)
        y = layers.Conv2D(growth_rate, 3, padding="same", activation="relu")(y)
        features.append(y)
    return layers.Concatenate()(features)


def _transition(x, filters: int):
    x = layers.Conv2D(filters, 1, padding="same", activation="relu")(x)
    return layers.AveragePooling2D()(x)


def build_model(config: dict[str, Any]):
    input_shape = tuple(config["data"]["input_shape"])
    num_classes = int(config["data"]["num_classes"])
    base = int(config.get("model", {}).get("base_filters", 24))
    dropout = float(config.get("model", {}).get("dropout", 0.2))

    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(base, 3, padding="same", activation="relu")(inputs)
    x = _dense_block(x, base // 2, 3)
    x = _transition(x, base * 2)
    x = _dense_block(x, base // 2, 3)
    x = _transition(x, base * 4)
    x = _dense_block(x, base // 2, 3)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="densenet_like")

