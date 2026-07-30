from __future__ import annotations

from typing import Any

from tensorflow import keras
from tensorflow.keras import layers

from oracle_builder.classification.features import classification_head


def _dense_block(x, growth_rate: int, layers_count: int):
    features = [x]
    for _ in range(layers_count):
        y = layers.Concatenate()(features)
        y = layers.Conv2D(growth_rate, 3, padding="same", activation="relu")(y)
        features.append(y)
    return layers.Concatenate()(features)


def _transition(x, filters: int):
    x = layers.Conv2D(filters, 1, padding="same", activation="relu")(x)
    return layers.AveragePooling2D(pool_size=2)(x)


def build_model(config: dict[str, Any]):
    input_shape = tuple(config["data"]["input_shape"])
    num_classes = int(config["data"]["num_classes"])
    base = int(config.get("model", {}).get("base_filters", 24))

    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(base, 3, padding="same", activation="relu")(inputs)
    x = _dense_block(x, base // 2, 3)
    x = _transition(x, base * 2)
    x = _dense_block(x, base // 2, 3)
    x = _transition(x, base * 4)
    x = _dense_block(x, base // 2, 3)
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    outputs = classification_head(x, num_classes, config, dropout_default=0.2)
    return keras.Model(inputs, outputs, name="densenet_like")
