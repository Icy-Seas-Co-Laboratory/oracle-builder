from __future__ import annotations

from typing import Any

from tensorflow import keras
from tensorflow.keras import layers


def _residual_block(x, filters: int, stride: int = 1):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same")(shortcut)
    x = layers.Add()([x, shortcut])
    return layers.Activation("relu")(x)


def build_model(config: dict[str, Any]):
    input_shape = tuple(config["data"]["input_shape"])
    num_classes = int(config["data"]["num_classes"])
    base = int(config.get("model", {}).get("base_filters", 32))
    dropout = float(config.get("model", {}).get("dropout", 0.2))

    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(base, 3, padding="same", activation="relu")(inputs)
    x = _residual_block(x, base)
    x = _residual_block(x, base * 2, stride=2)
    x = _residual_block(x, base * 4, stride=2)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="resnet_like")

