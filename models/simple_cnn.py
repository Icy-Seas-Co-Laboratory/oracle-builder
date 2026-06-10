from __future__ import annotations

from typing import Any

from tensorflow import keras
from tensorflow.keras import layers


def build_model(config: dict[str, Any]):
    input_shape = tuple(config["data"]["input_shape"])
    num_classes = int(config["data"]["num_classes"])
    base = int(config.get("model", {}).get("base_filters", 32))
    dropout = float(config.get("model", {}).get("dropout", 0.2))

    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(base, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(base * 2, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(base * 4, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="simple_cnn")

