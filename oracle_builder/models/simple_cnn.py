from __future__ import annotations

from typing import Any

from tensorflow import keras
from tensorflow.keras import layers

from oracle_builder.classification.features import classification_head


def build_model(config: dict[str, Any]):
    input_shape = tuple(config["data"]["input_shape"])
    num_classes = int(config["data"]["num_classes"])
    base = int(config.get("model", {}).get("base_filters", 32))

    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(base, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(base * 2, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(base * 4, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    outputs = classification_head(x, num_classes, config, dropout_default=0.2)
    return keras.Model(inputs, outputs, name="simple_cnn")
