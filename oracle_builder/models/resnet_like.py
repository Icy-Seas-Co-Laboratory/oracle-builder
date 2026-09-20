from __future__ import annotations

from typing import Any

from tensorflow import keras
from tensorflow.keras import layers

from oracle_builder.classification.features import classification_head, classifier_inputs, join_auxiliary_features


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

    inputs, metadata = classifier_inputs(input_shape, config)
    x = layers.Conv2D(base, 3, padding="same", activation="relu")(inputs)
    x = _residual_block(x, base)
    x = _residual_block(x, base * 2, stride=2)
    x = _residual_block(x, base * 4, stride=2)
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = join_auxiliary_features(x, metadata)
    outputs = classification_head(x, num_classes, config, dropout_default=0.2)
    return keras.Model([inputs, metadata] if metadata is not None else inputs, outputs, name="resnet_like")
