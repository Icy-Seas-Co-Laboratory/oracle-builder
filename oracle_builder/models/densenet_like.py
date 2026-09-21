from __future__ import annotations

from typing import Any

from tensorflow import keras
from tensorflow.keras import layers

from oracle_builder.classification.features import (
    classification_head, classifier_inputs, classifier_normalization,
    join_auxiliary_features, stratum_conditioning_input,
)


def _dense_block(x, growth_rate: int, layers_count: int, config: dict[str, Any], name: str):
    features = [x]
    for index in range(layers_count):
        y = layers.Concatenate()(features)
        y = layers.Conv2D(growth_rate, 3, padding="same", use_bias=False, name=f"{name}_conv{index}")(y)
        y = classifier_normalization(config, growth_rate, f"{name}_norm{index}")(y)
        y = layers.Activation("relu", name=f"{name}_relu{index}")(y)
        features.append(y)
    return layers.Concatenate()(features)


def _transition(x, filters: int, config: dict[str, Any], name: str):
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = classifier_normalization(config, filters, f"{name}_norm")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    return layers.AveragePooling2D(pool_size=2)(x)


def build_model(config: dict[str, Any]):
    input_shape = tuple(config["data"]["input_shape"])
    num_classes = int(config["data"]["num_classes"])
    base = int(config.get("model", {}).get("base_filters", 24))

    inputs, metadata = classifier_inputs(input_shape, config)
    stratum_dimension = stratum_conditioning_input(config)
    x = layers.Conv2D(base, 3, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = classifier_normalization(config, base, "stem_norm")(x)
    x = layers.Activation("relu", name="stem_relu")(x)
    x = _dense_block(x, base // 2, 3, config, "dense1")
    x = _transition(x, base * 2, config, "transition1")
    x = _dense_block(x, base // 2, 3, config, "dense2")
    x = _transition(x, base * 4, config, "transition2")
    x = _dense_block(x, base // 2, 3, config, "dense3")
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = join_auxiliary_features(x, metadata)
    outputs = classification_head(
        x, num_classes, config, dropout_default=0.2,
        stratum_dimension=stratum_dimension,
    )
    model_inputs = [inputs] + ([metadata] if metadata is not None else []) + ([stratum_dimension] if stratum_dimension is not None else [])
    return keras.Model(model_inputs if len(model_inputs) > 1 else inputs, outputs, name="densenet_like")
