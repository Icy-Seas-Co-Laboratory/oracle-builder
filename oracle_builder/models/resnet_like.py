from __future__ import annotations

from typing import Any

from tensorflow import keras
from tensorflow.keras import layers

from oracle_builder.classification.features import (
    build_composable_classification_model, classification_head, classifier_inputs, classifier_normalization,
    join_auxiliary_features, stratum_conditioning_input,
    uses_composable_graph,
)


def _residual_block(x, filters: int, config: dict[str, Any], name: str, stride: int = 1):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = classifier_normalization(config, filters, f"{name}_norm1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = classifier_normalization(config, filters, f"{name}_norm2")(x)
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = classifier_normalization(config, filters, f"{name}_shortcut_norm")(shortcut)
    x = layers.Add()([x, shortcut])
    return layers.Activation("relu")(x)


def build_model(config: dict[str, Any]):
    input_shape = tuple(config["data"]["input_shape"])
    num_classes = int(config["data"]["num_classes"])
    base = int(config.get("model", {}).get("base_filters", 32))

    inputs, metadata = classifier_inputs(input_shape, config)
    stratum_dimension = stratum_conditioning_input(config)
    x = layers.Conv2D(base, 3, padding="same", use_bias=False)(inputs)
    x = classifier_normalization(config, base, "stem_norm")(x)
    x = layers.Activation("relu")(x)
    x = _residual_block(x, base, config, "block1")
    x = _residual_block(x, base * 2, config, "block2", stride=2)
    x = _residual_block(x, base * 4, config, "block3", stride=2)
    if uses_composable_graph(config):
        return build_composable_classification_model(
            image=inputs, feature_map=x, metadata=metadata, num_classes=num_classes,
            config=config, name="resnet_like", stratum_dimension=stratum_dimension,
        )
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = join_auxiliary_features(x, metadata)
    outputs = classification_head(
        x, num_classes, config, dropout_default=0.2,
        stratum_dimension=stratum_dimension,
    )
    model_inputs = [inputs] + ([metadata] if metadata is not None else []) + ([stratum_dimension] if stratum_dimension is not None else [])
    return keras.Model(model_inputs if len(model_inputs) > 1 else inputs, outputs, name="resnet_like")
