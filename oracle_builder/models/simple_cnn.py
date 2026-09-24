from __future__ import annotations

from typing import Any

from tensorflow import keras
from tensorflow.keras import layers

from oracle_builder.classification.features import (
    build_composable_classification_model, classification_head, classifier_inputs, join_auxiliary_features,
    classifier_normalization,
    stratum_conditioning_input, uses_composable_graph,
)


def build_model(config: dict[str, Any]):
    input_shape = tuple(config["data"]["input_shape"])
    num_classes = int(config["data"]["num_classes"])
    base = int(config.get("model", {}).get("base_filters", 32))

    inputs, metadata = classifier_inputs(input_shape, config)
    stratum_dimension = stratum_conditioning_input(config)
    uses_group_norm = str(
        config.get("classification", {}).get("stratification", {}).get(
            "normalization", "batch"
        )
    ).lower() == "group"
    # V1 intentionally used un-normalized simple-CNN blocks unless resolution
    # stratification requested GroupNorm. V2 makes normalization a component
    # choice, so honor it without altering historical V1 graphs.
    uses_configurable_norm = uses_group_norm or uses_composable_graph(config)

    def normalize(value, channels: int, name: str):
        return classifier_normalization(config, channels, name)(value) if uses_configurable_norm else value

    x = layers.Conv2D(base, 3, padding="same", use_bias=not uses_configurable_norm)(inputs)
    x = normalize(x, base, "conv1_norm")
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(base * 2, 3, padding="same", use_bias=not uses_configurable_norm)(x)
    x = normalize(x, base * 2, "conv2_norm")
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(base * 4, 3, padding="same", use_bias=not uses_configurable_norm)(x)
    x = normalize(x, base * 4, "conv3_norm")
    x = layers.Activation("relu")(x)
    if uses_composable_graph(config):
        return build_composable_classification_model(
            image=inputs, feature_map=x, metadata=metadata, num_classes=num_classes,
            config=config, name="simple_cnn", stratum_dimension=stratum_dimension,
        )
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = join_auxiliary_features(x, metadata)
    outputs = classification_head(x, num_classes, config, dropout_default=0.2, stratum_dimension=stratum_dimension)
    model_inputs = [inputs] + ([metadata] if metadata is not None else []) + ([stratum_dimension] if stratum_dimension is not None else [])
    return keras.Model(model_inputs if len(model_inputs) > 1 else inputs, outputs, name="simple_cnn")
