from __future__ import annotations

import math
from typing import Any

from tensorflow import keras
from tensorflow.keras import layers

from oracle_builder.classification.features import (
    build_composable_classification_model, classification_head, classifier_inputs, join_auxiliary_features,
    classifier_normalization,
    stratum_conditioning_input, uses_composable_graph,
)


def _filter_counts(model_config: dict[str, Any]) -> list[int]:
    """Resolve the width of each Simple CNN convolutional block.

    ``filter_multiplier`` makes the small baseline model useful across a wider
    range of image sizes while retaining its original 32, 64, 128 default.
    Rounding is applied per block so non-integer multipliers still produce a
    valid number of convolution filters.
    """
    try:
        layers_count = int(model_config.get("conv_layers", 3))
        base_filters = int(model_config.get("base_filters", 32))
        multiplier = float(model_config.get("filter_multiplier", 2.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Simple CNN conv_layers, base_filters, and filter_multiplier must be numeric"
        ) from exc

    if layers_count < 1:
        raise ValueError("Simple CNN conv_layers must be at least 1")
    if base_filters < 1:
        raise ValueError("Simple CNN base_filters must be at least 1")
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Simple CNN filter_multiplier must be a finite value greater than 0")

    return [max(1, round(base_filters * multiplier**index)) for index in range(layers_count)]


def build_model(config: dict[str, Any]):
    input_shape = tuple(config["data"]["input_shape"])
    num_classes = int(config["data"]["num_classes"])
    model_config = config.get("model", {})
    filters_by_layer = _filter_counts(model_config)

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

    x = inputs
    for index, filters in enumerate(filters_by_layer, start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=not uses_configurable_norm)(x)
        x = normalize(x, filters, f"conv{index}_norm")
        x = layers.Activation("relu")(x)
        if index < len(filters_by_layer):
            x = layers.MaxPooling2D()(x)
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
