"""Native EfficientNetV2 image encoders for arbitrary-channel scientific data.

The implementation intentionally does not depend on ImageNet application
weights: Oracle Builder inputs may have derived/non-RGB channels.  It retains
the defining EfficientNetV2 fused-MBConv early stages and MBConv/SE later
stages, while the V2 representation graph supplies pooling and prediction.
"""

from __future__ import annotations

import math
from typing import Any

from tensorflow import keras
from tensorflow.keras import layers

from oracle_builder.classification.features import (
    build_composable_classification_model,
    classification_head,
    classifier_inputs,
    classifier_normalization,
    join_auxiliary_features,
    stratum_conditioning_input,
    uses_composable_graph,
)


# width coefficient, depth coefficient, classifier dropout.  B0--B3 are the
# compact variants appropriate for the small/medium image grids this project
# supports; model options can still override every scaling coefficient.
EFFICIENTNET_V2_VARIANTS = {
    "efficientnet_v2_b0": (1.0, 1.0, 0.2),
    "efficientnet_v2_b1": (1.0, 1.1, 0.2),
    "efficientnet_v2_b2": (1.1, 1.2, 0.3),
    "efficientnet_v2_b3": (1.2, 1.4, 0.3),
    "efficientnet_v2_s": (1.0, 1.8, 0.2),
    "efficientnet_v2_m": (1.25, 2.2, 0.3),
    "efficientnet_v2_l": (1.5, 2.6, 0.4),
}
LEGACY_EFFICIENTNET_VARIANTS = {f"efficientnet_b{index}" for index in range(8)}

# kind, expansion, kernel, stride, output filters, repeats, SE ratio
BLOCKS = [
    ("fused", 1, 3, 1, 16, 1, 0.0),
    ("fused", 4, 3, 2, 32, 2, 0.0),
    ("fused", 4, 3, 2, 48, 2, 0.0),
    ("mbconv", 4, 3, 2, 96, 3, 0.25),
    ("mbconv", 6, 3, 1, 112, 5, 0.25),
    ("mbconv", 6, 3, 2, 192, 8, 0.25),
]


def _round_filters(filters: int, width: float) -> int:
    value = filters * width
    rounded = max(8, int(value + 4) // 8 * 8)
    return rounded + 8 if rounded < 0.9 * value else rounded


def _round_repeats(repeats: int, depth: float) -> int:
    return int(math.ceil(repeats * depth))


def _squeeze_excite(x, input_filters: int, expanded_filters: int, ratio: float, name: str):
    if ratio <= 0:
        return x
    reduced = max(1, int(input_filters * ratio))
    gate = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_se_pool")(x)
    gate = layers.Conv2D(reduced, 1, activation="swish", name=f"{name}_se_reduce")(gate)
    gate = layers.Conv2D(expanded_filters, 1, activation="sigmoid", name=f"{name}_se_expand")(gate)
    return layers.Multiply(name=f"{name}_se_scale")([x, gate])


def _fused_mbconv(x, output_filters: int, expansion: int, kernel: int, stride: int, *, name: str, config: dict[str, Any]):
    input_filters = int(x.shape[-1])
    shortcut = x
    expanded = input_filters * expansion
    if expansion == 1:
        x = layers.Conv2D(output_filters, kernel, strides=stride, padding="same", use_bias=False, name=f"{name}_fused")(x)
        x = classifier_normalization(config, output_filters, f"{name}_fused_bn")(x)
        x = layers.Activation("swish", name=f"{name}_fused_activation")(x)
    else:
        x = layers.Conv2D(expanded, kernel, strides=stride, padding="same", use_bias=False, name=f"{name}_expand")(x)
        x = classifier_normalization(config, expanded, f"{name}_expand_bn")(x)
        x = layers.Activation("swish", name=f"{name}_expand_activation")(x)
        x = layers.Conv2D(output_filters, 1, padding="same", use_bias=False, name=f"{name}_project")(x)
        x = classifier_normalization(config, output_filters, f"{name}_project_bn")(x)
    if stride == 1 and input_filters == output_filters:
        x = layers.Add(name=f"{name}_add")([x, shortcut])
    return x


def _mbconv(x, output_filters: int, expansion: int, kernel: int, stride: int, se_ratio: float, *, name: str, config: dict[str, Any]):
    input_filters = int(x.shape[-1])
    shortcut = x
    expanded = input_filters * expansion
    if expansion != 1:
        x = layers.Conv2D(expanded, 1, padding="same", use_bias=False, name=f"{name}_expand")(x)
        x = classifier_normalization(config, expanded, f"{name}_expand_bn")(x)
        x = layers.Activation("swish", name=f"{name}_expand_activation")(x)
    x = layers.DepthwiseConv2D(kernel, strides=stride, padding="same", use_bias=False, name=f"{name}_depthwise")(x)
    x = classifier_normalization(config, expanded, f"{name}_depthwise_bn")(x)
    x = layers.Activation("swish", name=f"{name}_depthwise_activation")(x)
    x = _squeeze_excite(x, input_filters, expanded, se_ratio, name)
    x = layers.Conv2D(output_filters, 1, padding="same", use_bias=False, name=f"{name}_project")(x)
    x = classifier_normalization(config, output_filters, f"{name}_project_bn")(x)
    if stride == 1 and input_filters == output_filters:
        x = layers.Add(name=f"{name}_add")([x, shortcut])
    return x


def _normalise_variant(value: str) -> str:
    value = value.lower().replace("-", "_")
    aliases = {
        "efficientnetv2": "efficientnet_v2_b0",
        "efficientnetv2_b0": "efficientnet_v2_b0",
        "efficientnetv2_b1": "efficientnet_v2_b1",
        "efficientnetv2_b2": "efficientnet_v2_b2",
        "efficientnetv2_b3": "efficientnet_v2_b3",
        "efficientnetv2_s": "efficientnet_v2_s",
        "efficientnetv2_m": "efficientnet_v2_m",
        "efficientnetv2_l": "efficientnet_v2_l",
        "efficientnet_v2": "efficientnet_v2_b0",
    }
    return aliases.get(value, value)


def build_model(config: dict[str, Any]):
    model_config = config.get("model", {})
    requested = _normalise_variant(str(config.get("run", {}).get("model", "efficientnet_v2")))
    configured_variant = _normalise_variant(str(model_config.get("variant", "")))
    # The maintained EfficientNet V1 setup file is also used by non-GUI setup
    # callers as a baseline.  Its ``efficientnet_b0`` value must not override
    # a V2 architecture selected through ``run.model``.
    if configured_variant and configured_variant not in EFFICIENTNET_V2_VARIANTS:
        if configured_variant not in LEGACY_EFFICIENTNET_VARIANTS:
            raise ValueError(
                f"Unknown EfficientNetV2 variant {configured_variant!r}; "
                f"choose from {sorted(EFFICIENTNET_V2_VARIANTS)}"
            )
        variant = requested
    else:
        variant = configured_variant or requested
    if variant not in EFFICIENTNET_V2_VARIANTS:
        raise ValueError(f"Unknown EfficientNetV2 variant {variant!r}; choose from {sorted(EFFICIENTNET_V2_VARIANTS)}")
    default_width, default_depth, default_dropout = EFFICIENTNET_V2_VARIANTS[variant]
    width = float(model_config.get("width_coefficient", default_width))
    depth = float(model_config.get("depth_coefficient", default_depth))
    dropout = float(model_config.get("dropout", default_dropout))
    stem_filters = int(model_config.get("stem_filters", _round_filters(32, width)))
    top_filters = int(model_config.get("top_filters", _round_filters(1280, width)))
    stem_kernel = int(model_config.get("stem_kernel_size", 3))
    stem_stride = int(model_config.get("stem_stride", 2))
    se_ratio = float(model_config.get("se_ratio", 0.25))
    if width <= 0 or depth <= 0 or stem_filters < 1 or top_filters < 1 or stem_kernel < 1 or stem_stride < 1:
        raise ValueError("EfficientNetV2 width, depth, and filter parameters must be positive")
    if not 0 <= se_ratio <= 1:
        raise ValueError("model.se_ratio must be in [0, 1]")

    inputs, metadata = classifier_inputs(tuple(config["data"]["input_shape"]), config)
    stratum_dimension = stratum_conditioning_input(config)
    x = layers.Conv2D(stem_filters, stem_kernel, strides=stem_stride, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = classifier_normalization(config, stem_filters, "stem_bn")(x)
    x = layers.Activation("swish", name="stem_activation")(x)
    for stage, (kind, expansion, kernel, stride, filters, repeats, block_se_ratio) in enumerate(BLOCKS, start=1):
        output_filters = _round_filters(filters, width)
        for index in range(_round_repeats(repeats, depth)):
            block_name = f"stage{stage}_block{index + 1}"
            current_stride = stride if index == 0 else 1
            if kind == "fused":
                x = _fused_mbconv(x, output_filters, expansion, kernel, current_stride, name=block_name, config=config)
            else:
                x = _mbconv(x, output_filters, expansion, kernel, current_stride, se_ratio if block_se_ratio else 0.0, name=block_name, config=config)
    x = layers.Conv2D(top_filters, 1, padding="same", use_bias=False, name="top_conv")(x)
    x = classifier_normalization(config, top_filters, "top_bn")(x)
    x = layers.Activation("swish", name="top_activation")(x)
    if uses_composable_graph(config):
        return build_composable_classification_model(
            image=inputs, feature_map=x, metadata=metadata,
            num_classes=int(config["data"]["num_classes"]), config=config,
            name=variant, stratum_dimension=stratum_dimension,
        )
    pooled = layers.GlobalAveragePooling2D(name="global_pool")(x)
    pooled = join_auxiliary_features(pooled, metadata)
    outputs = classification_head(pooled, int(config["data"]["num_classes"]), config, dropout_default=dropout, stratum_dimension=stratum_dimension)
    model_inputs = [inputs] + ([metadata] if metadata is not None else []) + ([stratum_dimension] if stratum_dimension is not None else [])
    return keras.Model(model_inputs if len(model_inputs) > 1 else inputs, outputs, name=variant)
