from __future__ import annotations

import math
from typing import Any

from tensorflow import keras
from tensorflow.keras import layers

from oracle_builder.classification.features import classification_head


EFFICIENTNET_VARIANTS = {
    "efficientnet_b0": (1.0, 1.0, 0.2),
    "efficientnet_b1": (1.0, 1.1, 0.2),
    "efficientnet_b2": (1.1, 1.2, 0.3),
    "efficientnet_b3": (1.2, 1.4, 0.3),
    "efficientnet_b4": (1.4, 1.8, 0.4),
    "efficientnet_b5": (1.6, 2.2, 0.4),
    "efficientnet_b6": (1.8, 2.6, 0.5),
    "efficientnet_b7": (2.0, 3.1, 0.5),
}

BLOCKS = [
    # expansion, kernel, stride, input filters, output filters, repeats
    (1, 3, 1, 32, 16, 1),
    (6, 3, 2, 16, 24, 2),
    (6, 5, 2, 24, 40, 2),
    (6, 3, 2, 40, 80, 3),
    (6, 5, 1, 80, 112, 3),
    (6, 5, 2, 112, 192, 4),
    (6, 3, 1, 192, 320, 1),
]


def _round_filters(filters: int, width: float) -> int:
    value = filters * width
    rounded = max(8, int(value + 4) // 8 * 8)
    return rounded + 8 if rounded < 0.9 * value else rounded


def _round_repeats(repeats: int, depth: float) -> int:
    return int(math.ceil(repeats * depth))


def _mbconv(x, output_filters, expansion, kernel, stride, se_ratio, name):
    input_filters = int(x.shape[-1])
    expanded_filters = input_filters * expansion
    shortcut = x
    if expansion != 1:
        x = layers.Conv2D(
            expanded_filters, 1, padding="same", use_bias=False, name=f"{name}_expand"
        )(x)
        x = layers.BatchNormalization(name=f"{name}_expand_bn")(x)
        x = layers.Activation("swish", name=f"{name}_expand_activation")(x)
    x = layers.DepthwiseConv2D(
        kernel, strides=stride, padding="same", use_bias=False, name=f"{name}_depthwise"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_depthwise_bn")(x)
    x = layers.Activation("swish", name=f"{name}_depthwise_activation")(x)
    if se_ratio > 0:
        reduced_filters = max(1, int(input_filters * se_ratio))
        se = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_se_pool")(x)
        se = layers.Conv2D(reduced_filters, 1, activation="swish", name=f"{name}_se_reduce")(se)
        se = layers.Conv2D(expanded_filters, 1, activation="sigmoid", name=f"{name}_se_expand")(se)
        x = layers.Multiply(name=f"{name}_se_scale")([x, se])
    x = layers.Conv2D(
        output_filters, 1, padding="same", use_bias=False, name=f"{name}_project"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_project_bn")(x)
    if stride == 1 and input_filters == output_filters:
        x = layers.Add(name=f"{name}_add")([x, shortcut])
    return x


def build_model(config: dict[str, Any]):
    model_config = config.get("model", {})
    requested_model = str(config.get("run", {}).get("model", "efficientnet")).lower().replace(
        "-", "_"
    )
    default_variant = (
        requested_model if requested_model in EFFICIENTNET_VARIANTS else "efficientnet_b0"
    )
    variant = str(model_config.get("variant", default_variant)).lower().replace("-", "_")
    if variant not in EFFICIENTNET_VARIANTS:
        raise ValueError(
            f"Unknown EfficientNet variant {variant!r}; choose from {sorted(EFFICIENTNET_VARIANTS)}"
        )
    default_width, default_depth, default_dropout = EFFICIENTNET_VARIANTS[variant]
    width = float(model_config.get("width_coefficient", default_width))
    depth = float(model_config.get("depth_coefficient", default_depth))
    dropout = float(model_config.get("dropout", default_dropout))
    se_ratio = float(model_config.get("se_ratio", 0.25))
    stem_kernel = int(model_config.get("stem_kernel_size", 3))
    stem_stride = int(model_config.get("stem_stride", 2))
    stem_filters = int(model_config.get("stem_filters", _round_filters(32, width)))
    top_filters = int(model_config.get("top_filters", _round_filters(1280, width)))
    if (
        width <= 0
        or depth <= 0
        or stem_filters < 1
        or top_filters < 1
        or stem_kernel < 1
        or stem_stride < 1
    ):
        raise ValueError("EfficientNet width, depth, and filter parameters must be positive")
    if not 0 <= se_ratio <= 1:
        raise ValueError("model.se_ratio must be in [0, 1]")

    inputs = keras.Input(shape=tuple(config["data"]["input_shape"]))
    x = layers.Conv2D(
        stem_filters,
        stem_kernel,
        strides=stem_stride,
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("swish", name="stem_activation")(x)
    for stage, (expansion, kernel, stride, _in_filters, out_filters, repeats) in enumerate(BLOCKS):
        output_filters = _round_filters(out_filters, width)
        for index in range(_round_repeats(repeats, depth)):
            x = _mbconv(
                x,
                output_filters,
                expansion,
                kernel,
                stride if index == 0 else 1,
                se_ratio,
                name=f"stage{stage + 1}_block{index + 1}",
            )
    x = layers.Conv2D(top_filters, 1, padding="same", use_bias=False, name="top_conv")(x)
    x = layers.BatchNormalization(name="top_bn")(x)
    x = layers.Activation("swish", name="top_activation")(x)
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    outputs = classification_head(
        x,
        int(config["data"]["num_classes"]),
        config,
        dropout_default=default_dropout,
    )
    return keras.Model(inputs, outputs, name=variant)
