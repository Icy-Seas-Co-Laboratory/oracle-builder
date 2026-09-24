"""Compact MobileNetV3-style encoders compatible with arbitrary channels."""

from __future__ import annotations

from typing import Any

from tensorflow.keras import layers

from oracle_builder.classification.features import (
    build_composable_classification_model,
    classifier_inputs,
    classifier_normalization,
    stratum_conditioning_input,
)


MOBILENET_VARIANTS = {
    "mobilenet_v3_small": ([16, 24, 40, 48, 96], [1, 2, 2, 2, 3]),
    "mobilenet_v3_large": ([16, 24, 40, 80, 112, 160], [1, 2, 3, 3, 2, 3]),
}


def _block(x, filters: int, stride: int, config: dict[str, Any], name: str):
    input_channels = int(x.shape[-1])
    expansion = int(config.get("model", {}).get("expansion", 4))
    expanded = max(input_channels, input_channels * expansion)
    y = layers.Conv2D(expanded, 1, use_bias=False, name=f"{name}_expand")(x)
    y = classifier_normalization(config, expanded, f"{name}_expand_norm")(y)
    y = layers.Activation("swish", name=f"{name}_expand_activation")(y)
    y = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False, name=f"{name}_depthwise")(y)
    y = classifier_normalization(config, expanded, f"{name}_depthwise_norm")(y)
    y = layers.Activation("swish", name=f"{name}_depthwise_activation")(y)
    se = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_se_pool")(y)
    se = layers.Conv2D(max(1, expanded // 4), 1, activation="relu", name=f"{name}_se_reduce")(se)
    se = layers.Conv2D(expanded, 1, activation="sigmoid", name=f"{name}_se_expand")(se)
    y = layers.Multiply(name=f"{name}_se_scale")([y, se])
    y = layers.Conv2D(filters, 1, use_bias=False, name=f"{name}_project")(y)
    y = classifier_normalization(config, filters, f"{name}_project_norm")(y)
    if stride == 1 and input_channels == filters:
        y = layers.Add(name=f"{name}_add")([x, y])
    return y


def build_model(config: dict[str, Any]):
    model = config.get("model", {})
    requested = str(config.get("run", {}).get("model", "mobilenet_v3_small")).lower().replace("-", "_")
    variant = str(model.get("variant", requested)).lower().replace("-", "_")
    if variant not in MOBILENET_VARIANTS:
        raise ValueError(f"Unknown MobileNet variant {variant!r}; choose from {sorted(MOBILENET_VARIANTS)}")
    default_widths, default_repeats = MOBILENET_VARIANTS[variant]
    widths = [int(value) for value in model.get("stage_widths", default_widths)]
    repeats = [int(value) for value in model.get("stage_repeats", default_repeats)]
    if len(widths) != len(repeats) or not widths or any(value < 1 for value in [*widths, *repeats]):
        raise ValueError("MobileNet stage_widths and stage_repeats must be equal-length positive lists")
    kernel = int(model.get("stem_kernel_size", 3))
    stride = int(model.get("stem_stride", 2))
    stem_filters = int(model.get("stem_filters", 16))
    image, metadata = classifier_inputs(tuple(config["data"]["input_shape"]), config)
    stratum_dimension = stratum_conditioning_input(config)
    x = layers.Conv2D(stem_filters, kernel, strides=stride, padding="same", use_bias=False, name="stem_conv")(image)
    x = classifier_normalization(config, stem_filters, "stem_norm")(x)
    x = layers.Activation("swish", name="stem_activation")(x)
    for stage, (filters, count) in enumerate(zip(widths, repeats, strict=True)):
        for block in range(count):
            x = _block(x, filters, 2 if stage and block == 0 else 1, config, f"stage{stage + 1}_block{block + 1}")
    return build_composable_classification_model(
        image=image, feature_map=x, metadata=metadata,
        num_classes=int(config["data"]["num_classes"]), config=config,
        name=variant, stratum_dimension=stratum_dimension,
    )
