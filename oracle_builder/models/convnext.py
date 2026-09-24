"""Small, channel-agnostic ConvNeXt encoders for the composable classifier."""

from __future__ import annotations

from typing import Any

from tensorflow.keras import layers

from oracle_builder.classification.features import (
    build_composable_classification_model,
    classifier_inputs,
    classifier_normalization,
    stratum_conditioning_input,
)


CONVNEXT_VARIANTS = {
    "convnext_tiny": ([3, 3, 9, 3], [96, 192, 384, 768]),
    "convnext_small": ([3, 3, 27, 3], [96, 192, 384, 768]),
}


def _block(x, width: int, config: dict[str, Any], name: str):
    shortcut = x
    x = layers.DepthwiseConv2D(7, padding="same", name=f"{name}_depthwise")(x)
    x = classifier_normalization(config, width, f"{name}_norm")(x)
    x = layers.Conv2D(4 * width, 1, activation="gelu", name=f"{name}_expand")(x)
    x = layers.Conv2D(width, 1, name=f"{name}_project")(x)
    drop = float(config.get("model", {}).get("drop_path", 0.0))
    if drop:
        x = layers.Dropout(drop, noise_shape=(None, 1, 1, 1), name=f"{name}_drop_path")(x)
    return layers.Add(name=f"{name}_add")([shortcut, x])


def build_model(config: dict[str, Any]):
    model = config.get("model", {})
    requested = str(config.get("run", {}).get("model", "convnext_tiny")).lower().replace("-", "_")
    variant = str(model.get("variant", requested)).lower().replace("-", "_")
    if variant not in CONVNEXT_VARIANTS:
        raise ValueError(f"Unknown ConvNeXt variant {variant!r}; choose from {sorted(CONVNEXT_VARIANTS)}")
    default_depths, default_widths = CONVNEXT_VARIANTS[variant]
    depths = [int(value) for value in model.get("stage_depths", default_depths)]
    widths = [int(value) for value in model.get("stage_widths", default_widths)]
    if len(depths) != 4 or len(widths) != 4 or any(value < 1 for value in [*depths, *widths]):
        raise ValueError("ConvNeXt stage_depths and stage_widths must each contain four positive integers")
    kernel = int(model.get("stem_kernel_size", 4))
    stride = int(model.get("stem_stride", 4))
    if kernel < 1 or stride < 1:
        raise ValueError("ConvNeXt stem kernel and stride must be positive")
    image, metadata = classifier_inputs(tuple(config["data"]["input_shape"]), config)
    stratum_dimension = stratum_conditioning_input(config)
    x = layers.Conv2D(widths[0], kernel, strides=stride, padding="same", name="stem_conv")(image)
    x = classifier_normalization(config, widths[0], "stem_norm")(x)
    for stage, (depth, width) in enumerate(zip(depths, widths, strict=True)):
        if stage:
            x = classifier_normalization(config, int(x.shape[-1]), f"downsample{stage}_norm")(x)
            x = layers.Conv2D(width, 2, strides=2, padding="same", name=f"downsample{stage}_conv")(x)
        for block in range(depth):
            x = _block(x, width, config, f"stage{stage + 1}_block{block + 1}")
    return build_composable_classification_model(
        image=image, feature_map=x, metadata=metadata,
        num_classes=int(config["data"]["num_classes"]), config=config,
        name=variant, stratum_dimension=stratum_dimension,
    )
