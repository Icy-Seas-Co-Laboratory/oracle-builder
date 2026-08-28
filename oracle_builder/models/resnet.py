from __future__ import annotations

from typing import Any

from tensorflow import keras
from tensorflow.keras import layers

from oracle_builder.classification.features import classification_head


RESNET_VARIANTS = {
    "resnet18": ("basic", [2, 2, 2, 2]),
    "resnet34": ("basic", [3, 4, 6, 3]),
    "resnet50": ("bottleneck", [3, 4, 6, 3]),
    "resnet101": ("bottleneck", [3, 4, 23, 3]),
    "resnet152": ("bottleneck", [3, 8, 36, 3]),
}

# ROI classification commonly uses small batches.  Retain moving statistics
# more conservatively than the ImageNet-style 0.9 setting so validation and
# inference do not track an individual mini-batch too closely.
_BATCH_NORM_MOMENTUM = 0.99
_BATCH_NORM_EPSILON = 1e-3
_KERNEL_INITIALIZER = "he_normal"


def _conv_bn(x, filters, kernel_size, stride=1, activation=True, name="conv"):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_initializer=_KERNEL_INITIALIZER,
        name=name,
    )(x)
    x = layers.BatchNormalization(
        momentum=_BATCH_NORM_MOMENTUM,
        epsilon=_BATCH_NORM_EPSILON,
        name=f"{name}_bn",
    )(x)
    if activation:
        x = layers.Activation("relu", name=f"{name}_relu")(x)
    return x


def _basic_block(x, filters: int, stride: int, name: str):
    shortcut = x
    x = _conv_bn(x, filters, 3, stride=stride, name=f"{name}_conv1")
    x = _conv_bn(x, filters, 3, activation=False, name=f"{name}_conv2")
    if stride != 1 or int(shortcut.shape[-1]) != filters:
        shortcut = _conv_bn(
            shortcut, filters, 1, stride=stride, activation=False, name=f"{name}_projection"
        )
    x = layers.Add(name=f"{name}_add")([x, shortcut])
    return layers.Activation("relu", name=f"{name}_out")(x)


def _bottleneck_block(x, filters: int, stride: int, name: str):
    shortcut = x
    expanded_filters = filters * 4
    x = _conv_bn(x, filters, 1, name=f"{name}_conv1")
    x = _conv_bn(x, filters, 3, stride=stride, name=f"{name}_conv2")
    x = _conv_bn(x, expanded_filters, 1, activation=False, name=f"{name}_conv3")
    if stride != 1 or int(shortcut.shape[-1]) != expanded_filters:
        shortcut = _conv_bn(
            shortcut,
            expanded_filters,
            1,
            stride=stride,
            activation=False,
            name=f"{name}_projection",
        )
    x = layers.Add(name=f"{name}_add")([x, shortcut])
    return layers.Activation("relu", name=f"{name}_out")(x)


def build_model(config: dict[str, Any]):
    model_config = config.get("model", {})
    requested_model = str(config.get("run", {}).get("model", "resnet")).lower()
    default_variant = requested_model if requested_model in RESNET_VARIANTS else "resnet18"
    variant = str(model_config.get("variant", default_variant)).lower()
    if variant not in RESNET_VARIANTS:
        raise ValueError(f"Unknown ResNet variant {variant!r}; choose from {sorted(RESNET_VARIANTS)}")
    block_type, default_blocks = RESNET_VARIANTS[variant]
    block_counts = [int(value) for value in model_config.get("block_counts", default_blocks)]
    if len(block_counts) != 4 or any(value < 1 for value in block_counts):
        raise ValueError("model.block_counts must contain four positive integers")

    input_shape = tuple(config["data"]["input_shape"])
    num_classes = int(config["data"]["num_classes"])
    base_filters = int(model_config.get("base_filters", 64))
    stem_kernel = int(model_config.get("stem_kernel_size", 3))
    stem_stride = int(model_config.get("stem_stride", 1))
    stem_pool = bool(model_config.get("stem_pool", False))
    if base_filters < 1 or stem_kernel < 1 or stem_stride < 1:
        raise ValueError("ResNet filter, stem kernel, and stem stride parameters must be positive")
    block = _basic_block if block_type == "basic" else _bottleneck_block

    inputs = keras.Input(shape=input_shape)
    x = _conv_bn(inputs, base_filters, stem_kernel, stride=stem_stride, name="stem_conv")
    if stem_pool:
        x = layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)
    for stage, count in enumerate(block_counts):
        filters = base_filters * (2**stage)
        for index in range(count):
            stride = 2 if stage > 0 and index == 0 else 1
            x = block(x, filters, stride, name=f"stage{stage + 1}_block{index + 1}")
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    outputs = classification_head(x, num_classes, config, normalize_default=False)
    return keras.Model(inputs, outputs, name=variant)
