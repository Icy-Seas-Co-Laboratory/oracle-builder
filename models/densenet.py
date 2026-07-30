from __future__ import annotations

from typing import Any

from tensorflow import keras
from tensorflow.keras import layers

from oracle_builder.classification.features import classification_head


DENSENET_VARIANTS = {
    "densenet121": [6, 12, 24, 16],
    "densenet169": [6, 12, 32, 32],
    "densenet201": [6, 12, 48, 32],
}


def _dense_layer(x, growth_rate: int, bottleneck_multiplier: int, dropout: float, name: str):
    y = layers.BatchNormalization(name=f"{name}_bn1")(x)
    y = layers.Activation("relu", name=f"{name}_relu1")(y)
    y = layers.Conv2D(
        growth_rate * bottleneck_multiplier,
        1,
        padding="same",
        use_bias=False,
        name=f"{name}_bottleneck",
    )(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)
    y = layers.Activation("relu", name=f"{name}_relu2")(y)
    y = layers.Conv2D(growth_rate, 3, padding="same", use_bias=False, name=f"{name}_conv")(y)
    if dropout:
        y = layers.Dropout(dropout, name=f"{name}_dropout")(y)
    return layers.Concatenate(name=f"{name}_concat")([x, y])


def _transition(x, compression: float, name: str):
    filters = max(1, int(int(x.shape[-1]) * compression))
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_conv")(x)
    return layers.AveragePooling2D(2, strides=2, padding="same", name=f"{name}_pool")(x)


def build_model(config: dict[str, Any]):
    model_config = config.get("model", {})
    requested_model = str(config.get("run", {}).get("model", "densenet")).lower()
    default_variant = requested_model if requested_model in DENSENET_VARIANTS else "densenet121"
    variant = str(model_config.get("variant", default_variant)).lower()
    if variant not in DENSENET_VARIANTS:
        raise ValueError(f"Unknown DenseNet variant {variant!r}; choose from {sorted(DENSENET_VARIANTS)}")
    block_config = [
        int(value) for value in model_config.get("block_config", DENSENET_VARIANTS[variant])
    ]
    if len(block_config) != 4 or any(value < 1 for value in block_config):
        raise ValueError("model.block_config must contain four positive integers")

    input_shape = tuple(config["data"]["input_shape"])
    num_classes = int(config["data"]["num_classes"])
    growth_rate = int(model_config.get("growth_rate", 32))
    initial_filters = int(model_config.get("initial_filters", 64))
    bottleneck_multiplier = int(model_config.get("bottleneck_multiplier", 4))
    compression = float(model_config.get("compression", 0.5))
    stem_kernel = int(model_config.get("stem_kernel_size", 7))
    stem_stride = int(model_config.get("stem_stride", 2))
    stem_pool = bool(model_config.get("stem_pool", True))
    dropout = float(model_config.get("dropout", 0.0))
    if growth_rate < 1 or initial_filters < 1 or bottleneck_multiplier < 1:
        raise ValueError("DenseNet filter and bottleneck parameters must be positive")
    if stem_kernel < 1 or stem_stride < 1:
        raise ValueError("DenseNet stem kernel and stem stride must be positive")
    if not 0 < compression <= 1:
        raise ValueError("model.compression must be in (0, 1]")

    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(
        initial_filters,
        stem_kernel,
        strides=stem_stride,
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)
    if stem_pool:
        x = layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)
    for block_index, layer_count in enumerate(block_config):
        for layer_index in range(layer_count):
            x = _dense_layer(
                x,
                growth_rate,
                bottleneck_multiplier,
                dropout,
                name=f"dense{block_index + 1}_layer{layer_index + 1}",
            )
        if block_index < len(block_config) - 1:
            x = _transition(x, compression, name=f"transition{block_index + 1}")
    x = layers.BatchNormalization(name="final_bn")(x)
    x = layers.Activation("relu", name="final_relu")(x)
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    outputs = classification_head(x, num_classes, config)
    return keras.Model(inputs, outputs, name=variant)
