from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


MODEL_REGISTRY = {
    "simple_cnn": "oracle_builder.models.simple_cnn:build_model",
    "unet": "oracle_builder.models.unet:build_model",
    "residual_unet": "oracle_builder.models.residual_unet:build_model",
    "resunet": "oracle_builder.models.residual_unet:build_model",
    "unet_plus_plus": "oracle_builder.models.unet_plus_plus:build_model",
    "unetpp": "oracle_builder.models.unet_plus_plus:build_model",
    "resnet_like": "oracle_builder.models.resnet_like:build_model",
    "densenet_like": "oracle_builder.models.densenet_like:build_model",
    "resnet": "oracle_builder.models.resnet:build_model",
    "densenet": "oracle_builder.models.densenet:build_model",
    "efficientnet": "oracle_builder.models.efficientnet:build_model",
    "resnet18": "oracle_builder.models.resnet:build_model",
    "resnet34": "oracle_builder.models.resnet:build_model",
    "resnet50": "oracle_builder.models.resnet:build_model",
    "resnet101": "oracle_builder.models.resnet:build_model",
    "resnet152": "oracle_builder.models.resnet:build_model",
    "densenet121": "oracle_builder.models.densenet:build_model",
    "densenet169": "oracle_builder.models.densenet:build_model",
    "densenet201": "oracle_builder.models.densenet:build_model",
    "efficientnet_b0": "oracle_builder.models.efficientnet:build_model",
    "efficientnet_b1": "oracle_builder.models.efficientnet:build_model",
    "efficientnet_b2": "oracle_builder.models.efficientnet:build_model",
    "efficientnet_b3": "oracle_builder.models.efficientnet:build_model",
    "efficientnet_b4": "oracle_builder.models.efficientnet:build_model",
    "efficientnet_b5": "oracle_builder.models.efficientnet:build_model",
    "efficientnet_b6": "oracle_builder.models.efficientnet:build_model",
    "efficientnet_b7": "oracle_builder.models.efficientnet:build_model",
}


def get_model_builder(name: str) -> Callable[[dict[str, Any]], Any]:
    target = MODEL_REGISTRY.get(name)
    if not target:
        raise KeyError(f"Unknown model '{name}'. Available models: {', '.join(sorted(MODEL_REGISTRY))}")
    module_name, function_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)
