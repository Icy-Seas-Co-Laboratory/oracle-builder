from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


MODEL_REGISTRY = {
    "simple_cnn": "models.simple_cnn:build_model",
    "unet": "models.unet:build_model",
    "residual_unet": "models.residual_unet:build_model",
    "resunet": "models.residual_unet:build_model",
    "unet_plus_plus": "models.unet_plus_plus:build_model",
    "unetpp": "models.unet_plus_plus:build_model",
    "resnet_like": "models.resnet_like:build_model",
    "densenet_like": "models.densenet_like:build_model",
    "resnet": "models.resnet:build_model",
    "densenet": "models.densenet:build_model",
    "efficientnet": "models.efficientnet:build_model",
    "resnet18": "models.resnet:build_model",
    "resnet34": "models.resnet:build_model",
    "resnet50": "models.resnet:build_model",
    "resnet101": "models.resnet:build_model",
    "resnet152": "models.resnet:build_model",
    "densenet121": "models.densenet:build_model",
    "densenet169": "models.densenet:build_model",
    "densenet201": "models.densenet:build_model",
    "efficientnet_b0": "models.efficientnet:build_model",
    "efficientnet_b1": "models.efficientnet:build_model",
    "efficientnet_b2": "models.efficientnet:build_model",
    "efficientnet_b3": "models.efficientnet:build_model",
    "efficientnet_b4": "models.efficientnet:build_model",
    "efficientnet_b5": "models.efficientnet:build_model",
    "efficientnet_b6": "models.efficientnet:build_model",
    "efficientnet_b7": "models.efficientnet:build_model",
}


def get_model_builder(name: str) -> Callable[[dict[str, Any]], Any]:
    target = MODEL_REGISTRY.get(name)
    if not target:
        raise KeyError(f"Unknown model '{name}'. Available models: {', '.join(sorted(MODEL_REGISTRY))}")
    module_name, function_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)
