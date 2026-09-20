"""Resolution-stratum policy shared by training and inference."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def settings(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("classification", {}).get("stratification", {})
    return dict(value) if isinstance(value, dict) else {}


def enabled(config: dict[str, Any]) -> bool:
    return bool(settings(config).get("enabled", False))


def dimensions(config: dict[str, Any]) -> list[int]:
    return [int(value) for value in settings(config).get("dimensions", [])]


def validate(config: dict[str, Any]) -> None:
    if not enabled(config):
        return
    values = dimensions(config)
    if not values or values != sorted(set(values)) or any(value < 2 for value in values):
        raise ValueError("classification.stratification.dimensions must be ascending unique integers >= 2")
    if str(settings(config).get("basis", "max_original_dimension")) != "max_original_dimension":
        raise ValueError("classification.stratification.basis must be 'max_original_dimension'")
    if str(settings(config).get("batch_size_policy", "constant_input_tensor")) != "constant_input_tensor":
        raise ValueError("classification.stratification.batch_size_policy must be 'constant_input_tensor'")
    routing = settings(config).get("training_routing", {})
    if not isinstance(routing, dict):
        raise ValueError("classification.stratification.training_routing must be a table")
    probability = float(routing.get("adjacent_lower_probability", 0.10))
    if not 0 <= probability <= 1:
        raise ValueError("classification.stratification.training_routing.adjacent_lower_probability must be in [0, 1]")
    if config.get("run", {}).get("task") != "classification":
        raise ValueError("classification.stratification is only supported for classification")
    if config.get("self_supervised", config.get("pretraining", {})).get("enabled", False):
        raise ValueError("Resolution stratification is not yet compatible with self-supervised pretraining")


def stratum_for_shape(shape: Any, configured_dimensions: list[int]) -> int:
    array_shape = tuple(int(value) for value in shape)
    if len(array_shape) < 2:
        raise ValueError(f"Cannot assign a resolution stratum for shape {shape!r}")
    maximum = max(array_shape[:2])
    for dimension in configured_dimensions:
        if maximum <= dimension:
            return dimension
    return configured_dimensions[-1]


def stratum_for_array(array: Any, config: dict[str, Any]) -> int:
    return stratum_for_shape(np.asarray(array).shape, dimensions(config))


def architecture_supported(config: dict[str, Any]) -> bool:
    """Return whether this package-owned CNN family supports resolution bundles."""
    return str(config.get("run", {}).get("model", "")).lower() in {
        "simple_cnn", "resnet_like", "densenet_like", "resnet", "resnet18",
        "resnet34", "resnet50", "resnet101", "resnet152", "densenet",
        "densenet121", "densenet169", "densenet201", "efficientnet",
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
        "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    }


def batch_plan(config: dict[str, Any]) -> dict[int, int]:
    """Keep the raw input tensor count at or below the smallest stratum's."""
    values = dimensions(config)
    if not values:
        return {}
    channels = int(config["data"]["input_shape"][-1])
    baseline = int(config["data"].get("batch_size", 16))
    smallest = values[0]
    budget = baseline * smallest * smallest * channels
    return {
        dimension: max(1, budget // (dimension * dimension * channels))
        for dimension in values
    }


def training_stratum(
    canonical: int, *, item_id: str, epoch: int, config: dict[str, Any]
) -> int:
    """Return reproducible training-only stochastic routing for one sample."""
    routing = settings(config).get("training_routing", {})
    if not isinstance(routing, dict) or not routing.get("enabled", True):
        return canonical
    values = dimensions(config)
    index = values.index(canonical)
    probability = float(routing.get("adjacent_lower_probability", 0.10))
    if index == 0 or probability <= 0:
        return canonical
    seed = int(routing.get("seed", config.get("run", {}).get("seed", 123)))
    digest = hashlib.sha256(f"{seed}:{epoch}:{item_id}".encode("utf-8")).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    return values[index - 1] if draw < probability else canonical
