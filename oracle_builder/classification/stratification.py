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
    if str(settings(config).get("weight_sharing", "shared")) != "shared":
        raise ValueError("classification.stratification.weight_sharing must be 'shared'")
    if assignment_policy(config) not in {"smallest_fitting", "largest_not_exceeding"}:
        raise ValueError(
            "classification.stratification.assignment_policy must be "
            "'smallest_fitting' or 'largest_not_exceeding'"
        )
    supra_epochs = settings(config).get("supra_epochs", 5)
    if (
        isinstance(supra_epochs, bool)
        or not isinstance(supra_epochs, int)
        or supra_epochs < 1
    ):
        raise ValueError("classification.stratification.supra_epochs must be a positive integer")
    routing = settings(config).get("training_routing", {})
    if not isinstance(routing, dict):
        raise ValueError("classification.stratification.training_routing must be a table")
    probability = float(routing.get("adjacent_lower_probability", 0.10))
    if not 0 <= probability <= 1:
        raise ValueError("classification.stratification.training_routing.adjacent_lower_probability must be in [0, 1]")
    scheduler = settings(config).get("cycle_scheduler", {})
    if not isinstance(scheduler, dict):
        raise ValueError("classification.stratification.cycle_scheduler must be a table")
    if str(scheduler.get("aggregation", "sample_weighted")) not in {
        "sample_weighted",
        "equal_strata",
    }:
        raise ValueError(
            "classification.stratification.cycle_scheduler.aggregation must be "
            "'sample_weighted' or 'equal_strata'"
        )
    if int(scheduler.get("reduce_lr_patience", 3)) < 1:
        raise ValueError(
            "classification.stratification.cycle_scheduler.reduce_lr_patience "
            "must be positive"
        )
    factor = float(scheduler.get("reduce_lr_factor", 0.5))
    if not 0 < factor < 1:
        raise ValueError(
            "classification.stratification.cycle_scheduler.reduce_lr_factor "
            "must be in (0, 1)"
        )
    if config.get("run", {}).get("task") != "classification":
        raise ValueError("classification.stratification is only supported for classification")
    if config.get("self_supervised", config.get("pretraining", {})).get("enabled", False):
        raise ValueError("Resolution stratification is not yet compatible with self-supervised pretraining")


def assignment_policy(config: dict[str, Any]) -> str:
    """Return the canonical ROI-to-resolution routing policy."""
    value = str(settings(config).get("assignment_policy", "smallest_fitting"))
    aliases = {
        "prefer_next_largest": "smallest_fitting",
        "prefer_next_smallest": "largest_not_exceeding",
    }
    return aliases.get(value, value)


def stratum_for_shape(
    shape: Any,
    configured_dimensions: list[int],
    *,
    policy: str = "smallest_fitting",
) -> int:
    array_shape = tuple(int(value) for value in shape)
    if len(array_shape) < 2:
        raise ValueError(f"Cannot assign a resolution stratum for shape {shape!r}")
    maximum = max(array_shape[:2])
    if policy == "smallest_fitting":
        for dimension in configured_dimensions:
            if maximum <= dimension:
                return dimension
        return configured_dimensions[-1]
    if policy == "largest_not_exceeding":
        for dimension in reversed(configured_dimensions):
            if maximum >= dimension:
                return dimension
        return configured_dimensions[0]
    raise ValueError(f"Unsupported resolution assignment policy {policy!r}")


def stratum_for_array(array: Any, config: dict[str, Any]) -> int:
    return stratum_for_shape(
        np.asarray(array).shape,
        dimensions(config),
        policy=assignment_policy(config),
    )


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


def supra_epochs(config: dict[str, Any]) -> int:
    """Number of consecutive parent epochs a child remains active."""
    return int(settings(config).get("supra_epochs", 5))


def supra_epoch_schedule(config: dict[str, Any], total_epochs: int):
    """Yield ``(start, stop, dimension)`` child turns.

    Each child handles a contiguous block of parent epochs before the next
    resolution is loaded.  A final partial block is emitted when necessary.
    """
    block_size = supra_epochs(config)
    for start in range(0, int(total_epochs), block_size):
        stop = min(int(total_epochs), start + block_size)
        for dimension in dimensions(config):
            yield start, stop, dimension


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


def summarize_records(
    records: list[dict[str, Any]], config: dict[str, Any], *, split: str, epoch: int | None = None
) -> dict[str, Any]:
    """Return auditable canonical and (optionally) epoch routing counts."""
    configured = dimensions(config)
    canonical = {dimension: 0 for dimension in configured}
    routed = {dimension: 0 for dimension in configured}
    classes: dict[int, dict[int, int]] = {dimension: {} for dimension in configured}
    for record in records:
        shape = record.get("original_shape")
        if not shape:
            raise ValueError("Stratification requires original_shape on every classification record")
        assigned = stratum_for_shape(
            shape, configured, policy=assignment_policy(config)
        )
        canonical[assigned] += 1
        selected = training_stratum(
            assigned, item_id=str(record["uuid"]), epoch=int(epoch), config=config
        ) if epoch is not None else assigned
        routed[selected] += 1
        label = record.get("class_index")
        if label is not None:
            classes[selected][int(label)] = classes[selected].get(int(label), 0) + 1
    return {
        "split": split,
        "epoch": epoch,
        "canonical_counts": canonical,
        "routed_counts": routed,
        "class_counts": classes,
        "batch_plan": batch_plan(config),
    }
