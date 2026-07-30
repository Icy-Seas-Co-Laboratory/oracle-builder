from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class InferenceBatchPlan:
    batch_size: int
    mode: str
    initial_candidate: int
    attempts: tuple[int, ...]
    memory_budget_mb: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "mode": self.mode,
            "initial_candidate": self.initial_candidate,
            "attempts": list(self.attempts),
            "memory_budget_mb": self.memory_budget_mb,
        }


def _power_of_two_at_most(value: int) -> int:
    if value <= 1:
        return 1
    return 2 ** int(math.floor(math.log2(value)))


def _activation_factor(config: dict[str, Any]) -> int:
    architecture = str(config.get("run", {}).get("model", "")).lower()
    if "efficientnet" in architecture or "densenet" in architecture:
        return 72
    if "resnet" in architecture:
        return 56
    if "unet" in architecture:
        return 80
    return 32


def estimate_inference_batch_size(config: dict[str, Any]) -> int:
    settings = config.get("inference", {})
    minimum = max(1, int(settings.get("minimum_batch_size", 1)))
    maximum = max(minimum, int(settings.get("maximum_batch_size", 64)))
    memory_budget_mb = max(64, int(settings.get("memory_budget_mb", 512)))
    sample_values = int(np.prod(config["data"]["input_shape"]))
    estimated_sample_bytes = max(
        sample_values * 4 * _activation_factor(config),
        1,
    )
    candidate = int(memory_budget_mb * 1024 * 1024 / estimated_sample_bytes)
    return max(minimum, min(maximum, _power_of_two_at_most(candidate)))


def resolve_inference_batch_size(
    model,
    config: dict[str, Any],
    *,
    announce: Callable[[str], None] | None = print,
) -> InferenceBatchPlan:
    """Resolve and verify a bounded inference batch without retaining data."""
    settings = config.get("inference", {})
    requested = settings.get("batch_size", "auto")
    minimum = max(1, int(settings.get("minimum_batch_size", 1)))
    maximum = max(minimum, int(settings.get("maximum_batch_size", 64)))
    memory_budget_mb = max(64, int(settings.get("memory_budget_mb", 512)))
    if isinstance(requested, str) and requested.lower() == "auto":
        mode = "auto"
        candidate = estimate_inference_batch_size(config)
    else:
        mode = "fixed"
        candidate = int(requested)
        if candidate < 1:
            raise ValueError("inference.batch_size must be 'auto' or a positive integer")
        candidate = min(candidate, maximum)
    initial = candidate
    attempts: list[int] = []
    while True:
        attempts.append(candidate)
        try:
            sample = np.zeros(
                (candidate, *config["data"]["input_shape"]),
                dtype="float32",
            )
            model(sample, training=False)
            break
        except Exception as exc:
            try:
                import tensorflow as tf

                is_oom = isinstance(exc, tf.errors.ResourceExhaustedError)
            except ImportError:  # pragma: no cover
                is_oom = False
            if not is_oom or candidate <= minimum:
                raise
            candidate = max(minimum, candidate // 2)
            if announce is not None:
                announce(
                    "Inference batch probe exhausted device memory; "
                    f"retrying with batch size {candidate}."
                )
    plan = InferenceBatchPlan(
        batch_size=candidate,
        mode=mode,
        initial_candidate=initial,
        attempts=tuple(attempts),
        memory_budget_mb=memory_budget_mb,
    )
    if announce is not None:
        announce(
            f"Inference batch size: {plan.batch_size} "
            f"({plan.mode}, verified with a forward pass; "
            f"memory budget {plan.memory_budget_mb} MiB)."
        )
    return plan
