from __future__ import annotations

from typing import Any

import numpy as np


VALIDATED_MASK = "validated_mask"
CANDIDATE_DELTA = "candidate_delta"
SEGMENTATION_TARGETS = {VALIDATED_MASK, CANDIDATE_DELTA}


def segmentation_target_mode(config: dict[str, Any]) -> str:
    return str(config.get("training", {}).get("segmentation_target", VALIDATED_MASK)).lower()


def candidate_delta(candidate: Any, validated: Any) -> np.ndarray:
    return np.logical_xor(np.asarray(candidate) > 0.5, np.asarray(validated) > 0.5).astype("float32")


def reconstruct_validated_mask(candidate: Any, delta: Any) -> np.ndarray:
    return np.logical_xor(np.asarray(candidate) > 0.5, np.asarray(delta) > 0.5).astype("float32")


def reconstruct_validated_probability(candidate: Any, delta_probability: Any) -> np.ndarray:
    candidate_mask = (np.asarray(candidate) > 0.5).astype("float32")
    probability = np.asarray(delta_probability, dtype="float32")
    return candidate_mask * (1.0 - probability) + (1.0 - candidate_mask) * probability
