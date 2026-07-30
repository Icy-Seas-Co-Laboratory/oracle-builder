from __future__ import annotations

from typing import Any

import numpy as np


WEIGHTED_CROSS_ENTROPY_NAMES = {
    "weighted_sparse_categorical_crossentropy",
    "weighted_cross_entropy",
    "weighted_crossentropy",
}


def uses_weighted_cross_entropy(config: dict[str, Any]) -> bool:
    return (
        str(config.get("training", {}).get("loss", "")).lower()
        in WEIGHTED_CROSS_ENTROPY_NAMES
    )


def resolve_class_weights(
    labels: np.ndarray | list[int],
    num_classes: int,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = dict(settings or {})
    mode = str(settings.get("mode", "inverse_frequency")).lower()
    labels = np.asarray(labels, dtype="int64")
    counts = np.bincount(labels, minlength=int(num_classes)).astype("float64")
    if len(counts) != int(num_classes):
        raise ValueError("Training labels exceed data.num_classes")
    missing = np.flatnonzero(counts == 0)
    if missing.size:
        raise ValueError(
            "Weighted cross entropy cannot resolve weights because the training "
            f"split has no samples for class indices {missing.tolist()}"
        )
    if mode == "explicit":
        weights = np.asarray(settings.get("values", []), dtype="float64")
        if weights.shape != (int(num_classes),):
            raise ValueError(
                "training.class_weights.values must contain one value per class"
            )
    elif mode == "inverse_frequency":
        weights = 1.0 / counts
    elif mode == "effective_number":
        beta = float(settings.get("beta", 0.999))
        if not 0.0 <= beta < 1.0:
            raise ValueError("training.class_weights.beta must be in [0, 1)")
        weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    else:
        raise ValueError(
            "training.class_weights.mode must be explicit, inverse_frequency, "
            "or effective_number"
        )
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Resolved class weights must be finite and positive")
    if bool(settings.get("normalize", True)):
        # Weighted mean is one at the observed training distribution, keeping
        # the loss scale comparable to ordinary cross entropy.
        weights = weights / np.average(weights, weights=counts)
    return {
        "mode": mode,
        "beta": (
            float(settings.get("beta", 0.999))
            if mode == "effective_number"
            else None
        ),
        "normalize": bool(settings.get("normalize", True)),
        "counts": [int(value) for value in counts],
        "values": [float(value) for value in weights],
    }
