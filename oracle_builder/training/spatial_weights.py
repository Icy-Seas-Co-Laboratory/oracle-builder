from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt


def boundary_distance_weights(mask: Any, weight_lambda: float, sigma: float) -> np.ndarray:
    """Return 1 + lambda * exp(-d^2 / (2 sigma^2)) for a binary target mask."""
    if weight_lambda < 0:
        raise ValueError("training.edge_weight_lambda must be non-negative")
    if sigma <= 0:
        raise ValueError("training.edge_weight_sigma must be greater than zero")
    target = np.asarray(mask)
    if target.ndim == 3 and target.shape[-1] == 1:
        target = target[..., 0]
    if target.ndim != 2:
        raise ValueError(f"Spatial edge weighting requires a 2D single-channel mask, got {target.shape}")
    foreground = target > 0.5
    boundary = np.zeros(foreground.shape, dtype=bool)
    vertical = foreground[1:, :] != foreground[:-1, :]
    horizontal = foreground[:, 1:] != foreground[:, :-1]
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[0, :] |= foreground[0, :]
    boundary[-1, :] |= foreground[-1, :]
    boundary[:, 0] |= foreground[:, 0]
    boundary[:, -1] |= foreground[:, -1]
    if not boundary.any():
        return np.ones(foreground.shape, dtype="float32")
    distance = distance_transform_edt(~boundary)
    weights = 1.0 + weight_lambda * np.exp(-(distance**2) / (2.0 * sigma**2))
    return weights.astype("float32")


def batch_boundary_distance_weights(masks: Any, weight_lambda: float, sigma: float) -> np.ndarray:
    return np.stack(
        [boundary_distance_weights(mask, weight_lambda, sigma) for mask in np.asarray(masks)],
        axis=0,
    )
