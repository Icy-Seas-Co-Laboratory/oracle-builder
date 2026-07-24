from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt


def signed_distance_field(mask: Any, clip_distance: float = 32.0) -> np.ndarray:
    """Return a normalized SDF: positive inside, negative outside, clipped to [-1, 1]."""
    if clip_distance <= 0:
        raise ValueError("data.candidate_sdf_clip_distance must be greater than zero")
    value = np.asarray(mask)
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError(f"Candidate SDF requires a 2D single-channel mask, got {value.shape}")
    foreground = value > 0.5
    if not foreground.any():
        return np.full(foreground.shape, -1.0, dtype="float32")
    if foreground.all():
        return np.ones(foreground.shape, dtype="float32")
    inside_distance = distance_transform_edt(foreground)
    outside_distance = distance_transform_edt(~foreground)
    signed = (inside_distance - outside_distance) / float(clip_distance)
    return np.clip(signed, -1.0, 1.0).astype("float32")
