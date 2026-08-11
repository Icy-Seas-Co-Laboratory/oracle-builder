from __future__ import annotations

from typing import Any
import heapq

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


def geodesic_distance_field(
    image: Any,
    mask: Any,
    *,
    clip_distance: float = 32.0,
    epsilon: float = 1e-3,
    intensity_weight: float = 1.0,
    intensity_gamma: float = 1.0,
    gradient_weight: float = 1.0,
    connectivity: int = 8,
) -> np.ndarray:
    """Minimum image-cost travel distance from every candidate-mask pixel.

    Bright, continuous structure is cheap; dark pixels and intensity boundaries
    are expensive. The output is clipped and normalized to ``[0, 1]``.
    """
    if clip_distance <= 0 or epsilon <= 0 or connectivity not in {4, 8}:
        raise ValueError("geodesic distance requires positive clip/epsilon and 4 or 8 connectivity")
    values = np.asarray(image, dtype="float32")
    if values.ndim == 3:
        values = values[..., 0] if values.shape[-1] == 1 else np.mean(values[..., :3], axis=-1)
    seeds = np.asarray(mask) > 0.5
    if values.ndim != 2 or seeds.shape != values.shape:
        raise ValueError("Geodesic distance requires matching 2D image and candidate mask")
    lo, hi = np.percentile(values[np.isfinite(values)], [1, 99]) if np.isfinite(values).any() else (0.0, 1.0)
    intensity = np.zeros_like(values) if hi <= lo else np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    gy, gx = np.gradient(intensity)
    gradient = np.hypot(gx, gy)
    cost = epsilon + intensity_weight * (1.0 - intensity) ** intensity_gamma + gradient_weight * gradient
    distance = np.full(values.shape, np.inf, dtype="float32")
    heap: list[tuple[float, int, int]] = []
    for y, x in np.argwhere(seeds):
        distance[y, x] = 0.0
        heapq.heappush(heap, (0.0, int(y), int(x)))
    if not heap:
        return np.ones(values.shape, dtype="float32")
    steps = [(0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0)]
    if connectivity == 8:
        steps += [(1, 1, 2**0.5), (1, -1, 2**0.5), (-1, 1, 2**0.5), (-1, -1, 2**0.5)]
    height, width = values.shape
    while heap:
        current, y, x = heapq.heappop(heap)
        if current != distance[y, x]:
            continue
        for dy, dx, length in steps:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                proposed = current + length * float((cost[y, x] + cost[ny, nx]) / 2.0)
                if proposed < distance[ny, nx]:
                    distance[ny, nx] = proposed
                    heapq.heappush(heap, (proposed, ny, nx))
    return np.clip(distance / clip_distance, 0.0, 1.0).astype("float32")
