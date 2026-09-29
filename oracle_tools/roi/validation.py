"""Validation reports for ROI masks; independent of a dataset backend."""
from __future__ import annotations

import numpy as np
from oracle_tools.roi.morphology import connected_components


def _mask_2d(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    return array[..., 0] if array.ndim == 3 and array.shape[-1] == 1 else array


def summarize_mask(mask: np.ndarray) -> dict:
    array = _mask_2d(mask)
    finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.number) else array
    values, foreground = np.unique(finite).tolist() if finite.size else [], array > 0
    _, sizes = connected_components(foreground.astype("uint8")) if array.ndim == 2 else (None, [])
    total, count = (int(array.size), int(foreground.sum())) if array.ndim >= 2 else (0, 0)
    return {"height": int(array.shape[0]) if array.ndim >= 1 else None, "width": int(array.shape[1]) if array.ndim >= 2 else None, "channels": int(mask.shape[-1]) if np.asarray(mask).ndim == 3 else 1, "dtype": str(array.dtype), "unique_values": values, "foreground_pixel_count": count, "foreground_fraction": float(count / total) if total else 0.0, "connected_component_count": len(sizes), "touches_border": bool(array.ndim == 2 and foreground.any() and (foreground[0, :].any() or foreground[-1, :].any() or foreground[:, 0].any() or foreground[:, -1].any())), "has_nan": bool(np.issubdtype(array.dtype, np.number) and np.isnan(array).any()), "has_inf": bool(np.issubdtype(array.dtype, np.number) and np.isinf(array).any()), "is_binary": set(values).issubset({0, 1, False, True})}


def validate_mask(mask: np.ndarray, image: np.ndarray | None = None, min_foreground_fraction: float = 0.0001, max_foreground_fraction: float = 0.95) -> dict:
    report, warnings, valid = summarize_mask(mask), [], True
    if report["has_nan"] or report["has_inf"]: valid = False; warnings.append("Mask contains NaN or Inf values.")
    if not report["is_binary"]: valid = False; warnings.append("Mask must be binary for the initial mask builder workflow.")
    if report["foreground_pixel_count"] == 0: valid = False; warnings.append("Mask has no foreground pixels.")
    if report["foreground_fraction"] < min_foreground_fraction: warnings.append("Foreground fraction is below the configured minimum.")
    if report["foreground_fraction"] > max_foreground_fraction: valid = False; warnings.append("Foreground fraction is above the configured maximum.")
    shape = np.asarray(image).shape if image is not None else None
    matches = shape is None or (len(shape) >= 2 and tuple(shape[:2]) == (report["height"], report["width"]))
    if not matches: valid = False; warnings.append("Mask height/width do not match the image height/width.")
    if report["touches_border"]: warnings.append("Foreground touches the image border.")
    report.update(dimension_matches_image=bool(matches), valid=bool(valid), warnings=warnings)
    return report
