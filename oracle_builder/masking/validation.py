from __future__ import annotations

import numpy as np

from oracle_builder.masking.morphology import connected_components


def _mask_2d(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


def summarize_mask(mask: np.ndarray) -> dict:
    """Return descriptive statistics about a mask."""
    array = _mask_2d(mask)
    finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.number) else array
    unique_values = np.unique(finite).tolist() if finite.size else []
    foreground = array > 0
    labels, sizes = connected_components(foreground.astype("uint8")) if array.ndim == 2 else (None, [])
    touches_border = False
    if array.ndim == 2 and foreground.any():
        touches_border = bool(foreground[0, :].any() or foreground[-1, :].any() or foreground[:, 0].any() or foreground[:, -1].any())
    height = int(array.shape[0]) if array.ndim >= 1 else None
    width = int(array.shape[1]) if array.ndim >= 2 else None
    channels = int(mask.shape[-1]) if np.asarray(mask).ndim == 3 else 1
    total = int(array.size) if array.ndim >= 2 else 0
    foreground_count = int(foreground.sum()) if array.ndim >= 2 else 0
    return {
        "height": height,
        "width": width,
        "channels": channels,
        "dtype": str(array.dtype),
        "unique_values": unique_values,
        "foreground_pixel_count": foreground_count,
        "foreground_fraction": float(foreground_count / total) if total else 0.0,
        "connected_component_count": len(sizes),
        "touches_border": touches_border,
        "has_nan": bool(np.issubdtype(array.dtype, np.number) and np.isnan(array).any()),
        "has_inf": bool(np.issubdtype(array.dtype, np.number) and np.isinf(array).any()),
        "is_binary": set(unique_values).issubset({0, 1, False, True}),
    }


def validate_mask(
    mask: np.ndarray,
    image: np.ndarray | None = None,
    min_foreground_fraction: float = 0.0001,
    max_foreground_fraction: float = 0.95,
) -> dict:
    """Return a validation report dictionary."""
    report = summarize_mask(mask)
    warnings: list[str] = []
    valid = True

    if report["has_nan"] or report["has_inf"]:
        valid = False
        warnings.append("Mask contains NaN or Inf values.")
    if not report["is_binary"]:
        valid = False
        warnings.append("Mask must be binary for the initial mask builder workflow.")
    if report["foreground_pixel_count"] == 0:
        valid = False
        warnings.append("Mask has no foreground pixels.")
    if report["foreground_fraction"] < min_foreground_fraction:
        warnings.append("Foreground fraction is below the configured minimum.")
    if report["foreground_fraction"] > max_foreground_fraction:
        valid = False
        warnings.append("Foreground fraction is above the configured maximum.")
    if image is not None:
        image_shape = np.asarray(image).shape
        dimension_matches = len(image_shape) >= 2 and tuple(image_shape[:2]) == (report["height"], report["width"])
    else:
        dimension_matches = True
    if not dimension_matches:
        valid = False
        warnings.append("Mask height/width do not match the image height/width.")
    if report["touches_border"]:
        warnings.append("Foreground touches the image border.")

    report["dimension_matches_image"] = bool(dimension_matches)
    report["valid"] = bool(valid)
    report["warnings"] = warnings
    return report

