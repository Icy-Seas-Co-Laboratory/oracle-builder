"""Source-polarity provenance and conservative automatic estimation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


POLARITY_VALUES = ("auto", "dark_on_light", "light_on_dark", "mixed", "unknown")


def infer_source_polarity(
    paths: Iterable[Path], *, sample_count: int = 128
) -> dict[str, Any]:
    """Estimate polarity from robust border-versus-center intensity statistics.

    This deliberately returns ``unknown`` for weak or inconsistent evidence.
    It is a convenience for uniform ROI libraries, never a substitute for a
    declared acquisition convention.
    """
    candidates = sorted(Path(path) for path in paths)
    if sample_count < 1:
        raise ValueError("polarity sample_count must be positive")
    if len(candidates) > sample_count:
        positions = np.linspace(0, len(candidates) - 1, sample_count, dtype=int)
        candidates = [candidates[index] for index in positions]
    scores: list[float] = []
    for path in candidates:
        try:
            with Image.open(path) as image:
                value = np.asarray(image.convert("L"), dtype="float32")
        except (OSError, ValueError):
            continue
        height, width = value.shape
        if min(height, width) < 8:
            continue
        margin = max(1, int(round(min(height, width) * 0.15)))
        border = np.concatenate(
            (
                value[:margin, :].ravel(),
                value[-margin:, :].ravel(),
                value[margin:-margin, :margin].ravel(),
                value[margin:-margin, -margin:].ravel(),
            )
        )
        center = value[margin:-margin, margin:-margin]
        dynamic_range = float(np.percentile(value, 95) - np.percentile(value, 5))
        if dynamic_range <= 1e-6 or center.size == 0:
            continue
        scores.append(float((np.median(border) - np.median(center)) / dynamic_range))
    if not scores:
        return {
            "value": "unknown",
            "method": "automatic_border_center_v1",
            "confidence": 0.0,
            "sample_count": 0,
            "reason": "no_usable_images",
        }
    score = float(np.median(scores))
    signs = np.sign(scores)
    agreement = float(max(np.mean(signs >= 0), np.mean(signs <= 0)))
    strength = min(1.0, abs(score) / 0.25)
    confidence = round(strength * agreement, 4)
    if confidence < 0.65 or abs(score) < 0.08:
        value = "mixed" if agreement < 0.75 else "unknown"
    else:
        value = "dark_on_light" if score > 0 else "light_on_dark"
    return {
        "value": value,
        "method": "automatic_border_center_v1",
        "confidence": confidence,
        "sample_count": len(scores),
        "border_minus_center_normalized": round(score, 6),
        "agreement": round(agreement, 4),
    }


def resolve_polarity(
    requested: str,
    *,
    metadata: dict[str, Any] | None,
    paths: Iterable[Path],
    sample_count: int,
) -> dict[str, Any]:
    """Resolve CLI, sidecar, and automatic polarity in precedence order."""
    requested = str(requested).lower()
    if requested not in POLARITY_VALUES:
        raise ValueError(f"Unsupported source polarity: {requested}")
    imaging = dict(metadata or {})
    declared = str(
        imaging.get("source_polarity", imaging.get("object_polarity", "auto"))
    ).lower()
    if requested != "auto":
        result = {"value": requested, "method": "explicit_cli", "confidence": 1.0}
    elif declared in POLARITY_VALUES and declared != "auto":
        result = {"value": declared, "method": "metadata", "confidence": 1.0}
    else:
        result = infer_source_polarity(paths, sample_count=sample_count)
    result["model_polarity"] = "light_on_dark"
    result["model_invert"] = result["value"] == "dark_on_light"
    if imaging.get("illumination_mode") is not None:
        result["illumination_mode"] = str(imaging["illumination_mode"])
    return result
