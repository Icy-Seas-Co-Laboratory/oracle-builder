"""Auxiliary scalar feature extraction for image classifiers.

The values in this module deliberately remain outside image preprocessing: a
feature such as ROI area must describe the source ROI, not the resized canvas.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from oracle_builder.data.splits import assign_run_splits


def feature_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    specs = config.get("model", {}).get("auxiliary_features", [])
    if specs is None:
        return []
    if not isinstance(specs, list):
        raise ValueError("model.auxiliary_features must be an array of tables")
    return [dict(spec) for spec in specs]


def enabled(config: dict[str, Any]) -> bool:
    return bool(feature_specs(config))


def validate_feature_specs(config: dict[str, Any]) -> None:
    names: set[str] = set()
    allowed_transforms = {"identity", "log", "log1p", "sqrt"}
    for spec in feature_specs(config):
        name = str(spec.get("name", "")).strip()
        source = str(spec.get("source", "")).strip()
        if not name or not source:
            raise ValueError("Each model.auxiliary_features entry requires name and source")
        if name in names:
            raise ValueError(f"Duplicate auxiliary feature name {name!r}")
        names.add(name)
        if source not in {
            "roi.original_width_px", "roi.original_height_px", "roi.bounding_box_area_px"
        } and not source.startswith("metadata."):
            raise ValueError(
                f"Unsupported auxiliary feature source {source!r}; use an roi.* source "
                "or metadata.<dotted path>"
            )
        if str(spec.get("transform", "identity")).lower() not in allowed_transforms:
            raise ValueError("Auxiliary feature transform must be identity, log, log1p, or sqrt")
        if str(spec.get("missing", "error")).lower() not in {"error", "mean", "zero"}:
            raise ValueError("Auxiliary feature missing must be error, mean, or zero")


def _original_hw(shape: Any) -> tuple[int, int]:
    value = tuple(int(part) for part in shape)
    if len(value) < 2 or value[0] < 1 or value[1] < 1:
        raise ValueError(f"Cannot determine original ROI dimensions from {shape!r}")
    return value[0], value[1]


def _nested(metadata: dict[str, Any], dotted_path: str) -> Any:
    value: Any = metadata
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(dotted_path)
        value = value[key]
    return value


def raw_value(spec: dict[str, Any], metadata: dict[str, Any], original_shape: Any) -> float:
    source = str(spec["source"])
    height, width = _original_hw(original_shape)
    if source == "roi.original_width_px":
        value = width
    elif source == "roi.original_height_px":
        value = height
    elif source == "roi.bounding_box_area_px":
        value = height * width
    else:
        value = _nested(metadata, source.removeprefix("metadata."))
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Auxiliary feature {spec['name']!r} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"Auxiliary feature {spec['name']!r} must be finite")
    return result


def transform(value: float, spec: dict[str, Any]) -> float:
    kind = str(spec.get("transform", "identity")).lower()
    if kind == "identity":
        return value
    if kind == "log":
        if value <= 0:
            raise ValueError(f"Auxiliary feature {spec['name']!r} requires a positive value for log")
        return math.log(value)
    if kind == "log1p":
        if value <= -1:
            raise ValueError(f"Auxiliary feature {spec['name']!r} requires a value greater than -1 for log1p")
        return math.log1p(value)
    if kind == "sqrt":
        if value < 0:
            raise ValueError(f"Auxiliary feature {spec['name']!r} requires a non-negative value for sqrt")
        return math.sqrt(value)
    raise ValueError(f"Unsupported auxiliary feature transform {kind!r}")


def fitted_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(spec) for spec in config.get("model", {}).get("auxiliary_features_fitted", feature_specs(config))]


def vector(config: dict[str, Any], metadata: dict[str, Any], original_shape: Any) -> np.ndarray:
    result = []
    for spec in fitted_specs(config):
        try:
            value = transform(raw_value(spec, metadata, original_shape), spec)
        except (KeyError, ValueError):
            missing = str(spec.get("missing", "error")).lower()
            if missing == "error":
                raise
            value = float(spec.get("mean", 0.0)) if missing == "mean" else 0.0
        if bool(spec.get("standardize", True)):
            scale = float(spec.get("scale", 1.0))
            value = (value - float(spec.get("mean", 0.0))) / (scale if scale > 0 else 1.0)
        result.append(value)
    return np.asarray(result, dtype="float32")


def fit_feature_specs(sqlite_path: str | Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Fit standardization using training rows only and return persisted specs."""
    specs = feature_specs(config)
    if not specs:
        return []
    rows: list[tuple[str, str | None, str | None]]
    with sqlite3.connect(Path(sqlite_path)) as connection:
        rows = connection.execute(
            """
            SELECT di.item_id, di.metadata_json, a.shape_json
            FROM dataset_items di
            JOIN classification_items ci ON ci.item_id = di.item_id
            JOIN assets a ON a.asset_id = ci.image_asset_id
            """
        ).fetchall()
    assigned = assign_run_splits([{"uuid": str(row[0])} for row in rows], config)
    training_ids = {row["uuid"] for row in assigned if row["split"] == "train"}
    values: list[list[float]] = [[] for _ in specs]
    for item_id, metadata_json, shape_json in rows:
        if str(item_id) not in training_ids:
            continue
        metadata = json.loads(metadata_json) if metadata_json else {}
        shape = json.loads(shape_json) if shape_json else None
        for index, spec in enumerate(specs):
            try:
                values[index].append(transform(raw_value(spec, metadata, shape), spec))
            except (KeyError, ValueError):
                if str(spec.get("missing", "error")).lower() == "error":
                    raise ValueError(f"Training item {item_id}: missing/invalid auxiliary feature {spec['name']!r}")
    fitted = []
    for spec, feature_values in zip(specs, values, strict=True):
        if not feature_values:
            raise ValueError(f"No training values available for auxiliary feature {spec['name']!r}")
        copied = dict(spec)
        copied["mean"] = float(np.mean(feature_values))
        copied["scale"] = max(float(np.std(feature_values)), 1e-12)
        copied["fit_split"] = "train"
        fitted.append(copied)
    return fitted
