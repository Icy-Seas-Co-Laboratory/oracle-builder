from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from oracle_builder.data.decoders import decode_blob
from oracle_builder.data.sqlite_dataset import read_rows
from oracle_builder.datasets.schema import (
    dataset_fingerprint,
    read_dataset_info,
    validate_database,
)
from oracle_builder.masking.sqlite_io import decode_mask
from oracle_builder.datasets.legacy_roi import ensure_mask_refinement_database


def validate_unet_dataset(
    sqlite_path: str | Path,
    target_input_shape: list[int] | tuple[int, ...] | None = None,
    target_output_shape: list[int] | tuple[int, ...] | None = None,
    require_candidate_mask: bool = False,
) -> dict[str, Any]:
    migration = ensure_mask_refinement_database(sqlite_path)
    rows = _read_masked_sample_rows(sqlite_path)
    with sqlite3.connect(Path(sqlite_path).expanduser().resolve()) as connection:
        dataset_info = read_dataset_info(connection)
        schema_validation = validate_database(connection)
        fingerprint = dataset_fingerprint(connection)
        item_count = int(
            connection.execute("SELECT count(*) FROM dataset_items").fetchone()[0]
        )
        annotation_count = int(
            connection.execute("SELECT count(*) FROM mask_annotations").fetchone()[0]
        )
        candidate_mask_count = int(
            connection.execute(
                """
                SELECT count(*) FROM mask_refinement_items
                WHERE candidate_mask_asset_id IS NOT NULL
                """
            ).fetchone()[0]
        )
    target_input_shape = list(target_input_shape) if target_input_shape is not None else None
    target_output_shape = list(target_output_shape) if target_output_shape is not None else None
    report: dict[str, Any] = {
        "valid": True,
        "database": str(sqlite_path),
        "dataset": {
            "dataset_id": dataset_info["dataset_id"],
            "revision_id": dataset_info["revision_id"],
            "parent_revision_id": dataset_info.get("parent_revision_id"),
            "dataset_type": dataset_info["dataset_type"],
            "schema_name": dataset_info["schema_name"],
            "schema_version": dataset_info["schema_version"],
            "lifecycle": dataset_info["lifecycle"],
            "fingerprint_sha256": fingerprint,
        },
        "migration": migration,
        "schema_validation": {
            "valid": schema_validation["valid"],
            "errors": list(schema_validation["errors"]),
            "warnings": list(schema_validation["warnings"]),
        },
        "item_count": item_count,
        "sample_count": len(rows),
        "annotated_item_count": len(rows),
        "missing_current_mask_count": item_count - len(rows),
        "annotation_count": annotation_count,
        "historical_annotation_count": annotation_count - len(rows),
        "candidate_mask_count": candidate_mask_count,
        "usable_sample_count": 0,
        "input_shapes": {},
        "output_shapes": {},
        "target_input_shape": target_input_shape,
        "target_output_shape": target_output_shape,
        "inferred_input_shape": None,
        "inferred_output_shape": None,
        "warnings": [],
        "errors": [],
        "samples": [],
    }
    report["errors"].extend(
        f"Dataset schema: {error}" for error in schema_validation["errors"]
    )
    report["warnings"].extend(
        f"Dataset schema: {warning}" for warning in schema_validation["warnings"]
    )
    if not rows:
        report["valid"] = False
        report["errors"].append("No mask-refinement items with a current accepted mask were found.")
        return report

    input_shapes: Counter[tuple[int, ...]] = Counter()
    output_shapes: Counter[tuple[int, ...]] = Counter()
    for row in rows:
        sample_report = _validate_sample(row)
        sample_input_shape = sample_report.get("input_shape") or []
        if require_candidate_mask and (not sample_input_shape or sample_input_shape[-1] != 2):
            sample_report["errors"].append("Candidate-delta training requires a two-channel candidate-mask input.")
            sample_report["usable"] = False
        report["samples"].append(sample_report)
        if sample_report["usable"]:
            report["usable_sample_count"] += 1
            input_shapes[tuple(sample_report["input_shape"])] += 1
            output_shapes[tuple(sample_report["output_shape"])] += 1
        report["warnings"].extend(f"{sample_report['uuid']}: {warning}" for warning in sample_report["warnings"])
        report["errors"].extend(f"{sample_report['uuid']}: {error}" for error in sample_report["errors"])

    report["input_shapes"] = {str(list(shape)): count for shape, count in sorted(input_shapes.items())}
    report["output_shapes"] = {str(list(shape)): count for shape, count in sorted(output_shapes.items())}
    if input_shapes:
        report["inferred_input_shape"] = list(input_shapes.most_common(1)[0][0])
    if output_shapes:
        report["inferred_output_shape"] = list(output_shapes.most_common(1)[0][0])
    if target_input_shape:
        report["inferred_input_shape"] = target_input_shape
    if target_output_shape:
        report["inferred_output_shape"] = target_output_shape
    if len(input_shapes) > 1 and not target_input_shape:
        report["valid"] = False
        report["errors"].append("Usable samples have inconsistent input shapes.")
    elif len(input_shapes) > 1:
        report["warnings"].append("Usable samples have inconsistent input shapes and will be resized to target_input_shape.")
    if len(output_shapes) > 1 and not target_output_shape:
        report["valid"] = False
        report["errors"].append("Usable samples have inconsistent output shapes.")
    elif len(output_shapes) > 1:
        report["warnings"].append("Usable samples have inconsistent output shapes and will be resized to target_output_shape.")
    if report["usable_sample_count"] == 0:
        report["valid"] = False
        report["errors"].append("No usable U-Net segmentation samples were found.")
    if report["errors"]:
        report["valid"] = False
    return report


def write_unet_config_from_dataset(
    sqlite_path: str | Path,
    config_path: str | Path,
    run_name: str | None = None,
    model_name: str = "unet",
    batch_size: int = 8,
    epochs: int = 20,
    segmentation_target: str = "validated_mask",
    candidate_sdf: bool = False,
    candidate_sdf_clip_distance: float = 32.0,
    tiling_enabled: bool = False,
    tiling_overlap_fraction: float = 0.5,
    tiling_blend_mode: str = "hann",
    target_input_shape: list[int] | tuple[int, ...] | None = None,
    target_output_shape: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    report = validate_unet_dataset(
        sqlite_path,
        target_input_shape=target_input_shape,
        target_output_shape=target_output_shape,
        require_candidate_mask=segmentation_target == "candidate_delta",
    )
    if not report["valid"]:
        raise ValueError("Cannot write U-Net config for invalid dataset: " + "; ".join(report["errors"]))
    raw_candidate_inputs = all(
        sample.get("input_shape") and sample["input_shape"][-1] == 2
        for sample in report["samples"]
        if sample.get("usable")
    )
    if segmentation_target == "candidate_delta" and not raw_candidate_inputs:
        raise ValueError("candidate_delta config generation requires two-channel candidate-mask inputs")
    if candidate_sdf and not raw_candidate_inputs:
        raise ValueError("candidate SDF config generation requires two-channel candidate-mask inputs")
    model_input_shape = list(report["inferred_input_shape"])
    if candidate_sdf:
        model_input_shape[-1] = 3
    config = {
        "run": {
            "task": "segmentation",
            "model": model_name,
            "seed": 123,
            "notes": f"Generated from {Path(sqlite_path).name} by mask_builder",
        },
        "data": {
            "input_shape": model_input_shape,
            "output_shape": report["inferred_output_shape"],
            "batch_size": batch_size,
            "shuffle_buffer": max(32, report["usable_sample_count"]),
            "validation_split": 0.2,
            "test_split": 0.1,
            "candidate_sdf": candidate_sdf,
            "candidate_sdf_clip_distance": candidate_sdf_clip_distance,
        },
        "model": {
            "base_filters": 32,
            "depth": 4,
            "dropout": 0.1,
            "activation": "relu",
            "final_activation": "sigmoid",
        },
        "training": {
            "epochs": epochs,
            "optimizer": "adam",
            "learning_rate": 0.0001,
            "loss": "bce_soft_dice",
            "bce_weight": 1.0,
            "soft_dice_weight": 1.0,
            "soft_dice_smooth": 0.000001,
            "metrics": ["accuracy", "dice", "iou"],
            "segmentation_target": segmentation_target,
        },
        "callbacks": {
            "early_stopping": True,
            "early_stopping_patience": 8,
            "reduce_lr_on_plateau": True,
            "checkpoint_monitor": "val_loss",
        },
        "augmentation": {
            "enabled": False,
            "repeats_per_epoch": 1,
            "invert": False,
            "rotation": 0.05,
            "zoom": 0.15,
            "translation": [0.1, 0.1],
            "skew": 0.05,
            "flip_horizontal": True,
            "flip_vertical": True,
            "brightness": 0.15,
            "contrast": 0.15,
            "gaussian_noise": 0.03,
            "fill_value": 0.0,
            "mask_fill_value": 0.0,
            "photometric_channels": [0],
            "mask_input_channels": [1] if report["inferred_input_shape"][-1] == 2 else [],
        },
        "output": {
            "save_checkpoints": True,
            "save_predictions": True,
            "save_figures": True,
            "export_savedmodel": True,
        },
        "tiling": {
            "enabled": tiling_enabled,
            "overlap_fraction": tiling_overlap_fraction,
            "blend_mode": tiling_blend_mode,
            "tile_large_rois_only": True,
            "normalize_training_coverage": True,
        },
    }
    if run_name:
        config["run"]["notes"] = f"{config['run']['notes']} ({run_name})"
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config_path).write_text(_to_toml(config))
    return {"config": config, "validation": report}


def _read_masked_sample_rows(sqlite_path: str | Path) -> list[dict[str, Any]]:
    return [
        row
        for row in read_rows(sqlite_path)
        if row.get("output_blob") is not None
    ]


def _validate_sample(row: dict[str, Any]) -> dict[str, Any]:
    sample_report = {
        "uuid": row["uuid"],
        "usable": True,
        "input_shape": None,
        "output_shape": None,
        "warnings": [],
        "errors": [],
    }
    try:
        image = np.asarray(decode_blob(row["input_blob"], row["input_blob_encoding"], row["input_blob_dimensions"]))
        mask = decode_mask(row["output_blob"], row["output_blob_encoding"], row["output_blob_dimensions"])
        if mask is None:
            raise ValueError("Accepted mask could not be decoded.")
        mask = np.asarray(mask)
        input_shape = _training_input_shape(image)
        output_shape = _training_output_shape(mask)
        sample_report["input_shape"] = input_shape
        sample_report["output_shape"] = output_shape
        if tuple(input_shape[:2]) != tuple(output_shape[:2]):
            sample_report["errors"].append("Image and mask height/width do not match.")
        if not set(np.unique(mask).tolist()).issubset({0, 1, False, True}):
            sample_report["errors"].append("Mask is not binary after decoding.")
        if mask.size == 0:
            sample_report["errors"].append("Mask has no pixels.")
        if image.size == 0:
            sample_report["errors"].append("Image has no pixels.")
        if np.asarray(mask > 0).sum() == 0:
            sample_report["warnings"].append("Mask has no foreground pixels.")
    except Exception as exc:
        sample_report["errors"].append(str(exc))
    sample_report["usable"] = not sample_report["errors"]
    return sample_report


def _training_input_shape(image: np.ndarray) -> list[int]:
    if image.ndim == 2:
        return [int(image.shape[0]), int(image.shape[1]), 1]
    if image.ndim == 3:
        return [int(image.shape[0]), int(image.shape[1]), int(image.shape[2])]
    raise ValueError(f"Unsupported image shape for U-Net training: {image.shape}")


def _training_output_shape(mask: np.ndarray) -> list[int]:
    if mask.ndim == 2:
        return [int(mask.shape[0]), int(mask.shape[1]), 1]
    if mask.ndim == 3 and mask.shape[-1] == 1:
        return [int(mask.shape[0]), int(mask.shape[1]), 1]
    raise ValueError(f"Unsupported mask shape for binary U-Net training: {mask.shape}")


def _to_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for section, values in data.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _toml_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {value!r}")
