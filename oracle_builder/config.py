from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

try:
    import tomli_w
except ModuleNotFoundError:  # pragma: no cover
    tomli_w = None


DEFAULT_CONFIG: dict[str, Any] = {
    "run": {"seed": 123, "notes": ""},
    "data": {
        "batch_size": 16,
        "shuffle_buffer": 512,
        "validation_split": 0.2,
        "test_split": 0.1,
        "candidate_sdf": False,
        "candidate_sdf_clip_distance": 32.0,
    },
    "model": {},
    "training": {
        "epochs": 10,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "loss": None,
        "metrics": ["accuracy"],
        "segmentation_target": "validated_mask",
        "spatial_edge_weighting": False,
        "edge_weight_lambda": 1.0,
        "edge_weight_sigma": 5.0,
    },
    "callbacks": {
        "early_stopping": False,
        "early_stopping_patience": 5,
        "reduce_lr_on_plateau": False,
        "checkpoint_monitor": "val_loss",
    },
    "augmentation": {
        "enabled": False,
        "repeats_per_epoch": 1,
        "invert": False,
        "rotation": 0.0,
        "zoom": 0.0,
        "translation": 0.0,
        "skew": 0.0,
        "flip_horizontal": False,
        "flip_vertical": False,
        "brightness": 0.0,
        "contrast": 0.0,
        "gaussian_noise": 0.0,
        "fill_value": 0.0,
        "mask_fill_value": 0.0,
    },
    "output": {
        "save_checkpoints": True,
        "save_predictions": True,
        "save_figures": True,
        "export_savedmodel": True,
    },
    "evaluation": {"segmentation_threshold": 0.5},
    "tiling": {
        "enabled": False,
        "overlap_fraction": 0.5,
        "blend_mode": "hann",
        "tile_large_rois_only": True,
        "normalize_training_coverage": True,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def validate_config(config: dict[str, Any]) -> None:
    for section in ("run", "data", "training"):
        if section not in config:
            raise ValueError(f"Missing required config section [{section}]")
    task = config["run"].get("task")
    model = config["run"].get("model")
    if task not in {"classification", "segmentation"}:
        raise ValueError("run.task must be 'classification' or 'segmentation'")
    if not model:
        raise ValueError("run.model is required")
    if "input_shape" not in config["data"]:
        raise ValueError("data.input_shape is required")
    if task == "classification" and "num_classes" not in config["data"]:
        raise ValueError("data.num_classes is required for classification")
    if task == "segmentation" and "output_shape" not in config["data"]:
        raise ValueError("data.output_shape is required for segmentation")
    segmentation_target = str(config["training"].get("segmentation_target", "validated_mask")).lower()
    if segmentation_target not in {"validated_mask", "candidate_delta"}:
        raise ValueError("training.segmentation_target must be 'validated_mask' or 'candidate_delta'")
    if segmentation_target == "candidate_delta":
        input_shape = config["data"]["input_shape"]
        if task != "segmentation":
            raise ValueError("training.segmentation_target='candidate_delta' requires segmentation")
        expected_channels = 3 if config["data"].get("candidate_sdf", False) else 2
        if len(input_shape) != 3 or int(input_shape[-1]) != expected_channels:
            raise ValueError(f"candidate_delta training requires data.input_shape with {expected_channels} channels")
    if config["data"].get("candidate_sdf", False):
        input_shape = config["data"]["input_shape"]
        if task != "segmentation" or len(input_shape) != 3 or int(input_shape[-1]) != 3:
            raise ValueError("data.candidate_sdf requires segmentation with a three-channel input_shape")
        if float(config["data"].get("candidate_sdf_clip_distance", 32.0)) <= 0:
            raise ValueError("data.candidate_sdf_clip_distance must be greater than zero")
    tiling = config.get("tiling", {})
    if tiling.get("enabled", False):
        if task != "segmentation":
            raise ValueError("tiling is only supported for segmentation")
        overlap = float(tiling.get("overlap_fraction", 0.5))
        if not 0.0 <= overlap < 1.0:
            raise ValueError("tiling.overlap_fraction must be in [0, 1)")
        if tiling.get("blend_mode", "hann") not in {"uniform", "hann"}:
            raise ValueError("tiling.blend_mode must be 'uniform' or 'hann'")
        input_shape = config["data"]["input_shape"]
        output_shape = config["data"]["output_shape"]
        if tuple(input_shape[:2]) != tuple(output_shape[:2]):
            raise ValueError("tiling currently requires matching input and output spatial dimensions")
    if config["training"].get("spatial_edge_weighting", False):
        if task != "segmentation":
            raise ValueError("training.spatial_edge_weighting is only supported for segmentation")
        if float(config["training"].get("edge_weight_lambda", 1.0)) < 0:
            raise ValueError("training.edge_weight_lambda must be non-negative")
        if float(config["training"].get("edge_weight_sigma", 5.0)) <= 0:
            raise ValueError("training.edge_weight_sigma must be greater than zero")
    if not config["training"].get("loss"):
        raise ValueError("training.loss is required")


def resolve_config(config_path: str | Path, input_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
    user_config = load_toml(config_path)
    resolved = deep_merge(DEFAULT_CONFIG, user_config)
    validate_config(resolved)
    resolved["paths"] = {
        "config_path": str(Path(config_path).resolve()),
        "input_path": str(Path(input_path).resolve()),
        "run_dir": str(Path(run_dir).resolve()),
    }
    return resolved


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def write_toml(path: str | Path, data: dict[str, Any]) -> None:
    if tomli_w is None:
        raise RuntimeError("Writing TOML requires tomli-w. Install requirements.txt to enable this helper.")
    Path(path).write_text(tomli_w.dumps(data))


def copy_run_config(source: str | Path, run_dir: str | Path) -> None:
    shutil.copy2(source, Path(run_dir) / "run_config.toml")
