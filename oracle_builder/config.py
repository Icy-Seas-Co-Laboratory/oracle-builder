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
    },
    "model": {},
    "training": {
        "epochs": 10,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "loss": None,
        "metrics": ["accuracy"],
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
