from __future__ import annotations

import json
import shutil
import sqlite3
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
    "model": {
        "embedding_dim": 256,
        "normalize_embeddings": True,
    },
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
    "pretraining": {
        "enabled": False,
        "method": "byol",
        "epochs": 10,
        "learning_rate": 0.001,
        "teacher_momentum": 0.99,
        "projection_dim": 128,
        "projection_hidden_dim": 256,
        "use_training_augmentation": False,
        "augmentation": {},
    },
    "evidence": {
        "enabled": True,
        "knn_k": 5,
    },
    "preprocessing": {
        "resize_mode": "fit_pad",
        "normalization": "dtype",
        "rescale": True,
        "invert": False,
        "pad_value": 0.0,
        "interpolation": "bilinear",
        "channel_mode": "auto",
        "percentile_low": 1.0,
        "percentile_high": 99.0,
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
        raise ValueError(
            "Could not infer data.num_classes from the classification database"
        )
    if task == "classification" and int(config.get("model", {}).get("embedding_dim", 256)) < 1:
        raise ValueError("model.embedding_dim must be a positive integer")
    if int(config.get("evidence", {}).get("knn_k", 5)) < 1:
        raise ValueError("evidence.knn_k must be at least 1")
    preprocessing = config.get("preprocessing", {})
    if preprocessing.get("resize_mode", "fit_pad") not in {
        "fit_pad",
        "fill_crop",
        "stretch",
        "none",
        "fit",
    }:
        raise ValueError(
            "preprocessing.resize_mode must be fit_pad, fill_crop, stretch, none, or fit"
        )
    if preprocessing.get("normalization", "dtype") not in {
        "dtype",
        "minmax",
        "percentile",
        "none",
    }:
        raise ValueError(
            "preprocessing.normalization must be dtype, minmax, percentile, or none"
        )
    if preprocessing.get("interpolation", "bilinear") not in {
        "nearest",
        "bilinear",
        "bicubic",
        "lanczos",
    }:
        raise ValueError("Unsupported preprocessing.interpolation")
    if preprocessing.get("channel_mode", "auto") not in {"auto", "grayscale", "rgb", "rgba"}:
        raise ValueError("preprocessing.channel_mode must be auto, grayscale, rgb, or rgba")
    pretraining = config.get("pretraining", {})
    if pretraining.get("enabled", False):
        if task != "classification":
            raise ValueError("self-supervised pretraining is only supported for classification")
        if str(pretraining.get("method", "byol")).lower() not in {"byol", "student_teacher"}:
            raise ValueError("pretraining.method must be 'byol' or 'student_teacher'")
        if int(pretraining.get("epochs", 10)) < 1:
            raise ValueError("pretraining.epochs must be at least 1")
        if float(pretraining.get("learning_rate", 0.001)) <= 0:
            raise ValueError("pretraining.learning_rate must be greater than zero")
        momentum = float(pretraining.get("teacher_momentum", 0.99))
        if not 0 <= momentum < 1:
            raise ValueError("pretraining.teacher_momentum must be in [0, 1)")
        if int(pretraining.get("projection_dim", 128)) < 1:
            raise ValueError("pretraining.projection_dim must be positive")
        if int(pretraining.get("projection_hidden_dim", 256)) < 1:
            raise ValueError("pretraining.projection_hidden_dim must be positive")
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
    if (
        resolved.get("run", {}).get("task") == "classification"
        and "num_classes" not in resolved.get("data", {})
    ):
        resolved["data"]["num_classes"] = infer_classification_num_classes(input_path)
    validate_config(resolved)
    resolved["paths"] = {
        "config_path": str(Path(config_path).resolve()),
        "input_path": str(Path(input_path).resolve()),
        "run_dir": str(Path(run_dir).resolve()),
    }
    return resolved


def infer_classification_num_classes(input_path: str | Path) -> int:
    from oracle_builder.data.decoders import decode_blob

    connection = sqlite3.connect(Path(input_path).expanduser())
    try:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'class_labels'"
        ).fetchone()
        if table:
            indices = [
                int(row[0])
                for row in connection.execute(
                    "SELECT class_index FROM class_labels ORDER BY class_index"
                )
            ]
            if indices:
                expected = list(range(len(indices)))
                if indices != expected:
                    raise ValueError(
                        f"class_labels indices must be contiguous from zero; found {indices}"
                    )
                return len(indices)
        labels = set()
        for blob, encoding, label_text in connection.execute(
            "SELECT output_blob, output_blob_encoding, label_text FROM samples"
        ):
            if blob is not None:
                labels.add(int(decode_blob(blob, encoding)))
            elif label_text is not None:
                try:
                    labels.add(int(label_text))
                except ValueError:
                    continue
        if not labels:
            raise ValueError(
                "Could not infer classification classes: class_labels is empty and samples "
                "contain no numeric labels"
            )
        if labels != set(range(max(labels) + 1)):
            raise ValueError(
                f"Classification labels must be contiguous from zero; found {sorted(labels)}"
            )
        return max(labels) + 1
    finally:
        connection.close()


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def write_toml(path: str | Path, data: dict[str, Any]) -> None:
    if tomli_w is None:
        raise RuntimeError("Writing TOML requires tomli-w. Install requirements.txt to enable this helper.")
    Path(path).write_text(tomli_w.dumps(data))


def copy_run_config(source: str | Path, run_dir: str | Path) -> None:
    shutil.copy2(source, Path(run_dir) / "run_config.toml")
