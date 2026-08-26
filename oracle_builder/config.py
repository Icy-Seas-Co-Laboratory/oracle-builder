from __future__ import annotations

import copy
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

DEFAULT_CONFIG: dict[str, Any] = {
    "run": {"seed": 123, "notes": ""},
    "data": {
        "batch_size": 16,
        "shuffle_buffer": 512,
        "validation_split": 0.2,
        "test_split": 0.1,
        # "auto" honors a complete explicit source-partition layout when one
        # was imported; otherwise it creates a deterministic random protocol.
        # "random" always creates one; "source_partitions" requires one.
        "split_strategy": "auto",
        "candidate_sdf": False,
        "candidate_sdf_clip_distance": 32.0,
        "candidate_distance": "none",
        "candidate_distance_clip": 32.0,
        "geodesic_distance": {
            "epsilon": 0.001,
            "intensity_weight": 1.0,
            "intensity_gamma": 1.0,
            "gradient_weight": 1.0,
            "connectivity": 8,
        },
        "streaming": {
            "enabled": True,
            "reader_workers": 4,
            "prefetch_batches": 2,
            "deterministic": True,
            "sqlite_cache_kib": 65536,
        },
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
        "class_weights": {
            "mode": "effective_number",
            "beta": 0.999,
            "normalize": True,
            "values": [],
        },
        "segmentation_target": "validated_mask",
        "spatial_edge_weighting": False,
        "edge_weight_lambda": 1.0,
        "edge_weight_sigma": 5.0,
        "bce_weight": 1.0,
        "soft_dice_weight": 1.0,
        "soft_tversky_weight": 1.0,
        "tversky_alpha": 0.3,
        "tversky_beta": 0.7,
        "soft_tversky_smooth": 1e-6,
    },
    "pretraining": {
        "enabled": False,
        "method": "byol",
        "epochs": 10,
        "learning_rate": 0.001,
        "teacher_momentum": 0.99,
        "projection_dim": 128,
        "projection_hidden_dim": 256,
        "temperature": 0.1,
        "reconstruction_foreground_weight": 4.0,
        "augmentation": {},
        "minimum_global_batch_size": 2,
        "collapse_std_threshold": 1e-3,
        "collapse_embedding_spread_threshold": 0.005,
        "ssl_optimizer": "adam",
        "weight_decay": 1e-4,
        "vicreg_variance_weight": 25.0,
        "vicreg_covariance_weight": 1.0,
        "vicreg_target_std": 1.0,
    },
    "evidence": {
        "enabled": True,
        "knn_k": 5,
    },
    "inference": {
        "batch_size": "auto",
        "minimum_batch_size": 1,
        "maximum_batch_size": 64,
        "memory_budget_mb": 512,
        "progress": True,
    },
    "preprocessing": {
        "resize_mode": "fit_pad",
        "normalization": "dtype",
        "rescale": True,
        "invert": "auto",
        "pad_value": 0.0,
        "interpolation": "bilinear",
        "channel_mode": "grayscale",
        "percentile_low": 1.0,
        "percentile_high": 99.0,
    },
    "distribution": {
        "strategy": "auto",
        "devices": [],
        "cross_device_ops": "auto",
        "fallback_to_single": True,
        "memory_growth": True,
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
        "save_checkpoints": False,
        "save_predictions": True,
        "save_figures": True,
        "export_savedmodel": True,
    },
    "recovery": {
        "enabled": True,
        "save_every_epochs": 1,
    },
    "evaluation": {
        "segmentation_threshold": 0.5,
        # Classification uses argmax by default.  Grouped bootstrap intervals
        # are opt-in because nearby images are often not independent samples.
        "uncertainty": {
            "enabled": False,
            "group_metadata_key": None,
            "bootstrap_replicates": 1000,
            "confidence_level": 0.95,
            "seed": 123,
        },
        "benchmark": {
            "enabled": True,
            "warmup_batches": 2,
            "measured_batches": 10,
        },
    },
    "tiling": {
        "enabled": False,
        "overlap_fraction": 0.5,
        "blend_mode": "hann",
        "tile_large_rois_only": True,
        "normalize_training_coverage": True,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    # Resolution adds runtime details to nested sections.  Do not share those
    # mutable default objects across runs in a long-lived process.
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def self_supervised_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical self-supervised settings with legacy fallback.

    ``pretraining`` remains readable for existing configurations and artifacts,
    but new configurations should use ``self_supervised``.  When both sections
    are present, the canonical section wins.
    """
    canonical = config.get("self_supervised")
    if isinstance(canonical, dict):
        return canonical
    legacy = config.get("pretraining")
    return legacy if isinstance(legacy, dict) else {}


def normalize_self_supervised_config(
    config: dict[str, Any],
    user_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prefer the new config section while preserving legacy input support."""
    source = user_config if user_config is not None else config
    if isinstance(source.get("self_supervised"), dict):
        config.pop("pretraining", None)
    return config


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def validate_config(config: dict[str, Any]) -> None:
    for section in ("run", "data", "training"):
        if section not in config:
            raise ValueError(f"Missing required config section [{section}]")
    split_strategy = str(config["data"].get("split_strategy", "auto")).lower()
    if split_strategy not in {"auto", "random", "source_partitions"}:
        raise ValueError(
            "data.split_strategy must be 'auto', 'random', or 'source_partitions'"
        )
    invert = config.get("preprocessing", {}).get("invert", False)
    if not isinstance(invert, bool):
        raise ValueError("preprocessing.invert must resolve to a boolean")
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
    inference = config.get("inference", {})
    inference_batch = inference.get("batch_size", "auto")
    if not (
        (isinstance(inference_batch, str) and inference_batch.lower() == "auto")
        or (
            isinstance(inference_batch, int)
            and not isinstance(inference_batch, bool)
            and inference_batch >= 1
        )
    ):
        raise ValueError("inference.batch_size must be 'auto' or a positive integer")
    if int(inference.get("minimum_batch_size", 1)) < 1:
        raise ValueError("inference.minimum_batch_size must be positive")
    if int(inference.get("maximum_batch_size", 64)) < int(
        inference.get("minimum_batch_size", 1)
    ):
        raise ValueError(
            "inference.maximum_batch_size must be at least minimum_batch_size"
        )
    if int(inference.get("memory_budget_mb", 512)) < 64:
        raise ValueError("inference.memory_budget_mb must be at least 64")
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
    streaming = config.get("data", {}).get("streaming", {})
    if int(streaming.get("reader_workers", 4)) < 1:
        raise ValueError("data.streaming.reader_workers must be at least 1")
    if int(streaming.get("prefetch_batches", 2)) < 1:
        raise ValueError("data.streaming.prefetch_batches must be at least 1")
    if int(streaming.get("sqlite_cache_kib", 65536)) < 1:
        raise ValueError("data.streaming.sqlite_cache_kib must be at least 1")
    distribution = config.get("distribution", {})
    if distribution.get("strategy", "auto") not in {
        "auto",
        "single",
        "none",
        "mirrored",
        "cpu",
    }:
        raise ValueError(
            "distribution.strategy must be auto, single, mirrored, or cpu"
        )
    if distribution.get("cross_device_ops", "auto") not in {
        "auto",
        "nccl",
        "hierarchical_copy",
        "hierarchical-copy",
    }:
        raise ValueError(
            "distribution.cross_device_ops must be auto, nccl, or hierarchical_copy"
        )
    self_supervised = self_supervised_settings(config)
    if self_supervised.get("enabled", False):
        method = str(self_supervised.get("method", "byol")).lower()
        if method not in {
            "byol",
            "student_teacher",
            "simclr",
            "grayscale_reconstruction",
        }:
            raise ValueError(
                "self_supervised.method must be byol, student_teacher, simclr, or grayscale_reconstruction"
            )
        if task != "classification" and method != "grayscale_reconstruction":
            raise ValueError("segmentation pretraining requires method='grayscale_reconstruction'")
        if int(self_supervised.get("epochs", 10)) < 1:
            raise ValueError("self_supervised.epochs must be at least 1")
        if float(self_supervised.get("learning_rate", 0.001)) <= 0:
            raise ValueError("self_supervised.learning_rate must be greater than zero")
        momentum = float(self_supervised.get("teacher_momentum", 0.99))
        if not 0 <= momentum < 1:
            raise ValueError("self_supervised.teacher_momentum must be in [0, 1)")
        if int(self_supervised.get("projection_dim", 128)) < 1:
            raise ValueError("self_supervised.projection_dim must be positive")
        if int(self_supervised.get("projection_hidden_dim", 256)) < 1:
            raise ValueError("self_supervised.projection_hidden_dim must be positive")
        if float(self_supervised.get("temperature", 0.1)) <= 0:
            raise ValueError("self_supervised.temperature must be greater than zero")
        if float(self_supervised.get("reconstruction_foreground_weight", 4.0)) < 1:
            raise ValueError(
                "self_supervised.reconstruction_foreground_weight must be at least 1"
            )
        minimum_batch = self_supervised.get("minimum_global_batch_size", 2)
        if isinstance(minimum_batch, bool) or int(minimum_batch) < 2:
            raise ValueError("self_supervised.minimum_global_batch_size must be at least 2")
        collapse_threshold = float(self_supervised.get("collapse_std_threshold", 1e-3))
        if not np.isfinite(collapse_threshold) or collapse_threshold <= 0:
            raise ValueError("self_supervised.collapse_std_threshold must be finite and positive")
        spread_threshold = float(
            self_supervised.get("collapse_embedding_spread_threshold", 0.005)
        )
        if not np.isfinite(spread_threshold) or spread_threshold <= 0:
            raise ValueError(
                "self_supervised.collapse_embedding_spread_threshold must be finite and positive"
            )
        optimizer = str(self_supervised.get("ssl_optimizer", "adam")).lower()
        if optimizer not in {"adam", "adamw"}:
            raise ValueError("self_supervised.ssl_optimizer must be 'adam' or 'adamw'")
        weight_decay = float(self_supervised.get("weight_decay", 1e-4))
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("self_supervised.weight_decay must be finite and non-negative")
        for name in ("vicreg_variance_weight", "vicreg_covariance_weight"):
            value = float(self_supervised.get(name, 0.0))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"self_supervised.{name} must be finite and non-negative")
        target_std = float(self_supervised.get("vicreg_target_std", 1.0))
        if not np.isfinite(target_std) or target_std <= 0:
            raise ValueError("self_supervised.vicreg_target_std must be finite and positive")
    if task == "segmentation" and "output_shape" not in config["data"]:
        raise ValueError("data.output_shape is required for segmentation")
    distance_mode = str(config["data"].get("candidate_distance", "none")).lower()
    if distance_mode not in {"none", "euclidean_sdf", "geodesic"}:
        raise ValueError("data.candidate_distance must be none, euclidean_sdf, or geodesic")
    uses_distance = distance_mode != "none" or config["data"].get("candidate_sdf", False)
    segmentation_target = str(config["training"].get("segmentation_target", "validated_mask")).lower()
    if segmentation_target not in {"validated_mask", "candidate_delta"}:
        raise ValueError("training.segmentation_target must be 'validated_mask' or 'candidate_delta'")
    if segmentation_target == "candidate_delta":
        input_shape = config["data"]["input_shape"]
        if task != "segmentation":
            raise ValueError("training.segmentation_target='candidate_delta' requires segmentation")
        expected_channels = 3 if uses_distance else 2
        if len(input_shape) != 3 or int(input_shape[-1]) != expected_channels:
            raise ValueError(f"candidate_delta training requires data.input_shape with {expected_channels} channels")
    if uses_distance:
        input_shape = config["data"]["input_shape"]
        if task != "segmentation" or len(input_shape) != 3 or int(input_shape[-1]) != 3:
            raise ValueError("candidate distance requires segmentation with a three-channel input_shape")
        if float(config["data"].get("candidate_distance_clip", config["data"].get("candidate_sdf_clip_distance", 32.0))) <= 0:
            raise ValueError("candidate distance clip must be greater than zero")
        if distance_mode == "geodesic":
            settings = config["data"].get("geodesic_distance", {})
            if int(settings.get("connectivity", 8)) not in {4, 8}:
                raise ValueError("data.geodesic_distance.connectivity must be 4 or 8")
            if float(settings.get("epsilon", 0.001)) <= 0:
                raise ValueError("data.geodesic_distance.epsilon must be greater than zero")
    if str(config["training"].get("loss", "")).lower() in {"bce_soft_tversky", "bce+soft_tversky", "binary_crossentropy_soft_tversky"}:
        alpha = float(config["training"].get("tversky_alpha", 0.3))
        beta = float(config["training"].get("tversky_beta", 0.7))
        if alpha < 0 or beta < 0 or alpha + beta <= 0:
            raise ValueError("training Tversky alpha and beta must be non-negative with a positive sum")
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
    recovery = config.get("recovery", {})
    if int(recovery.get("save_every_epochs", 1)) < 1:
        raise ValueError("recovery.save_every_epochs must be at least 1")
    benchmark = config.get("evaluation", {}).get("benchmark", {})
    if int(benchmark.get("warmup_batches", 2)) < 0:
        raise ValueError("evaluation.benchmark.warmup_batches cannot be negative")
    if int(benchmark.get("measured_batches", 10)) < 1:
        raise ValueError("evaluation.benchmark.measured_batches must be positive")
    uncertainty = config.get("evaluation", {}).get("uncertainty", {})
    if int(uncertainty.get("bootstrap_replicates", 1000)) < 1:
        raise ValueError("evaluation.uncertainty.bootstrap_replicates must be positive")
    confidence_level = float(uncertainty.get("confidence_level", 0.95))
    if not 0 < confidence_level < 1:
        raise ValueError("evaluation.uncertainty.confidence_level must be between zero and one")
    if uncertainty.get("enabled", False) and not uncertainty.get("group_metadata_key"):
        raise ValueError("evaluation.uncertainty.group_metadata_key is required when uncertainty is enabled")
    from oracle_builder.training.class_weights import (
        WEIGHTED_CROSS_ENTROPY_NAMES,
    )

    if str(config["training"]["loss"]).lower() in WEIGHTED_CROSS_ENTROPY_NAMES:
        if task != "classification":
            raise ValueError(
                "weighted cross entropy is only supported for classification"
            )
        weights = config["training"].get("class_weights", {})
        mode = str(weights.get("mode", "inverse_frequency")).lower()
        if mode not in {"explicit", "inverse_frequency", "effective_number"}:
            raise ValueError(
                "training.class_weights.mode must be explicit, "
                "inverse_frequency, or effective_number"
            )
        if mode == "effective_number" and not (
            0 <= float(weights.get("beta", 0.999)) < 1
        ):
            raise ValueError("training.class_weights.beta must be in [0, 1)")


def resolve_config(config_path: str | Path, input_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
    user_config = load_toml(config_path)
    resolved = deep_merge(DEFAULT_CONFIG, user_config)
    normalize_self_supervised_config(resolved, user_config)
    task = resolved.get("run", {}).get("task")
    if task == "classification":
        if not resolved.get("training", {}).get("loss"):
            resolved["training"]["loss"] = (
                "weighted_sparse_categorical_crossentropy"
            )
        if "channel_mode" not in user_config.get("preprocessing", {}):
            resolved["preprocessing"]["channel_mode"] = "grayscale"
    if resolved.get("run", {}).get("task") == "segmentation":
        from oracle_builder.datasets.legacy_roi import (
            ensure_mask_refinement_database,
        )

        ensure_mask_refinement_database(input_path)
    if (
        resolved.get("run", {}).get("task") == "classification"
        and "num_classes" not in resolved.get("data", {})
    ):
        resolved["data"]["num_classes"] = infer_classification_num_classes(input_path)
    from oracle_builder.datasets.schema import (
        dataset_fingerprint,
        read_dataset_info,
        validate_database,
    )

    with sqlite3.connect(Path(input_path).expanduser()) as connection:
        dataset_info = read_dataset_info(connection)
        dataset_report = validate_database(connection)
        if not dataset_report["valid"]:
            raise ValueError(
                "Dataset validation failed: " + "; ".join(dataset_report["errors"])
            )
        expected_type = (
            "classification"
            if resolved.get("run", {}).get("task") == "classification"
            else "mask_refinement"
        )
        if dataset_info["dataset_type"] != expected_type:
            raise ValueError(
                f"Training task requires a {expected_type!r} dataset; "
                f"database contains {dataset_info['dataset_type']!r}"
            )
        resolved["dataset"] = {
            "dataset_id": dataset_info["dataset_id"],
            "revision_id": dataset_info["revision_id"],
            "parent_revision_id": dataset_info.get("parent_revision_id"),
            "dataset_type": dataset_info["dataset_type"],
            "schema_name": dataset_info["schema_name"],
            "schema_version": dataset_info["schema_version"],
            "version": dataset_info.get("version"),
            "lifecycle": dataset_info["lifecycle"],
            "fingerprint_sha256": dataset_fingerprint(connection),
        }
        imaging = dataset_info.get("metadata", {}).get("imaging", {})
        if not isinstance(imaging, dict):
            imaging = {}
        polarity = imaging.get("polarity", {})
        if not isinstance(polarity, dict):
            polarity = {}
        source_polarity = str(polarity.get("value", "unknown"))
        requested_invert = resolved["preprocessing"].get("invert", "auto")
        if requested_invert == "auto":
            resolved_invert = source_polarity == "dark_on_light"
            invert_method = "dataset_source_polarity"
        elif isinstance(requested_invert, bool):
            resolved_invert = requested_invert
            invert_method = "explicit_config"
        else:
            raise ValueError("preprocessing.invert must be true, false, or 'auto'")
        resolved["preprocessing"]["invert"] = resolved_invert
        resolved["preprocessing"]["polarity_resolution"] = {
            "requested": requested_invert,
            "source_polarity": source_polarity,
            "method": invert_method,
            "model_polarity": "light_on_dark",
        }
        resolved["dataset"]["imaging"] = imaging
        source_metadata = dataset_info.get("metadata", {}).get(
            "source_metadata", {}
        )
        source_dataset = (
            source_metadata.get("dataset", {})
            if isinstance(source_metadata, dict)
            else {}
        )
        if isinstance(source_dataset, dict):
            resolved["dataset"]["usage"] = {
                key: source_dataset.get(key)
                for key in (
                    "dataset_doi",
                    "dataset_url",
                    "license",
                    "license_url",
                    "access_restrictions",
                    "redistribution_allowed",
                )
                if source_dataset.get(key) not in (None, "")
            }
        if dataset_info["dataset_type"] == "classification":
            resolved["dataset"]["labels"] = [
                {
                    "label_id": row[0],
                    "class_index": int(row[1]),
                    "name": row[2],
                    "concept_id": row[3],
                    "concept_node_id": row[4],
                    "concept_relationship": row[5],
                }
                for row in connection.execute(
                    """
                    SELECT l.label_id, l.class_index, l.name,
                           lc.concept_id, tc.vocabulary_node_id, lc.relationship
                    FROM classification_labels l
                    LEFT JOIN classification_label_concepts lc ON lc.label_id = l.label_id
                    LEFT JOIN taxonomy_concepts tc ON tc.concept_id = lc.concept_id
                    ORDER BY l.class_index
                    """
                )
            ]
    validate_config(resolved)
    resolved["paths"] = {
        "config_path": str(Path(config_path).resolve()),
        "input_path": str(Path(input_path).resolve()),
        "run_dir": str(Path(run_dir).resolve()),
    }
    return resolved


def infer_classification_num_classes(input_path: str | Path) -> int:
    connection = sqlite3.connect(Path(input_path).expanduser())
    try:
        dataset_type = connection.execute(
            "SELECT dataset_type FROM dataset WHERE singleton = 1"
        ).fetchone()
        if dataset_type is None or dataset_type[0] != "classification":
            raise ValueError("Input is not an Oracle Builder V1 classification dataset")
        indices = [
            int(row[0])
            for row in connection.execute(
                "SELECT class_index FROM classification_labels ORDER BY class_index"
            )
        ]
        if not indices:
            raise ValueError("Classification dataset contains no labels")
        if indices != list(range(len(indices))):
            raise ValueError(
                f"Classification labels must be contiguous from zero; found {indices}"
            )
        return len(indices)
    finally:
        connection.close()
