from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from oracle_builder.datasets.repository import SQLiteDatasetRepository
from oracle_builder.datasets.schema import initialize_database, read_dataset_info, utc_now
from oracle_builder.data.decoders import (
    decode_blob,
    encode_npy,
    normalize_input,
    prepare_classification_input,
)
from oracle_builder.data.splits import assign_run_splits
from oracle_builder.data.signed_distance import signed_distance_field
from oracle_builder.data.tiling import coverage_map, extract_tile, plan_tiles
from oracle_builder.evaluation.segmentation_targets import CANDIDATE_DELTA, candidate_delta, segmentation_target_mode


def ensure_schema(
    connection: sqlite3.Connection,
    dataset_type: str = "mask_refinement",
    **metadata: Any,
) -> dict[str, Any]:
    """Create or validate the breaking V1 use-case-specific dataset schema."""
    return initialize_database(connection, dataset_type, **metadata)


def read_rows(sqlite_path: str | Path) -> list[dict[str, Any]]:
    # Keep compatibility at the lowest shared SQLite read boundary. This
    # helper ignores classification databases and only upgrades recognized
    # pre-V1 ROI databases before V1 repository queries are attempted.
    from oracle_builder.datasets.legacy_roi import migrate_legacy_roi_if_needed

    migrate_legacy_roi_if_needed(sqlite_path)
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    try:
        info = read_dataset_info(connection)
        repository = SQLiteDatasetRepository(connection)
        if info["dataset_type"] == "classification":
            return [
                {
                    "uuid": row["item_id"],
                    "input_blob": row["image_blob"],
                    "input_blob_encoding": row["image_encoding"],
                    "input_blob_dimensions": row["image_dimensions"],
                    "output_blob": (
                        str(row["class_index"]).encode("utf-8")
                        if row["class_index"] is not None
                        else None
                    ),
                    "output_blob_encoding": "int" if row["class_index"] is not None else None,
                    "output_blob_dimensions": None,
                    "label_text": row["class_name"],
                    "sample_weight": row["sample_weight"],
                    "metadata_json": row["metadata_json"],
                }
                for row in repository.classification_rows()
            ]
        records = []
        for row in repository.mask_rows():
            image = decode_blob(
                row["image_blob"], row["image_encoding"], row["image_dimensions"]
            )
            if row["candidate_blob"] is not None:
                candidate = decode_blob(
                    row["candidate_blob"],
                    row["candidate_encoding"],
                    row["candidate_dimensions"],
                )
                if np.asarray(candidate).ndim == 3:
                    candidate = np.asarray(candidate)[..., 0]
                roi = np.asarray(image)
                if roi.ndim == 3 and roi.shape[-1] == 1:
                    roi = roi[..., 0]
                if roi.ndim == 3 and roi.shape[-1] >= 3:
                    rgb = roi[..., :3].astype("float32")
                    roi = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
                candidate = np.asarray(candidate) > 0
                if np.asarray(roi).dtype.kind in {"u", "i"}:
                    candidate = candidate.astype(np.asarray(roi).dtype)
                    candidate *= np.iinfo(np.asarray(roi).dtype).max
                training_input = np.stack([roi, candidate], axis=-1)
                input_blob = encode_npy(training_input)
                input_encoding = "npy"
                input_dimensions = json.dumps(list(training_input.shape))
            else:
                input_blob = row["image_blob"]
                input_encoding = row["image_encoding"]
                input_dimensions = row["image_dimensions"]
            records.append(
                {
                    "uuid": row["item_id"],
                    "input_blob": input_blob,
                    "input_blob_encoding": input_encoding,
                    "input_blob_dimensions": input_dimensions,
                    "output_blob": row["validated_blob"],
                    "output_blob_encoding": row["validated_encoding"],
                    "output_blob_dimensions": row["validated_dimensions"],
                    "label_text": None,
                    "sample_weight": row["sample_weight"],
                    "metadata_json": row["metadata_json"],
                }
            )
        return records
    finally:
        connection.close()


def load_arrays(sqlite_path: str | Path, config: dict[str, Any], split: str | None = None) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    task = config["run"]["task"]
    if task == "segmentation":
        from oracle_builder.datasets.legacy_roi import (
            ensure_mask_refinement_database,
        )

        ensure_mask_refinement_database(sqlite_path)
    rows = read_rows(sqlite_path)
    if task == "segmentation":
        rows = [row for row in rows if row.get("output_blob")]
    rows = assign_run_splits(rows, config)
    if task == "classification":
        rows = [
            row
            for row in rows
            if row.get("output_blob") is not None or row.get("label_text") is not None
        ]
    if split:
        rows = [row for row in rows if row.get("split") == split]
    if not rows:
        raise ValueError(f"No samples found for split={split!r}")

    input_shape = config["data"]["input_shape"]
    output_shape = config["data"].get("output_shape")
    xs: list[np.ndarray] = []
    ys: list[Any] = []
    records: list[dict[str, Any]] = []
    for row in rows:
        x = decode_blob(row["input_blob"], row["input_blob_encoding"], row["input_blob_dimensions"])
        candidate = _candidate_mask_for_output(x, output_shape) if task == "segmentation" else None
        y = decode_blob(row["output_blob"], row["output_blob_encoding"], row["output_blob_dimensions"])
        if task == "segmentation" and _should_tile_segmentation(x, config):
            tiled = _expand_tiled_segmentation_row(row, x, y, config)
            for tile_x, tile_y, tile_record in tiled:
                xs.append(tile_x)
                ys.append(tile_y)
                records.append(tile_record)
            continue
        if task == "segmentation":
            x = prepare_segmentation_input(x, input_shape, config)
        xs.append(
            prepare_classification_input(x, input_shape, config)
            if task == "classification"
            else normalize_input(x, input_shape)
        )
        if task == "classification":
            ys.append(int(y if y is not None else row.get("label_text")))
        else:
            if output_shape is None:
                raise ValueError("data.output_shape is required for segmentation")
            target = np.asarray(y, dtype="float32")
            if (row.get("output_blob_encoding") or "").lower() == "png":
                target = (target > 0).astype("float32")
            target = resize_array_to_shape(target, output_shape, mask=True)
            validated_target = target.reshape(output_shape)
            if segmentation_target_mode(config) == CANDIDATE_DELTA:
                if candidate is None:
                    raise ValueError(f"Sample {row['uuid']} has no candidate mask required for candidate_delta training")
                target = candidate_delta(candidate, validated_target)
            ys.append(target.reshape(output_shape))
        records.append(
            {
                "uuid": row["uuid"],
                "split": row.get("split") or "train",
                "label_text": row.get("label_text"),
                "sample_weight": row.get("sample_weight"),
                "metadata": json.loads(row["metadata_json"]) if row.get("metadata_json") else {},
                "candidate_mask": candidate,
                "validated_mask": validated_target if task == "segmentation" else None,
            }
        )
    return np.stack(xs), np.asarray(ys), records


def load_prediction_arrays(
    sqlite_path: str | Path,
    config: dict[str, Any],
    split: str | None = None,
) -> tuple[np.ndarray, list[Any | None], list[dict[str, Any]]]:
    """Load every model input, including segmentation ROIs without validated masks."""
    task = config["run"]["task"]
    if task == "segmentation":
        from oracle_builder.datasets.legacy_roi import (
            ensure_mask_refinement_database,
        )

        ensure_mask_refinement_database(sqlite_path)
    rows = read_rows(sqlite_path)
    if task == "segmentation":
        with_targets = [row for row in rows if row.get("output_blob")]
        without_targets = [row for row in rows if not row.get("output_blob")]
        assigned = assign_run_splits(with_targets, config)
        assigned += assign_run_splits(without_targets, config)
    else:
        assigned = assign_run_splits(rows, config)
    if split:
        assigned = [row for row in assigned if row.get("split") == split]
    if not assigned:
        raise ValueError(f"No samples found for split={split!r}")

    input_shape = config["data"]["input_shape"]
    output_shape = config["data"].get("output_shape")
    xs: list[np.ndarray] = []
    targets: list[Any | None] = []
    records: list[dict[str, Any]] = []
    for row in assigned:
        x = decode_blob(row["input_blob"], row["input_blob_encoding"], row["input_blob_dimensions"])
        decoded_target = (
            decode_blob(row["output_blob"], row["output_blob_encoding"], row["output_blob_dimensions"])
            if row.get("output_blob") is not None
            else None
        )
        if task == "segmentation" and _should_tile_segmentation(x, config):
            tiled = _expand_tiled_segmentation_row(row, x, decoded_target, config)
            for tile_x, tile_y, tile_record in tiled:
                xs.append(tile_x)
                targets.append(tile_y)
                records.append(tile_record)
            continue
        candidate = _candidate_mask_for_output(x, output_shape) if task == "segmentation" else None
        if task == "segmentation" and segmentation_target_mode(config) == CANDIDATE_DELTA and candidate is None:
            raise ValueError(f"Sample {row['uuid']} has no candidate mask required for candidate_delta prediction")
        if task == "segmentation":
            x = prepare_segmentation_input(x, input_shape, config)
        xs.append(
            prepare_classification_input(x, input_shape, config)
            if task == "classification"
            else normalize_input(x, input_shape)
        )
        if row.get("output_blob") is None:
            target = None
        else:
            target = decode_blob(row["output_blob"], row["output_blob_encoding"], row["output_blob_dimensions"])
            if task == "classification":
                target = int(target if target is not None else row.get("label_text"))
            else:
                target = np.asarray(target, dtype="float32")
                if (row.get("output_blob_encoding") or "").lower() == "png":
                    target = (target > 0).astype("float32")
                target = resize_array_to_shape(target, output_shape, mask=True)
                validated_target = target.reshape(output_shape)
                if segmentation_target_mode(config) == CANDIDATE_DELTA:
                    target = candidate_delta(candidate, validated_target)
        targets.append(target)
        records.append(
            {
                "uuid": row["uuid"],
                "split": row.get("split") or "train",
                "label_text": row.get("label_text"),
                "sample_weight": row.get("sample_weight"),
                "metadata": json.loads(row["metadata_json"]) if row.get("metadata_json") else {},
                "candidate_mask": candidate,
                "validated_mask": validated_target if task == "segmentation" and row.get("output_blob") is not None else None,
            }
        )
    return np.stack(xs), targets, records


def _candidate_mask_for_output(array: Any, output_shape: list[int] | tuple[int, ...] | None) -> np.ndarray | None:
    if output_shape is None:
        return None
    value = np.asarray(array)
    if value.ndim != 3 or value.shape[-1] != 2:
        return None
    return resize_array_to_shape(value[..., 1], output_shape, mask=True).reshape(output_shape)


def _should_tile_segmentation(array: Any, config: dict[str, Any]) -> bool:
    tiling = config.get("tiling", {})
    if not tiling.get("enabled", False):
        return False
    value = np.asarray(array)
    tile_h, tile_w = (int(part) for part in config["data"]["input_shape"][:2])
    if tiling.get("tile_large_rois_only", True):
        return value.shape[0] > tile_h or value.shape[1] > tile_w
    return True


def _normalize_native_input(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.kind in {"u", "i"} and result.max(initial=0) > 1:
        result = result.astype("float32") / float(np.iinfo(result.dtype).max)
    return result.astype("float32")


def _native_model_input(array: Any, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None]:
    value = np.asarray(array)
    candidate = None
    if value.ndim == 3 and value.shape[-1] == 2:
        roi = _normalize_native_input(value[..., 0])
        candidate = (value[..., 1] > 0).astype("float32")
        channels = [roi, candidate]
        if config.get("data", {}).get("candidate_sdf", False):
            channels.append(
                signed_distance_field(
                    candidate,
                    float(config["data"].get("candidate_sdf_clip_distance", 32.0)),
                )
            )
        return np.stack(channels, axis=-1), candidate[..., None]
    return _normalize_native_input(value), candidate


def _extract_input_tile(full_input: np.ndarray, plan: dict[str, Any], candidate_sdf: bool) -> np.ndarray:
    if not candidate_sdf:
        return extract_tile(full_input, plan, fill_value=0.0)
    pieces = [
        extract_tile(full_input[..., 0], plan, fill_value=0.0),
        extract_tile(full_input[..., 1], plan, fill_value=0.0),
        extract_tile(full_input[..., 2], plan, fill_value=-1.0),
    ]
    return np.stack(pieces, axis=-1).astype("float32")


def _expand_tiled_segmentation_row(
    row: dict[str, Any],
    raw_input: Any,
    raw_target: Any | None,
    config: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray | None, dict[str, Any]]]:
    from oracle_builder.training.spatial_weights import boundary_distance_weights

    full_input, candidate = _native_model_input(raw_input, config)
    source_shape = tuple(int(part) for part in full_input.shape[:2])
    tile_shape = tuple(int(part) for part in config["data"]["input_shape"][:2])
    overlap = float(config.get("tiling", {}).get("overlap_fraction", 0.5))
    plans = plan_tiles(source_shape, tile_shape, overlap)
    validated = None
    full_target = None
    if raw_target is not None:
        validated = (np.asarray(raw_target) > 0).astype("float32")
        if validated.ndim == 2:
            validated = validated[..., None]
        if validated.shape[:2] != source_shape:
            raise ValueError(
                f"Sample {row['uuid']} input shape {source_shape} does not match target shape {validated.shape[:2]}"
            )
        full_target = validated
        if segmentation_target_mode(config) == CANDIDATE_DELTA:
            if candidate is None:
                raise ValueError(f"Sample {row['uuid']} has no candidate mask required for candidate_delta training")
            full_target = candidate_delta(candidate, validated)
    if segmentation_target_mode(config) == CANDIDATE_DELTA and candidate is None:
        raise ValueError(f"Sample {row['uuid']} has no candidate mask required for candidate_delta prediction")

    coverage = coverage_map(plans)
    normalize_coverage = bool(config.get("tiling", {}).get("normalize_training_coverage", True))
    spatial = bool(config.get("training", {}).get("spatial_edge_weighting", False))
    full_weights = None
    if full_target is not None and spatial:
        full_weights = boundary_distance_weights(
            full_target,
            float(config["training"].get("edge_weight_lambda", 1.0)),
            float(config["training"].get("edge_weight_sigma", 5.0)),
        )
    elif full_target is not None and normalize_coverage:
        full_weights = np.ones(source_shape, dtype="float32")
    if full_weights is not None and normalize_coverage:
        full_weights = full_weights / coverage

    source_record = {
        "uuid": row["uuid"],
        "split": row.get("split") or "train",
        "label_text": row.get("label_text"),
        "sample_weight": row.get("sample_weight"),
        "metadata": json.loads(row["metadata_json"]) if row.get("metadata_json") else {},
        "candidate_mask": candidate,
        "validated_mask": validated,
        "tile_count": len(plans),
        "source_shape": list(source_shape),
    }
    outputs = []
    for index, plan in enumerate(plans):
        record = {
            **source_record,
            "uuid": f"{row['uuid']}::tile:{index}",
            "source_uuid": row["uuid"],
            "tile_index": index,
            "tile_plan": plan,
            "source_record": source_record,
        }
        if full_weights is not None:
            record["pixel_weights"] = extract_tile(full_weights, plan, fill_value=0.0)
        outputs.append(
            (
                _extract_input_tile(
                    full_input, plan, bool(config.get("data", {}).get("candidate_sdf", False))
                ),
                extract_tile(full_target, plan, fill_value=0.0) if full_target is not None else None,
                record,
            )
        )
    return outputs


def make_tf_datasets(sqlite_path: str | Path, config: dict[str, Any]):
    if (
        config["run"]["task"] == "classification"
        and config.get("data", {}).get("streaming", {}).get("enabled", True)
    ):
        from oracle_builder.data.sqlite_stream import make_streaming_classification_bundle

        bundle = make_streaming_classification_bundle(sqlite_path, config)
        return bundle.datasets, {
            split: list(index.iter_records())
            for split, index in bundle.indices.items()
        }
    import tensorflow as tf
    from oracle_builder.training.augmentation import apply_training_augmentation
    from oracle_builder.training.spatial_weights import batch_boundary_distance_weights

    datasets = {}
    records_by_split = {}
    for split in ("train", "validation", "test"):
        try:
            x, y, records = load_arrays(sqlite_path, config, split=split)
        except ValueError as exc:
            if not str(exc).startswith("No samples found for split="):
                raise
            continue
        training_config = config.get("training", {})
        has_precomputed_weights = any("pixel_weights" in record for record in records)
        if training_config.get("spatial_edge_weighting", False) or has_precomputed_weights:
            generated = batch_boundary_distance_weights(
                y,
                weight_lambda=float(training_config.get("edge_weight_lambda", 1.0)),
                sigma=float(training_config.get("edge_weight_sigma", 5.0)),
            ) if training_config.get("spatial_edge_weighting", False) else np.ones(y.shape[:3], dtype="float32")
            weights = np.stack(
                [
                    record.get("pixel_weights", generated[index])
                    for index, record in enumerate(records)
                ]
            )
            dataset = tf.data.Dataset.from_tensor_slices((x, y, weights))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if split == "train":
            dataset = dataset.shuffle(config["data"].get("shuffle_buffer", 512), seed=config["run"].get("seed", 123))
            repeats_per_epoch = int(config.get("augmentation", {}).get("repeats_per_epoch", 1))
            if repeats_per_epoch < 1:
                raise ValueError("augmentation.repeats_per_epoch must be at least 1")
            if repeats_per_epoch > 1:
                dataset = dataset.repeat(repeats_per_epoch)
        dataset = dataset.batch(config["data"].get("batch_size", 16))
        if split == "train":
            dataset = apply_training_augmentation(dataset, config)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        datasets[split] = dataset
        records_by_split[split] = records
    if "train" not in datasets:
        raise ValueError("Dataset must contain or create a train split")
    return datasets, records_by_split


def resize_segmentation_input(array: Any, input_shape: list[int] | tuple[int, ...]) -> np.ndarray:
    value = np.asarray(array)
    target_shape = tuple(int(part) for part in input_shape)
    if value.shape == target_shape:
        return value
    if value.ndim == 3 and value.shape[-1] == 2 and len(target_shape) == 3 and target_shape[-1] == 2:
        roi = resize_array_to_shape(value[..., 0], target_shape[:2], mask=False)
        candidate = resize_array_to_shape(value[..., 1], target_shape[:2], mask=True)
        if value.dtype.kind in {"u", "i"}:
            candidate = np.where(candidate > 0, np.iinfo(value.dtype).max, 0).astype(value.dtype)
        return np.stack([roi, candidate], axis=-1)
    return resize_array_to_shape(value, target_shape, mask=False)


def prepare_segmentation_input(
    array: Any,
    input_shape: list[int] | tuple[int, ...],
    config: dict[str, Any],
) -> np.ndarray:
    distance_mode = config.get("data", {}).get("candidate_distance", "none")
    if distance_mode == "none" and config.get("data", {}).get("candidate_sdf", False):
        distance_mode = "euclidean_sdf"  # legacy configuration alias
    if distance_mode == "none":
        return resize_segmentation_input(array, input_shape)
    value = np.asarray(array)
    target_shape = tuple(int(part) for part in input_shape)
    if value.ndim != 3 or value.shape[-1] != 2 or len(target_shape) != 3 or target_shape[-1] != 3:
        raise ValueError(
            f"Candidate distance input requires raw ROI+candidate input and target shape [H, W, 3], got {value.shape} -> {target_shape}"
        )
    roi = resize_array_to_shape(value[..., 0], target_shape[:2], mask=False)
    candidate = resize_array_to_shape(value[..., 1], target_shape[:2], mask=True)
    roi = np.asarray(roi, dtype="float32")
    if value.dtype.kind in {"u", "i"} and roi.max(initial=0) > 1:
        roi /= float(np.iinfo(value.dtype).max)
    if distance_mode == "euclidean_sdf":
        distance = signed_distance_field(candidate, clip_distance=float(config["data"].get("candidate_sdf_clip_distance", 32.0)))
    elif distance_mode == "geodesic":
        from oracle_builder.data.signed_distance import geodesic_distance_field
        settings = config["data"].get("geodesic_distance", {})
        distance = geodesic_distance_field(roi, candidate, clip_distance=float(config["data"].get("candidate_distance_clip", 32.0)), **settings)
    else:
        raise ValueError(f"Unsupported data.candidate_distance: {distance_mode}")
    return np.stack([roi, candidate.astype("float32"), distance], axis=-1)


def resize_array_to_shape(
    array: Any,
    target_shape: list[int] | tuple[int, ...],
    mask: bool = False,
) -> np.ndarray:
    value = np.asarray(array)
    target = tuple(int(part) for part in target_shape)
    if value.shape == target:
        return value

    if len(target) == 2:
        target_h, target_w = target
        target_channels = None
    elif len(target) == 3:
        target_h, target_w, target_channels = target
    else:
        raise ValueError(f"Unsupported target shape for resizing: {target}")

    if value.ndim == 2 and target_channels is not None:
        value = value[..., None]
    if value.ndim == 3 and value.shape[-1] == 1 and target_channels is None:
        value = value[..., 0]
    if value.ndim not in {2, 3}:
        raise ValueError(f"Unsupported array shape for resizing: {value.shape}")
    if value.ndim == 3 and target_channels is not None and value.shape[-1] != target_channels:
        raise ValueError(f"Cannot resize array with {value.shape[-1]} channels to target shape {target}")

    source_h, source_w = value.shape[:2]
    scale = min(target_h / source_h, target_w / source_w)
    resized_h = min(target_h, max(1, int(round(source_h * scale))))
    resized_w = min(target_w, max(1, int(round(source_w * scale))))
    offset_y = (target_h - resized_h) // 2
    offset_x = (target_w - resized_w) // 2
    resample = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR

    def resize_channel(channel: np.ndarray) -> np.ndarray:
        channel_array = np.asarray(channel)
        pil_input = channel_array.astype("float32") if channel_array.dtype.kind == "f" else channel_array
        resized = Image.fromarray(pil_input).resize((resized_w, resized_h), resample=resample)
        result = np.zeros((target_h, target_w), dtype=np.asarray(resized).dtype)
        result[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = np.asarray(resized)
        if mask:
            return (result > 0).astype("float32")
        return result.astype(channel_array.dtype, copy=False)

    if value.ndim == 2:
        resized_value = resize_channel(value)
    else:
        resized_value = np.stack([resize_channel(value[..., index]) for index in range(value.shape[-1])], axis=-1)

    if len(target) == 3 and resized_value.ndim == 2:
        resized_value = resized_value[..., None]
    if mask:
        resized_value = (resized_value > 0).astype("float32")
    return resized_value.reshape(target)


def create_synthetic_classification(path: str | Path, n: int = 48, shape: tuple[int, int, int] = (128, 128, 3), classes: int = 3) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    ensure_schema(connection, "classification", name="synthetic-classification")
    repository = SQLiteDatasetRepository(connection)
    labels = [
        repository.add_classification_label(index, str(index))
        for index in range(classes)
    ]
    rng = np.random.default_rng(123)
    for i in range(n):
        label = i % classes
        image = rng.normal(0.15, 0.05, shape).astype("float32")
        channel = label % shape[-1]
        image[..., channel] += 0.7
        image = np.clip(image, 0, 1)
        asset = repository.add_asset(
            encode_npy(image),
            encoding="npy",
            media_type="application/x-npy",
            shape=shape,
            dtype=str(image.dtype),
        )
        item_id = repository.add_item(
            source_key=f"synthetic/{i}",
            metadata={"synthetic": True},
        )
        repository.add_classification_item(
            item_id=item_id,
            image_asset_id=asset.asset_id,
            label_id=labels[label],
            source="synthetic",
        )
    connection.commit()
    connection.close()


def create_synthetic_segmentation(path: str | Path, n: int = 24, shape: tuple[int, int, int] = (256, 256, 3)) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    ensure_schema(connection, "mask_refinement", name="synthetic-mask-refinement")
    repository = SQLiteDatasetRepository(connection)
    rng = np.random.default_rng(123)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    for i in range(n):
        center = rng.integers(shape[0] // 4, shape[0] * 3 // 4, size=2)
        radius = rng.integers(shape[0] // 8, shape[0] // 4)
        mask = (((yy - center[0]) ** 2 + (xx - center[1]) ** 2) <= radius**2).astype("float32")[..., None]
        image = np.repeat(mask, shape[-1], axis=-1) + rng.normal(0, 0.08, shape)
        image = np.clip(image, 0, 1).astype("float32")
        image_asset = repository.add_asset(
            encode_npy(image),
            encoding="npy",
            media_type="application/x-npy",
            shape=shape,
            dtype=str(image.dtype),
        )
        mask_asset = repository.add_asset(
            encode_npy(mask),
            encoding="npy",
            media_type="application/x-npy",
            shape=mask.shape,
            dtype=str(mask.dtype),
        )
        item_id = repository.add_item(
            source_key=f"synthetic/{i}",
            metadata={"synthetic": True},
        )
        repository.add_mask_item(
            item_id=item_id,
            image_asset_id=image_asset.asset_id,
            candidate_mask_asset_id=None,
        )
        connection.execute(
            """
            INSERT INTO mask_annotations (
                annotation_id, item_id, mask_asset_id, created_at, annotator, method,
                parameters_json, validation_json, status, is_current,
                parent_annotation_id, notes
            ) VALUES (?, ?, ?, ?, NULL, 'synthetic', '{}', '{}', 'accepted', 1, NULL, NULL)
            """,
            (str(uuid.uuid4()), item_id, mask_asset.asset_id, utc_now()),
        )
    connection.commit()
    connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tiny synthetic oracle-builder SQLite datasets.")
    parser.add_argument("--classification", type=Path)
    parser.add_argument("--segmentation", type=Path)
    args = parser.parse_args()
    if args.classification:
        create_synthetic_classification(args.classification)
    if args.segmentation:
        create_synthetic_segmentation(args.segmentation)
    if not args.classification and not args.segmentation:
        parser.error("Pass --classification PATH or --segmentation PATH")


if __name__ == "__main__":
    main()
