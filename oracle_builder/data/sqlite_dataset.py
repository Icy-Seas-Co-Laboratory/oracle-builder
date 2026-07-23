from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from oracle_builder.data.decoders import decode_blob, encode_npy, normalize_input
from oracle_builder.data.splits import assign_missing_splits


SAMPLE_COLUMNS = [
    "uuid",
    "split",
    "input_blob",
    "input_blob_encoding",
    "input_blob_dimensions",
    "output_blob",
    "output_blob_encoding",
    "output_blob_dimensions",
    "label_text",
    "sample_weight",
    "metadata_json",
]


def synthetic_splits(n: int, validation_fraction: float = 0.2, test_fraction: float = 0.1) -> list[str]:
    n_test = int(round(n * test_fraction))
    n_validation = int(round(n * validation_fraction))
    n_train = max(0, n - n_validation - n_test)
    return ["train"] * n_train + ["validation"] * n_validation + ["test"] * n_test


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS samples (
            uuid TEXT PRIMARY KEY,
            split TEXT,
            input_blob BLOB,
            input_blob_encoding TEXT,
            input_blob_dimensions TEXT,
            output_blob BLOB,
            output_blob_encoding TEXT,
            output_blob_dimensions TEXT,
            label_text TEXT,
            sample_weight REAL,
            metadata_json TEXT
        )
        """
    )


def read_rows(sqlite_path: str | Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(f"SELECT {', '.join(SAMPLE_COLUMNS)} FROM samples").fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def load_arrays(sqlite_path: str | Path, config: dict[str, Any], split: str | None = None) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rows = read_rows(sqlite_path)
    task = config["run"]["task"]
    if task == "segmentation":
        rows = [row for row in rows if row.get("output_blob")]
    rows = assign_missing_splits(
        rows,
        config["data"].get("validation_split", 0.2),
        config["data"].get("test_split", 0.1),
        config["run"].get("seed", 123),
    )
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
        y = decode_blob(row["output_blob"], row["output_blob_encoding"], row["output_blob_dimensions"])
        if task == "segmentation":
            x = resize_segmentation_input(x, input_shape)
        xs.append(normalize_input(x, input_shape))
        if task == "classification":
            ys.append(int(y if y is not None else row.get("label_text")))
        else:
            if output_shape is None:
                raise ValueError("data.output_shape is required for segmentation")
            target = np.asarray(y, dtype="float32")
            if (row.get("output_blob_encoding") or "").lower() == "png":
                target = (target > 0).astype("float32")
            target = resize_array_to_shape(target, output_shape, mask=True)
            ys.append(target.reshape(output_shape))
        records.append(
            {
                "uuid": row["uuid"],
                "split": row.get("split") or "train",
                "label_text": row.get("label_text"),
                "sample_weight": row.get("sample_weight"),
                "metadata": json.loads(row["metadata_json"]) if row.get("metadata_json") else {},
            }
        )
    return np.stack(xs), np.asarray(ys), records


def load_prediction_arrays(
    sqlite_path: str | Path,
    config: dict[str, Any],
    split: str | None = None,
) -> tuple[np.ndarray, list[Any | None], list[dict[str, Any]]]:
    """Load every model input, including segmentation ROIs without validated masks."""
    rows = read_rows(sqlite_path)
    task = config["run"]["task"]
    validation_split = config["data"].get("validation_split", 0.2)
    test_split = config["data"].get("test_split", 0.1)
    seed = config["run"].get("seed", 123)
    if task == "segmentation":
        with_targets = [row for row in rows if row.get("output_blob")]
        without_targets = [row for row in rows if not row.get("output_blob")]
        assigned = assign_missing_splits(with_targets, validation_split, test_split, seed)
        assigned += assign_missing_splits(without_targets, validation_split, test_split, seed)
    else:
        assigned = assign_missing_splits(rows, validation_split, test_split, seed)
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
        if task == "segmentation":
            x = resize_segmentation_input(x, input_shape)
        xs.append(normalize_input(x, input_shape))
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
        targets.append(target)
        records.append(
            {
                "uuid": row["uuid"],
                "split": row.get("split") or "train",
                "label_text": row.get("label_text"),
                "sample_weight": row.get("sample_weight"),
                "metadata": json.loads(row["metadata_json"]) if row.get("metadata_json") else {},
            }
        )
    return np.stack(xs), targets, records


def make_tf_datasets(sqlite_path: str | Path, config: dict[str, Any]):
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
        if training_config.get("spatial_edge_weighting", False):
            weights = batch_boundary_distance_weights(
                y,
                weight_lambda=float(training_config.get("edge_weight_lambda", 1.0)),
                sigma=float(training_config.get("edge_weight_sigma", 5.0)),
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
    ensure_schema(connection)
    rng = np.random.default_rng(123)
    splits = synthetic_splits(n)
    for i in range(n):
        label = i % classes
        image = rng.normal(0.15, 0.05, shape).astype("float32")
        channel = label % shape[-1]
        image[..., channel] += 0.7
        image = np.clip(image, 0, 1)
        connection.execute(
            "INSERT OR REPLACE INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                splits[i],
                encode_npy(image),
                "npy",
                json.dumps(list(shape)),
                str(label).encode("utf-8"),
                "int",
                None,
                str(label),
                None,
                json.dumps({"synthetic": True}),
            ),
        )
    connection.commit()
    connection.close()


def create_synthetic_segmentation(path: str | Path, n: int = 24, shape: tuple[int, int, int] = (256, 256, 3)) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    ensure_schema(connection)
    rng = np.random.default_rng(123)
    splits = synthetic_splits(n)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    for i in range(n):
        center = rng.integers(shape[0] // 4, shape[0] * 3 // 4, size=2)
        radius = rng.integers(shape[0] // 8, shape[0] // 4)
        mask = (((yy - center[0]) ** 2 + (xx - center[1]) ** 2) <= radius**2).astype("float32")[..., None]
        image = np.repeat(mask, shape[-1], axis=-1) + rng.normal(0, 0.08, shape)
        image = np.clip(image, 0, 1).astype("float32")
        connection.execute(
            "INSERT OR REPLACE INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                splits[i],
                encode_npy(image),
                "npy",
                json.dumps(list(shape)),
                encode_npy(mask),
                "npy",
                json.dumps([shape[0], shape[1], 1]),
                None,
                None,
                json.dumps({"synthetic": True}),
            ),
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
