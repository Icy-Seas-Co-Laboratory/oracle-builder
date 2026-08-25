from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import tensorflow as tf

from oracle_builder.data.decoders import decode_blob, prepare_classification_input
from oracle_builder.data.splits import assign_run_splits
from oracle_builder.training.augmentation import apply_training_augmentation


@dataclass(frozen=True)
class SQLiteSampleRef:
    item_id: str
    uuid: str
    split: str
    target: int | None
    input_encoding: str | None
    input_dimensions: str | None
    metadata_json: str | None

    def record(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "split": self.split,
            "label_text": None,
            "sample_weight": None,
            "metadata": json.loads(self.metadata_json) if self.metadata_json else {},
            "candidate_mask": None,
            "validated_mask": None,
        }


@dataclass
class SQLiteSplitIndex:
    sqlite_path: Path
    split: str
    refs: list[SQLiteSampleRef]

    def __len__(self) -> int:
        return len(self.refs)

    def iter_records(self) -> Iterator[dict[str, Any]]:
        for ref in self.refs:
            yield ref.record()


@dataclass
class SQLiteDatasetBundle:
    datasets: dict[str, tf.data.Dataset]
    indices: dict[str, SQLiteSplitIndex]
    source: "SQLiteClassificationSource"

    @property
    def counts(self) -> dict[str, int]:
        return {split: len(index) for split, index in self.indices.items()}


def build_classification_index(
    sqlite_path: str | Path,
    config: dict[str, Any],
    split: str,
    *,
    labeled_only: bool,
) -> SQLiteSplitIndex:
    path = Path(sqlite_path)
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            """
            SELECT di.item_id, l.class_index, a.encoding, a.shape_json,
                   di.metadata_json
            FROM dataset_items di
            JOIN classification_items ci ON ci.item_id = di.item_id
            JOIN assets a ON a.asset_id = ci.image_asset_id
            LEFT JOIN classification_annotations ca
              ON ca.item_id = di.item_id AND ca.is_current = 1
            LEFT JOIN classification_labels l ON l.label_id = ca.label_id
            ORDER BY di.item_id
            """
        ).fetchall()
    split_records = assign_run_splits(
        [{"uuid": str(row[0])} for row in rows],
        config,
    )
    resolved_splits = {row["uuid"]: row["split"] for row in split_records}
    refs = []
    for row in rows:
        resolved_split = resolved_splits[str(row[0])]
        if resolved_split != split:
            continue
        target = int(row[1]) if row[1] is not None else None
        if labeled_only and target is None:
            continue
        refs.append(
            SQLiteSampleRef(
                item_id=str(row[0]),
                uuid=str(row[0]),
                split=str(resolved_split),
                target=target,
                input_encoding=row[2],
                input_dimensions=row[3],
                metadata_json=row[4],
            )
        )
    return SQLiteSplitIndex(path, split, refs)


def build_all_classification_index(
    sqlite_path: str | Path,
    config: dict[str, Any],
    *,
    labeled_only: bool,
) -> SQLiteSplitIndex:
    if config.get("_external_inference"):
        return build_classification_index(
            sqlite_path, config, "inference", labeled_only=labeled_only
        )
    refs = []
    for split in ("train", "validation", "test"):
        refs.extend(
            build_classification_index(
                sqlite_path,
                config,
                split,
                labeled_only=labeled_only,
            ).refs
        )
    return SQLiteSplitIndex(Path(sqlite_path), "all", refs)


class SQLiteClassificationSource:
    """Thread-local, read-only SQLite image source used by tf.data maps."""

    def __init__(self, sqlite_path: str | Path, config: dict[str, Any]):
        self.sqlite_path = Path(sqlite_path).resolve()
        self.config = config
        self.input_shape = tuple(int(value) for value in config["data"]["input_shape"])
        self._local = threading.local()

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            uri = f"file:{self.sqlite_path}?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
            cache_kib = int(
                self.config.get("data", {})
                .get("streaming", {})
                .get("sqlite_cache_kib", 65536)
            )
            connection.execute(f"PRAGMA cache_size = {-max(cache_kib, 1)}")
            self._local.connection = connection
        return connection

    def read_image(self, item_id: str) -> np.ndarray:
        row = self._connection().execute(
            """
            SELECT a.payload, a.encoding, a.shape_json
            FROM classification_items ci
            JOIN assets a ON a.asset_id = ci.image_asset_id
            WHERE ci.item_id = ?
            """,
            (str(item_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"SQLite dataset item {item_id!r} no longer exists")
        decoded = decode_blob(row[0], row[1], row[2])
        return prepare_classification_input(decoded, self.input_shape, self.config)

    def _tf_read_image(self, item_id):
        image = tf.py_function(
            lambda value: self.read_image(value.numpy().decode("utf-8")),
            [item_id],
            Tout=tf.float32,
        )
        image.set_shape(self.input_shape)
        return image

    def training_dataset(
        self,
        index: SQLiteSplitIndex,
        *,
        shuffle: bool,
        augment: bool,
    ) -> tf.data.Dataset:
        if any(ref.target is None for ref in index.refs):
            raise ValueError("Supervised classification dataset contains unlabeled rows")
        item_ids = np.asarray([ref.item_id for ref in index.refs], dtype=str)
        labels = np.asarray([ref.target for ref in index.refs], dtype="int64")
        dataset = tf.data.Dataset.from_tensor_slices((item_ids, labels))
        streaming = self.config.get("data", {}).get("streaming", {})
        deterministic = bool(streaming.get("deterministic", True))
        workers = max(1, int(streaming.get("reader_workers", 4)))
        if shuffle:
            dataset = dataset.shuffle(
                min(
                    max(1, int(self.config["data"].get("shuffle_buffer", 512))),
                    max(1, len(index)),
                ),
                seed=int(self.config["run"].get("seed", 123)),
                reshuffle_each_iteration=True,
            )
            repeats = int(self.config.get("augmentation", {}).get("repeats_per_epoch", 1))
            if repeats < 1:
                raise ValueError("augmentation.repeats_per_epoch must be at least 1")
            if repeats > 1:
                dataset = dataset.repeat(repeats)
        dataset = dataset.map(
            lambda item_id, label: (self._tf_read_image(item_id), label),
            num_parallel_calls=workers,
            deterministic=deterministic,
        )
        dataset = dataset.batch(int(self.config["data"].get("batch_size", 16)))
        if augment:
            dataset = apply_training_augmentation(dataset, self.config)
        return dataset.prefetch(max(1, int(streaming.get("prefetch_batches", 2))))

    def indexed_image_dataset(
        self,
        index: SQLiteSplitIndex,
        *,
        batch_size: int | None = None,
        shuffle: bool = False,
    ) -> tf.data.Dataset:
        item_ids = np.asarray([ref.item_id for ref in index.refs], dtype=str)
        positions = np.arange(len(index.refs), dtype="int64")
        dataset = tf.data.Dataset.from_tensor_slices((item_ids, positions))
        streaming = self.config.get("data", {}).get("streaming", {})
        if shuffle:
            dataset = dataset.shuffle(
                min(
                    max(1, int(self.config["data"].get("shuffle_buffer", 512))),
                    max(1, len(index)),
                ),
                seed=int(self.config.get("run", {}).get("seed", 123)),
                reshuffle_each_iteration=True,
            )
        dataset = dataset.map(
            lambda item_id, position: (self._tf_read_image(item_id), position),
            num_parallel_calls=max(1, int(streaming.get("reader_workers", 4))),
            deterministic=bool(streaming.get("deterministic", True)),
        )
        dataset = dataset.batch(
            int(batch_size or self.config["data"].get("batch_size", 16))
        )
        return dataset.prefetch(max(1, int(streaming.get("prefetch_batches", 2))))

    def image_dataset(
        self,
        index: SQLiteSplitIndex,
        *,
        batch_size: int | None = None,
        shuffle: bool = False,
    ) -> tf.data.Dataset:
        dataset = self.indexed_image_dataset(
            index,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return dataset.map(
            lambda image, _position: image,
            deterministic=True,
        )


def make_streaming_classification_bundle(
    sqlite_path: str | Path,
    config: dict[str, Any],
) -> SQLiteDatasetBundle:
    source = SQLiteClassificationSource(sqlite_path, config)
    indices = {}
    datasets = {}
    for split in ("train", "validation", "test"):
        index = build_classification_index(
            sqlite_path,
            config,
            split,
            labeled_only=True,
        )
        if not index.refs:
            continue
        indices[split] = index
        datasets[split] = source.training_dataset(
            index,
            shuffle=split == "train",
            augment=split == "train",
        )
    if "train" not in datasets:
        raise ValueError("Dataset must contain or create a train split")
    return SQLiteDatasetBundle(datasets=datasets, indices=indices, source=source)
