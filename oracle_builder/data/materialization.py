"""Immutable, shared prepared-input caches for frozen classification datasets.

SQLite remains the authoritative dataset format.  A materialization is an
optional derived cache: it stores only deterministic decoded/preprocessed image
tensors and can always be recreated from the frozen source fingerprint.

The cache intentionally keeps random augmentation out of its payload.  Online
TensorFlow augmentation retains fresh views every epoch and avoids multiplying
cache size or accidentally sharing run-specific augmented images.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import tensorflow as tf

from oracle_builder.classification.metadata import (
    enabled as auxiliary_features_enabled,
    vector as auxiliary_feature_vector,
)
from oracle_builder.data.decoders import decode_blob, prepare_dataset_classification_input
from oracle_builder.training.augmentation import apply_training_augmentation

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from oracle_builder.data.sqlite_stream import SQLiteSplitIndex


MATERIALIZATION_SCHEMA_VERSION = "1.0.0"
_SUPPORTED_MODES = {"off", "run", "shared"}
_SUPPORTED_DTYPES = {"float16", "float32"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _item_ids_fingerprint(item_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(item_ids).encode("utf-8")).hexdigest()


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("data", {}).get("materialization", {})
    if not isinstance(value, dict):
        raise ValueError("data.materialization must be a table/object")
    return value


def _split_fingerprint(config: dict[str, Any]) -> str:
    manifest = config.get("_split_manifest", {})
    if not isinstance(manifest, dict):
        raise ValueError("data materialization requires a run split manifest")
    fingerprint = manifest.get("fingerprint_sha256")
    if fingerprint:
        return str(fingerprint)
    assignments = manifest.get("assignments")
    if isinstance(assignments, dict):
        assignments = [
            {"item_id": item_id, "split": split}
            for item_id, split in sorted(assignments.items())
        ]
    if not isinstance(assignments, list):
        raise ValueError("data materialization requires split assignments")
    return _sha256(assignments)


def _preprocessing_signature(config: dict[str, Any]) -> str:
    preprocessing = config.get("preprocessing", {})
    deterministic_keys = (
        "resize_mode", "normalization", "rescale", "invert", "pad_value",
        "upscale_limit", "pad_anchor", "pad_mode", "crop_anchor",
        "interpolation", "channel_mode", "percentile_low", "percentile_high",
        "channels", "resolved_channels", "derived_channels",
        "derive_after_augmentation",
    )
    semantic = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "input_shape": list(config["data"]["input_shape"]),
        "preprocessing": {
            key: preprocessing.get(key)
            for key in deterministic_keys
            if key in preprocessing
        },
    }
    return _sha256(semantic)


def _cache_root(config: dict[str, Any], settings: dict[str, Any]) -> Path:
    mode = str(settings.get("mode", "off")).lower()
    configured = settings.get("root", ".oracle-runtime/cache/datasets")
    root = Path(str(configured)).expanduser()
    if mode == "run":
        run_dir = config.get("paths", {}).get("run_dir")
        if not run_dir:
            raise ValueError("data.materialization.mode='run' requires config.paths.run_dir")
        return Path(run_dir) / "cache" / "prepared-inputs"
    return root.resolve()


def _cache_path(config: dict[str, Any], settings: dict[str, Any]) -> tuple[Path, str, str]:
    dataset = config.get("dataset", {})
    fingerprint = str(dataset.get("fingerprint_sha256") or "")
    if not fingerprint:
        raise ValueError("data materialization requires the resolved dataset fingerprint")
    split_fingerprint = _split_fingerprint(config)
    preprocessing_fingerprint = _preprocessing_signature(config)
    identity = _sha256(
        {
            "dataset_fingerprint": fingerprint,
            "split_fingerprint": split_fingerprint,
            "preprocessing_fingerprint": preprocessing_fingerprint,
            "dtype": str(settings.get("dtype", "float32")).lower(),
            "shard_samples": int(settings.get("shard_samples", 1024)),
        }
    )
    root = _cache_root(config, settings)
    return root / fingerprint / identity, identity, preprocessing_fingerprint


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_records(destination: Path, rows: list[dict[str, Any]]) -> tuple[str, str]:
    """Prefer Parquet for inspection/filtering, with a portable JSONL fallback."""
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(destination / "records.parquet", index=False)
        return "records.parquet", "parquet"
    except Exception:
        target = destination / "records.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_canonical_json(row) + "\n")
        return target.name, "jsonl"


class _DirectoryLock:
    """A small cross-process lock whose completed cache is always authoritative."""

    def __init__(self, target: Path, wait_seconds: float):
        self.target = target
        self.lock = target.with_name(f".{target.name}.lock")
        self.wait_seconds = float(wait_seconds)
        self.acquired = False

    def __enter__(self):
        deadline = time.monotonic() + self.wait_seconds
        while True:
            try:
                self.lock.mkdir(parents=True)
                self.acquired = True
                _atomic_write_json(
                    self.lock / "owner.json",
                    {"pid": os.getpid(), "created_at": time.time()},
                )
                return self
            except FileExistsError:
                if (self.target / "manifest.json").exists():
                    return self
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for prepared-input cache {self.target}"
                    )
                time.sleep(0.25)

    def __exit__(self, exc_type, exc, traceback):
        if self.acquired:
            shutil.rmtree(self.lock, ignore_errors=True)


@dataclass(frozen=True)
class PreparedSplit:
    split: str
    item_ids: tuple[str, ...]
    shards: tuple[Path, ...]
    shard_offsets: tuple[int, ...]


class MaterializedClassificationSource:
    """tf.data source backed by immutable NPY shards instead of SQLite blobs."""

    def __init__(self, config: dict[str, Any], splits: dict[str, PreparedSplit], refs_by_id: dict[str, Any], report: dict[str, Any]):
        self.config = config
        self.splits = splits
        self.refs_by_id = refs_by_id
        self.report = report
        self.input_shape = tuple(int(value) for value in config["data"]["input_shape"])
        self._locations = {
            item_id: (split.split, offset)
            for split in splits.values()
            for offset, item_id in enumerate(split.item_ids)
        }
        self._local = threading.local()
        self._statistics_lock = threading.Lock()
        self._statistics = {"reads": 0, "read_seconds": 0.0}

    def _arrays(self) -> dict[tuple[str, int], np.ndarray]:
        arrays = getattr(self._local, "arrays", None)
        if arrays is None:
            arrays = {}
            self._local.arrays = arrays
        return arrays

    def read_image(self, item_id: str) -> np.ndarray:
        started = time.perf_counter()
        try:
            split_name, position = self._locations[str(item_id)]
            split = self.splits[split_name]
            shard_index, shard_offset = split.shard_offsets[position]
            key = (split_name, shard_index)
            arrays = self._arrays()
            if key not in arrays:
                arrays[key] = np.load(split.shards[shard_index], mmap_mode="r")
            return np.asarray(arrays[key][shard_offset], dtype="float32")
        finally:
            with self._statistics_lock:
                self._statistics["reads"] += 1
                self._statistics["read_seconds"] += time.perf_counter() - started

    def read_input(self, item_id: str):
        image = self.read_image(item_id)
        if auxiliary_features_enabled(self.config):
            ref = self.refs_by_id[str(item_id)]
            metadata = json.loads(ref.metadata_json) if ref.metadata_json else {}
            original_shape = json.loads(ref.input_dimensions) if ref.input_dimensions else []
            return image, auxiliary_feature_vector(self.config, metadata, original_shape)
        return image

    def statistics(self) -> dict[str, Any]:
        with self._statistics_lock:
            reads = int(self._statistics["reads"])
            seconds = float(self._statistics["read_seconds"])
        return {
            "source": "materialized_npy",
            "reads": reads,
            "read_seconds": seconds,
            "mean_read_milliseconds": (1000 * seconds / reads) if reads else 0.0,
            "cache_id": self.report["cache_id"],
        }

    def supports_index(self, index) -> bool:
        """Whether every requested source item is part of this split cache."""
        return all(ref.item_id in self._locations for ref in index.refs)

    def _tf_read_image(self, item_id):
        image = tf.py_function(
            lambda value: self.read_image(value.numpy().decode("utf-8")),
            [item_id],
            Tout=tf.float32,
        )
        image.set_shape(self.input_shape)
        return image

    def _tf_read_input(self, item_id):
        if not auxiliary_features_enabled(self.config):
            return self._tf_read_image(item_id)
        image, metadata = tf.py_function(
            lambda value: self.read_input(value.numpy().decode("utf-8")),
            [item_id],
            Tout=[tf.float32, tf.float32],
        )
        image.set_shape(self.input_shape)
        metadata_count = len(
            self.config.get("model", {}).get(
                "auxiliary_features_fitted",
                self.config.get("model", {}).get("auxiliary_features", []),
            )
        )
        metadata.set_shape((metadata_count,))
        return {"image": image, "metadata": metadata}

    def training_dataset(self, index, *, shuffle: bool, augment: bool) -> tf.data.Dataset:
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
                min(max(1, int(self.config["data"].get("shuffle_buffer", 512))), max(1, len(index))),
                seed=int(self.config["run"].get("seed", 123)),
                reshuffle_each_iteration=True,
            )
            repeats = int(self.config.get("augmentation", {}).get("repeats_per_epoch", 1))
            if repeats < 1:
                raise ValueError("augmentation.repeats_per_epoch must be at least 1")
            if repeats > 1:
                dataset = dataset.repeat(repeats)
        dataset = dataset.map(
            lambda item_id, label: (self._tf_read_input(item_id), label),
            num_parallel_calls=workers,
            deterministic=deterministic,
        ).batch(int(self.config["data"].get("batch_size", 16)))
        if augment:
            dataset = apply_training_augmentation(dataset, self.config)
        return dataset.prefetch(max(1, int(streaming.get("prefetch_batches", 2))))

    def indexed_image_dataset(self, index, *, batch_size: int | None = None, shuffle: bool = False) -> tf.data.Dataset:
        item_ids = np.asarray([ref.item_id for ref in index.refs], dtype=str)
        positions = np.arange(len(index.refs), dtype="int64")
        dataset = tf.data.Dataset.from_tensor_slices((item_ids, positions))
        streaming = self.config.get("data", {}).get("streaming", {})
        if shuffle:
            dataset = dataset.shuffle(
                min(max(1, int(self.config["data"].get("shuffle_buffer", 512))), max(1, len(index))),
                seed=int(self.config.get("run", {}).get("seed", 123)),
                reshuffle_each_iteration=True,
            )
        dataset = dataset.map(
            lambda item_id, position: (self._tf_read_input(item_id), position),
            num_parallel_calls=max(1, int(streaming.get("reader_workers", 4))),
            deterministic=bool(streaming.get("deterministic", True)),
        ).batch(int(batch_size or self.config["data"].get("batch_size", 16)))
        return dataset.prefetch(max(1, int(streaming.get("prefetch_batches", 2))))

    def image_dataset(self, index, *, batch_size: int | None = None, shuffle: bool = False) -> tf.data.Dataset:
        return self.indexed_image_dataset(index, batch_size=batch_size, shuffle=shuffle).map(
            lambda image, _position: image, deterministic=True
        )


def _manifest_splits(cache_dir: Path, manifest: dict[str, Any]) -> dict[str, PreparedSplit]:
    result = {}
    for split, value in manifest["splits"].items():
        result[split] = PreparedSplit(
            split=split,
            item_ids=tuple(value["item_ids"]),
            shards=tuple(cache_dir / filename for filename in value["shards"]),
            shard_offsets=tuple((int(shard), int(offset)) for shard, offset in value["shard_offsets"]),
        )
    return result


def _load_cache(cache_dir: Path, config: dict[str, Any], indices: dict[str, Any], cache_id: str) -> MaterializedClassificationSource | None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if manifest.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION or manifest.get("cache_id") != cache_id:
        return None
    splits = _manifest_splits(cache_dir, manifest)
    for name, index in indices.items():
        expected = [ref.item_id for ref in index.refs]
        cached = list(splits.get(name, PreparedSplit(name, (), (), ())).item_ids)
        if expected != cached:
            return None
    if any(not shard.exists() for split in splits.values() for shard in split.shards):
        return None
    refs_by_id = {ref.item_id: ref for index in indices.values() for ref in index.refs}
    report = {
        "enabled": True,
        "status": "reused",
        "source": "materialized_npy",
        "cache_id": cache_id,
        "path": str(cache_dir),
        "manifest_path": str(manifest_path),
        "records_format": manifest.get("records_format"),
        "bytes": manifest.get("bytes", 0),
        "splits": {
            split: value.get("count", 0)
            for split, value in manifest.get("splits", {}).items()
        },
    }
    return MaterializedClassificationSource(config, splits, refs_by_id, report)


class _ShardWriter:
    def __init__(self, destination: Path, split: str, count: int, shape: tuple[int, ...], dtype: np.dtype, shard_samples: int):
        self.destination = destination
        self.split = split
        self.count = count
        self.shape = shape
        self.dtype = dtype
        self.shard_samples = shard_samples
        self.paths: list[str] = []
        self.offsets: list[tuple[int, int]] = []
        self._array = None
        self._shard_index = -1
        self._position = 0

    def _open_shard(self) -> None:
        self._shard_index += 1
        remaining = self.count - self._position
        length = min(self.shard_samples, remaining)
        relative = Path("tensors") / f"{self.split}-{self._shard_index:05d}.npy"
        target = self.destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        self._array = np.lib.format.open_memmap(
            target, mode="w+", dtype=self.dtype, shape=(length, *self.shape)
        )
        self.paths.append(str(relative))

    def write(self, image: np.ndarray) -> tuple[int, int]:
        if self._array is None or self._position % self.shard_samples == 0:
            self._open_shard()
        offset = self._position % self.shard_samples
        self._array[offset] = image
        self._position += 1
        self.offsets.append((self._shard_index, offset))
        return self.offsets[-1]

    def close(self) -> None:
        if self._array is not None:
            self._array.flush()
            self._array = None


def _build_cache(cache_dir: Path, config: dict[str, Any], indices: dict[str, Any], cache_id: str, preprocessing_fingerprint: str) -> None:
    settings = _settings(config)
    dtype_name = str(settings.get("dtype", "float32")).lower()
    dtype = np.dtype(dtype_name)
    shape = tuple(int(value) for value in config["data"]["input_shape"])
    shard_samples = int(settings.get("shard_samples", 1024))
    writers = {
        split: _ShardWriter(cache_dir, split, len(index.refs), shape, dtype, shard_samples)
        for split, index in indices.items()
    }
    refs_by_id = {ref.item_id: (split, ref) for split, index in indices.items() for ref in index.refs}
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    source = Path(config["paths"]["input_path"])
    try:
        with sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as connection:
            cursor = connection.execute(
                """
                SELECT ci.item_id, a.payload, a.encoding, a.shape_json, di.metadata_json
                FROM classification_items ci
                JOIN dataset_items di ON di.item_id = ci.item_id
                JOIN assets a ON a.asset_id = ci.image_asset_id
                ORDER BY ci.item_id
                """
            )
            for item_id, payload, encoding, dimensions, metadata_json in cursor:
                found = refs_by_id.get(str(item_id))
                if found is None:
                    continue
                split, ref = found
                metadata = json.loads(metadata_json) if metadata_json else {}
                image = prepare_dataset_classification_input(
                    decode_blob(payload, encoding, dimensions), shape, config, metadata
                )
                shard, offset = writers[split].write(np.asarray(image, dtype=dtype))
                records.append(
                    {
                        "item_id": str(item_id),
                        "split": split,
                        "label": ref.target,
                        "shard": shard,
                        "offset": offset,
                        "original_shape": ref.input_dimensions,
                    }
                )
    finally:
        for writer in writers.values():
            writer.close()
    missing = set(refs_by_id) - {row["item_id"] for row in records}
    if missing:
        raise ValueError(f"Prepared-input cache source is missing {len(missing)} indexed dataset items")
    records_file, records_format = _write_records(cache_dir, records)
    split_fingerprint = _split_fingerprint(config)
    bytes_written = sum(path.stat().st_size for path in (cache_dir / "tensors").glob("*.npy"))
    manifest = {
        "schema_name": "oracle_builder_prepared_input_cache",
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "cache_id": cache_id,
        "created_at": time.time(),
        "dataset": {
            "dataset_id": config.get("dataset", {}).get("dataset_id"),
            "fingerprint_sha256": config.get("dataset", {}).get("fingerprint_sha256"),
        },
        "split_manifest_fingerprint_sha256": split_fingerprint,
        "preprocessing_fingerprint_sha256": preprocessing_fingerprint,
        "input_shape": list(shape),
        "dtype": dtype_name,
        "bytes": bytes_written,
        "records_file": records_file,
        "records_format": records_format,
        "build_seconds": time.perf_counter() - started,
        "splits": {
            split: {
                "count": len(index.refs),
                "item_ids": [ref.item_id for ref in index.refs],
                "item_ids_fingerprint_sha256": _item_ids_fingerprint([ref.item_id for ref in index.refs]),
                "shards": writers[split].paths,
                "shard_offsets": writers[split].offsets,
            }
            for split, index in indices.items()
        },
    }
    _atomic_write_json(cache_dir / "manifest.json", manifest)


def materialized_classification_source(config: dict[str, Any], indices: dict[str, Any]) -> MaterializedClassificationSource | None:
    """Return a shared prepared-input source, building it once when requested."""
    settings = _settings(config)
    mode = str(settings.get("mode", "off")).lower()
    if mode == "off":
        return None
    if mode not in _SUPPORTED_MODES:
        raise ValueError("data.materialization.mode must be off, run, or shared")
    if config.get("dataset", {}).get("lifecycle") != "frozen":
        raise ValueError("data materialization is permitted only for frozen datasets")
    dtype = str(settings.get("dtype", "float32")).lower()
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError("data.materialization.dtype must be float16 or float32")
    if str(settings.get("format", "npy_shards")).lower() != "npy_shards":
        raise ValueError("data.materialization.format currently supports only npy_shards")
    shard_samples = int(settings.get("shard_samples", 1024))
    if shard_samples < 1:
        raise ValueError("data.materialization.shard_samples must be positive")
    cache_dir, cache_id, preprocessing_fingerprint = _cache_path(config, settings)
    existing = _load_cache(cache_dir, config, indices, cache_id)
    if existing is not None:
        return existing
    if not bool(settings.get("build_if_missing", True)):
        return None
    expected_bytes = sum(len(index.refs) for index in indices.values()) * int(np.prod(config["data"]["input_shape"])) * np.dtype(dtype).itemsize
    limit_gib = float(settings.get("max_cache_gib", 0.0))
    if limit_gib > 0 and expected_bytes > limit_gib * 1024**3:
        raise ValueError(
            "Prepared-input cache would exceed data.materialization.max_cache_gib "
            f"({expected_bytes / 1024**3:.2f} GiB requested)"
        )
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    with _DirectoryLock(cache_dir, float(settings.get("wait_seconds", 600))):
        existing = _load_cache(cache_dir, config, indices, cache_id)
        if existing is not None:
            return existing
        if cache_dir.exists():
            raise RuntimeError(
                f"Prepared-input cache path exists but does not match its requested immutable identity: {cache_dir}"
            )
        temporary = Path(tempfile.mkdtemp(prefix=f".{cache_dir.name}.", dir=cache_dir.parent))
        try:
            _build_cache(temporary, config, indices, cache_id, preprocessing_fingerprint)
            os.replace(temporary, cache_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    result = _load_cache(cache_dir, config, indices, cache_id)
    if result is None:  # pragma: no cover - protects against interrupted filesystem writes
        raise RuntimeError(f"Prepared-input cache {cache_dir} was not readable after build")
    result.report["status"] = "created"
    return result
