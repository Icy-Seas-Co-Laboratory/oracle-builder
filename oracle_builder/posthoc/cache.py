"""Portable, provenance-aware embedding caches.

Embeddings stay in a ``.npy`` file so callers can use mmap mode and process a
large run in batches.  Sample-level provenance is stored as Parquet when the
installed pandas engine supports it, otherwise JSON Lines.  The manifest is a
small, versioned contract that makes the fallback explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

CACHE_FORMAT = "oracle_builder.embedding_cache"
CACHE_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _records_list(records: Any, count: int) -> list[dict[str, Any]] | None:
    if records is None:
        return None
    if hasattr(records, "to_dict"):  # pandas without requiring pandas at import time
        rows = records.to_dict(orient="records")
    else:
        rows = list(records)
    if len(rows) != count:
        raise ValueError(f"records has {len(rows)} rows but embeddings has {count}")
    if not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("records must be mappings or a pandas DataFrame")
    return [dict(row) for row in rows]


@dataclass(frozen=True)
class EmbeddingCache:
    """A validated cache directory. ``embeddings`` opens as a read-only memmap."""

    path: Path
    manifest: Mapping[str, Any]

    @property
    def embeddings(self) -> np.ndarray:
        return np.load(self.path / self.manifest["embeddings_file"], mmap_mode="r")

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.manifest["shape"])  # type: ignore[return-value]

    def records(self) -> list[dict[str, Any]] | None:
        filename = self.manifest.get("records_file")
        if not filename:
            return None
        path = self.path / filename
        if self.manifest.get("records_format") == "parquet":
            try:
                import pandas as pd
                return pd.read_parquet(path).to_dict(orient="records")
            except Exception as exc:  # a cache should explain rather than silently corrupt
                raise RuntimeError("reading this cache's Parquet provenance requires a pandas Parquet engine") from exc
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Stream JSONL provenance; Parquet falls back to its dataframe reader."""
        filename = self.manifest.get("records_file")
        if not filename:
            return
        path = self.path / filename
        if self.manifest.get("records_format") == "jsonl":
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
            return
        records = self.records()
        yield from records or ()

    def iter_batches(self, batch_size: int = 4096) -> Iterator[tuple[slice, np.ndarray]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        values = self.embeddings
        for start in range(0, len(values), batch_size):
            yield slice(start, min(start + batch_size, len(values))), values[start : start + batch_size]


def export_embedding_cache(
    destination: str | Path,
    embeddings: np.ndarray,
    *,
    records: Any = None,
    representation: str = "fused_embedding",
    provenance: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    prefer_parquet: bool = True,
) -> EmbeddingCache:
    """Write a cache without duplicating the embedding matrix in a table.

    ``provenance`` should identify the producing model/data split/config hashes.
    It is intentionally unconstrained to permit future run formats while its
    exact values remain inspectable in the manifest.
    """
    values = np.asarray(embeddings)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("embeddings must be a non-empty rank-2 array")
    if values.dtype.kind not in "fiu":
        raise TypeError("embeddings must have a numeric dtype")
    rows = _records_list(records, values.shape[0])
    target = Path(destination)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"embedding cache already exists: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    embedding_file = "embeddings.npy"
    # np.save does not create a second in-memory copy for a normal ndarray.
    np.save(target / embedding_file, values, allow_pickle=False)
    manifest: dict[str, Any] = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "representation": representation,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "embeddings_file": embedding_file,
        "provenance": dict(provenance or {}),
    }
    if rows is not None:
        if prefer_parquet:
            try:
                import pandas as pd
                pd.DataFrame(rows).to_parquet(target / "records.parquet", index=False)
                manifest.update(records_file="records.parquet", records_format="parquet")
            except Exception:
                # Parquet is an optional acceleration, never a cache requirement.
                pass
        if "records_file" not in manifest:
            with (target / "records.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")
            manifest.update(records_file="records.jsonl", records_format="jsonl")
    with (target / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    return open_embedding_cache(target)


def export_embedding_cache_stream(
    destination: str | Path,
    batches: Iterable[tuple[np.ndarray, Sequence[int], Sequence[Mapping[str, Any]]]],
    *,
    count: int,
    representation: str = "fused_embedding",
    provenance: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> EmbeddingCache:
    """Write a cache incrementally, using a writable NPY memmap.

    ``batches`` yields representation rows, their stable output positions, and
    matching provenance rows.  This is the preferred path for training runs:
    no full-dataset embedding matrix or metadata dataframe is constructed in
    RAM.
    """
    if count < 1:
        raise ValueError("count must be positive")
    iterator = iter(batches)
    try:
        first_values, first_positions, first_records = next(iterator)
    except StopIteration as exc:
        raise ValueError("embedding stream produced no batches") from exc
    first_values = np.asarray(first_values)
    if first_values.ndim != 2 or first_values.shape[1] < 1:
        raise ValueError("streamed embeddings must be rank-2 with a positive dimension")
    if len(first_values) != len(first_positions) or len(first_values) != len(first_records):
        raise ValueError("stream batch embeddings, positions, and records must have equal length")
    target = Path(destination)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"embedding cache already exists: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    embedding_file = "embeddings.npy"
    mmap = np.lib.format.open_memmap(
        target / embedding_file, mode="w+", dtype=first_values.dtype,
        shape=(int(count), int(first_values.shape[1])),
    )
    expected_position = 0
    records_handle = (target / "records.jsonl").open("w", encoding="utf-8")

    def write(values: np.ndarray, positions: Sequence[int], records: Sequence[Mapping[str, Any]]) -> None:
        nonlocal expected_position
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != mmap.shape[1]:
            raise ValueError("streamed embedding dimensions are inconsistent")
        if len(values) != len(positions) or len(values) != len(records):
            raise ValueError("stream batch embeddings, positions, and records must have equal length")
        for value, position, record in zip(values, positions, records, strict=True):
            position = int(position)
            if position != expected_position:
                raise ValueError("stream positions must be ordered and contiguous")
            mmap[position] = value
            records_handle.write(json.dumps(dict(record), default=_json_default, sort_keys=True) + "\n")
            expected_position += 1

    write(first_values, first_positions, first_records)
    try:
        for values, positions, records in iterator:
            write(values, positions, records)
    finally:
        records_handle.close()
    mmap.flush()
    if expected_position != count:
        raise ValueError("stream did not produce every declared cache position")
    manifest = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "representation": representation,
        "shape": [int(count), int(mmap.shape[1])],
        "dtype": str(mmap.dtype),
        "embeddings_file": embedding_file,
        "records_file": "records.jsonl",
        "records_format": "jsonl",
        "provenance": dict(provenance or {}),
    }
    with (target / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    return open_embedding_cache(target)


def open_embedding_cache(path: str | Path) -> EmbeddingCache:
    """Open and validate a cache's stable on-disk contract."""
    root = Path(path)
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != CACHE_FORMAT or manifest.get("version") != CACHE_VERSION:
        raise ValueError("unsupported embedding cache format/version")
    required = {"embeddings_file", "shape", "dtype", "representation", "provenance"}
    if not required.issubset(manifest):
        raise ValueError("embedding cache manifest is incomplete")
    data = root / manifest["embeddings_file"]
    if not data.is_file():
        raise FileNotFoundError(data)
    shape = tuple(manifest["shape"])
    if len(shape) != 2 or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
        raise ValueError("embedding cache shape must contain two positive integers")
    loaded = np.load(data, mmap_mode="r")
    if tuple(loaded.shape) != shape or str(loaded.dtype) != manifest["dtype"]:
        raise ValueError("embedding file does not match manifest")
    return EmbeddingCache(root, manifest)
