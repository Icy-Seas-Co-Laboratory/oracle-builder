from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from oracle_builder.config import DEFAULT_CONFIG, deep_merge
from oracle_builder.data.sqlite_dataset import create_synthetic_classification
from oracle_builder.data.sqlite_stream import make_streaming_classification_bundle
from oracle_builder.datasets.schema import dataset_fingerprint, read_dataset_info, set_dataset_lifecycle


def _config(database, cache_root):
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "run": {"task": "classification", "model": "simple_cnn", "seed": 7},
            "data": {
                "input_shape": [12, 12, 1],
                "num_classes": 3,
                "batch_size": 2,
                "shuffle_buffer": 4,
                "validation_split": 0.2,
                "test_split": 0.1,
                "streaming": {
                    "enabled": True,
                    "reader_workers": 1,
                    "prefetch_batches": 1,
                    "deterministic": True,
                    "sqlite_cache_kib": 1024,
                },
                "materialization": {
                    "mode": "shared",
                    "root": str(cache_root),
                    "format": "npy_shards",
                    "dtype": "float32",
                    "shard_samples": 3,
                    "build_if_missing": True,
                    "wait_seconds": 2,
                    "max_cache_gib": 1.0,
                },
            },
            "preprocessing": {
                "resize_mode": "fit_pad",
                "normalization": "dtype",
                "rescale": True,
                "invert": False,
                "pad_value": 0.0,
                "interpolation": "bilinear",
                "channel_mode": "grayscale",
            },
            "augmentation": {"enabled": False, "repeats_per_epoch": 1},
        },
    )
    with sqlite3.connect(database) as connection:
        info = read_dataset_info(connection)
        fingerprint = dataset_fingerprint(connection)
        item_ids = [row[0] for row in connection.execute("SELECT item_id FROM dataset_items ORDER BY item_id")]
    config["dataset"] = {
        "dataset_id": info["dataset_id"],
        "fingerprint_sha256": fingerprint,
        "lifecycle": "frozen",
    }
    config["paths"] = {"input_path": str(database), "run_dir": str(cache_root / "run")}
    config["_split_manifest"] = {
        "fingerprint_sha256": "test-split-v1",
        "assignments": {
            item_id: "test" if index == 0 else "validation" if index < 3 else "train"
            for index, item_id in enumerate(item_ids)
        },
    }
    return config


def test_shared_materialization_builds_then_reuses_prepared_tensor_shards(tmp_path):
    database = tmp_path / "classification.sqlite"
    create_synthetic_classification(database, n=9, shape=(12, 12, 1), classes=3)
    with sqlite3.connect(database) as connection:
        set_dataset_lifecycle(connection, "frozen")
        connection.commit()
    config = _config(database, tmp_path / "cache")

    created = make_streaming_classification_bundle(database, config)

    assert created.materialization is not None
    assert created.materialization["status"] == "created"
    cache_path = created.materialization["path"]
    manifest = json.loads((Path(cache_path) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["records_format"] in {"parquet", "jsonl"}
    assert Path(cache_path).exists()
    assert manifest["splits"]["train"]["count"] == len(created.indices["train"])
    assert all((Path(cache_path) / shard).exists() for shard in manifest["splits"]["train"]["shards"])

    first_batch = next(iter(created.datasets["train"]))
    reused = make_streaming_classification_bundle(database, config)

    assert reused.materialization is not None
    assert reused.materialization["status"] == "reused"
    second_batch = next(iter(reused.datasets["train"]))
    np.testing.assert_allclose(first_batch[0], second_batch[0])
    assert reused.source.statistics()["reads"] > 0


def test_materialization_requires_a_frozen_resolved_dataset(tmp_path):
    database = tmp_path / "classification.sqlite"
    create_synthetic_classification(database, n=4, shape=(12, 12, 1), classes=2)
    config = _config(database, tmp_path / "cache")
    config["dataset"]["lifecycle"] = "working"

    with pytest.raises(ValueError, match="only for frozen datasets"):
        make_streaming_classification_bundle(database, config)


def test_materialization_manifest_is_json_serializable_and_split_owned(tmp_path):
    database = tmp_path / "classification.sqlite"
    create_synthetic_classification(database, n=6, shape=(12, 12, 1), classes=2)
    with sqlite3.connect(database) as connection:
        set_dataset_lifecycle(connection, "frozen")
        connection.commit()
    config = _config(database, tmp_path / "cache")
    bundle = make_streaming_classification_bundle(database, config)

    manifest_path = Path(bundle.materialization["path"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["split_manifest_fingerprint_sha256"] == "test-split-v1"
    assert manifest["dataset"]["fingerprint_sha256"] == config["dataset"]["fingerprint_sha256"]
