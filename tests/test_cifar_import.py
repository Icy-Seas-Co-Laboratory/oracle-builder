from __future__ import annotations

import json
import sqlite3
from argparse import Namespace

import numpy as np

from oracle_builder.data import cifar_import


def test_cifar_import_writes_lossless_arrays_labels_and_source_partitions(tmp_path, monkeypatch):
    train_images = np.asarray([np.zeros((32, 32, 3), dtype="uint8")])
    test_images = np.asarray([np.full((32, 32, 3), 255, dtype="uint8")])
    monkeypatch.setattr(
        cifar_import,
        "_load_cifar",
        lambda _: ((train_images, np.asarray([[0]])), (test_images, np.asarray([[1]])), ("airplane", "automobile")),
    )
    output = tmp_path / "cifar.sqlite"

    report = cifar_import.import_cifar(
        Namespace(dataset="cifar10", output=str(output), dataset_id=None, dry_run=False)
    )

    assert report["counts"] == {"train": 1, "test": 1}
    with sqlite3.connect(output) as connection:
        rows = connection.execute(
            "SELECT source_key, metadata_json FROM dataset_items ORDER BY source_key"
        ).fetchall()
        labels = connection.execute(
            "SELECT class_index, name FROM classification_labels ORDER BY class_index"
        ).fetchall()
        stored = connection.execute("SELECT encoding, shape_json, dtype FROM assets ORDER BY original_filename").fetchall()

    assert [row[0] for row in rows] == ["cifar10/test/00000", "cifar10/train/00000"]
    assert [json.loads(row[1])["source_partition"] for row in rows] == ["test", "train"]
    assert labels == [(0, "airplane"), (1, "automobile")]
    assert stored == [("npy", "[32, 32, 3]", "uint8"), ("npy", "[32, 32, 3]", "uint8")]
