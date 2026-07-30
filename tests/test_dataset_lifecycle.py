from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest
import model_training

from oracle_builder.data.sqlite_dataset import create_synthetic_classification
from oracle_builder.datasets.lifecycle import checkpoint_path, save_checkpoint, thaw_database
from oracle_builder.datasets.schema import (
    dataset_fingerprint,
    read_dataset_info,
    set_dataset_lifecycle,
    validate_database,
)


def test_checkpoint_path_is_timestamped():
    result = checkpoint_path(
        "library.sqlite",
        datetime(2026, 7, 30, 20, 15, 3, 125000, tzinfo=timezone.utc),
    )
    assert result.name == "library.checkpoint-20260730T201503.125000Z.sqlite"


def test_checkpoint_freezes_copy_but_leaves_working_database_editable(tmp_path):
    working = tmp_path / "library.sqlite"
    frozen = tmp_path / "library-release.sqlite"
    create_synthetic_classification(working, n=4, shape=(8, 8, 1), classes=2)

    result = save_checkpoint(working, frozen, actor="test")

    assert result["checkpoint"] == str(frozen.resolve())
    with sqlite3.connect(working) as connection:
        working_info = read_dataset_info(connection)
        assert working_info["lifecycle"] == "working"
        connection.execute(
            "UPDATE dataset_items SET sample_weight = 2 WHERE rowid = 1"
        )
    with sqlite3.connect(frozen) as connection:
        frozen_info = read_dataset_info(connection)
        assert frozen_info["lifecycle"] == "frozen"
        assert frozen_info["revision_id"] != working_info["revision_id"]
        assert frozen_info["parent_revision_id"] == working_info["revision_id"]
        assert validate_database(connection)["valid"]
        with pytest.raises(sqlite3.IntegrityError, match="dataset is frozen"):
            connection.execute(
                "UPDATE dataset_items SET sample_weight = 2 WHERE rowid = 1"
            )


def test_explicit_thaw_restores_editability_and_records_identity(tmp_path):
    database = tmp_path / "library.sqlite"
    create_synthetic_classification(database, n=3, shape=(8, 8, 1), classes=2)
    with sqlite3.connect(database) as connection:
        before_id = read_dataset_info(connection)["dataset_id"]
        frozen_revision_id = read_dataset_info(connection)["revision_id"]
        before_fingerprint = dataset_fingerprint(connection)
        set_dataset_lifecycle(connection, "frozen", actor="test")
        connection.commit()

    result = thaw_database(database, actor="test", reason="continue curation")

    assert result["dataset_id"] == before_id
    assert result["revision_id"] != frozen_revision_id
    assert result["parent_revision_id"] == frozen_revision_id
    assert result["fingerprint"] == before_fingerprint
    with sqlite3.connect(database) as connection:
        assert read_dataset_info(connection)["lifecycle"] == "working"
        connection.execute(
            "UPDATE dataset_items SET sample_weight = 3 WHERE rowid = 1"
        )


def test_checkpoint_refuses_to_overwrite_an_existing_file(tmp_path):
    source = tmp_path / "source.sqlite"
    destination = tmp_path / "checkpoint.sqlite"
    create_synthetic_classification(source, n=2, shape=(8, 8, 1), classes=2)
    destination.write_bytes(b"keep me")

    with pytest.raises(FileExistsError):
        save_checkpoint(source, destination)

    assert destination.read_bytes() == b"keep me"


def test_training_rejects_working_dataset_before_creating_run_directory(
    monkeypatch, tmp_path
):
    database = tmp_path / "working.sqlite"
    config = tmp_path / "classification.toml"
    runs = tmp_path / "runs"
    create_synthetic_classification(database, n=4, shape=(8, 8, 1), classes=2)
    config.write_text(
        """
[run]
task = "classification"
model = "resnet"

[data]
input_shape = [8, 8, 1]

[training]
loss = "sparse_categorical_crossentropy"
"""
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "model_training.py",
            "--config",
            str(config),
            "--input",
            str(database),
            "--output",
            "should-not-exist",
            "--runs-dir",
            str(runs),
        ],
    )

    with pytest.raises(ValueError, match="requires a frozen dataset checkpoint"):
        model_training.main()
    assert not (runs / "should-not-exist").exists()
