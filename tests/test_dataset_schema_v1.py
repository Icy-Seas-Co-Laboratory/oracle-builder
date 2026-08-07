from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from oracle_builder.data.sqlite_dataset import (
    create_synthetic_classification,
    create_synthetic_segmentation,
)
from oracle_builder.datasets.lifecycle import save_checkpoint, thaw_database
from oracle_builder.datasets.schema import (
    DatasetSchemaError,
    dataset_fingerprint,
    initialize_database,
    read_dataset_info,
    validate_database,
)
from oracle_builder.datasets.transfer import export_dataset, import_dataset_export


def test_one_sqlite_file_has_one_typed_dataset_identity():
    connection = sqlite3.connect(":memory:")
    first = initialize_database(
        connection,
        "classification",
        name="WHOI example",
        version="1",
    )

    second = initialize_database(connection, "classification")

    assert first["dataset_id"] == second["dataset_id"]
    assert second["dataset_type"] == "classification"
    assert second["schema_version"] == "1.2.0"
    assert "split" not in {
        row[1] for row in connection.execute("PRAGMA table_info(dataset_items)")
    }
    assert connection.execute("SELECT count(*) FROM dataset").fetchone()[0] == 1
    with pytest.raises(DatasetSchemaError, match="not 'mask_refinement'"):
        initialize_database(connection, "mask_refinement")


def test_legacy_generic_database_requires_explicit_migration():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE samples (uuid TEXT PRIMARY KEY)")

    with pytest.raises(DatasetSchemaError, match="Legacy generic samples-table"):
        initialize_database(connection, "classification")


def test_validation_reports_missing_use_case_table_instead_of_crashing():
    connection = sqlite3.connect(":memory:")
    initialize_database(connection, "classification")
    connection.execute("DROP TABLE classification_annotations")

    report = validate_database(connection)

    assert not report["valid"]
    assert any(
        "classification_annotations" in error for error in report["errors"]
    )


def test_checkpoint_is_frozen_and_source_remains_working(tmp_path: Path):
    source = tmp_path / "working.sqlite"
    create_synthetic_classification(source, n=4, shape=(8, 8, 1), classes=2)
    with sqlite3.connect(source) as connection:
        source_fingerprint = dataset_fingerprint(connection)
        source_revision_id = read_dataset_info(connection)["revision_id"]

    result = save_checkpoint(source, actor="test")
    checkpoint = Path(result["checkpoint"])

    assert checkpoint.name.startswith("working.checkpoint-")
    assert checkpoint.suffix == ".sqlite"
    with sqlite3.connect(source) as connection:
        assert read_dataset_info(connection)["lifecycle"] == "working"
    with sqlite3.connect(checkpoint) as connection:
        checkpoint_info = read_dataset_info(connection)
        assert checkpoint_info["lifecycle"] == "frozen"
        assert checkpoint_info["revision_id"] != source_revision_id
        assert checkpoint_info["parent_revision_id"] == source_revision_id
        assert dataset_fingerprint(connection) == source_fingerprint
        with pytest.raises(sqlite3.IntegrityError, match="dataset is frozen"):
            connection.execute(
                "UPDATE dataset_items SET sample_weight = 2 WHERE item_id = "
                "(SELECT item_id FROM dataset_items LIMIT 1)"
            )


def test_explicit_thaw_allows_edits_again(tmp_path: Path):
    source = tmp_path / "working.sqlite"
    create_synthetic_classification(source, n=4, shape=(8, 8, 1), classes=2)
    checkpoint = Path(save_checkpoint(source)["checkpoint"])
    with sqlite3.connect(checkpoint) as connection:
        frozen_revision_id = read_dataset_info(connection)["revision_id"]

    result = thaw_database(checkpoint, actor="test", reason="continue curation")

    assert result["lifecycle"] == "working"
    assert result["revision_id"] != frozen_revision_id
    assert result["parent_revision_id"] == frozen_revision_id
    with sqlite3.connect(checkpoint) as connection:
        connection.execute(
            "UPDATE dataset_items SET sample_weight = 2 WHERE item_id = "
            "(SELECT item_id FROM dataset_items LIMIT 1)"
        )
        connection.commit()


def test_schema_1_0_split_column_is_removed_automatically(tmp_path: Path):
    database = tmp_path / "old-v1.sqlite"
    create_synthetic_classification(database, n=4, shape=(8, 8, 1), classes=2)
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE dataset_items ADD COLUMN split TEXT")
        connection.execute(
            "UPDATE dataset_items SET split = 'train' WHERE item_id IN "
            "(SELECT item_id FROM dataset_items ORDER BY item_id LIMIT 2)"
        )
        connection.execute(
            """
            UPDATE dataset_items
            SET metadata_json = '{"import_split_hint":"train","synthetic":true}'
            WHERE item_id = (SELECT item_id FROM dataset_items ORDER BY item_id LIMIT 1)
            """
        )
        connection.execute(
            "UPDATE ob_schema SET schema_version = '1.0.0' WHERE singleton = 1"
        )
        connection.execute(
            "UPDATE dataset SET lifecycle = 'frozen' WHERE singleton = 1"
        )
        connection.commit()

    with sqlite3.connect(database) as connection:
        info = read_dataset_info(connection)
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(dataset_items)")
        }
        event = json.loads(
            connection.execute(
                """
                SELECT details_json FROM dataset_events
                WHERE event_type = 'schema.migrated'
                ORDER BY created_at DESC LIMIT 1
                """
            ).fetchone()[0]
        )
        report = validate_database(connection)
        metadata_values = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT metadata_json FROM dataset_items"
            )
        ]

    assert info["schema_version"] == "1.2.0"
    assert info["lifecycle"] == "frozen"
    assert "split" not in columns
    assert event["removed_non_null_assignments"] == 2
    assert event["removed_metadata_split_hints"] == 1
    assert all("import_split_hint" not in value for value in metadata_values)
    assert report["valid"]


def test_folder_exchange_round_trip_is_semantically_exact(tmp_path: Path):
    source = tmp_path / "classification.sqlite"
    export_root = tmp_path / "classification-export"
    restored = tmp_path / "classification-restored.sqlite"
    create_synthetic_classification(source, n=7, shape=(8, 8, 1), classes=3)
    with sqlite3.connect(source) as connection:
        expected = dataset_fingerprint(connection)

    export_dataset(source, export_root)
    import_dataset_export(export_root, restored)

    assert (export_root / "manifest.json").exists()
    assert (export_root / "metadata.toml").exists()
    assert (export_root / "checksums.sha256").exists()
    assert list((export_root / "images").rglob("*.npy"))
    manifest = json.loads((export_root / "manifest.json").read_text())
    assert manifest["fingerprint_sha256"] == expected
    with sqlite3.connect(restored) as connection:
        assert dataset_fingerprint(connection) == expected
        assert validate_database(connection)["valid"]


def test_mask_folder_exchange_preserves_annotation_history(tmp_path: Path):
    source = tmp_path / "masks.sqlite"
    export_root = tmp_path / "masks-export"
    restored = tmp_path / "masks-restored.sqlite"
    create_synthetic_segmentation(source, n=3, shape=(8, 8, 1))
    with sqlite3.connect(source) as connection:
        expected = dataset_fingerprint(connection)
        annotation_count = connection.execute(
            "SELECT count(*) FROM mask_annotations"
        ).fetchone()[0]

    export_dataset(source, export_root)
    import_dataset_export(export_root, restored)

    with sqlite3.connect(restored) as connection:
        assert dataset_fingerprint(connection) == expected
        assert connection.execute(
            "SELECT count(*) FROM mask_annotations"
        ).fetchone()[0] == annotation_count
        assert validate_database(connection)["valid"]
