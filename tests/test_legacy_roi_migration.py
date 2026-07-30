from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from oracle_builder.data.decoders import encode_npy
from oracle_builder.data.sqlite_dataset import load_arrays, read_rows
from oracle_builder.artifacts.splits import split_manifest_matches_dataset
from oracle_builder.config import resolve_config
from oracle_builder.datasets.legacy_roi import (
    inspect_dataset_kind,
    migrate_legacy_roi_database,
)
from oracle_builder.datasets.schema import read_dataset_info, validate_database
from oracle_builder.masking.sqlite_io import encode_mask, load_sample, open_database
from oracle_builder.masking.unet_dataset import validate_unet_dataset


LEGACY_SCHEMA = """
CREATE TABLE samples (
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
    metadata_json TEXT,
    input_aux_blob BLOB,
    input_aux_blob_encoding TEXT,
    input_aux_blob_dimensions TEXT
);
CREATE TABLE mask_annotations (
    annotation_id TEXT PRIMARY KEY,
    sample_uuid TEXT NOT NULL,
    created_at TEXT NOT NULL,
    annotator TEXT,
    mask_blob BLOB NOT NULL,
    mask_blob_encoding TEXT NOT NULL,
    mask_blob_dimensions TEXT NOT NULL,
    method TEXT,
    parameters_json TEXT,
    validation_json TEXT,
    accepted INTEGER DEFAULT 1,
    notes TEXT
);
CREATE TABLE predictions (
    uuid TEXT PRIMARY KEY,
    score REAL
);
"""


def _mask_blob(mask: np.ndarray) -> tuple[bytes, str, str]:
    return encode_mask(mask, "png")


def create_legacy_database(path: Path, *, orphan: bool = False) -> dict:
    image = np.arange(30, dtype="uint8").reshape(5, 6)
    candidate = np.zeros((5, 6), dtype="uint8")
    candidate[1:4, 2:5] = 255
    training_input = np.stack([image, candidate], axis=-1)
    first = np.zeros((5, 6), dtype="uint8")
    first[1:3, 1:3] = 1
    second = np.zeros((5, 6), dtype="uint8")
    second[2:5, 3:5] = 1
    output_only = np.zeros((5, 6), dtype="uint8")
    output_only[0:2, 0:2] = 1
    first_blob = _mask_blob(first)
    second_blob = _mask_blob(second)
    output_blob = _mask_blob(output_only)
    with sqlite3.connect(path) as connection:
        connection.executescript(LEGACY_SCHEMA)
        connection.execute(
            """
            INSERT INTO samples VALUES (
                'roi-1', 'test', ?, 'npy', ?, ?, 'png', ?, NULL, 2.5, ?,
                ?, 'png', ?
            )
            """,
            (
                encode_npy(training_input),
                json.dumps(list(training_input.shape)),
                second_blob[0],
                second_blob[2],
                json.dumps({"pelagia_detection_id": "roi-1", "kept": True}),
                _mask_blob(candidate)[0],
                _mask_blob(candidate)[2],
            ),
        )
        connection.execute(
            """
            INSERT INTO samples VALUES (
                'roi-output-only', 'train', ?, 'npy', ?, ?, 'png', ?,
                NULL, NULL, '{}', NULL, NULL, NULL
            )
            """,
            (
                encode_npy(training_input),
                json.dumps(list(training_input.shape)),
                output_blob[0],
                output_blob[2],
            ),
        )
        connection.execute(
            """
            INSERT INTO samples VALUES (
                'roi-unvalidated', NULL, ?, 'npy', ?, NULL, NULL, NULL,
                NULL, NULL, '{}', NULL, NULL, NULL
            )
            """,
            (encode_npy(image), json.dumps(list(image.shape))),
        )
        for annotation_id, created_at, blob in (
            ("annotation-1", "2026-01-01T00:00:00+00:00", first_blob),
            ("annotation-2", "2026-01-02T00:00:00+00:00", second_blob),
        ):
            connection.execute(
                """
                INSERT INTO mask_annotations VALUES (
                    ?, 'roi-1', ?, 'reviewer', ?, ?, ?, 'manual', '{}', '{}',
                    1, NULL
                )
                """,
                (annotation_id, created_at, blob[0], blob[1], blob[2]),
            )
        if orphan:
            connection.execute(
                """
                INSERT INTO mask_annotations VALUES (
                    'orphan', 'missing-roi', '2026-01-03T00:00:00+00:00',
                    NULL, ?, ?, ?, 'manual', '{}', '{}', 1, NULL
                )
                """,
                (first_blob[0], first_blob[1], first_blob[2]),
            )
        connection.execute("INSERT INTO predictions VALUES ('roi-1', 0.75)")
        connection.commit()
    return {
        "image": image,
        "candidate": candidate > 0,
        "current_mask": second > 0,
        "output_only": output_only > 0,
    }


def test_legacy_roi_migration_preserves_assets_history_and_additive_tables(tmp_path):
    database = tmp_path / "legacy.sqlite"
    expected = create_legacy_database(database)

    result = migrate_legacy_roi_database(database)

    assert inspect_dataset_kind(database) == "mask_refinement"
    assert Path(result["backup"]).exists()
    assert Path(result["report"]).exists()
    assert result["sample_count"] == 3
    assert result["legacy_annotation_count"] == 2
    assert result["synthetic_annotation_count"] == 1
    assert result["dropped_legacy_split_assignments"] == 2
    assert result["copied_additive_tables"] == ["predictions"]
    with sqlite3.connect(result["backup"]) as backup:
        assert backup.execute("SELECT count(*) FROM samples").fetchone()[0] == 3
    with sqlite3.connect(database) as connection:
        assert validate_database(connection)["valid"]
        assert read_dataset_info(connection)["dataset_type"] == "mask_refinement"
        assert "split" not in {
            row[1]
            for row in connection.execute("PRAGMA table_info(dataset_items)")
        }
        assert connection.execute(
            "SELECT count(*) FROM mask_annotations"
        ).fetchone()[0] == 3
        history = connection.execute(
            """
            SELECT annotation_id, is_current, parent_annotation_id
            FROM mask_annotations WHERE item_id = 'roi-1'
            ORDER BY created_at
            """
        ).fetchall()
        assert history == [
            ("annotation-1", 0, None),
            ("annotation-2", 1, "annotation-1"),
        ]
        assert connection.execute(
            "SELECT score FROM predictions WHERE uuid = 'roi-1'"
        ).fetchone()[0] == 0.75
        first = load_sample(connection, "roi-1")
        output_only = load_sample(connection, "roi-output-only")
    np.testing.assert_array_equal(first["image"], expected["image"])
    np.testing.assert_array_equal(first["candidate_mask"], expected["candidate"])
    np.testing.assert_array_equal(first["mask"], expected["current_mask"])
    np.testing.assert_array_equal(output_only["mask"], expected["output_only"])
    analysis = validate_unet_dataset(database)
    assert analysis["item_count"] == 3
    assert analysis["annotated_item_count"] == 2
    assert analysis["missing_current_mask_count"] == 1
    assert analysis["annotation_count"] == 3
    assert analysis["historical_annotation_count"] == 1
    assert analysis["candidate_mask_count"] == 2
    assert "split_counts" not in analysis


def test_open_database_automatically_migrates_recognized_legacy_roi_database(tmp_path):
    database = tmp_path / "legacy.sqlite"
    create_legacy_database(database)

    with pytest.warns(RuntimeWarning, match="Migrated legacy ROI database"):
        with open_database(database, create=False) as connection:
            assert read_dataset_info(connection)["schema_version"] == "1.1.0"

    assert list(tmp_path.glob("legacy.pre-v1-*.sqlite"))
    assert list(tmp_path.glob("legacy.migration-*.json"))


def test_segmentation_loader_automatically_migrates_before_reading(tmp_path):
    database = tmp_path / "legacy.sqlite"
    create_legacy_database(database)
    config = {
        "run": {"task": "segmentation", "seed": 123},
        "data": {
            "input_shape": [5, 6, 2],
            "output_shape": [5, 6, 1],
            "validation_split": 0.0,
            "test_split": 0.0,
        },
        "training": {"segmentation_target": "validated_mask"},
        "preprocessing": {"rescale": True},
        "tiling": {"enabled": False},
    }

    with pytest.warns(RuntimeWarning, match="Migrated legacy ROI database"):
        inputs, targets, records = load_arrays(database, config, split=None)

    assert inputs.shape == (2, 5, 6, 2)
    assert targets.shape == (2, 5, 6, 1)
    assert {record["uuid"] for record in records} == {
        "roi-1",
        "roi-output-only",
    }


def test_shared_row_reader_automatically_migrates_legacy_roi_database(tmp_path):
    database = tmp_path / "legacy.sqlite"
    create_legacy_database(database)

    with pytest.warns(RuntimeWarning, match="Migrated legacy ROI database"):
        rows = read_rows(database)

    assert inspect_dataset_kind(database) == "mask_refinement"
    assert {row["uuid"] for row in rows} == {
        "roi-1",
        "roi-output-only",
        "roi-unvalidated",
    }


def test_segmentation_config_resolution_migrates_before_schema_read(tmp_path):
    database = tmp_path / "legacy.sqlite"
    create_legacy_database(database)
    config_path = tmp_path / "segmentation.toml"
    config_path.write_text(
        """
[run]
task = "segmentation"
model = "unet"

[data]
input_shape = [5, 6, 2]
output_shape = [5, 6, 1]

[training]
loss = "bce_soft_dice"
""",
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="Migrated legacy ROI database"):
        config = resolve_config(
            config_path,
            database,
            tmp_path / "runs" / "migration-test",
        )

    assert config["dataset"]["dataset_type"] == "mask_refinement"
    assert config["dataset"]["schema_version"] == "1.1.0"
    assert config["dataset"]["lifecycle"] == "working"
    assert inspect_dataset_kind(database) == "mask_refinement"


def test_unet_analysis_reports_v1_identity_and_automatic_migration(tmp_path):
    database = tmp_path / "legacy.sqlite"
    create_legacy_database(database)

    with pytest.warns(RuntimeWarning, match="Migrated legacy ROI database"):
        report = validate_unet_dataset(
            database,
            target_input_shape=[5, 6, 2],
            target_output_shape=[5, 6, 1],
        )

    assert report["migration"]["migrated"]
    assert report["dataset"]["dataset_type"] == "mask_refinement"
    assert report["dataset"]["schema_version"] == "1.1.0"
    assert report["dataset"]["lifecycle"] == "working"
    assert len(report["dataset"]["dataset_id"]) == 36
    assert len(report["dataset"]["revision_id"]) == 36
    assert len(report["dataset"]["fingerprint_sha256"]) == 64
    assert report["schema_validation"]["valid"]


def test_split_manifest_identity_check_migrates_legacy_roi_before_reading(tmp_path):
    database = tmp_path / "legacy.sqlite"
    create_legacy_database(database)
    config = {
        "run": {"task": "segmentation"},
        "_split_manifest": {
            "dataset": {
                "dataset_id": "not-the-migrated-dataset",
                "revision_id": "not-the-migrated-revision",
                "fingerprint_sha256": "not-the-migrated-fingerprint",
            }
        },
    }

    with pytest.warns(RuntimeWarning, match="Migrated legacy ROI database"):
        assert not split_manifest_matches_dataset(config, database)

    assert inspect_dataset_kind(database) == "mask_refinement"
    assert list(tmp_path.glob("legacy.pre-v1-*.sqlite"))


def test_failed_migration_leaves_original_database_unchanged(tmp_path):
    database = tmp_path / "legacy.sqlite"
    create_legacy_database(database, orphan=True)

    with pytest.raises(ValueError, match="missing samples"):
        migrate_legacy_roi_database(database)

    assert inspect_dataset_kind(database) == "legacy_mask_refinement"
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM samples").fetchone()[0] == 3
        assert connection.execute(
            "SELECT count(*) FROM mask_annotations"
        ).fetchone()[0] == 3
    assert list(tmp_path.glob("legacy.pre-v1-*.sqlite"))
