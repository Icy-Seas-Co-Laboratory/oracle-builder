from __future__ import annotations

import hashlib
import sqlite3
import uuid

import numpy as np

from oracle_builder.datasets.lifecycle import (
    release_training_dataset,
    restore_workspace_snapshot,
    save_workspace_snapshot,
)
from oracle_builder.datasets.schema import (
    COMMON_SCHEMA_SQL,
    MASK_REFINEMENT_SCHEMA_SQL,
    initialize_database,
    read_dataset_info,
    validate_database,
    workspace_fingerprint,
)
from oracle_builder.datasets.workspace import (
    add_annotation_label,
    add_annotation_review,
    add_item_label_annotation,
    add_model_evidence,
    create_inference_run,
)


def _workspace_item(connection: sqlite3.Connection) -> str:
    info = read_dataset_info(connection)
    item_id, asset_id = str(uuid.uuid4()), str(uuid.uuid4())
    payload = b"roi-bytes"
    connection.execute(
        """
        INSERT INTO assets (asset_id, dataset_id, content_sha256, payload, encoding, created_at)
        VALUES (?, ?, ?, ?, 'raw', ?)
        """,
        (asset_id, info["dataset_id"], hashlib.sha256(payload).hexdigest(), payload, "now"),
    )
    connection.execute(
        """INSERT INTO dataset_items
           (item_id, dataset_id, metadata_json, created_at, updated_at)
           VALUES (?, ?, '{}', 'now', 'now')""",
        (item_id, info["dataset_id"]),
    )
    connection.execute(
        "INSERT INTO mask_refinement_items (item_id, image_asset_id) VALUES (?, ?)",
        (item_id, asset_id),
    )
    return item_id


def test_annotation_workspace_retains_human_and_model_evidence(tmp_path):
    path = tmp_path / "workspace.sqlite"
    with sqlite3.connect(path) as connection:
        initialize_database(connection, "mask_refinement", name="review")
        item_id = _workspace_item(connection)
        label_id = add_annotation_label(connection, "Calanus")
        annotation_id = add_item_label_annotation(
            connection, item_id, label_id, annotator="reviewer"
        )
        add_annotation_review(connection, annotation_id, "second-reviewer", "verified")
        run_id = create_inference_run(connection, name="classifier")
        add_model_evidence(
            connection,
            run_id,
            item_id,
            predicted_label_id=label_id,
            prediction_confidence=0.9,
            logits=np.array([0.1, 0.9], dtype="float32"),
            embedding=np.ones(4, dtype="float32"),
        )
        connection.commit()
        canonical = validate_database(connection)["fingerprint"]
        workspace = workspace_fingerprint(connection)
        assert canonical
        assert workspace and workspace != canonical
        assert connection.execute("SELECT count(*) FROM model_evidence").fetchone()[0] == 1


def test_workspace_snapshot_and_restore_preserve_evidence(tmp_path):
    workspace = tmp_path / "workspace.sqlite"
    snapshot = tmp_path / "snapshot.sqlite"
    restored = tmp_path / "restored.sqlite"
    with sqlite3.connect(workspace) as connection:
        initialize_database(connection, "mask_refinement")
        run_id = create_inference_run(connection, name="model")
        item_id = _workspace_item(connection)
        add_model_evidence(connection, run_id, item_id, output=np.zeros((2, 2)))
        connection.commit()
        source_fingerprint = workspace_fingerprint(connection)
    result = save_workspace_snapshot(workspace, snapshot, note="review milestone")
    assert result["workspace_fingerprint"] == source_fingerprint
    with sqlite3.connect(snapshot) as connection:
        assert read_dataset_info(connection)["lifecycle"] == "frozen"
        try:
            create_inference_run(connection, name="must-not-change-snapshot")
        except sqlite3.DatabaseError as exc:
            assert "dataset is frozen" in str(exc)
        else:
            raise AssertionError("frozen workspace snapshots must reject evidence writes")
    restored_result = restore_workspace_snapshot(snapshot, restored, reason="continue")
    assert restored_result["lifecycle"] == "working"
    with sqlite3.connect(restored) as connection:
        assert workspace_fingerprint(connection) == source_fingerprint


def test_migrates_v11_to_v12_annotation_workspace_schema(tmp_path):
    path = tmp_path / "v11.sqlite"
    dataset_id, revision_id = str(uuid.uuid4()), str(uuid.uuid4())
    with sqlite3.connect(path) as connection:
        connection.executescript(COMMON_SCHEMA_SQL)
        connection.executescript(MASK_REFINEMENT_SCHEMA_SQL)
        connection.execute(
            "INSERT INTO ob_schema VALUES (1, 'oracle_builder_dataset', '1.1.0', 'now')"
        )
        connection.execute(
            """
            INSERT INTO dataset VALUES
              (1, ?, ?, NULL, 'mask_refinement', 'old', NULL, NULL, NULL,
               'working', 'now', 'now', NULL, '{}')
            """,
            (dataset_id, revision_id),
        )
        connection.commit()
        assert initialize_database(connection, "mask_refinement")["schema_version"] == "1.2.0"
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'model_evidence'"
        ).fetchone()


def test_training_release_is_frozen_and_excludes_derived_evidence(tmp_path):
    workspace = tmp_path / "workspace.sqlite"
    release = tmp_path / "release.sqlite"
    with sqlite3.connect(workspace) as connection:
        source_info = initialize_database(connection, "mask_refinement")
        item_id = _workspace_item(connection)
        run_id = create_inference_run(connection, name="model")
        add_model_evidence(connection, run_id, item_id, output=np.zeros((2, 2)))
        connection.commit()
    result = release_training_dataset(workspace, release, name="training-release")
    with sqlite3.connect(release) as connection:
        info = read_dataset_info(connection)
        assert validate_database(connection)["valid"]
        assert info["lifecycle"] == "frozen"
        assert info["dataset_id"] != source_info["dataset_id"]
        assert connection.execute("SELECT count(*) FROM model_evidence").fetchone()[0] == 0
    assert result["lifecycle"] == "frozen"
