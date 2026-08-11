from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from oracle_builder.data.decoders import decode_blob
from oracle_builder.data.sqlite_dataset import ensure_schema
from oracle_builder.masking.sqlite_io import (
    create_or_update_image_sample,
    delete_image_sample,
    duplicate_image_sample,
    decode_mask,
    ensure_mask_annotation_table,
    encode_training_input,
    load_sample,
    make_training_input_tensor,
    open_database,
    save_mask_annotation,
)


def test_mask_annotation_table_is_created():
    conn = sqlite3.connect(":memory:")
    ensure_mask_annotation_table(conn)
    names = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    assert ("mask_annotations",) in names


def test_save_mask_annotation_updates_sample_and_preserves_history():
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    ensure_mask_annotation_table(conn)
    image = np.zeros((6, 6), dtype="uint8")
    create_or_update_image_sample(conn, "sample-1", image, "png", {"existing": "kept"})

    mask_a = np.zeros((6, 6), dtype="uint8")
    mask_a[1:3, 1:3] = 1
    first_id = save_mask_annotation(
        conn,
        "sample-1",
        mask_a,
        "png",
        method="test",
        parameters={"threshold": 0.5},
        validation={"valid": True},
    )
    mask_b = np.zeros((6, 6), dtype="uint8")
    mask_b[3:5, 3:5] = 1
    second_id = save_mask_annotation(
        conn,
        "sample-1",
        mask_b,
        "png",
        method="test",
        parameters={"threshold": 0.6},
        validation={"valid": True},
    )

    assert first_id != second_id
    assert conn.execute("SELECT count(*) FROM mask_annotations").fetchone()[0] == 2
    sample = load_sample(conn, "sample-1")
    decoded = sample["mask"]
    metadata = sample["metadata"]
    assert np.array_equal(decoded, mask_b)
    assert metadata["existing"] == "kept"
    current = conn.execute(
        "SELECT annotation_id FROM mask_annotations WHERE item_id = ? AND is_current = 1",
        ("sample-1",),
    ).fetchone()[0]
    assert current == second_id


def test_load_sample_reads_existing_mask():
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    image = np.zeros((4, 4), dtype="uint8")
    create_or_update_image_sample(conn, "sample-2", image, "png", {})
    mask = np.zeros((4, 4), dtype="uint8")
    mask[1:3, 1:3] = 1
    save_mask_annotation(conn, "sample-2", mask, "png", "test", {}, {"valid": True})
    sample = load_sample(conn, "sample-2")
    assert sample["image"].shape == (4, 4)
    assert np.array_equal(sample["mask"], mask)


def test_delete_image_sample_removes_item_and_annotation_history():
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    image = np.zeros((4, 4), dtype="uint8")
    create_or_update_image_sample(conn, "bad-sample", image, "png", {})
    save_mask_annotation(conn, "bad-sample", image, "png", "test", {}, {"valid": True})

    delete_image_sample(conn, "bad-sample")

    assert conn.execute("SELECT count(*) FROM dataset_items").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM mask_refinement_items").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM mask_annotations").fetchone()[0] == 0
    with pytest.raises(KeyError):
        delete_image_sample(conn, "bad-sample")


def test_duplicate_image_sample_reuses_image_and_copies_current_mask():
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    image = np.zeros((4, 4), dtype="uint8")
    image[1, 1] = 255
    create_or_update_image_sample(conn, "source", image, "png", {})
    save_mask_annotation(conn, "source", image, "png", "test", {}, {"valid": True})

    duplicate_uuid = duplicate_image_sample(conn, "source", "duplicate")

    assert duplicate_uuid == "duplicate"
    duplicate = load_sample(conn, duplicate_uuid)
    assert np.array_equal(duplicate["image"], image)
    assert np.array_equal(duplicate["mask"], (image > 0).astype("uint8"))
    assert duplicate["metadata"]["mask_builder_duplicate"]["source_uuid"] == "source"
    assert conn.execute("SELECT count(*) FROM mask_annotations WHERE item_id = 'duplicate'").fetchone()[0] == 1


def test_candidate_mask_is_stored_as_training_input_channel_and_aux_layer():
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    image = np.zeros((4, 5), dtype="uint8")
    image[:, 2:] = 100
    candidate = np.zeros((4, 5), dtype="uint8")
    candidate[1:3, 2:4] = 1

    create_or_update_image_sample(conn, "sample-candidate", image, "png", {}, candidate_mask=candidate)
    sample = load_sample(conn, "sample-candidate")
    training_input = make_training_input_tensor(sample["image"], sample["candidate_mask"])
    stored_candidate = sample["candidate_mask"]
    assert training_input.shape == (4, 5, 2)
    assert np.array_equal(training_input[..., 0], image)
    assert np.array_equal(training_input[..., 1] > 0, candidate > 0)
    assert np.array_equal(stored_candidate, candidate)
    assert np.array_equal(sample["image"], image)
    assert np.array_equal(sample["candidate_mask"], candidate)


def test_image_sample_update_with_candidate_mask_rewrites_two_channel_input():
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    image = np.zeros((4, 5), dtype="uint8")
    image[:, 2:] = 100
    candidate = np.zeros((4, 5), dtype="uint8")
    candidate[1:3, 2:4] = 1

    create_or_update_image_sample(conn, "sample-candidate", image, "png", {})
    create_or_update_image_sample(conn, "sample-candidate", image, "png", {}, candidate_mask=candidate)
    sample = load_sample(conn, "sample-candidate")
    training_input = make_training_input_tensor(sample["image"], sample["candidate_mask"])
    assert training_input.shape == (4, 5, 2)
    assert np.array_equal(training_input[..., 0], image)
    assert np.array_equal(training_input[..., 1] > 0, candidate > 0)
    assert sample["mask"] is None


def test_training_input_tensor_converts_rgb_roi_to_two_channels():
    image = np.zeros((3, 4, 3), dtype="uint8")
    image[..., 0] = 255
    candidate = np.zeros((3, 4), dtype="uint8")
    candidate[1, 2] = 1

    blob, encoding, dimensions = encode_training_input(image, candidate)
    training_input = np.asarray(decode_blob(blob, encoding, dimensions))
    expected_roi = make_training_input_tensor(image, candidate)[..., 0]

    assert encoding == "npy"
    assert training_input.shape == (3, 4, 2)
    assert np.array_equal(training_input[..., 0], expected_roi)
    assert training_input[1, 2, 1] == 255


def test_open_database_initializes_new_file_and_parent_directory(tmp_path):
    db_path = tmp_path / "nested" / "new_masks.sqlite"
    with open_database(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
    assert db_path.exists()
    assert {
        "dataset",
        "dataset_items",
        "assets",
        "mask_refinement_items",
        "mask_annotations",
    }.issubset(tables)


def test_open_database_can_refuse_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_database(tmp_path / "missing.sqlite", create=False)
