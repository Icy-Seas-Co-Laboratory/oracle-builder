from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from oracle_builder.data.decoders import decode_blob
from oracle_builder.data.sqlite_dataset import ensure_schema
from oracle_builder.masking.sqlite_io import (
    create_or_update_image_sample,
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
    row = conn.execute(
        "SELECT output_blob, output_blob_encoding, output_blob_dimensions, metadata_json FROM samples WHERE uuid = ?",
        ("sample-1",),
    ).fetchone()
    decoded = decode_mask(row[0], row[1], row[2])
    metadata = json.loads(row[3])
    assert np.array_equal(decoded, mask_b)
    assert metadata["existing"] == "kept"
    assert metadata["mask_builder"]["last_annotation_id"] == second_id


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


def test_candidate_mask_is_stored_as_training_input_channel_and_aux_layer():
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    image = np.zeros((4, 5), dtype="uint8")
    image[:, 2:] = 100
    candidate = np.zeros((4, 5), dtype="uint8")
    candidate[1:3, 2:4] = 1

    create_or_update_image_sample(conn, "sample-candidate", image, "png", {}, candidate_mask=candidate)
    row = conn.execute(
        """
        SELECT input_blob, input_blob_encoding, input_blob_dimensions,
               input_aux_blob, input_aux_blob_encoding, input_aux_blob_dimensions
        FROM samples WHERE uuid = ?
        """,
        ("sample-candidate",),
    ).fetchone()

    training_input = np.asarray(decode_blob(row[0], row[1], row[2]))
    stored_candidate = decode_mask(row[3], row[4], row[5])
    sample = load_sample(conn, "sample-candidate")
    assert row[1] == "npy"
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
    row = conn.execute(
        """
        SELECT input_blob, input_blob_encoding, input_blob_dimensions, output_blob
        FROM samples WHERE uuid = ?
        """,
        ("sample-candidate",),
    ).fetchone()

    training_input = np.asarray(decode_blob(row[0], row[1], row[2]))
    assert row[1] == "npy"
    assert training_input.shape == (4, 5, 2)
    assert np.array_equal(training_input[..., 0], image)
    assert np.array_equal(training_input[..., 1] > 0, candidate > 0)
    assert row[3] is None


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
    assert {"samples", "mask_annotations"}.issubset(tables)


def test_open_database_can_refuse_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_database(tmp_path / "missing.sqlite", create=False)
