from __future__ import annotations

import numpy as np

from oracle_builder.data.decoders import encode_npy
from oracle_builder.evaluation.predictions import init_predictions_db
from oracle_builder.masking.sqlite_io import create_or_update_image_sample, open_database, save_mask_annotation
from scripts.visualize_unet_training_rois import (
    build_contact_sheet,
    build_side_by_side_sheet,
    make_overlay_tile,
    prediction_to_roi_mask,
    read_predictions,
    read_training_rows,
)


def test_read_training_rows_uses_training_split_assignment(tmp_path):
    db_path = tmp_path / "masks.sqlite"
    with open_database(db_path) as conn:
        for index in range(10):
            image = np.ones((5, 6), dtype="uint8") * index
            candidate = np.zeros((5, 6), dtype="uint8")
            candidate[1:4, 2:5] = 1
            refined = np.zeros((5, 6), dtype="uint8")
            refined[2:4, 3:5] = 1
            create_or_update_image_sample(conn, f"sample-{index}", image, "png", {}, candidate_mask=candidate)
            save_mask_annotation(conn, f"sample-{index}", refined, "png", "test", {}, {"valid": True})

    rows = read_training_rows(db_path, validation_split=0.2, test_split=0.1, seed=123, split="train")

    assert len(rows) == 7
    assert all(row["split"] == "train" for row in rows)


def test_overlay_tile_draws_candidate_blue_and_refined_green():
    image = np.ones((8, 8), dtype="uint8") * 100
    candidate = np.zeros((8, 8), dtype="uint8")
    candidate[1:4, 1:4] = 1
    refined = np.zeros((8, 8), dtype="uint8")
    refined[4:7, 4:7] = 1
    sample = {"uuid": "sample", "image": image, "candidate_mask": candidate, "mask": refined}

    tile = make_overlay_tile(sample, thumbnail_size=8, candidate_alpha=0.5, refined_alpha=0.5, show_label=False)
    pixels = np.asarray(tile)

    assert pixels[2, 2, 2] > pixels[2, 2, 1]
    assert pixels[5, 5, 1] > pixels[5, 5, 2]


def test_contact_sheet_contains_all_tiles():
    samples = []
    for index in range(3):
        image = np.ones((8, 8), dtype="uint8") * 100
        mask = np.zeros((8, 8), dtype="uint8")
        mask[2:6, 2:6] = 1
        samples.append({"uuid": f"sample-{index}", "image": image, "candidate_mask": mask, "mask": mask})

    sheet = build_contact_sheet(
        samples,
        thumbnail_size=16,
        columns=2,
        candidate_alpha=0.35,
        refined_alpha=0.45,
        show_labels=True,
    )

    assert sheet.size == (32, 88)


def test_prediction_mask_removes_letterbox_padding_and_restores_roi_shape():
    prediction = np.zeros((8, 8, 1), dtype="float32")
    prediction[2:6, 2:6, 0] = 0.9

    mask = prediction_to_roi_mask(prediction, (2, 4), threshold=0.5)

    assert mask.shape == (2, 4)
    assert np.all(mask[:, 1:3] == 1)
    assert np.all(mask[:, [0, 3]] == 0)


def test_side_by_side_sheet_renders_three_columns_per_roi():
    image = np.ones((4, 8), dtype="uint8") * 100
    candidate = np.zeros((4, 8), dtype="uint8")
    candidate[:, :2] = 1
    validated = np.zeros((4, 8), dtype="uint8")
    validated[:, 6:] = 1
    sample = {"uuid": "sample", "image": image, "candidate_mask": candidate, "mask": validated}
    prediction = np.zeros((8, 8, 1), dtype="float32")
    prediction[2:6, 3:5, 0] = 1.0

    sheet = build_side_by_side_sheet(
        [sample],
        {"sample": prediction},
        thumbnail_size=16,
        candidate_alpha=0.5,
        prediction_alpha=0.5,
        refined_alpha=0.5,
        prediction_threshold=0.5,
        show_labels=False,
    )

    pixels = np.asarray(sheet)
    assert sheet.size == (48, 40)
    assert pixels[26, 5, 2] > pixels[26, 5, 1]
    assert pixels[26, 23, 0] > pixels[26, 23, 1]
    assert pixels[26, 42, 1] > pixels[26, 42, 2]


def test_read_predictions_selects_named_prediction_set(tmp_path):
    database = tmp_path / "predictions.sqlite"
    connection = init_predictions_db(database)
    connection.execute(
        "INSERT INTO prediction_sets VALUES (?, ?, ?, ?, ?)",
        ("run-a", "now", None, None, "{}"),
    )
    connection.execute(
        "INSERT INTO prediction_sets VALUES (?, ?, ?, ?, ?)",
        ("run-b", "now", None, None, "{}"),
    )
    for name, value in (("run-a", 0.25), ("run-b", 0.75)):
        connection.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (name, "sample", "test", encode_npy(np.zeros((2, 2))), "npy", encode_npy(np.full((2, 2), value)), "npy", None, "{}", "{}"),
        )
    connection.commit()
    connection.close()

    predictions = read_predictions(database, prediction_set="run-b")

    assert np.all(predictions["sample"] == 0.75)
