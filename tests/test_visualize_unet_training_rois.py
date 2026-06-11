from __future__ import annotations

import numpy as np

from oracle_builder.masking.sqlite_io import create_or_update_image_sample, open_database, save_mask_annotation
from scripts.visualize_unet_training_rois import build_contact_sheet, make_overlay_tile, read_training_rows


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
