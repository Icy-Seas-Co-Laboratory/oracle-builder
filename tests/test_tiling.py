from __future__ import annotations

import numpy as np

from oracle_builder.data.sqlite_dataset import load_arrays, load_prediction_arrays
from oracle_builder.data.tiling import coverage_map, extract_tile, plan_tiles, reassemble_tiles
from oracle_builder.evaluation.segmentation import predict_reassembled_segmentation
from oracle_builder.masking.sqlite_io import create_or_update_image_sample, open_database, save_mask_annotation


def test_512_square_produces_four_tiles_without_overlap():
    plans = plan_tiles((512, 512), (256, 256), overlap_fraction=0.0)

    assert len(plans) == 4
    assert [plan["origin"] for plan in plans] == [[0, 0], [0, 256], [256, 0], [256, 256]]
    assert np.all(coverage_map(plans) == 1)


def test_512_square_produces_nine_tiles_at_half_overlap():
    plans = plan_tiles((512, 512), (256, 256), overlap_fraction=0.5)

    assert len(plans) == 9
    assert sorted({plan["origin"][0] for plan in plans}) == [0, 128, 256]
    assert sorted({plan["origin"][1] for plan in plans}) == [0, 128, 256]
    assert coverage_map(plans).min() == 1


def test_non_divisible_dimensions_are_fully_covered():
    plans = plan_tiles((600, 730), (256, 256), overlap_fraction=0.3)
    coverage = coverage_map(plans)

    assert plans[-1]["origin"] == [344, 474]
    assert coverage.shape == (600, 730)
    assert coverage.min() >= 1


def test_small_dimension_is_center_padded():
    value = np.ones((100, 300, 1), dtype="float32")
    plans = plan_tiles((100, 300), (256, 256), overlap_fraction=0.0)
    tile = extract_tile(value, plans[0])

    assert tile.shape == (256, 256, 1)
    assert plans[0]["tile_offset"][0] == 78
    assert np.all(tile[:78] == 0)
    assert np.all(tile[78:178] == 1)


def test_overlapping_tiles_reassemble_by_weighted_average():
    plans = plan_tiles((4, 6), (4, 4), overlap_fraction=0.5)
    tiles = [
        np.full((4, 4, 1), index + 1, dtype="float32")
        for index in range(len(plans))
    ]

    result = reassemble_tiles(tiles, plans, blend_mode="uniform")

    assert result.shape == (4, 6, 1)
    assert np.all(result[:, :2] == 1)
    assert np.all(result[:, 2:4] == 1.5)
    assert np.all(result[:, 4:] == 2)


def tiled_config(overlap=0.0):
    return {
        "run": {"task": "segmentation", "model": "unet", "seed": 123},
        "data": {
            "input_shape": [8, 8, 2],
            "output_shape": [8, 8, 1],
            "validation_split": 0.0,
            "test_split": 0.0,
        },
        "training": {"loss": "bce_soft_dice", "segmentation_target": "validated_mask"},
        "tiling": {
            "enabled": True,
            "overlap_fraction": overlap,
            "blend_mode": "uniform",
            "tile_large_rois_only": True,
            "normalize_training_coverage": True,
        },
    }


def create_large_sample(path):
    image = np.arange(256, dtype="uint8").reshape(16, 16)
    candidate = np.zeros((16, 16), dtype="uint8")
    candidate[3:13, 3:13] = 1
    validated = candidate.copy()
    with open_database(path) as connection:
        create_or_update_image_sample(connection, "large", image, "png", {}, candidate_mask=candidate)
        save_mask_annotation(connection, "large", validated, "png", "test", {}, {"valid": True})
    return image, candidate


def test_large_roi_loader_expands_synchronized_tiles_after_split_assignment(tmp_path):
    database = tmp_path / "large.sqlite"
    _image, candidate = create_large_sample(database)

    x, y, records = load_arrays(database, tiled_config(overlap=0.0))

    assert x.shape == (4, 8, 8, 2)
    assert y.shape == (4, 8, 8, 1)
    assert {record["source_uuid"] for record in records} == {"large"}
    assert {record["split"] for record in records} == {"train"}
    assert [record["tile_index"] for record in records] == [0, 1, 2, 3]
    reconstructed_candidate = reassemble_tiles(
        [tile[..., 1:2] for tile in x],
        [record["tile_plan"] for record in records],
        blend_mode="uniform",
    )
    assert np.array_equal(reconstructed_candidate[..., 0] > 0.5, candidate > 0)


def test_overlap_coverage_weights_sum_to_one_per_source_pixel(tmp_path):
    database = tmp_path / "large.sqlite"
    create_large_sample(database)

    _x, _y, records = load_arrays(database, tiled_config(overlap=0.5))
    plans = [record["tile_plan"] for record in records]
    source_sum = np.zeros((16, 16), dtype="float32")
    for record, plan in zip(records, plans, strict=True):
        y, x = plan["origin"]
        crop_h, crop_w = plan["crop_shape"]
        offset_y, offset_x = plan["tile_offset"]
        source_sum[y : y + crop_h, x : x + crop_w] += record["pixel_weights"][
            offset_y : offset_y + crop_h, offset_x : offset_x + crop_w
        ]

    assert len(records) == 9
    assert np.allclose(source_sum, 1.0)


class EchoFirstChannelModel:
    def predict(self, x, verbose=0):
        return np.asarray(x)[..., :1]


def test_tiled_predictions_reassemble_to_one_original_roi(tmp_path):
    database = tmp_path / "large.sqlite"
    image, _candidate = create_large_sample(database)
    config = tiled_config(overlap=0.5)
    x, targets, records = load_prediction_arrays(database, config)

    predictions, reassembled_targets, source_records = predict_reassembled_segmentation(
        EchoFirstChannelModel(), x, targets, records, config
    )

    assert len(predictions) == len(reassembled_targets) == len(source_records) == 1
    assert predictions[0].shape == (16, 16, 1)
    assert np.allclose(predictions[0][..., 0], image.astype("float32") / 255.0)
    assert source_records[0]["uuid"] == "large"
