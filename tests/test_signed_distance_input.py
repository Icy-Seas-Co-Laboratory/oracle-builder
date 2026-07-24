from __future__ import annotations

import numpy as np
import pytest

import mask_builder
from oracle_builder.config import resolve_config, validate_config
from oracle_builder.data.signed_distance import signed_distance_field
from oracle_builder.data.sqlite_dataset import load_arrays
from oracle_builder.masking.sqlite_io import create_or_update_image_sample, open_database, save_mask_annotation


def create_candidate_dataset(path):
    image = np.arange(64, dtype="uint8").reshape(8, 8)
    candidate = np.zeros((8, 8), dtype="uint8")
    candidate[2:6, 2:6] = 1
    validated = candidate.copy()
    with open_database(path) as connection:
        create_or_update_image_sample(connection, "sample", image, "png", {}, candidate_mask=candidate)
        save_mask_annotation(connection, "sample", validated, "png", "test", {}, {"valid": True})
    return image, candidate


def sdf_config():
    return {
        "run": {"task": "segmentation", "model": "unet", "seed": 123},
        "data": {
            "input_shape": [8, 8, 3],
            "output_shape": [8, 8, 1],
            "candidate_sdf": True,
            "candidate_sdf_clip_distance": 4.0,
            "validation_split": 0.0,
            "test_split": 0.0,
        },
        "training": {"loss": "bce_soft_dice", "segmentation_target": "validated_mask"},
    }


def test_signed_distance_is_positive_inside_and_negative_outside():
    mask = np.zeros((5, 5), dtype="uint8")
    mask[2, 2] = 1

    sdf = signed_distance_field(mask, clip_distance=2.0)

    assert sdf[2, 2] == 0.5
    assert sdf[2, 1] == -0.5
    assert sdf[0, 0] == -1.0
    assert np.all(sdf >= -1.0)
    assert np.all(sdf <= 1.0)


def test_signed_distance_handles_empty_and_full_masks():
    assert np.all(signed_distance_field(np.zeros((3, 3))) == -1.0)
    assert np.all(signed_distance_field(np.ones((3, 3))) == 1.0)


def test_loader_appends_candidate_sdf_as_third_channel(tmp_path):
    database = tmp_path / "sdf.sqlite"
    _image, candidate = create_candidate_dataset(database)

    x, _y, records = load_arrays(database, sdf_config())

    assert x.shape == (1, 8, 8, 3)
    assert float(x[..., 0].min()) >= 0.0 and float(x[..., 0].max()) <= 1.0
    assert np.array_equal(x[0, ..., 1] > 0.5, candidate > 0)
    assert np.all(x[0, 2:6, 2:6, 2] > 0)
    assert x[0, 0, 0, 2] < 0
    assert records[0]["candidate_mask"].shape == (8, 8, 1)


def test_candidate_sdf_config_requires_three_model_channels():
    config = sdf_config()
    config["data"]["input_shape"] = [8, 8, 2]

    with pytest.raises(ValueError, match="three-channel"):
        validate_config(config)


def test_mask_builder_generates_candidate_sdf_config(monkeypatch, tmp_path):
    database = tmp_path / "sdf.sqlite"
    config_path = tmp_path / "sdf.toml"
    create_candidate_dataset(database)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mask_builder.py",
            "--database",
            str(database),
            "--write-unet-config",
            str(config_path),
            "--unet-candidate-sdf",
            "--unet-candidate-sdf-clip-distance",
            "12",
        ],
    )

    assert mask_builder.main() == 0
    resolved = resolve_config(config_path, database, tmp_path / "run")
    assert resolved["data"]["input_shape"] == [8, 8, 3]
    assert resolved["data"]["candidate_sdf"] is True
    assert resolved["data"]["candidate_sdf_clip_distance"] == 12.0
