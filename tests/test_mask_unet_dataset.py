from __future__ import annotations

import json
import sqlite3

import numpy as np

import mask_builder
import model_training
from oracle_builder.config import resolve_config
from oracle_builder.data.sqlite_dataset import load_arrays
from oracle_builder.masking.sqlite_io import create_or_update_image_sample, open_database, save_mask_annotation
from oracle_builder.masking.unet_dataset import validate_unet_dataset, write_unet_config_from_dataset


def _create_mask_builder_unet_dataset(db_path):
    with open_database(db_path) as conn:
        for index in range(2):
            image = np.zeros((8, 9, 3), dtype="uint8")
            image[..., index] = 120
            mask = np.zeros((8, 9), dtype="uint8")
            mask[2:6, 3:7] = 1
            uuid = f"sample-{index}"
            create_or_update_image_sample(conn, uuid, image, "png", {"source": "test"})
            save_mask_annotation(
                conn,
                uuid,
                mask,
                "png",
                method="test",
                parameters={"threshold": 0.5},
                validation={"valid": True},
            )
        image = np.zeros((8, 9, 3), dtype="uint8")
        create_or_update_image_sample(conn, "unmasked-sample", image, "png", {"source": "test"})


def test_mask_builder_dataset_validates_for_unet(tmp_path):
    db_path = tmp_path / "masks.sqlite"
    _create_mask_builder_unet_dataset(db_path)

    report = validate_unet_dataset(db_path)

    assert report["valid"]
    assert report["item_count"] == 3
    assert report["annotated_item_count"] == 2
    assert report["missing_current_mask_count"] == 1
    assert report["annotation_count"] == 2
    assert "split_counts" not in report
    assert report["usable_sample_count"] == 2
    assert report["inferred_input_shape"] == [8, 9, 3]
    assert report["inferred_output_shape"] == [8, 9, 1]


def test_api_candidate_dataset_validates_as_two_channel_unet_input(tmp_path):
    db_path = tmp_path / "api_masks.sqlite"
    with open_database(db_path) as conn:
        image = np.zeros((8, 9), dtype="uint8")
        image[:, 4:] = 120
        candidate = np.zeros((8, 9), dtype="uint8")
        candidate[1:5, 2:7] = 1
        validated = np.zeros((8, 9), dtype="uint8")
        validated[2:6, 3:7] = 1
        create_or_update_image_sample(conn, "api-sample", image, "png", {"source": "api"}, candidate_mask=candidate)
        save_mask_annotation(
            conn,
            "api-sample",
            validated,
            "png",
            method="test",
            parameters={},
            validation={"valid": True},
        )

    report = validate_unet_dataset(db_path)
    config = {
        "run": {"task": "segmentation", "seed": 123},
        "data": {
            "input_shape": [8, 9, 2],
            "output_shape": [8, 9, 1],
            "validation_split": 0.2,
            "test_split": 0.1,
        },
    }
    x, y, records = load_arrays(db_path, config, split=None)

    assert report["valid"]
    assert report["inferred_input_shape"] == [8, 9, 2]
    assert report["inferred_output_shape"] == [8, 9, 1]
    assert x.shape == (1, 8, 9, 2)
    assert y.shape == (1, 8, 9, 1)
    assert np.array_equal(x[0, ..., 1] > 0, candidate > 0)
    assert np.array_equal(y[0, ..., 0] > 0, validated > 0)
    assert records[0]["uuid"] == "api-sample"


def test_variable_api_roi_dataset_resizes_to_configured_unet_shape(tmp_path):
    db_path = tmp_path / "api_masks.sqlite"
    with open_database(db_path) as conn:
        for index, shape in enumerate(((7, 9), (5, 11))):
            image = np.zeros(shape, dtype="uint8")
            image[:, shape[1] // 2 :] = 120
            candidate = np.zeros(shape, dtype="uint8")
            candidate[1:-1, 2:-2] = 1
            validated = np.zeros(shape, dtype="uint8")
            validated[2:-1, 3:-1] = 1
            create_or_update_image_sample(
                conn,
                f"api-sample-{index}",
                image,
                "png",
                {"source": "api"},
                candidate_mask=candidate,
            )
            save_mask_annotation(
                conn,
                f"api-sample-{index}",
                validated,
                "png",
                method="test",
                parameters={},
                validation={"valid": True},
            )

    report = validate_unet_dataset(db_path, target_input_shape=[16, 16, 2], target_output_shape=[16, 16, 1])
    config = {
        "run": {"task": "segmentation", "seed": 123},
        "data": {
            "input_shape": [16, 16, 2],
            "output_shape": [16, 16, 1],
            "validation_split": 0.0,
            "test_split": 0.0,
        },
    }
    x, y, records = load_arrays(db_path, config, split="train")

    assert report["valid"]
    assert report["warnings"]
    assert x.shape == (2, 16, 16, 2)
    assert y.shape == (2, 16, 16, 1)
    assert set(np.unique(x[..., 1]).tolist()).issubset({0.0, 1.0})
    assert set(np.unique(y).tolist()).issubset({0.0, 1.0})
    assert len(records) == 2


def test_write_unet_config_accepts_target_shapes_for_variable_rois(tmp_path):
    db_path = tmp_path / "api_masks.sqlite"
    config_path = tmp_path / "configs" / "api_unet.toml"
    with open_database(db_path) as conn:
        for index, shape in enumerate(((7, 9), (5, 11))):
            image = np.zeros(shape, dtype="uint8")
            candidate = np.zeros(shape, dtype="uint8")
            candidate[1:-1, 2:-2] = 1
            validated = np.zeros(shape, dtype="uint8")
            validated[2:-1, 3:-1] = 1
            create_or_update_image_sample(
                conn,
                f"api-sample-{index}",
                image,
                "png",
                {"source": "api"},
                candidate_mask=candidate,
            )
            save_mask_annotation(
                conn,
                f"api-sample-{index}",
                validated,
                "png",
                method="test",
                parameters={},
                validation={"valid": True},
            )

    result = write_unet_config_from_dataset(
        db_path,
        config_path,
        batch_size=2,
        epochs=5,
        target_input_shape=[16, 16, 2],
        target_output_shape=[16, 16, 1],
    )
    resolved = resolve_config(config_path, db_path, tmp_path / "runs" / "run")

    assert result["validation"]["valid"]
    assert result["validation"]["warnings"]
    assert resolved["data"]["input_shape"] == [16, 16, 2]
    assert resolved["data"]["output_shape"] == [16, 16, 1]
    assert resolved["data"]["batch_size"] == 2
    assert resolved["training"]["epochs"] == 5


def test_mask_builder_dataset_loads_as_unet_training_arrays(tmp_path):
    db_path = tmp_path / "masks.sqlite"
    _create_mask_builder_unet_dataset(db_path)
    config = {
        "run": {"task": "segmentation", "seed": 123},
        "data": {
            "input_shape": [8, 9, 3],
            "output_shape": [8, 9, 1],
            "validation_split": 0.2,
            "test_split": 0.1,
        },
    }

    x, y, records = load_arrays(db_path, config, split=None)

    assert x.shape == (2, 8, 9, 3)
    assert y.shape == (2, 8, 9, 1)
    assert set(np.unique(y).tolist()) == {0.0, 1.0}
    assert [record["uuid"] for record in records] == ["sample-0", "sample-1"]


def test_write_unet_config_from_dataset(tmp_path):
    db_path = tmp_path / "masks.sqlite"
    config_path = tmp_path / "configs" / "unet.toml"
    _create_mask_builder_unet_dataset(db_path)

    result = write_unet_config_from_dataset(db_path, config_path, batch_size=4, epochs=3)
    resolved = resolve_config(config_path, db_path, tmp_path / "runs" / "run")

    assert config_path.exists()
    assert result["validation"]["valid"]
    assert resolved["run"]["task"] == "segmentation"
    assert resolved["run"]["model"] == "unet"
    assert resolved["data"]["input_shape"] == [8, 9, 3]
    assert resolved["data"]["output_shape"] == [8, 9, 1]
    assert resolved["data"]["batch_size"] == 4
    assert resolved["training"]["epochs"] == 3


def test_mask_save_metadata_records_unet_training_shapes(tmp_path):
    db_path = tmp_path / "masks.sqlite"
    _create_mask_builder_unet_dataset(db_path)

    with sqlite3.connect(db_path) as conn:
        metadata = json.loads(
            conn.execute(
                "SELECT metadata_json FROM dataset_items WHERE item_id = ?",
                ("sample-0",),
            ).fetchone()[0]
        )

    assert metadata["mask_builder"]["training_task"] == "segmentation"
    assert metadata["mask_builder"]["mask_type"] == "binary"
    assert metadata["mask_builder"]["input_shape"] == [8, 9, 3]
    assert metadata["mask_builder"]["output_shape"] == [8, 9, 1]


def test_mask_builder_cli_validate_and_write_unet_config(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "masks.sqlite"
    config_path = tmp_path / "unet.toml"
    _create_mask_builder_unet_dataset(db_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mask_builder.py",
            "--database",
            str(db_path),
            "--validate-unet-dataset",
            "--write-unet-config",
            str(config_path),
        ],
    )

    assert mask_builder.main() == 0
    output = capsys.readouterr().out
    assert '"valid": true' in output
    assert config_path.exists()


def test_mask_builder_cli_write_unet_config_with_target_shapes(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "api_masks.sqlite"
    config_path = tmp_path / "api_unet.toml"
    with open_database(db_path) as conn:
        for index, shape in enumerate(((7, 9), (5, 11))):
            image = np.zeros(shape, dtype="uint8")
            candidate = np.zeros(shape, dtype="uint8")
            candidate[1:-1, 2:-2] = 1
            validated = np.zeros(shape, dtype="uint8")
            validated[2:-1, 3:-1] = 1
            create_or_update_image_sample(conn, f"api-sample-{index}", image, "png", {}, candidate_mask=candidate)
            save_mask_annotation(conn, f"api-sample-{index}", validated, "png", "test", {}, {"valid": True})
    monkeypatch.setattr(
        "sys.argv",
        [
            "mask_builder.py",
            "--database",
            str(db_path),
            "--write-unet-config",
            str(config_path),
            "--unet-input-shape",
            "16,16,2",
            "--unet-output-shape",
            "16x16x1",
        ],
    )

    assert mask_builder.main() == 0
    output = capsys.readouterr().out
    resolved = resolve_config(config_path, db_path, tmp_path / "runs" / "run")
    assert '"valid": true' in output
    assert resolved["data"]["input_shape"] == [16, 16, 2]
    assert resolved["data"]["output_shape"] == [16, 16, 1]


def test_mask_builder_cli_can_generate_unet_plus_plus_config(monkeypatch, tmp_path):
    db_path = tmp_path / "masks.sqlite"
    config_path = tmp_path / "unetpp.toml"
    _create_mask_builder_unet_dataset(db_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mask_builder.py",
            "--database",
            str(db_path),
            "--write-unet-config",
            str(config_path),
            "--unet-model",
            "unet_plus_plus",
        ],
    )

    assert mask_builder.main() == 0
    resolved = resolve_config(config_path, db_path, tmp_path / "run")
    assert resolved["run"]["model"] == "unet_plus_plus"
    assert resolved["training"]["loss"] == "bce_soft_dice"


def test_mask_builder_cli_can_generate_candidate_delta_config(monkeypatch, tmp_path):
    db_path = tmp_path / "masks.sqlite"
    config_path = tmp_path / "delta.toml"
    with open_database(db_path) as connection:
        image = np.zeros((8, 9), dtype="uint8")
        candidate = np.zeros((8, 9), dtype="uint8")
        validated = np.zeros((8, 9), dtype="uint8")
        candidate[1:5, 1:5] = 1
        validated[2:6, 2:6] = 1
        create_or_update_image_sample(connection, "sample", image, "png", {}, candidate_mask=candidate)
        save_mask_annotation(connection, "sample", validated, "png", "test", {}, {"valid": True})
    monkeypatch.setattr(
        "sys.argv",
        [
            "mask_builder.py",
            "--database",
            str(db_path),
            "--write-unet-config",
            str(config_path),
            "--unet-segmentation-target",
            "candidate_delta",
        ],
    )

    assert mask_builder.main() == 0
    resolved = resolve_config(config_path, db_path, tmp_path / "run")
    assert resolved["training"]["segmentation_target"] == "candidate_delta"


def test_mask_builder_cli_can_generate_tiled_config(monkeypatch, tmp_path):
    db_path = tmp_path / "masks.sqlite"
    config_path = tmp_path / "tiled.toml"
    _create_mask_builder_unet_dataset(db_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mask_builder.py",
            "--database",
            str(db_path),
            "--write-unet-config",
            str(config_path),
            "--unet-tiling",
            "--unet-tiling-overlap",
            "0.3",
            "--unet-tiling-blend",
            "uniform",
        ],
    )

    assert mask_builder.main() == 0
    resolved = resolve_config(config_path, db_path, tmp_path / "run")
    assert resolved["tiling"]["enabled"] is True
    assert resolved["tiling"]["overlap_fraction"] == 0.3
    assert resolved["tiling"]["blend_mode"] == "uniform"


def test_model_training_preflight_validates_without_creating_run_dir(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "masks.sqlite"
    config_path = tmp_path / "unet.toml"
    runs_dir = tmp_path / "runs"
    _create_mask_builder_unet_dataset(db_path)
    write_unet_config_from_dataset(db_path, config_path, batch_size=4, epochs=3)
    monkeypatch.setattr(
        "sys.argv",
        [
            "model_training.py",
            "--config",
            str(config_path),
            "--input",
            str(db_path),
            "--output",
            "preflight-run",
            "--runs-dir",
            str(runs_dir),
            "--preflight",
        ],
    )

    assert model_training.main() == 0
    output = capsys.readouterr().out
    assert '"valid": true' in output
    assert not (runs_dir / "preflight-run").exists()
