from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image

from oracle_builder.data.classification_import import build_parser, import_folders
from oracle_builder.data.decoders import decode_blob, prepare_classification_input
from oracle_builder.data.polarity import infer_source_polarity
from oracle_builder.data.sqlite_dataset import load_arrays
from oracle_builder.data.sqlite_stream import SQLiteClassificationSource


def write_image(path: Path, color: tuple[int, int, int], size=(24, 12)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def options_for(source: Path, output: Path, *extra: str):
    return build_parser().parse_args(
        ["--input", str(source), "--output", str(output), *extra]
    )


def test_folder_import_preserves_originals_labels_audit_and_sidecars(tmp_path):
    source = tmp_path / "library"
    write_image(source / "cod" / "one.jpg", (255, 0, 0))
    write_image(source / "cod" / "nested" / "two.png", (200, 0, 0))
    write_image(source / "salmon" / "three.jpg", (0, 255, 0))
    write_image(source / "salmon" / "four.jpg", (0, 200, 0))
    (source / "metadata.toml").write_text('title = "Example library"\nyear = 2026\n')
    (source / "about.json").write_text('{"owner": "Icy Seas"}\n')
    output = tmp_path / "library.sqlite"

    summary = import_folders(
        options_for(
            source,
            output,
            "--duplicate-policy",
            "allow",
        )
    )

    assert summary["status_counts"] == {"ready": 4}
    with sqlite3.connect(output) as connection:
        assert connection.execute("SELECT count(*) FROM dataset_items").fetchone()[0] == 4
        assert connection.execute(
            "SELECT class_index, name FROM classification_labels ORDER BY class_index"
        ).fetchall() == [(0, "cod"), (1, "salmon")]
        assert connection.execute("SELECT count(*) FROM import_events").fetchone()[0] == 1
        sidecars = connection.execute(
            "SELECT name, parsed_json FROM metadata_documents ORDER BY name"
        ).fetchall()
        encoding, blob = connection.execute(
            """
            SELECT a.encoding, a.payload
            FROM dataset_items di
            JOIN classification_items ci ON ci.item_id = di.item_id
            JOIN assets a ON a.asset_id = ci.image_asset_id
            WHERE di.source_key LIKE '%one.jpg'
            """
        ).fetchone()
    assert [row[0] for row in sidecars] == ["about.json", "metadata.toml"]
    assert json.loads(sidecars[1][1])["title"] == "Example library"
    assert encoding == "jpg"
    assert blob[:2] == b"\xff\xd8"
    assert output.with_suffix(".labels.json").exists()
    assert output.with_suffix(".import_report.json").exists()
    assert output.with_suffix(".import_report.csv").exists()

    config = {
        "run": {"task": "classification", "seed": 123},
        "data": {"input_shape": [16, 16, 3]},
        "preprocessing": {
            "resize_mode": "fit_pad",
            "normalization": "dtype",
            "rescale": True,
            "invert": False,
            "channel_mode": "rgb",
            "interpolation": "bilinear",
            "pad_value": 0.0,
        },
    }
    x, y, records = load_arrays(output, config, split=None)
    assert x.shape == (4, 16, 16, 3)
    assert set(y.tolist()) == {0, 1}
    assert len(records) == 4


def test_dry_run_writes_reports_but_not_database(tmp_path):
    source = tmp_path / "library"
    write_image(source / "one" / "image.jpg", (1, 2, 3))
    output = tmp_path / "dry.sqlite"

    summary = import_folders(options_for(source, output, "--dry-run"))

    assert summary["dry_run"]
    assert not output.exists()
    assert output.with_suffix(".import_report.json").exists()


def test_import_records_explicit_source_polarity(tmp_path):
    source = tmp_path / "library"
    write_image(source / "copepod" / "image.jpg", (10, 10, 10))
    output = tmp_path / "polarity.sqlite"

    summary = import_folders(
        options_for(source, output, "--source-polarity", "dark_on_light")
    )

    assert summary["imaging"]["polarity"] == {
        "value": "dark_on_light",
        "method": "explicit_cli",
        "confidence": 1.0,
        "model_polarity": "light_on_dark",
        "model_invert": True,
    }
    with sqlite3.connect(output) as connection:
        metadata = json.loads(
            connection.execute("SELECT metadata_json FROM dataset").fetchone()[0]
        )
    assert metadata["imaging"]["polarity"]["model_invert"] is True


def test_auto_polarity_detects_dark_objects_on_a_light_background(tmp_path):
    image = np.full((32, 32), 240, dtype="uint8")
    image[8:24, 8:24] = 20
    path = tmp_path / "brightfield.png"
    Image.fromarray(image).save(path)

    polarity = infer_source_polarity([path])

    assert polarity["value"] == "dark_on_light"
    assert polarity["confidence"] >= 0.65


def test_materialized_import_applies_requested_preprocessing(tmp_path):
    source = tmp_path / "library"
    write_image(source / "one" / "wide.png", (255, 255, 255), size=(30, 10))
    output = tmp_path / "materialized.sqlite"

    import_folders(
        options_for(
            source,
            output,
            "--storage-mode",
            "materialized",
            "--input-shape",
            "12",
            "12",
            "3",
            "--invert",
        )
    )

    with sqlite3.connect(output) as connection:
        blob, encoding, dimensions = connection.execute(
            """
            SELECT a.payload, a.encoding, a.shape_json
            FROM classification_items ci
            JOIN assets a ON a.asset_id = ci.image_asset_id
            """
        ).fetchone()
    array = decode_blob(blob, encoding, dimensions)
    assert array.shape == (12, 12, 3)
    assert array.dtype == np.float32
    assert np.allclose(array[4:8], 0.0)
    assert np.allclose(array[:3], 1.0)


def test_materialized_import_resolves_derived_channels_from_height_and_width(tmp_path):
    source = tmp_path / "library"
    image = np.zeros((12, 20), dtype="uint8")
    image[:, 10:] = 255
    path = source / "one" / "edge.png"
    path.parent.mkdir(parents=True)
    Image.fromarray(image).save(path)
    output = tmp_path / "materialized.sqlite"

    import_folders(
        options_for(
            source,
            output,
            "--storage-mode",
            "materialized",
            "--input-shape",
            "12",
            "12",
            "--gradient-magnitude",
            "--local-contrast",
        )
    )

    with sqlite3.connect(output) as connection:
        blob, encoding, dimensions = connection.execute(
            """
            SELECT a.payload, a.encoding, a.shape_json
            FROM classification_items ci
            JOIN assets a ON a.asset_id = ci.image_asset_id
            """
        ).fetchone()
    array = decode_blob(blob, encoding, dimensions)
    assert array.shape == (12, 12, 3)
    assert array.dtype == np.float32
    assert array[..., 1].max() > 0.0

    config = {
        "run": {"task": "classification", "seed": 123},
        "data": {"input_shape": [12, 12, 3]},
        "preprocessing": {
            "resize_mode": "fit_pad",
            "normalization": "dtype",
            "rescale": True,
            "invert": False,
            "pad_value": 0.0,
            "interpolation": "bilinear",
            "channel_mode": "grayscale",
            "derived_channels": {
                "gradient_magnitude": True,
                "local_contrast": True,
                "local_contrast_sigma": 3.0,
            },
        },
    }
    eager, _, _ = load_arrays(output, config)
    with sqlite3.connect(output) as connection:
        item_id = connection.execute("SELECT item_id FROM dataset_items").fetchone()[0]
    streamed = SQLiteClassificationSource(output, config).read_image(item_id)
    np.testing.assert_allclose(eager[0], array)
    np.testing.assert_allclose(streamed, array)


def test_existing_source_partitions_are_provenance_not_dataset_splits(tmp_path):
    source = tmp_path / "library"
    write_image(source / "train" / "cod" / "one.jpg", (1, 2, 3))
    write_image(source / "validation" / "cod" / "two.jpg", (2, 3, 4))
    write_image(source / "test" / "cod" / "three.jpg", (3, 4, 5))
    output = tmp_path / "split.sqlite"

    summary = import_folders(options_for(source, output))

    with sqlite3.connect(output) as connection:
        assert "split" not in {
            row[1]
            for row in connection.execute("PRAGMA table_info(dataset_items)")
        }
        partitions = {
            row[0]: json.loads(row[1])["source_partition"]
            for row in connection.execute(
                "SELECT source_key, metadata_json FROM dataset_items ORDER BY source_key"
            )
        }
    assert summary["counts_by_source_partition"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert partitions == {
        "test/cod/three.jpg": "test",
        "train/cod/one.jpg": "train",
        "validation/cod/two.jpg": "validation",
    }


def test_preprocessing_supports_resize_inversion_and_channel_conversion():
    array = np.zeros((10, 20), dtype="uint8")
    array[:, 5:15] = 255
    base = {
        "preprocessing": {
            "resize_mode": "fit_pad",
            "normalization": "dtype",
            "rescale": True,
            "invert": True,
            "pad_value": 0.0,
            "interpolation": "nearest",
            "channel_mode": "rgb",
        }
    }

    value = prepare_classification_input(array, [16, 16, 3], base)

    assert value.shape == (16, 16, 3)
    assert value.dtype == np.float32
    assert value.min() == 0
    assert value.max() == 1


def test_preprocessing_adds_gradient_and_local_contrast_channels_from_grayscale_roi():
    array = np.zeros((12, 12), dtype="uint8")
    array[3:9, 3:9] = 255
    config = {
        "preprocessing": {
            "resize_mode": "fit_pad",
            "normalization": "dtype",
            "rescale": True,
            "invert": False,
            "pad_value": 0.0,
            "interpolation": "nearest",
            "channel_mode": "grayscale",
            "derived_channels": {
                "gradient_magnitude": True,
                "local_contrast": True,
                "local_contrast_sigma": 1.5,
            },
        }
    }

    value = prepare_classification_input(array, [16, 16, 3], config)

    assert value.shape == (16, 16, 3)
    assert value.dtype == np.float32
    assert value[..., 0].max() == 1.0
    assert value[..., 1].max() == 1.0
    assert not np.array_equal(value[..., 0], value[..., 1])
    assert np.all((0.0 <= value[..., 2]) & (value[..., 2] <= 1.0))


def test_local_contrast_uses_the_configured_sigma():
    array = np.zeros((32, 32), dtype="uint8")
    array[8:24, 8:24] = 255
    base = {
        "resize_mode": "none",
        "normalization": "dtype",
        "rescale": True,
        "invert": False,
        "channel_mode": "grayscale",
        "interpolation": "nearest",
    }
    small_sigma = prepare_classification_input(
        array,
        [32, 32, 2],
        {
            "preprocessing": {
                **base,
                "derived_channels": {
                    "local_contrast": True,
                    "local_contrast_sigma": 1.0,
                },
            }
        },
    )
    large_sigma = prepare_classification_input(
        array,
        [32, 32, 2],
        {
            "preprocessing": {
                **base,
                "derived_channels": {
                    "local_contrast": True,
                    "local_contrast_sigma": 8.0,
                },
            }
        },
    )

    assert not np.allclose(small_sigma[..., 1], large_sigma[..., 1])


def test_capped_fit_pad_modes_limit_small_roi_upscaling_and_downsize_large_rois():
    config = {
        "preprocessing": {
            "resize_mode": "fit_pad_max_2x",
            "normalization": "dtype",
            "rescale": True,
            "pad_value": 0.0,
            "interpolation": "nearest",
            "channel_mode": "grayscale",
        }
    }

    small = prepare_classification_input(
        np.full((8, 8), 255, dtype="uint8"), [64, 64, 1], config
    )
    # A normal fit-pad operation would make this 64x64; the capped mode makes
    # it 16x16 and pads the remaining canvas.
    assert small.shape == (64, 64, 1)
    assert np.count_nonzero(small) == 16 * 16

    large = prepare_classification_input(
        np.full((100, 200), 255, dtype="uint8"), [64, 64, 1], config
    )
    # The cap applies only to enlargement. A 100x200 ROI still scales down to
    # 32x64 while preserving aspect ratio.
    assert large.shape == (64, 64, 1)
    assert np.count_nonzero(large) == 32 * 64

    config["preprocessing"]["resize_mode"] = "fit_pad_max_3x"
    three_x = prepare_classification_input(
        np.full((8, 8), 255, dtype="uint8"), [64, 64, 1], config
    )
    assert three_x.shape == (64, 64, 1)
    assert np.count_nonzero(three_x) == 24 * 24


def test_repeat_import_skips_existing_samples_and_preserves_labels(tmp_path):
    source = tmp_path / "library"
    write_image(source / "cod" / "one.jpg", (10, 20, 30))
    output = tmp_path / "repeat.sqlite"
    arguments = options_for(source, output)

    import_folders(arguments)
    second = import_folders(arguments)

    assert second["status_counts"] == {"skipped_existing": 1}
    with sqlite3.connect(output) as connection:
        assert connection.execute("SELECT count(*) FROM dataset_items").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM import_events").fetchone()[0] == 2
        assert connection.execute(
            "SELECT class_index, name FROM classification_labels"
        ).fetchall() == [(0, "cod")]


def test_duplicate_content_is_skipped_within_a_class(tmp_path):
    source = tmp_path / "library"
    write_image(source / "cod" / "one.png", (10, 20, 30))
    write_image(source / "cod" / "two.png", (10, 20, 30))
    output = tmp_path / "duplicates.sqlite"

    summary = import_folders(options_for(source, output))

    assert summary["status_counts"] == {"ready": 1, "skipped_duplicate": 1}
    with sqlite3.connect(output) as connection:
        assert connection.execute("SELECT count(*) FROM dataset_items").fetchone()[0] == 1


def test_new_classes_require_explicit_permission_when_appending(tmp_path):
    source = tmp_path / "library"
    write_image(source / "cod" / "one.jpg", (1, 2, 3))
    output = tmp_path / "append.sqlite"
    import_folders(options_for(source, output))
    write_image(source / "salmon" / "two.jpg", (3, 2, 1))

    try:
        import_folders(options_for(source, output))
    except ValueError as exc:
        assert "--allow-new-classes" in str(exc)
    else:
        raise AssertionError("Expected a new-class safeguard")

    import_folders(options_for(source, output, "--allow-new-classes"))
    with sqlite3.connect(output) as connection:
        assert connection.execute(
            "SELECT class_index, name FROM classification_labels ORDER BY class_index"
        ).fetchall() == [(0, "cod"), (1, "salmon")]


def test_import_does_not_materialize_split_assignments(tmp_path):
    source = tmp_path / "library"
    for index in range(10):
        write_image(source / "cod" / f"{index}.png", (index, 0, 0))
    output = tmp_path / "stable.sqlite"
    import_folders(
        options_for(source, output, "--duplicate-policy", "allow")
    )
    with sqlite3.connect(output) as connection:
        before = {
            row[0] for row in connection.execute("SELECT item_id FROM dataset_items")
        }
    write_image(source / "cod" / "new.png", (250, 1, 1))

    import_folders(
        options_for(
            source,
            output,
            "--existing-policy",
            "update",
            "--duplicate-policy",
            "allow",
        )
    )

    with sqlite3.connect(output) as connection:
        after = {
            row[0] for row in connection.execute("SELECT item_id FROM dataset_items")
        }
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(dataset_items)")
        }
    assert before.issubset(after)
    assert "split" not in columns
