from __future__ import annotations

import sqlite3

import numpy as np

from oracle_builder.data.sqlite_dataset import (
    create_synthetic_classification,
    create_synthetic_segmentation,
    load_arrays,
    load_prediction_arrays,
)
from oracle_builder.evaluation.predictions import write_predictions_db


class FakeClassificationModel:
    def predict(self, x, verbose=0):
        probabilities = np.zeros((len(x), 2), dtype="float32")
        probabilities[:, 0] = 0.75
        probabilities[:, 1] = 0.25
        return probabilities


class FakeSegmentationModel:
    def predict(self, x, verbose=0):
        return np.zeros((*x.shape[:3], 1), dtype="float32")


def test_prediction_database_copies_source_and_supports_multiple_sets(tmp_path):
    source = tmp_path / "training.sqlite"
    output = tmp_path / "training_with_predictions.sqlite"
    create_synthetic_classification(source, n=10, shape=(8, 8, 1), classes=2)
    config = {
        "run": {"task": "classification", "model": "simple_cnn", "seed": 123},
        "data": {
            "input_shape": [8, 8, 1],
            "num_classes": 2,
            "validation_split": 0.2,
            "test_split": 0.1,
        },
    }
    model = FakeClassificationModel()

    total = 0
    for split in ("train", "validation", "test"):
        x, y, records = load_arrays(source, config, split=split)
        total += len(records)
        write_predictions_db(
            model,
            x,
            y,
            records,
            config,
            output,
            source_sqlite=source,
            prediction_set="run-a",
        )
    x, y, records = load_arrays(source, config, split="test")
    write_predictions_db(
        model,
        x,
        y,
        records,
        config,
        output,
        source_sqlite=source,
        prediction_set="run-b",
    )

    with sqlite3.connect(output) as connection:
        assert connection.execute("SELECT count(*) FROM samples").fetchone()[0] == 10
        assert connection.execute("SELECT count(*) FROM predictions WHERE prediction_set = 'run-a'").fetchone()[0] == total
        assert connection.execute("SELECT count(*) FROM predictions WHERE prediction_set = 'run-b'").fetchone()[0] == 1
        assert connection.execute(
            "SELECT split, count(*) FROM predictions WHERE prediction_set = 'run-a' GROUP BY split ORDER BY split"
        ).fetchall() == [("test", 1), ("train", 7), ("validation", 2)]
        assert connection.execute("SELECT prediction_set FROM prediction_sets ORDER BY prediction_set").fetchall() == [
            ("run-a",),
            ("run-b",),
        ]


def test_all_roi_predictions_include_rows_without_validated_masks(tmp_path):
    source = tmp_path / "segmentation.sqlite"
    output = tmp_path / "segmentation_with_predictions.sqlite"
    create_synthetic_segmentation(source, n=5, shape=(8, 8, 1))
    with sqlite3.connect(source) as connection:
        missing_uuid = connection.execute("SELECT uuid FROM samples ORDER BY rowid LIMIT 1").fetchone()[0]
        connection.execute(
            "UPDATE samples SET output_blob = NULL, output_blob_encoding = NULL, output_blob_dimensions = NULL WHERE uuid = ?",
            (missing_uuid,),
        )
    config = {
        "run": {"task": "segmentation", "model": "unet", "seed": 123},
        "data": {
            "input_shape": [8, 8, 1],
            "output_shape": [8, 8, 1],
            "validation_split": 0.2,
            "test_split": 0.1,
        },
    }

    x, targets, records = load_prediction_arrays(source, config)
    write_predictions_db(
        FakeSegmentationModel(),
        x,
        targets,
        records,
        config,
        output,
        source_sqlite=source,
        prediction_set="all-rois",
    )

    with sqlite3.connect(output) as connection:
        assert connection.execute("SELECT count(*) FROM predictions").fetchone()[0] == 5
        assert connection.execute(
            "SELECT y_true_blob, metrics_json FROM predictions WHERE uuid = ?",
            (missing_uuid,),
        ).fetchone() == (None, None)
        assert connection.execute(
            "SELECT split FROM predictions WHERE uuid = ?",
            (missing_uuid,),
        ).fetchone()[0] in {"train", "validation", "test"}
