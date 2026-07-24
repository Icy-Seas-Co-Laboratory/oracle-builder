from __future__ import annotations

import json
import sqlite3

import numpy as np

from oracle_builder.config import validate_config
from oracle_builder.data.decoders import decode_blob
from oracle_builder.data.sqlite_dataset import load_arrays, load_prediction_arrays
from oracle_builder.evaluation.predictions import write_predictions_db
from oracle_builder.evaluation.segmentation_targets import candidate_delta, reconstruct_validated_mask, reconstruct_validated_probability
from oracle_builder.evaluation.thresholds import optimize_delta_threshold
from oracle_builder.masking.sqlite_io import create_or_update_image_sample, open_database, save_mask_annotation


def delta_config():
    return {
        "run": {"task": "segmentation", "model": "unet", "seed": 123},
        "data": {
            "input_shape": [8, 8, 2],
            "output_shape": [8, 8, 1],
            "validation_split": 0.0,
            "test_split": 0.0,
        },
        "training": {"loss": "bce_soft_dice", "segmentation_target": "candidate_delta"},
    }


def create_delta_sample(path):
    image = np.arange(64, dtype="uint8").reshape(8, 8)
    candidate = np.zeros((8, 8), dtype="uint8")
    candidate[1:6, 1:6] = 1
    validated = candidate.copy()
    validated[2:4, 2:4] = 0
    validated[5:7, 5:7] = 1
    with open_database(path) as connection:
        create_or_update_image_sample(connection, "sample", image, "png", {}, candidate_mask=candidate)
        save_mask_annotation(connection, "sample", validated, "png", "test", {}, {"valid": True})
    return candidate, validated


def test_candidate_delta_target_and_reconstruction_are_exact(tmp_path):
    database = tmp_path / "delta.sqlite"
    candidate, validated = create_delta_sample(database)

    _x, y, records = load_arrays(database, delta_config())

    expected = np.logical_xor(candidate > 0, validated > 0)
    assert np.array_equal(y[0, ..., 0] > 0, expected)
    assert np.array_equal(records[0]["candidate_mask"][..., 0] > 0, candidate > 0)
    assert np.array_equal(reconstruct_validated_mask(records[0]["candidate_mask"], y[0]) > 0, validated[..., None] > 0)


def test_reconstructed_probability_inverts_delta_probability_inside_candidate():
    candidate = np.array([[0, 1]], dtype="float32")
    delta_probability = np.array([[0.2, 0.2]], dtype="float32")

    reconstructed = reconstruct_validated_probability(candidate, delta_probability)

    assert np.allclose(reconstructed, [[0.2, 0.8]])


def test_delta_threshold_optimization_uses_reconstructed_mask_dice():
    candidate = np.array([[[[1], [1], [0], [0]]]], dtype="float32")
    validated = np.array([[[[1], [0], [1], [0]]]], dtype="float32")
    probabilities = np.array([[[[0.1], [0.7], [0.6], [0.4]]]], dtype="float32")

    result = optimize_delta_threshold(
        validated, candidate, probabilities, thresholds=np.array([0.5, 0.8])
    )

    assert result["best_threshold"] == 0.5
    assert result["best_aggregate_dice"] == 1.0
    assert result["target_mode"] == "candidate_delta"


def test_delta_config_requires_two_input_channels():
    config = delta_config()
    config["data"]["input_shape"] = [8, 8, 1]

    try:
        validate_config(config)
    except ValueError as exc:
        assert "2 channels" in str(exc)
    else:
        raise AssertionError("Expected candidate_delta config validation to fail")


class FakeDeltaModel:
    def predict(self, x, verbose=0):
        return np.full((len(x), 8, 8, 1), 0.25, dtype="float32")


def test_delta_prediction_database_stores_raw_and_reconstructed_outputs(tmp_path):
    source = tmp_path / "source.sqlite"
    output = tmp_path / "predictions.sqlite"
    create_delta_sample(source)
    config = delta_config()
    x, targets, records = load_prediction_arrays(source, config)

    write_predictions_db(
        FakeDeltaModel(), x, targets, records, config, output,
        source_sqlite=source, prediction_set="delta-run"
    )

    with sqlite3.connect(output) as connection:
        row = connection.execute(
            "SELECT target_mode, y_pred_blob, y_pred_encoding, reconstructed_pred_blob, reconstructed_pred_encoding, metrics_json FROM predictions"
        ).fetchone()
    raw = decode_blob(row[1], row[2])
    reconstructed = decode_blob(row[3], row[4])
    metrics = json.loads(row[5])
    assert row[0] == "candidate_delta"
    assert np.allclose(raw, 0.25)
    assert not np.array_equal(raw, reconstructed)
    assert "delta_dice" in metrics
    assert "candidate_dice" in metrics
