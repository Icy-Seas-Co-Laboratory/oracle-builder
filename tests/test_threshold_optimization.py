from __future__ import annotations

import json

import numpy as np

from oracle_builder.evaluation.thresholds import analyze_validation_threshold, optimize_dice_threshold


class FakeModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, x, verbose=0):
        return self.predictions


def test_dice_threshold_optimizer_finds_best_probability_cutoff():
    y_true = np.array([[[[1], [1], [0], [0]]]], dtype="float32")
    probabilities = np.array([[[[0.8], [0.6], [0.55], [0.1]]]], dtype="float32")

    result = optimize_dice_threshold(
        y_true,
        probabilities,
        thresholds=np.array([0.5, 0.6, 0.7]),
    )

    assert result["best_threshold"] == 0.6
    assert result["best_aggregate_dice"] == 1.0
    assert result["thresholds_evaluated"] == 3


def test_threshold_optimizer_breaks_ties_toward_half():
    y_true = np.zeros((1, 2, 2, 1), dtype="float32")
    probabilities = np.zeros_like(y_true)

    result = optimize_dice_threshold(
        y_true,
        probabilities,
        thresholds=np.array([0.2, 0.5, 0.8]),
    )

    assert result["best_threshold"] == 0.5


def test_validation_threshold_analysis_writes_json_and_csv(tmp_path):
    y_true = np.array([[[[1], [0]]]], dtype="float32")
    probabilities = np.array([[[[0.7], [0.2]]]], dtype="float32")

    result = analyze_validation_threshold(
        FakeModel(probabilities),
        np.zeros_like(y_true),
        y_true,
        tmp_path,
        thresholds=np.array([0.5, 0.8]),
    )

    saved = json.loads((tmp_path / "evaluation" / "validation_threshold_analysis.json").read_text())
    assert saved["best_threshold"] == result["best_threshold"] == 0.5
    assert (tmp_path / "evaluation" / "validation_threshold_curve.csv").exists()
