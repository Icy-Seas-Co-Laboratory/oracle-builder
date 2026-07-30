from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("sklearn")

from oracle_builder.evaluation.classification import (
    ClassificationMetricAccumulator,
    write_classification_evaluation,
)


def test_large_class_confusion_matrix_writes_scalable_outputs(tmp_path):
    class_count = 120
    targets = np.repeat(np.arange(class_count), 2)
    predicted = targets.copy()
    predicted[::7] = (predicted[::7] + 1) % class_count
    names = {index: f"class-{index:03d}" for index in range(class_count)}
    sample_rows = [
        {
            "uuid": str(index),
            "split": "test",
            "y_true": int(true),
            "y_pred": int(prediction),
            "correct": bool(true == prediction),
            "confidence": 0.9,
        }
        for index, (true, prediction) in enumerate(
            zip(targets, predicted, strict=True)
        )
    ]

    result = write_classification_evaluation(
        targets,
        predicted,
        sample_rows,
        tmp_path,
        class_names=names,
    )

    assert result["summary"]["class_count"] == class_count
    assert result["summary"]["confusion_matrix_representation"] == (
        "sparse_normalized"
    )
    normalized = pd.read_csv(
        tmp_path / "evaluation" / "confusion_matrix_normalized.csv",
        index_col=0,
    )
    assert normalized.shape == (class_count, class_count)
    assert np.allclose(normalized.sum(axis=1), 1.0)
    confusions = pd.read_csv(
        tmp_path / "evaluation" / "top_confusions.csv"
    )
    assert len(confusions) > 0
    assert confusions.iloc[0]["count"] >= 1
    with Image.open(tmp_path / "figures" / "confusion_matrix.png") as image:
        assert image.width > 1000
        assert image.height > 1000
    assert (tmp_path / "figures" / "top_confusions.png").exists()
    payload = json.loads(
        (tmp_path / "evaluation" / "confusion_matrix.json").read_text()
    )
    assert payload["class_names"][0] == "class-000"


def test_probability_metrics_include_ranking_calibration_and_proper_scores():
    accumulator = ClassificationMetricAccumulator(3)
    targets = np.array([0, 1, 2])
    probabilities = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )

    accumulator.update(targets, probabilities)
    result = accumulator.result()

    assert result["sample_count"] == 3
    assert result["top_3_accuracy"] == 1.0
    assert result["log_loss"] > 0
    assert result["multiclass_brier_score"] > 0
    assert 0 <= result["expected_calibration_error"] <= 1
