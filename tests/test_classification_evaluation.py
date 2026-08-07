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


def test_evaluation_writes_ranking_calibration_and_canonical_metric_tables(tmp_path):
    targets = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.1, 0.8],
            [0.6, 0.1, 0.3],
        ]
    )
    predicted = probabilities.argmax(axis=1)
    accumulator = ClassificationMetricAccumulator(3)
    accumulator.update(targets, probabilities)
    rows = [
        {
            "uuid": str(index),
            "split": "test",
            "y_true": int(target),
            "y_pred": int(prediction),
            "correct": bool(target == prediction),
            "confidence": float(score.max()),
            "metadata": {"cruise_id": "one" if index < 3 else "two"},
        }
        for index, (target, prediction, score) in enumerate(
            zip(targets, predicted, probabilities, strict=True)
        )
    ]

    result = write_classification_evaluation(
        targets,
        predicted,
        rows,
        tmp_path,
        class_names={0: "a", 1: "b", 2: "c"},
        probability_metrics=accumulator.result(),
        probabilities=probabilities,
        calibration_rows=accumulator.calibration_rows(),
        evaluation_context={"run_id": "run-1", "dataset_id": "dataset-1", "split": "test"},
    )

    assert result["summary"]["top_1_accuracy"] == result["summary"]["accuracy"]
    assert result["summary"]["macro_average_precision"] is not None
    assert result["summary"]["macro_roc_auc"] is not None
    per_class = pd.read_csv(tmp_path / "evaluation" / "per_class_metrics.csv")
    assert {"average_precision", "roc_auc"}.issubset(per_class.columns)
    metrics = pd.read_csv(tmp_path / "evaluation" / "metrics_long.csv")
    assert {"artifact_id", "run_id", "dataset_id", "metric_family", "metric_name"}.issubset(metrics.columns)
    assert "macro_average_precision" in set(metrics["metric_name"])
    assert (tmp_path / "evaluation" / "calibration_bins.csv").exists()
    assert (tmp_path / "figures" / "reliability_diagram.png").exists()
