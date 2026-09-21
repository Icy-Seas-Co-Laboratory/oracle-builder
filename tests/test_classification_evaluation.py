from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("sklearn")

from oracle_builder.evaluation.classification import (
    ClassificationMetricAccumulator,
    plot_classification_roi_size_metrics,
    plot_classification_training_metrics,
    write_classification_evaluation,
)


def test_default_classification_plots_include_epoch_and_roi_size_metrics(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    records = [
        {"phase": "epoch_evaluation", "epoch": epoch, "split": "validation", "metric": metric, "value": value, **({"label": "copepod"} if metric in {"recall", "f1_score"} else {})}
        for epoch, value in enumerate((0.5, 0.7))
        for metric in ("accuracy", "weighted_recall", "top_3_accuracy", "macro_f1", "weighted_f1", "log_loss", "recall", "f1_score")
    ]
    (metrics_dir / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in records))
    plot_classification_training_metrics(
        {"loss": [1.0, 0.5], "val_loss": [1.2, 0.6], "macro_f1": [0.4, 0.6]},
        tmp_path,
    )
    for filename in ("classification_accuracy_by_epoch.png", "classification_f1_by_epoch.png", "classification_loss_by_epoch.png"):
        assert (tmp_path / "figures" / filename).exists()

    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    pd.DataFrame(
        {"roi_area_px": [10, 20, 30, 40, 50], "y_true": [0, 0, 1, 1, 1], "y_pred": [0, 1, 1, 1, 0]}
    ).to_csv(evaluation_dir / "sample_metrics.csv", index=False)
    plot_classification_roi_size_metrics(tmp_path)
    assert (evaluation_dir / "roi_size_metrics.csv").exists()
    assert (tmp_path / "figures" / "classification_accuracy_by_roi_size.png").exists()
    assert (tmp_path / "figures" / "classification_f1_by_roi_size.png").exists()


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


def test_top_k_metric_names_remain_stable_for_small_classifiers():
    accumulator = ClassificationMetricAccumulator(2)
    accumulator.update(
        np.asarray([0, 1]),
        np.asarray([[0.8, 0.2], [0.3, 0.7]]),
    )
    result = accumulator.result()
    assert result["top_3_accuracy"] == 1.0
    assert result["top_5_accuracy"] == 1.0


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
