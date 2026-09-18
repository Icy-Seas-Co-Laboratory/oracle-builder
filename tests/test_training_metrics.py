from __future__ import annotations

import json

import numpy as np
import tensorflow as tf
from tensorflow import keras

from oracle_builder.evaluation.classification import classification_epoch_metric_records
from oracle_builder.training.logging_callbacks import ClassificationEpochMetricsLogger
from oracle_builder.training.metrics import BinaryDice, SparseCategoricalMacroF1


def test_binary_dice_accumulates_confusion_counts_across_batches():
    metric = BinaryDice(threshold=0.5)
    metric.update_state(
        np.array([[[[1], [1], [0], [0]]]], dtype="float32"),
        np.array([[[[0.9], [0.2], [0.8], [0.1]]]], dtype="float32"),
    )

    assert np.isclose(float(metric.result()), 0.5)


def test_binary_dice_returns_one_for_two_empty_masks():
    metric = BinaryDice()
    metric.update_state(np.zeros((1, 2, 2, 1)), np.zeros((1, 2, 2, 1)))

    assert float(metric.result()) == 1.0


def test_sparse_categorical_macro_f1_accumulates_one_confusion_matrix_across_batches():
    metric = SparseCategoricalMacroF1(num_classes=3)
    metric.update_state(
        np.array([0, 1]),
        np.array([[0.9, 0.1, 0.0], [0.8, 0.1, 0.1]], dtype="float32"),
    )
    metric.update_state(
        np.array([2]),
        np.array([[0.0, 0.0, 1.0]], dtype="float32"),
    )

    # Class F1 values are 2/3, 0, and 1; macro F1 is their mean.
    assert np.isclose(float(metric.result()), 5.0 / 9.0)

    metric.reset_state()
    assert float(metric.result()) == 0.0


def test_epoch_classification_records_include_summary_and_per_class_metrics():
    records = classification_epoch_metric_records(
        np.array([0, 1, 1]),
        np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]], dtype="float32"),
        class_names={0: "copepod", 1: "diatom"},
    )

    assert {"accuracy", "macro_f1", "log_loss", "macro_roc_auc"} <= {
        row["metric"] for row in records
    }
    assert any(
        row["metric"] == "f1_score" and row.get("label") == "copepod"
        for row in records
    )
    assert any(
        row["metric"] == "confusion_count"
        and row.get("label") == "diatom"
        and row.get("predicted_label") == "copepod"
        for row in records
    )


def test_epoch_classification_logger_writes_train_and_validation_but_not_test(tmp_path):
    inputs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    labels = np.array([0, 1], dtype="int64")
    datasets = {
        split: tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(2)
        for split in ("train", "validation", "test")
    }
    model_input = keras.Input(shape=(2,))
    model = keras.Model(model_input, model_input)
    log_path = tmp_path / "logs" / "training.sqlite"
    callback = ClassificationEpochMetricsLogger(
        log_path, "run-1", datasets, {0: "copepod", 1: "diatom"}
    )
    callback.set_model(model)
    callback.on_epoch_end(0)

    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics" / "metrics.jsonl").read_text().splitlines()
    ]
    assert {row["split"] for row in rows} == {"train", "validation"}
    assert all(row["phase"] == "epoch_evaluation" for row in rows)
    assert any(row["metric"] == "macro_f1" for row in rows)
