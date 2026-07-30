from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from oracle_builder.progress import BatchProgress


class ClassificationMetricAccumulator:
    def __init__(self, class_count: int, calibration_bins: int = 15):
        self.class_count = int(class_count)
        self.calibration_bins = int(calibration_bins)
        self.sample_count = 0
        self.log_loss_sum = 0.0
        self.brier_sum = 0.0
        self.top_correct = {k: 0 for k in (3, 5)}
        self.bin_count = np.zeros(self.calibration_bins, dtype="int64")
        self.bin_confidence = np.zeros(self.calibration_bins, dtype="float64")
        self.bin_correct = np.zeros(self.calibration_bins, dtype="float64")

    def update(self, targets: np.ndarray, probabilities: np.ndarray) -> None:
        targets = np.asarray(targets, dtype="int64")
        probabilities = np.asarray(probabilities, dtype="float64")
        rows = np.arange(len(targets))
        true_probabilities = np.clip(
            probabilities[rows, targets], 1e-12, 1.0
        )
        self.log_loss_sum += float(-np.log(true_probabilities).sum())
        one_hot = np.eye(self.class_count, dtype="float64")[targets]
        self.brier_sum += float(np.square(probabilities - one_hot).sum())
        for requested in self.top_correct:
            k = min(requested, self.class_count)
            top = np.argpartition(probabilities, -k, axis=1)[:, -k:]
            self.top_correct[requested] += int(
                np.any(top == targets[:, None], axis=1).sum()
            )
        confidence = probabilities.max(axis=1)
        predicted = probabilities.argmax(axis=1)
        correct = predicted == targets
        bins = np.minimum(
            (confidence * self.calibration_bins).astype(int),
            self.calibration_bins - 1,
        )
        for index in range(self.calibration_bins):
            selected = bins == index
            self.bin_count[index] += int(selected.sum())
            self.bin_confidence[index] += float(confidence[selected].sum())
            self.bin_correct[index] += float(correct[selected].sum())
        self.sample_count += len(targets)

    def result(self) -> dict[str, float | int]:
        count = max(self.sample_count, 1)
        calibration_error = 0.0
        for bin_count, confidence, correct in zip(
            self.bin_count,
            self.bin_confidence,
            self.bin_correct,
            strict=True,
        ):
            if bin_count:
                calibration_error += (
                    bin_count
                    / count
                    * abs(correct / bin_count - confidence / bin_count)
                )
        result: dict[str, float | int] = {
            "sample_count": self.sample_count,
            "log_loss": self.log_loss_sum / count,
            "multiclass_brier_score": self.brier_sum / count,
            "expected_calibration_error": float(calibration_error),
        }
        for requested, correct in self.top_correct.items():
            result[f"top_{min(requested, self.class_count)}_accuracy"] = (
                correct / count
            )
        return result


def evaluate_classification(
    model,
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict[str, Any]],
    run_dir: str | Path,
    *,
    class_names: dict[int, str] | None = None,
    batch_size: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    probabilities = model.predict(
        x,
        batch_size=batch_size,
        verbose=1 if progress else 0,
    )
    probability_metrics = ClassificationMetricAccumulator(
        probabilities.shape[1]
    )
    probability_metrics.update(y, probabilities)
    predicted = np.argmax(probabilities, axis=1)
    sample_rows = [
        {
            "uuid": row["uuid"],
            "split": row["split"],
            "y_true": int(true_value),
            "y_pred": int(pred_value),
            "correct": bool(int(true_value) == int(pred_value)),
            "confidence": float(np.max(probs)),
        }
        for row, true_value, pred_value, probs in zip(
            records, y, predicted, probabilities, strict=False
        )
    ]
    result = write_classification_evaluation(
        np.asarray(y, dtype="int64"),
        predicted,
        sample_rows,
        run_dir,
        class_names=class_names,
        probability_metrics=probability_metrics.result(),
    )
    result["probabilities"] = probabilities
    return result


def evaluate_classification_streaming(
    model,
    dataset,
    sample_index,
    run_dir: str | Path,
    *,
    class_names: dict[int, str] | None = None,
    progress: bool = True,
):
    targets = np.asarray([ref.target for ref in sample_index.refs], dtype="int64")
    predicted = np.empty(len(sample_index), dtype="int64")
    sample_rows: list[dict[str, Any] | None] = [None] * len(sample_index)
    probability_metrics = ClassificationMetricAccumulator(
        int(model.output_shape[-1])
    )
    display = BatchProgress(
        "Evaluating classification",
        len(sample_index),
        enabled=progress,
    )
    for images, positions in dataset:
        probabilities = np.asarray(model(images, training=False))
        positions_array = np.asarray(positions, dtype="int64")
        batch_predicted = np.argmax(probabilities, axis=1)
        probability_metrics.update(targets[positions_array], probabilities)
        predicted[positions_array] = batch_predicted
        for position, pred_value, probs in zip(
            positions_array, batch_predicted, probabilities, strict=True
        ):
            ref = sample_index.refs[int(position)]
            sample_rows[int(position)] = {
                "uuid": ref.uuid,
                "split": ref.split,
                "y_true": int(ref.target),
                "y_pred": int(pred_value),
                "correct": bool(int(ref.target) == int(pred_value)),
                "confidence": float(np.max(probs)),
            }
        display.update(len(positions_array))
    display.close()
    return write_classification_evaluation(
        targets,
        predicted,
        [row for row in sample_rows if row is not None],
        run_dir,
        class_names=class_names,
        probability_metrics=probability_metrics.result(),
    )


def _display_names(
    labels: list[int], class_names: dict[int, str] | None
) -> list[str]:
    names = class_names or {}
    return [str(names.get(label, label)) for label in labels]


def _tick_positions(size: int, maximum: int = 30) -> np.ndarray:
    if size <= maximum:
        return np.arange(size)
    return np.unique(np.linspace(0, size - 1, maximum, dtype=int))


def _plot_matrix(
    matrix: np.ndarray,
    names: list[str],
    path: Path,
    *,
    title: str,
    colorbar_label: str,
    annotate: bool,
) -> None:
    size = len(names)
    side = min(24.0, max(7.0, size * 0.09))
    fig, ax = plt.subplots(figsize=(side, side))
    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ticks = _tick_positions(size)
    ax.set_xticks(ticks, labels=[names[index] for index in ticks])
    ax.set_yticks(ticks, labels=[names[index] for index in ticks])
    ax.tick_params(axis="x", labelrotation=90, labelsize=6 if size > 40 else 8)
    ax.tick_params(axis="y", labelsize=6 if size > 40 else 8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    if annotate:
        threshold = float(np.max(matrix)) / 2.0 if matrix.size else 0.0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                label = (
                    f"{value:.2f}"
                    if np.issubdtype(matrix.dtype, np.floating)
                    else str(int(value))
                )
                ax.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > threshold else "black",
                )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _top_confusions(
    matrix: np.ndarray,
    labels: list[int],
    names: list[str],
    *,
    limit: int = 100,
) -> pd.DataFrame:
    row_totals = matrix.sum(axis=1)
    rows = []
    for true_index, predicted_index in zip(
        *np.nonzero(matrix - np.diag(np.diag(matrix)))
    ):
        count = int(matrix[true_index, predicted_index])
        rows.append(
            {
                "true_class_index": int(labels[true_index]),
                "true_class": names[true_index],
                "predicted_class_index": int(labels[predicted_index]),
                "predicted_class": names[predicted_index],
                "count": count,
                "fraction_of_true_class": (
                    float(count / row_totals[true_index])
                    if row_totals[true_index]
                    else 0.0
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["count"],
            -row["fraction_of_true_class"],
            row["true_class"],
            row["predicted_class"],
        )
    )
    return pd.DataFrame(rows[:limit])


def _plot_top_confusions(rows: pd.DataFrame, path: Path) -> None:
    shown = rows.head(30).iloc[::-1]
    if shown.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No off-diagonal confusions", ha="center", va="center")
        ax.axis("off")
    else:
        height = min(16.0, max(4.0, len(shown) * 0.36))
        fig, ax = plt.subplots(figsize=(11, height))
        labels = [
            f"{row.true_class} → {row.predicted_class}"
            for row in shown.itertuples()
        ]
        ax.barh(labels, shown["count"], color="#377eb8")
        ax.set_xlabel("Misclassified samples")
        ax.set_title("Most frequent class confusions")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_classification_evaluation(
    targets: np.ndarray,
    predicted: np.ndarray,
    sample_rows: list[dict[str, Any]],
    run_dir: str | Path,
    *,
    class_names: dict[int, str] | None = None,
    probability_metrics: dict[str, Any] | None = None,
):
    labels = sorted(
        set(targets.tolist())
        | set(predicted.tolist())
        | set((class_names or {}).keys())
    )
    names = _display_names(labels, class_names)
    matrix = confusion_matrix(targets, predicted, labels=labels)
    report = classification_report(
        targets,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    evaluation_dir = Path(run_dir) / "evaluation"
    figures_dir = Path(run_dir) / "figures"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(matrix, index=names, columns=names).to_csv(
        evaluation_dir / "confusion_matrix.csv"
    )
    (evaluation_dir / "confusion_matrix.json").write_text(
        json.dumps(
            {"labels": labels, "class_names": names, "matrix": matrix.tolist()},
            indent=2,
        )
        + "\n"
    )
    (evaluation_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    pd.DataFrame(sample_rows).to_csv(
        evaluation_dir / "sample_metrics.csv", index=False
    )

    normalized = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype="float64"),
        where=matrix.sum(axis=1, keepdims=True) > 0,
    )
    pd.DataFrame(normalized, index=names, columns=names).to_csv(
        evaluation_dir / "confusion_matrix_normalized.csv"
    )
    _plot_matrix(
        normalized,
        names,
        figures_dir / "confusion_matrix.png",
        title="Confusion matrix normalized by true class",
        colorbar_label="Fraction of true class",
        annotate=len(labels) <= 20,
    )
    _plot_matrix(
        np.log1p(matrix),
        names,
        figures_dir / "confusion_matrix_counts_log.png",
        title="Confusion matrix counts (logₑ(1 + count))",
        colorbar_label="logₑ(1 + count)",
        annotate=False,
    )

    confusion_rows = _top_confusions(matrix, labels, names)
    confusion_rows.to_csv(evaluation_dir / "top_confusions.csv", index=False)
    _plot_top_confusions(confusion_rows, figures_dir / "top_confusions.png")

    class_rows = []
    for label, name in zip(labels, names, strict=True):
        values = report[str(label)]
        class_rows.append(
            {
                "class_index": label,
                "class_name": name,
                "support": int(values["support"]),
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1_score": float(values["f1-score"]),
            }
        )
    pd.DataFrame(class_rows).sort_values(
        ["f1_score", "support"], ascending=[True, False]
    ).to_csv(evaluation_dir / "per_class_metrics.csv", index=False)

    row_totals = matrix.sum(axis=1).astype("float64")
    column_totals = matrix.sum(axis=0).astype("float64")
    total = float(matrix.sum())
    trace = float(np.trace(matrix))
    expected = (
        float(np.dot(row_totals, column_totals)) / total**2 if total else 0.0
    )
    observed = trace / total if total else 0.0
    kappa = (
        (observed - expected) / (1.0 - expected)
        if total and expected < 1.0
        else 0.0
    )
    mcc_denominator = np.sqrt(
        max(total**2 - float(np.dot(column_totals, column_totals)), 0.0)
        * max(total**2 - float(np.dot(row_totals, row_totals)), 0.0)
    )
    mcc = (
        (trace * total - float(np.dot(row_totals, column_totals)))
        / mcc_denominator
        if mcc_denominator
        else 0.0
    )
    summary = {
        "task": "classification",
        "class_count": len(labels),
        "sample_count": int(total),
        "accuracy": float(report.get("accuracy", 0.0)),
        "balanced_accuracy": float(
            report.get("macro avg", {}).get("recall", 0.0)
        ),
        "macro_precision": float(
            report.get("macro avg", {}).get("precision", 0.0)
        ),
        "macro_recall": float(
            report.get("macro avg", {}).get("recall", 0.0)
        ),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_precision": float(
            report.get("weighted avg", {}).get("precision", 0.0)
        ),
        "weighted_recall": float(
            report.get("weighted avg", {}).get("recall", 0.0)
        ),
        "weighted_f1": float(
            report.get("weighted avg", {}).get("f1-score", 0.0)
        ),
        "cohen_kappa": float(kappa),
        "matthews_correlation_coefficient": float(mcc),
        "confusion_matrix_representation": (
            "annotated_normalized" if len(labels) <= 20 else "sparse_normalized"
        ),
    }
    summary.update(probability_metrics or {})
    (evaluation_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return {"summary": summary, "predicted": predicted}


# Kept as a private alias for callers written against the initial streaming API.
_write_classification_evaluation = write_classification_evaluation
