from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from oracle_builder.progress import BatchProgress


class ClassificationMetricAccumulator:
    def __init__(self, class_count: int, calibration_bins: int = 15):
        self.class_count = int(class_count)
        self.calibration_bins = int(calibration_bins)
        self.sample_count = 0
        self.log_loss_sum = 0.0
        self.brier_sum = 0.0
        self.top_correct = {k: 0 for k in (1, 3, 5)}
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

    def calibration_rows(self) -> list[dict[str, float | int]]:
        """Return reliability-diagram data without retaining individual scores."""
        rows = []
        for index, (count, confidence, correct) in enumerate(
            zip(self.bin_count, self.bin_confidence, self.bin_correct, strict=True)
        ):
            average_confidence = float(confidence / count) if count else None
            accuracy = float(correct / count) if count else None
            rows.append(
                {
                    "bin_index": index,
                    "lower_bound": index / self.calibration_bins,
                    "upper_bound": (index + 1) / self.calibration_bins,
                    "sample_count": int(count),
                    "average_confidence": average_confidence,
                    "accuracy": accuracy,
                    "absolute_gap": (
                        abs(accuracy - average_confidence)
                        if accuracy is not None and average_confidence is not None
                        else None
                    ),
                }
            )
        return rows


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
    evaluation_settings: dict[str, Any] | None = None,
    evaluation_context: dict[str, Any] | None = None,
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
            "metadata": row.get("metadata", {}),
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
        probabilities=probabilities,
        calibration_rows=probability_metrics.calibration_rows(),
        evaluation_settings=evaluation_settings,
        evaluation_context=evaluation_context,
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
    evaluation_settings: dict[str, Any] | None = None,
    evaluation_context: dict[str, Any] | None = None,
):
    targets = np.asarray([ref.target for ref in sample_index.refs], dtype="int64")
    predicted = np.empty(len(sample_index), dtype="int64")
    sample_rows: list[dict[str, Any] | None] = [None] * len(sample_index)
    probability_metrics = ClassificationMetricAccumulator(
        int(model.output_shape[-1])
    )
    evaluation_dir = Path(run_dir) / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    # Ranking metrics require every score, but evaluation should not turn the
    # streaming data path into an in-memory one.  A temporary .npy memmap keeps
    # memory bounded and is removed after the final report is written.
    score_path = evaluation_dir / ".classification_scores.npy"
    probabilities_by_sample = np.lib.format.open_memmap(
        score_path,
        mode="w+",
        dtype="float32",
        shape=(len(sample_index), int(model.output_shape[-1])),
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
        probabilities_by_sample[positions_array] = probabilities
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
                "metadata": ref.record()["metadata"],
            }
        display.update(len(positions_array))
    display.close()
    probabilities_by_sample.flush()
    try:
        return write_classification_evaluation(
            targets,
            predicted,
            [row for row in sample_rows if row is not None],
            run_dir,
            class_names=class_names,
            probability_metrics=probability_metrics.result(),
            probabilities=probabilities_by_sample,
            calibration_rows=probability_metrics.calibration_rows(),
            evaluation_settings=evaluation_settings,
            evaluation_context=evaluation_context,
        )
    finally:
        del probabilities_by_sample
        score_path.unlink(missing_ok=True)


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


def _plot_reliability(rows: list[dict[str, float | int]], path: Path) -> None:
    usable = [row for row in rows if row["sample_count"]]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#777777", label="Perfect calibration")
    if usable:
        confidence = [float(row["average_confidence"]) for row in usable]
        accuracy = [float(row["accuracy"]) for row in usable]
        ax.plot(confidence, accuracy, "o-", color="#377eb8", label="Model")
    else:
        ax.text(0.5, 0.5, "No predictions", ha="center", va="center")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted confidence", ylabel="Observed accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _safe_binary_metrics(targets: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    """One-vs-rest ranking metrics, with undefined values represented by null."""
    positives = int(targets.sum())
    negatives = int(len(targets) - positives)
    if not positives or not negatives:
        return {"average_precision": None, "roc_auc": None}
    return {
        "average_precision": float(average_precision_score(targets, scores)),
        "roc_auc": float(roc_auc_score(targets, scores)),
    }


def _binary_decision_metrics(targets: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    """Balanced accuracy and MCC for one label treated as positive."""
    positives = int(targets.sum())
    negatives = int(len(targets) - positives)
    if not positives or not negatives:
        return {"balanced_accuracy": None, "matthews_correlation_coefficient": None}
    true_positive = int(np.logical_and(targets, predicted).sum())
    true_negative = int(np.logical_and(~targets, ~predicted).sum())
    false_positive = int(np.logical_and(~targets, predicted).sum())
    false_negative = int(np.logical_and(targets, ~predicted).sum())
    balanced_accuracy = 0.5 * (
        true_positive / positives + true_negative / negatives
    )
    denominator = np.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = (
        (true_positive * true_negative - false_positive * false_negative) / denominator
        if denominator
        else 0.0
    )
    return {
        "balanced_accuracy": float(balanced_accuracy),
        "matthews_correlation_coefficient": float(mcc),
    }


def _ranking_metrics(
    targets: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray | None,
    labels: list[int],
) -> tuple[dict[int, dict[str, float | None]], dict[str, float | None]]:
    if probabilities is None:
        return {}, {}
    probabilities = np.asarray(probabilities, dtype="float64")
    per_class: dict[int, dict[str, float | None]] = {}
    valid_labels = [label for label in labels if 0 <= label < probabilities.shape[1]]
    for label in valid_labels:
        true_binary = targets == label
        predicted_binary = predicted == label
        per_class[label] = {
            **_safe_binary_metrics(true_binary, probabilities[:, label]),
            **_binary_decision_metrics(true_binary, predicted_binary),
        }

    def aggregate(metric: str, weighted: bool = False) -> float | None:
        values = []
        weights = []
        for label in valid_labels:
            value = per_class[label].get(metric)
            if value is not None:
                values.append(float(value))
                weights.append(float((targets == label).sum()))
        if not values:
            return None
        return float(np.average(values, weights=weights)) if weighted and sum(weights) else float(np.mean(values))

    aggregate_metrics: dict[str, float | None] = {}
    for metric in ("average_precision", "roc_auc"):
        aggregate_metrics[f"macro_{metric}"] = aggregate(metric)
        aggregate_metrics[f"weighted_{metric}"] = aggregate(metric, weighted=True)
    if len(valid_labels) > 1 and len(targets):
        score_columns = probabilities[:, valid_labels]
        true_columns = np.column_stack([targets == label for label in valid_labels])
        if true_columns.any() and (~true_columns).any():
            aggregate_metrics["micro_average_precision"] = float(
                average_precision_score(true_columns, score_columns, average="micro")
            )
            aggregate_metrics["micro_roc_auc"] = float(
                roc_auc_score(true_columns, score_columns, average="micro")
            )
    else:
        aggregate_metrics["micro_average_precision"] = None
        aggregate_metrics["micro_roc_auc"] = None
    return per_class, aggregate_metrics


def _nested_metadata_value(metadata: Any, key: str) -> Any:
    current = metadata if isinstance(metadata, dict) else {}
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _bootstrap_intervals(
    targets: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray | None,
    labels: list[int],
    sample_rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> pd.DataFrame:
    """Grouped bootstrap intervals; disabled unless a real provenance key is named."""
    if not settings.get("enabled", False):
        return pd.DataFrame(columns=["metric_name", "confidence_level", "ci_low", "ci_high", "replicates", "group_metadata_key"])
    group_key = settings.get("group_metadata_key")
    if not group_key:
        raise ValueError("evaluation.uncertainty.group_metadata_key is required when grouped bootstrap is enabled")
    groups = [_nested_metadata_value(row.get("metadata"), str(group_key)) for row in sample_rows]
    if any(value is None for value in groups):
        raise ValueError(f"Evaluation uncertainty group metadata key {group_key!r} is absent for one or more samples")
    group_values = np.asarray([str(value) for value in groups])
    unique_groups = np.unique(group_values)
    if len(unique_groups) < 2:
        raise ValueError("Grouped bootstrap requires at least two distinct metadata groups")
    replicates = int(settings.get("bootstrap_replicates", 1000))
    confidence_level = float(settings.get("confidence_level", 0.95))
    if replicates < 1 or not 0 < confidence_level < 1:
        raise ValueError("Bootstrap replicates must be positive and confidence_level must be between zero and one")
    rng = np.random.default_rng(int(settings.get("seed", 123)))
    values: dict[str, list[float]] = {"accuracy": [], "balanced_accuracy": [], "macro_f1": [], "macro_average_precision": []}
    indices_by_group = {group: np.flatnonzero(group_values == group) for group in unique_groups}
    for _ in range(replicates):
        selected_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        positions = np.concatenate([indices_by_group[group] for group in selected_groups])
        report = classification_report(targets[positions], predicted[positions], labels=labels, output_dict=True, zero_division=0)
        values["accuracy"].append(float(report.get("accuracy", 0.0)))
        values["balanced_accuracy"].append(float(report.get("macro avg", {}).get("recall", 0.0)))
        values["macro_f1"].append(float(report.get("macro avg", {}).get("f1-score", 0.0)))
        ranking = _ranking_metrics(targets[positions], predicted[positions], probabilities[positions] if probabilities is not None else None, labels)[1]
        if ranking.get("macro_average_precision") is not None:
            values["macro_average_precision"].append(float(ranking["macro_average_precision"]))
    alpha = (1.0 - confidence_level) / 2.0
    return pd.DataFrame(
        [
            {
                "metric_name": metric,
                "confidence_level": confidence_level,
                "ci_low": float(np.quantile(metric_values, alpha)),
                "ci_high": float(np.quantile(metric_values, 1.0 - alpha)),
                "replicates": replicates,
                "group_metadata_key": group_key,
            }
            for metric, metric_values in values.items()
            if metric_values
        ]
    )


def write_classification_evaluation(
    targets: np.ndarray,
    predicted: np.ndarray,
    sample_rows: list[dict[str, Any]],
    run_dir: str | Path,
    *,
    class_names: dict[int, str] | None = None,
    probability_metrics: dict[str, Any] | None = None,
    probabilities: np.ndarray | None = None,
    calibration_rows: list[dict[str, float | int]] | None = None,
    evaluation_settings: dict[str, Any] | None = None,
    evaluation_context: dict[str, Any] | None = None,
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

    calibration_rows = calibration_rows or []
    pd.DataFrame(calibration_rows).to_csv(
        evaluation_dir / "calibration_bins.csv", index=False
    )
    _plot_reliability(calibration_rows, figures_dir / "reliability_diagram.png")

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

    ranking_by_class, ranking_summary = _ranking_metrics(
        targets, predicted, probabilities, labels
    )
    class_rows = []
    for label, name in zip(labels, names, strict=True):
        values = report[str(label)]
        ranking = ranking_by_class.get(label, {})
        class_rows.append(
            {
                "class_index": label,
                "class_name": name,
                "support": int(values["support"]),
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1_score": float(values["f1-score"]),
                "average_precision": ranking.get("average_precision"),
                "roc_auc": ranking.get("roc_auc"),
                "one_vs_rest_balanced_accuracy": ranking.get("balanced_accuracy"),
                "one_vs_rest_matthews_correlation_coefficient": ranking.get("matthews_correlation_coefficient"),
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
        # For exclusive single-label classification, micro P/R/F1 are exactly
        # accuracy.  They are still written explicitly for comparison tooling.
        "micro_precision": float(report.get("accuracy", 0.0)),
        "micro_recall": float(report.get("accuracy", 0.0)),
        "micro_f1": float(report.get("accuracy", 0.0)),
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
        "decision_rule": "argmax",
        "probability_threshold": None,
    }
    summary.update(probability_metrics or {})
    summary.update(ranking_summary)

    context = evaluation_context or {}
    long_rows: list[dict[str, Any]] = []
    base = {
        "schema_name": "oracle_builder.classification_metrics",
        "schema_version": "1.0.0",
        "artifact_id": context.get("artifact_id"),
        "run_id": context.get("run_id"),
        "dataset_id": context.get("dataset_id"),
        "dataset_fingerprint_sha256": context.get("dataset_fingerprint_sha256"),
        "split": context.get("split"),
        "task": "classification",
        "decision_rule": "argmax",
        "threshold": None,
    }
    for row in class_rows:
        for metric_name in (
            "precision", "recall", "f1_score", "average_precision", "roc_auc",
            "one_vs_rest_balanced_accuracy", "one_vs_rest_matthews_correlation_coefficient",
        ):
            long_rows.append({
                **base,
                "metric_family": "ranking" if metric_name in {"average_precision", "roc_auc"} else "decision",
                "averaging": "one_vs_rest",
                "label": row["class_name"],
                "label_index": row["class_index"],
                "support": row["support"],
                "metric_name": metric_name,
                "value": row[metric_name],
            })
    for metric_name, value in summary.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if metric_name in {"sample_count", "class_count"}:
            continue
        averaging = "overall"
        if metric_name.startswith("macro_"):
            averaging = "macro"
        elif metric_name.startswith("micro_"):
            averaging = "micro"
        elif metric_name.startswith("weighted_"):
            averaging = "weighted"
        long_rows.append({
            **base,
            "metric_family": "ranking" if metric_name in {"macro_average_precision", "weighted_average_precision", "micro_average_precision", "macro_roc_auc", "weighted_roc_auc", "micro_roc_auc"} else ("calibration" if metric_name in {"log_loss", "multiclass_brier_score", "expected_calibration_error"} else "decision"),
            "averaging": averaging,
            "label": None,
            "label_index": None,
            "support": int(total),
            "metric_name": metric_name,
            "value": value,
        })
    uncertainty = _bootstrap_intervals(
        targets,
        predicted,
        probabilities,
        labels,
        sample_rows,
        (evaluation_settings or {}).get("uncertainty", {}),
    )
    uncertainty.to_csv(evaluation_dir / "metric_confidence_intervals.csv", index=False)
    if not uncertainty.empty:
        intervals = uncertainty.set_index("metric_name").to_dict("index")
        for row in long_rows:
            interval = intervals.get(row["metric_name"])
            if interval:
                row["confidence_level"] = interval["confidence_level"]
                row["ci_low"] = interval["ci_low"]
                row["ci_high"] = interval["ci_high"]
            else:
                row["confidence_level"] = row["ci_low"] = row["ci_high"] = None
    else:
        for row in long_rows:
            row["confidence_level"] = row["ci_low"] = row["ci_high"] = None
    pd.DataFrame(long_rows).to_csv(evaluation_dir / "metrics_long.csv", index=False)
    (evaluation_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return {"summary": summary, "predicted": predicted}


# Kept as a private alias for callers written against the initial streaming API.
_write_classification_evaluation = write_classification_evaluation
