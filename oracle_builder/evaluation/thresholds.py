from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oracle_builder.evaluation.segmentation import binary_metrics
from oracle_builder.evaluation.segmentation import predict_reassembled_segmentation
from oracle_builder.evaluation.segmentation_targets import CANDIDATE_DELTA, candidate_delta, reconstruct_validated_mask, segmentation_target_mode


def optimize_dice_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    true_samples = [np.asarray(value) > 0.5 for value in y_true]
    probability_samples = [np.asarray(value, dtype="float32") for value in probabilities]
    rows: list[dict[str, float]] = []
    for threshold_value in thresholds:
        threshold = float(threshold_value)
        predicted_samples = [value >= threshold for value in probability_samples]
        tp = float(sum(np.logical_and(target, prediction).sum() for target, prediction in zip(true_samples, predicted_samples, strict=False)))
        fp = float(sum(np.logical_and(~target, prediction).sum() for target, prediction in zip(true_samples, predicted_samples, strict=False)))
        fn = float(sum(np.logical_and(target, ~prediction).sum() for target, prediction in zip(true_samples, predicted_samples, strict=False)))
        denominator = 2 * tp + fp + fn
        aggregate_dice = (2 * tp / denominator) if denominator else 1.0
        sample_dice = [
            binary_metrics(target, prediction, threshold=threshold)["dice"]
            for target, prediction in zip(true_samples, probability_samples, strict=False)
        ]
        rows.append(
            {
                "threshold": threshold,
                "aggregate_dice": aggregate_dice,
                "mean_sample_dice": float(np.mean(sample_dice)) if sample_dice else 1.0,
            }
        )
    best = max(rows, key=lambda row: (row["aggregate_dice"], -abs(row["threshold"] - 0.5)))
    return {
        "objective": "aggregate_dice",
        "best_threshold": best["threshold"],
        "best_aggregate_dice": best["aggregate_dice"],
        "best_mean_sample_dice": best["mean_sample_dice"],
        "thresholds_evaluated": len(rows),
        "curve": rows,
    }


def analyze_validation_threshold(
    model,
    x: np.ndarray,
    y: np.ndarray,
    run_dir: str | Path,
    config: dict[str, Any] | None = None,
    records: list[dict[str, Any]] | None = None,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    if config is not None and records is not None:
        probabilities, targets, records = predict_reassembled_segmentation(
            model, x, y, records, config
        )
        y = targets
    else:
        probabilities = model.predict(x, verbose=0)
    if config is not None and segmentation_target_mode(config) == CANDIDATE_DELTA:
        if not records:
            raise ValueError("Candidate-delta threshold analysis requires validation records")
        candidates = [record["candidate_mask"] for record in records]
        validated = [record["validated_mask"] for record in records]
        analysis = optimize_delta_threshold(validated, candidates, probabilities, thresholds=thresholds)
    else:
        analysis = optimize_dice_threshold(y, probabilities, thresholds=thresholds)
    evaluation_dir = Path(run_dir) / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "validation_threshold_analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n"
    )
    pd.DataFrame(analysis["curve"]).to_csv(
        evaluation_dir / "validation_threshold_curve.csv", index=False
    )
    return analysis


def optimize_delta_threshold(
    validated: np.ndarray,
    candidates: np.ndarray,
    delta_probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    validated_masks = [np.asarray(value) > 0.5 for value in validated]
    candidate_masks = [np.asarray(value) > 0.5 for value in candidates]
    probability_samples = [np.asarray(value, dtype="float32") for value in delta_probabilities]
    true_deltas = [
        candidate_delta(candidate, target)
        for candidate, target in zip(candidate_masks, validated_masks, strict=False)
    ]
    rows: list[dict[str, float]] = []
    for threshold_value in thresholds:
        threshold = float(threshold_value)
        predicted_deltas = [value >= threshold for value in probability_samples]
        reconstructed = [
            reconstruct_validated_mask(candidate, delta)
            for candidate, delta in zip(candidate_masks, predicted_deltas, strict=False)
        ]
        reconstructed_metrics = _aggregate_binary_metrics(validated_masks, reconstructed)
        delta_metrics = _aggregate_binary_metrics(true_deltas, predicted_deltas)
        sample_dice = [
            binary_metrics(target, prediction)["dice"]
            for target, prediction in zip(validated_masks, reconstructed, strict=False)
        ]
        rows.append(
            {
                "threshold": threshold,
                "aggregate_dice": reconstructed_metrics["dice"],
                "mean_sample_dice": float(np.mean(sample_dice)) if sample_dice else 1.0,
                "delta_dice": delta_metrics["dice"],
            }
        )
    best = max(rows, key=lambda row: (row["aggregate_dice"], -abs(row["threshold"] - 0.5)))
    return {
        "objective": "reconstructed_validated_aggregate_dice",
        "target_mode": CANDIDATE_DELTA,
        "best_threshold": best["threshold"],
        "best_aggregate_dice": best["aggregate_dice"],
        "best_mean_sample_dice": best["mean_sample_dice"],
        "best_delta_dice": best["delta_dice"],
        "thresholds_evaluated": len(rows),
        "curve": rows,
    }


def _aggregate_binary_metrics(targets, predictions) -> dict[str, float]:
    tp = float(sum(np.logical_and(target, prediction).sum() for target, prediction in zip(targets, predictions, strict=False)))
    fp = float(sum(np.logical_and(~np.asarray(target, dtype=bool), prediction).sum() for target, prediction in zip(targets, predictions, strict=False)))
    fn = float(sum(np.logical_and(target, ~np.asarray(prediction, dtype=bool)).sum() for target, prediction in zip(targets, predictions, strict=False)))
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}
