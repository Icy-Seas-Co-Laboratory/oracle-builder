from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oracle_builder.evaluation.segmentation import binary_metrics
from oracle_builder.evaluation.segmentation_targets import CANDIDATE_DELTA, candidate_delta, reconstruct_validated_mask, segmentation_target_mode


def optimize_dice_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    true_mask = np.asarray(y_true) > 0.5
    probabilities = np.asarray(probabilities, dtype="float32")
    rows: list[dict[str, float]] = []
    for threshold_value in thresholds:
        threshold = float(threshold_value)
        pred_mask = probabilities >= threshold
        tp = float(np.logical_and(true_mask, pred_mask).sum())
        fp = float(np.logical_and(~true_mask, pred_mask).sum())
        fn = float(np.logical_and(true_mask, ~pred_mask).sum())
        denominator = 2 * tp + fp + fn
        aggregate_dice = (2 * tp / denominator) if denominator else 1.0
        sample_dice = [
            binary_metrics(target, prediction, threshold=threshold)["dice"]
            for target, prediction in zip(y_true, probabilities, strict=False)
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
    probabilities = model.predict(x, verbose=0)
    if config is not None and segmentation_target_mode(config) == CANDIDATE_DELTA:
        if not records:
            raise ValueError("Candidate-delta threshold analysis requires validation records")
        candidates = np.stack([record["candidate_mask"] for record in records])
        validated = np.stack([record["validated_mask"] for record in records])
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
    validated_mask = np.asarray(validated) > 0.5
    candidate_mask = np.asarray(candidates) > 0.5
    true_delta = candidate_delta(candidate_mask, validated_mask)
    rows: list[dict[str, float]] = []
    for threshold_value in thresholds:
        threshold = float(threshold_value)
        predicted_delta = np.asarray(delta_probabilities) >= threshold
        reconstructed = reconstruct_validated_mask(candidate_mask, predicted_delta)
        reconstructed_metrics = binary_metrics(validated_mask, reconstructed)
        delta_metrics = binary_metrics(true_delta, predicted_delta)
        sample_dice = [
            binary_metrics(target, prediction)["dice"]
            for target, prediction in zip(validated_mask, reconstructed, strict=False)
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
