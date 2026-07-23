from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oracle_builder.evaluation.segmentation import binary_metrics


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
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    probabilities = model.predict(x, verbose=0)
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
