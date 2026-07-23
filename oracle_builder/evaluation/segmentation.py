from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    true_mask = y_true.astype("float32") > 0.5
    pred_mask = y_pred.astype("float32") >= threshold
    tp = float(np.logical_and(true_mask, pred_mask).sum())
    fp = float(np.logical_and(~true_mask, pred_mask).sum())
    fn = float(np.logical_and(true_mask, ~pred_mask).sum())
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


def evaluate_segmentation(
    model,
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict[str, Any]],
    run_dir: str | Path,
    threshold: float = 0.5,
) -> dict[str, Any]:
    predictions = model.predict(x, verbose=0)
    rows = []
    for row, true_mask, pred_mask in zip(records, y, predictions, strict=False):
        metrics = binary_metrics(true_mask, pred_mask, threshold=threshold)
        rows.append({"uuid": row["uuid"], "split": row["split"], **metrics})

    evaluation_dir = Path(run_dir) / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(evaluation_dir / "sample_metrics.csv", index=False)
    metrics_df.to_csv(evaluation_dir / "segmentation_metrics.csv", index=False)
    summary = {
        "task": "segmentation",
        "probability_threshold": threshold,
        "mean_dice": float(metrics_df["dice"].mean()) if not metrics_df.empty else None,
        "mean_iou": float(metrics_df["iou"].mean()) if not metrics_df.empty else None,
        "mean_precision": float(metrics_df["precision"].mean()) if not metrics_df.empty else None,
        "mean_recall": float(metrics_df["recall"].mean()) if not metrics_df.empty else None,
    }
    (evaluation_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"summary": summary, "predictions": predictions, "sample_metrics": rows}
