from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oracle_builder.data.tiling import group_and_reassemble
from oracle_builder.evaluation.segmentation_targets import CANDIDATE_DELTA, reconstruct_validated_mask, segmentation_target_mode


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


def predict_reassembled_segmentation(
    model,
    x: np.ndarray,
    y: Any,
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray | None], list[dict[str, Any]]]:
    tile_predictions = model.predict(x, verbose=0)
    blend_mode = config.get("tiling", {}).get("blend_mode", "hann")
    predictions, source_records = group_and_reassemble(tile_predictions, records, blend_mode=blend_mode)
    target_values = list(y)
    if any(value is None for value in target_values):
        grouped_targets: dict[str, list[np.ndarray | None]] = {}
        for value, record in zip(target_values, records, strict=False):
            source_uuid = record.get("source_uuid", record["uuid"])
            grouped_targets.setdefault(source_uuid, []).append(value)
        targets = []
        for record in source_records:
            values = grouped_targets[record["uuid"]]
            if values[0] is None:
                targets.append(None)
            else:
                source_tiles = [
                    value for value in values if value is not None
                ]
                source_tile_records = [
                    tile_record
                    for tile_record in records
                    if tile_record.get("source_uuid", tile_record["uuid"]) == record["uuid"]
                ]
                reassembled, _ = group_and_reassemble(
                    source_tiles, source_tile_records, blend_mode="uniform"
                )
                targets.append(reassembled[0])
    else:
        targets, _ = group_and_reassemble(target_values, records, blend_mode="uniform")
    return predictions, targets, source_records


def evaluate_segmentation(
    model,
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict[str, Any]],
    run_dir: str | Path,
    threshold: float = 0.5,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    predictions, targets, records = predict_reassembled_segmentation(
        model, x, y, records, config or {}
    )
    rows = []
    target_mode = segmentation_target_mode(config or {})
    for row, true_mask, pred_mask in zip(records, targets, predictions, strict=False):
        if target_mode == CANDIDATE_DELTA:
            candidate = row["candidate_mask"]
            validated = row["validated_mask"]
            predicted_delta = np.asarray(pred_mask) >= threshold
            reconstructed = reconstruct_validated_mask(candidate, predicted_delta)
            delta_metrics = binary_metrics(true_mask, pred_mask, threshold=threshold)
            reconstructed_metrics = binary_metrics(validated, reconstructed)
            candidate_metrics = binary_metrics(validated, candidate)
            candidate_binary = np.asarray(candidate) > 0.5
            validated_binary = np.asarray(validated) > 0.5
            addition_pixels = int(np.logical_and(~candidate_binary, validated_binary).sum())
            removal_pixels = int(np.logical_and(candidate_binary, ~validated_binary).sum())
            rows.append(
                {
                    "uuid": row["uuid"],
                    "split": row["split"],
                    **reconstructed_metrics,
                    **{f"delta_{key}": value for key, value in delta_metrics.items()},
                    "candidate_dice": candidate_metrics["dice"],
                    "dice_improvement": reconstructed_metrics["dice"] - candidate_metrics["dice"],
                    "correction_fraction": float(np.logical_xor(candidate_binary, validated_binary).mean()),
                    "addition_pixels": addition_pixels,
                    "removal_pixels": removal_pixels,
                }
            )
        else:
            metrics = binary_metrics(true_mask, pred_mask, threshold=threshold)
            rows.append({"uuid": row["uuid"], "split": row["split"], **metrics})

    evaluation_dir = Path(run_dir) / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(evaluation_dir / "sample_metrics.csv", index=False)
    metrics_df.to_csv(evaluation_dir / "segmentation_metrics.csv", index=False)
    summary = {
        "task": "segmentation",
        "segmentation_target": target_mode,
        "probability_threshold": threshold,
        "mean_dice": float(metrics_df["dice"].mean()) if not metrics_df.empty else None,
        "mean_iou": float(metrics_df["iou"].mean()) if not metrics_df.empty else None,
        "mean_precision": float(metrics_df["precision"].mean()) if not metrics_df.empty else None,
        "mean_recall": float(metrics_df["recall"].mean()) if not metrics_df.empty else None,
    }
    if target_mode == CANDIDATE_DELTA and not metrics_df.empty:
        summary.update(
            {
                "mean_delta_dice": float(metrics_df["delta_dice"].mean()),
                "mean_candidate_dice": float(metrics_df["candidate_dice"].mean()),
                "mean_dice_improvement": float(metrics_df["dice_improvement"].mean()),
                "mean_correction_fraction": float(metrics_df["correction_fraction"].mean()),
                "total_addition_pixels": int(metrics_df["addition_pixels"].sum()),
                "total_removal_pixels": int(metrics_df["removal_pixels"].sum()),
            }
        )
    (evaluation_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"summary": summary, "predictions": predictions, "sample_metrics": rows}
