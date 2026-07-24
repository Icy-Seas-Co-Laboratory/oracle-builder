from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from oracle_builder.data.decoders import encode_npy
from oracle_builder.evaluation.segmentation import binary_metrics
from oracle_builder.evaluation.segmentation import predict_reassembled_segmentation
from oracle_builder.evaluation.segmentation_targets import CANDIDATE_DELTA, reconstruct_validated_mask, reconstruct_validated_probability, segmentation_target_mode


def init_predictions_db(
    path: str | Path,
    source_sqlite: str | Path | None = None,
) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() and source_sqlite is not None:
        source = Path(source_sqlite)
        if source.resolve() == path.resolve():
            raise ValueError("Prediction output must differ from the source SQLite database")
        with sqlite3.connect(source) as source_connection, sqlite3.connect(path) as destination:
            source_connection.backup(destination)
    connection = sqlite3.connect(path)
    existing = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'predictions'"
    ).fetchone()
    if existing:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(predictions)")}
        if "prediction_set" not in columns:
            connection.close()
            raise ValueError(
                f"{path} uses the legacy single-set predictions schema; choose a new output path"
            )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_sets (
            prediction_set TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            run_id TEXT,
            run_name TEXT,
            config_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_set TEXT NOT NULL,
            uuid TEXT NOT NULL,
            split TEXT,
            y_true_blob BLOB,
            y_true_encoding TEXT,
            y_pred_blob BLOB,
            y_pred_encoding TEXT,
            y_prob_json TEXT,
            metrics_json TEXT,
            metadata_json TEXT,
            target_mode TEXT NOT NULL DEFAULT 'validated_mask',
            reconstructed_pred_blob BLOB,
            reconstructed_pred_encoding TEXT,
            PRIMARY KEY (prediction_set, uuid),
            FOREIGN KEY (prediction_set) REFERENCES prediction_sets(prediction_set),
            FOREIGN KEY (uuid) REFERENCES samples(uuid)
        )
        """
    )
    prediction_columns = {row[1] for row in connection.execute("PRAGMA table_info(predictions)")}
    for column, definition in {
        "target_mode": "TEXT NOT NULL DEFAULT 'validated_mask'",
        "reconstructed_pred_blob": "BLOB",
        "reconstructed_pred_encoding": "TEXT",
    }.items():
        if column not in prediction_columns:
            connection.execute(f"ALTER TABLE predictions ADD COLUMN {column} {definition}")
    return connection


def write_predictions_db(
    model,
    x: np.ndarray,
    y: Any,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    sqlite_path: str | Path,
    *,
    source_sqlite: str | Path | None = None,
    prediction_set: str = "default",
) -> None:
    task = config["run"]["task"]
    if task == "segmentation":
        predictions, y, records = predict_reassembled_segmentation(model, x, y, records, config)
    else:
        predictions = model.predict(x, verbose=0)
    connection = init_predictions_db(sqlite_path, source_sqlite=source_sqlite)
    run = config.get("run", {})
    connection.execute(
        """
        INSERT INTO prediction_sets (prediction_set, created_at, run_id, run_name, config_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(prediction_set) DO UPDATE SET
            run_id = excluded.run_id,
            run_name = excluded.run_name,
            config_json = excluded.config_json
        """,
        (
            prediction_set,
            datetime.now(timezone.utc).isoformat(),
            run.get("run_id"),
            run.get("name") or run.get("run_name"),
            json.dumps(config, sort_keys=True, default=str),
        ),
    )
    target_mode = segmentation_target_mode(config)
    segmentation_threshold = float(config.get("evaluation", {}).get("segmentation_threshold", 0.5))
    for row, true_value, prediction in zip(records, y, predictions, strict=False):
        prediction_metadata = dict(row.get("metadata", {}))
        prediction_metadata["prediction"] = {
            "tile_count": int(row.get("tile_count", 1)),
            "source_shape": row.get("source_shape"),
            "tiling_enabled": bool(config.get("tiling", {}).get("enabled", False)),
            "overlap_fraction": float(config.get("tiling", {}).get("overlap_fraction", 0.0)),
            "blend_mode": config.get("tiling", {}).get("blend_mode", "uniform"),
        }
        if true_value is None:
            y_true_blob = None
            true_encoding = None
            if task == "classification":
                pred_class = int(np.argmax(prediction))
                y_pred_blob = str(pred_class).encode("utf-8")
                y_prob_json = json.dumps([float(v) for v in prediction])
                pred_encoding = "int"
            else:
                y_pred_blob = encode_npy(np.asarray(prediction))
                y_prob_json = None
                pred_encoding = "npy"
                reconstructed_prediction = _reconstructed_prediction(row, prediction, target_mode)
            metrics_json = None
        elif task == "classification":
            pred_class = int(np.argmax(prediction))
            y_true_blob = str(int(true_value)).encode("utf-8")
            y_pred_blob = str(pred_class).encode("utf-8")
            y_prob_json = json.dumps([float(v) for v in prediction])
            metrics_json = json.dumps({"correct": bool(pred_class == int(true_value))})
            true_encoding = pred_encoding = "int"
        else:
            if target_mode == CANDIDATE_DELTA:
                predicted_delta = np.asarray(prediction) >= segmentation_threshold
                reconstructed_mask = reconstruct_validated_mask(row["candidate_mask"], predicted_delta)
                reconstructed_prediction = reconstruct_validated_probability(row["candidate_mask"], prediction)
                delta_metrics = binary_metrics(true_value, prediction, threshold=segmentation_threshold)
                reconstructed_metrics = binary_metrics(row["validated_mask"], reconstructed_mask)
                candidate_metrics = binary_metrics(row["validated_mask"], row["candidate_mask"])
                candidate_binary = np.asarray(row["candidate_mask"]) > 0.5
                validated_binary = np.asarray(row["validated_mask"]) > 0.5
                metrics = {
                    **reconstructed_metrics,
                    **{f"delta_{key}": value for key, value in delta_metrics.items()},
                    "candidate_dice": candidate_metrics["dice"],
                    "dice_improvement": reconstructed_metrics["dice"] - candidate_metrics["dice"],
                    "correction_fraction": float(np.logical_xor(candidate_binary, validated_binary).mean()),
                    "addition_pixels": int(np.logical_and(~candidate_binary, validated_binary).sum()),
                    "removal_pixels": int(np.logical_and(candidate_binary, ~validated_binary).sum()),
                }
            else:
                metrics = binary_metrics(true_value, prediction, threshold=segmentation_threshold)
                reconstructed_prediction = np.asarray(prediction)
            metrics["probability_threshold"] = segmentation_threshold
            metrics["segmentation_target"] = target_mode
            y_true_blob = encode_npy(np.asarray(true_value))
            y_pred_blob = encode_npy(np.asarray(prediction))
            y_prob_json = None
            metrics_json = json.dumps(metrics)
            true_encoding = pred_encoding = "npy"
        connection.execute(
            """
            INSERT OR REPLACE INTO predictions (
                prediction_set, uuid, split, y_true_blob, y_true_encoding,
                y_pred_blob, y_pred_encoding, y_prob_json, metrics_json, metadata_json,
                target_mode, reconstructed_pred_blob, reconstructed_pred_encoding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_set,
                row["uuid"],
                row["split"],
                y_true_blob,
                true_encoding,
                y_pred_blob,
                pred_encoding,
                y_prob_json,
                metrics_json,
                json.dumps(prediction_metadata),
                target_mode,
                encode_npy(np.asarray(reconstructed_prediction)) if task == "segmentation" else None,
                "npy" if task == "segmentation" else None,
            ),
        )
    connection.commit()
    connection.close()


def _reconstructed_prediction(row: dict[str, Any], prediction: Any, target_mode: str) -> np.ndarray:
    if target_mode == CANDIDATE_DELTA:
        return reconstruct_validated_probability(row["candidate_mask"], prediction)
    return np.asarray(prediction)
