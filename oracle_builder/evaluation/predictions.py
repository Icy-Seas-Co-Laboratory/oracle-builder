from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from oracle_builder.data.decoders import encode_npy
from oracle_builder.evaluation.segmentation import binary_metrics


def init_predictions_db(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            uuid TEXT PRIMARY KEY,
            split TEXT,
            y_true_blob BLOB,
            y_true_encoding TEXT,
            y_pred_blob BLOB,
            y_pred_encoding TEXT,
            y_prob_json TEXT,
            metrics_json TEXT,
            metadata_json TEXT
        )
        """
    )
    return connection


def write_predictions_db(
    model,
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    sqlite_path: str | Path,
) -> None:
    predictions = model.predict(x, verbose=0)
    connection = init_predictions_db(sqlite_path)
    task = config["run"]["task"]
    for row, true_value, prediction in zip(records, y, predictions, strict=False):
        if task == "classification":
            pred_class = int(np.argmax(prediction))
            y_true_blob = str(int(true_value)).encode("utf-8")
            y_pred_blob = str(pred_class).encode("utf-8")
            y_prob_json = json.dumps([float(v) for v in prediction])
            metrics_json = json.dumps({"correct": bool(pred_class == int(true_value))})
            true_encoding = pred_encoding = "int"
        else:
            metrics = binary_metrics(true_value, prediction)
            y_true_blob = encode_npy(np.asarray(true_value))
            y_pred_blob = encode_npy(np.asarray(prediction))
            y_prob_json = None
            metrics_json = json.dumps(metrics)
            true_encoding = pred_encoding = "npy"
        connection.execute(
            "INSERT OR REPLACE INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["uuid"],
                row["split"],
                y_true_blob,
                true_encoding,
                y_pred_blob,
                pred_encoding,
                y_prob_json,
                metrics_json,
                json.dumps(row.get("metadata", {})),
            ),
        )
    connection.commit()
    connection.close()

