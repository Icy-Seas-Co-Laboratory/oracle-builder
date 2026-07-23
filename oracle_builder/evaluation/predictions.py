from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from oracle_builder.data.decoders import encode_npy
from oracle_builder.evaluation.segmentation import binary_metrics


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
            PRIMARY KEY (prediction_set, uuid),
            FOREIGN KEY (prediction_set) REFERENCES prediction_sets(prediction_set),
            FOREIGN KEY (uuid) REFERENCES samples(uuid)
        )
        """
    )
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
    task = config["run"]["task"]
    segmentation_threshold = float(config.get("evaluation", {}).get("segmentation_threshold", 0.5))
    for row, true_value, prediction in zip(records, y, predictions, strict=False):
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
            metrics_json = None
        elif task == "classification":
            pred_class = int(np.argmax(prediction))
            y_true_blob = str(int(true_value)).encode("utf-8")
            y_pred_blob = str(pred_class).encode("utf-8")
            y_prob_json = json.dumps([float(v) for v in prediction])
            metrics_json = json.dumps({"correct": bool(pred_class == int(true_value))})
            true_encoding = pred_encoding = "int"
        else:
            metrics = binary_metrics(true_value, prediction, threshold=segmentation_threshold)
            metrics["probability_threshold"] = segmentation_threshold
            y_true_blob = encode_npy(np.asarray(true_value))
            y_pred_blob = encode_npy(np.asarray(prediction))
            y_prob_json = None
            metrics_json = json.dumps(metrics)
            true_encoding = pred_encoding = "npy"
        connection.execute(
            "INSERT OR REPLACE INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                json.dumps(row.get("metadata", {})),
            ),
        )
    connection.commit()
    connection.close()
