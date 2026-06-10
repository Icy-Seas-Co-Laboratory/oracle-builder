from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tensorflow import keras


def init_training_log(path: str | Path, run_id: str, run_name: str, config: dict[str, Any], environment: dict[str, Any]) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS run (
            run_id TEXT PRIMARY KEY,
            run_name TEXT,
            started_at TEXT,
            completed_at TEXT,
            status TEXT,
            config_json TEXT,
            environment_json TEXT
        );
        CREATE TABLE IF NOT EXISTS epoch_metrics (
            run_id TEXT,
            epoch INTEGER,
            split TEXT,
            metric TEXT,
            value REAL
        );
        CREATE TABLE IF NOT EXISTS events (
            run_id TEXT,
            timestamp TEXT,
            level TEXT,
            message TEXT,
            details_json TEXT
        );
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO run VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            run_name,
            datetime.now(timezone.utc).isoformat(),
            None,
            "running",
            json.dumps(config, default=str),
            json.dumps(environment, default=str),
        ),
    )
    connection.commit()
    connection.close()


def log_event(path: str | Path, run_id: str, level: str, message: str, details: dict[str, Any] | None = None) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
        (run_id, datetime.now(timezone.utc).isoformat(), level, message, json.dumps(details or {}, default=str)),
    )
    connection.commit()
    connection.close()


def mark_run_complete(path: str | Path, run_id: str, status: str) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE run SET completed_at = ?, status = ? WHERE run_id = ?",
        (datetime.now(timezone.utc).isoformat(), status, run_id),
    )
    connection.commit()
    connection.close()


class SQLiteMetricLogger(keras.callbacks.Callback):
    def __init__(self, sqlite_path: str | Path, run_id: str):
        super().__init__()
        self.sqlite_path = sqlite_path
        self.run_id = run_id

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        connection = sqlite3.connect(self.sqlite_path)
        for key, value in logs.items():
            split = "validation" if key.startswith("val_") else "train"
            metric = key[4:] if key.startswith("val_") else key
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            connection.execute(
                "INSERT INTO epoch_metrics VALUES (?, ?, ?, ?, ?)",
                (self.run_id, int(epoch), split, metric, numeric),
            )
        connection.commit()
        connection.close()

