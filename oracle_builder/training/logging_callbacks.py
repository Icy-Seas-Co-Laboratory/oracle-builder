from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tensorflow import keras


EVENT_SCHEMA = "oracle_training_event"
METRIC_SCHEMA = "oracle_training_metric"


def _events_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.suffix == ".jsonl" else value.parent / "events.jsonl"


def _metrics_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.suffix == ".jsonl" else value.parent.parent / "metrics" / "metrics.jsonl"


def _append_jsonl(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str, allow_nan=False) + "\n")


def append_jsonl_event(
    path: str | Path,
    run_id: str,
    level: str,
    event: str,
    data: dict[str, Any] | None = None,
) -> None:
    _append_jsonl(
        path,
        {
            "schema": EVENT_SCHEMA,
            "schema_version": "1.0.0",
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": str(run_id),
            "level": str(level),
            "event": str(event),
            "data": data or {},
        },
    )


def write_history_jsonl(
    history: dict[str, list[float]],
    path: str | Path,
    *,
    run_id: str | None = None,
    phase: str = "training",
) -> None:
    target = Path(path)
    if target.exists():
        target.unlink()
    names = sorted(history)
    for epoch in range(max((len(values) for values in history.values()), default=0)):
        for name in names:
            values = history[name]
            if epoch >= len(values):
                continue
            value = values[epoch]
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value != value:
                continue
            _append_jsonl(
                target,
                {
                    "schema": METRIC_SCHEMA,
                    "schema_version": "1.0.0",
                    "metric_id": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "run_id": run_id,
                    "phase": phase,
                    "epoch": epoch,
                    "metric": name[4:] if name.startswith("val_") else name,
                    "split": "validation" if name.startswith("val_") else "train",
                    "value": value,
                },
            )


def init_training_log(
    path: str | Path,
    run_id: str,
    run_name: str,
    config: dict[str, Any],
    environment: dict[str, Any],
    *,
    resume: bool = False,
) -> None:
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
    if resume:
        connection.execute(
            """
            UPDATE run
            SET completed_at = NULL, status = 'running', config_json = ?, environment_json = ?
            WHERE run_id = ?
            """,
            (json.dumps(config, default=str), json.dumps(environment, default=str), run_id),
        )
    else:
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
    append_jsonl_event(
        _events_path(path),
        run_id,
        "INFO",
        "run_resumed" if resume else "run_started",
        {"run_name": run_name},
    )


def log_event(path: str | Path, run_id: str, level: str, message: str, details: dict[str, Any] | None = None) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
        (run_id, datetime.now(timezone.utc).isoformat(), level, message, json.dumps(details or {}, default=str)),
    )
    connection.commit()
    connection.close()
    append_jsonl_event(_events_path(path), run_id, level, message, details)


def mark_run_complete(path: str | Path, run_id: str, status: str) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE run SET completed_at = ?, status = ? WHERE run_id = ?",
        (datetime.now(timezone.utc).isoformat(), status, run_id),
    )
    connection.commit()
    connection.close()
    append_jsonl_event(_events_path(path), run_id, "INFO", "run_finished", {"status": status})


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
                "DELETE FROM epoch_metrics WHERE run_id = ? AND epoch = ? AND split = ? AND metric = ?",
                (self.run_id, int(epoch), split, metric),
            )
            connection.execute(
                "INSERT INTO epoch_metrics VALUES (?, ?, ?, ?, ?)",
                (self.run_id, int(epoch), split, metric, numeric),
            )
            _append_jsonl(
                _metrics_path(self.sqlite_path),
                {
                    "schema": METRIC_SCHEMA,
                    "schema_version": "1.0.0",
                    "metric_id": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "run_id": self.run_id,
                    "phase": "training",
                    "epoch": int(epoch),
                    "metric": metric,
                    "split": split,
                    "value": numeric,
                },
            )
        connection.commit()
        connection.close()


def history_from_training_log(path: str | Path, run_id: str) -> dict[str, list[float]]:
    """Reconstruct one continuous supervised history, including resumed epochs."""
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            """
            SELECT epoch, split, metric, value
            FROM epoch_metrics
            WHERE run_id = ?
            ORDER BY epoch, split, metric
            """,
            (run_id,),
        ).fetchall()
    finally:
        connection.close()
    values: dict[str, dict[int, float]] = {}
    max_epoch = -1
    for epoch, split, metric, value in rows:
        name = f"val_{metric}" if split == "validation" else str(metric)
        values.setdefault(name, {})[int(epoch)] = float(value)
        max_epoch = max(max_epoch, int(epoch))
    return {
        name: [by_epoch.get(epoch, float("nan")) for epoch in range(max_epoch + 1)]
        for name, by_epoch in values.items()
    }
