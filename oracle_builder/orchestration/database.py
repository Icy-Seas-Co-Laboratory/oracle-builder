from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS datasets (
  dataset_id TEXT PRIMARY KEY, revision_id TEXT, name TEXT NOT NULL,
  dataset_type TEXT, lifecycle TEXT, fingerprint_sha256 TEXT, path TEXT NOT NULL,
  metadata_json TEXT NOT NULL, discovered_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id TEXT PRIMARY KEY, run_id TEXT, artifact_type TEXT NOT NULL,
  name TEXT NOT NULL, task TEXT, architecture TEXT, variant TEXT,
  status TEXT, lifecycle TEXT, dataset_id TEXT, dataset_fingerprint_sha256 TEXT,
  fingerprint_sha256 TEXT, path TEXT NOT NULL, manifest_json TEXT NOT NULL,
  discovered_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS artifacts_dataset_idx ON artifacts(dataset_id);
CREATE TABLE IF NOT EXISTS recipes (
  recipe_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
  config_path TEXT NOT NULL, config_sha256 TEXT NOT NULL,
  task TEXT NOT NULL, model TEXT NOT NULL, summary_json TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS recipes_config_hash_idx ON recipes(config_sha256);
CREATE TABLE IF NOT EXISTS experiments (
  experiment_id TEXT PRIMARY KEY, project_id TEXT, name TEXT NOT NULL,
  description TEXT NOT NULL, dataset_id TEXT, status TEXT NOT NULL,
  plan_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_specifications (
  specification_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
  ordinal INTEGER NOT NULL, name TEXT NOT NULL, action TEXT NOT NULL,
  parameters_json TEXT NOT NULL, resources_json TEXT NOT NULL, config_hash TEXT,
  status TEXT NOT NULL, artifact_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE(experiment_id, ordinal)
);
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY, specification_id TEXT REFERENCES run_specifications(specification_id),
  oracle_serve_url TEXT NOT NULL, action TEXT NOT NULL, parameters_json TEXT NOT NULL,
  resources_json TEXT NOT NULL, status TEXT NOT NULL, remote_status TEXT,
  worker_id TEXT, error TEXT, submitted_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  completed_at TEXT, started_at TEXT, output_path TEXT,
  validation_status TEXT, validation_report_json TEXT
);
CREATE TABLE IF NOT EXISTS job_events (
  job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
  sequence INTEGER NOT NULL, timestamp TEXT NOT NULL, event_type TEXT NOT NULL,
  message TEXT NOT NULL, data_json TEXT NOT NULL,
  PRIMARY KEY (job_id, sequence)
);
CREATE TABLE IF NOT EXISTS compute_endpoints (
  endpoint_id TEXT PRIMARY KEY, name TEXT NOT NULL, base_url TEXT NOT NULL UNIQUE,
  enabled INTEGER NOT NULL, status TEXT NOT NULL, last_checked_at TEXT,
  error TEXT, readiness_json TEXT, workers_json TEXT, queue_json TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS comparisons (
  comparison_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
  selection_json TEXT NOT NULL, protocol_json TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    database = Path(path).expanduser().resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    existing = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
    migrations = {
        "started_at": "TEXT",
        "output_path": "TEXT",
        "validation_status": "TEXT",
        "validation_report_json": "TEXT",
    }
    for column, data_type in migrations.items():
        if column not in existing:
            connection.execute(f"ALTER TABLE jobs ADD COLUMN {column} {data_type}")
    return connection
