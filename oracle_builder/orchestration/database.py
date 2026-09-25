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
-- UI-owned annotations deliberately live outside sealed artifact manifests.
CREATE TABLE IF NOT EXISTS artifact_tags (
  tag_id TEXT PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE UNIQUE,
  color TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifact_tag_assignments (
  artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  tag_id TEXT NOT NULL REFERENCES artifact_tags(tag_id) ON DELETE CASCADE,
  created_at TEXT NOT NULL, PRIMARY KEY (artifact_id, tag_id)
);
CREATE INDEX IF NOT EXISTS artifact_tag_assignments_tag_idx ON artifact_tag_assignments(tag_id);
-- Denormalized facts make catalog filtering predictable without mutating runs.
CREATE TABLE IF NOT EXISTS artifact_facts (
  artifact_id TEXT PRIMARY KEY REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  training_set TEXT, classifier_type TEXT, stem_size INTEGER,
  macro_f1 REAL, loss REAL, training_seconds REAL, facts_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS artifact_facts_macro_f1_idx ON artifact_facts(macro_f1);
CREATE INDEX IF NOT EXISTS artifact_facts_classifier_idx ON artifact_facts(classifier_type);
CREATE TABLE IF NOT EXISTS model_drafts (
  draft_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
  source_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE SET NULL,
  revision INTEGER NOT NULL, config_json TEXT NOT NULL, layout_json TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_draft_revisions (
  draft_id TEXT NOT NULL REFERENCES model_drafts(draft_id) ON DELETE CASCADE,
  revision INTEGER NOT NULL, config_json TEXT NOT NULL, layout_json TEXT NOT NULL,
  created_at TEXT NOT NULL, PRIMARY KEY (draft_id, revision)
);
-- V2 model definitions are the user-owned, versioned scientific source of
-- truth.  They intentionally do not share the draft tables: queued work must
-- pin a definition revision without inheriting mutable draft semantics.
CREATE TABLE IF NOT EXISTS model_definitions (
  definition_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
  revision INTEGER NOT NULL, template_id TEXT, template_sha256 TEXT,
  parent_definition_id TEXT REFERENCES model_definitions(definition_id) ON DELETE SET NULL,
  parent_revision INTEGER, lineage_kind TEXT NOT NULL,
  config_json TEXT NOT NULL, config_sha256 TEXT NOT NULL, catalog_fingerprint TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
-- Definition names are user-facing identifiers in the construction library.
-- Keeping them unique makes server-generated "V.N" duplicate names safe even
-- when multiple browser sessions create copies at the same time.
CREATE UNIQUE INDEX IF NOT EXISTS model_definitions_name_idx
  ON model_definitions(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS model_definitions_updated_idx ON model_definitions(updated_at DESC);
CREATE TABLE IF NOT EXISTS model_definition_revisions (
  definition_id TEXT NOT NULL REFERENCES model_definitions(definition_id) ON DELETE CASCADE,
  revision INTEGER NOT NULL, config_json TEXT NOT NULL, config_sha256 TEXT NOT NULL,
  catalog_fingerprint TEXT, created_at TEXT NOT NULL,
  PRIMARY KEY (definition_id, revision)
);
-- A queued run is a sealed pairing of a model-definition revision and a
-- frozen dataset.  It is deliberately distinct from ``jobs``: a job is one
-- dispatch attempt while the queued run remains the durable user decision.
CREATE TABLE IF NOT EXISTS queued_runs (
  queued_run_id TEXT PRIMARY KEY,
  definition_id TEXT NOT NULL REFERENCES model_definitions(definition_id),
  definition_revision INTEGER NOT NULL,
  dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
  dataset_fingerprint_sha256 TEXT,
  name TEXT NOT NULL,
  description TEXT NOT NULL,
  specification_id TEXT REFERENCES run_specifications(specification_id),
  resolved_toml_path TEXT NOT NULL,
  resolved_toml_sha256 TEXT NOT NULL,
  config_schema_fingerprint TEXT,
  resources_json TEXT NOT NULL,
  initialization_json TEXT NOT NULL,
  preflight_endpoint_id TEXT REFERENCES compute_endpoints(endpoint_id),
  preflight_status TEXT NOT NULL,
  preflight_report_json TEXT NOT NULL,
  status TEXT NOT NULL,
  start_authorized INTEGER NOT NULL DEFAULT 0,
  priority INTEGER NOT NULL DEFAULT 0,
  failure_reason TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS queued_runs_ready_idx
  ON queued_runs(status, start_authorized, priority, created_at);
CREATE TABLE IF NOT EXISTS training_catalog_entries (
  catalog_id TEXT PRIMARY KEY, root_id TEXT NOT NULL, name TEXT NOT NULL,
  path TEXT NOT NULL, source_type TEXT NOT NULL, fingerprint_sha256 TEXT,
  metadata_json TEXT NOT NULL, scanned_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS training_catalog_entries_root_idx ON training_catalog_entries(root_id);
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
-- Comparison groups are deliberately permissive: a group records a user's
-- relationship claim, while the service reports protocol compatibility.
CREATE TABLE IF NOT EXISTS comparison_groups (
  comparison_group_id TEXT PRIMARY KEY, name TEXT NOT NULL,
  description TEXT NOT NULL, relationship_label TEXT NOT NULL,
  baseline_artifact_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS comparison_group_members (
  comparison_group_id TEXT NOT NULL REFERENCES comparison_groups(comparison_group_id) ON DELETE CASCADE,
  artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id), ordinal INTEGER NOT NULL,
  relationship_label TEXT, note TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (comparison_group_id, artifact_id),
  UNIQUE(comparison_group_id, ordinal)
);
CREATE INDEX IF NOT EXISTS comparison_group_members_artifact_idx ON comparison_group_members(artifact_id);
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
        "queued_run_id": "TEXT",
    }
    for column, data_type in migrations.items():
        if column not in existing:
            connection.execute(f"ALTER TABLE jobs ADD COLUMN {column} {data_type}")
    return connection
