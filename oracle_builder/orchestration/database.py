from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA = """
PRAGMA foreign_keys = ON;
-- One-way control-plane migrations are recorded separately from scientific
-- catalog data.  This keeps historical projects readable while preventing
-- their old endpoint jobs from becoming executable after an upgrade.
CREATE TABLE IF NOT EXISTS schema_metadata (
  key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
);
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
  worker_pool_id TEXT REFERENCES worker_pools(pool_id),
  preflight_endpoint_id TEXT REFERENCES compute_endpoints(endpoint_id),
  preflight_status TEXT NOT NULL,
  preflight_report_json TEXT NOT NULL,
  status TEXT NOT NULL,
  start_authorized INTEGER NOT NULL DEFAULT 0,
  priority INTEGER NOT NULL DEFAULT 0,
  -- A dispatch claim is a short, durable lease held while the scheduler is
  -- making the remote submission.  It prevents two scheduler ticks (or a
  -- future second API process) from dispatching the same sealed run.
  dispatch_claim_token TEXT,
  dispatch_claim_owner TEXT,
  dispatch_claimed_at TEXT,
  dispatch_attempt INTEGER NOT NULL DEFAULT 0,
  failure_reason TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS queued_runs_ready_idx
  ON queued_runs(status, start_authorized, priority, created_at);
-- Inference requests are a first-class durable user decision.  They do not
-- reuse queued_runs: a completed model artifact and a frozen input dataset
-- are immutable inputs, rather than a mutable model definition to train.
CREATE TABLE IF NOT EXISTS inference_runs (
  inference_run_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  model_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
  model_ref_json TEXT NOT NULL,
  dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
  input_ref_json TEXT NOT NULL,
  parameters_json TEXT NOT NULL,
  resources_json TEXT NOT NULL,
  worker_pool_id TEXT NOT NULL REFERENCES worker_pools(pool_id),
  work_unit_json TEXT NOT NULL,
  work_unit_sha256 TEXT NOT NULL,
  job_id TEXT UNIQUE REFERENCES jobs(job_id) ON DELETE SET NULL,
  status TEXT NOT NULL,
  output_ref_json TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS inference_runs_status_idx ON inference_runs(status, created_at DESC);
CREATE INDEX IF NOT EXISTS inference_runs_model_idx ON inference_runs(model_artifact_id, created_at DESC);
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
  resources_json TEXT NOT NULL, work_unit_json TEXT, work_unit_sha256 TEXT,
  worker_pool_id TEXT REFERENCES worker_pools(pool_id),
  status TEXT NOT NULL, remote_status TEXT,
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
-- Remote event cursors are intentionally separate from the locally ordered
-- job-event sequence.  The orchestrator adds its own diagnostics/events, so
-- using a local sequence as Serve's ``after`` cursor could silently skip a
-- worker log line after a local status update.
CREATE TABLE IF NOT EXISTS job_remote_event_cursors (
  job_id TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
  remote_sequence INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL
);
-- Durable control-plane work.  These rows deliberately survive API restarts;
-- event sequence numbers provide a stable cursor for clients reconnecting to
-- the event stream.
CREATE TABLE IF NOT EXISTS operations (
  operation_id TEXT PRIMARY KEY, operation_type TEXT NOT NULL,
  status TEXT NOT NULL, parameters_json TEXT NOT NULL, result_json TEXT,
  error TEXT, created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS operations_status_idx ON operations(status, created_at);
CREATE TABLE IF NOT EXISTS operation_events (
  sequence INTEGER PRIMARY KEY AUTOINCREMENT,
  operation_id TEXT NOT NULL REFERENCES operations(operation_id) ON DELETE CASCADE,
  timestamp TEXT NOT NULL, event_type TEXT NOT NULL, message TEXT NOT NULL,
  data_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS operation_events_operation_idx ON operation_events(operation_id, sequence);
CREATE TABLE IF NOT EXISTS compute_endpoints (
  endpoint_id TEXT PRIMARY KEY, name TEXT NOT NULL, base_url TEXT NOT NULL UNIQUE,
  enabled INTEGER NOT NULL, status TEXT NOT NULL, last_checked_at TEXT,
  error TEXT, readiness_json TEXT, workers_json TEXT, queue_json TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
-- Desired state is durable. The local-process provider's process handles are
-- intentionally not: after an orchestrator restart a worker is reconciled as
-- unknown rather than being adopted or signalled by an untrusted PID.
CREATE TABLE IF NOT EXISTS managed_workers (
  worker_id TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT NOT NULL,
  provider TEXT NOT NULL, desired_state TEXT NOT NULL, host TEXT NOT NULL,
  port INTEGER NOT NULL, compute_queue_size INTEGER NOT NULL,
  compute_worker_slots INTEGER NOT NULL, compute_cpu_capacity INTEGER,
  endpoint_id TEXT REFERENCES compute_endpoints(endpoint_id),
  state TEXT NOT NULL, pid INTEGER, error TEXT, last_health_at TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS managed_workers_state_idx ON managed_workers(desired_state, state);
CREATE TABLE IF NOT EXISTS managed_worker_events (
  worker_id TEXT NOT NULL REFERENCES managed_workers(worker_id) ON DELETE CASCADE,
  sequence INTEGER NOT NULL, timestamp TEXT NOT NULL, event_type TEXT NOT NULL,
  message TEXT NOT NULL, data_json TEXT NOT NULL,
  PRIMARY KEY (worker_id, sequence)
);
-- Pull workers are intentionally separate from lifecycle-managed local
-- processes.  A pool is a scheduling and admission boundary; registration
-- secrets and per-worker bearer secrets are stored only as SHA-256 digests.
CREATE TABLE IF NOT EXISTS worker_pools (
  pool_id TEXT PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE UNIQUE,
  enabled INTEGER NOT NULL DEFAULT 1, allowed_actions_json TEXT NOT NULL,
  registration_token_sha256 TEXT NOT NULL, max_workers INTEGER,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS registered_workers (
  worker_id TEXT PRIMARY KEY, pool_id TEXT NOT NULL REFERENCES worker_pools(pool_id) ON DELETE CASCADE,
  name TEXT NOT NULL, endpoint TEXT, capabilities_json TEXT NOT NULL,
  auth_token_sha256 TEXT NOT NULL, state TEXT NOT NULL, last_seen_at TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE(pool_id, name)
);
CREATE INDEX IF NOT EXISTS registered_workers_pool_idx ON registered_workers(pool_id, state);
-- Deployment intent is distinct from worker registration.  This table holds
-- only an operator-selected profile identifier and public scheduling facts;
-- provider paths, images, and secrets remain process configuration.
CREATE TABLE IF NOT EXISTS worker_deployments (
  deployment_id TEXT PRIMARY KEY,
  name TEXT NOT NULL COLLATE NOCASE UNIQUE,
  pool_id TEXT NOT NULL REFERENCES worker_pools(pool_id),
  provider TEXT NOT NULL,
  profile_id TEXT NOT NULL,
  allowed_actions_json TEXT NOT NULL,
  capabilities_json TEXT NOT NULL,
  desired_state TEXT NOT NULL,
  state TEXT NOT NULL,
  endpoint TEXT,
  error TEXT,
  last_reconciled_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS worker_deployments_state_idx ON worker_deployments(desired_state, state);
CREATE TABLE IF NOT EXISTS worker_deployment_events (
  deployment_id TEXT NOT NULL REFERENCES worker_deployments(deployment_id) ON DELETE CASCADE,
  sequence INTEGER NOT NULL, timestamp TEXT NOT NULL, event_type TEXT NOT NULL,
  message TEXT NOT NULL, data_json TEXT NOT NULL,
  PRIMARY KEY (deployment_id, sequence)
);
-- A lease is a durable, single-worker claim over a sealed work unit.  The
-- raw lease token is returned exactly once and never written to SQLite.
CREATE TABLE IF NOT EXISTS worker_leases (
  lease_id TEXT PRIMARY KEY, worker_id TEXT NOT NULL REFERENCES registered_workers(worker_id) ON DELETE CASCADE,
  work_unit_id TEXT NOT NULL, job_id TEXT REFERENCES jobs(job_id) ON DELETE SET NULL,
  token_sha256 TEXT NOT NULL, status TEXT NOT NULL, issued_at TEXT NOT NULL,
  expires_at TEXT NOT NULL, released_at TEXT, outcome TEXT, created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS worker_leases_active_worker_idx
  ON worker_leases(worker_id) WHERE status='active';
CREATE UNIQUE INDEX IF NOT EXISTS worker_leases_active_work_unit_idx
  ON worker_leases(work_unit_id) WHERE status='active';
CREATE INDEX IF NOT EXISTS worker_leases_expiry_idx ON worker_leases(status, expires_at);
CREATE TABLE IF NOT EXISTS worker_lease_events (
  lease_id TEXT NOT NULL REFERENCES worker_leases(lease_id) ON DELETE CASCADE,
  sequence INTEGER NOT NULL, timestamp TEXT NOT NULL, event_type TEXT NOT NULL,
  message TEXT NOT NULL, data_json TEXT NOT NULL,
  PRIMARY KEY (lease_id, sequence)
);
-- Worker-originated events have a caller-selected idempotency key.  This
-- lets a remote worker retry after a dropped response without duplicating
-- progress entries in the durable lease history.
CREATE TABLE IF NOT EXISTS worker_lease_event_receipts (
  lease_id TEXT NOT NULL REFERENCES worker_leases(lease_id) ON DELETE CASCADE,
  event_id TEXT NOT NULL, sequence INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (lease_id, event_id)
);
-- Worker output publication is deliberately separate from execution.  A
-- completed worker may resume an interrupted transfer without rerunning its
-- expensive work.  Parts live only in the control-plane owned spool root.
CREATE TABLE IF NOT EXISTS worker_output_uploads (
  upload_id TEXT PRIMARY KEY,
  lease_id TEXT NOT NULL UNIQUE REFERENCES worker_leases(lease_id) ON DELETE CASCADE,
  worker_id TEXT NOT NULL REFERENCES registered_workers(worker_id) ON DELETE CASCADE,
  job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
  archive_size INTEGER NOT NULL,
  archive_sha256 TEXT NOT NULL,
  part_size INTEGER NOT NULL,
  part_count INTEGER NOT NULL,
  status TEXT NOT NULL,
  archive_path TEXT NOT NULL,
  created_at TEXT NOT NULL,
  finalized_at TEXT,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS worker_output_uploads_status_idx
  ON worker_output_uploads(status, updated_at);
CREATE TABLE IF NOT EXISTS worker_output_upload_parts (
  upload_id TEXT NOT NULL REFERENCES worker_output_uploads(upload_id) ON DELETE CASCADE,
  part_number INTEGER NOT NULL,
  size_bytes INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (upload_id, part_number)
);
-- Each worker lease is one durable execution attempt.  A job may have at
-- most one automatic retry, and only after a classified infrastructure loss.
CREATE TABLE IF NOT EXISTS execution_attempts (
  attempt_id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
  lease_id TEXT REFERENCES worker_leases(lease_id) ON DELETE SET NULL,
  ordinal INTEGER NOT NULL, status TEXT NOT NULL, classification TEXT,
  worker_id TEXT REFERENCES registered_workers(worker_id) ON DELETE SET NULL,
  work_unit_sha256 TEXT NOT NULL, error TEXT, output_ref_json TEXT,
  created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, updated_at TEXT NOT NULL,
  UNIQUE(job_id, ordinal), UNIQUE(lease_id)
);
CREATE INDEX IF NOT EXISTS execution_attempts_job_idx ON execution_attempts(job_id, ordinal DESC);
CREATE TABLE IF NOT EXISTS worker_commands (
  command_id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(job_id),
  lease_id TEXT REFERENCES worker_leases(lease_id), generation INTEGER NOT NULL,
  sequence INTEGER NOT NULL, action TEXT NOT NULL, status TEXT NOT NULL,
  reason TEXT NOT NULL, result_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE(job_id, sequence)
);
CREATE TABLE IF NOT EXISTS execution_runs (
  run_id TEXT PRIMARY KEY, specification_id TEXT NOT NULL,
  queued_run_id TEXT, worker_pool_id TEXT NOT NULL, status TEXT NOT NULL,
  current_job_id TEXT NOT NULL, completed_epoch INTEGER NOT NULL DEFAULT 0,
  total_epochs INTEGER NOT NULL, unit_epochs INTEGER NOT NULL,
  checkpoint_ref_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS worker_commands_lease_idx ON worker_commands(lease_id,sequence);
CREATE TABLE IF NOT EXISTS worker_heartbeats (
  worker_id TEXT PRIMARY KEY REFERENCES registered_workers(worker_id),
  lease_id TEXT NOT NULL REFERENCES worker_leases(lease_id),
  generation INTEGER NOT NULL, telemetry_json TEXT NOT NULL, received_at TEXT NOT NULL
);
-- Local folders remain primary.  This ledger makes off-host replica progress
-- durable, resumable, and observable without storing any object-store secret.
CREATE TABLE IF NOT EXISTS artifact_replications (
  artifact_ref_uri TEXT PRIMARY KEY, artifact_ref_json TEXT NOT NULL,
  status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
  manifest_sha256 TEXT, last_error TEXT, replicated_at TEXT,
  verified_at TEXT, restored_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS artifact_replications_status_idx ON artifact_replications(status, updated_at);
-- Browser transfers use explicit, resumable sessions. A session contains no
-- credential or arbitrary host path: it only identifies an operator-owned
-- temporary file under the artifact root. Part rows make an interrupted
-- multi-gigabyte transfer restartable without trusting client progress.
CREATE TABLE IF NOT EXISTS upload_sessions (
  upload_id TEXT PRIMARY KEY, kind TEXT NOT NULL, filename TEXT NOT NULL,
  size_bytes INTEGER NOT NULL, chunk_size_bytes INTEGER NOT NULL,
  temporary_path TEXT NOT NULL, destination_path TEXT NOT NULL,
  status TEXT NOT NULL, created_at TEXT NOT NULL, completed_at TEXT,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS upload_sessions_status_idx ON upload_sessions(status, updated_at);
CREATE TABLE IF NOT EXISTS upload_session_parts (
  upload_id TEXT NOT NULL REFERENCES upload_sessions(upload_id) ON DELETE CASCADE,
  offset_bytes INTEGER NOT NULL, size_bytes INTEGER NOT NULL,
  sha256 TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY (upload_id, offset_bytes)
);
-- Audit records intentionally omit request bodies, paths, and credentials.
CREATE TABLE IF NOT EXISTS audit_events (
  sequence INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
  actor_role TEXT NOT NULL, method TEXT NOT NULL, path TEXT NOT NULL,
  outcome TEXT NOT NULL, request_id TEXT
);
CREATE INDEX IF NOT EXISTS audit_events_timestamp_idx ON audit_events(timestamp DESC);
-- Serialize capacity snapshots per endpoint.  Queue-row claims protect an
-- individual dispatch; this lease additionally prevents two schedulers from
-- independently spending the same idle worker snapshot.
CREATE TABLE IF NOT EXISTS scheduler_leases (
  endpoint_id TEXT PRIMARY KEY REFERENCES compute_endpoints(endpoint_id) ON DELETE CASCADE,
  owner TEXT NOT NULL,
  claimed_at TEXT NOT NULL
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
    from oracle_builder.orchestration.inference_sharding import INFERENCE_SHARD_TABLE_SQL
    connection.executescript(INFERENCE_SHARD_TABLE_SQL)
    existing = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
    migrations = {
        "started_at": "TEXT",
        "output_path": "TEXT",
        "validation_status": "TEXT",
        "validation_report_json": "TEXT",
        "queued_run_id": "TEXT",
        "work_unit_json": "TEXT",
        "work_unit_sha256": "TEXT",
        "worker_pool_id": "TEXT",
        "execution_generation": "TEXT NOT NULL DEFAULT 'worker-v2'",
        "retry_count": "INTEGER NOT NULL DEFAULT 0",
        "max_retries": "INTEGER NOT NULL DEFAULT 1",
        "cancel_requested_at": "TEXT",
        "cancelled_at": "TEXT",
    }
    for column, data_type in migrations.items():
        if column not in existing:
            connection.execute(f"ALTER TABLE jobs ADD COLUMN {column} {data_type}")
    lease_columns = {row[1] for row in connection.execute("PRAGMA table_info(worker_leases)")}
    lease_migrations = {
        # The staging attempt is control-plane allocated and never supplied by
        # a remote worker. It forms the revision of the generic job output.
        "acknowledged_at": "TEXT",
        "output_attempt_id": "TEXT",
        "output_archive_sha256": "TEXT",
        "output_ref_json": "TEXT",
    }
    for column, data_type in lease_migrations.items():
        if column not in lease_columns:
            connection.execute(f"ALTER TABLE worker_leases ADD COLUMN {column} {data_type}")
    queued_columns = {row[1] for row in connection.execute("PRAGMA table_info(queued_runs)")}
    queued_migrations = {
        "worker_pool_id": "TEXT",
        "dispatch_claim_token": "TEXT",
        "dispatch_claim_owner": "TEXT",
        "dispatch_claimed_at": "TEXT",
        "dispatch_attempt": "INTEGER NOT NULL DEFAULT 0",
    }
    for column, data_type in queued_migrations.items():
        if column not in queued_columns:
            connection.execute(f"ALTER TABLE queued_runs ADD COLUMN {column} {data_type}")
    connection.execute("CREATE INDEX IF NOT EXISTS queued_runs_claim_idx ON queued_runs(preflight_endpoint_id, dispatch_claimed_at)")
    # Migration 5 is deliberately one-way. Endpoint-oriented jobs are kept as
    # historical evidence, but can never be picked up by the v2 pull queue.
    # New v2 jobs explicitly set this value at enqueue time.
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    version = connection.execute("SELECT value FROM schema_metadata WHERE key='schema_version'").fetchone()
    if version is None or int(version[0]) < 5:
        connection.execute(
            "UPDATE jobs SET execution_generation='historical', status='historical', remote_status='historical' "
            "WHERE worker_pool_id IS NULL AND status NOT IN ('historical','indexed','artifact_invalid')"
        )
        connection.execute(
            "INSERT INTO schema_metadata(key,value,updated_at) VALUES('schema_version','5',?) "
            "ON CONFLICT(key) DO UPDATE SET value='5',updated_at=excluded.updated_at", (now,),
        )
    return connection
