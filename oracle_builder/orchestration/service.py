from __future__ import annotations

import csv
import hmac
import hashlib
import io
import json
import os
import re
import secrets
import sqlite3
import shutil
import tempfile
import urllib.error
import urllib.request
import uuid
import threading
import tarfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from oracle_data_contracts.datasets import dataset_fingerprint, read_dataset_info, set_dataset_lifecycle
from oracle_builder.data.decoders import decode_blob
from oracle_builder.config import (
    DEFAULT_CONFIG,
    deep_merge,
    load_toml,
    normalize_component_config,
    resolve_v2_config,
    validate_config,
    validate_v2_config,
)
from oracle_builder.config_schema import configuration_schema as v2_configuration_schema
from oracle_builder.orchestration.database import connect
from oracle_builder.orchestration.lifecycle import (
    LocalProcessWorkerProvider,
    LocalWorkerSpec,
    PullWorkerProcessProvider,
    PullWorkerSpec,
    PullWorkerDeploymentProfile,
    PullWorkerLifecycleProvider,
    WorkerProvider,
)
from oracle_builder.orchestration.storage import ArtifactStore, LocalArtifactStore
from oracle_builder.orchestration.work_units import LocalPathWorkUnitAdapter, build_work_unit
from oracle_data_contracts.artifacts import ArtifactRef


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    for key in ("metadata_json", "manifest_json", "plan_json", "parameters_json", "resources_json", "work_unit_json", "selection_json", "protocol_json", "summary_json", "validation_report_json", "readiness_json", "workers_json", "queue_json", "facts_json", "config_json", "layout_json", "initialization_json", "preflight_report_json", "allowed_actions_json", "capabilities_json", "output_ref_json", "model_ref_json", "input_ref_json"):
        if key in result:
            raw = result.pop(key)
            result[key.removesuffix("_json")] = json.loads(raw) if raw is not None else None
    return result


from oracle_builder.orchestration.worker_control import WorkerControlMixin
from oracle_builder.orchestration.sequential import SequentialExecutionMixin
from oracle_builder.orchestration.inference_sharding import InferenceShardingMixin


class Orchestrator(WorkerControlMixin, SequentialExecutionMixin, InferenceShardingMixin):
    """SQLite control plane for Oracle Builder artifacts and compute requests."""

    def __init__(
        self,
        database: str | Path,
        *,
        artifact_root: str | Path | None = None,
        runs_root: str | Path | None = None,
        datasets_root: str | Path | None = None,
        browse_roots: list[str | Path] | None = None,
        training_catalog_roots: list[str | Path] | None = None,
        log_root: str | Path | None = None,
        upload_limit_bytes: int = 10 * 1024 * 1024 * 1024,
        workspace_root: str | Path | None = None,
        compute_endpoints: list[tuple[str, str]] | None = None,
        artifact_store: ArtifactStore | None = None,
        worker_providers: dict[str, WorkerProvider] | None = None,
        pull_worker_provider: PullWorkerProcessProvider | None = None,
        deployment_profiles: dict[str, PullWorkerDeploymentProfile] | None = None,
        deployment_providers: dict[str, PullWorkerLifecycleProvider] | None = None,
    ):
        self.database = Path(database).expanduser().resolve()
        # The stack launcher writes only these service logs beneath the
        # runtime directory.  Keep this a fixed allow-list rather than a
        # general file browser so diagnostics cannot disclose arbitrary files.
        self.log_root = (Path(log_root).expanduser().resolve() if log_root else self.database.parent / "logs")
        self.artifact_root = Path(artifact_root).expanduser().resolve() if artifact_root else self.database.parent / "oracle-artifacts"
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        # This is deliberately injected at the control-plane boundary.  The
        # current path-based worker is still a compatibility adapter, but new
        # work-unit and publication flows have one authoritative store owner.
        self.artifact_store: ArtifactStore = artifact_store or LocalArtifactStore(
            self.artifact_root / "artifact-store"
        )
        self.worker_providers: dict[str, WorkerProvider] = {
            "local-process": LocalProcessWorkerProvider(),
            **(worker_providers or {}),
        }
        # Pull workers are the current execution runtime.  Keep their
        # lifecycle separate from retired managed HTTP compute services: a
        # process provider owns only children it created and exposes no remote
        # shell channel.  Deployment-specific providers can implement the
        # same typed start/stop/status surface.
        self.pull_worker_provider = pull_worker_provider or PullWorkerProcessProvider()
        self.deployment_profiles = dict(deployment_profiles or {})
        self.deployment_providers: dict[str, PullWorkerLifecycleProvider] = {
            "local-process": self.pull_worker_provider,
            **(deployment_providers or {}),
        }
        for profile_id, profile in self.deployment_profiles.items():
            if profile_id != profile.profile_id:
                raise ValueError("deployment profile mapping key must match profile_id")
            if profile.provider not in self.deployment_providers:
                raise ValueError(f"deployment profile {profile_id!r} references unavailable provider {profile.provider!r}")
        if upload_limit_bytes < 1:
            raise ValueError("upload_limit_bytes must be positive")
        self.upload_limit_bytes = upload_limit_bytes
        self.workspace_root = Path(workspace_root).expanduser().resolve() if workspace_root else Path.cwd().resolve()
        if not self.workspace_root.is_dir():
            raise NotADirectoryError(self.workspace_root)
        # The control database and upload staging area are runtime concerns;
        # ordinary Oracle Builder science products stay in the familiar project
        # directories unless an operator chooses different roots explicitly.
        self.runs_root = Path(runs_root).expanduser().resolve() if runs_root else self.workspace_root / "runs"
        self.datasets_root = Path(datasets_root).expanduser().resolve() if datasets_root else self.workspace_root / "datasets"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.datasets_root.mkdir(parents=True, exist_ok=True)
        self.browse_roots: dict[str, Path] = {"workspace": self.workspace_root, "artifacts": self.artifact_root}
        for index, root in enumerate(browse_roots or [], start=1):
            location = Path(root).expanduser().resolve()
            if not location.is_dir():
                raise NotADirectoryError(location)
            self.browse_roots[f"root-{index}"] = location
        for root_id, location in (("runs", self.runs_root), ("datasets", self.datasets_root)):
            if not any(location.is_relative_to(allowed) for allowed in self.browse_roots.values()):
                self.browse_roots[root_id] = location
        configured_catalog_roots = [self.datasets_root, *(Path(root).expanduser().resolve() for root in training_catalog_roots or [])]
        self.training_catalog_roots: dict[str, Path] = {}
        for root in configured_catalog_roots:
            location = Path(root).expanduser().resolve()
            if not location.is_dir():
                raise NotADirectoryError(location)
            if location not in self.training_catalog_roots.values():
                self.training_catalog_roots[f"catalog-{len(self.training_catalog_roots) + 1}"] = location
            if not any(location.is_relative_to(allowed) for allowed in self.browse_roots.values()):
                self.browse_roots[f"catalog-{len(self.browse_roots) + 1}"] = location
        with connect(self.database):
            pass
        self._operation_stop: threading.Event | None = None
        self._operation_thread: threading.Thread | None = None
        for name, base_url in compute_endpoints or []:
            self.register_compute_endpoint(name=name, base_url=base_url)

    def _connection(self) -> sqlite3.Connection:
        return connect(self.database)

    def server_logs(self, *, tail_lines: int = 400) -> dict[str, Any]:
        """Return bounded service tails plus the scheduler's cached diagnosis.

        Reading logs must stay cheap and local: the modal must not itself
        create a burst of requests to Serve.  The scheduler summary therefore
        uses the most recently persisted endpoint snapshot, which is also the
        snapshot used by the Queue page between refreshes.
        """
        lines = max(1, min(int(tail_lines), 2_000))
        services = {
            "orchestrator": ("Oracle Orchestrator", "orchestrator.log"),
            "worker": ("Oracle worker", "oracle-worker.log"),
            "webgui": ("Web GUI", "webgui.log"),
        }
        logs: list[dict[str, Any]] = []
        for service_id, (name, filename) in services.items():
            path = (self.log_root / filename).resolve()
            if not path.is_relative_to(self.log_root) or not path.is_file():
                logs.append({"service": service_id, "name": name, "available": False, "message": "Log file is not available."})
                continue
            try:
                # Tails are capped before decoding to keep a noisy process
                # from turning a health-modal request into a large response.
                with path.open("rb") as handle:
                    handle.seek(0, 2)
                    handle.seek(max(0, handle.tell() - 512 * 1024))
                    tail = handle.read().decode("utf-8", errors="replace").splitlines()[-lines:]
                logs.append({
                    "service": service_id, "name": name, "available": True,
                    "updated_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                    "line_count": len(tail), "text": "\n".join(tail),
                })
            except OSError as exc:
                logs.append({"service": service_id, "name": name, "available": False, "message": f"Could not read log: {exc}"})
        endpoint_diagnostics: list[dict[str, Any]] = []
        findings: list[str] = []
        with self._connection() as connection:
            queue_counts_by_endpoint: dict[str, dict[str, int]] = {}
            for row in connection.execute(
                """SELECT preflight_endpoint_id, status, COUNT(*) AS count FROM queued_runs
                   WHERE start_authorized=1
                     AND status IN ('ready', 'waiting_for_resources')
                     AND preflight_endpoint_id IS NOT NULL
                   GROUP BY preflight_endpoint_id, status"""
            ).fetchall():
                endpoint_counts = queue_counts_by_endpoint.setdefault(str(row["preflight_endpoint_id"]), {})
                endpoint_counts[str(row["status"])] = int(row["count"])
        for endpoint in self.compute_endpoints():
            endpoint_id = str(endpoint["endpoint_id"])
            workers = endpoint.get("workers") or []
            worker_states: dict[str, int] = {}
            for worker in workers:
                state = str(worker.get("status") or "unknown")
                worker_states[state] = worker_states.get(state, 0) + 1
            idle_workers = worker_states.get("idle", 0)
            queue_counts = queue_counts_by_endpoint.get(endpoint_id, {})
            authorized = sum(queue_counts.values())
            status = str(endpoint.get("status") or "unknown")
            diagnosis = "No authorized runs are waiting for this endpoint."
            if status != "ready":
                diagnosis = f"Endpoint is {status}; scheduling requires a ready endpoint."
            elif authorized and not workers:
                diagnosis = "Serve has not reported any workers. The scheduler dispatches only to workers explicitly reported as idle."
            elif authorized and not idle_workers:
                states = ", ".join(f"{count} {state}" for state, count in sorted(worker_states.items())) or "no worker telemetry"
                diagnosis = f"{authorized} authorized run(s) are blocked: Serve reports no idle worker slot ({states})."
            elif authorized:
                diagnosis = f"{authorized} authorized run(s) can be considered on the next scheduler retry."
            if authorized and (status != "ready" or not idle_workers):
                findings.append(f"{endpoint.get('name') or endpoint_id}: {diagnosis}")
            endpoint_diagnostics.append({
                "endpoint_id": endpoint_id,
                "name": endpoint.get("name") or endpoint_id,
                "status": status,
                "worker_slots": (endpoint.get("capacity") or {}).get("worker_slots", len(workers)),
                "idle_workers": idle_workers,
                "worker_states": worker_states,
                "authorized_runs": authorized,
                "waiting_runs": int(queue_counts.get("waiting_for_resources", 0)),
                "diagnosis": diagnosis,
            })
        return {
            "log_root_configured": self.log_root.is_dir(),
            "logs": logs,
            "scheduler": {"endpoints": endpoint_diagnostics, "findings": findings},
        }

    def record_audit_event(
        self, *, actor_role: str, method: str, path: str, outcome: str,
        request_id: str | None = None,
    ) -> None:
        """Record a bounded control-plane audit fact without request data.

        Callers deliberately pass only role, HTTP verb/path, outcome, and an
        optional correlation id.  Credentials, bodies, filesystem paths, and
        artifact grants are never accepted here, so they cannot leak into the
        durable audit ledger by accident.
        """
        if not isinstance(actor_role, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", actor_role):
            raise ValueError("audit actor_role is invalid")
        if not isinstance(method, str) or not re.fullmatch(r"[A-Z]{3,10}", method):
            raise ValueError("audit method is invalid")
        if not isinstance(path, str) or not path.startswith("/") or len(path) > 512 or "?" in path or "#" in path:
            raise ValueError("audit path is invalid")
        if not isinstance(outcome, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", outcome):
            raise ValueError("audit outcome is invalid")
        if request_id is not None and (not isinstance(request_id, str) or len(request_id) > 128):
            raise ValueError("audit request_id is invalid")
        with self._connection() as db:
            db.execute(
                "INSERT INTO audit_events(timestamp,actor_role,method,path,outcome,request_id) VALUES(?,?,?,?,?,?)",
                (_now(), actor_role, method, path, outcome, request_id),
            )

    # Durable operations ------------------------------------------------
    # Long-lived filesystem scans and remote reconciliation must not happen
    # in a GET request or vanish with an API process.  The runner is deliberately
    # single-slot for SQLite/local-filesystem deployments; a later distributed
    # scheduler can claim the same queued rows.
    def create_operation(self, operation_type: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        if operation_type not in {"catalog_scan", "training_catalog_scan", "artifact_reindex", "worker_reconciliation", "validated_queue_validation", "artifact_replication"}:
            raise ValueError(f"Unsupported operation type: {operation_type}")
        now, operation_id = _now(), str(uuid.uuid4())
        with self._connection() as connection:
            connection.execute("""INSERT INTO operations(operation_id,operation_type,status,parameters_json,created_at,updated_at)
                VALUES(?,?, 'queued',?,?,?)""", (operation_id, operation_type, _json(parameters or {}), now, now))
            # Commit the durable operation and its first event together.  A
            # runner can otherwise claim a just-inserted row between these
            # two writes, making the observable event stream begin at
            # ``started`` rather than ``queued``.
            connection.execute("""INSERT INTO operation_events(operation_id,timestamp,event_type,message,data_json)
                VALUES(?,?,?,?,?)""", (operation_id, now, "queued", "Operation accepted", _json({"status": "queued"})))
        return self.operation(operation_id) or {}

    def operation(self, operation_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM operations WHERE operation_id=?", (operation_id,)).fetchone()
            latest_event = connection.execute(
                "SELECT * FROM operation_events WHERE operation_id=? ORDER BY sequence DESC LIMIT 1",
                (operation_id,),
            ).fetchone()
        result = _row(row)
        if result and result.get("result_json") is not None:
            result["result"] = json.loads(result.pop("result_json"))
        if result:
            result.pop("result_json", None)
            if latest_event is not None:
                event = dict(latest_event)
                event["data"] = json.loads(event.pop("data_json"))
                result["latest_event"] = event
        return result

    def operations(self, *, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM operations " + ("WHERE status=? " if status else "") + "ORDER BY created_at DESC LIMIT ?", ((status, limit) if status else (limit,))).fetchall()
        result = []
        for row in rows:
            item = self.operation(row["operation_id"])
            if item: result.append(item)
        return result

    def operation_events(self, *, after: int = 0, operation_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        query = "SELECT * FROM operation_events WHERE sequence>?"
        values: list[Any] = [after]
        if operation_id:
            query += " AND operation_id=?"; values.append(operation_id)
        query += " ORDER BY sequence ASC LIMIT ?"; values.append(limit)
        with self._connection() as connection: rows = connection.execute(query, values).fetchall()
        return [{key: value for key, value in dict(row).items() if key != "data_json"} | {"data": json.loads(row["data_json"])} for row in rows]

    def emit_operation_event(self, operation_id: str, event_type: str, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        now = _now()
        with self._connection() as connection:
            cursor = connection.execute("""INSERT INTO operation_events(operation_id,timestamp,event_type,message,data_json)
                VALUES(?,?,?,?,?)""", (operation_id, now, event_type, message, _json(data or {})))
            connection.execute("UPDATE operations SET updated_at=? WHERE operation_id=?", (now, operation_id))
            sequence = cursor.lastrowid
        return {"sequence": sequence, "operation_id": operation_id, "timestamp": now, "event_type": event_type, "message": message, "data": data or {}}

    def _run_operation(self, operation: dict[str, Any]) -> None:
        operation_id, kind, parameters = operation["operation_id"], operation["operation_type"], operation["parameters"]
        self.emit_operation_event(operation_id, "started", "Operation started", {"status": "running"})
        try:
            if kind == "catalog_scan": result = self.scan(parameters["root"])
            elif kind == "training_catalog_scan": result = self.scan_training_catalog(parameters.get("root_id"))
            elif kind == "artifact_reindex": result = self.reindex_artifact_catalog(parameters.get("artifact_ids"))
            elif kind == "artifact_replication":
                action = parameters.get("action", "retry")
                refs = [ArtifactRef.from_dict(value) for value in parameters.get("refs", [])]
                if action == "retry": result = {"replications": [self.replicate_artifact(ref) for ref in refs]}
                elif action == "verify": result = {"replications": [self.verify_artifact_replica(ref) for ref in refs]}
                elif action == "restore": result = {"replications": [self.restore_artifact_replica(ref) for ref in refs]}
                else: raise ValueError("artifact replication action must be retry, verify, or restore")
            elif kind == "validated_queue_validation":
                definition_id = parameters.pop("definition_id")
                def progress(event_type: str, message: str, data: dict[str, Any] | None = None) -> None:
                    self.emit_operation_event(operation_id, event_type, message, data)
                queued_run = self.queue_model_definition_for_pool(definition_id, operation_progress=progress, **parameters)
                result = {"queued_run": queued_run, "queued_run_id": queued_run["queued_run_id"]}
            elif kind == "worker_reconciliation":
                result = {
                    "liveness": self.reconcile_worker_liveness(),
                    "published_outputs": self.reconcile_worker_publications(),
                }
            else: raise RuntimeError(f"unsupported durable operation: {kind}")
            now = _now()
            with self._connection() as connection:
                connection.execute("UPDATE operations SET status='completed',result_json=?,completed_at=?,updated_at=? WHERE operation_id=?", (_json(result), now, now, operation_id))
            self.emit_operation_event(operation_id, "completed", "Operation completed", {"status": "completed"})
        except Exception as exc:
            now = _now()
            with self._connection() as connection:
                connection.execute("UPDATE operations SET status='failed',error=?,completed_at=?,updated_at=? WHERE operation_id=?", (str(exc), now, now, operation_id))
            self.emit_operation_event(operation_id, "failed", "Operation failed", {"status": "failed", "error": str(exc)})

    def run_next_operation(self) -> bool:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM operations WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if row is not None:
                now = _now()
                claimed = connection.execute("UPDATE operations SET status='running',started_at=?,updated_at=? WHERE operation_id=? AND status='queued'", (now, now, row["operation_id"])).rowcount
        if row is None: return False
        if claimed != 1: return False
        operation = _row(row) or {}
        self._run_operation(operation)
        return True

    def start_operation_runner(self, *, interval_seconds: float = 0.25, reconciliation_seconds: float = 15.0) -> None:
        if self._operation_thread and self._operation_thread.is_alive(): return
        self._operation_stop = threading.Event()
        def loop() -> None:
            last_reconciliation = 0.0
            while not self._operation_stop.is_set():
                # Pull workers are reconciled through durable leases and their
                # staged publications.  The retired HTTP push scheduler is
                # deliberately not polled or restarted here.
                import time
                if time.monotonic() - last_reconciliation >= reconciliation_seconds:
                    with self._connection() as connection:
                        busy = connection.execute("SELECT 1 FROM operations WHERE operation_type='worker_reconciliation' AND status IN ('queued','running') LIMIT 1").fetchone()
                        active = connection.execute(
                            "SELECT 1 FROM worker_leases WHERE status='active' LIMIT 1"
                        ).fetchone()
                    if busy is None and active is not None:
                        self.create_operation("worker_reconciliation")
                    self.expire_worker_leases()
                    last_reconciliation = time.monotonic()
                if not self.run_next_operation(): self._operation_stop.wait(interval_seconds)
        self._operation_thread = threading.Thread(target=loop, name="oracle-operation-runner", daemon=True)
        self._operation_thread.start()

    def stop_operation_runner(self) -> None:
        if self._operation_stop: self._operation_stop.set()
        if self._operation_thread: self._operation_thread.join(timeout=5)

    # Artifact replica ledger -----------------------------------------
    # Replica implementation details stay inside the injected artifact store;
    # this durable ledger deliberately persists only ArtifactRef, integrity
    # digest, state, and a bounded diagnostic.  Object-store credentials and
    # paths never enter SQLite.
    def _record_artifact_replica_in_connection(
        self, db: sqlite3.Connection, ref: ArtifactRef, *, status: str,
        manifest_sha256: str | None = None, error: str | None = None,
        increment_attempt: bool = False, restored: bool = False, verified: bool = False,
    ) -> dict[str, Any]:
        if status not in {"pending", "replicated", "failed"}:
            raise ValueError("artifact replica status is invalid")
        now = _now()
        # Failure text can come from a third-party client; bound it so the
        # durable operational ledger cannot become an unbounded error sink.
        error = error[:1000] if error else None
        db.execute("""INSERT INTO artifact_replications(
            artifact_ref_uri,artifact_ref_json,status,attempts,manifest_sha256,last_error,replicated_at,verified_at,restored_at,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(artifact_ref_uri) DO UPDATE SET
              artifact_ref_json=excluded.artifact_ref_json,status=excluded.status,
              attempts=artifact_replications.attempts + excluded.attempts,
              manifest_sha256=COALESCE(excluded.manifest_sha256,artifact_replications.manifest_sha256),
              last_error=excluded.last_error,
              replicated_at=COALESCE(excluded.replicated_at,artifact_replications.replicated_at),
              verified_at=COALESCE(excluded.verified_at,artifact_replications.verified_at),
              restored_at=COALESCE(excluded.restored_at,artifact_replications.restored_at),updated_at=excluded.updated_at""",
            (ref.uri, _json(ref.to_dict()), status, 1 if increment_attempt else 0, manifest_sha256, error,
             now if status == "replicated" else None, now if verified else None, now if restored else None, now, now))
        row = db.execute("SELECT * FROM artifact_replications WHERE artifact_ref_uri=?", (ref.uri,)).fetchone()
        return self._replica_row(row)

    @staticmethod
    def _replica_row(row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None: return {}
        result = dict(row)
        result["artifact_ref"] = json.loads(result.pop("artifact_ref_json"))
        return result

    def record_artifact_replica(self, ref: ArtifactRef) -> dict[str, Any]:
        status_method = getattr(self.artifact_store, "replication_status", None)
        if not callable(status_method) and not callable(getattr(self.artifact_store, "replicate", None)):
            return {"artifact_ref": ref.to_dict(), "status": "not_configured"}
        report = status_method(ref) if callable(status_method) else {"status": "pending"}
        status = report.get("status") if isinstance(report, dict) else "pending"
        if status not in {"pending", "replicated", "failed"}: status = "pending"
        with self._connection() as db:
            return self._record_artifact_replica_in_connection(
                db, ref, status=status, manifest_sha256=report.get("manifest_sha256") if isinstance(report, dict) else None,
                error=report.get("error") if isinstance(report, dict) else None,
            )

    def artifact_replications(self, *, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        if status is not None and status not in {"pending", "replicated", "failed"}: raise ValueError("artifact replica status is invalid")
        limit = max(1, min(limit, 500))
        with self._connection() as db:
            rows = db.execute("SELECT * FROM artifact_replications " + ("WHERE status=? " if status else "") + "ORDER BY updated_at DESC LIMIT ?", (status, limit) if status else (limit,)).fetchall()
        return [self._replica_row(row) for row in rows]

    def _replica_capability(self, name: str) -> Callable[[ArtifactRef], dict[str, Any] | Path]:
        method = getattr(self.artifact_store, name, None)
        if not callable(method): raise RuntimeError("Artifact replication is not configured")
        return method

    def replicate_artifact(self, ref: ArtifactRef) -> dict[str, Any]:
        try:
            report = self._replica_capability("replicate")(ref)
            if not isinstance(report, dict): raise RuntimeError("artifact replica returned an invalid report")
            with self._connection() as db:
                return self._record_artifact_replica_in_connection(db, ref, status="replicated", increment_attempt=True,
                    manifest_sha256=report.get("manifest_sha256"))
        except Exception as exc:
            with self._connection() as db:
                return self._record_artifact_replica_in_connection(db, ref, status="failed", increment_attempt=True, error=str(exc))

    def verify_artifact_replica(self, ref: ArtifactRef) -> dict[str, Any]:
        try:
            report = self._replica_capability("verify_replica")(ref)
            if not isinstance(report, dict): raise RuntimeError("artifact replica returned an invalid report")
            with self._connection() as db:
                return self._record_artifact_replica_in_connection(db, ref, status="replicated", verified=True,
                    manifest_sha256=report.get("manifest_sha256"))
        except Exception as exc:
            with self._connection() as db:
                return self._record_artifact_replica_in_connection(db, ref, status="failed", error=str(exc))

    def restore_artifact_replica(self, ref: ArtifactRef) -> dict[str, Any]:
        try:
            self._replica_capability("restore")(ref)
            # Verify after restore, before representing it as usable.
            report = self._replica_capability("verify_replica")(ref)
            if not isinstance(report, dict): raise RuntimeError("artifact replica returned an invalid report")
            with self._connection() as db:
                result = self._record_artifact_replica_in_connection(db, ref, status="replicated", restored=True, verified=True,
                    manifest_sha256=report.get("manifest_sha256"))
            return result | {"restored": True}
        except Exception as exc:
            with self._connection() as db:
                return self._record_artifact_replica_in_connection(db, ref, status="failed", error=str(exc))

    @staticmethod
    def _architecture_config_path(architecture: str) -> Path:
        """Return the explicit V2 starter recipe for a supported architecture.

        Family names remain useful shorthand in the API, but resolve to a
        concrete supported variant.  This deliberately avoids treating the
        old EfficientNet V1 recipe as an EfficientNetV2 baseline.
        """
        root = Path(__file__).resolve().parents[2] / "configs"
        classification = root / "classification_defaults"
        paths = {
            "simple_cnn": classification / "simple_cnn.toml",
            "resnet_like": classification / "resnet_like.toml",
            "densenet_like": classification / "densenet_like.toml",
            "resnet": classification / "resnet18.toml",
            "resnet18": classification / "resnet18.toml",
            "resnet34": classification / "resnet34.toml",
            "resnet50": classification / "resnet50.toml",
            "resnet101": classification / "resnet101.toml",
            "resnet152": classification / "resnet152.toml",
            "densenet": classification / "densenet121.toml",
            "densenet121": classification / "densenet121.toml",
            "densenet169": classification / "densenet169.toml",
            "densenet201": classification / "densenet201.toml",
            "efficientnet": classification / "efficientnet_b0.toml",
            "efficientnet_b0": classification / "efficientnet_b0.toml",
            "efficientnet_b1": classification / "efficientnet_b1.toml",
            "efficientnet_b2": classification / "efficientnet_b2.toml",
            "efficientnet_b3": classification / "efficientnet_b3.toml",
            "efficientnet_b4": classification / "efficientnet_b4.toml",
            "efficientnet_b5": classification / "efficientnet_b5.toml",
            "efficientnet_b6": classification / "efficientnet_b6.toml",
            "efficientnet_b7": classification / "efficientnet_b7.toml",
            "efficientnet_v2": classification / "efficientnet_v2_b0.toml",
            "efficientnet_v2_b0": classification / "efficientnet_v2_b0.toml",
            "efficientnet_v2_b1": classification / "efficientnet_v2_b1.toml",
            "efficientnet_v2_b2": classification / "efficientnet_v2_b2.toml",
            "efficientnet_v2_b3": classification / "efficientnet_v2_b3.toml",
            "efficientnet_v2_s": classification / "efficientnet_v2_s.toml",
            "efficientnet_v2_m": classification / "efficientnet_v2_m.toml",
            "efficientnet_v2_l": classification / "efficientnet_v2_l.toml",
            "efficientnetv2": classification / "efficientnet_v2_b0.toml",
            "efficientnetv2_b0": classification / "efficientnet_v2_b0.toml",
            "efficientnetv2_b1": classification / "efficientnet_v2_b1.toml",
            "efficientnetv2_b2": classification / "efficientnet_v2_b2.toml",
            "efficientnetv2_b3": classification / "efficientnet_v2_b3.toml",
            "efficientnetv2_s": classification / "efficientnet_v2_s.toml",
            "efficientnetv2_m": classification / "efficientnet_v2_m.toml",
            "efficientnetv2_l": classification / "efficientnet_v2_l.toml",
            "convnext": classification / "convnext_tiny.toml",
            "convnext_tiny": classification / "convnext_tiny.toml",
            "convnext_small": classification / "convnext_small.toml",
            "mobilenet": classification / "mobilenet_v3_small.toml",
            "mobilenet_v3_small": classification / "mobilenet_v3_small.toml",
            "mobilenet_v3_large": classification / "mobilenet_v3_large.toml",
            "unet": root / "example_segmentation_unet.toml",
            "residual_unet": root / "example_segmentation_residual_unet.toml",
            "resunet": root / "example_segmentation_residual_unet.toml",
            "unet_plus_plus": root / "example_segmentation_unet_plus_plus.toml",
            "unetpp": root / "example_segmentation_unet_plus_plus.toml",
        }
        try:
            return paths[architecture]
        except KeyError as exc:
            raise ValueError(f"Unsupported architecture: {architecture}") from exc

    def model_setup(self, architecture: str) -> dict[str, Any]:
        """Expose the maintained TOML defaults as a safe UI setup schema."""
        source = load_toml(self._architecture_config_path(architecture))
        source.setdefault("run", {})["model"] = architecture
        config = deep_merge(DEFAULT_CONFIG, source)
        # model_setup is a runtime-preview helper, not the V2 authoring API.
        # Translate its temporary alias so existing builders and previews see
        # the same internal structure they will receive at execution.
        normalize_component_config(config, source)
        fields: list[dict[str, Any]] = []

        def collect(value: dict[str, Any], prefix: str) -> None:
            for key, item in value.items():
                path = f"{prefix}.{key}"
                if isinstance(item, (str, int, float, bool)):
                    fields.append({"path": path, "value": item, "type": "boolean" if isinstance(item, bool) else "number" if isinstance(item, (int, float)) else "text"})
                elif isinstance(item, list):
                    fields.append({"path": path, "value": item, "type": "list"})

        # These are the knobs users can safely understand before later advanced
        # configuration support. They still come directly from each TOML.
        for section in ("data", "model", "training"):
            collect(config.get(section, {}), section)
        return {"architecture": architecture, "task": config["run"].get("task"), "config": config, "fields": fields}

    @staticmethod
    def configuration_schema() -> dict[str, Any]:
        """Return the complete V2 contract for TOML, GUI, and planning clients."""
        return v2_configuration_schema()

    # Model definitions -------------------------------------------------
    # Maintained TOMLs are deliberately read-only sources.  A user must make
    # a definition (or duplicate an existing definition) before changing any
    # scientific setting, which gives queueing a stable revision to pin.
    @property
    def _model_template_root(self) -> Path:
        # Templates ship with Oracle Builder rather than with an arbitrary
        # project workspace.  Definitions persist the template digest, so a
        # later package/template change cannot rewrite prior science.
        return Path(__file__).resolve().parents[2] / "configs" / "classification_defaults"

    @property
    def _first_class_template_root(self) -> Path:
        """Maintained V2 templates whose task is part of the product surface.

        Classification family defaults predate model definitions and retain
        their stable short identifiers (for example ``resnet18``).  New
        first-class workflows live separately so their template names describe
        the scientific task rather than being confused with a family preset.
        """
        return Path(__file__).resolve().parents[2] / "configs" / "model_definition_templates"

    @staticmethod
    def _config_digest(config: dict[str, Any]) -> str:
        return hashlib.sha256(_json(config).encode("utf-8")).hexdigest()

    def _definition_catalog_fingerprint(self) -> str | None:
        # Schema work evolves independently of persistence.  A definition is
        # still valid without a catalog fingerprint; once the catalog is
        # available its immutable fingerprint is recorded on new revisions.
        try:
            schema = self.configuration_schema()
        except (TypeError, ValueError):
            return None
        return schema.get("fingerprint") or schema.get("schema_fingerprint")

    def _definition_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Resolve the reusable model portion of a V2 configuration.

        Training duration is intentionally excluded from a model definition.
        It is an execution decision made when a frozen definition is paired
        with a dataset in the validated queue.
        """
        # Definition records are public V2 documents.  Runtime-only aliases
        # (run.model/model/pretraining) are created only by resolve_config at
        # execution time and never leak back through this API.
        validate_v2_config(config)
        resolved = resolve_v2_config(config)
        if isinstance(resolved.get("training"), dict):
            resolved["training"].pop("epochs", None)
        if isinstance(resolved.get("self_supervised"), dict):
            resolved["self_supervised"].pop("epochs", None)
        return resolved

    def _template_path(self, template_id: str) -> Path:
        if not isinstance(template_id, str) or not template_id or "/" in template_id or "\\" in template_id:
            raise ValueError("Invalid model-definition template ID")
        for item_id, candidate in self._template_sources():
            if item_id == template_id:
                return candidate
        raise KeyError(template_id)

    def _template_sources(self) -> list[tuple[str, Path]]:
        classification = [
            (path.stem, path) for path in sorted(self._model_template_root.glob("*.toml"))
        ]
        first_class = [
            (path.stem, path)
            for path in sorted(self._first_class_template_root.glob("*.toml"))
        ]
        # Only templates which already satisfy the V2 definition contract are
        # offered here.  Historical examples remain runnable through their CLI
        # paths, but cannot silently become authoring templates.
        sources = classification + first_class
        ids = [template_id for template_id, _ in sources]
        if len(ids) != len(set(ids)):
            raise ValueError("Model-definition template IDs must be unique")
        return [
            (template_id, path.resolve())
            for template_id, path in sources
            if path.is_file()
        ]

    def model_definition_templates(self) -> list[dict[str, Any]]:
        templates: list[dict[str, Any]] = []
        for template_id, path in self._template_sources():
            raw = path.read_bytes()
            source = load_toml(path)
            try:
                config = self._definition_config(source)
            except (TypeError, ValueError):
                # A maintained file that is not V2 must never leak into the
                # authoring catalog.  It remains visible on disk for migration.
                continue
            encoder = config.get("encoder") or {}
            templates.append({
                "template_id": template_id,
                "name": template_id.replace("_", " "),
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "task": (config.get("run") or {}).get("task"),
                # These are the user-facing creation dimensions.  ``model``
                # deliberately aliases the V2 encoder family so clients do
                # not need to understand source TOML filenames.
                "model": encoder.get("family"),
                "variant": encoder.get("variant"),
                "architecture": {"family": encoder.get("family"), "variant": encoder.get("variant")},
                "config": config,
            })
        return templates

    def model_definition_template(self, template_id: str) -> dict[str, Any]:
        for template in self.model_definition_templates():
            if template["template_id"] == template_id:
                return template
        raise KeyError(template_id)

    def _definition_result(self, definition: dict[str, Any]) -> dict[str, Any]:
        definition["architecture"] = self.architecture_view_from_config(definition["config"])
        return definition

    def model_definitions(self) -> list[dict[str, Any]]:
        return [self._definition_result(item) for item in self._many("SELECT * FROM model_definitions ORDER BY updated_at DESC")]

    def model_definition(self, definition_id: str, *, revision: int | None = None) -> dict[str, Any] | None:
        with self._connection() as db:
            if revision is None:
                item = _row(db.execute("SELECT * FROM model_definitions WHERE definition_id=?", (definition_id,)).fetchone())
            else:
                item = _row(db.execute("""SELECT d.definition_id,d.name,d.description,r.revision,d.template_id,d.template_sha256,
                    d.parent_definition_id,d.parent_revision,d.lineage_kind,r.config_json,r.config_sha256,r.catalog_fingerprint,
                    d.created_at,r.created_at AS updated_at FROM model_definitions d JOIN model_definition_revisions r
                    ON d.definition_id=r.definition_id WHERE d.definition_id=? AND r.revision=?""", (definition_id, int(revision))).fetchone())
        return self._definition_result(item) if item is not None else None

    def model_definition_revisions(self, definition_id: str) -> list[dict[str, Any]]:
        if self.model_definition(definition_id) is None:
            raise KeyError(definition_id)
        with self._connection() as db:
            rows = db.execute("SELECT definition_id,revision,config_json,config_sha256,catalog_fingerprint,created_at FROM model_definition_revisions WHERE definition_id=? ORDER BY revision DESC", (definition_id,)).fetchall()
        return [_row(row) for row in rows]  # type: ignore[list-item]

    def create_model_definition(self, *, name: str, template_id: str, description: str = "", config: dict[str, Any] | None = None) -> dict[str, Any]:
        template = self.model_definition_template(template_id)
        if config is not None and not isinstance(config, dict):
            raise ValueError("Model definition config must be an object")
        resolved = self._definition_config(deep_merge(template["config"], config or {}))
        definition_id, now = str(uuid.uuid4()), _now()
        digest, fingerprint = self._config_digest(resolved), self._definition_catalog_fingerprint()
        with self._connection() as db:
            definition_name = name.strip() or template["name"]
            try:
                db.execute("""INSERT INTO model_definitions VALUES (?, ?, ?, 1, ?, ?, NULL, NULL, 'template', ?, ?, ?, ?, ?)""",
                           (definition_id, definition_name, description, template_id, template["sha256"], _json(resolved), digest, fingerprint, now, now))
            except sqlite3.IntegrityError as exc:
                if "model_definitions_name_idx" in str(exc) or "model_definitions.name" in str(exc):
                    raise ValueError(f"Model definition name already exists: {definition_name}") from exc
                raise
            db.execute("INSERT INTO model_definition_revisions VALUES (?, 1, ?, ?, ?, ?)", (definition_id, _json(resolved), digest, fingerprint, now))
        return self.model_definition(definition_id)  # type: ignore[return-value]

    def update_model_definition(self, definition_id: str, *, expected_revision: int, name: str | None = None, description: str | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
        current = self.model_definition(definition_id)
        if current is None: raise KeyError(definition_id)
        if int(current["revision"]) != int(expected_revision):
            raise ValueError("Model definition revision conflict; refresh before saving")
        if config is not None and not isinstance(config, dict): raise ValueError("Model definition config must be an object")
        resolved = self._definition_config(config if config is not None else current["config"])
        revision, now = int(current["revision"]) + 1, _now()
        digest, fingerprint = self._config_digest(resolved), self._definition_catalog_fingerprint()
        with self._connection() as db:
            definition_name = name.strip() if isinstance(name, str) and name.strip() else current["name"]
            try:
                updated = db.execute("""UPDATE model_definitions SET name=?,description=?,revision=?,config_json=?,config_sha256=?,catalog_fingerprint=?,updated_at=?
                    WHERE definition_id=? AND revision=?""", (definition_name, description if description is not None else current["description"], revision, _json(resolved), digest, fingerprint, now, definition_id, expected_revision)).rowcount
            except sqlite3.IntegrityError as exc:
                if "model_definitions_name_idx" in str(exc) or "model_definitions.name" in str(exc):
                    raise ValueError(f"Model definition name already exists: {definition_name}") from exc
                raise
            if updated != 1: raise ValueError("Model definition revision conflict; refresh before saving")
            db.execute("INSERT INTO model_definition_revisions VALUES (?, ?, ?, ?, ?, ?)", (definition_id, revision, _json(resolved), digest, fingerprint, now))
        return self.model_definition(definition_id)  # type: ignore[return-value]

    def duplicate_model_definition(self, definition_id: str, *, revision: int | None = None, name: str | None = None, lineage_kind: str = "duplicate") -> dict[str, Any]:
        source = self.model_definition(definition_id, revision=revision)
        if source is None: raise KeyError(definition_id)
        if lineage_kind not in {"duplicate", "sensitivity", "fork"}: raise ValueError("Unsupported definition lineage kind")
        new_id, now = str(uuid.uuid4()), _now()
        digest, fingerprint = self._config_digest(source["config"]), self._definition_catalog_fingerprint()
        requested_name = name.strip() if isinstance(name, str) and name.strip() else None
        with self._connection() as db:
            # Naming has to be decided inside a write transaction.  Otherwise
            # two users duplicating the same definition could both observe
            # "V.2" as available and create indistinguishable library rows.
            db.execute("BEGIN IMMEDIATE")
            duplicate_name = requested_name or self._next_definition_duplicate_name(db, source["name"])
            try:
                db.execute("""INSERT INTO model_definitions VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                           (new_id, duplicate_name, source["description"], source.get("template_id"), source.get("template_sha256"), definition_id, source["revision"], lineage_kind, _json(source["config"]), digest, fingerprint, now, now))
                db.execute("INSERT INTO model_definition_revisions VALUES (?, 1, ?, ?, ?, ?)", (new_id, _json(source["config"]), digest, fingerprint, now))
            except sqlite3.IntegrityError as exc:
                if "model_definitions_name_idx" in str(exc) or "model_definitions.name" in str(exc):
                    raise ValueError(f"Model definition name already exists: {duplicate_name}") from exc
                raise
        return self.model_definition(new_id)  # type: ignore[return-value]

    def delete_model_definition(self, definition_id: str) -> dict[str, Any]:
        """Delete an unused definition without breaking sealed run provenance."""
        current = self.model_definition(definition_id)
        if current is None:
            raise KeyError(definition_id)
        with self._connection() as db:
            references = db.execute(
                "SELECT queued_run_id,status FROM queued_runs WHERE definition_id=? ORDER BY created_at",
                (definition_id,),
            ).fetchall()
            if references:
                states = sorted({str(row["status"]) for row in references})
                raise ValueError(
                    "This definition is pinned by "
                    f"{len(references)} queued or historical run{'s' if len(references) != 1 else ''} "
                    f"({', '.join(states)}), so it cannot be deleted."
                )
            db.execute("DELETE FROM model_definitions WHERE definition_id=?", (definition_id,))
        return {"definition_id": definition_id, "deleted": True, "name": current["name"]}

    @staticmethod
    def _next_definition_duplicate_name(db: sqlite3.Connection, source_name: str) -> str:
        """Choose the next free ``base V.N`` name for an automatic duplicate.

        A copy of either ``baseline`` or ``baseline V.2`` belongs to the same
        sequence.  We start at V.2 for a base definition and immediately after
        a versioned source, then fill the next available number.  The caller
        holds a write transaction, and the unique name index is the final
        guard against collisions from another process.
        """
        match = re.fullmatch(r"(?P<base>.+?)\s+[Vv]\.(?P<version>[1-9]\d*)", source_name.strip())
        base = match.group("base").strip() if match else source_name.strip()
        version = int(match.group("version")) + 1 if match else 2
        while True:
            candidate = f"{base} V.{version}"
            exists = db.execute(
                "SELECT 1 FROM model_definitions WHERE name = ? COLLATE NOCASE LIMIT 1",
                (candidate,),
            ).fetchone()
            if exists is None:
                return candidate
            version += 1

    # Validated queue ---------------------------------------------------
    # Definitions stay editable and versioned.  A queued run instead owns a
    # sealed TOML and dataset pairing, so later edits cannot change queued
    # science or execution behaviour.
    @staticmethod
    def _toml_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: Orchestrator._toml_safe(item) for key, item in value.items() if item is not None}
        if isinstance(value, list):
            return [Orchestrator._toml_safe(item) for item in value if item is not None]
        return value

    def _definition_queue_config(self, definition: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        """Bind trusted dataset facts before sealing a definition revision."""
        config = deep_merge(definition["config"], {})
        dataset_facts: dict[str, Any] = {}
        task = str((config.get("run") or {}).get("task", "classification"))
        if task in {"classification", "embedding"}:
            try:
                with sqlite3.connect(dataset["path"]) as dataset_db:
                    count = int(dataset_db.execute("SELECT count(*) FROM classification_labels").fetchone()[0])
            except sqlite3.DatabaseError as exc:
                raise ValueError("Could not read class labels from the frozen dataset") from exc
            if count < 1:
                raise ValueError("Frozen classification dataset has no class labels")
            dataset_facts["num_classes"] = count
        # Keep the sealed TOML V2-only.  The training runtime translates this
        # isolated input to internal builder aliases while resolving it.
        return resolve_v2_config(config, dataset_facts=dataset_facts)

    def queued_runs(self) -> list[dict[str, Any]]:
        rows = self._many("SELECT * FROM queued_runs WHERE status != 'archived' ORDER BY priority DESC, created_at")
        for row in rows:
            row["start_authorized"] = bool(row.get("start_authorized"))
            definition = self.model_definition(str(row["definition_id"]), revision=int(row["definition_revision"]))
            row["definition_name"] = definition.get("name") if definition else None
            try:
                sealed = load_toml(row["resolved_toml_path"])
                row["batch_size"] = int(sealed.get("data", {}).get("batch_size"))
                row["epochs"] = int(sealed.get("training", {}).get("epochs"))
            except (OSError, TypeError, ValueError):
                row["batch_size"] = None
                row["epochs"] = None
            specification = self.specification(str(row["specification_id"])) if row.get("specification_id") else None
            row["batch_execution"] = (specification or {}).get("parameters", {}).get("queue_execution")
            if (row["batch_execution"] or {}).get("mode") == "verified":
                row["batch_size"] = row["batch_execution"]["batch_size"]
        return rows

    def queued_run(self, queued_run_id: str) -> dict[str, Any] | None:
        with self._connection() as db:
            row = _row(db.execute("SELECT * FROM queued_runs WHERE queued_run_id=?", (queued_run_id,)).fetchone())
        if row is not None:
            row["start_authorized"] = bool(row.get("start_authorized"))
            try:
                sealed = load_toml(row["resolved_toml_path"])
                row["batch_size"] = int(sealed.get("data", {}).get("batch_size"))
                row["epochs"] = int(sealed.get("training", {}).get("epochs"))
            except (OSError, TypeError, ValueError):
                row["batch_size"] = None
                row["epochs"] = None
            specification = self.specification(str(row["specification_id"])) if row.get("specification_id") else None
            row["batch_execution"] = (specification or {}).get("parameters", {}).get("queue_execution")
            if (row["batch_execution"] or {}).get("mode") == "verified":
                row["batch_size"] = row["batch_execution"]["batch_size"]
        return row

    def validate_and_queue_model_definition(
        self,
        definition_id: str,
        *,
        name: str,
        dataset_id: str,
        endpoint_id: str,
        revision: int | None = None,
        description: str = "",
        resources: dict[str, Any] | None = None,
        initialization: dict[str, Any] | None = None,
        batch_size_mode: str = "manual",
        batch_size: int | None = None,
        maximum_batch_size: int = 256,
        epochs: int = 10,
        operation_progress: Callable[[str, str, dict[str, Any] | None], None] | None = None,
    ) -> dict[str, Any]:
        def progress(event_type: str, message: str, data: dict[str, Any] | None = None) -> None:
            if operation_progress is not None:
                operation_progress(event_type, message, data)

        progress("validating", "Validating definition, frozen dataset, and requested resources")
        if not name.strip():
            raise ValueError("A queued run requires a name")
        definition = self.model_definition(definition_id, revision=revision)
        dataset = self.dataset(dataset_id)
        if definition is None:
            raise KeyError("Model definition was not found")
        if dataset is None:
            raise KeyError("Dataset was not found")
        if dataset.get("lifecycle") != "frozen":
            raise ValueError("Queued training requires a frozen registered dataset")
        endpoint = self.compute_endpoint(endpoint_id)
        if endpoint is None:
            raise KeyError("Compute endpoint was not found")
        requested_resources = dict(resources or {})
        raw_gpu_ids = requested_resources.get("gpu_ids")
        if raw_gpu_ids is not None:
            if not isinstance(raw_gpu_ids, list) or any(not isinstance(item, (str, int)) or not str(item).strip() for item in raw_gpu_ids):
                raise ValueError("GPU allocation must be a list of non-empty GPU IDs")
            gpu_ids = [str(item).strip() for item in raw_gpu_ids]
            if len(gpu_ids) != len(set(gpu_ids)):
                raise ValueError("GPU allocation cannot contain the same GPU more than once")
            if len(gpu_ids) > 1:
                raise ValueError("Validated Queue currently allocates one GPU per training run")
            gpu_count = requested_resources.get("gpu_count", len(gpu_ids))
            if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count != len(gpu_ids):
                raise ValueError("gpu_count must equal the number of selected gpu_ids")
            # Preserve an explicit empty allocation: it is an intentional CPU
            # run, not an omitted legacy GPU request.
            requested_resources["gpu_ids"] = gpu_ids
            requested_resources["gpu_count"] = gpu_count
        else:
            gpu_count = requested_resources.get("gpu_count", 0)
        if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 0:
            raise ValueError("GPU request must be a non-negative integer")
        if batch_size_mode not in {"manual", "auto"}:
            raise ValueError("batch_size_mode must be 'manual' or 'auto'")
        if batch_size is not None and (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")
        if isinstance(maximum_batch_size, bool) or not isinstance(maximum_batch_size, int) or maximum_batch_size < 1:
            raise ValueError("maximum_batch_size must be a positive integer")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if batch_size_mode == "auto" and batch_size is not None:
            raise ValueError("batch_size is only valid with manual batch_size_mode")
        queue_id, experiment_id, specification_id, now = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4()), _now()
        config = self._definition_queue_config(definition, dataset)
        # Run length belongs to this sealed dataset/definition pairing, not to
        # the reusable model definition.  The resolved TOML remains the audit
        # record of the exact epoch budget that was executed.
        config.setdefault("training", {})["epochs"] = epochs
        if (config.get("self_supervised") or {}).get("enabled"):
            config.setdefault("self_supervised", {})["epochs"] = epochs
        # Queue allocation owns physical-device choice.  The runtime sees just
        # that selected device via CUDA_VISIBLE_DEVICES, so use one logical
        # device rather than allowing the legacy automatic selector to choose
        # another host GPU.
        if "gpu_ids" in requested_resources:
            distribution = config.setdefault("distribution", {})
            distribution["strategy"] = "single" if requested_resources["gpu_ids"] else "cpu"
            distribution["devices"] = []
        if batch_size_mode == "manual":
            resolved_batch_size = batch_size if batch_size is not None else int(config.get("data", {}).get("batch_size", 16))
            config.setdefault("data", {})["batch_size"] = resolved_batch_size
            batch_execution: dict[str, Any] = {"mode": "manual", "batch_size": resolved_batch_size}
        else:
            probe_minimum = max(1, int(requested_resources.get("gpu_count", 0) or len(requested_resources.get("gpu_ids", [])) or 1))
            batch_execution = {
                "mode": "auto", "minimum_batch_size": probe_minimum,
                "maximum_batch_size": maximum_batch_size,
                "target_vram_fraction": {"minimum": 0.30, "maximum": 0.80},
            }
        queue_dir = self.artifact_root / "queued-runs" / queue_id
        queue_dir.mkdir(parents=True, exist_ok=False)
        config_path = queue_dir / "resolved.toml"
        import tomli_w
        config_path.write_text(tomli_w.dumps(self._toml_safe(config)), encoding="utf-8")
        config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        parameters = self._assign_output_path(
            "train",
            {
                "config": str(config_path), "input": dataset["path"], "dataset_id": dataset_id,
                "definition_id": definition_id, "definition_revision": int(definition["revision"]),
                "initialization": dict(initialization or {}),
                "queue_execution": {**batch_execution, "epochs": epochs},
            },
            specification_id,
        )
        plan = {
            "kind": "validated_queue", "queued_run_id": queue_id,
            "definition_id": definition_id, "definition_revision": int(definition["revision"]),
            "dataset_id": dataset_id, "dataset_fingerprint_sha256": dataset.get("fingerprint_sha256"),
            "batch_execution": {**batch_execution, "epochs": epochs},
        }
        with self._connection() as db:
            db.execute("INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                experiment_id, None, name.strip(), description, dataset_id, "queued", _json(plan), now, now,
            ))
            db.execute("INSERT INTO run_specifications VALUES (?, ?, 1, ?, 'train', ?, ?, ?, 'planned', NULL, ?, ?)", (
                specification_id, experiment_id, name.strip(), _json(parameters), _json(requested_resources),
                hashlib.sha256(_json({"parameters": parameters, "config_sha256": config_digest}).encode()).hexdigest(), now, now,
            ))
        # Both the orchestrator and compute host inspect the immutable inputs.
        # A host-side preflight is advisory when telemetry is unavailable; its
        # provenance stays attached to the queue row and is rechecked at launch.
        progress("preflight", "Checking compute endpoint capacity and compatibility", {"endpoint_id": endpoint_id})
        report = self.preflight(specification_id, endpoint_id)
        if report.get("ready") and batch_size_mode == "auto":
            capable_workers = report.get("capable_workers") or []
            idle_workers = [worker for worker in capable_workers if worker.get("status") == "idle"]
            if not idle_workers:
                states = sorted({str(worker.get("status") or "unknown") for worker in capable_workers})
                report["ready"] = False
                report.setdefault("reasons", []).append(
                    "Batch-size auto-tuning requires an idle compute worker"
                    + (f" (currently: {', '.join(states)})" if states else "")
                    + ". Wait for the current training or calibration to finish, then retry."
                )
        if report.get("ready") and batch_size_mode == "auto":
            try:
                progress("calibrating", "Calibrating a safe batch size on the selected compute resource", {"maximum_batch_size": maximum_batch_size})
                tune_report = self._request(endpoint["base_url"], "POST", "/compute/batch-size-tune", {
                    "parameters": {
                        "config": str(config_path), "input": dataset["path"],
                        "minimum_batch_size": probe_minimum, "maximum_batch_size": maximum_batch_size,
                        "safety_factor": 0.8, "target_vram_min": 0.30, "target_vram_max": 0.80,
                    },
                    "resources": requested_resources,
                }, timeout_seconds=930)
                report["batch_tune"] = tune_report
                if tune_report.get("ready") and isinstance(tune_report.get("recommended_batch_size"), int):
                    resolved_batch_size = int(tune_report["recommended_batch_size"])
                    config.setdefault("data", {})["batch_size"] = resolved_batch_size
                    batch_execution = {
                        "mode": "auto", "batch_size": resolved_batch_size,
                        "maximum_batch_size": maximum_batch_size,
                        "probe_kind": tune_report.get("probe_kind"),
                        "largest_verified_batch_size": tune_report.get("largest_verified_batch_size"),
                        "tuning_strategy": tune_report.get("tuning_strategy"),
                        "target_vram_fraction": tune_report.get("target_vram_fraction"),
                        "vram_total_mib": tune_report.get("vram_total_mib"),
                        "selected_peak_memory_mib": tune_report.get("selected_peak_memory_mib"),
                    }
                    parameters["queue_execution"] = {**batch_execution, "epochs": epochs}
                    plan["batch_execution"] = {**batch_execution, "epochs": epochs}
                    import tomli_w
                    config_path.write_text(tomli_w.dumps(self._toml_safe(config)), encoding="utf-8")
                    config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
                    with self._connection() as db:
                        db.execute(
                            "UPDATE run_specifications SET parameters_json=?, config_hash=?, updated_at=? WHERE specification_id=?",
                            (
                                _json(parameters),
                                hashlib.sha256(_json({"parameters": parameters, "config_sha256": config_digest}).encode()).hexdigest(),
                                _now(), specification_id,
                            ),
                        )
                        db.execute(
                            "UPDATE experiments SET plan_json=?, updated_at=? WHERE experiment_id=?",
                            (_json(plan), _now(), experiment_id),
                        )
                else:
                    report["ready"] = False
                    report.setdefault("reasons", []).extend(tune_report.get("reasons") or ["Batch-size auto-tuning did not produce a safe batch size"])
            except RuntimeError as exc:
                report["ready"] = False
                report.setdefault("reasons", []).append(f"Batch-size auto-tuning unavailable: {exc}")
        if report.get("ready"):
            try:
                progress("compute_preflight", "Running final compute-host preflight", {"endpoint_id": endpoint_id})
                compute_report = self._request(endpoint["base_url"], "POST", "/compute/preflight", {
                    "action": "train", "parameters": parameters, "resources": requested_resources,
                })
                report["compute_preflight"] = compute_report
                if not compute_report.get("ready"):
                    report["ready"] = False
                    report.setdefault("reasons", []).extend(compute_report.get("reasons") or ["Compute preflight failed"])
            except RuntimeError as exc:
                report["ready"] = False
                report.setdefault("reasons", []).append(f"Compute preflight unavailable: {exc}")
        status = "ready" if report.get("ready") else "needs_attention"
        schema_fingerprint = self._definition_catalog_fingerprint()
        with self._connection() as db:
            db.execute("""INSERT INTO queued_runs (
                queued_run_id,definition_id,definition_revision,dataset_id,dataset_fingerprint_sha256,
                name,description,specification_id,resolved_toml_path,resolved_toml_sha256,
                config_schema_fingerprint,resources_json,initialization_json,preflight_endpoint_id,
                preflight_status,preflight_report_json,status,start_authorized,priority,failure_reason,
                created_at,updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                queue_id, definition_id, int(definition["revision"]), dataset_id, dataset.get("fingerprint_sha256"),
                name.strip(), description, specification_id, str(config_path), config_digest, schema_fingerprint,
                _json(requested_resources), _json(initialization or {}), endpoint_id, "valid" if report.get("ready") else "invalid",
                _json(report), status, 0, 0, "; ".join(report.get("reasons") or []) or None, now, now,
            ))
        progress("queued_run_created", "Validated run was added to the queue", {"queued_run_id": queue_id, "status": status})
        return self.queued_run(queue_id)  # type: ignore[return-value]

    def queue_model_definition_for_pool(
        self,
        definition_id: str,
        *,
        name: str,
        dataset_id: str,
        worker_pool_id: str,
        revision: int | None = None,
        description: str = "",
        resources: dict[str, Any] | None = None,
        initialization: dict[str, Any] | None = None,
        batch_size_mode: str = "manual",
        batch_size: int | None = None,
        maximum_batch_size: int = 256,
        epochs: int = 10,
        operation_progress: Callable[[str, str, dict[str, Any] | None], None] | None = None,
    ) -> dict[str, Any]:
        """Pin a definition and frozen input without verifying or training.

        Input artifacts capture the requested revision. A later verification
        job performs hardware preflight; training needs separate consent.
        """
        from oracle_builder.orchestration.storage import artifact_ref_for_file

        def progress(event_type: str, message: str, data: dict[str, Any] | None = None) -> None:
            if operation_progress is not None:
                operation_progress(event_type, message, data)

        progress("validating", "Sealing definition, frozen dataset, and portable inputs")
        if not name.strip():
            raise ValueError("A queued run requires a name")
        if batch_size_mode not in {"manual", "auto"}:
            raise ValueError("batch_size_mode must be 'manual' or 'auto'")
        if batch_size is not None and (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if isinstance(maximum_batch_size, bool) or not isinstance(maximum_batch_size, int) or maximum_batch_size < 1:
            raise ValueError("maximum_batch_size must be a positive integer")
        if batch_size_mode == "auto" and batch_size is not None:
            raise ValueError("batch_size is only valid with automatic batch calibration disabled")
        definition = self.model_definition(definition_id, revision=revision)
        dataset = self.dataset(dataset_id)
        pool = self.worker_pool(worker_pool_id)
        if definition is None:
            raise KeyError("Model definition was not found")
        if dataset is None:
            raise KeyError("Dataset was not found")
        if pool is None:
            raise KeyError("Worker pool was not found")
        if not pool["enabled"] or "train" not in set(pool["allowed_actions"]):
            raise ValueError("Worker pool is not enabled for train work")
        if dataset.get("lifecycle") != "frozen":
            raise ValueError("Queued training requires a frozen registered dataset")
        requested_resources = dict(resources or {})
        gpu_count = requested_resources.get("gpu_count", 0)
        if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 0:
            raise ValueError("GPU request must be a non-negative integer")
        queue_id, experiment_id, specification_id, now = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4()), _now()
        config = self._definition_queue_config(definition, dataset)
        config.setdefault("training", {})["epochs"] = epochs
        if (config.get("self_supervised") or {}).get("enabled"):
            config.setdefault("self_supervised", {})["epochs"] = epochs
        # Auto calibration is intentionally deferred to the worker that will
        # execute the run.  Only that worker knows its accelerator, runtime,
        # and current isolation boundary.  The configuration artifact remains
        # an immutable baseline; the selected batch size is applied to a
        # worker-local copy immediately before training.
        resolved_batch_size = batch_size if batch_size is not None else int(config.get("data", {}).get("batch_size", 16))
        if batch_size_mode == "manual":
            config.setdefault("data", {})["batch_size"] = resolved_batch_size
        queue_dir = self.artifact_root / "queued-runs" / queue_id
        queue_dir.mkdir(parents=True, exist_ok=False)
        config_path = queue_dir / "resolved.toml"
        import tomli_w
        config_path.write_text(tomli_w.dumps(self._toml_safe(config)), encoding="utf-8")
        config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        dataset_ref = artifact_ref_for_file(
            "dataset", dataset_id, dataset["path"], revision=str(dataset["revision_id"]) if dataset.get("revision_id") else None,
        )
        config_ref = artifact_ref_for_file("configuration", specification_id, config_path, revision="1")
        self.artifact_store.ingest_file(dataset_ref, dataset["path"])
        self.artifact_store.ingest_file(config_ref, config_path)
        # Portable inputs are published artifacts too.  Record their replica
        # outcome immediately; a replica outage must not prevent local queue
        # creation, but it must remain visible and repairable.
        self.record_artifact_replica(dataset_ref)
        self.record_artifact_replica(config_ref)
        if batch_size_mode == "manual":
            execution: dict[str, Any] = {"mode": "manual", "batch_size": resolved_batch_size, "epochs": epochs}
        else:
            execution = {
                "mode": "auto",
                "minimum_batch_size": max(1, gpu_count or 1),
                "maximum_batch_size": maximum_batch_size,
                "target_vram_fraction": {"minimum": 0.30, "maximum": 0.80},
                "epochs": epochs,
            }
        # New queue intake is V2-only even though the eventual training unit
        # is created after verification. This prevents a V1 worker from
        # silently producing a legacy preflight report for a resumable run.
        execution["required_execution_contract_version"] = 2
        parameters = {
            "artifact_inputs": {"input": dataset_ref.to_dict()},
            "configuration_artifact": config_ref.to_dict(),
            "dataset_id": dataset_id,
            "definition_id": definition_id,
            "definition_revision": int(definition["revision"]),
            "initialization": dict(initialization or {}),
            "queue_execution": execution,
        }
        plan = {
            "kind": "validated_queue", "queued_run_id": queue_id,
            "definition_id": definition_id, "definition_revision": int(definition["revision"]),
            "dataset_id": dataset_id, "dataset_fingerprint_sha256": dataset.get("fingerprint_sha256"),
            "worker_pool_id": worker_pool_id, "batch_execution": execution,
        }
        with self._connection() as db:
            db.execute("INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                experiment_id, None, name.strip(), description, dataset_id, "queued", _json(plan), now, now,
            ))
            db.execute("INSERT INTO run_specifications VALUES (?, ?, 1, ?, 'train', ?, ?, ?, 'planned', NULL, ?, ?)", (
                specification_id, experiment_id, name.strip(), _json(parameters), _json(requested_resources),
                hashlib.sha256(_json({"parameters": parameters, "config_sha256": config_digest}).encode()).hexdigest(), now, now,
            ))
            db.execute(
                """INSERT INTO queued_runs (
                   queued_run_id,definition_id,definition_revision,dataset_id,dataset_fingerprint_sha256,
                   name,description,specification_id,resolved_toml_path,resolved_toml_sha256,
                   config_schema_fingerprint,resources_json,initialization_json,worker_pool_id,
                   preflight_endpoint_id,preflight_status,preflight_report_json,status,start_authorized,
                   priority,failure_reason,created_at,updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, 'pending_verification', 0, 0, NULL, ?, ?)""",
                (queue_id, definition_id, int(definition["revision"]), dataset_id, dataset.get("fingerprint_sha256"),
                 name.strip(), description, specification_id, str(config_path), config_digest,
                 self._definition_catalog_fingerprint(), _json(requested_resources), _json(initialization or {}),
                 worker_pool_id, "pending", _json({"ready": False, "reasons": ["Verification has not been requested"]}), now, now),
            )
        progress("queued_run_created", "Portable run was added to the worker-pool queue", {"queued_run_id": queue_id, "worker_pool_id": worker_pool_id})
        return self.queued_run(queue_id)  # type: ignore[return-value]

    def verify_queued_runs_for_pool(
        self, *, queued_run_ids: list[str], start_after_verification: bool = False,
    ) -> dict[str, Any]:
        """Schedule hardware preflight, with explicit optional training consent.

        Selection and job creation are one transaction. No worker can observe
        a partially authorized batch and repeated requests cannot duplicate work.
        """
        selected = list(dict.fromkeys(queued_run_ids))
        if not selected:
            raise ValueError("Select at least one run to verify")
        jobs = []
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            for queue_id in selected:
                row = db.execute("SELECT * FROM queued_runs WHERE queued_run_id=?", (queue_id,)).fetchone()
                if row is None:
                    raise KeyError(queue_id)
                if row["status"] not in {"pending_verification", "needs_attention", "ready"} or row["start_authorized"]:
                    raise ValueError("Only unstarted runs can be verified")
                spec = self.specification(row["specification_id"], connection=db)
                parameters = dict(spec["parameters"])
                execution = dict(parameters["queue_execution"])
                # Reverification repeats the originally requested batch policy.
                execution = dict(execution.get("requested_policy") or execution)
                execution["phase"] = "verify"
                parameters["queue_execution"] = execution
                db.execute("UPDATE run_specifications SET parameters_json=?,status='planned',updated_at=? WHERE specification_id=?",
                           (_json(parameters), _now(), row["specification_id"]))
                db.execute("UPDATE queued_runs SET preflight_status='pending',preflight_report_json=?,start_authorized=?,failure_reason=NULL WHERE queued_run_id=?",
                           (_json({"ready": False, "reasons": ["Waiting for worker preflight"]}), int(start_after_verification), queue_id))
                jobs.append(self._enqueue_specification_in_connection(db, row["specification_id"], row["worker_pool_id"]))
        return {"verification_ids": selected, "dispatched": jobs}

    def authorize_queued_runs_for_pool(
        self, *, queued_run_ids: list[str] | None, worker_pool_id: str, all_ready: bool = False
    ) -> dict[str, Any]:
        """Start only successfully verified runs, atomically across a selection."""
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            selected = list(dict.fromkeys(queued_run_ids or []))
            if all_ready:
                selected = [row[0] for row in db.execute(
                    "SELECT queued_run_id FROM queued_runs WHERE status='ready' AND preflight_status='valid' AND worker_pool_id=?", (worker_pool_id,))]
            if not selected:
                raise ValueError("Select at least one verified run")
            jobs = []
            for queue_id in selected:
                row = db.execute("SELECT * FROM queued_runs WHERE queued_run_id=?", (queue_id,)).fetchone()
                if row is None:
                    raise KeyError(queue_id)
                if row["status"] != "ready" or row["preflight_status"] != "valid" or row["worker_pool_id"] != worker_pool_id:
                    raise ValueError("Only verified runs assigned to this worker pool can be started")
                db.execute("UPDATE queued_runs SET start_authorized=1 WHERE queued_run_id=?", (queue_id,))
                jobs.append(self._enqueue_specification_in_connection(db, row["specification_id"], worker_pool_id))
        return {"authorized_ids": selected, "dispatched": jobs, "queued_runs": [self.queued_run(item) for item in selected]}

    def authorize_queued_runs(
        self, *, queued_run_ids: list[str] | None, endpoint_id: str, all_ready: bool = False
    ) -> dict[str, Any]:
        endpoint = self.compute_endpoint(endpoint_id)
        if endpoint is None:
            raise KeyError("Compute endpoint was not found")
        if all_ready:
            selected = [
                row["queued_run_id"] for row in self.queued_runs()
                if row["status"] == "ready" and row.get("preflight_endpoint_id") == endpoint_id
            ]
        else:
            selected = list(dict.fromkeys(str(item) for item in (queued_run_ids or []) if item))
        if not selected:
            raise ValueError("Select at least one ready queued run")
        placeholders = ",".join("?" for _ in selected)
        with self._connection() as db:
            rows = db.execute(f"SELECT queued_run_id,status,preflight_endpoint_id FROM queued_runs WHERE queued_run_id IN ({placeholders})", selected).fetchall()
            if len(rows) != len(selected):
                raise KeyError("One or more queued runs were not found")
            blocked = [
                row["queued_run_id"] for row in rows
                if row["status"] != "ready" or row["preflight_endpoint_id"] != endpoint_id
            ]
            if blocked:
                raise ValueError("Only ready runs validated for this compute endpoint can be started")
            db.execute(f"UPDATE queued_runs SET start_authorized=1, updated_at=? WHERE queued_run_id IN ({placeholders})", [_now(), *selected])
        dispatched = self.schedule_queued_runs(endpoint_id)
        return {"authorized_ids": selected, "dispatched": dispatched, "queued_runs": [self.queued_run(item) for item in selected]}

    def _release_expired_dispatch_claims(self, endpoint_id: str) -> None:
        """Recover claims abandoned by a stopped scheduler process.

        A claim is only held around synchronous preflight/submission, normally
        a few seconds.  Five minutes is deliberately generous so a slow
        endpoint cannot create a duplicate remote submission.
        """
        expires_at = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        with self._connection() as db:
            db.execute(
                """UPDATE queued_runs
                   SET status='waiting_for_resources', dispatch_claim_token=NULL,
                       dispatch_claim_owner=NULL, dispatch_claimed_at=NULL,
                       failure_reason=COALESCE(failure_reason, 'Dispatch claim expired; retrying automatically'),
                       updated_at=?
                   WHERE preflight_endpoint_id=? AND status='dispatching'
                     AND dispatch_claimed_at IS NOT NULL AND dispatch_claimed_at < ?""",
                (_now(), endpoint_id, expires_at),
            )

    @staticmethod
    def _resource_request(resources: dict[str, Any] | None) -> tuple[int, list[str] | None]:
        request = resources or {}
        cpu = request.get("cpu_count", request.get("cpu_cores", 1))
        cpu_count = int(cpu) if isinstance(cpu, int) and not isinstance(cpu, bool) and cpu > 0 else 1
        explicit = request.get("gpu_ids")
        if isinstance(explicit, list):
            return cpu_count, [str(item) for item in explicit]
        gpu_count = request.get("gpu_count", 0)
        return cpu_count, None if isinstance(gpu_count, int) and gpu_count > 0 else []

    def _scheduler_capacity(self, endpoint: dict[str, Any]) -> dict[str, Any]:
        """Make a conservative local reservation view from Serve's snapshot."""
        workers = endpoint.get("workers") or []
        idle = [worker for worker in workers if worker.get("status") == "idle"]
        gpu_ids = {
            str(gpu.get("id")) for worker in workers
            for gpu in ((worker.get("capabilities") or {}).get("gpus") or [])
            if isinstance(gpu, dict) and gpu.get("id") is not None
        }
        resources = (endpoint.get("queue") or {}).get("resources") or {}
        leased = {str(item) for item in resources.get("gpu_leases") or []}
        cpu_capacity = resources.get("cpu_capacity")
        cpu_in_use = resources.get("cpu_in_use", 0)
        cpu_free = None
        if isinstance(cpu_capacity, int) and isinstance(cpu_in_use, int):
            cpu_free = max(0, cpu_capacity - cpu_in_use)
        return {"slots": len(idle), "gpu_ids": gpu_ids, "leased_gpus": leased, "cpu_free": cpu_free}

    def _claim_queued_run(self, endpoint_id: str, queued_run_id: str, owner: str) -> dict[str, Any] | None:
        token, now = str(uuid.uuid4()), _now()
        with self._connection() as db:
            result = db.execute(
                """UPDATE queued_runs SET status='dispatching', dispatch_claim_token=?,
                       dispatch_claim_owner=?, dispatch_claimed_at=?, dispatch_attempt=dispatch_attempt+1,
                       failure_reason=NULL, updated_at=?
                   WHERE queued_run_id=? AND preflight_endpoint_id=? AND start_authorized=1
                     AND status IN ('ready','waiting_for_resources') AND dispatch_claim_token IS NULL""",
                (token, owner, now, now, queued_run_id, endpoint_id),
            )
            if result.rowcount != 1:
                return None
            return _row(db.execute("SELECT * FROM queued_runs WHERE queued_run_id=?", (queued_run_id,)).fetchone())

    def _release_dispatch_claim(self, queued_run_id: str, *, reason: str, report: dict[str, Any] | None = None) -> None:
        with self._connection() as db:
            db.execute(
                """UPDATE queued_runs SET status='waiting_for_resources', dispatch_claim_token=NULL,
                   dispatch_claim_owner=NULL, dispatch_claimed_at=NULL, failure_reason=?,
                   preflight_report_json=COALESCE(?, preflight_report_json), updated_at=?
                   WHERE queued_run_id=? AND status='dispatching'""",
                (reason, _json({"launch_preflight": report}) if report is not None else None, _now(), queued_run_id),
            )

    def _acquire_scheduler_lease(self, endpoint_id: str, owner: str) -> bool:
        """Claim an endpoint-wide capacity snapshot for one scheduler tick."""
        expires_at = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        with self._connection() as db:
            claimed = db.execute(
                """INSERT INTO scheduler_leases(endpoint_id,owner,claimed_at) VALUES(?,?,?)
                   ON CONFLICT(endpoint_id) DO UPDATE SET owner=excluded.owner, claimed_at=excluded.claimed_at
                   WHERE scheduler_leases.claimed_at < ?""",
                (endpoint_id, owner, _now(), expires_at),
            ).rowcount
        return claimed == 1

    def _release_scheduler_lease(self, endpoint_id: str, owner: str) -> None:
        with self._connection() as db:
            db.execute("DELETE FROM scheduler_leases WHERE endpoint_id=? AND owner=?", (endpoint_id, owner))

    def schedule_queued_runs(self, endpoint_id: str) -> list[dict[str, Any]]:
        """Fill capacity, serializing each endpoint's volatile snapshot."""
        if self.compute_endpoint(endpoint_id) is None:
            raise KeyError("Compute endpoint was not found")
        owner = str(uuid.uuid4())
        if not self._acquire_scheduler_lease(endpoint_id, owner):
            return []
        try:
            return self._schedule_queued_runs_locked(endpoint_id, owner)
        finally:
            self._release_scheduler_lease(endpoint_id, owner)

    def _schedule_queued_runs_locked(self, endpoint_id: str, owner: str) -> list[dict[str, Any]]:
        """Fill currently advertised Serve capacity with authorized sealed runs.

        The endpoint is refreshed once, then this invocation reserves slots,
        CPU and GPU IDs locally while it submits.  Serve remains the final
        resource authority; the durable claim prevents concurrent scheduler
        invocations from submitting a queue row twice.
        """
        endpoint = self.refresh_compute_endpoint(endpoint_id)
        if endpoint is None:
            raise KeyError("Compute endpoint was not found")
        self._release_expired_dispatch_claims(endpoint_id)
        if not endpoint.get("enabled") or endpoint.get("status") != "ready":
            return []
        capacity = self._scheduler_capacity(endpoint)
        if capacity["slots"] < 1:
            states: dict[str, int] = {}
            for worker in endpoint.get("workers") or []:
                state = str(worker.get("status") or "unknown")
                states[state] = states.get(state, 0) + 1
            telemetry = ", ".join(f"{count} {state}" for state, count in sorted(states.items())) or "no worker telemetry"
            reason = f"Waiting for an idle compute worker slot ({telemetry})"
            with self._connection() as db:
                db.execute(
                    """UPDATE queued_runs SET status='waiting_for_resources', failure_reason=?, updated_at=?
                       WHERE preflight_endpoint_id=? AND start_authorized=1
                         AND status IN ('ready', 'waiting_for_resources')""",
                    (reason, _now(), endpoint_id),
                )
            return []
        dispatched = []
        # A bounded candidate snapshot means a malformed/high-priority row
        # cannot starve a compatible one behind it.
        with self._connection() as db:
            candidates = [_row(row) for row in db.execute(
                """SELECT * FROM queued_runs WHERE preflight_endpoint_id=? AND start_authorized=1
                   AND status IN ('ready','waiting_for_resources')
                   ORDER BY priority DESC, created_at""", (endpoint_id,)
            ).fetchall()]
        for candidate in candidates:
            if candidate is None or capacity["slots"] < 1:
                break
            resources = candidate.get("resources") or {}
            cpu_count, explicit_gpu_ids = self._resource_request(resources)
            requested_gpu_count = int(resources.get("gpu_count") or 0)
            used_gpus = capacity["leased_gpus"]
            if capacity["cpu_free"] is not None and cpu_count > capacity["cpu_free"]:
                with self._connection() as db:
                    db.execute("UPDATE queued_runs SET status='waiting_for_resources', failure_reason=?, updated_at=? WHERE queued_run_id=? AND status IN ('ready','waiting_for_resources')", ("Waiting for CPU capacity", _now(), candidate["queued_run_id"]))
                continue
            if explicit_gpu_ids is not None:
                required_gpus = set(explicit_gpu_ids)
                if required_gpus & used_gpus:
                    # No claim is necessary to report a wait; a running
                    # scheduler must not hold a lease while capacity is absent.
                    with self._connection() as db:
                        db.execute("UPDATE queued_runs SET status='waiting_for_resources', failure_reason=?, updated_at=? WHERE queued_run_id=? AND status IN ('ready','waiting_for_resources')", ("Waiting for selected GPU(s): " + ", ".join(sorted(required_gpus & used_gpus)), _now(), candidate["queued_run_id"]))
                    continue
                if required_gpus - capacity["gpu_ids"]:
                    with self._connection() as db:
                        db.execute("UPDATE queued_runs SET status='waiting_for_resources', failure_reason=?, updated_at=? WHERE queued_run_id=? AND status IN ('ready','waiting_for_resources')", ("Selected GPU(s) are unavailable: " + ", ".join(sorted(required_gpus - capacity["gpu_ids"])), _now(), candidate["queued_run_id"]))
                    continue
            elif requested_gpu_count:
                free = capacity["gpu_ids"] - used_gpus
                if len(free) < requested_gpu_count:
                    with self._connection() as db:
                        db.execute("UPDATE queued_runs SET status='waiting_for_resources', failure_reason=?, updated_at=? WHERE queued_run_id=? AND status IN ('ready','waiting_for_resources')", ("Waiting for GPU capacity", _now(), candidate["queued_run_id"]))
                    continue
                required_gpus = set(sorted(free)[:requested_gpu_count])
            else:
                # Serve preserves legacy unconstrained CUDA visibility by
                # leasing every GPU; reserve the same set here.
                required_gpus = set(capacity["gpu_ids"])
                if required_gpus & used_gpus:
                    continue
            row = self._claim_queued_run(endpoint_id, candidate["queued_run_id"], owner)
            if row is None:
                continue
            specification = self.specification(str(row["specification_id"]))
            if specification is None:
                self._release_dispatch_claim(row["queued_run_id"], reason="Queued run specification was not found")
                continue
            try:
                compute_report = self._request(endpoint["base_url"], "POST", "/compute/preflight", {
                    "action": specification["action"], "parameters": specification["parameters"],
                    "resources": specification["resources"],
                })
            except RuntimeError as exc:
                compute_report = {"ready": False, "reasons": [f"Compute preflight unavailable: {exc}"]}
            if not compute_report.get("ready"):
                self._release_dispatch_claim(row["queued_run_id"], reason="; ".join(compute_report.get("reasons") or ["Compute capacity is no longer ready"]), report=compute_report)
                continue
            try:
                job = self.dispatch(str(row["specification_id"]), endpoint_id)
            except (RuntimeError, ValueError) as exc:
                self._release_dispatch_claim(row["queued_run_id"], reason=str(exc))
                continue
            with self._connection() as db:
                db.execute("UPDATE jobs SET queued_run_id=? WHERE job_id=?", (row["queued_run_id"], job["job_id"]))
                db.execute("""UPDATE queued_runs SET status='submitted', dispatch_claim_token=NULL,
                    dispatch_claim_owner=NULL, dispatch_claimed_at=NULL, failure_reason=NULL, updated_at=?
                    WHERE queued_run_id=? AND dispatch_claim_token=?""", (_now(), row["queued_run_id"], row["dispatch_claim_token"]))
            dispatched.append(job)
            capacity["slots"] -= 1
            capacity["leased_gpus"].update(required_gpus)
            if capacity["cpu_free"] is not None:
                capacity["cpu_free"] -= cpu_count
        return dispatched

    def cancel_queued_run(self, queued_run_id: str) -> dict[str, Any]:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            queued = db.execute("SELECT * FROM queued_runs WHERE queued_run_id=?", (queued_run_id,)).fetchone()
            if queued is None:
                raise KeyError(queued_run_id)
            if queued["status"] not in {"pending_verification", "ready", "needs_attention", "cancelled"}:
                raise ValueError("Cancel the active execution job before cancelling its queued run")
            db.execute("UPDATE queued_runs SET status='cancelled',start_authorized=0,updated_at=? WHERE queued_run_id=?", (_now(), queued_run_id))
        return self.queued_run(queued_run_id)

    def update_queued_run_batch_size(self, queued_run_id: str, *, batch_size: int) -> dict[str, Any]:
        """Replace an unstarted run's batch size without changing its definition.

        This is intentionally a queue-level amendment: the sealed TOML,
        dispatch specification, and experiment plan change together while the
        reusable model-definition revision stays untouched.  Once authorized,
        a queue row is immutable because a scheduler may claim it at any time.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        queued = self.queued_run(queued_run_id)
        if queued is None:
            raise KeyError(queued_run_id)
        if queued.get("worker_pool_id"):
            raise ValueError("Create a new queued run to change portable batch settings")
        if queued.get("start_authorized") or queued["status"] not in {"ready", "needs_attention", "waiting_for_resources"}:
            raise ValueError("Batch size can only be changed before the run is authorized")
        config_path = Path(str(queued["resolved_toml_path"])).resolve()
        if not config_path.is_file():
            raise FileNotFoundError("The sealed queued-run configuration is unavailable")
        config = load_toml(config_path)
        prior_batch_size = config.get("data", {}).get("batch_size")
        config.setdefault("data", {})["batch_size"] = batch_size
        import tomli_w
        config_path.write_text(tomli_w.dumps(self._toml_safe(config)), encoding="utf-8")
        config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        with self._connection() as db:
            specification = self.specification(str(queued["specification_id"]), connection=db)
            if specification is None:
                raise KeyError("Queued run specification was not found")
            parameters = dict(specification["parameters"])
            previous_execution = dict(parameters.get("queue_execution") or {})
            execution = {
                **previous_execution,
                "mode": "manual_override",
                "batch_size": batch_size,
                "auto_tuned_batch_size": previous_execution.get("batch_size") if previous_execution.get("mode") == "auto" else None,
            }
            parameters["queue_execution"] = execution
            experiment = _row(db.execute("SELECT plan_json FROM experiments WHERE experiment_id=?", (specification["experiment_id"],)).fetchone())
            if experiment is not None:
                plan = dict(experiment.get("plan") or {})
                plan["batch_execution"] = execution
                db.execute("UPDATE experiments SET plan_json=?, updated_at=? WHERE experiment_id=?", (_json(plan), _now(), specification["experiment_id"]))
            report = dict(queued.get("preflight_report") or {})
            report["batch_size_override"] = {
                "previous_batch_size": prior_batch_size,
                "batch_size": batch_size,
                "message": "Manual override selected after queue validation; launch preflight will recheck compute compatibility.",
            }
            now = _now()
            db.execute(
                "UPDATE run_specifications SET parameters_json=?, config_hash=?, updated_at=? WHERE specification_id=?",
                (_json(parameters), hashlib.sha256(_json({"parameters": parameters, "config_sha256": config_digest}).encode()).hexdigest(), now, specification["specification_id"]),
            )
            db.execute(
                "UPDATE queued_runs SET resolved_toml_sha256=?, preflight_report_json=?, failure_reason=NULL, updated_at=? WHERE queued_run_id=?",
                (config_digest, _json(report), now, queued_run_id),
            )
        return self.queued_run(queued_run_id)  # type: ignore[return-value]

    def archive_terminal_queued_run(self, queued_run_id: str) -> dict[str, Any]:
        """Clear a finished queue row while preserving the immutable job audit."""
        queued = self.queued_run(queued_run_id)
        if queued is None:
            raise KeyError(queued_run_id)
        with self._connection() as db:
            job = _row(db.execute("SELECT job_id,status FROM jobs WHERE queued_run_id=? ORDER BY submitted_at DESC LIMIT 1", (queued_run_id,)).fetchone())
            terminal = {"failed", "cancelled", "indexed", "artifact_invalid", "dispatch_failed", "succeeded"}
            if job is not None and job["status"] not in terminal:
                raise ValueError("Only a finished or failed run can be cleared from the queue")
            if job is None and queued["status"] not in {"failed", "cancelled", "complete", "needs_attention", "waiting_for_resources"}:
                raise ValueError("Only a non-running queue entry can be cleared")
            db.execute("UPDATE queued_runs SET status='archived', start_authorized=0, updated_at=? WHERE queued_run_id=?", (_now(), queued_run_id))
        if job is not None:
            self._record_event(str(job["job_id"]), "queue_cleared", "Queue row cleared by user; execution logs remain retained", {"queued_run_id": queued_run_id})
        return {"queued_run_id": queued_run_id, "archived": True, "job_id": job.get("job_id") if job else None}

    def control_job(self, job_id: str, action: str) -> dict[str, Any]:
        """Proxy an explicit lifecycle control to the owning compute service.

        The worker remains authoritative for the process, while the
        orchestrator immediately records the observed lifecycle and a durable
        audit event.  A cancellation may remain ``running`` briefly while the
        worker drains the process; normal reconciliation records its terminal
        state.
        """
        if action not in {"pause", "resume", "cancel"}:
            raise ValueError("Unsupported job control")
        local = self.job(job_id)
        if local is None:
            raise KeyError(job_id)
        if local["status"] in {"indexed", "artifact_invalid", "failed", "cancelled"}:
            raise ValueError("This job has already reached a terminal state")
        remote = self._request(local["oracle_serve_url"], "POST", f"/compute/jobs/{job_id}/{action}")
        remote_status = str(remote.get("status") or local.get("remote_status") or local["status"])
        now = _now()
        with self._connection() as db:
            db.execute(
                "UPDATE jobs SET status=?, remote_status=?, worker_id=?, error=?, updated_at=? WHERE job_id=?",
                (remote_status, remote_status, remote.get("worker_id"), remote.get("error"), now, job_id),
            )
            if remote_status in {"paused", "running"}:
                db.execute("UPDATE run_specifications SET status=?, updated_at=? WHERE specification_id=?", (remote_status, now, local["specification_id"]))
            queued_run_id = local.get("queued_run_id")
            if queued_run_id:
                queue_status = {"paused": "paused", "running": "running", "queued": "submitted"}.get(remote_status, remote_status)
                db.execute("UPDATE queued_runs SET status=?, updated_at=? WHERE queued_run_id=?", (queue_status, now, queued_run_id))
        verb = {"pause": "paused", "resume": "resumed", "cancel": "cancellation requested"}[action]
        self._record_event(job_id, action, f"Job {verb} by user", {"remote_status": remote_status})
        return self.job(job_id)  # type: ignore[return-value]

    def clear_queued_runs(self, *, worker_pool_id: str | None = None) -> dict[str, Any]:
        """Cancel only local, non-running queue entries.

        This is intentionally not a bulk compute-job cancellation endpoint.
        Jobs that have been submitted or are running remain untouched and must
        be handled through their own execution controls.
        """
        query, params = "SELECT queued_run_id,status FROM queued_runs WHERE status != 'archived'", []
        if worker_pool_id:
            if self.worker_pool(worker_pool_id) is None:
                raise KeyError(worker_pool_id)
            query += " AND worker_pool_id=?"; params.append(worker_pool_id)
        with self._connection() as db:
            rows = [dict(row) for row in db.execute(query, params).fetchall()]
            cancellable = [row["queued_run_id"] for row in rows if row["status"] in {"ready", "needs_attention", "waiting_for_resources"}]
            skipped = [{"queued_run_id": row["queued_run_id"], "status": row["status"]} for row in rows if row["queued_run_id"] not in cancellable and row["status"] != "cancelled"]
            if cancellable:
                db.execute(
                    f"UPDATE queued_runs SET status='cancelled', start_authorized=0, updated_at=? WHERE queued_run_id IN ({','.join('?' for _ in cancellable)})",
                    [_now(), *cancellable],
                )
        return {"cleared": cancellable, "skipped": skipped, "worker_pool_id": worker_pool_id}

    def model_preview(self, *, architecture: str, dataset_id: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build a disposable model for an honest pre-run structural summary."""
        dataset = self.dataset(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        setup = self.model_setup(architecture)
        config = deep_merge(setup["config"], overrides or {})
        config.setdefault("run", {})["model"] = architecture
        with sqlite3.connect(dataset["path"]) as db:
            if config["run"].get("task") == "classification":
                config.setdefault("data", {})["num_classes"] = int(db.execute("SELECT count(*) FROM classification_labels").fetchone()[0])
        try:
            from oracle_builder.registry import get_model_builder
            model = get_model_builder(architecture)(config)
            lines: list[str] = []
            model.summary(print_fn=lines.append)
            return {"architecture": architecture, "input_shape": list(model.input_shape), "output_shape": list(model.output_shape), "parameters": int(model.count_params()), "layers": len(model.layers), "summary": "\n".join(lines[:35]), "config": config}
        except Exception as exc:  # A preview must never prevent configuring a run.
            return {"architecture": architecture, "error": str(exc), "config": config}

    def register_compute_endpoint(self, *, name: str, base_url: str) -> dict[str, Any]:
        normalized = base_url.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("Compute endpoint URL must start with http:// or https://")
        endpoint_id = str(uuid.uuid5(uuid.NAMESPACE_URL, normalized))
        now = _now()
        with self._connection() as db:
            db.execute("""INSERT INTO compute_endpoints VALUES (?, ?, ?, 1, 'unknown', NULL, NULL, NULL, NULL, NULL, ?, ?)
                ON CONFLICT(base_url) DO UPDATE SET name=excluded.name, enabled=1, updated_at=excluded.updated_at""",
                (endpoint_id, name.strip() or normalized, normalized, now, now))
        return self.compute_endpoint(endpoint_id)  # type: ignore[return-value]

    # Pull worker registration and durable leases ----------------------
    # These APIs deliberately do not create processes or accept command-line
    # fragments.  A pool admits a worker with a one-time bearer secret, then
    # all subsequent calls use that worker's own secret.  This lets a worker
    # live on another host without granting it control-plane credentials.
    @staticmethod
    def _token_digest(token: str) -> str:
        # Generated credentials are high entropy.  This helper accepts any
        # non-empty presented string so bad credentials consistently produce
        # an authorization failure rather than leaking a format oracle.
        if not isinstance(token, str) or not token:
            raise ValueError("Token is invalid")
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _safe_worker_endpoint(endpoint: str | None) -> str | None:
        if endpoint is None:
            return None
        if not isinstance(endpoint, str):
            raise ValueError("Worker endpoint must be a URL")
        normalized = endpoint.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Worker endpoint must be a plain http(s) URL without credentials")
        return normalized

    @staticmethod
    def _safe_worker_name(name: str, label: str = "Worker") -> str:
        if not isinstance(name, str) or not (value := name.strip()) or len(value) > 160:
            raise ValueError(f"{label} name must be between 1 and 160 characters")
        if any(ord(char) < 32 for char in value):
            raise ValueError(f"{label} name contains control characters")
        return value

    @staticmethod
    def _worker_lease_public(row: sqlite3.Row | dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        result.pop("token_sha256", None)
        if result.get("output_ref_json") is not None:
            result["output_ref"] = json.loads(result.pop("output_ref_json"))
        else:
            result.pop("output_ref_json", None)
        return result

    def create_worker_pool(
        self,
        *,
        name: str,
        allowed_actions: list[str] | tuple[str, ...],
        max_workers: int | None = None,
    ) -> dict[str, Any]:
        """Create an admission/scheduling boundary and return its join secret once."""
        pool_name = self._safe_worker_name(name, "Worker pool")
        actions = sorted({str(action) for action in allowed_actions})
        from oracle_data_contracts.work_units import WORK_UNIT_ACTIONS
        if not actions or any(action not in WORK_UNIT_ACTIONS for action in actions):
            raise ValueError("Worker pool actions must be supported work-unit actions")
        if max_workers is not None and (isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1):
            raise ValueError("max_workers must be a positive integer")
        pool_id, now, token = str(uuid.uuid4()), _now(), secrets.token_urlsafe(32)
        with self._connection() as db:
            db.execute(
                """INSERT INTO worker_pools(pool_id,name,enabled,allowed_actions_json,registration_token_sha256,max_workers,created_at,updated_at)
                   VALUES(?,?,1,?,?,?,?,?)""",
                (pool_id, pool_name, _json(actions), self._token_digest(token), max_workers, now, now),
            )
        result = self.worker_pool(pool_id) or {}
        result["registration_token"] = token
        return result

    def worker_pool(self, pool_id: str) -> dict[str, Any] | None:
        with self._connection() as db:
            result = _row(db.execute("SELECT * FROM worker_pools WHERE pool_id=?", (pool_id,)).fetchone())
        if result is not None:
            result.pop("registration_token_sha256", None)
        return result

    def worker_pools(self) -> list[dict[str, Any]]:
        with self._connection() as db:
            rows = db.execute("SELECT * FROM worker_pools ORDER BY created_at DESC").fetchall()
        result = [_row(row) for row in rows]
        for pool in result:
            if pool is not None:
                pool.pop("registration_token_sha256", None)
        return [pool for pool in result if pool is not None]

    def register_worker(
        self,
        *,
        pool_id: str,
        name: str,
        registration_token: str,
        endpoint: str | None = None,
        capabilities: dict[str, Any] | None = None,
        worker_id: str | None = None,
    ) -> dict[str, Any]:
        """Register a remote worker; raw secrets are never persisted or listed."""
        worker_name = self._safe_worker_name(name)
        endpoint = self._safe_worker_endpoint(endpoint)
        if capabilities is not None and not isinstance(capabilities, dict):
            raise ValueError("Worker capabilities must be an object")
        # Enforce JSON-only durable metadata, avoiding arbitrary Python values.
        try:
            encoded_capabilities = _json(capabilities or {})
        except (TypeError, ValueError) as exc:
            raise ValueError("Worker capabilities must be JSON serializable") from exc
        token_digest = self._token_digest(registration_token)
        if worker_id is None:
            worker_id = str(uuid.uuid4())
        else:
            try:
                worker_id = str(uuid.UUID(worker_id))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError("worker_id must be a UUID") from exc
        now, worker_token = _now(), secrets.token_urlsafe(32)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            pool = db.execute("SELECT * FROM worker_pools WHERE pool_id=?", (pool_id,)).fetchone()
            if pool is None:
                raise KeyError(pool_id)
            if not pool["enabled"]:
                raise ValueError("Worker pool is disabled")
            if not hmac.compare_digest(str(pool["registration_token_sha256"]), token_digest):
                raise PermissionError("Worker pool registration token is invalid")
            maximum = pool["max_workers"]
            if maximum is not None:
                count = db.execute("SELECT COUNT(*) FROM registered_workers WHERE pool_id=?", (pool_id,)).fetchone()[0]
                if count >= int(maximum):
                    raise ValueError("Worker pool has reached its registration limit")
            db.execute(
                """INSERT INTO registered_workers(worker_id,pool_id,name,endpoint,capabilities_json,auth_token_sha256,state,last_seen_at,created_at,updated_at)
                   VALUES(?,?,?,?,?,?, 'idle', ?,?,?)""",
                (worker_id, pool_id, worker_name, endpoint, encoded_capabilities, self._token_digest(worker_token), now, now, now),
            )
        result = self.registered_worker(worker_id) or {}
        result["worker_token"] = worker_token
        return result

    def registered_worker(self, worker_id: str) -> dict[str, Any] | None:
        with self._connection() as db:
            worker = _row(db.execute("SELECT * FROM registered_workers WHERE worker_id=?", (worker_id,)).fetchone())
        if worker is not None:
            worker.pop("auth_token_sha256", None)
        return worker

    def registered_workers(self, *, pool_id: str | None = None) -> list[dict[str, Any]]:
        with self._connection() as db:
            rows = db.execute(
                "SELECT * FROM registered_workers " + ("WHERE pool_id=? " if pool_id else "") + "ORDER BY created_at DESC",
                (pool_id,) if pool_id else (),
            ).fetchall()
        result = [_row(row) for row in rows]
        for worker in result:
            if worker is not None:
                worker.pop("auth_token_sha256", None)
        return [worker for worker in result if worker is not None]

    @staticmethod
    def _lease_expiry(ttl_seconds: int) -> str:
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or not 5 <= ttl_seconds <= 3600:
            raise ValueError("Lease TTL must be between 5 and 3600 seconds")
        return (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()

    def _authenticate_registered_worker(self, db: sqlite3.Connection, worker_id: str, worker_token: str) -> sqlite3.Row:
        worker = db.execute("SELECT * FROM registered_workers WHERE worker_id=?", (worker_id,)).fetchone()
        if worker is None:
            raise KeyError(worker_id)
        if not hmac.compare_digest(str(worker["auth_token_sha256"]), self._token_digest(worker_token)):
            raise PermissionError("Worker token is invalid")
        if worker["state"] == "disabled":
            raise ValueError("Worker is disabled")
        pool = db.execute("SELECT enabled FROM worker_pools WHERE pool_id=?", (worker["pool_id"],)).fetchone()
        if pool is None or not pool["enabled"]:
            raise ValueError("Worker pool is disabled")
        return worker

    def authenticate_worker(self, *, worker_id: str, worker_token: str) -> dict[str, Any]:
        """Authenticate a pull worker without exposing its stored credential."""
        with self._connection() as db:
            worker = self._authenticate_registered_worker(db, worker_id, worker_token)
        result = _row(worker) or {}
        result.pop("auth_token_sha256", None)
        return result

    @staticmethod
    def _worker_supports_action(worker: sqlite3.Row, action: str) -> bool:
        """Workers opt into actions explicitly; absence is not an implicit grant."""
        capabilities = json.loads(worker["capabilities_json"])
        actions = capabilities.get("actions") if isinstance(capabilities, dict) else None
        return isinstance(actions, list) and action in {str(value) for value in actions}

    def _worker_can_execute_unit(self, worker: sqlite3.Row, unit: dict[str, Any]) -> bool:
        """Conservative local compatibility check before granting a pull lease."""
        if not self._worker_supports_action(worker, str(unit.get("action", ""))):
            return False
        capabilities = json.loads(worker["capabilities_json"])
        execution = (unit.get("parameters") or {}).get("queue_execution") or {}
        if execution.get("required_execution_contract_version") == 2 and not all(capabilities.get(key) is True for key in ("work_unit_v2", "training_segments_v1", "worker_control_v1")):
            return False
        if unit.get('schema', {}).get('version') == 2:
            if any(capabilities.get(key) != required for key, required in unit.get('compatibility', {}).items()):
                return False
        if (execution.get("phase") == "verify" or execution.get("mode") == "verified") and capabilities.get("training_verification_v1") is not True:
            return False
        parameters = unit.get('parameters') or {}
        if ('shard_id' in parameters or 'merge_item_ids' in parameters) and capabilities.get('inference_shards_v1') is not True:
            return False
        portable = unit.get('schema', {}).get('version') == 2 and execution.get('revalidate_fixed_batch') is True
        if not portable and execution.get("verified_worker_id") and execution["verified_worker_id"] != worker["worker_id"]:
            return False
        if not portable and execution.get("verified_environment_sha256") and execution["verified_environment_sha256"] != hashlib.sha256(_json(capabilities).encode()).hexdigest():
            return False
        resources = unit.get("resources") if isinstance(unit.get("resources"), dict) else {}
        cpu_requested, explicit_gpu_ids = self._resource_request(resources)
        cpu_capacity = capabilities.get("cpu_capacity")
        if isinstance(cpu_capacity, int) and not isinstance(cpu_capacity, bool) and cpu_requested > cpu_capacity:
            return False
        gpu_ids = {str(item) for item in capabilities.get("gpu_ids", [])} if isinstance(capabilities.get("gpu_ids", []), list) else set()
        requested_gpu_count = resources.get("gpu_count", 0)
        if explicit_gpu_ids is not None and explicit_gpu_ids and not set(explicit_gpu_ids).issubset(gpu_ids):
            return False
        if isinstance(requested_gpu_count, int) and not isinstance(requested_gpu_count, bool) and requested_gpu_count > len(gpu_ids):
            return False
        refs = list((unit.get("inputs") or {}).values())
        if unit.get("configuration") is not None:
            refs.append(unit["configuration"])
        if any(isinstance(ref, dict) and ref.get("kind") == "legacy_file" for ref in refs):
            return capabilities.get("execution_mode") == "local_path_compat"
        return True

    def _expire_worker_leases_in_connection(self, db: sqlite3.Connection, now: str) -> int:
        expired_rows = db.execute("SELECT lease_id,job_id FROM worker_leases WHERE status='active' AND expires_at<=?", (now,)).fetchall()
        expired = db.execute(
            """UPDATE worker_leases SET status='expired',outcome='expired',released_at=?,updated_at=?
               WHERE status='active' AND expires_at<=?""", (now, now, now),
        ).rowcount
        for lease in expired_rows:
            db.execute("UPDATE worker_commands SET status='expired',updated_at=? WHERE lease_id=? AND status IN ('pending','received','accepted')", (now, lease['lease_id']))
            self._finish_execution_attempt_in_connection(
                db, str(lease["lease_id"]), status="expired", classification="infrastructure",
                error="Worker lease expired before durable completion", now=now,
            )
            if lease["job_id"] is not None:
                self._retry_or_fail_infrastructure_in_connection(
                    db, str(lease["job_id"]), lease_id=str(lease["lease_id"]),
                    reason="Worker lease expired before durable completion", now=now,
                )
        if expired_rows:
            db.execute("UPDATE registered_workers SET state='idle',updated_at=? WHERE state='leased' AND worker_id NOT IN (SELECT worker_id FROM worker_leases WHERE status='active')", (now,))
        return expired

    def _record_worker_lease_event(self, db: sqlite3.Connection, lease_id: str, event_type: str, message: str, data: dict[str, Any]) -> None:
        sequence = db.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM worker_lease_events WHERE lease_id=?", (lease_id,)).fetchone()[0]
        db.execute("INSERT INTO worker_lease_events VALUES(?,?,?,?,?,?)", (lease_id, sequence, _now(), event_type, message, _json(data)))

    def _record_job_event_in_connection(self, db: sqlite3.Connection, job_id: str, event_type: str, message: str, data: dict[str, Any]) -> None:
        sequence = db.execute("SELECT COALESCE(MAX(sequence), 0) + 1 FROM job_events WHERE job_id=?", (job_id,)).fetchone()[0]
        db.execute("INSERT INTO job_events VALUES (?, ?, ?, ?, ?, ?)", (job_id, sequence, _now(), event_type, message, _json(data)))
        # The current unit is the authority for a sequential run's live state.
        # An event from a completed predecessor cannot regress its successor.
        current = db.execute("SELECT j.status,j.queued_run_id FROM jobs j JOIN execution_runs r ON r.current_job_id=j.job_id WHERE j.job_id=?", (job_id,)).fetchone()
        if current:
            db.execute("UPDATE execution_runs SET status=?,updated_at=? WHERE current_job_id=?", (current['status'], _now(), job_id))
            if current['queued_run_id']:
                db.execute("UPDATE queued_runs SET status=?,updated_at=? WHERE queued_run_id=?", (current['status'], _now(), current['queued_run_id']))

    def _finish_execution_attempt_in_connection(
        self, db: sqlite3.Connection, lease_id: str, *, status: str, classification: str,
        error: str | None, now: str, output_ref: ArtifactRef | None = None,
    ) -> None:
        db.execute(
            "UPDATE execution_attempts SET status=?,classification=?,error=?,output_ref_json=?,finished_at=?,updated_at=? "
            "WHERE lease_id=? AND status IN ('leased','running')",
            (status, classification, error, _json(output_ref.to_dict()) if output_ref else None, now, now, lease_id),
        )

    def _retry_or_fail_infrastructure_in_connection(
        self, db: sqlite3.Connection, job_id: str, *, lease_id: str, reason: str, now: str,
    ) -> None:
        """Apply the sole automatic retry policy: one infrastructure retry."""
        job = db.execute("SELECT retry_count,max_retries,cancel_requested_at,status,queued_run_id FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if job is None:
            return
        if job["cancel_requested_at"] is not None:
            db.execute("UPDATE jobs SET status='cancelled',remote_status='cancelled',cancelled_at=?,completed_at=?,updated_at=? WHERE job_id=?", (now, now, now, job_id))
            self._record_job_event_in_connection(db, job_id, "cancelled", "Cancellation completed after worker loss", {"lease_id": lease_id})
            if job["queued_run_id"]:
                db.execute("UPDATE queued_runs SET status='cancelled',start_authorized=0,updated_at=? WHERE queued_run_id=?", (now, job["queued_run_id"]))
        elif int(job["retry_count"] or 0) < int(job["max_retries"] or 0):
            db.execute(
                "UPDATE jobs SET status='queued',remote_status='queued',worker_id=NULL,retry_count=retry_count+1,error=?,updated_at=? WHERE job_id=?",
                (reason, now, job_id),
            )
            self._record_job_event_in_connection(db, job_id, "retry_queued", "Infrastructure loss queued the one automatic retry", {"lease_id": lease_id, "reason": reason})
        else:
            db.execute("UPDATE jobs SET status='failed',remote_status='failed',worker_id=NULL,error=?,completed_at=?,updated_at=? WHERE job_id=?", (reason, now, now, job_id))
            self._record_job_event_in_connection(db, job_id, "retry_exhausted", "Infrastructure retry budget exhausted", {"lease_id": lease_id, "reason": reason})
            failed_job = db.execute("SELECT queued_run_id,work_unit_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if failed_job["queued_run_id"]:
                verifying = (json.loads(failed_job["work_unit_json"]).get("parameters", {}).get("queue_execution", {}).get("phase") == "verify")
                db.execute("UPDATE queued_runs SET status=?,start_authorized=0,failure_reason=?,updated_at=? WHERE queued_run_id=?",
                           ("needs_attention" if verifying else "failed", reason, now, failed_job["queued_run_id"]))

    def _acquire_worker_lease_in_connection(
        self, db: sqlite3.Connection, *, worker: sqlite3.Row, work_unit_id: str, job_id: str | None, ttl_seconds: int,
    ) -> dict[str, Any]:
        now, expires_at = _now(), self._lease_expiry(ttl_seconds)
        self._expire_worker_leases_in_connection(db, now)
        lease_id, lease_token = str(uuid.uuid4()), secrets.token_urlsafe(32)
        try:
            db.execute(
                """INSERT INTO worker_leases(lease_id,worker_id,work_unit_id,job_id,token_sha256,status,issued_at,expires_at,released_at,outcome,created_at,updated_at)
                   VALUES(?,?,?,?,?,'active',?,?,NULL,NULL,?,?)""",
                (lease_id, worker["worker_id"], work_unit_id, job_id, self._token_digest(lease_token), now, expires_at, now, now),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Worker or work unit already has an active lease") from exc
        db.execute("UPDATE registered_workers SET state='leased',last_seen_at=?,updated_at=? WHERE worker_id=?", (now, now, worker["worker_id"]))
        if job_id is not None:
            db.execute("UPDATE jobs SET status='leased',remote_status='leased',worker_id=?,updated_at=? WHERE job_id=? AND status='queued'", (worker["worker_id"], now, job_id))
            job = db.execute("SELECT work_unit_sha256 FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            ordinal = int(db.execute("SELECT COALESCE(MAX(ordinal),0)+1 FROM execution_attempts WHERE job_id=?", (job_id,)).fetchone()[0])
            db.execute(
                "INSERT INTO execution_attempts(attempt_id,job_id,lease_id,ordinal,status,classification,worker_id,work_unit_sha256,created_at,updated_at) "
                "VALUES(?,?,?,?,'leased',NULL,?,?,?,?)",
                (str(uuid.uuid4()), job_id, lease_id, ordinal, worker["worker_id"], str(job["work_unit_sha256"] or ""), now, now),
            )
            self._record_job_event_in_connection(db, job_id, "leased", "Work unit leased to pull worker", {"lease_id": lease_id, "worker_id": worker["worker_id"], "expires_at": expires_at, "environment_sha256": hashlib.sha256(_json(json.loads(worker["capabilities_json"])).encode()).hexdigest()})
        self._record_worker_lease_event(db, lease_id, "acquired", "Worker lease acquired", {"worker_id": worker["worker_id"], "work_unit_id": work_unit_id, "expires_at": expires_at})
        result = self._worker_lease_public(db.execute("SELECT * FROM worker_leases WHERE lease_id=?", (lease_id,)).fetchone()) or {}
        result["lease_token"] = lease_token
        if job_id is not None:
            attempt = db.execute('SELECT attempt_id,ordinal FROM execution_attempts WHERE lease_id=?', (lease_id,)).fetchone()
            result['attempt_id'] = attempt['attempt_id']
            result['generation'] = attempt['ordinal']
            result['worker_boot_id'] = json.loads(worker['capabilities_json']).get('execution_instance_id')
            result['work_unit_sha256'] = job['work_unit_sha256']
        return result

    def acquire_worker_lease(self, *, worker_id: str, worker_token: str, work_unit_id: str, ttl_seconds: int = 60) -> dict[str, Any]:
        """Atomically claim one known sealed work unit for one admitted worker."""
        try:
            work_unit_id = str(uuid.UUID(work_unit_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("work_unit_id must be a UUID") from exc
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            worker = self._authenticate_registered_worker(db, worker_id, worker_token)
            job = db.execute("SELECT job_id,action,work_unit_json,worker_pool_id,status FROM jobs WHERE job_id=?", (work_unit_id,)).fetchone()
            if job is None or job["work_unit_json"] is None:
                raise KeyError("Sealed work unit was not found")
            active = db.execute("SELECT 1 FROM worker_leases WHERE work_unit_id=? AND status='active'", (work_unit_id,)).fetchone()
            if active is not None:
                raise ValueError("Worker or work unit already has an active lease")
            if job["status"] != "queued" or job["worker_pool_id"] != worker["pool_id"]:
                raise ValueError("Work unit is not available to this worker pool")
            allowed = set(json.loads(db.execute("SELECT allowed_actions_json FROM worker_pools WHERE pool_id=?", (worker["pool_id"],)).fetchone()[0]))
            unit = json.loads(job["work_unit_json"])
            if job["action"] not in allowed or not self._worker_can_execute_unit(worker, unit):
                raise ValueError("Worker is not capable of this work-unit action")
            lease = self._acquire_worker_lease_in_connection(db, worker=worker, work_unit_id=work_unit_id, job_id=str(job["job_id"]), ttl_seconds=ttl_seconds)
            lease["work_unit"] = unit
            return lease

    def _invalidate_changed_verifications_in_connection(self, db: sqlite3.Connection, worker: sqlite3.Row) -> None:
        """A restarted/reconfigured executor must reverify unstarted work."""
        digest = hashlib.sha256(_json(json.loads(worker["capabilities_json"])).encode()).hexdigest()
        rows = db.execute("""SELECT q.queued_run_id,q.specification_id,s.parameters_json
            FROM queued_runs q JOIN run_specifications s USING(specification_id)
            WHERE q.status IN ('ready','queued') AND q.worker_pool_id=?
            AND NOT EXISTS (SELECT 1 FROM jobs j WHERE j.queued_run_id=q.queued_run_id
                AND j.status NOT IN ('queued','completed','indexed','failed','cancelled'))""", (worker["pool_id"],)).fetchall()
        for row in rows:
            # Once authorized, V2 attempts verify their frozen batch locally.
            # Worker replacement must preserve the committed continuation.
            if db.execute("SELECT 1 FROM execution_runs WHERE queued_run_id=?", (row['queued_run_id'],)).fetchone():
                continue
            execution = json.loads(row["parameters_json"]).get("queue_execution", {})
            if execution.get("verified_worker_id") != worker["worker_id"] or execution.get("verified_environment_sha256") == digest:
                continue
            reason = "Worker restarted or its capabilities changed. Verify this run again before training."
            now = _now()
            jobs = db.execute("SELECT job_id FROM jobs WHERE queued_run_id=? AND status='queued'", (row["queued_run_id"],)).fetchall()
            for job in jobs:
                db.execute("UPDATE jobs SET status='cancelled',remote_status='cancelled',error=?,completed_at=?,updated_at=? WHERE job_id=?", (reason, now, now, job["job_id"]))
                self._record_job_event_in_connection(db, job["job_id"], "verification_invalidated", reason, {})
            db.execute("UPDATE queued_runs SET status='needs_attention',preflight_status='invalid',start_authorized=0,failure_reason=?,updated_at=? WHERE queued_run_id=?", (reason, now, row["queued_run_id"]))
            db.execute("UPDATE run_specifications SET status='planned',updated_at=? WHERE specification_id=?", (now, row["specification_id"]))

    def acquire_next_worker_lease(self, *, worker_id: str, worker_token: str, ttl_seconds: int = 60, capabilities: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Atomically select and claim a compatible locally queued work unit.

        Push dispatch does not create ``queued`` jobs, so this is deliberately
        an opt-in pull lane until queue intake is migrated.  It cannot steal a
        running legacy/push job.
        """
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            worker = self._authenticate_registered_worker(db, worker_id, worker_token)
            self._expire_worker_leases_in_connection(db, _now())
            if capabilities is not None:
                if not isinstance(capabilities, dict):
                    raise ValueError("Worker capabilities must be an object")
                if _json(capabilities) != worker["capabilities_json"] and db.execute(
                    "SELECT 1 FROM worker_leases WHERE worker_id=? AND status='active'", (worker_id,)
                ).fetchone():
                    raise ValueError("Worker capabilities cannot change during an active lease")
                # A restarted worker keeps its identity but advertises the
                # features of the installed executor before accepting work.
                db.execute("UPDATE registered_workers SET capabilities_json=?,updated_at=? WHERE worker_id=?",
                           (_json(capabilities), _now(), worker_id))
                worker = db.execute("SELECT * FROM registered_workers WHERE worker_id=?", (worker_id,)).fetchone()
                self._invalidate_changed_verifications_in_connection(db, worker)
            allowed = set(json.loads(db.execute("SELECT allowed_actions_json FROM worker_pools WHERE pool_id=?", (worker["pool_id"],)).fetchone()[0]))
            rows = db.execute("SELECT job_id,action,work_unit_json FROM jobs WHERE status='queued' AND worker_pool_id=? AND work_unit_json IS NOT NULL ORDER BY submitted_at", (worker["pool_id"],)).fetchall()
            candidate = next((row for row in rows if row["action"] in allowed and self._worker_can_execute_unit(worker, json.loads(row["work_unit_json"]))), None)
            if candidate is None:
                now = _now()
                db.execute("UPDATE registered_workers SET state='idle',last_seen_at=?,updated_at=? WHERE worker_id=?", (now, now, worker_id))
                return None
            lease = self._acquire_worker_lease_in_connection(db, worker=worker, work_unit_id=str(candidate["job_id"]), job_id=str(candidate["job_id"]), ttl_seconds=ttl_seconds)
            lease["work_unit"] = json.loads(candidate["work_unit_json"])
            return lease

    def enqueue_work_unit_for_pool(self, *, pool_id: str, work_unit: dict[str, Any]) -> dict[str, Any]:
        """Durably queue a sealed, path-free work unit for pull workers.

        This is intentionally distinct from legacy ``dispatch``: it does no
        network I/O and creates no compatibility path envelope.  A worker can
        claim it only through a pool whose policy admits its action.
        """
        from oracle_data_contracts.work_units import WorkUnit
        unit = WorkUnit.from_dict(work_unit)
        now = _now()
        with self._connection() as db:
            pool = db.execute("SELECT * FROM worker_pools WHERE pool_id=?", (pool_id,)).fetchone()
            if pool is None:
                raise KeyError(pool_id)
            if not pool["enabled"]:
                raise ValueError("Worker pool is disabled")
            if unit.action not in set(json.loads(pool["allowed_actions_json"])):
                raise ValueError("Work-unit action is not allowed by this worker pool")
            try:
                db.execute(
                    """INSERT INTO jobs(job_id,specification_id,oracle_serve_url,action,parameters_json,resources_json,
                       work_unit_json,work_unit_sha256,worker_pool_id,status,remote_status,worker_id,error,submitted_at,updated_at,completed_at)
                       VALUES(?,NULL,?,?,?,?,?,?,?,'queued',NULL,NULL,NULL,?,?,NULL)""",
                    (unit.work_unit_id, f"pull://{pool_id}", unit.action, _json({}), _json(unit.resources),
                     unit.canonical_json(), unit.sha256, pool_id, now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("A job already exists for this work unit") from exc
        self._record_event(unit.work_unit_id, "queued", "Work unit queued for pull workers", {"pool_id": pool_id, "work_unit_sha256": unit.sha256})
        return self.job(unit.work_unit_id) or {}

    # Durable inference -------------------------------------------------
    # Inference deliberately has its own request record instead of pretending
    # that a sealed model artifact is a training definition.  The record pins
    # both input ArtifactRefs before a worker is allowed to lease anything.
    @staticmethod
    def _inference_task_compatible(model: dict[str, Any], dataset: dict[str, Any]) -> bool:
        model_task = str(model.get("task") or "").strip().lower()
        dataset_task = str(dataset.get("dataset_type") or "").strip().lower()
        # Dataset types have historically been less precise than model tasks;
        # reject only an explicit, meaningful disagreement and let the worker
        # perform the detailed contract validation against materialized data.
        aliases = {"classification": "classification", "cluster": "clustering", "clustering": "clustering", "embedding": "embedding"}
        return not model_task or not dataset_task or aliases.get(model_task, model_task) == aliases.get(dataset_task, dataset_task)

    def _model_artifact_ref_for_inference(self, artifact: dict[str, Any]) -> ArtifactRef:
        fingerprint = artifact.get("fingerprint_sha256")
        ref = ArtifactRef(
            str(artifact.get("artifact_type") or "model_run"), str(artifact["artifact_id"]),
            revision=str(fingerprint) if fingerprint else "catalog",
            fingerprint_sha256=str(fingerprint) if fingerprint else None,
        )
        # Current worker-produced runs already live in the store.  Catalogued
        # sealed runs from before that boundary are registered as an explicit
        # read-only migration binding; the worker still receives only a grant.
        try:
            self.artifact_store.resolve(ref)
        except FileNotFoundError:
            register = getattr(self.artifact_store, "register_existing", None)
            if not callable(register):
                raise ValueError("Selected model artifact is not materializable by this artifact store")
            try:
                register(ref, artifact["path"])
            except FileExistsError:
                pass
        return ref

    def create_inference_run(
        self, *, name: str, model_artifact_id: str, dataset_id: str, worker_pool_id: str,
        split: str = "all", prediction_set: str | None = None,
        resources: dict[str, Any] | None = None,
        shard_size: int | None = None,
    ) -> dict[str, Any]:
        """Validate and seal a portable inference request without queueing it."""
        if not isinstance(name, str) or not (name := name.strip()):
            raise ValueError("An inference run requires a name")
        if split not in {"all", "train", "validation", "test"}:
            raise ValueError("split must be one of all, train, validation, or test")
        if shard_size is not None and (isinstance(shard_size, bool) or not isinstance(shard_size, int) or not 1 <= shard_size <= 100000):
            raise ValueError('Inference shard size must be between 1 and 100000')
        if prediction_set is not None and (not isinstance(prediction_set, str) or not prediction_set.strip() or len(prediction_set) > 120):
            raise ValueError("prediction_set must be a non-empty value up to 120 characters")
        artifact, dataset, pool = self.artifact(model_artifact_id), self.dataset(dataset_id), self.worker_pool(worker_pool_id)
        if artifact is None: raise KeyError("Model artifact was not found")
        if dataset is None: raise KeyError("Dataset was not found")
        if pool is None: raise KeyError("Worker pool was not found")
        if not pool["enabled"] or "infer" not in set(pool["allowed_actions"]):
            raise ValueError("Worker pool is not enabled for infer work")
        if artifact.get("lifecycle") != "sealed" or artifact.get("status") != "complete":
            raise ValueError("Inference requires a sealed, complete model artifact")
        if artifact.get("artifact_type") not in {"model_run", "model_product"}:
            raise ValueError("Inference requires a cataloged model artifact")
        if dataset.get("lifecycle") != "frozen":
            raise ValueError("Inference requires a frozen registered dataset")
        if not self._inference_task_compatible(artifact, dataset):
            raise ValueError("Selected model artifact and frozen dataset have incompatible tasks")
        model_ref = self._model_artifact_ref_for_inference(artifact)
        from oracle_builder.orchestration.storage import artifact_ref_for_file
        input_ref = artifact_ref_for_file("dataset", dataset_id, dataset["path"], revision=str(dataset.get("revision_id") or "frozen"))
        self.artifact_store.ingest_file(input_ref, dataset["path"])
        self.record_artifact_replica(input_ref)
        parameters = {"split": split, **({"prediction_set": prediction_set.strip()} if prediction_set else {})}
        if shard_size is not None:
            parameters['shard_size'] = shard_size
        run_id, attempt_id, now = str(uuid.uuid4()), str(uuid.uuid4()), _now()
        work_unit = build_work_unit(
            work_unit_id=run_id, attempt_id=attempt_id, specification_id=run_id, action="infer",
            parameters={"artifact_inputs": {"model": model_ref.to_dict(), "input": input_ref.to_dict()}, "inference": parameters},
            resources=dict(resources or {}), require_portable=True,
        )
        with self._connection() as db:
            db.execute("""INSERT INTO inference_runs(
                inference_run_id,name,model_artifact_id,model_ref_json,dataset_id,input_ref_json,
                parameters_json,resources_json,worker_pool_id,work_unit_json,work_unit_sha256,status,
                created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,'ready',?,?)""", (
                run_id, name, model_artifact_id, _json(model_ref.to_dict()), dataset_id, _json(input_ref.to_dict()),
                _json(parameters), _json(resources or {}), worker_pool_id, work_unit.canonical_json(), work_unit.sha256, now, now,
            ))
        return self.inference_run(run_id) or {}

    def inference_run(self, inference_run_id: str) -> dict[str, Any] | None:
        with self._connection() as db:
            row = db.execute("""SELECT inference_runs.*, jobs.status AS job_status, jobs.error AS job_error
                FROM inference_runs LEFT JOIN jobs USING(job_id) WHERE inference_run_id=?""", (inference_run_id,)).fetchone()
        return _row(row)

    def inference_runs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connection() as db:
            rows = db.execute("""SELECT inference_runs.*, jobs.status AS job_status, jobs.error AS job_error
                FROM inference_runs LEFT JOIN jobs USING(job_id) ORDER BY inference_runs.created_at DESC LIMIT ?""", (limit,)).fetchall()
        return [_row(row) for row in rows if row is not None]  # type: ignore[list-item]

    def start_inference_run(self, inference_run_id: str) -> dict[str, Any]:
        """Make a sealed inference request eligible for one pull-worker lease."""
        now = _now()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM inference_runs WHERE inference_run_id=?", (inference_run_id,)).fetchone()
            if row is None: raise KeyError(inference_run_id)
            if row["status"] != "ready": raise ValueError("Only ready inference runs can be started")
            pool = db.execute("SELECT * FROM worker_pools WHERE pool_id=?", (row["worker_pool_id"],)).fetchone()
            if pool is None or not pool["enabled"] or "infer" not in set(json.loads(pool["allowed_actions_json"])):
                raise ValueError("Worker pool is not enabled for infer work")
            unit_json, unit_sha = row['work_unit_json'], row['work_unit_sha256']
            if json.loads(row['parameters_json']).get('shard_size'):
                first_shard = self._prepare_inference_shards(db, row, now)
                if first_shard:
                    unit_json, unit_sha = first_shard.canonical_json(), first_shard.sha256
            db.execute("""INSERT INTO jobs(job_id,specification_id,oracle_serve_url,action,parameters_json,resources_json,
                work_unit_json,work_unit_sha256,worker_pool_id,status,remote_status,worker_id,error,submitted_at,updated_at,completed_at)
                VALUES(?,NULL,?,'infer',?,?,?,?,?,'queued',NULL,NULL,NULL,?,?,NULL)""", (
                inference_run_id, f"pull://{row['worker_pool_id']}", _json({}), row["resources_json"], unit_json,
                unit_sha, row["worker_pool_id"], now, now,
            ))
            db.execute("UPDATE inference_runs SET job_id=?,status='queued',started_at=?,updated_at=? WHERE inference_run_id=?", (inference_run_id, now, now, inference_run_id))
        self._record_event(inference_run_id, "queued", "Inference run queued for pull workers", {"pool_id": row["worker_pool_id"]})
        return self.inference_run(inference_run_id) or {}

    def cancel_inference_run(self, inference_run_id: str) -> dict[str, Any]:
        run = self.inference_run(inference_run_id)
        if run is None: raise KeyError(inference_run_id)
        if run["status"] == "ready":
            with self._connection() as db:
                db.execute("UPDATE inference_runs SET status='cancelled',completed_at=?,updated_at=? WHERE inference_run_id=?", (_now(), _now(), inference_run_id))
            return self.inference_run(inference_run_id) or {}
        if run.get("job_id"):
            job = self.request_pull_job_cancellation(str(run["job_id"]))
            if job["status"] == "cancelled":
                with self._connection() as db:
                    db.execute("UPDATE inference_runs SET status='cancelled',completed_at=?,updated_at=? WHERE inference_run_id=?", (_now(), _now(), inference_run_id))
        return self.inference_run(inference_run_id) or {}

    def enqueue_specification_for_pool(self, specification_id: str, pool_id: str) -> dict[str, Any]:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            queued = db.execute("SELECT * FROM queued_runs WHERE specification_id=?", (specification_id,)).fetchone()
            if queued is not None:
                if queued["status"] != "ready" or queued["preflight_status"] != "valid":
                    raise ValueError("Verify the queued run before starting training")
                if queued["worker_pool_id"] != pool_id:
                    raise ValueError("Training must use the verified worker pool")
                db.execute("UPDATE queued_runs SET start_authorized=1 WHERE specification_id=?", (specification_id,))
            return self._enqueue_specification_in_connection(db, specification_id, pool_id)

    def _enqueue_specification_in_connection(self, db: sqlite3.Connection, specification_id: str, pool_id: str) -> dict[str, Any]:
        now = _now()
        pool = db.execute("SELECT * FROM worker_pools WHERE pool_id=?", (pool_id,)).fetchone()
        if pool is None:
            raise KeyError(pool_id)
        if not pool["enabled"]:
            raise ValueError("Worker pool is disabled")
        spec = self.specification(specification_id, connection=db)
        if spec is None:
            raise KeyError(specification_id)
        if spec["status"] != "planned":
            raise ValueError("Only planned run specifications can be queued")
        # Specifications are retained as immutable provenance records for
        # current queued definitions, but the former model-import and
        # serve-dispatch actions have no pull-worker executor.  Rejecting
        # them here prevents an old database record from becoming an
        # indefinitely leased unit during a migration.
        if spec["action"] != "train":
            raise ValueError(
                "Only train specifications are supported by pull workers; "
                "legacy specification actions are retired"
            )
        if spec["action"] not in set(json.loads(pool["allowed_actions_json"])):
            raise ValueError("Run action is not allowed by this worker pool")
        job_id, attempt_id = str(uuid.uuid4()), str(uuid.uuid4())
        dataset_id = spec["parameters"].get("dataset_id")
        dataset = self.dataset(str(dataset_id), connection=db) if dataset_id else None
        work_unit = build_work_unit(
            work_unit_id=job_id,
            attempt_id=attempt_id,
            specification_id=specification_id,
            action=spec["action"],
            parameters=spec["parameters"],
            resources=spec["resources"],
            dataset=dataset,
            require_portable=True,
        )
        queued_run = db.execute(
            "SELECT queued_run_id FROM queued_runs WHERE specification_id=?", (specification_id,)
        ).fetchone()
        if spec['parameters'].get('queue_execution', {}).get('execution_contract_version') == 2:
            work_unit = self._begin_sequential_in_connection(db, unit=work_unit,
                queued_run_id=queued_run['queued_run_id'] if queued_run else None, pool_id=pool_id, now=now)
        db.execute(
            """INSERT INTO jobs(job_id,specification_id,oracle_serve_url,action,parameters_json,resources_json,
               work_unit_json,work_unit_sha256,worker_pool_id,queued_run_id,status,remote_status,worker_id,error,submitted_at,updated_at,completed_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,'queued',NULL,NULL,NULL,?,?,NULL)""",
            (job_id, specification_id, f"pull://{pool_id}", spec["action"], _json({}),
             _json(spec["resources"]), work_unit.canonical_json(), work_unit.sha256,
             pool_id, queued_run["queued_run_id"] if queued_run is not None else None, now, now),
        )
        db.execute(
            "UPDATE run_specifications SET status='queued', updated_at=? WHERE specification_id=?",
            (now, specification_id),
        )
        if queued_run is not None:
            db.execute(
                "UPDATE queued_runs SET status=?, updated_at=? WHERE queued_run_id=?",
                ("verifying" if spec["parameters"].get("queue_execution", {}).get("phase") == "verify" else "queued", now, queued_run["queued_run_id"]),
            )
        self._record_job_event_in_connection(db, job_id, "queued", "Verification queued" if spec["parameters"].get("queue_execution", {}).get("phase") == "verify" else "Training queued", {"pool_id": pool_id})
        return _row(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()) or {}

    # Pull execution completion ----------------------------------------
    # A remote worker only gets a lease-scoped authority.  It cannot select a
    # storage location, artifact id, or publication destination: the control
    # plane assigns the staging attempt and publishes a generic job-output
    # artifact after validating the candidate directory.
    def _authenticate_active_worker_lease(
        self,
        db: sqlite3.Connection,
        *,
        lease_id: str,
        worker_id: str,
        worker_token: str,
        lease_token: str,
    ) -> tuple[sqlite3.Row, sqlite3.Row, sqlite3.Row]:
        worker = self._authenticate_registered_worker(db, worker_id, worker_token)
        lease = db.execute("SELECT * FROM worker_leases WHERE lease_id=?", (lease_id,)).fetchone()
        if lease is None:
            raise KeyError(lease_id)
        if str(lease["worker_id"]) != str(worker["worker_id"]):
            raise PermissionError("Lease does not belong to this worker")
        if not hmac.compare_digest(str(lease["token_sha256"]), self._token_digest(lease_token)):
            raise PermissionError("Lease token is invalid")
        now = _now()
        if lease["status"] != "active" or str(lease["expires_at"]) <= now:
            self._expire_worker_leases_in_connection(db, now)
            raise ValueError("Lease is no longer active")
        if lease["job_id"] is None:
            raise ValueError("Lease has no durable work-unit job")
        job = db.execute("SELECT * FROM jobs WHERE job_id=?", (lease["job_id"],)).fetchone()
        if job is None or job["work_unit_json"] is None:
            raise ValueError("Lease work unit is unavailable")
        return lease, worker, job

    def authenticate_active_worker_lease(
        self, *, lease_id: str, worker_id: str, worker_token: str, lease_token: str,
    ) -> dict[str, Any]:
        """Validate worker + lease credentials and return its sealed unit.

        This is the sole authentication primitive for materialization and
        completion routes.  The returned WorkUnit remains path-free.
        """
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            lease, _worker, job = self._authenticate_active_worker_lease(
                db, lease_id=lease_id, worker_id=worker_id,
                worker_token=worker_token, lease_token=lease_token,
            )
            now = _now()
            db.execute("UPDATE registered_workers SET last_seen_at=?,updated_at=? WHERE worker_id=?", (now, now, worker_id))
            return {
                "lease": self._worker_lease_public(lease),
                "work_unit": json.loads(job["work_unit_json"]),
            }

    @staticmethod
    def _worker_event_value(event_id: str, event_type: str, message: str, data: dict[str, Any] | None) -> tuple[str, str, str, dict[str, Any]]:
        try:
            event_id = str(uuid.UUID(event_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("event_id must be a UUID") from exc
        if not isinstance(event_type, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", event_type):
            raise ValueError("event_type must be a short lowercase identifier")
        if not isinstance(message, str) or not (value := message.strip()) or len(value) > 2000:
            raise ValueError("event message must be between 1 and 2000 characters")
        if data is not None and not isinstance(data, dict):
            raise ValueError("event data must be an object")
        try:
            encoded = _json(data or {})
        except (TypeError, ValueError) as exc:
            raise ValueError("event data must be JSON serializable") from exc
        if len(encoded.encode("utf-8")) > 32 * 1024:
            raise ValueError("event data exceeds 32 KiB")
        return event_id, event_type, value, json.loads(encoded)

    def append_worker_lease_event(
        self, *, lease_id: str, worker_id: str, worker_token: str, lease_token: str,
        event_id: str, event_type: str, message: str, data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one bounded, idempotent worker progress event."""
        event_id, event_type, message, data = self._worker_event_value(event_id, event_type, message, data)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            lease, _worker, job = self._authenticate_active_worker_lease(
                db, lease_id=lease_id, worker_id=worker_id, worker_token=worker_token, lease_token=lease_token,
            )
            receipt = db.execute(
                "SELECT sequence FROM worker_lease_event_receipts WHERE lease_id=? AND event_id=?", (lease_id, event_id),
            ).fetchone()
            if receipt is not None:
                event = db.execute("SELECT * FROM worker_lease_events WHERE lease_id=? AND sequence=?", (lease_id, receipt["sequence"])).fetchone()
                return _row(event) or {}
            count = db.execute("SELECT COUNT(*) FROM worker_lease_events WHERE lease_id=?", (lease_id,)).fetchone()[0]
            if count >= 1000:
                raise ValueError("Lease event limit reached")
            sequence = int(db.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM worker_lease_events WHERE lease_id=?", (lease_id,)).fetchone()[0])
            now = _now()
            db.execute("INSERT INTO worker_lease_events VALUES(?,?,?,?,?,?)", (lease_id, sequence, now, event_type, message, _json(data)))
            db.execute("INSERT INTO worker_lease_event_receipts VALUES(?,?,?,?)", (lease_id, event_id, sequence, now))
            db.execute("UPDATE registered_workers SET last_seen_at=?,updated_at=? WHERE worker_id=?", (now, now, worker_id))
            self._record_job_event_in_connection(db, str(job["job_id"]), "worker." + event_type, message, {"lease_id": lease_id, "worker_id": worker_id, **data})
            return {"lease_id": lease_id, "sequence": sequence, "timestamp": now, "event_type": event_type, "message": message, "data": data}

    def acknowledge_worker_lease(
        self, *, lease_id: str, worker_id: str, worker_token: str, lease_token: str,
    ) -> dict[str, Any]:
        """Acknowledge execution and allocate a control-plane owned staging area."""
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            lease, _worker, job = self._authenticate_active_worker_lease(
                db, lease_id=lease_id, worker_id=worker_id, worker_token=worker_token, lease_token=lease_token,
            )
            attempt_id = lease["output_attempt_id"] or str(uuid.uuid4())
            if lease["acknowledged_at"] is None:
                # Allocate storage before recording acknowledgement. A failed
                # allocation leaves the lease retryable and unacknowledged.
                self.artifact_store.begin_staging(str(job["job_id"]), str(attempt_id))
                now = _now()
                db.execute(
                    "UPDATE worker_leases SET acknowledged_at=?,output_attempt_id=?,updated_at=? WHERE lease_id=?",
                    (now, attempt_id, now, lease_id),
                )
                db.execute("UPDATE jobs SET status='running',remote_status='running',started_at=COALESCE(started_at,?),updated_at=? WHERE job_id=?", (now, now, job["job_id"]))
                db.execute("UPDATE execution_attempts SET status='running',started_at=COALESCE(started_at,?),updated_at=? WHERE lease_id=? AND status='leased'", (now, now, lease_id))
                self._record_worker_lease_event(db, lease_id, "acknowledged", "Worker acknowledged execution", {"attempt_id": attempt_id})
                self._record_job_event_in_connection(db, str(job["job_id"]), "execution_acknowledged", "Pull worker acknowledged execution", {"lease_id": lease_id, "worker_id": worker_id})
            updated = db.execute("SELECT * FROM worker_leases WHERE lease_id=?", (lease_id,)).fetchone()
            return self._worker_lease_public(updated) or {}

    @staticmethod
    def _safe_archive_member(name: str) -> PurePosixPath:
        from pathlib import PurePosixPath
        candidate = PurePosixPath(name)
        if not name or candidate.is_absolute() or ".." in candidate.parts or any(part in {"", "."} for part in candidate.parts):
            raise ValueError("Output archive contains an unsafe member path")
        if len(candidate.parts) > 32 or len(name) > 512:
            raise ValueError("Output archive member path is too large")
        return candidate

    def _archive_input(self, archive: bytes | Path) -> tuple[io.BytesIO | Any, int]:
        """Open a bounded archive upload without trusting a worker path.

        ``Path`` is intentionally for an API-owned spool file only.  It is not
        exposed on the worker protocol and the service never derives a path
        from a remote request.
        """
        if isinstance(archive, bytes):
            return io.BytesIO(archive), len(archive)
        if isinstance(archive, Path) and archive.is_file():
            return archive.open("rb"), archive.stat().st_size
        raise ValueError("Output archive must be bytes or a trusted spool file")

    def _archive_digest(self, archive: bytes | Path) -> tuple[str, int]:
        if isinstance(archive, bytes):
            return hashlib.sha256(archive).hexdigest(), len(archive)
        if not isinstance(archive, Path) or not archive.is_file():
            raise ValueError("Output archive must be bytes or a trusted spool file")
        digest = hashlib.sha256()
        size = 0
        with archive.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                if size > self.upload_limit_bytes:
                    raise ValueError("Output archive exceeds the upload limit")
                digest.update(chunk)
        return digest.hexdigest(), size

    def _extract_worker_output_archive(self, archive: bytes | Path, destination: Path) -> None:
        """Extract a small, ordinary-file-only zip/tar candidate into staging."""
        archive_source, archive_size = self._archive_input(archive)
        if not archive_size or archive_size > self.upload_limit_bytes:
            archive_source.close()
            raise ValueError("Output archive is empty or exceeds the upload limit")
        if any(destination.iterdir()):
            raise ValueError("Output staging area is not empty")
        member_count = 0
        total_size = 0

        def check(name: str, size: int) -> Path:
            nonlocal member_count, total_size
            relative = self._safe_archive_member(name)
            if not isinstance(size, int) or size < 0:
                raise ValueError("Output archive member has an invalid size")
            member_count += 1
            total_size += size
            if member_count > 10_000 or total_size > self.upload_limit_bytes:
                raise ValueError("Output archive exceeds unpacked safety limits")
            target = destination.joinpath(*relative.parts)
            if not target.is_relative_to(destination):
                raise ValueError("Output archive escapes its staging area")
            return target

        # A tar may end in a .keras file (itself ZIP). is_zipfile alone
        # accepts that embedded archive and silently discards the outer run.
        magic = archive_source.read(4)
        archive_source.seek(0)
        if magic in {b'PK\x03\x04', b'PK\x05\x06'} and zipfile.is_zipfile(archive_source):
            try:
                archive_source.seek(0)
                with zipfile.ZipFile(archive_source) as bundle:
                    for member in bundle.infolist():
                        if member.is_dir():
                            self._safe_archive_member(member.filename.rstrip("/"))
                            continue
                        # Unix symlink bits are stored in the upper mode word.
                        if (member.external_attr >> 16) & 0o170000 == 0o120000:
                            raise ValueError("Output archive may not contain symlinks")
                        target = check(member.filename, member.file_size)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with bundle.open(member) as member_source, target.open("xb") as output:
                            shutil.copyfileobj(member_source, output, length=1024 * 1024)
            finally:
                archive_source.close()
            return
        archive_source.seek(0)
        try:
            with tarfile.open(fileobj=archive_source, mode="r:*") as bundle:
                for member in bundle:
                    if member.isdir():
                        self._safe_archive_member(member.name.rstrip("/"))
                        continue
                    if not member.isreg():
                        raise ValueError("Output archive may contain only regular files")
                    target = check(member.name, member.size)
                    member_source = bundle.extractfile(member)
                    if member_source is None:
                        raise ValueError("Could not read output archive member")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with member_source, target.open("xb") as output:
                        shutil.copyfileobj(member_source, output, length=1024 * 1024)
        except tarfile.TarError as exc:
            raise ValueError("Output upload must be a zip or tar archive") from exc
        finally:
            archive_source.close()

    @staticmethod
    def _valid_worker_output(path: Path) -> dict[str, Any]:
        files = 0
        size = 0
        for item in path.rglob("*"):
            if item.is_symlink() or not item.is_relative_to(path):
                return {"valid": False, "reason": "output contains a symlink or escapes staging"}
            if item.is_file():
                files += 1
                size += item.stat().st_size
        return {"valid": files > 0, "files": files, "bytes": size, "reason": None if files else "output archive contains no files"}

    @staticmethod
    def _valid_inference_output(path: Path) -> dict[str, Any]:
        """Validate the small, portable inference-result artifact contract."""
        generic = Orchestrator._valid_worker_output(path)
        if not generic["valid"]:
            return generic
        try:
            manifest = json.loads((path / "artifact.json").read_text(encoding="utf-8"))
            if manifest.get("schema", {}).get("name") != "oracle_builder_inference_result":
                raise ValueError("unsupported inference result schema")
            if manifest.get("artifact_type") != "inference_result":
                raise ValueError("output is not an inference result")
            if manifest.get("lifecycle") != "sealed" or manifest.get("status") != "complete":
                raise ValueError("inference result must be sealed and complete")
            str(uuid.UUID(str(manifest["artifact_id"])))
            if not (path / "predictions.sqlite").is_file() or not (path / "checksums.sha256").is_file():
                raise ValueError("inference result is missing predictions or checksums")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {"valid": False, "reason": str(exc)}
        return {**generic, "manifest": manifest}

    def upload_worker_lease_output_archive(
        self, *, lease_id: str, worker_id: str, worker_token: str, lease_token: str, archive: bytes | Path,
    ) -> dict[str, Any]:
        """Accept one bounded archive and extract it only into lease staging.

        This endpoint never accepts a destination or artifact reference.  A
        retry of the identical archive is idempotent; a different archive for
        the same acknowledged attempt is rejected.
        """
        digest, archive_size = self._archive_digest(archive)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            lease, _worker, job = self._authenticate_active_worker_lease(
                db, lease_id=lease_id, worker_id=worker_id, worker_token=worker_token, lease_token=lease_token,
            )
            if lease["acknowledged_at"] is None or lease["output_attempt_id"] is None:
                raise ValueError("Lease must be acknowledged before uploading output")
            if lease["output_archive_sha256"] is not None:
                if hmac.compare_digest(str(lease["output_archive_sha256"]), digest):
                    return {"lease_id": lease_id, "archive_sha256": digest, "idempotent": True}
                raise ValueError("Output archive has already been uploaded for this lease")
            staging = self.artifact_store.staging(str(job["job_id"]), str(lease["output_attempt_id"]))
            # The store verifies/reconstructs its own canonical path; obtain it
            # through staging_path rather than deriving a storage-root path.
            destination = self.artifact_store.staging_path(staging)
            try:
                self._extract_worker_output_archive(archive, destination)
            except Exception:
                # The digest is not recorded until extraction succeeds.  Clear
                # only this store-owned, lease-specific staging candidate so a
                # transient or malformed upload cannot poison a safe retry.
                for child in destination.iterdir():
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                raise
            now = _now()
            db.execute("UPDATE worker_leases SET output_archive_sha256=?,updated_at=? WHERE lease_id=?", (digest, now, lease_id))
            db.execute("UPDATE registered_workers SET last_seen_at=?,updated_at=? WHERE worker_id=?", (now, now, worker_id))
            self._record_worker_lease_event(db, lease_id, "output_uploaded", "Worker output archive staged", {"archive_sha256": digest, "bytes": archive_size})
            self._record_job_event_in_connection(db, str(job["job_id"]), "output_uploaded", "Pull worker output archive staged", {"lease_id": lease_id, "archive_sha256": digest})
            return {"lease_id": lease_id, "archive_sha256": digest, "bytes": archive_size, "idempotent": False}

    # Resumable worker-output publication --------------------------------
    # Do not share the browser upload-session tables here.  The worker
    # protocol has a different authority model (a short lived worker + lease
    # credential), and one upload is bound to exactly one execution lease.
    _WORKER_OUTPUT_MIN_PART_BYTES = 1
    _WORKER_OUTPUT_MAX_PART_BYTES = 64 * 1024 * 1024
    _WORKER_OUTPUT_MAX_PARTS = 10_000

    @staticmethod
    def _output_sha256(value: str) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("archive_sha256 must be a lowercase SHA-256 digest")
        return value

    def _worker_output_upload_root(self) -> Path:
        root = (self.artifact_root / "worker-output-uploads").resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _worker_output_upload_public(db: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        parts = db.execute(
            "SELECT part_number, size_bytes, sha256 FROM worker_output_upload_parts WHERE upload_id=? ORDER BY part_number",
            (row["upload_id"],),
        ).fetchall()
        return {
            "upload_id": str(row["upload_id"]), "lease_id": str(row["lease_id"]),
            "status": str(row["status"]), "archive_size": int(row["archive_size"]),
            "archive_sha256": str(row["archive_sha256"]), "part_size": int(row["part_size"]),
            "part_count": int(row["part_count"]),
            "uploaded_parts": [int(part["part_number"]) for part in parts],
            "uploaded_bytes": sum(int(part["size_bytes"]) for part in parts),
            "finalized_at": row["finalized_at"], "updated_at": row["updated_at"],
        }

    def create_worker_output_upload(
        self, *, lease_id: str, worker_id: str, worker_token: str, lease_token: str,
        archive_size: int, archive_sha256: str, part_size: int,
    ) -> dict[str, Any]:
        """Allocate (or resume) one bounded, lease-scoped output transfer."""
        if isinstance(archive_size, bool) or not isinstance(archive_size, int) or not 0 < archive_size <= self.upload_limit_bytes:
            raise ValueError("archive_size must be positive and within the output upload limit")
        archive_sha256 = self._output_sha256(archive_sha256)
        if isinstance(part_size, bool) or not isinstance(part_size, int) or not self._WORKER_OUTPUT_MIN_PART_BYTES <= part_size <= self._WORKER_OUTPUT_MAX_PART_BYTES:
            raise ValueError("part_size must be between 1 byte and 64 MiB")
        part_count = (archive_size + part_size - 1) // part_size
        if part_count > self._WORKER_OUTPUT_MAX_PARTS:
            raise ValueError("output archive would exceed the 10,000 part limit")
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            lease, _worker, job = self._authenticate_active_worker_lease(
                db, lease_id=lease_id, worker_id=worker_id, worker_token=worker_token, lease_token=lease_token,
            )
            if lease["acknowledged_at"] is None or lease["output_attempt_id"] is None:
                raise ValueError("Lease must be acknowledged before uploading output")
            existing = db.execute("SELECT * FROM worker_output_uploads WHERE lease_id=?", (lease_id,)).fetchone()
            if existing is not None:
                if (int(existing["archive_size"]), str(existing["archive_sha256"]), int(existing["part_size"])) != (archive_size, archive_sha256, part_size):
                    raise ValueError("Lease already has an output upload with different metadata")
                return self._worker_output_upload_public(db, existing)
            upload_id, now = str(uuid.uuid4()), _now()
            root = self._worker_output_upload_root() / upload_id
            root.mkdir(mode=0o700)
            archive_path = root / "output.archive"
            try:
                db.execute(
                    """INSERT INTO worker_output_uploads(upload_id,lease_id,worker_id,job_id,archive_size,archive_sha256,part_size,part_count,status,archive_path,created_at,finalized_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?, 'uploading', ?, ?, NULL, ?)""",
                    (upload_id, lease_id, worker_id, str(job["job_id"]), archive_size, archive_sha256, part_size, part_count, str(archive_path), now, now),
                )
            except Exception:
                shutil.rmtree(root, ignore_errors=True)
                raise
            self._record_worker_lease_event(db, lease_id, "output_upload_created", "Worker output upload session created", {"upload_id": upload_id, "bytes": archive_size, "part_count": part_count})
            return self._worker_output_upload_public(db, db.execute("SELECT * FROM worker_output_uploads WHERE upload_id=?", (upload_id,)).fetchone())

    def worker_output_upload(self, *, upload_id: str, worker_id: str, worker_token: str, lease_token: str) -> dict[str, Any]:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM worker_output_uploads WHERE upload_id=?", (upload_id,)).fetchone()
            if row is None: raise KeyError(upload_id)
            # The upload itself binds the worker.  Looking it up server-side
            # avoids adding a caller-controlled worker identifier to this API.
            if worker_id and worker_id != str(row["worker_id"]):
                raise PermissionError("Output upload does not belong to this worker")
            self._authenticate_active_worker_lease(db, lease_id=str(row["lease_id"]), worker_id=str(row["worker_id"]), worker_token=worker_token, lease_token=lease_token)
            return self._worker_output_upload_public(db, row)

    def job_worker_output_upload(self, job_id: str) -> dict[str, Any] | None:
        """Return browser-safe transfer progress without paths or credentials."""
        with self._connection() as db:
            row = db.execute(
                "SELECT * FROM worker_output_uploads WHERE job_id=? ORDER BY created_at DESC LIMIT 1",
                (job_id,),
            ).fetchone()
            return self._worker_output_upload_public(db, row) if row is not None else None

    def upload_worker_output_part(
        self, *, upload_id: str, worker_id: str, worker_token: str, lease_token: str,
        part_number: int, part_sha256: str, part: bytes | Path,
    ) -> dict[str, Any]:
        """Persist a single idempotent part without ever buffering an archive."""
        if isinstance(part_number, bool) or not isinstance(part_number, int) or part_number < 0:
            raise ValueError("part_number must be a non-negative integer")
        part_sha256 = self._output_sha256(part_sha256)
        digest, size = self._archive_digest(part)
        if not hmac.compare_digest(digest, part_sha256):
            raise ValueError("Part SHA-256 does not match uploaded bytes")
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM worker_output_uploads WHERE upload_id=?", (upload_id,)).fetchone()
            if row is None: raise KeyError(upload_id)
            self._authenticate_active_worker_lease(db, lease_id=str(row["lease_id"]), worker_id=worker_id, worker_token=worker_token, lease_token=lease_token)
            if row["status"] == "finalized": return self._worker_output_upload_public(db, row)
            if row["status"] == "assembling": raise ValueError("Output upload is being finalized")
            count, expected_size = int(row["part_count"]), int(row["part_size"])
            if part_number >= count: raise ValueError("part_number is outside this upload")
            if part_number == count - 1:
                expected_size = int(row["archive_size"]) - part_number * int(row["part_size"])
            if size != expected_size:
                raise ValueError("Part size does not match the declared upload layout")
            existing = db.execute("SELECT * FROM worker_output_upload_parts WHERE upload_id=? AND part_number=?", (upload_id, part_number)).fetchone()
            if existing is not None:
                if int(existing["size_bytes"]) == size and hmac.compare_digest(str(existing["sha256"]), digest):
                    return {**self._worker_output_upload_public(db, row), "part_idempotent": True}
                raise ValueError("Part number was already uploaded with different bytes")
            directory = Path(str(row["archive_path"])).parent
            target = directory / f"part-{part_number:06d}"
            if not target.resolve().is_relative_to(self._worker_output_upload_root()):
                raise ValueError("Invalid output upload spool path")
            temporary = target.with_suffix(".incoming-" + uuid.uuid4().hex)
            try:
                if isinstance(part, bytes):
                    temporary.write_bytes(part)
                else:
                    shutil.copyfile(part, temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            now = _now()
            db.execute("INSERT INTO worker_output_upload_parts(upload_id,part_number,size_bytes,sha256,created_at) VALUES(?,?,?,?,?)", (upload_id, part_number, size, digest, now))
            db.execute("UPDATE worker_output_uploads SET status='uploading',updated_at=? WHERE upload_id=?", (now, upload_id))
            return {**self._worker_output_upload_public(db, db.execute("SELECT * FROM worker_output_uploads WHERE upload_id=?", (upload_id,)).fetchone()), "part_idempotent": False}

    def finalize_worker_output_upload(self, *, upload_id: str, worker_id: str, worker_token: str, lease_token: str) -> dict[str, Any]:
        """Assemble verified parts and stage output; completion remains separate."""
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM worker_output_uploads WHERE upload_id=?", (upload_id,)).fetchone()
            if row is None: raise KeyError(upload_id)
            lease, _worker, _job = self._authenticate_active_worker_lease(db, lease_id=str(row["lease_id"]), worker_id=worker_id, worker_token=worker_token, lease_token=lease_token)
            if row["status"] == "finalized": return {**self._worker_output_upload_public(db, row), "idempotent": True}
            if row["status"] == "assembling":
                # A process can die after the archive was staged but before it
                # recorded this session as finalized. The lease's digest is
                # the durable publication receipt, so recover that narrow
                # window rather than stranding completed scientific output.
                if lease["output_archive_sha256"] is not None and hmac.compare_digest(str(lease["output_archive_sha256"]), str(row["archive_sha256"])):
                    now = _now()
                    db.execute("UPDATE worker_output_uploads SET status='finalized',finalized_at=COALESCE(finalized_at,?),updated_at=? WHERE upload_id=?", (now, now, upload_id))
                    recovered = db.execute("SELECT * FROM worker_output_uploads WHERE upload_id=?", (upload_id,)).fetchone()
                    return {**self._worker_output_upload_public(db, recovered), "idempotent": True}
                # No receipt means an interrupted assembly; it is safe to
                # recreate the archive deterministically from immutable parts.
                db.execute("UPDATE worker_output_uploads SET status='uploading',updated_at=? WHERE upload_id=?", (_now(), upload_id))
                row = db.execute("SELECT * FROM worker_output_uploads WHERE upload_id=?", (upload_id,)).fetchone()
            parts = db.execute("SELECT * FROM worker_output_upload_parts WHERE upload_id=? ORDER BY part_number", (upload_id,)).fetchall()
            if len(parts) != int(row["part_count"]) or [int(part["part_number"]) for part in parts] != list(range(int(row["part_count"]))):
                raise ValueError("Output upload is incomplete")
            db.execute("UPDATE worker_output_uploads SET status='assembling',updated_at=? WHERE upload_id=?", (_now(), upload_id))
            row = db.execute("SELECT * FROM worker_output_uploads WHERE upload_id=?", (upload_id,)).fetchone()
        archive_path = Path(str(row["archive_path"]))
        assembling = archive_path.with_suffix(".assembling")
        try:
            digest = hashlib.sha256(); written = 0
            with assembling.open("wb") as output:
                for number in range(int(row["part_count"])):
                    source = archive_path.parent / f"part-{number:06d}"
                    with source.open("rb") as input_file:
                        while chunk := input_file.read(8 * 1024 * 1024):
                            output.write(chunk); digest.update(chunk); written += len(chunk)
            if written != int(row["archive_size"]) or not hmac.compare_digest(digest.hexdigest(), str(row["archive_sha256"])):
                raise ValueError("Assembled output archive does not match declared digest or size")
            os.replace(assembling, archive_path)
            result = self.upload_worker_lease_output_archive(
                lease_id=str(row["lease_id"]), worker_id=worker_id, worker_token=worker_token,
                lease_token=lease_token, archive=archive_path,
            )
        except Exception:
            assembling.unlink(missing_ok=True)
            with self._connection() as db:
                db.execute("UPDATE worker_output_uploads SET status='uploading',updated_at=? WHERE upload_id=? AND status='assembling'", (_now(), upload_id))
            raise
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            now = _now()
            db.execute("UPDATE worker_output_uploads SET status='finalized',finalized_at=?,updated_at=? WHERE upload_id=?", (now, now, upload_id))
            final = db.execute("SELECT * FROM worker_output_uploads WHERE upload_id=?", (upload_id,)).fetchone()
            self._record_worker_lease_event(db, str(row["lease_id"]), "output_upload_finalized", "Worker output upload verified and staged", {"upload_id": upload_id, "bytes": int(row["archive_size"])})
            return {**self._worker_output_upload_public(db, final), "idempotent": bool(result.get("idempotent", False))}

    def defer_worker_output_publication(
        self, *, lease_id: str, worker_id: str, worker_token: str, lease_token: str,
        message: str = "Output publication deferred for worker recovery",
    ) -> dict[str, Any]:
        """Keep completed output recoverable without re-queuing its execution.

        This transition is intentionally neither a completed nor a failed
        lease.  The assigned worker identity may later obtain a fresh
        short-lived recovery token; the original raw lease token is never
        persisted on the worker.
        """
        now = _now()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            lease, _worker, job = self._authenticate_active_worker_lease(
                db, lease_id=lease_id, worker_id=worker_id,
                worker_token=worker_token, lease_token=lease_token,
            )
            db.execute(
                "UPDATE worker_leases SET status='publication_pending',outcome='publication_pending',updated_at=? WHERE lease_id=?",
                (now, lease_id),
            )
            db.execute(
                "UPDATE registered_workers SET state='publishing',last_seen_at=?,updated_at=? WHERE worker_id=?",
                (now, now, worker_id),
            )
            db.execute(
                "UPDATE jobs SET status='publication_pending',remote_status='publication_pending',error=?,updated_at=? WHERE job_id=?",
                (message[:2000], now, job["job_id"]),
            )
            self._record_worker_lease_event(db, lease_id, "output_publication_deferred", message, {})
            self._record_job_event_in_connection(db, str(job["job_id"]), "output_publication_deferred", message, {"lease_id": lease_id})
            return self._worker_lease_public(db.execute("SELECT * FROM worker_leases WHERE lease_id=?", (lease_id,)).fetchone()) or {}

    def resume_worker_output_publication(
        self, *, lease_id: str, worker_id: str, worker_token: str, ttl_seconds: int = 60,
    ) -> dict[str, Any]:
        """Rotate an expired-in-memory lease secret for its owning worker only."""
        expires_at, now = self._lease_expiry(ttl_seconds), _now()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            worker = self._authenticate_registered_worker(db, worker_id, worker_token)
            lease = db.execute("SELECT * FROM worker_leases WHERE lease_id=?", (lease_id,)).fetchone()
            if lease is None:
                raise KeyError(lease_id)
            if str(lease["worker_id"]) != str(worker["worker_id"]):
                raise PermissionError("Output recovery does not belong to this worker")
            if lease["status"] != "publication_pending" or lease["job_id"] is None:
                raise ValueError("Lease is not awaiting output recovery")
            lease_token = secrets.token_urlsafe(32)
            db.execute(
                "UPDATE worker_leases SET token_sha256=?,status='active',outcome=NULL,expires_at=?,updated_at=? WHERE lease_id=?",
                (self._token_digest(lease_token), expires_at, now, lease_id),
            )
            db.execute("UPDATE registered_workers SET state='leased',last_seen_at=?,updated_at=? WHERE worker_id=?", (now, now, worker_id))
            db.execute("UPDATE jobs SET status='publishing',remote_status='publishing',error=NULL,updated_at=? WHERE job_id=?", (now, lease["job_id"]))
            self._record_worker_lease_event(db, lease_id, "output_publication_resumed", "Worker resumed retained output publication", {"expires_at": expires_at})
            self._record_job_event_in_connection(db, str(lease["job_id"]), "output_publication_resumed", "Worker resumed retained output publication", {"lease_id": lease_id})
            result = self._worker_lease_public(db.execute("SELECT * FROM worker_leases WHERE lease_id=?", (lease_id,)).fetchone()) or {}
            result["lease_token"] = lease_token
            return result

    def _record_verification_in_connection(self, db: sqlite3.Connection, job: sqlite3.Row, worker_id: str, report: dict[str, Any], now: str, lease_id: str) -> None:
        """Apply a published verification once, also used after crash recovery."""
        queue = db.execute("SELECT * FROM queued_runs WHERE queued_run_id=?", (job["queued_run_id"],)).fetchone()
        if queue is None or queue["status"] != "verifying" or job["cancel_requested_at"] is not None:
            return
        execution = json.loads(job["work_unit_json"])["parameters"]["queue_execution"]
        # A V2 queue is only safe to advance when the verifier has produced
        # the matching contract.  This is repeated here because published
        # outputs can be reconciled after a process crash without traversing
        # the normal completion validation path.
        if execution.get("required_execution_contract_version") == 2:
            if report.get("execution_contract_version") != 2:
                raise ValueError("Verification report does not satisfy required execution contract V2")
            unit_epochs = report.get("unit_epochs")
            unit_steps = report.get("unit_steps", 0)
            if (isinstance(unit_epochs, bool) or not isinstance(unit_epochs, int) or unit_epochs < 1
                    or isinstance(unit_steps, bool) or not isinstance(unit_steps, int) or unit_steps < 0):
                raise ValueError("Verification report has invalid V2 work-unit bounds")
        spec = self.specification(job["specification_id"], connection=db)
        parameters = dict(spec["parameters"])
        requested = dict(execution)
        requested.pop("phase", None)
        lease_events = db.execute("SELECT data_json FROM job_events WHERE job_id=? AND event_type='leased' ORDER BY sequence DESC", (job["job_id"],)).fetchall()
        issued = next((data for event in lease_events if (data := json.loads(event[0])).get("lease_id") == lease_id), {})
        environment_digest = issued.get("environment_sha256")
        if not environment_digest:
            raise ValueError("Verification lease has no recorded worker environment; reverify the run")
        parameters["queue_execution"] = {
            "mode": "verified", "batch_size": report["batch_size"],
            "epochs": execution.get("epochs", 10), "verified_worker_id": worker_id,
            "verified_environment_sha256": environment_digest, "requested_policy": requested,
        }
        if report.get('execution_contract_version') == 2:
            output_row = db.execute('SELECT output_ref_json FROM worker_leases WHERE lease_id=?', (lease_id,)).fetchone()
            if not output_row or not output_row[0]:
                raise ValueError('Verification has no committed split artifact')
            parameters.setdefault('artifact_inputs', {})['split_manifest'] = json.loads(output_row[0])
            parameters['queue_execution']['execution_contract_version'] = 2
            parameters['queue_execution']['unit_epochs'] = report.get('unit_epochs', 1)
            parameters['queue_execution']['unit_steps'] = report.get('unit_steps', 0)
        evidence = {**report, "worker_id": worker_id, "environment_sha256": environment_digest, "verified_at": now, "job_id": job["job_id"]}
        db.execute("UPDATE run_specifications SET parameters_json=?,config_hash=?,status='planned',updated_at=? WHERE specification_id=?",
                   (_json(parameters), hashlib.sha256(_json(parameters).encode()).hexdigest(), now, job["specification_id"]))
        db.execute("UPDATE queued_runs SET status='ready',preflight_status='valid',preflight_report_json=?,failure_reason=NULL,updated_at=? WHERE queued_run_id=?",
                   (_json(evidence), now, job["queued_run_id"]))
        pool = db.execute("SELECT * FROM worker_pools WHERE pool_id=?", (queue["worker_pool_id"],)).fetchone()
        if queue["start_authorized"] and pool["enabled"] and "train" in json.loads(pool["allowed_actions_json"]):
            # The durable consent, completion, and next job commit
            # together, including across control-plane restarts.
            self._enqueue_specification_in_connection(db, job["specification_id"], queue["worker_pool_id"])
        elif queue["start_authorized"]:
            db.execute("UPDATE queued_runs SET start_authorized=0,failure_reason=? WHERE queued_run_id=?",
                       ("Verification passed; pool no longer allows training. Enable the pool, then start explicitly.", job["queued_run_id"]))

    def complete_worker_lease(
        self, *, lease_id: str, worker_id: str, worker_token: str, lease_token: str,
        success: bool, message: str = "Worker completed execution",
    ) -> dict[str, Any]:
        """Finalize an acknowledged lease and publish a sealed generic output.

        Failure is terminal for this attempt but does not publish anything.
        Successful completion requires an uploaded archive whose extracted
        candidate validates and seals through the configured ArtifactStore.
        """
        if not isinstance(success, bool):
            raise ValueError("success must be boolean")
        if not isinstance(message, str) or not (message := message.strip()) or len(message) > 2000:
            raise ValueError("completion message must be between 1 and 2000 characters")
        published_path: Path | None = None
        published_model_artifact_id: str | None = None
        published_inference_artifact_id: str | None = None
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            existing_lease = db.execute('SELECT * FROM worker_leases WHERE lease_id=?', (lease_id,)).fetchone()
            if existing_lease is not None and existing_lease['status'] == 'released' and existing_lease['outcome'] == ('completed' if success else 'failed'):
                self._authenticate_registered_worker(db, worker_id, worker_token)
                if existing_lease['worker_id'] != worker_id or not hmac.compare_digest(existing_lease['token_sha256'], self._token_digest(lease_token)):
                    raise PermissionError('Completion receipt does not belong to this worker lease')
                return self._worker_lease_public(existing_lease) or {}
            lease, _worker, job = self._authenticate_active_worker_lease(
                db, lease_id=lease_id, worker_id=worker_id, worker_token=worker_token, lease_token=lease_token,
            )
            if lease["acknowledged_at"] is None:
                raise ValueError("Lease must be acknowledged before completion")
            now = _now()
            execution = (json.loads(job["work_unit_json"]).get("parameters") or {}).get("queue_execution") or {}
            verifying = execution.get("phase") == "verify"
            sealed_unit = json.loads(job['work_unit_json'])
            checkpointing = sealed_unit.get('schema', {}).get('version') == 2 and sealed_unit.get('phase') == 'train'
            segment_result = None
            verification_report = None
            pending_stop = db.execute("SELECT action FROM worker_commands WHERE lease_id=? AND action IN ('stop_now','restart') AND status IN ('pending','received','accepted')", (lease_id,)).fetchone()
            if success and (pending_stop or job['cancel_requested_at'] is not None):
                raise ValueError('Attempt was stopped or restarted; output cannot be committed')
            if success and verifying and job["cancel_requested_at"] is not None:
                raise ValueError("Verification was cancelled; training cannot be authorized")
            output_ref: ArtifactRef | None = None
            if success:
                if lease["output_attempt_id"] is None or lease["output_archive_sha256"] is None:
                    raise ValueError("Successful completion requires a staged output archive")
                staging = self.artifact_store.staging(str(job["job_id"]), str(lease["output_attempt_id"]))
                validator = self._valid_worker_output
                if verifying:
                    candidate = self.artifact_store.staging_path(staging) / "verification.json"
                    if not candidate.is_file() or candidate.stat().st_size > 65536:
                        raise ValueError("Verification output requires a bounded verification report")
                    verification_report = json.loads(candidate.read_text(encoding="utf-8"))
                    if not isinstance(verification_report, dict):
                        raise ValueError("Invalid worker verification report")
                    batch = verification_report.get("batch_size")
                    if (verification_report.get("ready") is not True or isinstance(batch, bool)
                            or not isinstance(batch, int) or batch < 1
                            or verification_report.get("work_unit_id") != job["job_id"]):
                        raise ValueError("Invalid worker verification report")
                    if execution.get("required_execution_contract_version") == 2:
                        if verification_report.get("execution_contract_version") != 2:
                            raise ValueError("Verification report does not satisfy required execution contract V2")
                        unit_epochs = verification_report.get("unit_epochs")
                        unit_steps = verification_report.get("unit_steps", 0)
                        if (isinstance(unit_epochs, bool) or not isinstance(unit_epochs, int) or unit_epochs < 1
                                or isinstance(unit_steps, bool) or not isinstance(unit_steps, int) or unit_steps < 0):
                            raise ValueError("Verification report has invalid V2 work-unit bounds")
                    if execution.get("mode") == "auto":
                        if not execution.get("minimum_batch_size", 1) <= batch <= execution["maximum_batch_size"]:
                            raise ValueError("Verified batch size is outside the requested bounds")
                    elif batch != execution.get("batch_size"):
                        raise ValueError("Verification changed the manually selected batch size")
                if verifying and verification_report.get('execution_contract_version') == 2:
                    from oracle_data_contracts.artifacts.splits import read_split_manifest
                    pinned_split = read_split_manifest(self.artifact_store.staging_path(staging))
                    dataset_ref = sealed_unit['inputs']['input']
                    recorded = pinned_split.get('dataset', {})
                    if recorded.get('dataset_id') != dataset_ref['artifact_id'] or (dataset_ref.get('revision') and recorded.get('revision_id') != dataset_ref['revision']):
                        raise ValueError('Verified split belongs to a different dataset revision')
                    if verification_report.get('split_manifest', {}).get('fingerprint_sha256') != pinned_split['fingerprint_sha256']:
                        raise ValueError('Verification report does not identify its pinned split manifest')
                # Generic/package executors may legitimately produce a
                # non-model output for a train-capable pool smoke test. Only
                # a candidate that actually presents a run manifest enters
                # the stricter model-run publication and catalog path.
                is_model_run = not verifying and not checkpointing and job["action"] == "train" and (self.artifact_store.staging_path(staging) / "artifact.json").is_file()
                is_inference_result = job["action"] == "infer"
                if sealed_unit.get('schema', {}).get('version') == 2:
                    self._validate_training_output_inputs(self.artifact_store.staging_path(staging), sealed_unit)
                if checkpointing:
                    from oracle_builder.artifacts import validate_run_artifact
                    validator = validate_run_artifact
                    segment_result = self._read_segment_result(self.artifact_store.staging_path(staging), sealed_unit)
                elif is_model_run:
                    from oracle_builder.artifacts import validate_run_artifact
                    validator = validate_run_artifact
                elif is_inference_result:
                    validator = self._valid_inference_output
                    if sealed_unit.get('parameters', {}).get('shard_id'):
                        from oracle_data_contracts.work_units import WorkUnit
                        self._validate_inference_shard_output(self.artifact_store.staging_path(staging), WorkUnit.from_dict(sealed_unit))
                    elif sealed_unit.get('parameters', {}).get('merge_item_ids'):
                        from oracle_data_contracts.work_units import WorkUnit
                        self._validate_inference_merge_output(self.artifact_store.staging_path(staging), WorkUnit.from_dict(sealed_unit))
                if is_model_run:
                    try:
                        manifest = json.loads((self.artifact_store.staging_path(staging) / "artifact.json").read_text(encoding="utf-8"))
                        published_model_artifact_id = str(uuid.UUID(str(manifest["artifact_id"])))
                    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise ValueError(f"Staged training artifact has no valid identity: {exc}") from exc
                    output_ref = ArtifactRef("model_run", published_model_artifact_id, revision=str(lease["output_attempt_id"]))
                elif is_inference_result:
                    try:
                        manifest = json.loads((self.artifact_store.staging_path(staging) / "artifact.json").read_text(encoding="utf-8"))
                        published_inference_artifact_id = str(uuid.UUID(str(manifest["artifact_id"])))
                    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise ValueError(f"Staged inference artifact has no valid identity: {exc}") from exc
                    output_ref = ArtifactRef("inference_result", published_inference_artifact_id, revision=str(lease["output_attempt_id"]))
                else:
                    from oracle_data_contracts.artifacts import directory_content_digest
                    output_ref = ArtifactRef("job_output", str(job["job_id"]), revision=str(lease["output_attempt_id"]),
                        fingerprint_sha256=directory_content_digest(self.artifact_store.staging_path(staging)))
                # Resolve the identity while staging is still readable through
                # the writable-handle API. Validation and sealing close it.
                result = self.artifact_store.validate(staging, validator)
                if not result.valid:
                    raise ValueError("Staged output did not pass publication validation")
                self.artifact_store.seal(staging)
                published_path = self.artifact_store.publish(staging, output_ref)
                # Publication is local and atomic.  A replica failure must
                # never invalidate it; record the store's immediate outcome
                # so a background operator retry can recover it later.
                status_method = getattr(self.artifact_store, "replication_status", None)
                if callable(status_method) or callable(getattr(self.artifact_store, "replicate", None)):
                    replica_report = status_method(output_ref) if callable(status_method) else {"status": "pending"}
                    replica_status = replica_report.get("status") if isinstance(replica_report, dict) else "pending"
                    if replica_status not in {"pending", "replicated", "failed"}: replica_status = "pending"
                    self._record_artifact_replica_in_connection(
                        db, output_ref, status=replica_status,
                        manifest_sha256=replica_report.get("manifest_sha256") if isinstance(replica_report, dict) else None,
                        error=replica_report.get("error") if isinstance(replica_report, dict) else None,
                    )
            outcome = "completed" if success else "failed"
            db.execute(
                "UPDATE worker_leases SET status='released',outcome=?,released_at=?,output_ref_json=?,updated_at=? WHERE lease_id=?",
                (outcome, now, _json(output_ref.to_dict()) if output_ref else None, now, lease_id),
            )
            db.execute("UPDATE registered_workers SET state='idle',last_seen_at=?,updated_at=? WHERE worker_id=?", (now, now, worker_id))
            if success:
                db.execute("UPDATE jobs SET status='completed',remote_status='completed',completed_at=?,output_path=NULL,updated_at=? WHERE job_id=?", (now, now, job["job_id"]))
                if is_inference_result and sealed_unit.get('parameters', {}).get('shard_id'):
                    self._commit_inference_shard(db, job, lease_id, output_ref, now)
                elif is_inference_result:
                    db.execute("""UPDATE inference_runs SET status='completed',output_ref_json=?,completed_at=?,updated_at=?
                        WHERE job_id=?""", (_json(output_ref.to_dict()) if output_ref else None, now, now, job["job_id"]))
                self._finish_execution_attempt_in_connection(db, lease_id, status="completed", classification="success", error=None, now=now, output_ref=output_ref)
            else:
                db.execute("UPDATE jobs SET status='failed',remote_status='failed',completed_at=?,error=?,updated_at=? WHERE job_id=?", (now, message, now, job["job_id"]))
                if job["action"] == "infer":
                    db.execute("UPDATE inference_runs SET status='failed',error=?,completed_at=?,updated_at=? WHERE job_id=?", (message, now, now, job["job_id"]))
                # A worker-reported failure is scientific/action failure; it
                # never consumes the infrastructure-only retry path.
                self._finish_execution_attempt_in_connection(db, lease_id, status="failed", classification="scientific", error=message, now=now)
            if job["queued_run_id"]:
                queue = db.execute("SELECT * FROM queued_runs WHERE queued_run_id=?", (job["queued_run_id"],)).fetchone()
                if verifying and success:
                    self._record_verification_in_connection(db, job, worker_id, verification_report, now, lease_id)
                elif not success:
                    db.execute("UPDATE queued_runs SET status=?,preflight_status=?,start_authorized=0,failure_reason=?,updated_at=? WHERE queued_run_id=?",
                               ("needs_attention" if verifying else "failed", "invalid" if verifying else queue["preflight_status"], message, now, job["queued_run_id"]))
                elif not verifying and not checkpointing:
                    db.execute("UPDATE queued_runs SET status='complete',updated_at=? WHERE queued_run_id=?", (now, job["queued_run_id"]))
            if success and checkpointing:
                self._commit_segment_in_connection(db, job=job, lease_id=lease_id, output_ref=output_ref, result=segment_result, now=now)
            if not success and sealed_unit.get('schema', {}).get('version') == 2:
                db.execute("UPDATE execution_runs SET status='failed',updated_at=? WHERE run_id=?", (now, sealed_unit['run_id']))
            data = {"lease_id": lease_id, "outcome": outcome}
            if output_ref:
                data["output_ref"] = output_ref.to_dict()
            self._record_worker_lease_event(db, lease_id, outcome, message, data)
            self._record_job_event_in_connection(db, str(job["job_id"]), outcome, message, data)
            updated = db.execute("SELECT * FROM worker_leases WHERE lease_id=?", (lease_id,)).fetchone()
            response = self._worker_lease_public(updated) or {}
        if success and published_model_artifact_id is not None and published_path is not None:
            # A published train result is a real model-run artifact, not a
            # generic opaque job output. Index only after the store's atomic
            # publication transition has completed.
            report = self.scan(published_path)
            if published_model_artifact_id not in report["artifacts"]:
                raise RuntimeError("Published training artifact could not be indexed: " + "; ".join(item["reason"] for item in report["skipped"]))
            with self._connection() as db:
                now = _now()
                db.execute("UPDATE jobs SET status='indexed',remote_status='indexed',updated_at=? WHERE job_id=?", (now, job["job_id"]))
                db.execute("UPDATE run_specifications SET status='indexed',artifact_id=?,updated_at=? WHERE specification_id=?", (published_model_artifact_id, now, job["specification_id"]))
                if job["queued_run_id"]:
                    db.execute("UPDATE queued_runs SET status='indexed',updated_at=? WHERE queued_run_id=?", (now, job["queued_run_id"]))
                if sealed_unit.get('schema', {}).get('version') == 2:
                    db.execute("UPDATE execution_runs SET status='indexed',updated_at=? WHERE run_id=?", (now, sealed_unit['run_id']))
            self._record_event(str(job["job_id"]), "artifact_indexed", "Published training artifact was indexed", {"artifact_id": published_model_artifact_id})
        return response

    def renew_worker_lease(self, *, lease_id: str, lease_token: str, ttl_seconds: int = 60) -> dict[str, Any]:
        expires_at, now = self._lease_expiry(ttl_seconds), _now()
        expired = False
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            lease = db.execute("SELECT * FROM worker_leases WHERE lease_id=?", (lease_id,)).fetchone()
            if lease is None:
                raise KeyError(lease_id)
            if not hmac.compare_digest(str(lease["token_sha256"]), self._token_digest(lease_token)):
                raise PermissionError("Lease token is invalid")
            if lease["status"] != "active" or lease["expires_at"] <= now:
                self._expire_worker_leases_in_connection(db, now)
                expired = True
            else:
                db.execute("UPDATE worker_leases SET expires_at=?,updated_at=? WHERE lease_id=?", (expires_at, now, lease_id))
                db.execute("UPDATE registered_workers SET last_seen_at=?,updated_at=? WHERE worker_id=?", (now, now, lease["worker_id"]))
                self._record_worker_lease_event(db, lease_id, "renewed", "Worker lease renewed", {"expires_at": expires_at})
                result = self._worker_lease_public(db.execute("SELECT * FROM worker_leases WHERE lease_id=?", (lease_id,)).fetchone()) or {}
        if expired:
            raise ValueError("Lease is no longer active")
        return result

    def release_worker_lease(self, *, lease_id: str, lease_token: str, outcome: str = "released") -> dict[str, Any]:
        if outcome not in {"released", "completed", "failed"}:
            raise ValueError("Lease outcome must be released, completed, or failed")
        now = _now()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            lease = db.execute("SELECT * FROM worker_leases WHERE lease_id=?", (lease_id,)).fetchone()
            if lease is None:
                raise KeyError(lease_id)
            if not hmac.compare_digest(str(lease["token_sha256"]), self._token_digest(lease_token)):
                raise PermissionError("Lease token is invalid")
            if lease["status"] != "active":
                raise ValueError("Lease is no longer active")
            db.execute("UPDATE worker_leases SET status='released',outcome=?,released_at=?,updated_at=? WHERE lease_id=?", (outcome, now, now, lease_id))
            db.execute("UPDATE registered_workers SET state='idle',last_seen_at=?,updated_at=? WHERE worker_id=?", (now, now, lease["worker_id"]))
            if lease["job_id"] is not None:
                # A release is only an execution handoff acknowledgement.  It
                # never declares an artifact complete; publication owns that
                # later state transition.
                # `release` never proves action failure or success—it has no
                # sealed output.  Even historical callers that used its
                # completed/failed labels therefore take only the bounded
                # infrastructure recovery path.  A scientific failure must
                # use complete_worker_lease(success=False), which is terminal.
                reason = f"Worker released lease ({outcome})"
                self._finish_execution_attempt_in_connection(db, lease_id, status="released", classification="infrastructure", error=reason, now=now)
                self._retry_or_fail_infrastructure_in_connection(db, str(lease["job_id"]), lease_id=lease_id, reason=reason, now=now)
            self._record_worker_lease_event(db, lease_id, "released", "Worker lease released", {"outcome": outcome})
            return self._worker_lease_public(db.execute("SELECT * FROM worker_leases WHERE lease_id=?", (lease_id,)).fetchone()) or {}

    def expire_worker_leases(self) -> int:
        """Release abandoned claims; scheduler startup/ticks can call this safely."""
        now = _now()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            expired = self._expire_worker_leases_in_connection(db, now)
            db.execute("UPDATE registered_workers SET state='idle',updated_at=? WHERE state='leased' AND worker_id NOT IN (SELECT worker_id FROM worker_leases WHERE status='active')", (now,))
        return expired

    def request_pull_job_cancellation(self, job_id: str) -> dict[str, Any]:
        """Durably request cancellation without trusting a worker to decide it.

        Queued jobs stop immediately.  A leased/running job remains assigned
        until its worker observes the request, reports cancellation, or loses
        its lease; this avoids concurrent duplicate execution.
        """
        now = _now()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            job = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(job_id)
            if job["worker_pool_id"] is None:
                raise ValueError("Only pull-worker jobs can be cancelled through this control plane")
            if job["status"] in {"completed", "indexed", "failed", "cancelled", "historical"}:
                return _row(job) or {}
            if job["status"] == "queued":
                db.execute("UPDATE jobs SET status='cancelled',remote_status='cancelled',cancel_requested_at=?,cancelled_at=?,completed_at=?,updated_at=? WHERE job_id=?", (now, now, now, now, job_id))
                if job["queued_run_id"]:
                    db.execute("UPDATE queued_runs SET status='cancelled',start_authorized=0,updated_at=? WHERE queued_run_id=?", (now, job["queued_run_id"]))
                self._record_job_event_in_connection(db, job_id, "cancelled", "Queued pull job cancelled", {})
            else:
                db.execute("UPDATE jobs SET status='cancel_requested',remote_status='cancel_requested',cancel_requested_at=COALESCE(cancel_requested_at,?),updated_at=? WHERE job_id=?", (now, now, job_id))
                self._record_job_event_in_connection(db, job_id, "cancellation_requested", "Cancellation requested for active pull job", {})
            return _row(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()) or {}

    def worker_lease_cancellation(
        self, *, lease_id: str, worker_id: str, worker_token: str, lease_token: str,
    ) -> dict[str, Any]:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            _lease, _worker, job = self._authenticate_active_worker_lease(
                db, lease_id=lease_id, worker_id=worker_id, worker_token=worker_token, lease_token=lease_token,
            )
            return {"cancel_requested": job["cancel_requested_at"] is not None}

    def cancel_worker_lease(
        self, *, lease_id: str, worker_id: str, worker_token: str, lease_token: str,
        message: str = "Worker acknowledged cancellation",
    ) -> dict[str, Any]:
        now = _now()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            lease, _worker, job = self._authenticate_active_worker_lease(
                db, lease_id=lease_id, worker_id=worker_id, worker_token=worker_token, lease_token=lease_token,
            )
            if job["cancel_requested_at"] is None:
                raise ValueError("Cancellation was not requested")
            db.execute("UPDATE worker_leases SET status='released',outcome='cancelled',released_at=?,updated_at=? WHERE lease_id=?", (now, now, lease_id))
            db.execute("UPDATE jobs SET status='cancelled',remote_status='cancelled',cancelled_at=?,completed_at=?,updated_at=? WHERE job_id=?", (now, now, now, job["job_id"]))
            db.execute("UPDATE registered_workers SET state='idle',last_seen_at=?,updated_at=? WHERE worker_id=?", (now, now, worker_id))
            self._finish_execution_attempt_in_connection(db, lease_id, status="cancelled", classification="cancelled", error=message, now=now)
            db.execute("UPDATE worker_commands SET status='applied',updated_at=? WHERE lease_id=? AND action='stop_now' AND status IN ('pending','received','accepted')", (now, lease_id))
            if job["queued_run_id"]:
                db.execute("UPDATE queued_runs SET status='cancelled',start_authorized=0,updated_at=? WHERE queued_run_id=?", (now, job["queued_run_id"]))
            self._record_worker_lease_event(db, lease_id, "cancelled", message, {})
            self._record_job_event_in_connection(db, str(job["job_id"]), "cancelled", message, {"lease_id": lease_id})
            return self._worker_lease_public(db.execute("SELECT * FROM worker_leases WHERE lease_id=?", (lease_id,)).fetchone()) or {}

    def reconcile_worker_liveness(self, *, stale_after_seconds: int = 180) -> dict[str, int]:
        """Mark silent workers offline and recover their active pull jobs."""
        if isinstance(stale_after_seconds, bool) or not isinstance(stale_after_seconds, int) or stale_after_seconds < 1:
            raise ValueError("stale_after_seconds must be a positive integer")
        now_dt = datetime.now(timezone.utc)
        cutoff = (now_dt - timedelta(seconds=stale_after_seconds)).isoformat()
        now = now_dt.isoformat()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            stale = db.execute("SELECT worker_id FROM registered_workers WHERE last_seen_at IS NOT NULL AND last_seen_at<?", (cutoff,)).fetchall()
            worker_ids = [str(row["worker_id"]) for row in stale]
            if worker_ids:
                placeholders = ",".join("?" for _ in worker_ids)
                db.execute(f"UPDATE worker_leases SET expires_at=? WHERE status='active' AND worker_id IN ({placeholders})", (now, *worker_ids))
                db.execute(f"UPDATE registered_workers SET state='offline',updated_at=? WHERE worker_id IN ({placeholders})", (now, *worker_ids))
            expired = self._expire_worker_leases_in_connection(db, now)
        return {"offline_workers": len(worker_ids), "expired_leases": expired}

    def worker_lease(self, lease_id: str) -> dict[str, Any] | None:
        with self._connection() as db:
            return self._worker_lease_public(db.execute("SELECT * FROM worker_leases WHERE lease_id=?", (lease_id,)).fetchone())

    def worker_leases(self, *, worker_id: str | None = None) -> list[dict[str, Any]]:
        with self._connection() as db:
            rows = db.execute("SELECT l.*,a.attempt_id,a.ordinal AS generation FROM worker_leases l LEFT JOIN execution_attempts a ON a.lease_id=l.lease_id " + ("WHERE l.worker_id=? " if worker_id else "") + "ORDER BY l.created_at DESC", (worker_id,) if worker_id else ()).fetchall()
        return [self._worker_lease_public(row) for row in rows if row is not None]  # type: ignore[list-item]

    # Pull-worker lifecycle ---------------------------------------------
    # These methods are deliberately configuration-facing rather than public
    # browser endpoints.  A deployment constructs a typed spec from its own
    # approved mounts/secrets; callers cannot submit a path, token, image, or
    # command through the API and turn the control plane into a remote shell.
    def start_pull_worker(self, spec: PullWorkerSpec) -> dict[str, Any]:
        pool = self.worker_pool(spec.pool_id)
        if pool is None:
            raise KeyError(f"worker pool does not exist: {spec.pool_id}")
        if not pool["enabled"]:
            raise ValueError(f"worker pool is disabled: {spec.pool_id}")
        unsupported = set(spec.executors) - set(pool["allowed_actions"])
        if unsupported:
            raise ValueError(f"worker pool does not allow actions: {', '.join(sorted(unsupported))}")
        status = self.pull_worker_provider.start(spec)
        self.record_audit_event(
            actor_role="orchestrator", method="LIFECYCLE", path="/pull-worker:start",
            outcome="success", request_id=status.worker_id,
        )
        return self._worker_status_payload(status)

    def stop_pull_worker(self, worker_id: str, *, timeout_seconds: float = 10.0) -> dict[str, Any]:
        status = self.pull_worker_provider.stop(worker_id, timeout_seconds=timeout_seconds)
        self.record_audit_event(
            actor_role="orchestrator", method="LIFECYCLE", path="/pull-worker:stop",
            outcome="success", request_id=worker_id,
        )
        return self._worker_status_payload(status)

    def pull_worker_status(self, worker_id: str) -> dict[str, Any] | None:
        status = self.pull_worker_provider.status(worker_id)
        return None if status is None else self._worker_status_payload(status)

    # Durable pull-worker deployments ---------------------------------
    # Only a configured profile can turn durable intent into a provider call.
    # This means the database/API never holds bootstrap tokens, filesystem
    # paths, images, mounts, networks, or arbitrary command arguments.
    def _deployment(self, deployment_id: str) -> dict[str, Any] | None:
        with self._connection() as db:
            return _row(db.execute("SELECT * FROM worker_deployments WHERE deployment_id=?", (deployment_id,)).fetchone())

    def worker_deployment(self, deployment_id: str) -> dict[str, Any] | None:
        return self._deployment(deployment_id)

    def worker_deployments(self) -> list[dict[str, Any]]:
        with self._connection() as db:
            return [_row(row) for row in db.execute("SELECT * FROM worker_deployments ORDER BY created_at DESC").fetchall()]  # type: ignore[list-item]

    def _record_deployment_event(self, deployment_id: str, event_type: str, message: str, data: dict[str, Any]) -> None:
        now = _now()
        with self._connection() as db:
            sequence = int(db.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM worker_deployment_events WHERE deployment_id=?", (deployment_id,)).fetchone()[0])
            db.execute("INSERT INTO worker_deployment_events VALUES(?,?,?,?,?,?)", (deployment_id, sequence, now, event_type, message, _json(data)))

    def create_worker_deployment(self, *, name: str, pool_id: str, profile_id: str) -> dict[str, Any]:
        profile = self.deployment_profiles.get(profile_id)
        if profile is None:
            raise ValueError("worker deployment profile is not configured")
        pool = self.worker_pool(pool_id)
        if pool is None:
            raise KeyError(f"worker pool does not exist: {pool_id}")
        if not pool["enabled"]:
            raise ValueError("worker pool is disabled")
        unsupported = set(profile.executors) - set(pool["allowed_actions"])
        if unsupported:
            raise ValueError(f"worker pool does not allow actions: {', '.join(sorted(unsupported))}")
        if not name or len(name) > 120 or any(char in name for char in "\x00\r\n"):
            raise ValueError("deployment name is invalid")
        deployment_id = f"deploy-{uuid.uuid4().hex}"
        now = _now()
        with self._connection() as db:
            try:
                db.execute("""INSERT INTO worker_deployments(
                    deployment_id,name,pool_id,provider,profile_id,allowed_actions_json,capabilities_json,
                    desired_state,state,endpoint,error,last_reconciled_at,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    deployment_id, name, pool_id, profile.provider, profile_id, _json(list(profile.executors)),
                    _json(dict(profile.capabilities)), "stopped", "stopped", None, None, None, now, now,
                ))
            except sqlite3.IntegrityError as exc:
                raise ValueError("worker deployment name already exists") from exc
        self._record_deployment_event(deployment_id, "created", "Worker deployment created", {"provider": profile.provider, "profile_id": profile_id})
        return self._deployment(deployment_id)  # type: ignore[return-value]

    def _deployment_spec(self, deployment: dict[str, Any]) -> PullWorkerSpec:
        profile = self.deployment_profiles.get(str(deployment["profile_id"]))
        if profile is None or profile.provider != deployment["provider"]:
            raise RuntimeError("deployment profile is unavailable; configure it before lifecycle actions")
        return profile.spec_for(str(deployment["deployment_id"]), str(deployment["name"]), str(deployment["pool_id"]))

    def start_worker_deployment(self, deployment_id: str) -> dict[str, Any]:
        deployment = self._deployment(deployment_id)
        if deployment is None:
            raise KeyError(f"worker deployment does not exist: {deployment_id}")
        provider = self.deployment_providers.get(str(deployment["provider"]))
        if provider is None:
            raise RuntimeError("worker deployment provider is unavailable")
        pool = self.worker_pool(str(deployment["pool_id"]))
        if pool is None or not pool["enabled"]:
            raise ValueError("worker deployment pool is unavailable or disabled")
        status = provider.start(self._deployment_spec(deployment))
        now = _now()
        with self._connection() as db:
            db.execute("UPDATE worker_deployments SET desired_state='running',state=?,endpoint=?,error=NULL,last_reconciled_at=?,updated_at=? WHERE deployment_id=?", (status.state, status.endpoint, now, now, deployment_id))
        self._record_deployment_event(deployment_id, "started", "Worker deployment started", self._worker_status_payload(status))
        return self._deployment(deployment_id)  # type: ignore[return-value]

    def stop_worker_deployment(self, deployment_id: str, *, timeout_seconds: float = 10.0) -> dict[str, Any]:
        deployment = self._deployment(deployment_id)
        if deployment is None:
            raise KeyError(f"worker deployment does not exist: {deployment_id}")
        provider = self.deployment_providers.get(str(deployment["provider"]))
        if provider is None:
            raise RuntimeError("worker deployment provider is unavailable")
        status = provider.stop(deployment_id, timeout_seconds=timeout_seconds)
        now = _now()
        with self._connection() as db:
            db.execute("UPDATE worker_deployments SET desired_state='stopped',state=?,endpoint=?,error=NULL,last_reconciled_at=?,updated_at=? WHERE deployment_id=?", (status.state, status.endpoint, now, now, deployment_id))
        self._record_deployment_event(deployment_id, "stopped", "Worker deployment stopped", self._worker_status_payload(status))
        return self._deployment(deployment_id)  # type: ignore[return-value]

    def reconcile_worker_deployments(self) -> list[dict[str, Any]]:
        """Reconcile known provider identities; never inspect/adopt local PIDs."""
        reconciled: list[dict[str, Any]] = []
        for deployment in self.worker_deployments():
            provider = self.deployment_providers.get(str(deployment["provider"]))
            status = provider.status(str(deployment["deployment_id"])) if provider else None
            state = status.state if status else "unknown"
            endpoint = status.endpoint if status else deployment.get("endpoint")
            error = None if status else "provider has no known deployment instance"
            now = _now()
            with self._connection() as db:
                db.execute("UPDATE worker_deployments SET state=?,endpoint=?,error=?,last_reconciled_at=?,updated_at=? WHERE deployment_id=?", (state, endpoint, error, now, now, deployment["deployment_id"]))
            self._record_deployment_event(str(deployment["deployment_id"]), "reconciled", "Worker deployment reconciled", {"state": state})
            reconciled.append(self._deployment(str(deployment["deployment_id"])) or deployment)
        return reconciled

    # Managed worker lifecycle -----------------------------------------
    # A desired worker is durable control-plane state.  The provider owns
    # privileged creation/termination and receives a fixed typed spec rather
    # than any browser-supplied command string.
    @staticmethod
    def _worker_status_payload(status: Any) -> dict[str, Any]:
        return {
            "worker_id": status.worker_id,
            "state": status.state,
            "pid": status.pid,
            "endpoint": status.endpoint,
            "started_at": status.started_at,
            "exit_code": status.exit_code,
        }

    def _managed_worker_spec(self, worker: dict[str, Any]) -> LocalWorkerSpec:
        if worker["provider"] != "local-process":
            raise ValueError(f"Provider {worker['provider']!r} does not use a local worker specification")
        return LocalWorkerSpec(
            worker_id=worker["worker_id"], host=worker["host"], port=int(worker["port"]),
            compute_queue_size=int(worker["compute_queue_size"]),
            compute_worker_slots=int(worker["compute_worker_slots"]),
            compute_cpu_capacity=(int(worker["compute_cpu_capacity"])
                                  if worker.get("compute_cpu_capacity") is not None else None),
        )

    def _record_managed_worker_event(self, worker_id: str, event_type: str, message: str, data: dict[str, Any]) -> None:
        with self._connection() as db:
            sequence = db.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM managed_worker_events WHERE worker_id=?",
                (worker_id,),
            ).fetchone()[0]
            db.execute(
                "INSERT INTO managed_worker_events VALUES (?, ?, ?, ?, ?, ?)",
                (worker_id, sequence, _now(), event_type, message, _json(data)),
            )

    def create_managed_worker(
        self,
        *,
        name: str,
        host: str = "127.0.0.1",
        port: int = 8100,
        compute_queue_size: int = 128,
        compute_worker_slots: int = 1,
        compute_cpu_capacity: int | None = None,
        provider: str = "local-process",
        role: str = "compute",
    ) -> dict[str, Any]:
        if provider not in self.worker_providers:
            raise ValueError(f"Unsupported worker provider: {provider}")
        if role != "compute":
            raise ValueError("Only the compute role is lifecycle-managed in this migration slice")
        worker_id, now = str(uuid.uuid4()), _now()
        # Construct before the INSERT so unsafe ports/identifiers never become
        # durable desired state.
        spec = LocalWorkerSpec(
            worker_id=worker_id, host=host, port=port,
            compute_queue_size=compute_queue_size,
            compute_worker_slots=compute_worker_slots,
            compute_cpu_capacity=compute_cpu_capacity,
        )
        with self._connection() as db:
            db.execute("""INSERT INTO managed_workers(
                worker_id,name,role,provider,desired_state,host,port,compute_queue_size,
                compute_worker_slots,compute_cpu_capacity,endpoint_id,state,pid,error,last_health_at,
                created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                worker_id, name.strip() or f"Local worker {port}", role, provider, "stopped",
                spec.host, spec.port, spec.compute_queue_size, spec.compute_worker_slots,
                spec.compute_cpu_capacity, None, "stopped", None, None, None, now, now,
            ))
        self._record_managed_worker_event(worker_id, "created", "Managed worker created", {"provider": provider, "role": role})
        return self.managed_worker(worker_id)  # type: ignore[return-value]

    def managed_worker(self, worker_id: str) -> dict[str, Any] | None:
        with self._connection() as db:
            return _row(db.execute("SELECT * FROM managed_workers WHERE worker_id=?", (worker_id,)).fetchone())

    def managed_workers(self) -> list[dict[str, Any]]:
        with self._connection() as db:
            return [_row(row) for row in db.execute("SELECT * FROM managed_workers ORDER BY created_at DESC").fetchall()]  # type: ignore[list-item]

    def managed_worker_events(self, worker_id: str) -> list[dict[str, Any]]:
        with self._connection() as db:
            rows = db.execute("SELECT * FROM managed_worker_events WHERE worker_id=? ORDER BY sequence", (worker_id,)).fetchall()
        return [{**dict(row), "data": json.loads(row["data_json"])} for row in rows]

    def start_managed_worker(self, worker_id: str) -> dict[str, Any]:
        worker = self.managed_worker(worker_id)
        if worker is None:
            raise KeyError(worker_id)
        provider = self.worker_providers.get(worker["provider"])
        if provider is None:
            raise ValueError(f"Worker provider is unavailable: {worker['provider']}")
        status = provider.start(self._managed_worker_spec(worker))
        endpoint = self.register_compute_endpoint(name=worker["name"], base_url=status.endpoint)
        now = _now()
        with self._connection() as db:
            db.execute("""UPDATE managed_workers SET desired_state='running',state=?,pid=?,endpoint_id=?,
                error=NULL,updated_at=? WHERE worker_id=?""", (status.state, status.pid, endpoint["endpoint_id"], now, worker_id))
        self._record_managed_worker_event(worker_id, "started", "Local worker process started", self._worker_status_payload(status))
        return self.managed_worker(worker_id)  # type: ignore[return-value]

    def stop_managed_worker(self, worker_id: str, *, timeout_seconds: float = 10.0) -> dict[str, Any]:
        worker = self.managed_worker(worker_id)
        if worker is None:
            raise KeyError(worker_id)
        provider = self.worker_providers.get(worker["provider"])
        if provider is None:
            raise ValueError(f"Worker provider is unavailable: {worker['provider']}")
        status = provider.stop(worker_id, timeout_seconds=timeout_seconds)
        now = _now()
        with self._connection() as db:
            db.execute("UPDATE managed_workers SET desired_state='stopped',state=?,pid=?,error=NULL,updated_at=? WHERE worker_id=?", (status.state, status.pid, now, worker_id))
            if worker.get("endpoint_id"):
                db.execute("UPDATE compute_endpoints SET enabled=0,status='stopped',updated_at=? WHERE endpoint_id=?", (now, worker["endpoint_id"]))
        self._record_managed_worker_event(worker_id, "stopped", "Local worker process stopped", self._worker_status_payload(status))
        return self.managed_worker(worker_id)  # type: ignore[return-value]

    def reconcile_managed_worker(self, worker_id: str) -> dict[str, Any]:
        worker = self.managed_worker(worker_id)
        if worker is None:
            raise KeyError(worker_id)
        provider = self.worker_providers.get(worker["provider"])
        if provider is None:
            raise ValueError(f"Worker provider is unavailable: {worker['provider']}")
        status = provider.status(worker_id)
        if status is None:
            state, pid, error, health_data = "unknown", None, "Worker is not managed by this orchestrator process", None
        else:
            health = provider.health(worker_id)
            state, pid = status.state, status.pid
            error = None if health.ready else health.detail
            health_data = {"ready": health.ready, "status_code": health.status_code, "detail": health.detail}
        now = _now()
        with self._connection() as db:
            db.execute("UPDATE managed_workers SET state=?,pid=?,error=?,last_health_at=?,updated_at=? WHERE worker_id=?", (state, pid, error, now, now, worker_id))
        self._record_managed_worker_event(worker_id, "reconciled", "Managed worker reconciled", {"state": state, "pid": pid, "health": health_data})
        return self.managed_worker(worker_id)  # type: ignore[return-value]

    def refresh_compute_endpoint(self, endpoint_id: str) -> dict[str, Any]:
        endpoint = self.compute_endpoint(endpoint_id)
        if endpoint is None:
            raise KeyError(endpoint_id)
        now = _now()
        try:
            readiness = self._request(endpoint["base_url"], "GET", "/health/ready")
            compute = self._request(endpoint["base_url"], "GET", "/compute/status")
            status = "ready" if readiness.get("status") == "ready" and compute.get("status") == "ready" else "degraded"
            error = None
            workers, queue = compute.get("workers") or [], dict(compute.get("queue") or {})
            # Keep Serve's live capacity snapshot beside the existing queue
            # payload for backward-compatible endpoint responses.
            queue["resources"] = compute.get("resources") or {}
        except RuntimeError as exc:
            status, error, readiness, workers, queue = "unavailable", str(exc), None, [], {}
        with self._connection() as db:
            db.execute("""UPDATE compute_endpoints SET status=?, last_checked_at=?, error=?,
                readiness_json=?, workers_json=?, queue_json=?, updated_at=? WHERE endpoint_id=?""",
                (status, now, error, _json(readiness) if readiness is not None else None,
                 _json(workers), _json(queue), now, endpoint_id))
        return self.compute_endpoint(endpoint_id)  # type: ignore[return-value]

    def preflight(self, specification_id: str, endpoint_id: str, *, refresh: bool = True) -> dict[str, Any]:
        specification = self.specification(specification_id)
        if specification is None:
            raise KeyError(specification_id)
        endpoint = self.refresh_compute_endpoint(endpoint_id) if refresh else self.compute_endpoint(endpoint_id)
        if endpoint is None:
            raise KeyError(endpoint_id)
        reasons: list[str] = []
        if not endpoint["enabled"]:
            reasons.append("Compute endpoint is disabled")
        if endpoint["status"] != "ready":
            reasons.append(endpoint.get("error") or f"Compute endpoint is {endpoint['status']}")
        workers = endpoint.get("workers") or []
        capable = [worker for worker in workers if specification["action"] in (worker.get("capabilities") or {}).get("actions", [])]
        if endpoint["status"] == "ready" and not capable:
            reasons.append(f"No worker supports the {specification['action']} action")
        requested_resources = specification.get("resources") or {}
        requested_gpus = int(requested_resources.get("gpu_count") or 0)
        if capable and requested_gpus > max((len((worker.get("capabilities") or {}).get("gpus") or []) for worker in capable), default=0):
            reasons.append(f"Run requests {requested_gpus} GPU(s), but no capable worker advertises that capacity")
        requested_gpu_ids = requested_resources.get("gpu_ids")
        if isinstance(requested_gpu_ids, list) and requested_gpu_ids:
            advertised_gpu_ids = {
                str(gpu.get("id"))
                for worker in capable
                for gpu in ((worker.get("capabilities") or {}).get("gpus") or [])
                if isinstance(gpu, dict) and gpu.get("id") is not None
            }
            missing_gpu_ids = [str(item) for item in requested_gpu_ids if str(item) not in advertised_gpu_ids]
            if missing_gpu_ids:
                reasons.append(
                    "Selected GPU(s) are not advertised by a capable worker: "
                    + ", ".join(missing_gpu_ids)
                )
        queue = endpoint.get("queue") or {}
        if queue.get("capacity") is not None and queue.get("depth", 0) >= queue["capacity"]:
            reasons.append("Compute queue is full")
        return {
            "ready": not reasons,
            "reasons": reasons,
            "specification_id": specification_id,
            "action": specification["action"],
            "requested_resources": specification["resources"],
            "endpoint": endpoint,
            "capable_workers": capable,
        }

    def file_roots(self) -> list[dict[str, str]]:
        return [{"id": root_id, "path": str(path)} for root_id, path in self.browse_roots.items()]

    def upload_destination(self, kind: str, filename: str) -> Path:
        allowed_extensions = {
            "datasets": {".sqlite"},
            "configs": {".toml"},
            "models": {".keras", ".h5", ".hdf5"},
        }
        if kind not in allowed_extensions:
            raise ValueError(f"Unsupported upload kind: {kind}")
        candidate = Path(filename)
        if candidate.name != filename or filename in {"", ".", ".."}:
            raise ValueError("Upload filename must be a plain filename")
        if candidate.suffix.lower() not in allowed_extensions[kind]:
            expected = ", ".join(sorted(allowed_extensions[kind]))
            raise ValueError(f"{kind} uploads must use one of: {expected}")
        destination = (self.artifact_root / "uploads" / kind / candidate.name).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    # Resumable browser transfers ------------------------------------
    # The old one-request upload remains useful to simple CLI clients.  The
    # browser uses these durable sessions instead: a lost connection only
    # requires re-sending the missing chunk, never the whole dataset/model.
    _UPLOAD_CHUNK_BYTES = 16 * 1024 * 1024

    @staticmethod
    def _upload_session_view(row: sqlite3.Row | None, *, received_bytes: int = 0, part_count: int = 0) -> dict[str, Any]:
        if row is None:
            raise KeyError("Upload session was not found")
        return {
            "upload_id": row["upload_id"], "kind": row["kind"], "filename": row["filename"],
            "size_bytes": row["size_bytes"], "chunk_size_bytes": row["chunk_size_bytes"],
            "status": row["status"], "received_bytes": received_bytes, "part_count": part_count,
            "completed_at": row["completed_at"],
            # Only a completed, operator-owned location may be used by the
            # existing registration APIs. Never disclose partial-file paths.
            **({"path": row["destination_path"]} if row["status"] == "completed" else {}),
        }

    def create_upload_session(self, *, kind: str, filename: str, size_bytes: int) -> dict[str, Any]:
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 1:
            raise ValueError("Upload size must be a positive integer")
        if size_bytes > self.upload_limit_bytes:
            raise ValueError("Upload exceeds configured size limit")
        destination = self.upload_destination(kind, filename)
        if destination.exists():
            raise FileExistsError("An upload with this filename already exists")
        upload_id, now = uuid.uuid4().hex, _now()
        session_root = self.artifact_root / "uploads" / ".sessions"
        session_root.mkdir(parents=True, exist_ok=True)
        temporary = session_root / f"{upload_id}.part"
        # Exclusive creation ensures a restart cannot accidentally append to
        # data owned by an older session.
        with temporary.open("xb"):
            pass
        with self._connection() as db:
            db.execute(
                """INSERT INTO upload_sessions(
                    upload_id,kind,filename,size_bytes,chunk_size_bytes,temporary_path,destination_path,status,created_at,completed_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (upload_id, kind, filename, size_bytes, self._UPLOAD_CHUNK_BYTES,
                 str(temporary), str(destination), "uploading", now, None, now),
            )
            row = db.execute("SELECT * FROM upload_sessions WHERE upload_id=?", (upload_id,)).fetchone()
        return self._upload_session_view(row)

    def upload_session(self, upload_id: str) -> dict[str, Any]:
        with self._connection() as db:
            row = db.execute("SELECT * FROM upload_sessions WHERE upload_id=?", (upload_id,)).fetchone()
            if row is None:
                raise KeyError("Upload session was not found")
            totals = db.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) AS received_bytes, COUNT(*) AS part_count "
                "FROM upload_session_parts WHERE upload_id=?", (upload_id,),
            ).fetchone()
        return self._upload_session_view(row, received_bytes=int(totals["received_bytes"]), part_count=int(totals["part_count"]))

    def prepare_upload_session_part(self, upload_id: str, *, offset_bytes: int, size_bytes: int, total_bytes: int) -> dict[str, Any]:
        """Validate a chunk before the API writes its request body to disk."""
        if offset_bytes < 0 or size_bytes < 1:
            raise ValueError("Upload chunk range is invalid")
        with self._connection() as db:
            row = db.execute("SELECT * FROM upload_sessions WHERE upload_id=?", (upload_id,)).fetchone()
            if row is None:
                raise KeyError("Upload session was not found")
            if row["status"] != "uploading":
                raise ValueError("Upload session is not accepting parts")
            if total_bytes != row["size_bytes"] or offset_bytes + size_bytes > total_bytes:
                raise ValueError("Upload chunk does not match the declared file size")
            chunk_size = int(row["chunk_size_bytes"])
            if offset_bytes % chunk_size != 0 or (size_bytes != chunk_size and offset_bytes + size_bytes != total_bytes):
                raise ValueError("Upload chunks must use the session chunk size except for the final chunk")
            existing = db.execute(
                "SELECT size_bytes,sha256 FROM upload_session_parts WHERE upload_id=? AND offset_bytes=?",
                (upload_id, offset_bytes),
            ).fetchone()
        if existing is not None:
            return {"idempotent": True, "size_bytes": int(existing["size_bytes"]), "sha256": existing["sha256"]}
        return {"idempotent": False, "temporary_path": row["temporary_path"]}

    def record_upload_session_part(self, upload_id: str, *, offset_bytes: int, size_bytes: int, sha256_hex: str) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{64}", sha256_hex):
            raise ValueError("Upload chunk digest is invalid")
        now = _now()
        with self._connection() as db:
            row = db.execute("SELECT * FROM upload_sessions WHERE upload_id=?", (upload_id,)).fetchone()
            if row is None:
                raise KeyError("Upload session was not found")
            if row["status"] != "uploading":
                raise ValueError("Upload session is not accepting parts")
            try:
                db.execute(
                    "INSERT INTO upload_session_parts(upload_id,offset_bytes,size_bytes,sha256,created_at) VALUES(?,?,?,?,?)",
                    (upload_id, offset_bytes, size_bytes, sha256_hex, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("Upload chunk was already recorded") from exc
        return self.upload_session(upload_id)

    def complete_upload_session(self, upload_id: str) -> dict[str, Any]:
        """Atomically expose a fully received file to dataset/model workflows."""
        with self._connection() as db:
            row = db.execute("SELECT * FROM upload_sessions WHERE upload_id=?", (upload_id,)).fetchone()
            if row is None:
                raise KeyError("Upload session was not found")
            if row["status"] == "completed":
                return self.upload_session(upload_id)
            if row["status"] != "uploading":
                raise ValueError("Upload session cannot be completed")
            parts = db.execute(
                "SELECT offset_bytes,size_bytes FROM upload_session_parts WHERE upload_id=? ORDER BY offset_bytes",
                (upload_id,),
            ).fetchall()
            expected = 0
            for part in parts:
                if int(part["offset_bytes"]) != expected:
                    raise ValueError("Upload is incomplete")
                expected += int(part["size_bytes"])
            if expected != int(row["size_bytes"]):
                raise ValueError("Upload is incomplete")
            temporary, destination = Path(row["temporary_path"]), Path(row["destination_path"])
            if not temporary.is_file() or temporary.stat().st_size != expected:
                raise ValueError("Upload data is incomplete")
            if destination.exists():
                raise FileExistsError("An upload with this filename already exists")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(destination)
            now = _now()
            db.execute("UPDATE upload_sessions SET status='completed',completed_at=?,updated_at=? WHERE upload_id=?", (now, now, upload_id))
        return self.upload_session(upload_id)

    def cancel_upload_session(self, upload_id: str) -> dict[str, Any]:
        with self._connection() as db:
            row = db.execute("SELECT * FROM upload_sessions WHERE upload_id=?", (upload_id,)).fetchone()
            if row is None:
                raise KeyError("Upload session was not found")
            if row["status"] == "completed":
                raise ValueError("Completed uploads cannot be cancelled")
            Path(row["temporary_path"]).unlink(missing_ok=True)
            now = _now()
            db.execute("UPDATE upload_sessions SET status='cancelled',updated_at=? WHERE upload_id=?", (now, upload_id))
        return self.upload_session(upload_id)

    def dataset_download_path(self, dataset_id: str) -> Path:
        dataset = self.dataset(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        path = Path(dataset["path"]).resolve()
        self._require_browse_path(path)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        return path

    def artifact_download_archive(self, artifact_id: str, *, chunk_size: int = 1024 * 1024) -> tuple[str, Any]:
        """Create a bounded-memory archive stream for a sealed model artifact."""
        artifact = self.artifact(artifact_id)
        if artifact is None:
            raise KeyError(artifact_id)
        root = Path(artifact["path"]).resolve()
        self._require_browse_path(root)
        if not root.is_dir() or root.is_symlink():
            raise FileNotFoundError(root)
        temporary = tempfile.TemporaryFile(mode="w+b")
        try:
            entries = [root, *sorted(root.rglob("*"), key=lambda item: item.as_posix())]
            with tarfile.open(fileobj=temporary, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
                for item in entries:
                    if item.is_symlink():
                        raise ValueError("Artifact contains a symlink and cannot be downloaded")
                    relative = item.relative_to(root)
                    arcname = "artifact" if relative == Path(".") else f"artifact/{relative.as_posix()}"
                    info = archive.gettarinfo(str(item), arcname=arcname)
                    if info.isdir():
                        archive.addfile(info)
                    elif info.isreg():
                        with item.open("rb") as payload:
                            archive.addfile(info, payload)
                    else:
                        raise ValueError("Artifact contains a non-regular file and cannot be downloaded")
            temporary.seek(0)
        except Exception:
            temporary.close()
            raise
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(artifact.get("name") or artifact_id)).strip(".-") or artifact_id
        def stream() -> Any:
            try:
                while chunk := temporary.read(chunk_size):
                    yield chunk
            finally:
                temporary.close()
        return f"{safe_name}.tar.gz", stream()

    def inference_result_download_archive(self, inference_run_id: str, *, chunk_size: int = 1024 * 1024) -> tuple[str, Any]:
        """Archive one published inference result through its durable ref."""
        run = self.inference_run(inference_run_id)
        if run is None: raise KeyError(inference_run_id)
        raw = run.get("output_ref")
        if not isinstance(raw, dict): raise ValueError("Inference result is not published")
        ref = ArtifactRef.from_dict(raw)
        root = self.artifact_store.resolve(ref)
        temporary = tempfile.TemporaryFile(mode="w+b")
        try:
            entries = [root, *sorted(root.rglob("*"), key=lambda item: item.as_posix())]
            with tarfile.open(fileobj=temporary, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
                for item in entries:
                    if item.is_symlink(): raise ValueError("Inference result contains a symlink")
                    relative = item.relative_to(root)
                    arcname = "inference-result" if relative == Path(".") else f"inference-result/{relative.as_posix()}"
                    info = archive.gettarinfo(str(item), arcname=arcname)
                    if info.isdir(): archive.addfile(info)
                    elif info.isreg():
                        with item.open("rb") as payload: archive.addfile(info, payload)
                    else: raise ValueError("Inference result contains a non-regular file")
            temporary.seek(0)
        except Exception:
            temporary.close()
            raise
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(run["name"])).strip(".-") or inference_run_id
        def stream() -> Any:
            try:
                while chunk := temporary.read(chunk_size): yield chunk
            finally: temporary.close()
        return f"{safe_name}.tar.gz", stream()

    def files(self, root_id: str, relative_path: str = ".") -> dict[str, Any]:
        """List an allow-listed directory without exposing arbitrary filesystem paths."""
        try:
            root = self.browse_roots[root_id]
        except KeyError as exc:
            raise KeyError(f"Unknown browse root: {root_id}") from exc
        requested = Path(relative_path)
        if requested.is_absolute():
            raise ValueError("File-browser paths must be relative to the selected root")
        directory = (root / requested).resolve()
        try:
            relative = directory.relative_to(root)
        except ValueError as exc:
            raise ValueError("File-browser path escapes the selected root") from exc
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        hidden = {".git", ".venv", "node_modules", ".svelte-kit", "__pycache__"}
        entries: list[dict[str, Any]] = []
        for entry in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if entry.name in hidden:
                continue
            entry_relative = entry.relative_to(root).as_posix()
            entries.append({
                "name": entry.name,
                "path": entry_relative,
                "kind": "directory" if entry.is_dir() else "file",
                "size_bytes": None if entry.is_dir() else entry.stat().st_size,
            })
            if len(entries) >= 500:
                break
        return {
            "root": {"id": root_id, "path": str(root)},
            "path": relative.as_posix() if relative.as_posix() != "." else "",
            "parent": None if relative == Path(".") else relative.parent.as_posix(),
            "entries": entries,
        }

    def ingest_dataset(self, path: str | Path) -> dict[str, Any]:
        source = Path(path).expanduser().resolve()
        self._require_browse_path(source)
        with sqlite3.connect(source) as dataset_connection:
            info = read_dataset_info(dataset_connection)
            fingerprint = dataset_fingerprint(dataset_connection)
        if info["lifecycle"] != "frozen":
            raise ValueError("Only frozen dataset revisions can be registered for training")
        now = _now()
        payload = (info["dataset_id"], info.get("revision_id"), info.get("name") or source.stem,
                   info.get("dataset_type"), info.get("lifecycle"), fingerprint, str(source), _json(info), now, now)
        with self._connection() as db:
            db.execute("""INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET revision_id=excluded.revision_id,name=excluded.name,dataset_type=excluded.dataset_type,lifecycle=excluded.lifecycle,fingerprint_sha256=excluded.fingerprint_sha256,path=excluded.path,metadata_json=excluded.metadata_json,updated_at=excluded.updated_at""", payload)
            return self.dataset(info["dataset_id"], connection=db)  # type: ignore[return-value]

    def create_recipe(self, *, name: str, config_path: str | Path, description: str = "") -> dict[str, Any]:
        source = Path(config_path).expanduser().resolve()
        self._require_browse_path(source)
        if source.suffix.lower() != ".toml":
            raise ValueError("Training recipes must reference a TOML configuration")
        raw = load_toml(source)
        resolved = deep_merge(DEFAULT_CONFIG, raw)
        run = resolved.get("run") or {}
        data = resolved.get("data") or {}
        task, model = str(run.get("task") or ""), str(run.get("model") or "")
        if task not in {"classification", "segmentation"}:
            raise ValueError("Recipe [run].task must be classification or segmentation")
        if not model or not data.get("input_shape"):
            raise ValueError("Recipe requires [run].model and data.input_shape")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        recipe_id, now = str(uuid.uuid4()), _now()
        summary = {"task": task, "model": model, "input_shape": data["input_shape"], "epochs": resolved.get("training", {}).get("epochs"), "loss": resolved.get("training", {}).get("loss"), "seed": run.get("seed", 123)}
        with self._connection() as db:
            db.execute("INSERT INTO recipes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (recipe_id, name.strip() or source.stem, description, str(source), digest, task, model, _json(summary), now, now))
        return self.recipe(recipe_id)  # type: ignore[return-value]

    def create_training_experiment(self, *, name: str, dataset_id: str, recipe_ids: list[str], seeds: list[int], description: str = "", resources: dict[str, Any] | None = None, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        if not name.strip() or not recipe_ids or not seeds:
            raise ValueError("Training experiments require a name, at least one recipe, and at least one seed")
        dataset = self.dataset(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        if dataset["lifecycle"] != "frozen":
            raise ValueError("Training experiments require a frozen registered dataset")
        recipes = [self.recipe(recipe_id) for recipe_id in recipe_ids]
        if any(recipe is None for recipe in recipes):
            raise KeyError("One or more recipes were not found")
        selected = [recipe for recipe in recipes if recipe is not None]
        experiment_id, now = str(uuid.uuid4()), _now()
        if config_overrides is not None and not isinstance(config_overrides, dict):
            raise ValueError("Configuration overrides must be an object")
        overrides = config_overrides or {}
        allowed_override_sections = {"run", "data", "model", "training", "augmentation", "callbacks", "output", "architecture", "input", "encoder", "stem", "normalization", "pooling", "image_embedding", "metadata", "fusion", "classifier", "preprocessing"}
        unknown_sections = set(overrides) - allowed_override_sections
        if unknown_sections or any(not isinstance(value, dict) for value in overrides.values()):
            raise ValueError("Configuration overrides may contain only supported object-valued configuration sections")
        plan = {"kind": "training", "dataset_id": dataset_id, "recipe_ids": recipe_ids, "seeds": seeds, "resources": resources or {}, "config_overrides": overrides}
        config_dir = self.artifact_root / "experiments" / experiment_id / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute("INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (experiment_id, None, name.strip(), description, dataset_id, "expanded", _json(plan), now, now))
            ordinal = 0
            for recipe in selected:
                for seed in seeds:
                    ordinal += 1
                    specification_id = str(uuid.uuid4())
                    source = deep_merge(load_toml(recipe["config_path"]), overrides)
                    source.setdefault("run", {})["seed"] = int(seed)
                    generated = config_dir / f"{ordinal:03d}-{recipe['recipe_id'][:8]}-seed-{seed}.toml"
                    import tomli_w
                    generated.write_text(tomli_w.dumps(source), encoding="utf-8")
                    parameters = self._assign_output_path("train", {"config": str(generated), "input": dataset["path"], "dataset_id": dataset_id, "recipe_id": recipe["recipe_id"], "seed": int(seed)}, specification_id)
                    digest = hashlib.sha256(_json({"parameters": parameters, "dataset_fingerprint": dataset["fingerprint_sha256"]}).encode()).hexdigest()
                    spec_name = f"{name}-{recipe['model']}-seed-{seed}".replace(" ", "-")
                    db.execute("INSERT INTO run_specifications VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (specification_id, experiment_id, ordinal, spec_name, "train", _json(parameters), _json(resources or {}), digest, "planned", None, now, now))
        return self.experiment(experiment_id)  # type: ignore[return-value]

    def create_model_import(self, *, name: str, model_path: str | Path, info_path: str | Path, dataset_id: str | None = None, description: str = "", resources: dict[str, Any] | None = None) -> dict[str, Any]:
        source, info = Path(model_path).expanduser().resolve(), Path(info_path).expanduser().resolve()
        self._require_browse_path(source)
        self._require_browse_path(info)
        if source.suffix.lower() not in {".keras", ".h5", ".hdf5"}:
            raise ValueError("Model import requires a .keras, .h5, or .hdf5 source")
        if info.suffix.lower() != ".toml":
            raise ValueError("Model import metadata must be a TOML file")
        metadata = load_toml(info)
        if not isinstance(metadata.get("product"), dict):
            raise ValueError("Model import metadata requires a [product] section")
        dataset = self.dataset(dataset_id) if dataset_id else None
        if dataset_id and dataset is None:
            raise KeyError(dataset_id)
        experiment_id, specification_id, now = str(uuid.uuid4()), str(uuid.uuid4()), _now()
        plan = {"kind": "model_import", "model_path": str(source), "info_path": str(info), "dataset_id": dataset_id, "resources": resources or {}}
        parameters = self._assign_output_path("model_ingest", {"model": str(source), "info": str(info), **({"dataset": dataset["path"]} if dataset else {})}, specification_id)
        digest = hashlib.sha256(_json(parameters).encode()).hexdigest()
        with self._connection() as db:
            db.execute("INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (experiment_id, None, name.strip() or source.stem, description, dataset_id, "expanded", _json(plan), now, now))
            db.execute("INSERT INTO run_specifications VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (specification_id, experiment_id, 1, f"{name or source.stem}-import", "model_ingest", _json(parameters), _json(resources or {}), digest, "planned", None, now, now))
        return self.experiment(experiment_id)  # type: ignore[return-value]

    def _require_browse_path(self, candidate: Path) -> None:
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        if not any(candidate.is_relative_to(root) for root in self.browse_roots.values()):
            raise ValueError("Path is outside the configured workspace roots")

    def scan(self, root: str | Path, *, only_unindexed: bool = False) -> dict[str, Any]:
        root_path = Path(root).expanduser().resolve()
        self._require_browse_path(root_path)
        if not root_path.is_dir():
            raise NotADirectoryError(root_path)
        discovered: list[str] = []
        already_indexed: list[str] = []
        skipped: list[dict[str, str]] = []
        for manifest_path in root_path.rglob("artifact.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                schema = manifest.get("artifact_schema", {})
                if schema.get("name") != "oracle_builder_model_run":
                    raise ValueError("unsupported artifact schema")
                artifact_id = str(uuid.UUID(str(manifest["artifact_id"])))
                if only_unindexed and self.artifact(artifact_id) is not None:
                    already_indexed.append(artifact_id)
                    continue
                from oracle_builder.artifacts import validate_run_artifact
                validation = validate_run_artifact(manifest_path.parent)
                if not validation.get("valid"):
                    raise ValueError("artifact validation failed: " + "; ".join(validation.get("errors") or []))
                now = _now()
                model, dataset = manifest.get("model") or {}, manifest.get("dataset") or {}
                facts = self._build_artifact_facts(manifest, manifest_path.parent)
                values = (artifact_id, manifest.get("run_id"), manifest.get("artifact_type", "model_run"), manifest.get("name") or manifest_path.parent.name,
                          facts.get("task") or model.get("task"), facts.get("architecture") or model.get("architecture"), facts.get("variant") or model.get("variant"), manifest.get("status"), manifest.get("lifecycle"), dataset.get("dataset_id"), dataset.get("fingerprint_sha256"), manifest.get("fingerprint_sha256"), str(manifest_path.parent), _json(manifest), now, now)
                with self._connection() as db:
                    db.execute("""INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                      ON CONFLICT(artifact_id) DO UPDATE SET run_id=excluded.run_id,artifact_type=excluded.artifact_type,name=excluded.name,task=excluded.task,architecture=excluded.architecture,variant=excluded.variant,status=excluded.status,lifecycle=excluded.lifecycle,dataset_id=excluded.dataset_id,dataset_fingerprint_sha256=excluded.dataset_fingerprint_sha256,fingerprint_sha256=excluded.fingerprint_sha256,path=excluded.path,manifest_json=excluded.manifest_json,updated_at=excluded.updated_at""", values)
                    db.execute("""INSERT INTO artifact_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(artifact_id) DO UPDATE SET training_set=excluded.training_set,classifier_type=excluded.classifier_type,stem_size=excluded.stem_size,macro_f1=excluded.macro_f1,loss=excluded.loss,training_seconds=excluded.training_seconds,facts_json=excluded.facts_json,updated_at=excluded.updated_at""",
                        (artifact_id, facts.get("training_set"), facts.get("classifier_type"), facts.get("stem_size"), facts.get("macro_f1"), facts.get("loss"), facts.get("training_seconds"), _json(facts), now))
                discovered.append(artifact_id)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                skipped.append({"path": str(manifest_path.parent), "reason": str(exc)})
        return {"root": str(root_path), "artifacts": discovered, "already_indexed": already_indexed, "skipped": skipped}

    def reindex_artifact_catalog(self, artifact_ids: list[str] | None = None) -> dict[str, Any]:
        """Refresh DB-owned catalog facts from registered artifacts only.

        This intentionally reads the sealed run directory and writes only the
        control-plane SQLite database. It never calls artifact update/seal APIs
        and never changes a file within the artifact.
        """
        requested = list(dict.fromkeys(artifact_ids or []))
        with self._connection() as db:
            if requested:
                rows = db.execute(f"SELECT * FROM artifacts WHERE artifact_id IN ({','.join('?' for _ in requested)})", requested).fetchall()
            else:
                rows = db.execute("SELECT * FROM artifacts ORDER BY artifact_id").fetchall()
        found = {row["artifact_id"] for row in rows}
        missing = [artifact_id for artifact_id in requested if artifact_id not in found]
        refreshed: list[str] = []
        skipped: list[dict[str, str]] = []
        for row in rows:
            artifact = _row(row)
            assert artifact is not None
            root = Path(artifact["path"]).resolve()
            try:
                # Registration only occurs from allow-listed browse roots; keep
                # that invariant on later maintenance calls too.
                self._require_browse_path(root)
                manifest = artifact["manifest"]
                if not isinstance(manifest, dict):
                    raise ValueError("Registered artifact has no manifest")
                facts = self._build_artifact_facts(manifest, root)
                now = _now()
                with self._connection() as db:
                    db.execute("""UPDATE artifacts SET task=?,architecture=?,variant=?,updated_at=? WHERE artifact_id=?""",
                        (facts.get("task") or artifact.get("task"), facts.get("architecture") or artifact.get("architecture"), facts.get("variant") or artifact.get("variant"), now, artifact["artifact_id"]))
                    db.execute("""INSERT INTO artifact_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(artifact_id) DO UPDATE SET training_set=excluded.training_set,classifier_type=excluded.classifier_type,stem_size=excluded.stem_size,macro_f1=excluded.macro_f1,loss=excluded.loss,training_seconds=excluded.training_seconds,facts_json=excluded.facts_json,updated_at=excluded.updated_at""",
                        (artifact["artifact_id"], facts.get("training_set"), facts.get("classifier_type"), facts.get("stem_size"), facts.get("macro_f1"), facts.get("loss"), facts.get("training_seconds"), _json(facts), now))
                refreshed.append(artifact["artifact_id"])
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                skipped.append({"artifact_id": artifact["artifact_id"], "reason": str(exc)})
        return {"requested": requested or None, "refreshed": refreshed, "missing": missing, "skipped": skipped, "artifact_files_changed": False}

    def reconcile_startup(self) -> dict[str, Any]:
        """Re-index durable local work without changing its source files.

        This is deliberately safe to repeat.  Sealed run artifacts and frozen
        dataset revisions have stable IDs, so existing database rows are left
        intact and newly discovered records are inserted by the normal catalog
        and ingestion paths.
        """
        result: dict[str, Any] = {"runs_root": str(self.runs_root), "datasets_root": str(self.datasets_root)}
        try:
            known_artifacts = {artifact["artifact_id"] for artifact in self.artifacts()}
            artifact_report = self.scan(self.runs_root)
            result["artifacts"] = {
                "indexed": [artifact_id for artifact_id in artifact_report["artifacts"] if artifact_id not in known_artifacts],
                "refreshed": [artifact_id for artifact_id in artifact_report["artifacts"] if artifact_id in known_artifacts],
                "skipped": artifact_report["skipped"],
            }
        except (OSError, ValueError) as exc:
            result["artifacts"] = {"indexed": [], "refreshed": [], "skipped": [{"path": str(self.runs_root), "reason": str(exc)}]}

        dataset_report = self.scan_training_catalog()
        result["datasets"] = {
            "catalog_entries": len(dataset_report["entries"]),
            **dataset_report["registration"],
        }
        result["expired_worker_leases"] = self.expire_worker_leases()
        result["recovered_publications"] = self.reconcile_worker_publications()
        return result

    def reconcile_worker_publications(self) -> int:
        """Finish DB state after a crash just after atomic artifact publish.

        Filesystem publication is atomic but intentionally occurs before the
        lease/job transaction can record completion.  The local store's
        private staging ledger is the recovery authority for that narrow
        window.  Other stores may omit this optional capability; their normal
        completion transaction remains authoritative.
        """
        published_ref = getattr(self.artifact_store, "published_ref", None)
        if not callable(published_ref):
            return 0
        recovered = 0
        now = _now()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT worker_leases.lease_id,worker_leases.job_id,worker_leases.output_attempt_id,jobs.action FROM worker_leases "
                "JOIN jobs ON jobs.job_id=worker_leases.job_id "
                "WHERE output_attempt_id IS NOT NULL AND output_ref_json IS NULL AND worker_leases.job_id IS NOT NULL"
            ).fetchall()
            for lease in rows:
                try:
                    ref = published_ref(str(lease["job_id"]), str(lease["output_attempt_id"]))
                except (OSError, ValueError):
                    continue
                if ref is None:
                    continue
                recovered_job = db.execute("SELECT * FROM jobs WHERE job_id=?", (lease["job_id"],)).fetchone()
                latest = db.execute('SELECT lease_id FROM execution_attempts WHERE job_id=? ORDER BY ordinal DESC LIMIT 1', (lease['job_id'],)).fetchone()
                if (latest and latest[0] != lease['lease_id']) or recovered_job['cancel_requested_at'] is not None:
                    continue
                stopped = db.execute("SELECT 1 FROM worker_commands WHERE lease_id=? AND action IN ('stop_now','restart') AND status!='superseded'", (lease['lease_id'],)).fetchone()
                if stopped:
                    continue
                recovered_unit = json.loads(recovered_job["work_unit_json"] or "{}")
                execution = recovered_unit.get("parameters", {}).get("queue_execution", {})
                db.execute('UPDATE worker_leases SET output_ref_json=? WHERE lease_id=?', (_json(ref.to_dict()), lease['lease_id']))
                if execution.get("phase") == "verify" and recovered_job["queued_run_id"]:
                    report = json.loads((self.artifact_store.resolve(ref) / "verification.json").read_text(encoding="utf-8"))
                    owner = db.execute("SELECT worker_id FROM worker_leases WHERE lease_id=?", (lease["lease_id"],)).fetchone()[0]
                    self._record_verification_in_connection(db, recovered_job, owner, report, now, str(lease["lease_id"]))
                if recovered_unit.get('schema', {}).get('version') == 2 and recovered_unit.get('phase') == 'train':
                    run = db.execute('SELECT current_job_id FROM execution_runs WHERE run_id=?', (recovered_unit['run_id'],)).fetchone()
                    if run and run[0] == recovered_job['job_id']:
                        segment = self._read_segment_result(self.artifact_store.resolve(ref), recovered_unit)
                        self._commit_segment_in_connection(db, job=recovered_job, lease_id=lease['lease_id'], output_ref=ref, result=segment, now=now)
                db.execute("UPDATE worker_leases SET status='released',outcome='completed',released_at=COALESCE(released_at,?),output_ref_json=?,updated_at=? WHERE lease_id=?", (now, _json(ref.to_dict()), now, lease["lease_id"]))
                db.execute("UPDATE jobs SET status='completed',remote_status='completed',completed_at=COALESCE(completed_at,?),updated_at=? WHERE job_id=? AND status NOT IN ('indexed','cancelled')", (now, now, lease["job_id"]))
                if lease["action"] == "infer" and ref.kind == "inference_result":
                    if recovered_unit.get('parameters', {}).get('shard_id'):
                        self._commit_inference_shard(db, recovered_job, lease['lease_id'], ref, now)
                    else:
                        db.execute("""UPDATE inference_runs SET status='completed',output_ref_json=?,completed_at=COALESCE(completed_at,?),updated_at=?
                            WHERE job_id=?""", (_json(ref.to_dict()), now, now, lease["job_id"]))
                self._finish_execution_attempt_in_connection(db, str(lease["lease_id"]), status="completed", classification="success", error=None, now=now, output_ref=ref)
                self._record_job_event_in_connection(db, str(lease["job_id"]), "publication_reconciled", "Recovered durable artifact publication after restart", {"lease_id": lease["lease_id"], "output_ref": ref.to_dict()})
                recovered += 1
        # Publication and catalog indexing are deliberately separate durable
        # transitions.  If the process died after an atomic store publish but
        # before (or during) catalog indexing, the lease now has an output ref
        # and this pass can safely finish promotion without re-running work.
        with self._connection() as db:
            pending_models = db.execute(
                """SELECT jobs.job_id,jobs.specification_id,jobs.queued_run_id,
                          worker_leases.output_ref_json
                   FROM jobs JOIN worker_leases ON worker_leases.job_id=jobs.job_id
                   WHERE jobs.status='completed' AND worker_leases.output_ref_json IS NOT NULL"""
            ).fetchall()
        for row in pending_models:
            try:
                ref = ArtifactRef.from_dict(json.loads(row["output_ref_json"]))
                if ref.kind != "model_run":
                    continue
                report = self.scan(self.artifact_store.resolve(ref))
                if ref.artifact_id not in report["artifacts"]:
                    continue
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            with self._connection() as db:
                db.execute("BEGIN IMMEDIATE")
                indexed_at = _now()
                db.execute(
                    "UPDATE jobs SET status='indexed',remote_status='indexed',updated_at=? WHERE job_id=? AND status='completed'",
                    (indexed_at, row["job_id"]),
                )
                db.execute(
                    "UPDATE run_specifications SET status='indexed',artifact_id=?,updated_at=? WHERE specification_id=?",
                    (ref.artifact_id, indexed_at, row["specification_id"]),
                )
                db.execute("UPDATE execution_runs SET status='indexed',updated_at=? WHERE current_job_id=?", (indexed_at, row['job_id']))
                if row["queued_run_id"]:
                    db.execute("UPDATE queued_runs SET status='indexed',updated_at=? WHERE queued_run_id=?", (indexed_at, row["queued_run_id"]))
                self._record_job_event_in_connection(
                    db, str(row["job_id"]), "artifact_indexed",
                    "Recovered published training artifact was indexed", {"artifact_id": ref.artifact_id},
                )
            recovered += 1
        return recovered

    @staticmethod
    def _nested(value: dict[str, Any], *keys: str) -> Any:
        current: Any = value
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    def _build_artifact_facts(self, manifest: dict[str, Any], root: Path) -> dict[str, Any]:
        """Extract catalog facts without ever changing an artifact.

        The catalog is deliberately allowed to be more helpful than a legacy
        manifest.  Every fallback is retained alongside its provenance so a
        client never has to mistake an inferred display value for sealed
        evaluation evidence.
        """
        config = self._json_file(root / "config" / "resolved.json") or {}
        summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
        manifest_evaluation = summary.get("evaluation") if isinstance(summary.get("evaluation"), dict) else {}
        summary_evaluation = self._json_file(root / "evaluation" / "evaluation_summary.json") or {}
        evaluation = manifest_evaluation or summary_evaluation
        runtime = self._json_file(root / "provenance" / "runtime.json") or {}
        contract = self._json_file(root / "model" / "contract.json") or {}
        history = self._evidence_csv(root / "metrics" / "history.csv", limit=2_000)
        final_history = history[-1] if history else {}
        model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
        dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
        model_config = config.get("model") if isinstance(config.get("model"), dict) else {}
        outputs = model.get("outputs") if isinstance(model.get("outputs"), dict) else {}
        classifier = self._nested(config, "classifier") or self._nested(config, "model", "classifier") or {}
        stem = self._nested(config, "stem") or self._nested(config, "model", "stem") or {}
        pooling = self._nested(config, "pooling") or self._nested(config, "model", "pooling") or {}
        metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else {}
        posthoc = config.get("posthoc") if isinstance(config.get("posthoc"), dict) else config.get("post_hoc") if isinstance(config.get("post_hoc"), dict) else {}
        training = config.get("training") if isinstance(config.get("training"), dict) else {}
        sources: dict[str, dict[str, str]] = {}
        availability: dict[str, str] = {}

        def choose(field: str, *candidates: tuple[Any, str, str]) -> Any:
            """Use the first meaningful source in explicit provenance order."""
            for value, source, status in candidates:
                if value is not None and value != "":
                    sources[field] = {"source": source, "status": status}
                    availability[field] = status
                    return value
            sources[field] = {"source": "unavailable", "status": "not_recorded"}
            availability[field] = "not_recorded"
            return None

        seconds = choose(
            "training_seconds",
            (runtime.get("training_seconds") if isinstance(runtime, dict) else None, "runtime_provenance", "recorded"),
            (runtime.get("duration_seconds") if isinstance(runtime, dict) else None, "runtime_provenance", "recorded"),
        )
        try:
            artifact_size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
            model_size = sum(path.stat().st_size for path in (root / "model").rglob("*") if path.is_file())
        except OSError:
            artifact_size, model_size = None, None
        summary_text = ""
        try:
            summary_text = (root / "model" / "model_summary.txt").read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
        def summary_number(label: str) -> int | None:
            match = re.search(rf"{label}\s*:\s*([\d,]+)", summary_text, flags=re.IGNORECASE)
            return int(match.group(1).replace(",", "")) if match else None
        metric_values = {key: value for key, value in evaluation.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
        # A history value is operational evidence, not a substitute for a
        # held-out evaluation; mark it inferred when it fills an old catalog.
        for key, value in final_history.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metric_values.setdefault(key.removeprefix("val_"), value)
        evaluation_source = "sealed_manifest" if manifest_evaluation else "evaluation_summary"
        classifier_type = choose(
            "classifier_type",
            (classifier.get("type") if isinstance(classifier, dict) else None, "resolved_config", "recorded"),
            (model_config.get("classifier_type") or model_config.get("classifier"), "resolved_config", "recorded"),
            (self._nested(contract, "classifier", "type") or contract.get("classifier_type") if isinstance(contract, dict) else None, "model_contract", "inferred"),
            ("linear" if model.get("task") == "classification" else None, "legacy_classification_default", "inferred"),
        )
        task = choose("task", (model.get("task"), "sealed_manifest", "recorded"), (config.get("task") if isinstance(config, dict) else None, "resolved_config", "recorded"), (contract.get("task") if isinstance(contract, dict) else None, "model_contract", "inferred"))
        architecture = choose("architecture", (model.get("architecture"), "sealed_manifest", "recorded"), (model_config.get("architecture"), "resolved_config", "recorded"), (contract.get("architecture") if isinstance(contract, dict) else None, "model_contract", "inferred"))
        variant = choose("variant", (model.get("variant"), "sealed_manifest", "recorded"), (model_config.get("variant"), "resolved_config", "recorded"), (contract.get("variant") if isinstance(contract, dict) else None, "model_contract", "inferred"))
        epochs_from_history = max((int(row["epoch"]) + 1 for row in history if isinstance(row.get("epoch"), (int, float)) and not isinstance(row.get("epoch"), bool)), default=None)
        facts = {
            "training_set": choose("training_set", (dataset.get("dataset_id") or dataset.get("name"), "sealed_manifest", "recorded"), (self._nested(config, "data", "dataset_id"), "resolved_config", "recorded")),
            "classifier_type": classifier_type,
            "stem_size": choose("stem_size", (stem.get("filters", stem.get("kernel_size", stem.get("size"))) if isinstance(stem, dict) else None, "resolved_config", "recorded"), (self._nested(contract, "stem", "filters") if isinstance(contract, dict) else None, "model_contract", "inferred")),
            "macro_f1": choose("macro_f1", (manifest_evaluation.get("macro_f1", manifest_evaluation.get("f1_macro")), "sealed_manifest", "recorded"), (summary_evaluation.get("macro_f1", summary_evaluation.get("f1_macro")) if isinstance(summary_evaluation, dict) else None, "evaluation_summary", "recorded"), (final_history.get("val_macro_f1", final_history.get("macro_f1")), "training_history", "inferred")),
            "loss": choose("loss", (manifest_evaluation.get("loss", summary.get("loss")), "sealed_manifest", "recorded"), (summary_evaluation.get("loss") if isinstance(summary_evaluation, dict) else None, "evaluation_summary", "recorded"), (final_history.get("val_loss", final_history.get("loss")), "training_history", "inferred")),
            "training_seconds": seconds,
            "created_at": choose("created_at", (manifest.get("created_at"), "sealed_manifest", "recorded")), "completed_at": choose("completed_at", (manifest.get("completed_at"), "sealed_manifest", "recorded")),
            "modified_at": choose("modified_at", (datetime.fromtimestamp(root.stat().st_mtime, timezone.utc).isoformat() if root.exists() else None, "filesystem_metadata", "inferred")),
            "artifact_size_bytes": choose("artifact_size_bytes", (artifact_size, "filesystem_metadata", "inferred")), "model_size_bytes": choose("model_size_bytes", (model_size, "filesystem_metadata", "inferred")),
            "epochs": choose("epochs", (training.get("epochs"), "resolved_config", "recorded"), (epochs_from_history, "training_history", "inferred")), "seed": choose("seed", ((config.get("run") or {}).get("seed") if isinstance(config.get("run"), dict) else None, "resolved_config", "recorded")),
            "parameter_count": choose("parameter_count", (summary_number("Total params"), "model_summary", "inferred")), "trainable_parameters": choose("trainable_parameters", (summary_number("Trainable params"), "model_summary", "inferred")),
            "input_shape": choose("input_shape", ((config.get("data") or {}).get("input_shape") if isinstance(config.get("data"), dict) else None, "resolved_config", "recorded"), (model.get("input", {}).get("shape") if isinstance(model.get("input"), dict) else None, "sealed_manifest", "recorded"), (contract.get("input_shape") if isinstance(contract, dict) else None, "model_contract", "inferred")),
            "num_classes": choose("num_classes", ((config.get("data") or {}).get("num_classes") if isinstance(config.get("data"), dict) else None, "resolved_config", "recorded"), (outputs.get("class_count"), "sealed_manifest", "recorded"), (contract.get("num_classes") if isinstance(contract, dict) else None, "model_contract", "inferred")),
            "embedding_dim": choose("embedding_dim", ((config.get("image_embedding") or {}).get("dimension") if isinstance(config.get("image_embedding"), dict) else model_config.get("embedding_dim"), "resolved_config", "recorded"), (outputs.get("embedding_dimension"), "sealed_manifest", "recorded"), (contract.get("embedding_dim") if isinstance(contract, dict) else None, "model_contract", "inferred")),
            "pooling_type": choose("pooling_type", (pooling.get("type") if isinstance(pooling, dict) else model_config.get("pooling"), "resolved_config", "recorded"), (contract.get("pooling") if isinstance(contract, dict) else None, "model_contract", "inferred")),
            "metadata_field_count": choose("metadata_field_count", (len(metadata.get("fields", metadata.get("features", []))) if metadata and isinstance(metadata.get("fields", metadata.get("features", [])), list) else None, "resolved_config", "recorded")),
            "posthoc_type": choose("posthoc_type", (posthoc.get("type") if isinstance(posthoc, dict) else None, "resolved_config", "recorded")),
            "architecture_version": choose("architecture_version", ((config.get("architecture") or {}).get("version") if isinstance(config.get("architecture"), dict) else None, "resolved_config", "recorded")),
            "architecture": architecture, "variant": variant, "task": task, "dataset_fingerprint": choose("dataset_fingerprint", (dataset.get("fingerprint_sha256"), "sealed_manifest", "recorded")),
            "metrics": metric_values, "config": config, "evaluation": evaluation, "runtime": runtime, "field_sources": sources, "field_availability": availability,
        }
        for name, value in metric_values.items():
            # Evaluation values retain their source; history-only values are explicitly inferred.
            choose(name, (manifest_evaluation.get(name), "sealed_manifest", "recorded"), (summary_evaluation.get(name) if isinstance(summary_evaluation, dict) else None, "evaluation_summary", "recorded"), (value, "training_history", "inferred"))
        facts.update(metric_values)
        for key in ("stem_size", "epochs", "seed", "parameter_count", "trainable_parameters", "num_classes", "embedding_dim", "metadata_field_count", "architecture_version"):
            if isinstance(facts[key], bool): facts[key] = None
        for key in ("macro_f1", "loss", "training_seconds", "artifact_size_bytes", "model_size_bytes"):
            if not isinstance(facts[key], (int, float)) or isinstance(facts[key], bool): facts[key] = None
        return facts

    def training_catalog_roots_info(self) -> list[dict[str, str]]:
        return [{"root_id": root_id, "path": str(path)} for root_id, path in self.training_catalog_roots.items()]

    def _register_frozen_catalog_entries(self, entries: list[dict[str, Any]]) -> dict[str, list[Any]]:
        """Mirror discovered frozen datasets into Queue's durable registry.

        Catalog entries are safe, allow-listed SQLite paths, while the
        ``datasets`` table is the durable identity Queue pins into a run.  The
        scan is therefore the single boundary that makes an already-frozen
        revision usable for training; working revisions remain catalog-only.
        """
        registered, already_registered, skipped = [], [], []
        for entry in entries:
            if entry.get("source_type") != "oracle_sqlite" or entry.get("status") != "frozen":
                continue
            info = entry.get("dataset_info") if isinstance(entry.get("dataset_info"), dict) else {}
            dataset_id = info.get("dataset_id")
            if not isinstance(dataset_id, str) or not dataset_id:
                skipped.append({"path": entry["path"], "reason": "Frozen catalog entry has no dataset identity"})
                continue
            try:
                existed = self.dataset(dataset_id) is not None
                self.ingest_dataset(entry["path"])
                (already_registered if existed else registered).append(dataset_id)
            except (OSError, ValueError, sqlite3.DatabaseError) as exc:
                skipped.append({"path": entry["path"], "reason": str(exc)})
        return {"registered": registered, "already_registered": already_registered, "skipped": skipped}

    def scan_training_catalog(self, root_id: str | None = None) -> dict[str, Any]:
        """Safely inspect an allow-listed source directory without importing it."""
        from oracle_builder.orchestration.training_catalog import scan_training_catalog
        if root_id is not None and root_id not in self.training_catalog_roots:
            raise KeyError(f"Unknown training catalog root: {root_id}")
        root_ids = [root_id] if root_id else list(self.training_catalog_roots)
        now = _now()
        reports: list[tuple[str, dict[str, Any]]] = []
        for selected_root_id in root_ids:
            reports.append((selected_root_id, scan_training_catalog(self.training_catalog_roots[selected_root_id])))
        with self._connection() as db:
            for selected_root_id, report in reports:
                catalog_ids = [entry["catalog_id"] for entry in report["entries"]]
                if catalog_ids:
                    db.execute(f"DELETE FROM training_catalog_entries WHERE root_id=? AND catalog_id NOT IN ({','.join('?' for _ in catalog_ids)})", [selected_root_id, *catalog_ids])
                else:
                    db.execute("DELETE FROM training_catalog_entries WHERE root_id=?", (selected_root_id,))
                for entry in report["entries"]:
                    db.execute("""INSERT INTO training_catalog_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(catalog_id) DO UPDATE SET root_id=excluded.root_id,name=excluded.name,path=excluded.path,source_type=excluded.source_type,fingerprint_sha256=excluded.fingerprint_sha256,metadata_json=excluded.metadata_json,scanned_at=excluded.scanned_at,updated_at=excluded.updated_at""",
                        (entry["catalog_id"], selected_root_id, entry["name"], entry["path"], entry["source_type"], entry.get("fingerprint_sha256"), _json(entry), report["scanned_at"], now))
        entries = [entry for _, report in reports for entry in report["entries"]]
        return {
            "root_id": root_id,
            "roots": [{"root_id": selected_root_id, "path": report["root"]} for selected_root_id, report in reports],
            "entries": entries,
            "registration": self._register_frozen_catalog_entries(entries),
            "scanned_at": now,
        }

    def training_catalog(self, *, root_id: str | None = None) -> list[dict[str, Any]]:
        query, params = "SELECT * FROM training_catalog_entries", []
        if root_id:
            query += " WHERE root_id=?"; params.append(root_id)
        query += " ORDER BY name COLLATE NOCASE"
        with self._connection() as db:
            return [self._training_catalog_view(_row(row)) for row in db.execute(query, params).fetchall()]  # type: ignore[list-item]

    def training_catalog_bundles(self, *, root_id: str | None = None) -> list[dict[str, Any]]:
        """Group compatible revisions without hiding their immutable identity."""
        bundles: dict[str, dict[str, Any]] = {}
        for entry in self.training_catalog(root_id=root_id):
            family_id = str(entry.get("family_id") or entry.get("training_set_family") or entry["catalog_id"])
            bundle = bundles.setdefault(family_id, {
                "family_id": family_id,
                "name": entry.get("training_set_family") or entry.get("name"),
                "task": entry.get("task"),
                "versions": [],
            })
            bundle["versions"].append(entry)
        for bundle in bundles.values():
            bundle["versions"].sort(key=lambda entry: (str(entry.get("modified_at") or ""), str(entry.get("training_set_version") or "")), reverse=True)
            versions = bundle["versions"]
            bundle["frozen_count"] = sum(entry.get("status") == "frozen" for entry in versions)
            bundle["unfrozen_count"] = len(versions) - bundle["frozen_count"]
            bundle["latest"] = versions[0]
        return sorted(bundles.values(), key=lambda bundle: str(bundle["name"]).casefold())

    def training_catalog_entry(self, catalog_id: str) -> dict[str, Any] | None:
        with self._connection() as db:
            return _row(db.execute("SELECT * FROM training_catalog_entries WHERE catalog_id=?", (catalog_id,)).fetchone())

    @staticmethod
    def _training_catalog_view(entry: dict[str, Any] | None) -> dict[str, Any]:
        """Expose catalog metadata as convenient read-only fields without losing its source map."""
        if entry is None:
            return {}
        result = dict(entry)
        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                result.setdefault(key, value)
        return result

    def _training_catalog_sqlite_path(self, entry: dict[str, Any]) -> Path:
        if entry.get("source_type") != "oracle_sqlite":
            raise KeyError(entry.get("catalog_id"))
        path = Path(entry["path"]).resolve()
        if not path.is_file() or not any(path.is_relative_to(root) for root in self.training_catalog_roots.values()):
            raise FileNotFoundError(path)
        return path

    def training_catalog_previews(self, catalog_id: str, *, label: str | None = None, offset: int = 0, limit: int = 24) -> dict[str, Any]:
        entry = self.training_catalog_entry(catalog_id)
        if entry is None: raise KeyError(catalog_id)
        path = self._training_catalog_sqlite_path(entry)
        offset, limit = max(0, offset), min(max(1, limit), 100)
        with sqlite3.connect(path) as db:
            db.row_factory = sqlite3.Row
            info = read_dataset_info(db)
            where, params = "", []
            if info["dataset_type"] == "classification":
                if label:
                    where, params = " WHERE l.name=?", [label]
                total = int(db.execute("""SELECT count(*) FROM dataset_items di
                    LEFT JOIN classification_annotations ca ON ca.item_id=di.item_id AND ca.is_current=1 AND ca.status='accepted'
                    LEFT JOIN classification_labels l ON l.label_id=ca.label_id""" + where, params).fetchone()[0])
                query = """SELECT di.item_id,di.source_key,l.name AS label FROM dataset_items di
                    JOIN classification_items ci ON ci.item_id=di.item_id
                    LEFT JOIN classification_annotations ca ON ca.item_id=di.item_id AND ca.is_current=1 AND ca.status='accepted'
                    LEFT JOIN classification_labels l ON l.label_id=ca.label_id""" + where + " ORDER BY di.item_id LIMIT ? OFFSET ?"
            else:
                total = int(db.execute("SELECT count(*) FROM dataset_items").fetchone()[0])
                query, params = """SELECT di.item_id,di.source_key,NULL AS label FROM dataset_items di
                    JOIN mask_refinement_items mi ON mi.item_id=di.item_id ORDER BY di.item_id LIMIT ? OFFSET ?""", []
            rows = [dict(row) for row in db.execute(query, [*params, limit, offset])]
        items = [{**row, "preview_url": f"/api/v1/training-catalog/{catalog_id}/previews/{row['item_id']}"} for row in rows]
        return {"catalog_id": catalog_id, "items": items, "total": total, "offset": offset, "limit": limit}

    def training_catalog_preview_image(self, catalog_id: str, item_id: str, *, max_size: int = 320) -> bytes:
        """Render a bounded JPEG preview for an allow-listed catalog image.

        The UI receives opaque item IDs rather than source paths, so this never
        turns the catalog endpoint into a general file reader.
        """
        entry = self.training_catalog_entry(catalog_id)
        if entry is None: raise KeyError(catalog_id)
        path = self._training_catalog_sqlite_path(entry)
        return self._sqlite_preview_image(path, item_id, max_size=max_size)

    def freeze_training_catalog_entry(self, catalog_id: str) -> dict[str, Any]:
        """Freeze one explicit SQLite revision, then register its immutable state."""
        entry = self.training_catalog_entry(catalog_id)
        if entry is None: raise KeyError(catalog_id)
        path = self._training_catalog_sqlite_path(entry)
        with sqlite3.connect(path) as db:
            info = read_dataset_info(db)
            if info["lifecycle"] not in {"working", "frozen"}:
                raise ValueError(f"Only working datasets can be frozen; this revision is {info['lifecycle']!r}")
            if info["lifecycle"] == "working":
                set_dataset_lifecycle(db, "frozen", actor="oracle-orchestrator", details={"catalog_id": catalog_id})
            db.commit()
        report = self.scan_training_catalog(entry["root_id"])
        refreshed = next((item for item in report["entries"] if item["catalog_id"] == catalog_id), None)
        if refreshed is None: raise RuntimeError("Frozen dataset was not returned by catalog reconciliation")
        dataset = self.ingest_dataset(path)
        return {"entry": refreshed, "dataset": dataset}

    def compare_training_catalog(self, catalog_ids: list[str]) -> dict[str, Any]:
        entries = [self.training_catalog_entry(value) for value in dict.fromkeys(catalog_ids)]
        if len(entries) < 2 or any(entry is None for entry in entries): raise KeyError("Select at least two catalog entries")
        values = [entry for entry in entries if entry]
        return {"entries": [self._training_catalog_view(entry) for entry in values], "comparison": {"item_counts": {entry["catalog_id"]: entry["metadata"].get("item_count") for entry in values}, "class_counts": {entry["catalog_id"]: entry["metadata"].get("class_count") for entry in values}, "fingerprints": {entry["catalog_id"]: entry.get("fingerprint_sha256") for entry in values}}}

    @staticmethod
    def _duration_seconds(started_at: str | None, completed_at: str | None) -> float | None:
        if not started_at or not completed_at:
            return None
        try:
            return round((datetime.fromisoformat(completed_at.replace("Z", "+00:00")) - datetime.fromisoformat(started_at.replace("Z", "+00:00"))).total_seconds(), 3)
        except ValueError:
            return None

    def _artifact_result(self, artifact: dict[str, Any]) -> dict[str, Any]:
        manifest = artifact["manifest"]
        root = Path(artifact["path"])
        summary = manifest.get("summary") or {}
        evaluation = summary.get("evaluation") if isinstance(summary.get("evaluation"), dict) else None
        if evaluation is None:
            summary_path = root / "evaluation" / "evaluation_summary.json"
            try:
                loaded = json.loads(summary_path.read_text(encoding="utf-8"))
                evaluation = loaded if isinstance(loaded, dict) else None
            except (OSError, json.JSONDecodeError):
                evaluation = None
        metrics = {
            key: float(value)
            for key, value in (evaluation or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        task = artifact.get("task") or (evaluation or {}).get("task")
        split, metric_schema, split_source = None, None, None
        for relative in ("evaluation/metrics_long.csv", "evaluation/segmentation_metrics.csv", "evaluation/sample_metrics.csv"):
            path = root / relative
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    row = next(csv.DictReader(handle), None)
                if row:
                    split = row.get("split") or None
                    metric_schema = row.get("schema_name") or None
                    split_source = relative
                    break
            except OSError:
                continue
        if evaluation and not split:
            # Oracle Builder training evaluates the held-out test split before
            # sealing. Older summaries did not repeat that field explicitly.
            split, split_source = "test", "standard_training_summary"
        primary_names = {
            "classification": ["accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall", "macro_average_precision"],
            "segmentation": ["mean_dice", "mean_iou", "mean_precision", "mean_recall", "mean_pixel_accuracy"],
        }.get(str(task), [])
        primary = {name: metrics[name] for name in primary_names if name in metrics}
        return {
            "artifact_id": artifact["artifact_id"], "name": artifact["name"],
            "artifact_type": artifact["artifact_type"], "task": task,
            "architecture": artifact.get("architecture"), "variant": artifact.get("variant"),
            "status": artifact.get("status"), "lifecycle": artifact.get("lifecycle"),
            "path": artifact["path"], "fingerprint_sha256": artifact.get("fingerprint_sha256"),
            "dataset_id": artifact.get("dataset_id"),
            "dataset_fingerprint_sha256": artifact.get("dataset_fingerprint_sha256"),
            "metrics": metrics, "primary_metrics": primary,
            "protocol": {
                "task": task, "dataset_fingerprint_sha256": artifact.get("dataset_fingerprint_sha256"),
                "split": split, "split_source": split_source, "metric_schema": metric_schema,
                "decision_rule": (evaluation or {}).get("decision_rule"),
                "segmentation_target": (evaluation or {}).get("segmentation_target"),
            },
        }

    @staticmethod
    def _evidence_csv(path: Path, *, limit: int = 200) -> list[dict[str, Any]]:
        """Read a bounded, display-oriented view of a standard evidence table."""
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))[:limit]
        except OSError:
            return []
        converted: list[dict[str, Any]] = []
        for row in rows:
            clean: dict[str, Any] = {}
            for key, value in row.items():
                if value in {None, ""}:
                    clean[key] = None
                    continue
                try:
                    clean[key] = float(value)
                except ValueError:
                    clean[key] = value
            converted.append(clean)
        return converted

    def artifact_evidence(self, artifact_id: str) -> dict[str, Any]:
        """Return bounded detail from files already sealed into an artifact.

        This endpoint never evaluates a model or derives new scientific results.
        It only makes Oracle Builder's standard evidence inspectable by clients.
        """
        artifact = self.artifact(artifact_id)
        if artifact is None:
            raise KeyError(artifact_id)
        root = Path(artifact["path"])
        result = self._artifact_result(artifact)
        confusion: dict[str, Any] | None = None
        try:
            loaded = json.loads((root / "evaluation" / "confusion_matrix.json").read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("matrix"), list):
                confusion = loaded
        except (OSError, json.JSONDecodeError):
            pass

        per_class = self._evidence_csv(root / "evaluation" / "per_class_metrics.csv")
        top_confusions = self._evidence_csv(root / "evaluation" / "top_confusions.csv", limit=50)
        samples = self._evidence_csv(root / "evaluation" / "sample_metrics.csv", limit=500)
        if result.get("task") == "segmentation":
            samples.sort(key=lambda row: float(row.get("dice") or 0.0))

        media: list[dict[str, Any]] = []
        supported = {".png", ".jpg", ".jpeg", ".webp"}
        for directory in (root / "figures", root / "evaluation", root / "activations", root / "overlays"):
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in supported:
                    continue
                relative = path.relative_to(root).as_posix()
                label = path.stem.replace("_", " ").replace("-", " ").strip().title()
                lowered = relative.lower()
                kind = "activation" if "activat" in lowered or "saliency" in lowered or "gradcam" in lowered else (
                    "overlay" if "overlay" in lowered or "mask" in lowered or "prediction" in lowered else "figure"
                )
                media.append({
                    "name": label, "kind": kind, "path": relative,
                    "url": f"/api/v1/artifacts/{artifact_id}/evidence/files/{relative}",
                })

        return {
            "artifact": result,
            "classification": {
                "confusion_matrix": confusion,
                "per_class_metrics": per_class,
                "top_confusions": top_confusions,
            } if result.get("task") == "classification" else None,
            "segmentation": {
                "sample_metrics": samples,
                "worst_samples": samples[:25],
            } if result.get("task") == "segmentation" else None,
            "media": media,
            "availability": {
                "confusion_matrix": confusion is not None,
                "per_class_metrics": bool(per_class),
                "sample_metrics": bool(samples),
                "overlays": any(item["kind"] == "overlay" for item in media),
                "activations": any(item["kind"] == "activation" for item in media),
            },
        }

    def artifact_evidence_file(self, artifact_id: str, relative_path: str) -> Path:
        artifact = self.artifact(artifact_id)
        if artifact is None:
            raise KeyError(artifact_id)
        root = Path(artifact["path"]).resolve()
        requested = Path(relative_path)
        if requested.is_absolute():
            raise ValueError("Evidence paths must be relative")
        path = (root / requested).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Evidence path escapes the artifact") from exc
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _json_file(path: Path) -> dict[str, Any] | list[Any] | None:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            return loaded if isinstance(loaded, (dict, list)) else None
        except (OSError, json.JSONDecodeError):
            return None

    def dataset_detail(self, dataset_id: str) -> dict[str, Any]:
        """Summarize a registered dataset without returning item payloads."""
        dataset = self.dataset(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        path = Path(dataset["path"])
        with sqlite3.connect(path) as db:
            db.row_factory = sqlite3.Row
            info = read_dataset_info(db)
            total = int(db.execute("SELECT count(*) FROM dataset_items").fetchone()[0])
            asset_count = int(db.execute("SELECT count(*) FROM assets").fetchone()[0])
            if info["dataset_type"] == "classification":
                labels = [dict(row) for row in db.execute("""SELECT l.label_id,l.class_index,l.name,count(ca.annotation_id) AS item_count
                    FROM classification_labels l LEFT JOIN classification_annotations ca
                    ON ca.label_id=l.label_id AND ca.is_current=1 AND ca.status='accepted'
                    GROUP BY l.label_id ORDER BY l.class_index""")]
            else:
                labels = []
            annotations_table = "classification_annotations" if info["dataset_type"] == "classification" else "mask_annotations"
            annotated = int(db.execute(f"SELECT count(*) FROM {annotations_table} WHERE is_current=1 AND status='accepted'").fetchone()[0])
        return {
            "dataset": dataset,
            "info": info,
            "counts": {"items": total, "assets": asset_count, "current_annotations": annotated},
            "labels": labels,
            "preview": {"available": total > 0, "max_limit": 100, "max_size": 1024},
        }

    def dataset_previews(self, dataset_id: str, *, offset: int = 0, limit: int = 24) -> dict[str, Any]:
        """List page-sized preview metadata; image bytes remain behind item URLs."""
        dataset = self.dataset(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        offset, limit = max(0, offset), min(max(1, limit), 100)
        with sqlite3.connect(dataset["path"]) as db:
            db.row_factory = sqlite3.Row
            info = read_dataset_info(db)
            total = int(db.execute("SELECT count(*) FROM dataset_items").fetchone()[0])
            if info["dataset_type"] == "classification":
                query = """SELECT di.item_id,di.source_key,di.metadata_json,a.shape_json,l.name AS label
                    FROM dataset_items di JOIN classification_items ci ON ci.item_id=di.item_id
                    JOIN assets a ON a.asset_id=ci.image_asset_id
                    LEFT JOIN classification_annotations ca ON ca.item_id=di.item_id AND ca.is_current=1 AND ca.status='accepted'
                    LEFT JOIN classification_labels l ON l.label_id=ca.label_id
                    ORDER BY di.item_id LIMIT ? OFFSET ?"""
            else:
                query = """SELECT di.item_id,di.source_key,di.metadata_json,a.shape_json,NULL AS label
                    FROM dataset_items di JOIN mask_refinement_items mi ON mi.item_id=di.item_id
                    JOIN assets a ON a.asset_id=mi.image_asset_id ORDER BY di.item_id LIMIT ? OFFSET ?"""
            items = []
            for row in db.execute(query, (limit, offset)):
                item = dict(row)
                shape = json.loads(item.pop("shape_json") or "null")
                item["shape"] = shape
                item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
                item["image_url"] = f"/api/v1/datasets/{dataset_id}/previews/{item['item_id']}"
                if info["dataset_type"] == "mask_refinement":
                    item["mask_url"] = f"/api/v1/datasets/{dataset_id}/previews/{item['item_id']}?kind=mask"
                    item["candidate_mask_url"] = f"/api/v1/datasets/{dataset_id}/previews/{item['item_id']}?kind=candidate_mask"
                items.append(item)
        return {"dataset_id": dataset_id, "offset": offset, "limit": limit, "total": total, "items": items}

    @staticmethod
    def _sqlite_preview_image(path: Path, item_id: str, *, kind: str = "image", max_size: int = 320) -> bytes:
        """Decode an item from an allow-listed Oracle SQLite revision as JPEG."""
        if kind not in {"image", "mask", "candidate_mask"}:
            raise ValueError("Preview kind must be image, mask, or candidate_mask")
        max_size = min(max(32, max_size), 1024)
        with sqlite3.connect(path) as db:
            db.row_factory = sqlite3.Row
            info = read_dataset_info(db)
            if info["dataset_type"] == "classification" and kind != "image":
                raise FileNotFoundError(item_id)
            if info["dataset_type"] == "classification":
                query = """SELECT a.payload,a.encoding,a.shape_json,a.external_uri FROM classification_items ci
                    JOIN assets a ON a.asset_id=ci.image_asset_id WHERE ci.item_id=?"""
            elif kind == "image":
                query = """SELECT a.payload,a.encoding,a.shape_json,a.external_uri FROM mask_refinement_items mi
                    JOIN assets a ON a.asset_id=mi.image_asset_id WHERE mi.item_id=?"""
            elif kind == "candidate_mask":
                query = """SELECT a.payload,a.encoding,a.shape_json,a.external_uri FROM mask_refinement_items mi
                    JOIN assets a ON a.asset_id=mi.candidate_mask_asset_id WHERE mi.item_id=?"""
            else:
                query = """SELECT a.payload,a.encoding,a.shape_json,a.external_uri FROM mask_annotations ma
                    JOIN assets a ON a.asset_id=ma.mask_asset_id WHERE ma.item_id=? AND ma.is_current=1 AND ma.status='accepted'"""
            row = db.execute(query, (item_id,)).fetchone()
        if row is None or row["payload"] is None:
            # External asset URIs are intentionally not fetched by the API.
            raise FileNotFoundError(item_id)
        array = np.asarray(decode_blob(row["payload"], row["encoding"], row["shape_json"]))
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim not in {2, 3}:
            raise ValueError("Dataset asset is not an image array")
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        if array.dtype.kind == "f":
            low, high = float(array.min(initial=0)), float(array.max(initial=1))
            array = ((array - low) / (high - low) * 255 if high > low else array * 255).clip(0, 255).astype("uint8")
        else:
            array = np.clip(array, 0, 255).astype("uint8")
        image = Image.fromarray(array)
        if image.mode not in {"L", "RGB"}:
            image = image.convert("RGB")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=85, optimize=True)
        return output.getvalue()

    def dataset_preview_image(self, dataset_id: str, item_id: str, *, kind: str = "image", max_size: int = 320) -> bytes:
        """Decode one registered local dataset asset and emit a bounded JPEG preview."""
        dataset = self.dataset(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        return self._sqlite_preview_image(Path(dataset["path"]), item_id, kind=kind, max_size=max_size)

    def artifact_history(self, artifact_id: str, *, limit: int = 500) -> dict[str, Any]:
        artifact = self.artifact(artifact_id)
        if artifact is None:
            raise KeyError(artifact_id)
        limit = min(max(1, limit), 2000)
        root = Path(artifact["path"])
        rows = self._evidence_csv(root / "metrics" / "history.csv", limit=limit)
        columns = list(rows[0]) if rows else []
        return {"artifact_id": artifact_id, "rows": rows, "columns": columns, "limit": limit,
                "truncated": len(rows) == limit and (root / "metrics" / "history.csv").is_file()}

    def artifact_detail(self, artifact_id: str) -> dict[str, Any]:
        """Read-only inspector payload for a sealed artifact; all large media stays separate."""
        artifact = self.artifact(artifact_id)
        if artifact is None:
            raise KeyError(artifact_id)
        root = Path(artifact["path"])
        model_summary = None
        try:
            model_summary = (root / "model" / "model_summary.txt").read_text(encoding="utf-8")[:100_000]
        except OSError:
            pass
        runtime = self._json_file(root / "provenance" / "runtime.json")
        config = self._json_file(root / "config" / "resolved.json")
        contract = self._json_file(root / "model" / "contract.json")
        return {"artifact": self._artifact_result(artifact), "facts": self._artifact_facts(artifact_id), "manifest": artifact["manifest"],
                "runtime": runtime, "config": config, "model_contract": contract,
                "architecture": {"summary": model_summary, "available": model_summary is not None},
                "history_url": f"/api/v1/artifacts/{artifact_id}/history",
                "evidence_url": f"/api/v1/artifacts/{artifact_id}/evidence"}

    @staticmethod
    def _comparison_check(results: list[dict[str, Any]]) -> dict[str, Any]:
        reasons: list[str] = []
        if len(results) < 2:
            reasons.append("Select at least two indexed artifacts")
        for field, label in (("task", "task"), ("dataset_fingerprint_sha256", "dataset revision"), ("split", "evaluation split")):
            values = {result["protocol"].get(field) for result in results}
            if None in values:
                reasons.append(f"One or more artifacts do not record the {label}")
            elif len(values) > 1:
                reasons.append(f"Artifacts use different {label} values")
        for field, label in (("metric_schema", "metric schema"), ("decision_rule", "decision rule"), ("segmentation_target", "segmentation target")):
            values = {result["protocol"].get(field) for result in results if result["protocol"].get(field) is not None}
            if len(values) > 1:
                reasons.append(f"Artifacts use different {label} values")
        if any(not result.get("metrics") for result in results):
            reasons.append("One or more artifacts have no standard evaluation metrics")
        common_metrics = sorted(set.intersection(*(set(result["metrics"]) for result in results))) if results else []
        if results and not common_metrics:
            reasons.append("Artifacts have no evaluation metrics in common")
        return {
            "compatible": not reasons, "reasons": reasons, "common_metrics": common_metrics,
            "protocol": results[0]["protocol"] if results and not reasons else None,
        }

    def experiment_results(self, experiment_id: str) -> dict[str, Any]:
        experiment = self.experiment(experiment_id)
        if experiment is None:
            raise KeyError(experiment_id)
        specifications = self.specifications(experiment_id)
        candidates: list[dict[str, Any]] = []
        with self._connection() as db:
            for specification in specifications:
                job_row = db.execute("SELECT * FROM jobs WHERE specification_id=? ORDER BY submitted_at DESC LIMIT 1", (specification["specification_id"],)).fetchone()
                job = _row(job_row)
                artifact = self.artifact(specification["artifact_id"]) if specification.get("artifact_id") else None
                result = self._artifact_result(artifact) if artifact else None
                recipe_id = specification["parameters"].get("recipe_id")
                recipe = self.recipe(recipe_id) if recipe_id else None
                candidates.append({
                    "specification_id": specification["specification_id"], "name": specification["name"],
                    "ordinal": specification["ordinal"], "action": specification["action"],
                    "status": specification["status"], "seed": specification["parameters"].get("seed"),
                    "recipe_id": recipe_id, "recipe_name": recipe.get("name") if recipe else None,
                    "resources": specification["resources"], "job": job,
                    "runtime_seconds": self._duration_seconds(job.get("started_at") if job else None, job.get("completed_at") if job else None),
                    "artifact": result,
                })
        comparable = [candidate["artifact"] for candidate in candidates if candidate["artifact"]]
        return {
            "experiment": experiment, "dataset": self.dataset(experiment["dataset_id"]) if experiment.get("dataset_id") else None,
            "candidates": candidates, "comparison": self._comparison_check(comparable),
            "summary": {
                "total": len(candidates), "planned": sum(candidate["status"] == "planned" for candidate in candidates),
                "active": sum(candidate["status"] in {"dispatched", "queued", "running", "paused", "validating"} for candidate in candidates),
                "indexed": sum(candidate["artifact"] is not None for candidate in candidates),
                "failed": sum(candidate["status"] in {"failed", "dispatch_failed", "artifact_invalid", "cancelled"} for candidate in candidates),
            },
        }

    def create_comparison(self, *, name: str, artifact_ids: list[str], description: str = "") -> dict[str, Any]:
        """Create a legacy comparison.

        The historical endpoint remains strict so older API consumers retain
        their reproducibility guardrail. New UI work should use comparison
        groups, which intentionally allow a user to relate unlike runs.
        """
        unique_ids = list(dict.fromkeys(artifact_ids))
        artifacts = [self.artifact(artifact_id) for artifact_id in unique_ids]
        if any(artifact is None for artifact in artifacts):
            raise KeyError("One or more artifacts were not found")
        results = [self._artifact_result(artifact) for artifact in artifacts if artifact is not None]
        check = self._comparison_check(results)
        if not check["compatible"]:
            raise ValueError("Artifacts are not comparable: " + "; ".join(check["reasons"]))
        comparison_id, now = str(uuid.uuid4()), _now()
        selection = {"artifact_ids": unique_ids, "artifacts": results}
        protocol = {**check["protocol"], "common_metrics": check["common_metrics"]}
        with self._connection() as db:
            db.execute("INSERT INTO comparisons VALUES (?, ?, ?, ?, ?, ?, ?)", (comparison_id, name.strip() or "Model comparison", description, _json(selection), _json(protocol), now, now))
        return self.comparison(comparison_id)  # type: ignore[return-value]

    def _artifact_runtime(self, artifact_id: str) -> dict[str, float | None]:
        """Return recorded queue and training duration when this artifact has a job."""
        with self._connection() as db:
            row = db.execute("""SELECT jobs.submitted_at, jobs.started_at, jobs.completed_at
                FROM jobs JOIN run_specifications USING (specification_id)
                WHERE run_specifications.artifact_id=?
                ORDER BY jobs.submitted_at DESC LIMIT 1""", (artifact_id,)).fetchone()
        if row is None:
            return {"queue_seconds": None, "runtime_seconds": None}
        return {
            "queue_seconds": self._duration_seconds(row["submitted_at"], row["started_at"]),
            "runtime_seconds": self._duration_seconds(row["started_at"], row["completed_at"]),
        }

    def artifact_catalog(self) -> list[dict[str, Any]]:
        """A compact, display-ready global run/artifact catalog for selection UIs."""
        return self.artifact_catalog_query()["artifacts"]

    def _artifact_facts(self, artifact_id: str) -> dict[str, Any]:
        with self._connection() as db:
            row = db.execute("SELECT training_set,classifier_type,stem_size,macro_f1,loss,training_seconds,facts_json FROM artifact_facts WHERE artifact_id=?", (artifact_id,)).fetchone()
        if row is None:
            return {}
        raw = dict(row)
        raw["facts"] = json.loads(raw.pop("facts_json"))
        return raw

    _CATALOG_COLUMNS = {
        "artifact_id": "artifacts.artifact_id", "name": "artifacts.name", "task": "artifacts.task",
        "architecture": "artifacts.architecture", "variant": "artifacts.variant", "status": "artifacts.status",
        "lifecycle": "artifacts.lifecycle", "dataset_id": "artifacts.dataset_id", "updated_at": "artifacts.updated_at",
        "discovered_at": "artifacts.discovered_at", "training_set": "artifact_facts.training_set",
        "classifier_type": "artifact_facts.classifier_type", "stem_size": "artifact_facts.stem_size",
        "macro_f1": "artifact_facts.macro_f1", "loss": "artifact_facts.loss", "training_seconds": "artifact_facts.training_seconds",
        "created_at": "json_extract(artifact_facts.facts_json, '$.created_at')",
        "completed_at": "json_extract(artifact_facts.facts_json, '$.completed_at')",
        "modified_at": "json_extract(artifact_facts.facts_json, '$.modified_at')",
        "artifact_size_bytes": "json_extract(artifact_facts.facts_json, '$.artifact_size_bytes')",
        "model_size_bytes": "json_extract(artifact_facts.facts_json, '$.model_size_bytes')",
        "epochs": "json_extract(artifact_facts.facts_json, '$.epochs')",
        "parameter_count": "json_extract(artifact_facts.facts_json, '$.parameter_count')",
        "trainable_parameters": "json_extract(artifact_facts.facts_json, '$.trainable_parameters')",
        "accuracy": "json_extract(artifact_facts.facts_json, '$.accuracy')",
        "balanced_accuracy": "json_extract(artifact_facts.facts_json, '$.balanced_accuracy')",
        "macro_precision": "json_extract(artifact_facts.facts_json, '$.macro_precision')",
        "macro_recall": "json_extract(artifact_facts.facts_json, '$.macro_recall')",
        "macro_average_precision": "json_extract(artifact_facts.facts_json, '$.macro_average_precision')",
        "num_classes": "json_extract(artifact_facts.facts_json, '$.num_classes')",
        "embedding_dim": "json_extract(artifact_facts.facts_json, '$.embedding_dim')",
        "pooling_type": "json_extract(artifact_facts.facts_json, '$.pooling_type')",
        "metadata_field_count": "json_extract(artifact_facts.facts_json, '$.metadata_field_count')",
        "posthoc_type": "json_extract(artifact_facts.facts_json, '$.posthoc_type')",
        "architecture_version": "json_extract(artifact_facts.facts_json, '$.architecture_version')",
    }

    def artifact_filter_schema(self) -> dict[str, Any]:
        fields = [
            {"key": key, "label": key.replace("_", " ").title(),
             "type": "number" if key in {"stem_size", "macro_f1", "loss", "training_seconds", "artifact_size_bytes", "model_size_bytes", "epochs", "parameter_count", "trainable_parameters", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_average_precision", "num_classes", "embedding_dim", "metadata_field_count", "architecture_version"} else "datetime" if key in {"updated_at", "discovered_at", "created_at", "completed_at", "modified_at"} else "text",
             "operators": ["eq", "in", "contains"] if key not in {"stem_size", "macro_f1", "loss", "training_seconds", "artifact_size_bytes", "model_size_bytes", "epochs", "parameter_count", "trainable_parameters", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_average_precision", "num_classes", "embedding_dim", "metadata_field_count", "architecture_version", "updated_at", "discovered_at", "created_at", "completed_at", "modified_at"} else ["eq", "gte", "lte"]}
            for key in self._CATALOG_COLUMNS
        ]
        fields.append({"key": "tag", "label": "Tag", "type": "text", "operators": ["eq", "in"]})
        return {"fields": fields, "default_columns": ["name", "status", "architecture", "variant", "classifier_type", "training_set", "macro_f1", "accuracy", "epochs", "artifact_size_bytes", "created_at", "tags"]}

    def artifact_catalog_query(self, *, filters: dict[str, Any] | None = None, sort: str = "updated_at", order: str = "desc", offset: int = 0, limit: int = 100) -> dict[str, Any]:
        """Server-side catalog paging with a deliberately small, typed filter language.

        A filter can be a scalar, a list (membership), or ``{"gte": value}``,
        ``{"lte": value}``, ``{"contains": text}``, and ``{"eq": value}``.
        """
        filters, params, clauses = filters or {}, [], []
        for key, expression in filters.items():
            if key == "search":
                clauses.append("(LOWER(artifacts.name) LIKE ? OR LOWER(COALESCE(artifacts.architecture, '')) LIKE ? OR LOWER(COALESCE(artifacts.variant, '')) LIKE ? OR LOWER(COALESCE(artifact_facts.training_set, '')) LIKE ?)")
                params.extend([f"%{str(expression).lower()}%"] * 4)
                continue
            if key == "tag":
                values = expression if isinstance(expression, list) else [expression]
                if not values: continue
                clauses.append("EXISTS (SELECT 1 FROM artifact_tag_assignments ata JOIN artifact_tags at ON at.tag_id=ata.tag_id WHERE ata.artifact_id=artifacts.artifact_id AND at.name IN (%s))" % ",".join("?" for _ in values))
                params.extend(str(item) for item in values)
                continue
            column = self._CATALOG_COLUMNS.get(key)
            if column is None:
                raise ValueError(f"Unsupported artifact catalog filter: {key}")
            if isinstance(expression, dict):
                for operator, value in expression.items():
                    if operator == "eq": clauses.append(f"{column}=?"); params.append(value)
                    elif operator == "gte": clauses.append(f"{column}>=?"); params.append(value)
                    elif operator == "lte": clauses.append(f"{column}<=?"); params.append(value)
                    elif operator == "contains": clauses.append(f"LOWER(CAST({column} AS TEXT)) LIKE ?"); params.append(f"%{str(value).lower()}%")
                    else: raise ValueError(f"Unsupported artifact catalog operator: {operator}")
            elif isinstance(expression, list):
                if expression: clauses.append(f"{column} IN ({','.join('?' for _ in expression)})"); params.extend(expression)
            else:
                clauses.append(f"{column}=?"); params.append(expression)
        sort_column = self._CATALOG_COLUMNS.get(sort)
        if sort_column is None: raise ValueError(f"Unsupported artifact catalog sort: {sort}")
        direction = "ASC" if order.lower() == "asc" else "DESC"
        offset, limit = max(0, int(offset)), min(max(1, int(limit)), 500)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        source = " FROM artifacts LEFT JOIN artifact_facts USING (artifact_id)"
        with self._connection() as db:
            total = int(db.execute("SELECT COUNT(*)" + source + where, params).fetchone()[0])
            rows = db.execute("SELECT artifacts.*,artifact_facts.training_set,artifact_facts.classifier_type,artifact_facts.stem_size,artifact_facts.macro_f1,artifact_facts.loss,artifact_facts.training_seconds,artifact_facts.facts_json" + source + where + f" ORDER BY {sort_column} {direction}, artifacts.artifact_id ASC LIMIT ? OFFSET ?", [*params, limit, offset]).fetchall()
        values = []
        for row in rows:
            artifact = _row(row)
            if artifact:
                facts = {**(artifact.get("facts") if isinstance(artifact.get("facts"), dict) else {}), **{key: artifact.get(key) for key in ("training_set", "classifier_type", "stem_size", "macro_f1", "loss", "training_seconds")}}
                values.append({**self._artifact_result(artifact), **facts, "timing": self._artifact_runtime(artifact["artifact_id"]), "tags": self.tags_for_artifact(artifact["artifact_id"])})
        return {"artifacts": values, "total": total, "offset": offset, "limit": limit, "sort": sort, "order": direction.lower()}

    def tags(self) -> list[dict[str, Any]]:
        return self._many("SELECT * FROM artifact_tags ORDER BY name COLLATE NOCASE")

    def create_tag(self, *, name: str, color: str | None = None) -> dict[str, Any]:
        normalized = name.strip()
        if not normalized or len(normalized) > 80: raise ValueError("Tag name must be between 1 and 80 characters")
        now, tag_id = _now(), str(uuid.uuid4())
        with self._connection() as db:
            db.execute("INSERT INTO artifact_tags VALUES (?, ?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET color=COALESCE(excluded.color, artifact_tags.color),updated_at=excluded.updated_at", (tag_id, normalized, color, now, now))
            row = db.execute("SELECT * FROM artifact_tags WHERE name=? COLLATE NOCASE", (normalized,)).fetchone()
        return _row(row)  # type: ignore[return-value]

    def tags_for_artifact(self, artifact_id: str) -> list[dict[str, Any]]:
        with self._connection() as db:
            return [_row(row) for row in db.execute("SELECT artifact_tags.* FROM artifact_tags JOIN artifact_tag_assignments USING (tag_id) WHERE artifact_id=? ORDER BY name COLLATE NOCASE", (artifact_id,)).fetchall()]  # type: ignore[list-item]

    def set_artifact_tags(self, artifact_id: str, tag_names: list[str]) -> list[dict[str, Any]]:
        if self.artifact(artifact_id) is None: raise KeyError(artifact_id)
        names = list(dict.fromkeys(str(name).strip() for name in tag_names if str(name).strip()))
        if len(names) > 30: raise ValueError("An artifact can have at most 30 tags")
        now = _now()
        with self._connection() as db:
            db.execute("DELETE FROM artifact_tag_assignments WHERE artifact_id=?", (artifact_id,))
            for name in names:
                if len(name) > 80: raise ValueError("Tag name must be between 1 and 80 characters")
                tag_id = str(uuid.uuid4())
                db.execute("INSERT INTO artifact_tags VALUES (?, ?, NULL, ?, ?) ON CONFLICT(name) DO UPDATE SET updated_at=excluded.updated_at", (tag_id, name, now, now))
                saved = db.execute("SELECT tag_id FROM artifact_tags WHERE name=? COLLATE NOCASE", (name,)).fetchone()[0]
                db.execute("INSERT INTO artifact_tag_assignments VALUES (?, ?, ?)", (artifact_id, saved, now))
        return self.tags_for_artifact(artifact_id)

    @staticmethod
    def architecture_view_from_config(config: dict[str, Any]) -> dict[str, Any]:
        """Stable UI graph, intentionally derived from config rather than Keras internals."""
        version = int((config.get("architecture") or {}).get("version", 1))
        model = config.get("model") or {}
        input_module = config.get("input") or config.get("data") or {}
        modules = [
            {"id": "input", "kind": "input", "label": "Input", "config": input_module},
            {"id": "preprocessing", "kind": "preprocessing", "label": "Geometry & channels", "config": config.get("preprocessing") or {}},
            {"id": "encoder", "kind": "encoder", "label": "CNN encoder", "config": config.get("encoder") or {"architecture": model.get("architecture") or (config.get("run") or {}).get("model"), "variant": model.get("variant")}},
            {"id": "stem", "kind": "stem", "label": "Stem", "config": config.get("stem") or {}},
            {"id": "pooling", "kind": "pooling", "label": "Pooling", "config": config.get("pooling") or {}},
            {"id": "image_embedding", "kind": "embedding", "label": "Image embedding", "config": config.get("image_embedding") or {}},
        ]
        metadata = config.get("metadata") or {}
        fusion = config.get("fusion") or {}
        if metadata.get("fields") or metadata.get("enabled") or metadata.get("features"):
            modules.extend([
                {"id": "metadata", "kind": "metadata", "label": "Metadata encoder", "config": metadata},
                {"id": "fusion", "kind": "fusion", "label": "Fusion", "config": fusion},
            ])
        modules.append({"id": "classifier", "kind": "classifier", "label": "Primary classifier", "config": config.get("classifier") or {}})
        posthoc = (config.get("posthoc") or config.get("post_hoc") or {})
        if posthoc.get("enabled") or posthoc.get("type"):
            modules.append({"id": "posthoc", "kind": "posthoc", "label": "Post-hoc predictor", "config": posthoc})
        return {"version": version, "modules": modules, "edges": [{"from": modules[index]["id"], "to": modules[index + 1]["id"]} for index in range(len(modules) - 1)],
                "representations": ["feature_map", "image_embedding", "metadata_embedding", "fused_embedding", "projection_embedding"] if version >= 2 else ["features"]}

    def artifact_architecture_view(self, artifact_id: str) -> dict[str, Any]:
        artifact = self.artifact(artifact_id)
        if artifact is None: raise KeyError(artifact_id)
        config = self._json_file(Path(artifact["path"]) / "config" / "resolved.json") or {}
        return {"artifact_id": artifact_id, "architecture": self.architecture_view_from_config(config), "config_available": bool(config)}

    @staticmethod
    def _draft_config(config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(config, dict): raise ValueError("Draft config must be an object")
        resolved = deep_merge(DEFAULT_CONFIG, config)
        # Drafts are authored against the composable contract unless explicitly
        # cloned from an archived V1 run.
        resolved.setdefault("architecture", {}).setdefault("version", 2)
        # A new visual draft is intentionally constructible before a dataset is
        # selected. These placeholders are replaced during training planning.
        resolved.setdefault("run", {}).setdefault("task", "classification")
        resolved["run"].setdefault("model", "resnet18")
        resolved.setdefault("data", {}).setdefault("input_shape", [128, 128])
        resolved["data"].setdefault("num_classes", 2)
        if not resolved.setdefault("training", {}).get("loss"):
            resolved["training"]["loss"] = "sparse_categorical_crossentropy"
        resolved.setdefault("preprocessing", {})["invert"] = bool(resolved["preprocessing"].get("invert") or False)
        validate_config(resolved)
        return resolved

    def create_model_draft(self, *, name: str, config: dict[str, Any] | None = None, description: str = "", layout: dict[str, Any] | None = None, source_artifact_id: str | None = None) -> dict[str, Any]:
        if source_artifact_id and self.artifact(source_artifact_id) is None: raise KeyError(source_artifact_id)
        source_config: dict[str, Any] = {}
        if source_artifact_id:
            source = self.artifact(source_artifact_id)
            source_config = self._json_file(Path(source["path"]) / "config" / "resolved.json") if source else {}
            source_config = source_config or {}
        resolved = self._draft_config(deep_merge(source_config, config or {}))
        now, draft_id = _now(), str(uuid.uuid4())
        safe_layout = layout if isinstance(layout, dict) else {}
        with self._connection() as db:
            db.execute("INSERT INTO model_drafts VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)", (draft_id, name.strip() or "Untitled model", description, source_artifact_id, _json(resolved), _json(safe_layout), now, now))
            db.execute("INSERT INTO model_draft_revisions VALUES (?, 1, ?, ?, ?)", (draft_id, _json(resolved), _json(safe_layout), now))
        return self.model_draft(draft_id)  # type: ignore[return-value]

    def model_drafts(self) -> list[dict[str, Any]]:
        return self._many("SELECT * FROM model_drafts ORDER BY updated_at DESC")

    def model_draft(self, draft_id: str) -> dict[str, Any] | None:
        with self._connection() as db:
            draft = _row(db.execute("SELECT * FROM model_drafts WHERE draft_id=?", (draft_id,)).fetchone())
            if draft is not None:
                draft["architecture"] = self.architecture_view_from_config(draft["config"])
            return draft

    def update_model_draft(self, draft_id: str, *, name: str | None = None, description: str | None = None, config: dict[str, Any] | None = None, layout: dict[str, Any] | None = None) -> dict[str, Any]:
        current = self.model_draft(draft_id)
        if current is None: raise KeyError(draft_id)
        if config is not None and not isinstance(config, dict): raise ValueError("Draft config must be an object")
        if layout is not None and not isinstance(layout, dict): raise ValueError("Draft layout must be an object")
        resolved = self._draft_config(config if config is not None else current["config"])
        next_layout = layout if layout is not None else current["layout"]
        revision, now = int(current["revision"]) + 1, _now()
        with self._connection() as db:
            db.execute("UPDATE model_drafts SET name=?,description=?,revision=?,config_json=?,layout_json=?,updated_at=? WHERE draft_id=?", (name.strip() if isinstance(name, str) and name.strip() else current["name"], description if description is not None else current["description"], revision, _json(resolved), _json(next_layout), now, draft_id))
            db.execute("INSERT INTO model_draft_revisions VALUES (?, ?, ?, ?, ?)", (draft_id, revision, _json(resolved), _json(next_layout), now))
        return self.model_draft(draft_id)  # type: ignore[return-value]

    def clone_model_draft(self, draft_id: str, *, name: str | None = None) -> dict[str, Any]:
        draft = self.model_draft(draft_id)
        if draft is None: raise KeyError(draft_id)
        return self.create_model_draft(name=name or f"{draft['name']} copy", description=draft["description"], config=draft["config"], layout=draft["layout"], source_artifact_id=draft.get("source_artifact_id"))

    def validate_model_draft(self, draft_id: str) -> dict[str, Any]:
        draft = self.model_draft(draft_id)
        if draft is None: raise KeyError(draft_id)
        try:
            config = self._draft_config(draft["config"])
            return {"valid": True, "errors": [], "warnings": [], "architecture": self.architecture_view_from_config(config)}
        except (TypeError, ValueError) as exc:
            return {"valid": False, "errors": [str(exc)], "warnings": []}

    def preview_model_draft(self, draft_id: str, *, dataset_id: str | None = None) -> dict[str, Any]:
        draft = self.model_draft(draft_id)
        if draft is None: raise KeyError(draft_id)
        config = self._draft_config(draft["config"])
        architecture = str((config.get("run") or {}).get("model") or (config.get("encoder") or {}).get("architecture") or "")
        result = {"draft_id": draft_id, "architecture": self.architecture_view_from_config(config), "config": config}
        if dataset_id:
            result["model_preview"] = self.model_preview(architecture=architecture, dataset_id=dataset_id, overrides=config)
        return result

    def plan_draft_training(self, draft_id: str, *, name: str, dataset_id: str, description: str = "", resources: dict[str, Any] | None = None, training_overrides: dict[str, Any] | None = None, initialization: dict[str, Any] | None = None) -> dict[str, Any]:
        """Seal a draft revision into one planned training specification.

        The draft itself remains editable; the generated TOML and initialization
        record are owned by the experiment, making later UI changes harmless.
        """
        if not name.strip(): raise ValueError("A training plan requires a name")
        draft, dataset = self.model_draft(draft_id), self.dataset(dataset_id)
        if draft is None: raise KeyError("Model draft was not found")
        if dataset is None: raise KeyError("Dataset was not found")
        if dataset["lifecycle"] != "frozen": raise ValueError("Training requires a frozen registered dataset")
        overrides, initialization = training_overrides or {}, initialization or {}
        if not isinstance(overrides, dict) or not isinstance(initialization, dict): raise ValueError("Training overrides and initialization must be objects")
        config = deep_merge(draft["config"], overrides)
        # The selected frozen dataset is authoritative for the classifier
        # output dimension. A draft can estimate capacity before data is
        # selected, but it cannot accidentally seal an incompatible class head.
        if str((config.get("run") or {}).get("task")) == "classification":
            try:
                with sqlite3.connect(dataset["path"]) as dataset_db:
                    class_count = int(dataset_db.execute("SELECT count(*) FROM classification_labels").fetchone()[0])
                if class_count > 0:
                    config.setdefault("data", {})["num_classes"] = class_count
            except sqlite3.DatabaseError:
                pass
        config = self._draft_config(config)
        source_id = initialization.get("source_artifact_id")
        if source_id:
            source = self.artifact(str(source_id))
            if source is None: raise KeyError("Initialization source artifact was not found")
            if source.get("task") and source.get("task") != (config.get("run") or {}).get("task"):
                raise ValueError("Initialization source task is incompatible with the draft task")
            source_config = self._json_file(Path(source["path"]) / "config" / "resolved.json") or {}
            source_shape = self._nested(source_config, "data", "input_shape")
            target_shape = self._nested(config, "data", "input_shape")
            if source_shape and target_shape and list(source_shape) != list(target_shape):
                raise ValueError("Initialization source input shape is incompatible with the draft")
        mode = str(initialization.get("mode", "scratch"))
        if mode not in {"scratch", "fine_tune", "transfer_encoder", "resume"}: raise ValueError("Unsupported initialization mode")
        if mode != "scratch" and not source_id: raise ValueError("A non-scratch initialization requires source_artifact_id")
        experiment_id, specification_id, now = str(uuid.uuid4()), str(uuid.uuid4()), _now()
        config_dir = self.artifact_root / "experiments" / experiment_id / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"draft-{draft_id[:8]}-revision-{draft['revision']}.toml"
        import tomli_w
        def toml_safe(value: Any) -> Any:
            if isinstance(value, dict): return {key: toml_safe(item) for key, item in value.items() if item is not None}
            if isinstance(value, list): return [toml_safe(item) for item in value if item is not None]
            return value
        config_path.write_text(tomli_w.dumps(toml_safe(config)), encoding="utf-8")
        parameters = self._assign_output_path("train", {"config": str(config_path), "input": dataset["path"], "dataset_id": dataset_id, "draft_id": draft_id, "draft_revision": draft["revision"], "initialization": initialization}, specification_id)
        digest = hashlib.sha256(_json({"parameters": parameters, "dataset_fingerprint": dataset["fingerprint_sha256"]}).encode()).hexdigest()
        plan = {"kind": "training", "dataset_id": dataset_id, "draft_id": draft_id, "draft_revision": draft["revision"], "resources": resources or {}, "initialization": initialization}
        with self._connection() as db:
            db.execute("INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (experiment_id, None, name.strip(), description, dataset_id, "expanded", _json(plan), now, now))
            db.execute("INSERT INTO run_specifications VALUES (?, ?, 1, ?, 'train', ?, ?, ?, 'planned', NULL, ?, ?)", (specification_id, experiment_id, name.strip(), _json(parameters), _json(resources or {}), digest, now, now))
        return self.experiment(experiment_id)  # type: ignore[return-value]

    def _comparison_group_detail(self, group: dict[str, Any], members: list[dict[str, Any]]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for member in members:
            artifact = self.artifact(member["artifact_id"])
            # Members should always resolve due to the foreign key. Keeping a
            # missing row visible makes imported/corrupt old databases diagnosable.
            if artifact is None:
                results.append({**member, "artifact": None, "timing": {"queue_seconds": None, "runtime_seconds": None}})
                continue
            results.append({**member, "artifact": self._artifact_result(artifact), "timing": self._artifact_runtime(member["artifact_id"])})

        artifact_results = [row["artifact"] for row in results if row["artifact"]]
        compatibility = self._comparison_check(artifact_results)
        baseline_id = group.get("baseline_artifact_id") or (results[0]["artifact_id"] if results else None)
        baseline = next((row for row in results if row["artifact_id"] == baseline_id), None)
        baseline_metrics = (baseline or {}).get("artifact", {}).get("metrics", {})
        all_metrics = sorted(set().union(*(set(row["artifact"]["metrics"]) for row in results if row["artifact"]))) if results else []
        matrix: list[dict[str, Any]] = []
        for row in results:
            artifact = row.get("artifact")
            metrics = artifact.get("metrics", {}) if artifact else {}
            values = {}
            for metric in all_metrics:
                value = metrics.get(metric)
                reference = baseline_metrics.get(metric)
                values[metric] = {
                    "value": value,
                    "delta": (value - reference) if isinstance(value, (int, float)) and isinstance(reference, (int, float)) else None,
                    "relative_delta": ((value - reference) / abs(reference)) if isinstance(value, (int, float)) and isinstance(reference, (int, float)) and reference != 0 else None,
                }
            matrix.append({"artifact_id": row["artifact_id"], "values": values,
                           "runtime_seconds": row["timing"]["runtime_seconds"],
                           "queue_seconds": row["timing"]["queue_seconds"]})
        return {
            **group, "members": results, "baseline_artifact_id": baseline_id,
            "compatibility": compatibility,
            "metrics": {"names": all_metrics, "baseline_artifact_id": baseline_id, "matrix": matrix},
        }

    def create_comparison_group(
        self, *, name: str, members: list[dict[str, Any]], description: str = "",
        relationship_label: str = "related", baseline_artifact_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist a manual relationship group without requiring protocol equality."""
        if not members:
            raise ValueError("Select at least one artifact")
        artifact_ids = [str(member.get("artifact_id", "")) for member in members]
        if not all(artifact_ids) or len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("Each comparison group member must reference a unique artifact")
        if any(self.artifact(artifact_id) is None for artifact_id in artifact_ids):
            raise KeyError("One or more artifacts were not found")
        if baseline_artifact_id is not None and baseline_artifact_id not in artifact_ids:
            raise ValueError("The baseline artifact must be a member of the comparison group")
        group_id, now = str(uuid.uuid4()), _now()
        with self._connection() as db:
            db.execute("INSERT INTO comparison_groups VALUES (?, ?, ?, ?, ?, ?, ?)", (
                group_id, name.strip() or "Model comparison", description, relationship_label.strip() or "related",
                baseline_artifact_id, now, now,
            ))
            for ordinal, member in enumerate(members):
                db.execute("INSERT INTO comparison_group_members VALUES (?, ?, ?, ?, ?)", (
                    group_id, artifact_ids[ordinal], ordinal, member.get("relationship_label"), str(member.get("note") or ""),
                ))
        return self.comparison_group(group_id)  # type: ignore[return-value]

    def _assign_output_path(self, action: str, parameters: dict[str, Any], specification_id: str) -> dict[str, Any]:
        """Assign paths owned by the orchestrator, never supplied by a UI client."""
        resolved = dict(parameters)
        if action == "train":
            resolved["runs_dir"] = str(self.runs_root)
            resolved["output"] = specification_id
        elif action == "evaluate":
            output = self.artifact_root / "evaluations" / specification_id
            output.parent.mkdir(parents=True, exist_ok=True)
            resolved["output"] = str(output)
        elif action == "model_ingest":
            output = self.artifact_root / "models" / specification_id
            output.parent.mkdir(parents=True, exist_ok=True)
            resolved["output"] = str(output)
        elif action == "run_pack":
            output = self.artifact_root / "packages" / f"{specification_id}.oracle-run.zip"
            output.parent.mkdir(parents=True, exist_ok=True)
            resolved["output"] = str(output)
        return resolved

    def dispatch(self, specification_id: str, endpoint_id: str) -> dict[str, Any]:
        check = self.preflight(specification_id, endpoint_id)
        if not check["ready"]:
            raise ValueError("Dispatch preflight failed: " + "; ".join(check["reasons"]))
        oracle_serve_url = check["endpoint"]["base_url"]
        with self._connection() as db:
            spec = self.specification(specification_id, connection=db)
            if spec is None:
                raise KeyError(specification_id)
            if spec["status"] != "planned":
                raise ValueError("Only planned run specifications can be dispatched")
            job_id, attempt_id, now = str(uuid.uuid4()), str(uuid.uuid4()), _now()
            dataset_id = spec["parameters"].get("dataset_id")
            dataset = self.dataset(str(dataset_id), connection=db) if dataset_id else None
            work_unit = build_work_unit(
                work_unit_id=job_id,
                attempt_id=attempt_id,
                specification_id=specification_id,
                action=spec["action"],
                parameters=spec["parameters"],
                resources=spec["resources"],
                dataset=dataset,
            )
            request = LocalPathWorkUnitAdapter.request_envelope(work_unit, spec["parameters"])
            db.execute("""INSERT INTO jobs (
                job_id, specification_id, oracle_serve_url, action, parameters_json,
                resources_json, work_unit_json, work_unit_sha256, status, remote_status, worker_id, error,
                submitted_at, updated_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                job_id, specification_id, oracle_serve_url.rstrip("/"), spec["action"],
                _json(spec["parameters"]), _json(spec["resources"]), work_unit.canonical_json(),
                work_unit.sha256, "dispatching", None, None, None, now, now, None,
            ))
        try:
            response = self._request(oracle_serve_url, "POST", "/compute/jobs", request)
        except Exception as exc:
            with self._connection() as db:
                db.execute("UPDATE jobs SET status='dispatch_failed', error=?, updated_at=? WHERE job_id=?", (str(exc), _now(), job_id))
            raise
        with self._connection() as db:
            db.execute("UPDATE jobs SET status='submitted', remote_status=?, worker_id=?, updated_at=? WHERE job_id=?", (response.get("status"), response.get("worker_id"), _now(), job_id))
            db.execute("UPDATE run_specifications SET status='dispatched', updated_at=? WHERE specification_id=?", (_now(), specification_id))
        return self.job(job_id)  # type: ignore[return-value]

    def specification_detail(self, specification_id: str) -> dict[str, Any]:
        specification = self.specification(specification_id)
        if specification is None:
            raise KeyError(specification_id)
        config_path = specification.get("parameters", {}).get("config")
        config = load_toml(config_path) if config_path else {}
        return {"specification": specification, "config": config}

    def update_planned_specification(self, specification_id: str, *, name: str | None = None, resources: dict[str, Any] | None = None, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Edit a saved run before it has been handed to compute."""
        with self._connection() as db:
            specification = self.specification(specification_id, connection=db)
            if specification is None:
                raise KeyError(specification_id)
            if specification["status"] != "planned":
                raise ValueError("Only runs that have not started can be edited. Create a new run to change a queued or completed execution.")
            next_name = specification["name"] if name is None else name.strip()
            if not next_name:
                raise ValueError("Run name cannot be empty")
            next_resources = dict(specification.get("resources") or {})
            if resources:
                if set(resources) - {"gpu_count"}:
                    raise ValueError("Only the GPU request can be changed after a run is saved")
                gpu_count = resources.get("gpu_count")
                if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 0:
                    raise ValueError("GPU request must be a non-negative integer")
                next_resources["gpu_count"] = gpu_count
            overrides = config_overrides or {}
            allowed_sections = {"run", "data", "model", "training", "augmentation", "callbacks", "output", "architecture", "input", "encoder", "stem", "normalization", "pooling", "image_embedding", "metadata", "fusion", "classifier", "preprocessing"}
            if set(overrides) - allowed_sections or any(not isinstance(value, dict) for value in overrides.values()):
                raise ValueError("Configuration changes must use known configuration sections")
            if overrides:
                config_path = specification.get("parameters", {}).get("config")
                if not config_path:
                    raise ValueError("This run has no editable training configuration")
                import tomli_w
                Path(config_path).write_text(tomli_w.dumps(deep_merge(load_toml(config_path), overrides)), encoding="utf-8")
            db.execute("UPDATE run_specifications SET name=?, resources_json=?, updated_at=? WHERE specification_id=?", (next_name, _json(next_resources), _now(), specification_id))
        return self.specification(specification_id)  # type: ignore[return-value]

    def reconcile_job(self, job_id: str) -> dict[str, Any]:
        local = self.job(job_id)
        if local is None:
            raise KeyError(job_id)
        if local["status"] in {"indexed", "artifact_invalid", "failed", "cancelled"}:
            return local
        remote = self._request(local["oracle_serve_url"], "GET", f"/compute/jobs/{job_id}")
        try:
            self._capture_job_events(local)
        except RuntimeError:
            # Job state is authoritative for reconciliation. Event collection is
            # best-effort and can recover during a later poll.
            pass
        remote_status, now = remote["status"], _now()
        result = remote.get("result") or {}
        output_path = result.get("output_path") or self._expected_output_path(local)
        local_status = "validating" if remote_status == "succeeded" and local["action"] in {"train", "model_ingest"} else remote_status
        with self._connection() as db:
            db.execute("""UPDATE jobs SET status=?, remote_status=?, worker_id=?, error=?,
                updated_at=?, completed_at=?, started_at=?, output_path=? WHERE job_id=?""",
                (local_status, remote_status, remote.get("worker_id"), remote.get("error"), now,
                 remote.get("finished_at"), remote.get("started_at"), output_path, job_id))
            if remote_status in {"failed", "cancelled"}:
                db.execute("UPDATE run_specifications SET status=?, updated_at=? WHERE specification_id=?", (remote_status, now, local["specification_id"]))
            elif local_status == "validating":
                db.execute("UPDATE run_specifications SET status='validating', updated_at=? WHERE specification_id=?", (now, local["specification_id"]))
            elif remote_status == "succeeded":
                db.execute("UPDATE run_specifications SET status='succeeded', updated_at=? WHERE specification_id=?", (now, local["specification_id"]))
        if local_status != local.get("status") or remote_status != local.get("remote_status"):
            self._record_event(
                job_id,
                "status_changed",
                f"Oracle Serve status changed to {remote_status}.",
                {
                    "source": "oracle-serve",
                    "previous_status": local.get("status"),
                    "previous_remote_status": local.get("remote_status"),
                    "status": local_status,
                    "remote_status": remote_status,
                    "worker_id": remote.get("worker_id"),
                },
            )
        if remote_status == "succeeded" and local["action"] in {"train", "model_ingest"}:
            self._record_event(job_id, "validating", "Compute completed; validating the produced artifact", {"output_path": output_path})
            try:
                report = self._index_completed_artifact({**local, "output_path": output_path})
            except Exception as exc:
                report = {"valid": False, "errors": [str(exc)], "warnings": [], "output_path": output_path}
            if report["valid"]:
                with self._connection() as db:
                    db.execute("UPDATE jobs SET status='indexed', validation_status='valid', validation_report_json=?, error=NULL, updated_at=? WHERE job_id=?", (_json(report), _now(), job_id))
                    db.execute("UPDATE run_specifications SET status='indexed', updated_at=? WHERE specification_id=?", (_now(), local["specification_id"]))
            else:
                error = "Completed output failed artifact validation: " + "; ".join(report.get("errors") or ["unknown validation error"])
                with self._connection() as db:
                    db.execute("UPDATE jobs SET status='artifact_invalid', validation_status='invalid', validation_report_json=?, error=?, updated_at=? WHERE job_id=?", (_json(report), error, _now(), job_id))
                    db.execute("UPDATE run_specifications SET status='artifact_invalid', updated_at=? WHERE specification_id=?", (_now(), local["specification_id"]))
                self._record_event(job_id, "artifact_invalid", error, {"validation": report})
        result_job = self.job(job_id)  # type: ignore[assignment]
        queued_run_id = result_job.get("queued_run_id") if result_job else None
        if queued_run_id:
            queue_statuses = {
                "dispatching": "starting", "submitted": "submitted", "queued": "submitted",
                "running": "running", "paused": "paused", "validating": "validating_artifact", "indexed": "complete",
                "succeeded": "complete", "failed": "failed", "cancelled": "cancelled",
                "dispatch_failed": "waiting_for_resources", "artifact_invalid": "failed",
            }
            with self._connection() as db:
                db.execute(
                    "UPDATE queued_runs SET status=?, failure_reason=?, updated_at=? WHERE queued_run_id=?",
                    (queue_statuses.get(str(result_job.get("status")), str(result_job.get("status"))), result_job.get("error"), _now(), queued_run_id),
                )
        return result_job  # type: ignore[return-value]

    def reconcile_active_jobs(self) -> list[dict[str, Any]]:
        """Retired push-job reconciliation shim.

        New jobs use durable worker leases and staged publication; they must
        never trigger a request to a historical ``oracle_serve_url``.  The
        private compatibility readers below remain only for offline inspection
        of an old control database, not as a runnable execution path.
        """
        return []

    def _record_remote_job_unavailable(self, job: dict[str, Any], error: RuntimeError) -> None:
        """Persist an endpoint-confirmed missing job once, without resetting it.

        A 404 proves this Serve process no longer owns the dispatch, but state
        reset remains an explicit user action.  Recording that fact is safe,
        makes the UI honest immediately, and retains the recovery decision.
        """
        message = str(error)
        if "returned 404" not in message or job.get("remote_status") == "missing":
            return
        reason = "Compute endpoint no longer has this job; it may have restarted or the process was interrupted."
        with self._connection() as db:
            db.execute(
                "UPDATE jobs SET remote_status='missing', error=?, updated_at=? WHERE job_id=?",
                (reason, _now(), job["job_id"]),
            )
        self._record_event(job["job_id"], "remote_missing", reason, {"source": "oracle-serve", "error": message})

    def reset_stuck_jobs(self, *, endpoint_id: str | None = None) -> dict[str, Any]:
        """Reset only jobs proven absent from their compute endpoint.

        A transient network failure is never treated as proof a job is gone.
        The original dispatch record is retained as ``stale`` for audit, and a
        linked validated queue row is returned to ``ready`` so the user must
        explicitly authorize its next dispatch attempt.
        """
        active_states = {"dispatching", "submitted", "queued", "running", "paused", "validating"}
        candidates = [job for job in self.jobs() if job.get("status") in active_states]
        if endpoint_id:
            endpoint = self.compute_endpoint(endpoint_id)
            if endpoint is None:
                raise KeyError("Compute endpoint was not found")
            candidates = [job for job in candidates if job.get("oracle_serve_url", "").rstrip("/") == endpoint["base_url"].rstrip("/")]
        reset: list[str] = []
        still_active: list[str] = []
        unavailable: list[dict[str, str]] = []
        for job in candidates:
            try:
                remote = self._request(job["oracle_serve_url"], "GET", f"/compute/jobs/{job['job_id']}")
            except RuntimeError as exc:
                message = str(exc)
                if "returned 404" not in message:
                    unavailable.append({"job_id": job["job_id"], "reason": message})
                    continue
                now = _now()
                reason = "Compute endpoint confirmed this dispatch no longer exists; reset for an explicit retry."
                with self._connection() as db:
                    db.execute("UPDATE jobs SET status='stale', remote_status='missing', error=?, completed_at=?, updated_at=? WHERE job_id=?", (reason, now, now, job["job_id"]))
                    db.execute("UPDATE run_specifications SET status='planned', updated_at=? WHERE specification_id=?", (now, job["specification_id"]))
                    queue = db.execute("SELECT queued_run_id,preflight_status FROM queued_runs WHERE specification_id=?", (job["specification_id"],)).fetchone()
                    if queue is not None:
                        next_status = "ready" if queue["preflight_status"] == "valid" else "needs_attention"
                        db.execute("UPDATE queued_runs SET status=?, start_authorized=0, failure_reason=?, updated_at=? WHERE queued_run_id=?", (next_status, reason, now, queue["queued_run_id"]))
                self._record_event(job["job_id"], "stale_reset", reason, {"remote_status": "missing"})
                reset.append(job["job_id"])
                continue
            if remote.get("status") in active_states:
                still_active.append(job["job_id"])
            else:
                # A completed remote record follows normal reconciliation;
                # this endpoint is only a recovery tool for missing records.
                self.reconcile_job(job["job_id"])
        return {"reset": reset, "still_active": still_active, "unavailable": unavailable, "endpoint_id": endpoint_id}

    def _capture_job_events(self, job: dict[str, Any]) -> None:
        with self._connection() as db:
            cursor = db.execute("SELECT remote_sequence FROM job_remote_event_cursors WHERE job_id=?", (job["job_id"],)).fetchone()
            remote_after = int(cursor["remote_sequence"]) if cursor is not None else 0
        payload = self._request(job["oracle_serve_url"], "GET", f"/compute/jobs/{job['job_id']}/events?after={remote_after}")
        with self._connection() as db:
            remote_cursor = remote_after
            for event in payload.get("events", []):
                remote_sequence = event.get("sequence")
                if not isinstance(remote_sequence, int) or remote_sequence <= remote_cursor:
                    continue
                sequence = db.execute("SELECT COALESCE(MAX(sequence), 0) + 1 FROM job_events WHERE job_id=?", (job["job_id"],)).fetchone()[0]
                data = {**(event.get("data") or {}), "source": "oracle-serve", "remote_sequence": remote_sequence}
                db.execute("INSERT INTO job_events VALUES (?, ?, ?, ?, ?, ?)", (job["job_id"], sequence, event.get("timestamp") or _now(), event.get("type") or "log", event.get("message") or "", _json(data)))
                remote_cursor = remote_sequence
            db.execute(
                """INSERT INTO job_remote_event_cursors(job_id,remote_sequence,updated_at) VALUES(?,?,?)
                   ON CONFLICT(job_id) DO UPDATE SET remote_sequence=excluded.remote_sequence, updated_at=excluded.updated_at""",
                (job["job_id"], remote_cursor, _now()),
            )

    @staticmethod
    def _expected_output_path(job: dict[str, Any]) -> str | None:
        params = job["parameters"]
        output = params.get("output")
        if not isinstance(output, str):
            return None
        return str(Path(params["runs_dir"]) / output) if job["action"] == "train" and isinstance(params.get("runs_dir"), str) else output

    def _index_completed_artifact(self, job: dict[str, Any]) -> dict[str, Any]:
        path_value = job.get("output_path") or self._expected_output_path(job)
        if not path_value:
            raise RuntimeError("Compute service did not report an artifact output path")
        path = Path(path_value)
        from oracle_builder.artifacts import validate_run_artifact
        report = {**validate_run_artifact(path), "output_path": str(path)}
        try:
            artifact_type = json.loads((path / "artifact.json").read_text(encoding="utf-8")).get("artifact_type", "model_run")
        except (OSError, json.JSONDecodeError) as exc:
            report["valid"] = False
            report.setdefault("errors", []).append(f"Artifact manifest could not be read: {exc}")
            artifact_type = None
        expected_type = "model_run" if job["action"] == "train" else "model_product"
        if report.get("valid") and artifact_type != expected_type:
            report["valid"] = False
            report.setdefault("errors", []).append(f"Expected {expected_type} output, received {artifact_type}")
        report["artifact_type"] = artifact_type
        if report.get("valid") and report.get("status") != "complete":
            report["valid"] = False
            report.setdefault("errors", []).append("Artifact status must be complete before indexing")
        if report.get("valid") and report.get("lifecycle") != "sealed":
            report["valid"] = False
            report.setdefault("errors", []).append("Artifact must be sealed before indexing")
        if not report.get("valid"):
            return report
        result = self.scan(path)
        artifact_id = report.get("artifact_id")
        if not artifact_id or artifact_id not in result["artifacts"]:
            report["valid"] = False
            report.setdefault("errors", []).append("Validated artifact could not be added to the catalog")
            return report
        with self._connection() as db:
            db.execute("UPDATE run_specifications SET artifact_id=?, updated_at=? WHERE specification_id=?", (artifact_id, _now(), job["specification_id"]))
        self._record_event(job["job_id"], "artifact_indexed", "Validated and indexed completed artifact", {"artifact_id": artifact_id, "path": str(path), "validation": report})
        return report

    def _record_event(self, job_id: str, event_type: str, message: str, data: dict[str, Any]) -> None:
        with self._connection() as db:
            sequence = db.execute("SELECT COALESCE(MAX(sequence), 0) + 1 FROM job_events WHERE job_id=?", (job_id,)).fetchone()[0]
            db.execute("INSERT INTO job_events VALUES (?, ?, ?, ?, ?, ?)", (job_id, sequence, _now(), event_type, message, _json(data)))

    @staticmethod
    def _request(
        base: str,
        method: str,
        endpoint: str,
        body: dict[str, Any] | None = None,
        *,
        timeout_seconds: float = 15,
    ) -> dict[str, Any]:
        request = urllib.request.Request(base.rstrip("/") + endpoint, method=method)
        if body is not None:
            request.data = _json(body).encode()
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"oracle-serve returned {exc.code}: {exc.read().decode()}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not communicate with oracle-serve: {exc}") from exc

    def datasets(self) -> list[dict[str, Any]]: return self._many("SELECT * FROM datasets ORDER BY name")
    def artifacts(self) -> list[dict[str, Any]]: return self._many("SELECT * FROM artifacts ORDER BY updated_at DESC")
    def experiments(self) -> list[dict[str, Any]]: return self._many("SELECT * FROM experiments ORDER BY created_at DESC")
    def recipes(self) -> list[dict[str, Any]]: return self._many("SELECT * FROM recipes ORDER BY name")
    def specifications(self, experiment_id: str | None = None) -> list[dict[str, Any]]:
        if experiment_id:
            with self._connection() as db:
                return [_row(item) for item in db.execute("SELECT * FROM run_specifications WHERE experiment_id=? ORDER BY ordinal", (experiment_id,)).fetchall()]  # type: ignore[list-item]
        return self._many("SELECT * FROM run_specifications ORDER BY created_at DESC")
    def jobs(self) -> list[dict[str, Any]]:
        return self._many("""SELECT jobs.*, run_specifications.artifact_id,
            artifacts.name AS artifact_name, artifacts.path AS artifact_path,
            artifacts.status AS artifact_status, artifacts.lifecycle AS artifact_lifecycle
            FROM jobs
            LEFT JOIN run_specifications USING (specification_id)
            LEFT JOIN artifacts USING (artifact_id)
            ORDER BY jobs.submitted_at DESC""")
    def job_events(self, job_id: str) -> list[dict[str, Any]]:
        with self._connection() as db:
            rows = db.execute("SELECT * FROM job_events WHERE job_id=? ORDER BY sequence", (job_id,)).fetchall()
        return [{**dict(row), "data": json.loads(row["data_json"])} for row in rows]

    def job_timing(self, job_id: str) -> dict[str, Any]:
        """Return a compact timing breakdown for a durable pull-worker job.

        Stage measurements are emitted by workers as ``oracle_timing_v1``
        events.  The small derived section keeps runs created before that
        instrumentation useful, without pretending its timestamps are a
        substitute for the worker's monotonic-clock measurements.
        """
        job = self.job(job_id)
        if job is None:
            raise KeyError(job_id)
        events = self.job_events(job_id)
        stages: list[dict[str, Any]] = []
        for event in events:
            data = event.get("data")
            if event.get("event_type") != "worker.stage_timing" or not isinstance(data, dict):
                continue
            duration = data.get("duration_seconds")
            stage = data.get("stage")
            if (isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0
                    or not isinstance(stage, str) or not stage):
                continue
            stages.append({
                "stage": stage, "duration_seconds": round(float(duration), 3),
                "timestamp": event.get("timestamp"), "outcome": data.get("outcome"),
                "bytes": data.get("bytes"), "details": {
                    key: value for key, value in data.items()
                    if key not in {"schema", "stage", "duration_seconds", "outcome", "bytes"}
                },
            })

        def timestamp(event_type: str) -> str | None:
            return next((str(item["timestamp"]) for item in events if item.get("event_type") == event_type and isinstance(item.get("timestamp"), str)), None)

        derived: list[dict[str, Any]] = []
        materializing, executing = timestamp("worker.materializing"), timestamp("worker.executing")
        materialization = self._duration_seconds(materializing, executing)
        if materialization is not None:
            derived.append({"stage": "artifact_materialization", "duration_seconds": materialization, "source": "event_timestamps"})
        submitted, started = job.get("submitted_at"), job.get("started_at")
        queue_seconds = self._duration_seconds(submitted, started)
        latest_progress = next((item.get("data") for item in reversed(events) if item.get("event_type") == "worker.training_progress" and isinstance(item.get("data"), dict)), None)
        reported_training_seconds = None
        if isinstance(latest_progress, dict):
            timing = latest_progress.get("timing")
            value = timing.get("elapsed_seconds") if isinstance(timing, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                reported_training_seconds = round(float(value), 3)
        runtime_end = job.get("completed_at") or _now()
        runtime_seconds = self._duration_seconds(started, runtime_end)
        return {
            "job_id": job_id, "status": job.get("status"),
            "summary": {
                "queue_seconds": queue_seconds,
                "runtime_seconds": runtime_seconds,
                "training_elapsed_seconds": reported_training_seconds,
            },
            "stages": stages,
            "derived_stages": derived,
        }

    def _last_persisted_training_snapshot(self, job: dict[str, Any]) -> dict[str, Any] | None:
        """Return safe, last-known progress when a local worker has forgotten a job.

        The live worker owns process state, but a training run records a
        status document in its sealed output.  It is valuable recovery context
        after a worker restart; keep the response deliberately path-free and
        bounded to fields useful in the UI.
        """
        output = job.get("output_path") or self._expected_output_path(job)
        if not isinstance(output, str):
            return None
        run_path = Path(output).expanduser().resolve()
        if not run_path.is_relative_to(self.runs_root):
            return None
        try:
            snapshot = json.loads((run_path / "training-status.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(snapshot, dict):
            return None
        allowed = {
            "state", "phase", "epoch", "total_epochs", "batch", "total_batches", "updated_at",
            "message", "progress", "timing", "metrics", "latest_metrics", "history", "learning_rate",
            "batches_per_second", "samples_per_second", "total_eta_seconds",
        }
        return {key: value for key, value in snapshot.items() if key in allowed}

    def job_training_status(self, job_id: str) -> dict[str, Any]:
        """Return the worker's live training projection without exposing paths.

        A live snapshot is intentionally best-effort: a job may be queued, run
        on an older worker, or briefly lose contact with its compute service.
        The web client can therefore render a useful, explicit unavailable
        state instead of treating a transient worker failure as a failed job.
        """
        job = self.job(job_id)
        if job is None:
            raise KeyError(job_id)

        fallback = {
            "job_id": job_id,
            "job_status": job["status"],
            "controls": (
                ["pause", "cancel"] if job["status"] == "running"
                else ["resume", "cancel"] if job["status"] == "paused"
                else ["cancel"] if job["status"] in {"dispatching", "submitted", "queued"}
                else []
            ),
            "available": False,
            # A process can be running normally while it creates its split,
            # imports TensorFlow, or compiles its first graph.  That is an
            # initializing state, not a stale/lost worker.  ``stale`` remains
            # reserved for the confirmed-404 recovery path below.
            "stale": False,
            "phase": "initializing" if job["status"] == "running" else (job.get("remote_status") or job["status"]),
            "message": "Training is starting; the first live metrics snapshot is not available yet.",
        }
        try:
            remote = self._request(job["oracle_serve_url"], "GET", f"/compute/jobs/{job_id}/training-status")
        except RuntimeError as exc:
            if "returned 404" in str(exc):
                snapshot = self._last_persisted_training_snapshot(job)
                return {
                    **fallback,
                    "stale": True,
                    "snapshot": snapshot,
                    "last_known": snapshot is not None,
                    "recovery_action": "reconcile_stuck_jobs",
                    "message": "The compute worker no longer has this job. It was interrupted or the worker restarted; use Reconcile stuck jobs before retrying.",
                }
            return {**fallback, "message": f"Live training status is temporarily unavailable: {exc}"}

        # A compute service controls the metric shape, but not the identity or
        # lifecycle information returned to the browser.  This prevents a
        # malformed/old worker response from making the local job ambiguous.
        if not isinstance(remote, dict):
            return {**fallback, "message": "Compute returned an invalid live training status response."}
        available = bool(remote.get("available", True))
        return {
            **remote,
            "job_id": job_id,
            "job_status": job["status"],
            "controls": fallback["controls"],
            "available": available,
            # Older/newer Serve instances may omit ``stale`` while they are
            # still starting their first snapshot.  Missing telemetry alone
            # is not evidence that a running worker was lost.
            "stale": bool(remote.get("stale", False)),
            "message": remote.get("message") or (None if available else fallback["message"]),
        }
    def comparisons(self) -> list[dict[str, Any]]: return self._many("SELECT * FROM comparisons ORDER BY created_at DESC")
    def comparison_groups(self) -> list[dict[str, Any]]:
        with self._connection() as db:
            groups = [_row(item) for item in db.execute("SELECT * FROM comparison_groups ORDER BY created_at DESC").fetchall()]
            counts = {row["comparison_group_id"]: row["member_count"] for row in db.execute(
                "SELECT comparison_group_id, COUNT(*) AS member_count FROM comparison_group_members GROUP BY comparison_group_id"
            ).fetchall()}
        return [{**group, "member_count": counts.get(group["comparison_group_id"], 0)} for group in groups if group]
    def _endpoint_with_scheduler_capacity(self, endpoint: dict[str, Any]) -> dict[str, Any]:
        """Expose the cached Serve resource snapshot in a UI-friendly shape."""
        queue = endpoint.get("queue") or {}
        resources = queue.get("resources") or {}
        workers = endpoint.get("workers") or []
        worker_slots = resources.get("worker_slots")
        if not isinstance(worker_slots, int):
            worker_slots = len(workers)
        free_slots = sum(worker.get("status") == "idle" for worker in workers)
        return {
            **endpoint,
            "resources": resources,
            "capacity": {
                "worker_slots": worker_slots,
                "free_slots": free_slots,
                "active_slots": max(0, worker_slots - free_slots),
                "cpu_capacity": resources.get("cpu_capacity"),
                "cpu_in_use": resources.get("cpu_in_use"),
                "gpu_leases": resources.get("gpu_leases") or [],
            },
            "scheduler_resources": resources,
        }

    def compute_endpoints(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        endpoints = self._many("SELECT * FROM compute_endpoints WHERE enabled=1 ORDER BY name")
        return [self.refresh_compute_endpoint(item["endpoint_id"]) for item in endpoints] if refresh else [self._endpoint_with_scheduler_capacity(item) for item in endpoints]
    def _many(self, query: str) -> list[dict[str, Any]]:
        with self._connection() as db: return [_row(item) for item in db.execute(query).fetchall()]  # type: ignore[list-item]
    def dataset(self, value: str, *, connection: sqlite3.Connection | None = None) -> dict[str, Any] | None:
        if connection is not None: return _row(connection.execute("SELECT * FROM datasets WHERE dataset_id=?", (value,)).fetchone())
        with self._connection() as db: return _row(db.execute("SELECT * FROM datasets WHERE dataset_id=?", (value,)).fetchone())
    def artifact(self, value: str) -> dict[str, Any] | None:
        with self._connection() as db: return _row(db.execute("SELECT * FROM artifacts WHERE artifact_id=?", (value,)).fetchone())
    def recipe(self, value: str) -> dict[str, Any] | None:
        with self._connection() as db: return _row(db.execute("SELECT * FROM recipes WHERE recipe_id=?", (value,)).fetchone())
    def experiment(self, value: str) -> dict[str, Any] | None:
        with self._connection() as db: return _row(db.execute("SELECT * FROM experiments WHERE experiment_id=?", (value,)).fetchone())
    def specification(self, value: str, *, connection: sqlite3.Connection | None = None) -> dict[str, Any] | None:
        if connection is not None: return _row(connection.execute("SELECT * FROM run_specifications WHERE specification_id=?", (value,)).fetchone())
        with self._connection() as db: return _row(db.execute("SELECT * FROM run_specifications WHERE specification_id=?", (value,)).fetchone())
    def job(self, value: str) -> dict[str, Any] | None:
        with self._connection() as db:
            return _row(db.execute("""SELECT jobs.*, run_specifications.artifact_id,
                artifacts.name AS artifact_name, artifacts.path AS artifact_path,
                artifacts.status AS artifact_status, artifacts.lifecycle AS artifact_lifecycle
                FROM jobs
                LEFT JOIN run_specifications USING (specification_id)
                LEFT JOIN artifacts USING (artifact_id)
                WHERE jobs.job_id=?""", (value,)).fetchone())
    def compute_endpoint(self, value: str) -> dict[str, Any] | None:
        with self._connection() as db:
            endpoint = _row(db.execute("SELECT * FROM compute_endpoints WHERE endpoint_id=?", (value,)).fetchone())
        return self._endpoint_with_scheduler_capacity(endpoint) if endpoint is not None else None
    def comparison(self, value: str) -> dict[str, Any] | None:
        with self._connection() as db:
            return _row(db.execute("SELECT * FROM comparisons WHERE comparison_id=?", (value,)).fetchone())
    def comparison_group(self, value: str) -> dict[str, Any] | None:
        with self._connection() as db:
            group = _row(db.execute("SELECT * FROM comparison_groups WHERE comparison_group_id=?", (value,)).fetchone())
            if group is None:
                return None
            members = [_row(row) for row in db.execute(
                "SELECT * FROM comparison_group_members WHERE comparison_group_id=? ORDER BY ordinal", (value,)
            ).fetchall()]
        return self._comparison_group_detail(group, [member for member in members if member])
