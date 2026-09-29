from __future__ import annotations

import io
import json
import uuid
import zipfile

from oracle_builder.orchestration.service import Orchestrator
from oracle_data_contracts.artifacts import ArtifactRef
from oracle_data_contracts.work_units import WorkUnit


def _archive() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        bundle.writestr("result.json", b"{}")
    return output.getvalue()


def _leased(tmp_path):
    orch = Orchestrator(tmp_path / "control.sqlite", workspace_root=tmp_path)
    pool = orch.create_worker_pool(name="pool", allowed_actions=["train"])
    worker = orch.register_worker(
        pool_id=pool["pool_id"], name="worker", registration_token=pool["registration_token"],
        capabilities={"actions": ["train"]},
    )
    job_id = str(uuid.uuid4())
    unit = WorkUnit(
        work_unit_id=job_id, attempt_id=str(uuid.uuid4()), specification_id=str(uuid.uuid4()), action="train",
        inputs={"dataset": ArtifactRef("dataset", "dataset")}, configuration=None,
        staging=ArtifactRef("staging", job_id, revision=str(uuid.uuid4())), resources={},
    )
    # The unit needs no materialization for these durability-only tests.
    orch.enqueue_work_unit_for_pool(pool_id=pool["pool_id"], work_unit=unit.to_dict())
    lease = orch.acquire_next_worker_lease(worker_id=worker["worker_id"], worker_token=worker["worker_token"])
    assert lease is not None
    credentials = {"lease_id": lease["lease_id"], "worker_id": worker["worker_id"], "worker_token": worker["worker_token"], "lease_token": lease["lease_token"]}
    return orch, worker, lease, credentials


def _expire(orch, lease_id: str) -> None:
    with orch._connection() as db:
        db.execute("UPDATE worker_leases SET expires_at=? WHERE lease_id=?", ("2000-01-01T00:00:00+00:00", lease_id))
    assert orch.expire_worker_leases() == 1


def test_expiry_gets_exactly_one_infrastructure_retry_with_attempt_history(tmp_path):
    orch, worker, lease, _ = _leased(tmp_path)
    _expire(orch, lease["lease_id"])
    job = orch.job(lease["job_id"])
    assert job["status"] == "queued" and job["retry_count"] == 1
    retry = orch.acquire_next_worker_lease(worker_id=worker["worker_id"], worker_token=worker["worker_token"])
    assert retry is not None
    _expire(orch, retry["lease_id"])
    assert orch.job(lease["job_id"])["status"] == "failed"
    with orch._connection() as db:
        attempts = db.execute("SELECT ordinal,status,classification FROM execution_attempts WHERE job_id=? ORDER BY ordinal", (lease["job_id"],)).fetchall()
    assert [(item["ordinal"], item["status"], item["classification"]) for item in attempts] == [
        (1, "expired", "infrastructure"), (2, "expired", "infrastructure"),
    ]


def test_scientific_failure_is_terminal_and_never_retried(tmp_path):
    orch, _worker, lease, credentials = _leased(tmp_path)
    orch.acknowledge_worker_lease(**credentials)
    orch.complete_worker_lease(**credentials, success=False, message="invalid training data")
    job = orch.job(lease["job_id"])
    assert job["status"] == "failed" and job["retry_count"] == 0
    with orch._connection() as db:
        attempt = db.execute("SELECT status,classification FROM execution_attempts WHERE job_id=?", (lease["job_id"],)).fetchone()
    assert tuple(attempt) == ("failed", "scientific")


def test_cancel_request_is_durable_and_worker_acknowledgement_is_terminal(tmp_path):
    orch, _worker, lease, credentials = _leased(tmp_path)
    requested = orch.request_pull_job_cancellation(lease["job_id"])
    assert requested["status"] == "cancel_requested"
    assert orch.worker_lease_cancellation(**credentials) == {"cancel_requested": True}
    cancelled = orch.cancel_worker_lease(**credentials)
    assert cancelled["outcome"] == "cancelled"
    assert orch.job(lease["job_id"])["status"] == "cancelled"


def test_startup_reconciles_a_store_publication_left_between_filesystem_and_db_commit(tmp_path):
    orch, _worker, lease, credentials = _leased(tmp_path)
    acknowledged = orch.acknowledge_worker_lease(**credentials)
    orch.upload_worker_lease_output_archive(**credentials, archive=_archive())
    staging = orch.artifact_store.staging(lease["job_id"], acknowledged["output_attempt_id"])
    assert orch.artifact_store.validate(staging, orch._valid_worker_output).valid
    orch.artifact_store.seal(staging)
    expected = ArtifactRef("job_output", lease["job_id"], revision=acknowledged["output_attempt_id"])
    orch.artifact_store.publish(staging, expected)
    assert orch.reconcile_worker_publications() == 1
    assert orch.worker_lease(lease["lease_id"])["output_ref"] == expected.to_dict()
    assert orch.job(lease["job_id"])["status"] == "completed"


def test_publication_reconciliation_finishes_model_catalog_promotion(tmp_path, monkeypatch):
    orch, _worker, lease, _credentials = _leased(tmp_path)
    model_ref = ArtifactRef("model_run", str(uuid.uuid4()), revision="attempt")
    with orch._connection() as db:
        db.execute(
            "UPDATE worker_leases SET output_ref_json=? WHERE lease_id=?",
            (json.dumps(model_ref.to_dict()), lease["lease_id"]),
        )
        db.execute("UPDATE jobs SET status='completed',remote_status='completed' WHERE job_id=?", (lease["job_id"],))
    monkeypatch.setattr(orch.artifact_store, "resolve", lambda ref: tmp_path / "published")
    monkeypatch.setattr(orch, "scan", lambda _path: {"artifacts": [model_ref.artifact_id], "skipped": []})

    assert orch.reconcile_worker_publications() == 1
    assert orch.job(lease["job_id"])["status"] == "indexed"


def test_liveness_marks_silent_worker_offline_and_recovers_its_job(tmp_path):
    orch, worker, lease, _ = _leased(tmp_path)
    with orch._connection() as db:
        db.execute("UPDATE registered_workers SET last_seen_at=? WHERE worker_id=?", ("2000-01-01T00:00:00+00:00", worker["worker_id"]))
    report = orch.reconcile_worker_liveness(stale_after_seconds=1)
    assert report == {"offline_workers": 1, "expired_leases": 1}
    assert orch.registered_worker(worker["worker_id"])["state"] == "offline"
    assert orch.job(lease["job_id"])["status"] == "queued"


def test_audit_ledger_never_accepts_or_stores_request_payloads(tmp_path):
    orch = Orchestrator(tmp_path / "control.sqlite", workspace_root=tmp_path)
    orch.record_audit_event(actor_role="operator", method="POST", path="/v1/jobs/job:cancel", outcome="accepted", request_id="req-1")
    with orch._connection() as db:
        row = db.execute("SELECT actor_role,method,path,outcome,request_id FROM audit_events").fetchone()
    assert tuple(row) == ("operator", "POST", "/v1/jobs/job:cancel", "accepted", "req-1")


def test_one_way_schema_migration_keeps_legacy_job_as_history_not_executable(tmp_path):
    database = tmp_path / "control.sqlite"
    orch = Orchestrator(database, workspace_root=tmp_path)
    with orch._connection() as db:
        db.execute("UPDATE schema_metadata SET value='4' WHERE key='schema_version'")
        db.execute(
            "INSERT INTO jobs(job_id,specification_id,oracle_serve_url,action,parameters_json,resources_json,status,submitted_at,updated_at) "
            "VALUES('legacy-job',NULL,'http://old-serve','train','{}','{}','running','2000-01-01','2000-01-01')"
        )
    # `connect` runs the forward-only migration on each control-plane open.
    upgraded = Orchestrator(database, workspace_root=tmp_path)
    historical = upgraded.job("legacy-job")
    assert historical["status"] == "historical"
    assert historical["execution_generation"] == "historical"
