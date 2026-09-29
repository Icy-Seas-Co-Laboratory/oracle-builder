from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from oracle_builder.orchestration.service import Orchestrator
from oracle_data_contracts.artifacts import ArtifactRef
from oracle_data_contracts.work_units import WorkUnit


def _unit(*, action: str = "train", legacy: bool = False) -> dict:
    unit_id = str(uuid.uuid4())
    input_ref = ArtifactRef("legacy_file", "a" * 64) if legacy else ArtifactRef("dataset", "a" * 64)
    return WorkUnit(
        work_unit_id=unit_id, attempt_id=str(uuid.uuid4()), specification_id=str(uuid.uuid4()),
        action=action, inputs={"input": input_ref}, configuration=None,
        staging=ArtifactRef("staging", unit_id, revision=str(uuid.uuid4())), resources={"cpu_count": 1},
    ).to_dict()


def _pool_and_worker(orch: Orchestrator, *, capabilities: dict | None = None) -> tuple[dict, dict]:
    pool = orch.create_worker_pool(name="Remote compute", allowed_actions=["train"])
    worker = orch.register_worker(
        pool_id=pool["pool_id"], name="gpu-a", registration_token=pool["registration_token"],
        capabilities=capabilities or {"actions": ["train"], "cpu_capacity": 4, "gpu_ids": []},
    )
    return pool, worker


def test_registered_worker_can_claim_and_release_path_free_work_unit(tmp_path):
    orch = Orchestrator(tmp_path / "control.sqlite", workspace_root=tmp_path)
    pool, worker = _pool_and_worker(orch)
    queued = orch.enqueue_work_unit_for_pool(pool_id=pool["pool_id"], work_unit=_unit())

    lease = orch.acquire_next_worker_lease(worker_id=worker["worker_id"], worker_token=worker["worker_token"])

    assert lease and lease["work_unit"]["work_unit_id"] == queued["job_id"]
    assert "lease_token" in lease and "token_sha256" not in lease
    assert orch.job(queued["job_id"])["status"] == "leased"
    assert orch.release_worker_lease(lease_id=lease["lease_id"], lease_token=lease["lease_token"], outcome="completed")["status"] == "released"
    assert orch.job(queued["job_id"])["status"] == "queued"


def test_job_timing_collects_worker_measurements_and_legacy_event_boundaries(tmp_path):
    orch = Orchestrator(tmp_path / "control.sqlite", workspace_root=tmp_path)
    pool, worker = _pool_and_worker(orch)
    queued = orch.enqueue_work_unit_for_pool(pool_id=pool["pool_id"], work_unit=_unit())
    lease = orch.acquire_next_worker_lease(worker_id=worker["worker_id"], worker_token=worker["worker_token"])
    assert lease
    credentials = {
        "lease_id": lease["lease_id"], "worker_id": worker["worker_id"],
        "worker_token": worker["worker_token"], "lease_token": lease["lease_token"],
    }
    orch.acknowledge_worker_lease(**credentials)
    orch.append_worker_lease_event(**credentials, event_id=str(uuid.uuid4()), event_type="materializing", message="starting", data={})
    orch.append_worker_lease_event(**credentials, event_id=str(uuid.uuid4()), event_type="executing", message="running", data={})
    orch.append_worker_lease_event(
        **credentials, event_id=str(uuid.uuid4()), event_type="stage_timing", message="done",
        data={"schema": "oracle_timing_v1", "stage": "artifact_materialization", "duration_seconds": 1.25, "input_count": 2},
    )

    timing = orch.job_timing(queued["job_id"])

    assert timing["summary"]["queue_seconds"] is not None
    assert len(timing["stages"]) == 1
    assert timing["stages"][0]["stage"] == "artifact_materialization"
    assert timing["stages"][0]["duration_seconds"] == 1.25
    assert timing["stages"][0]["details"]["input_count"] == 2
    assert timing["derived_stages"][0]["stage"] == "artifact_materialization"


def test_lease_is_atomic_and_worker_auth_and_action_capability_are_enforced(tmp_path):
    orch = Orchestrator(tmp_path / "control.sqlite", workspace_root=tmp_path)
    pool, worker = _pool_and_worker(orch, capabilities={"actions": [], "cpu_capacity": 4})
    queued = orch.enqueue_work_unit_for_pool(pool_id=pool["pool_id"], work_unit=_unit())
    with pytest.raises(PermissionError):
        orch.authenticate_worker(worker_id=worker["worker_id"], worker_token="not-the-token")
    assert orch.acquire_next_worker_lease(worker_id=worker["worker_id"], worker_token=worker["worker_token"]) is None

    capable = orch.register_worker(pool_id=pool["pool_id"], name="gpu-b", registration_token=pool["registration_token"], capabilities={"actions": ["train"], "cpu_capacity": 4})
    lease = orch.acquire_worker_lease(worker_id=capable["worker_id"], worker_token=capable["worker_token"], work_unit_id=queued["job_id"])
    with pytest.raises(ValueError, match="active lease"):
        orch.acquire_worker_lease(worker_id=capable["worker_id"], worker_token=capable["worker_token"], work_unit_id=queued["job_id"])
    with pytest.raises(PermissionError):
        orch.renew_worker_lease(lease_id=lease["lease_id"], lease_token="wrong-token")


def test_expired_lease_returns_job_to_retryable_queue_and_legacy_paths_require_opt_in(tmp_path):
    orch = Orchestrator(tmp_path / "control.sqlite", workspace_root=tmp_path)
    pool, worker = _pool_and_worker(orch)
    queued = orch.enqueue_work_unit_for_pool(pool_id=pool["pool_id"], work_unit=_unit(legacy=True))
    assert orch.acquire_next_worker_lease(worker_id=worker["worker_id"], worker_token=worker["worker_token"]) is None

    compat = orch.register_worker(pool_id=pool["pool_id"], name="local-compat", registration_token=pool["registration_token"], capabilities={"actions": ["train"], "execution_mode": "local_path_compat", "cpu_capacity": 2})
    lease = orch.acquire_next_worker_lease(worker_id=compat["worker_id"], worker_token=compat["worker_token"], ttl_seconds=5)
    assert lease
    with orch._connection() as db:
        db.execute("UPDATE worker_leases SET expires_at=? WHERE lease_id=?", ("2000-01-01T00:00:00+00:00", lease["lease_id"]))
    with pytest.raises(ValueError, match="no longer active"):
        orch.renew_worker_lease(lease_id=lease["lease_id"], lease_token=lease["lease_token"])
    assert orch.job(queued["job_id"])["status"] == "queued"
    assert orch.worker_lease(lease["lease_id"])["status"] == "expired"


def test_registration_token_and_pool_policy_are_not_disclosed(tmp_path):
    orch = Orchestrator(tmp_path / "control.sqlite", workspace_root=tmp_path)
    pool = orch.create_worker_pool(name="Safe pool", allowed_actions=["train"], max_workers=1)
    assert "registration_token" not in orch.worker_pool(pool["pool_id"])
    with pytest.raises(PermissionError):
        orch.register_worker(pool_id=pool["pool_id"], name="bad", registration_token="bad", capabilities={"actions": ["train"]})
    worker = orch.register_worker(pool_id=pool["pool_id"], name="good", registration_token=pool["registration_token"], capabilities={"actions": ["train"]})
    assert "worker_token" not in orch.registered_worker(worker["worker_id"])
    with pytest.raises(ValueError, match="registration limit"):
        orch.register_worker(pool_id=pool["pool_id"], name="too-many", registration_token=pool["registration_token"], capabilities={"actions": ["train"]})


def test_two_workers_cannot_concurrently_claim_one_work_unit(tmp_path):
    orch = Orchestrator(tmp_path / "control.sqlite", workspace_root=tmp_path)
    pool, first = _pool_and_worker(orch)
    second = orch.register_worker(pool_id=pool["pool_id"], name="second", registration_token=pool["registration_token"], capabilities={"actions": ["train"], "cpu_capacity": 2})
    orch.enqueue_work_unit_for_pool(pool_id=pool["pool_id"], work_unit=_unit())

    def claim(worker: dict):
        return orch.acquire_next_worker_lease(worker_id=worker["worker_id"], worker_token=worker["worker_token"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(claim, (first, second)))
    assert sum(claim is not None for claim in claims) == 1
    assert len([lease for lease in orch.worker_leases() if lease["status"] == "active"]) == 1


def test_pull_jobs_are_not_reconciled_as_legacy_http_push_jobs(tmp_path):
    orch = Orchestrator(tmp_path / "control.sqlite", workspace_root=tmp_path)
    pool, _ = _pool_and_worker(orch)
    orch.enqueue_work_unit_for_pool(pool_id=pool["pool_id"], work_unit=_unit())

    assert orch.reconcile_active_jobs() == []
