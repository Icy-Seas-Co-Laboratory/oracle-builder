from __future__ import annotations

import io
import uuid
import zipfile

import pytest
from fastapi.testclient import TestClient

from oracle_builder.orchestration.api import create_app
from oracle_builder.orchestration.service import Orchestrator
from oracle_data_contracts.artifacts import ArtifactRef
from oracle_data_contracts.work_units import WorkUnit


def _archive(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)
    return output.getvalue()


def _leased(tmp_path):
    orch = Orchestrator(tmp_path / "control.sqlite", workspace_root=tmp_path)
    pool = orch.create_worker_pool(name="Remote", allowed_actions=["train"])
    worker = orch.register_worker(
        pool_id=pool["pool_id"], name="worker", registration_token=pool["registration_token"],
        capabilities={"actions": ["train"], "cpu_capacity": 4},
    )
    unit_id = str(uuid.uuid4())
    unit = WorkUnit(
        work_unit_id=unit_id, attempt_id=str(uuid.uuid4()), specification_id=str(uuid.uuid4()), action="train",
        inputs={"dataset": ArtifactRef("dataset", "a" * 64)}, configuration=None,
        staging=ArtifactRef("staging", unit_id, revision=str(uuid.uuid4())), resources={"cpu_count": 1},
    )
    orch.enqueue_work_unit_for_pool(pool_id=pool["pool_id"], work_unit=unit.to_dict())
    lease = orch.acquire_next_worker_lease(worker_id=worker["worker_id"], worker_token=worker["worker_token"])
    assert lease
    return orch, worker, lease


def test_acknowledge_event_upload_and_complete_publish_generic_output(tmp_path):
    orch, worker, lease = _leased(tmp_path)
    credentials = {"lease_id": lease["lease_id"], "worker_id": worker["worker_id"], "worker_token": worker["worker_token"], "lease_token": lease["lease_token"]}

    acknowledged = orch.acknowledge_worker_lease(**credentials)
    assert acknowledged["acknowledged_at"]
    assert orch.acknowledge_worker_lease(**credentials)["output_attempt_id"] == acknowledged["output_attempt_id"]
    event_id = str(uuid.uuid4())
    event = orch.append_worker_lease_event(**credentials, event_id=event_id, event_type="training.started", message="starting", data={"epoch": 0})
    assert orch.append_worker_lease_event(**credentials, event_id=event_id, event_type="training.started", message="starting", data={"epoch": 0})["sequence"] == event["sequence"]
    archive = _archive({"metrics.json": b'{"loss": 0.1}', "models/model.bin": b"model"})
    assert not orch.upload_worker_lease_output_archive(**credentials, archive=archive)["idempotent"]
    assert orch.upload_worker_lease_output_archive(**credentials, archive=archive)["idempotent"]
    complete = orch.complete_worker_lease(**credentials, success=True)
    assert complete["status"] == "released"
    assert complete["outcome"] == "completed"
    assert complete["output_ref"]["kind"] == "job_output"
    assert orch.artifact_store.resolve(ArtifactRef.from_dict(complete["output_ref"])).joinpath("metrics.json").is_file()
    assert orch.job(lease["job_id"])["status"] == "completed"


def test_completion_requires_acknowledgement_and_sealed_output(tmp_path):
    orch, worker, lease = _leased(tmp_path)
    credentials = {"lease_id": lease["lease_id"], "worker_id": worker["worker_id"], "worker_token": worker["worker_token"], "lease_token": lease["lease_token"]}
    with pytest.raises(ValueError, match="acknowledged"):
        orch.complete_worker_lease(**credentials, success=True)
    orch.acknowledge_worker_lease(**credentials)
    with pytest.raises(ValueError, match="staged output"):
        orch.complete_worker_lease(**credentials, success=True)
    failed = orch.complete_worker_lease(**credentials, success=False, message="compiler failed")
    assert failed["outcome"] == "failed"
    assert orch.job(lease["job_id"])["status"] == "failed"


def test_expiry_requeues_an_acknowledged_running_pull_job(tmp_path):
    orch, worker, lease = _leased(tmp_path)
    credentials = {"lease_id": lease["lease_id"], "worker_id": worker["worker_id"], "worker_token": worker["worker_token"], "lease_token": lease["lease_token"]}
    orch.acknowledge_worker_lease(**credentials)
    assert orch.job(lease["job_id"])["status"] == "running"
    with orch._connection() as db:
        db.execute("UPDATE worker_leases SET expires_at=? WHERE lease_id=?", ("2000-01-01T00:00:00+00:00", lease["lease_id"]))
    assert orch.expire_worker_leases() == 1
    assert orch.job(lease["job_id"])["status"] == "queued"


def test_spooled_archive_upload_streams_from_control_plane_file(tmp_path):
    orch, worker, lease = _leased(tmp_path)
    credentials = {"lease_id": lease["lease_id"], "worker_id": worker["worker_id"], "worker_token": worker["worker_token"], "lease_token": lease["lease_token"]}
    orch.acknowledge_worker_lease(**credentials)
    spool = tmp_path / "api-owned-upload.tar"
    spool.write_bytes(_archive({"result.txt": b"spooled"}))
    result = orch.upload_worker_lease_output_archive(**credentials, archive=spool)
    assert result["bytes"] == spool.stat().st_size


def test_output_archive_rejects_traversal_and_other_worker_credentials(tmp_path):
    orch, worker, lease = _leased(tmp_path)
    credentials = {"lease_id": lease["lease_id"], "worker_id": worker["worker_id"], "worker_token": worker["worker_token"], "lease_token": lease["lease_token"]}
    orch.acknowledge_worker_lease(**credentials)
    with pytest.raises(ValueError, match="unsafe"):
        orch.upload_worker_lease_output_archive(**credentials, archive=_archive({"partial.txt": b"partial", "../escape": b"no"}))
    # A rejected archive must not leave partial candidate files that prevent a
    # worker from retrying its upload under the same active lease.
    uploaded = orch.upload_worker_lease_output_archive(**credentials, archive=_archive({"result.json": b"{}"}))
    assert not uploaded["idempotent"]
    with pytest.raises(PermissionError):
        orch.append_worker_lease_event(**{**credentials, "worker_token": "wrong"}, event_id=str(uuid.uuid4()), event_type="test", message="no")


def test_worker_completion_api_spools_stages_and_publishes_without_worker_paths(tmp_path):
    orch, worker, lease = _leased(tmp_path)
    headers = {
        "Authorization": f"Bearer {worker['worker_token']}",
        "X-Oracle-Lease-Token": lease["lease_token"],
    }
    with TestClient(create_app(orch)) as client:
        acknowledged = client.post(
            f"/v1/worker-leases/{lease['lease_id']}:acknowledge",
            headers={"Authorization": headers["Authorization"]},
            json={"lease_token": lease["lease_token"]},
        )
        assert acknowledged.status_code == 200
        event = client.post(
            f"/v1/worker-leases/{lease['lease_id']}:events",
            headers={"Authorization": headers["Authorization"]},
            json={"lease_token": lease["lease_token"], "events": [{
                "event_id": str(uuid.uuid4()), "type": "execution.started", "message": "starting", "data": {},
            }]},
        )
        assert event.status_code == 200
        archive = _archive({"result.json": b"{}"})
        uploaded = client.put(
            f"/v1/worker-leases/{lease['lease_id']}/staging",
            headers={**headers, "Content-Type": "application/x-tar"}, content=archive,
        )
        assert uploaded.status_code == 200
        assert not list((orch.artifact_root / "worker-upload-spool").glob("*.archive"))
        completed = client.post(
            f"/v1/worker-leases/{lease['lease_id']}:complete",
            headers={"Authorization": headers["Authorization"]},
            json={"lease_token": lease["lease_token"], "outcome": "succeeded"},
        )
        assert completed.status_code == 200
        assert completed.json()["output_ref"]["kind"] == "job_output"
