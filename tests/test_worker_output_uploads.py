from __future__ import annotations

import hashlib
import io
import uuid
import zipfile

import pytest
from fastapi.testclient import TestClient

from oracle_builder.orchestration.api import create_app
from oracle_builder.orchestration.service import Orchestrator
from oracle_data_contracts.artifacts import ArtifactRef
from oracle_data_contracts.work_units import WorkUnit


def _archive() -> bytes:
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w") as bundle:
        bundle.writestr("result.json", b"{}")
        bundle.writestr("metrics.json", b'{"loss": 0.1}')
    return result.getvalue()


def _leased(tmp_path):
    orch = Orchestrator(tmp_path / "control.sqlite", workspace_root=tmp_path)
    pool = orch.create_worker_pool(name="Remote", allowed_actions=["train"])
    worker = orch.register_worker(pool_id=pool["pool_id"], name="worker", registration_token=pool["registration_token"], capabilities={"actions": ["train"], "cpu_capacity": 4})
    unit_id = str(uuid.uuid4())
    unit = WorkUnit(work_unit_id=unit_id, attempt_id=str(uuid.uuid4()), specification_id=str(uuid.uuid4()), action="train", inputs={"dataset": ArtifactRef("dataset", "a" * 64)}, configuration=None, staging=ArtifactRef("staging", unit_id, revision=str(uuid.uuid4())), resources={"cpu_count": 1})
    orch.enqueue_work_unit_for_pool(pool_id=pool["pool_id"], work_unit=unit.to_dict())
    lease = orch.acquire_next_worker_lease(worker_id=worker["worker_id"], worker_token=worker["worker_token"])
    assert lease
    credentials = {"lease_id": lease["lease_id"], "worker_id": worker["worker_id"], "worker_token": worker["worker_token"], "lease_token": lease["lease_token"]}
    orch.acknowledge_worker_lease(**credentials)
    return orch, worker, lease, credentials


def test_resumable_output_upload_accepts_out_of_order_parts_and_finalizes(tmp_path):
    orch, _worker, lease, credentials = _leased(tmp_path)
    archive = _archive()
    part_size = 17
    upload = orch.create_worker_output_upload(**credentials, archive_size=len(archive), archive_sha256=hashlib.sha256(archive).hexdigest(), part_size=part_size)
    assert orch.job_worker_output_upload(lease["job_id"])["uploaded_bytes"] == 0
    count = upload["part_count"]
    assert count > 2
    for number in reversed(range(count)):
        data = archive[number * part_size:(number + 1) * part_size]
        response = orch.upload_worker_output_part(**{key: value for key, value in credentials.items() if key != "lease_id"}, upload_id=upload["upload_id"], part_number=number, part_sha256=hashlib.sha256(data).hexdigest(), part=data)
        assert not response["part_idempotent"]
    assert orch.job_worker_output_upload(lease["job_id"])["uploaded_bytes"] == len(archive)
    first = archive[:part_size]
    assert orch.upload_worker_output_part(**{key: value for key, value in credentials.items() if key != "lease_id"}, upload_id=upload["upload_id"], part_number=0, part_sha256=hashlib.sha256(first).hexdigest(), part=first)["part_idempotent"]
    final = orch.finalize_worker_output_upload(**{key: value for key, value in credentials.items() if key != "lease_id"}, upload_id=upload["upload_id"])
    assert final["status"] == "finalized"
    assert orch.finalize_worker_output_upload(**{key: value for key, value in credentials.items() if key != "lease_id"}, upload_id=upload["upload_id"])["idempotent"]
    assert orch.complete_worker_lease(**credentials, success=True)["outcome"] == "completed"
    assert orch.job(lease["job_id"])["status"] == "completed"


def test_resumable_output_upload_rejects_bad_digest_and_incomplete_finalize(tmp_path):
    orch, _worker, _lease, credentials = _leased(tmp_path)
    archive = _archive()
    upload = orch.create_worker_output_upload(**credentials, archive_size=len(archive), archive_sha256=hashlib.sha256(archive).hexdigest(), part_size=32)
    with pytest.raises(ValueError, match="incomplete"):
        orch.finalize_worker_output_upload(**{key: value for key, value in credentials.items() if key != "lease_id"}, upload_id=upload["upload_id"])
    part = archive[:32]
    with pytest.raises(ValueError, match="does not match"):
        orch.upload_worker_output_part(**{key: value for key, value in credentials.items() if key != "lease_id"}, upload_id=upload["upload_id"], part_number=0, part_sha256="0" * 64, part=part)
    assert orch.worker_output_upload(upload_id=upload["upload_id"], worker_id=credentials["worker_id"], worker_token=credentials["worker_token"], lease_token=credentials["lease_token"])["uploaded_parts"] == []


def test_deferred_output_recovery_rotates_a_fresh_lease_token_for_owning_worker(tmp_path):
    orch, worker, lease, credentials = _leased(tmp_path)
    deferred = orch.defer_worker_output_publication(**credentials)
    assert deferred["status"] == "publication_pending"
    assert orch.job(lease["job_id"])["status"] == "publication_pending"
    with pytest.raises(PermissionError):
        orch.resume_worker_output_publication(
            lease_id=lease["lease_id"], worker_id=worker["worker_id"], worker_token="wrong",
        )
    resumed = orch.resume_worker_output_publication(
        lease_id=lease["lease_id"], worker_id=worker["worker_id"], worker_token=worker["worker_token"],
    )
    assert resumed["status"] == "active" and resumed["lease_token"] != credentials["lease_token"]
    assert orch.job(lease["job_id"])["status"] == "publishing"


def test_resumable_output_upload_api_contract(tmp_path):
    orch, worker, lease, _credentials = _leased(tmp_path)
    archive, part_size = _archive(), 31
    headers = {"Authorization": f"Bearer {worker['worker_token']}"}
    with TestClient(create_app(orch)) as client:
        created = client.post(f"/v1/worker-leases/{lease['lease_id']}/output-uploads", headers=headers, json={"lease_token": lease["lease_token"], "archive_size": len(archive), "archive_sha256": hashlib.sha256(archive).hexdigest(), "part_size": part_size})
        assert created.status_code == 201
        upload_id = created.json()["upload_id"]
        for number in range(created.json()["part_count"]):
            value = archive[number * part_size:(number + 1) * part_size]
            start = number * part_size
            response = client.put(f"/v1/worker-output-uploads/{upload_id}/parts/{number}", headers={**headers, "X-Oracle-Lease-Token": lease["lease_token"], "X-Oracle-Part-SHA256": hashlib.sha256(value).hexdigest(), "Content-Range": f"bytes {start}-{start + len(value) - 1}/{len(archive)}"}, content=value)
            assert response.status_code == 200, response.text
        finalized = client.post(f"/v1/worker-output-uploads/{upload_id}:finalize", headers=headers, json={"lease_token": lease["lease_token"]})
        assert finalized.status_code == 200, finalized.text
        status = client.get(f"/v1/worker-output-uploads/{upload_id}", headers={**headers, "X-Oracle-Lease-Token": lease["lease_token"]})
        assert status.status_code == 200 and status.json()["status"] == "finalized"


def test_tar_ending_in_keras_zip_preserves_outer_run_artifact(tmp_path):
    import tarfile
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, 'w') as zipped:
        zipped.writestr('config.json', '{}')
        zipped.writestr('model.weights.h5', b'weights')
    outer = io.BytesIO()
    with tarfile.open(fileobj=outer, mode='w') as tar:
        for name, data in [('segment-result.json', b'checkpoint'), ('model.keras', nested.getvalue())]:
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
    orchestrator = Orchestrator(tmp_path / 'control.sqlite', workspace_root=tmp_path)
    destination = tmp_path / 'unpacked'
    destination.mkdir()
    orchestrator._extract_worker_output_archive(outer.getvalue(), destination)
    assert (destination / 'segment-result.json').read_bytes() == b'checkpoint'
    assert (destination / 'model.keras').read_bytes() == nested.getvalue()
    assert not (destination / 'model.weights.h5').exists()


def test_worker_archive_flattens_checkpoint_hardlinks_to_regular_files(tmp_path):
    from oracle_builder.worker.pull import archive_directory_to_file
    import os, tarfile
    source = tmp_path / 'checkpoint'
    source.mkdir()
    (source / 'model.keras').write_bytes(b'model')
    os.link(source / 'model.keras', source / 'latest.keras')
    archive = tmp_path / 'output.tar'
    archive_directory_to_file(source, archive)
    with tarfile.open(archive) as tar:
        assert all(member.isreg() for member in tar.getmembers())
    orchestrator = Orchestrator(tmp_path / 'control.sqlite', workspace_root=tmp_path)
    target = tmp_path / 'unpacked'
    target.mkdir()
    orchestrator._extract_worker_output_archive(archive, target)
    assert (target / 'latest.keras').read_bytes() == b'model'
