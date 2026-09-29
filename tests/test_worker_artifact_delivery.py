from __future__ import annotations

import io
import json
import tarfile
import uuid

import pytest
from fastapi.testclient import TestClient

from oracle_data_contracts.artifacts import ArtifactRef
from oracle_data_contracts.work_units import WorkUnit
from oracle_builder.orchestration.api import create_app
from oracle_builder.orchestration.service import Orchestrator
from oracle_builder.orchestration.storage import LocalArtifactStore
from oracle_builder.orchestration.storage import artifact_ref_for_file
from oracle_builder.orchestration.work_units import build_work_unit


def _store_with_artifact(tmp_path):
    source = tmp_path / "legacy-dataset"
    source.mkdir()
    (source / "payload.txt").write_text("remote-safe\n", encoding="utf-8")
    nested = source / "nested"
    nested.mkdir()
    (nested / "metadata.json").write_text('{"version": 1}\n', encoding="utf-8")
    store = LocalArtifactStore(tmp_path / "artifact-store")
    ref = ArtifactRef("dataset", "dataset-remote-1", "1")
    store.register_existing(ref, source)
    return store, ref, source


def test_lease_scoped_grant_streams_a_safe_gzip_archive_without_persisting_secrets(tmp_path):
    store, ref, _ = _store_with_artifact(tmp_path)
    grant = store.issue_materialization_grant(ref, lease_id="lease-1", job_id="job-1")

    payload = b"".join(store.stream_grant_archive(
        grant.token, lease_id="lease-1", job_id="job-1", chunk_size=1024,
    ))
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        assert archive.getnames() == ["artifact", "artifact/nested", "artifact/nested/metadata.json", "artifact/payload.txt"]
        assert archive.extractfile("artifact/payload.txt").read() == b"remote-safe\n"

    persisted = (tmp_path / "artifact-store" / ".materialization-grants.json").read_text(encoding="utf-8")
    assert grant.token not in persisted
    assert json.loads(persisted)["grants"][grant.grant_id]["status"] == "used"
    with pytest.raises(PermissionError, match="already"):
        list(store.stream_grant_archive(grant.token, lease_id="lease-1", job_id="job-1"))


def test_delivery_rejects_cross_lease_use_and_does_not_burn_grant_on_unsafe_source(tmp_path):
    store, ref, source = _store_with_artifact(tmp_path)
    grant = store.issue_materialization_grant(ref, lease_id="lease-1", job_id="job-1")
    with pytest.raises(PermissionError, match="scope"):
        list(store.stream_grant_archive(grant.token, lease_id="other-lease", job_id="job-1"))

    (source / "unsafe-link").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        list(store.stream_grant_archive(grant.token, lease_id="lease-1", job_id="job-1"))
    (source / "unsafe-link").unlink()
    assert b"".join(store.stream_grant_archive(grant.token, lease_id="lease-1", job_id="job-1"))


def test_worker_can_issue_and_download_only_its_leased_input_artifact(tmp_path):
    store, ref, _ = _store_with_artifact(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    orchestrator = Orchestrator(
        tmp_path / "orchestrator.sqlite", workspace_root=workspace, artifact_store=store,
    )
    unit = WorkUnit(
        work_unit_id=str(uuid.uuid4()), attempt_id=str(uuid.uuid4()), specification_id=str(uuid.uuid4()),
        action="train", inputs={"dataset": ref}, configuration=None,
        staging=ArtifactRef("staging", str(uuid.uuid4())), resources={"cpu": 1},
    )
    with TestClient(create_app(orchestrator)) as client:
        pool = client.post("/v1/worker-pools", json={"name": "remote", "allowed_actions": ["train"]}).json()
        worker = client.post("/v1/workers:register", json={
            "pool_id": pool["pool_id"], "name": "worker", "registration_token": pool["registration_token"],
            "capabilities": {"actions": ["train"]},
        }).json()
        orchestrator.enqueue_work_unit_for_pool(pool_id=pool["pool_id"], work_unit=unit.to_dict())
        worker_headers = {"Authorization": f"Bearer {worker['worker_token']}"}
        leased = client.post(f"/v1/workers/{worker['worker_id']}:lease", headers=worker_headers, json={}).json()
        lease = leased["lease"]

        refused = client.post(f"/v1/worker-leases/{lease['lease_id']}:artifact-grants", headers=worker_headers, json={
            "lease_token": lease["lease_token"], "ref": unit.staging.to_dict(),
        })
        assert refused.status_code == 403
        issued = client.post(f"/v1/worker-leases/{lease['lease_id']}:artifact-grants", headers=worker_headers, json={
            "lease_token": lease["lease_token"], "ref": ref.to_dict(),
        })
        assert issued.status_code == 201
        delivery = issued.json()
        assert delivery["grant"]["token"]

        downloaded = client.get(delivery["download_path"], headers={
            **worker_headers,
            "X-Oracle-Lease-Token": lease["lease_token"],
            "X-Oracle-Artifact-Grant": delivery["grant"]["token"],
        })
        assert downloaded.status_code == 200
        assert downloaded.headers["content-type"].startswith("application/gzip")
        with tarfile.open(fileobj=io.BytesIO(downloaded.content), mode="r:gz") as archive:
            assert archive.extractfile("artifact/payload.txt").read() == b"remote-safe\n"
        assert delivery["grant"]["token"] not in (tmp_path / "artifact-store" / ".materialization-grants.json").read_text(encoding="utf-8")
        assert client.get(delivery["download_path"], headers={
            **worker_headers,
            "X-Oracle-Lease-Token": lease["lease_token"],
            "X-Oracle-Artifact-Grant": delivery["grant"]["token"],
        }).status_code == 403


def test_portable_registered_dataset_and_config_are_delivered_without_host_paths(tmp_path):
    """A pull lease sees ref identities and bytes, never submission paths."""
    dataset_file = tmp_path / "registered.sqlite"
    config_file = tmp_path / "registered.toml"
    dataset_file.write_bytes(b"SQLite format 3\x00remote dataset")
    config_file.write_text("[training]\nepochs=1\n", encoding="utf-8")
    store = LocalArtifactStore(tmp_path / "artifact-store")
    dataset_ref = artifact_ref_for_file("dataset", "registered-dataset", dataset_file, revision="1")
    config_ref = artifact_ref_for_file("configuration", "registered-config", config_file)
    store.ingest_file(dataset_ref, dataset_file)
    store.ingest_file(config_ref, config_file)
    unit = build_work_unit(
        work_unit_id=str(uuid.uuid4()), attempt_id=str(uuid.uuid4()), specification_id=str(uuid.uuid4()),
        action="train", resources={}, require_portable=True,
        parameters={
            "input": str(dataset_file), "config": str(config_file),
            "artifact_inputs": {"input": dataset_ref.to_dict()},
            "configuration_artifact": config_ref.to_dict(),
        },
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    orchestrator = Orchestrator(tmp_path / "orchestrator.sqlite", workspace_root=workspace, artifact_store=store)
    with TestClient(create_app(orchestrator)) as client:
        pool = client.post("/v1/worker-pools", json={"name": "remote", "allowed_actions": ["train"]}).json()
        worker = client.post("/v1/workers:register", json={
            "pool_id": pool["pool_id"], "name": "worker", "registration_token": pool["registration_token"],
            "capabilities": {"actions": ["train"]},
        }).json()
        orchestrator.enqueue_work_unit_for_pool(pool_id=pool["pool_id"], work_unit=unit.to_dict())
        headers = {"Authorization": f"Bearer {worker['worker_token']}"}
        lease_response = client.post(f"/v1/workers/{worker['worker_id']}:lease", headers=headers, json={})
        assert lease_response.status_code == 200, lease_response.text
        leased = lease_response.json()
        lease = leased["lease"]
        assert lease is not None, lease_response.text
        assert str(tmp_path) not in json.dumps(leased["work_unit"], sort_keys=True)
        assert leased["work_unit"]["configuration"] == config_ref.to_dict()
        grant = client.post(f"/v1/worker-leases/{lease['lease_id']}:artifact-grants", headers=headers, json={
            "lease_token": lease["lease_token"], "ref": dataset_ref.to_dict(),
        }).json()
        response = client.get(grant["download_path"], headers={
            **headers, "X-Oracle-Lease-Token": lease["lease_token"],
            "X-Oracle-Artifact-Grant": grant["grant"]["token"],
        })
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
            assert archive.extractfile("artifact/payload").read() == dataset_file.read_bytes()
