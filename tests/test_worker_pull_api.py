from __future__ import annotations

import uuid
import json

from fastapi.testclient import TestClient

from oracle_builder.orchestration.api import create_app
from oracle_builder.orchestration.service import Orchestrator
from oracle_builder.orchestration.storage import artifact_ref_for_file
from oracle_data_contracts.artifacts import ArtifactRef
from oracle_data_contracts.work_units import WorkUnit


def _orchestrator(tmp_path) -> Orchestrator:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return Orchestrator(tmp_path / "orchestrator.sqlite", workspace_root=workspace)


def _queued_work_unit(orchestrator: Orchestrator, pool_id: str) -> WorkUnit:
    work_unit = WorkUnit(
        work_unit_id=str(uuid.uuid4()), attempt_id=str(uuid.uuid4()), specification_id=str(uuid.uuid4()),
        action="train", inputs={"input": ArtifactRef("dataset", "dataset-1", revision="1")},
        configuration=None, staging=ArtifactRef("staging", str(uuid.uuid4())), resources={"cpu": 1},
    )
    orchestrator.enqueue_work_unit_for_pool(pool_id=pool_id, work_unit=work_unit.to_dict())
    return work_unit


def test_pull_registration_and_path_free_durable_lease_api(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    with TestClient(create_app(orchestrator)) as client:
        pool_response = client.post("/v1/worker-pools", json={"name": "Training", "allowed_actions": ["train"]})
        assert pool_response.status_code == 201
        pool = pool_response.json()
        assert pool["registration_token"]
        assert "registration_token_sha256" not in pool

        bad = client.post("/v1/workers:register", json={"pool_id": pool["pool_id"], "name": "w1", "registration_token": "wrong-registration-token-value"})
        assert bad.status_code == 403
        registered = client.post("/v1/workers:register", json={
            "pool_id": pool["pool_id"], "name": "w1", "registration_token": pool["registration_token"],
            "endpoint": "http://worker.example:8100", "capabilities": {"accelerator": "cpu", "actions": ["train"]},
        })
        assert registered.status_code == 201
        worker = registered.json()
        assert worker["worker_token"]
        assert "auth_token_sha256" not in worker

        unauthorized = client.post(f"/v1/workers/{worker['worker_id']}:lease", json={})
        assert unauthorized.status_code == 401
        headers = {"Authorization": f"Bearer {worker['worker_token']}"}
        assert client.post(f"/v1/workers/{worker['worker_id']}:lease", headers=headers, json={}).status_code == 204

        expected = _queued_work_unit(orchestrator, pool["pool_id"])
        leased = client.post(f"/v1/workers/{worker['worker_id']}:lease", headers=headers, json={"ttl_seconds": 60})
        assert leased.status_code == 200
        body = leased.json()
        assert set(body) == {"lease", "work_unit"}
        assert body["work_unit"] == expected.to_dict()
        assert "parameters" not in body["work_unit"]
        assert body["lease"]["work_unit_id"] == expected.work_unit_id
        assert body["lease"]["lease_token"]

        assert client.post(f"/v1/worker-leases/{body['lease']['lease_id']}:renew", json={
            "lease_token": body["lease"]["lease_token"], "ttl_seconds": 60,
        }).status_code == 401
        renewed = client.post(f"/v1/worker-leases/{body['lease']['lease_id']}:renew", json={
            "lease_token": body["lease"]["lease_token"], "ttl_seconds": 60,
        }, headers=headers)
        assert renewed.status_code == 200
        released = client.post(f"/v1/worker-leases/{body['lease']['lease_id']}:release", json={
            "lease_token": body["lease"]["lease_token"], "outcome": "released",
        }, headers=headers)
        assert released.status_code == 200
        assert released.json()["status"] == "released"


def test_registration_rejects_server_owned_worker_id_and_command(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    with TestClient(create_app(orchestrator)) as client:
        pool = client.post("/v1/worker-pools", json={"name": "Evaluation", "allowed_actions": ["evaluate"]}).json()
        body = {"pool_id": pool["pool_id"], "name": "w1", "registration_token": pool["registration_token"]}
        assert client.post("/v1/workers:register", json={**body, "command": ["not", "allowed"]}).status_code == 422
        assert client.post("/v1/workers:register", json={**body, "worker_id": str(uuid.uuid4())}).status_code == 422


def test_planned_specification_can_be_queued_to_a_worker_pool_without_path_payload(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    now = "2026-09-28T00:00:00+00:00"
    experiment_id, specification_id = str(uuid.uuid4()), str(uuid.uuid4())
    dataset_source, config_source = tmp_path / "training.sqlite", tmp_path / "resolved.toml"
    dataset_source.write_bytes(b"portable dataset")
    config_source.write_text("[run]\ntask = 'classification'\n", encoding="utf-8")
    dataset_ref = artifact_ref_for_file("dataset", str(uuid.uuid4()), dataset_source, revision="1")
    config_ref = artifact_ref_for_file("configuration", specification_id, config_source, revision="1")
    orchestrator.artifact_store.ingest_file(dataset_ref, dataset_source)
    orchestrator.artifact_store.ingest_file(config_ref, config_source)
    with orchestrator._connection() as db:
        db.execute(
            """INSERT INTO experiments(experiment_id,project_id,name,description,dataset_id,status,plan_json,created_at,updated_at)
               VALUES (?, NULL, 'Test', '', NULL, 'planned', '{}', ?, ?)""",
            (experiment_id, now, now),
        )
        db.execute(
            """INSERT INTO run_specifications(specification_id,experiment_id,ordinal,name,action,parameters_json,
               resources_json,config_hash,status,artifact_id,created_at,updated_at)
               VALUES (?, ?, 1, 'Run', 'train', ?, '{}', 'digest', 'planned', NULL, ?, ?)""",
                (specification_id, experiment_id, json.dumps({"artifact_inputs": {"input": dataset_ref.to_dict()}, "configuration_artifact": config_ref.to_dict()}), now, now),
        )
    with TestClient(create_app(orchestrator)) as client:
        pool = client.post("/v1/worker-pools", json={"name": "Training", "allowed_actions": ["train"]}).json()
        queued = client.post(f"/v1/specifications/{specification_id}:enqueue", json={"worker_pool_id": pool["pool_id"]})

    assert queued.status_code == 201
    job = queued.json()
    assert job["status"] == "queued"
    assert job["worker_pool_id"] == pool["pool_id"]
    assert job["parameters"] == {}
    with orchestrator._connection() as db:
        stored_unit = db.execute("SELECT work_unit_json FROM jobs WHERE job_id=?", (job["job_id"],)).fetchone()[0]
    assert str(tmp_path) not in stored_unit
    assert orchestrator.specification(specification_id)["status"] == "queued"
    assert orchestrator.job_events(job["job_id"])[0]["event_type"] == "queued"
