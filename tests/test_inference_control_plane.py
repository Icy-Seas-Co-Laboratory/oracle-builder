from __future__ import annotations

import hashlib
import json
import uuid

from fastapi.testclient import TestClient

from oracle_builder.orchestration.api import create_app
from oracle_builder.orchestration.service import Orchestrator


def _ready_orchestrator(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    dataset_path = workspace / "frozen.sqlite"
    dataset_path.write_bytes(b"frozen-input")
    model_path = workspace / "model"
    model_path.mkdir()
    model_id = str(uuid.uuid4())
    model_path.joinpath("artifact.json").write_text(json.dumps({"artifact_id": model_id}), encoding="utf-8")
    model_fingerprint = hashlib.sha256(b"model").hexdigest()
    orch = Orchestrator(tmp_path / "control.sqlite", workspace_root=workspace)
    now = "2026-01-01T00:00:00+00:00"
    with orch._connection() as db:  # Fixtures create catalog facts directly; API behavior is under test.
        db.execute("""INSERT INTO datasets VALUES(?,?,?,?,?,?,?,?,?,?)""", (
            "dataset-1", "revision-1", "Frozen", "classification", "frozen", hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            str(dataset_path), "{}", now, now,
        ))
        db.execute("""INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            model_id, None, "model_run", "Model", "classification", "tiny", None, "complete", "sealed", None,
            None, model_fingerprint, str(model_path), "{}", now, now,
        ))
    pool = orch.create_worker_pool(name="Inference", allowed_actions=["infer"])
    return orch, model_id, pool["pool_id"]


def test_inference_request_seals_path_free_work_and_lifecycle_api(tmp_path):
    orch, model_id, pool_id = _ready_orchestrator(tmp_path)
    with TestClient(create_app(orch)) as client:
        created = client.post("/v1/inference-runs", json={
            "name": "demo inference", "model_artifact_id": model_id, "dataset_id": "dataset-1",
            "worker_pool_id": pool_id, "split": "test", "prediction_set": "demo",
            "shard_size": None,  # Exercise the retained unsharded V1 contract.
        })
        assert created.status_code == 201, created.text
        run = created.json()["inference_run"]
        assert run["status"] == "ready"
        work = run["work_unit"]
        assert work["action"] == "infer"
        assert work["parameters"] == {"split": "test", "prediction_set": "demo"}
        assert "/" not in json.dumps(work)

        started = client.post(f"/v1/inference-runs/{run['inference_run_id']}:start")
        assert started.status_code == 200, started.text
        assert started.json()["inference_run"]["status"] == "queued"
        # Action routes are registered before the bare id route.
        assert client.get(f"/v1/inference-runs/{run['inference_run_id']}:download").status_code == 422
