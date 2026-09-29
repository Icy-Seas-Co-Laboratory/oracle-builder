from __future__ import annotations

from fastapi.testclient import TestClient

from oracle_builder.orchestration.api import create_app
from oracle_builder.orchestration.service import Orchestrator


def test_legacy_push_and_model_import_routes_are_explicitly_gone(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    orchestrator = Orchestrator(tmp_path / "control.sqlite", workspace_root=workspace)

    with TestClient(create_app(orchestrator)) as client:
        # These routes intentionally do not parse their historical bodies: a
        # caller must receive the migration signal, rather than a misleading
        # validation error from the old contract.
        training = client.post("/v1/experiments:train", json={"old": "payload"})
        imported = client.post("/v1/model-imports", json={"old": "payload"})
        refreshed = client.get("/v1/jobs?refresh=true")
        jobs = client.get("/v1/jobs")

    assert training.status_code == 410
    assert "model definition" in training.json()["detail"]
    assert imported.status_code == 410
    assert "Model import" in imported.json()["detail"]
    assert refreshed.status_code == 410
    assert jobs.status_code == 200

