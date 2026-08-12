from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from oracle_builder.api.app import create_app
from oracle_builder.api.compute import ComputeService
from oracle_builder.api.registry import InferenceModelRegistry


def test_compute_api_exposes_local_worker_and_validates_orchestrator_job_ids():
    compute = ComputeService()
    app = create_app(InferenceModelRegistry(), compute=compute, preload=False)
    with TestClient(app) as client:
        workers = client.get("/compute/workers")
        assert workers.status_code == 200
        assert workers.json()["workers"][0]["status"] == "idle"
        assert "train" in workers.json()["workers"][0]["capabilities"]["actions"]
        status = client.get("/compute/status")
        assert status.status_code == 200
        assert status.json()["queue"] == {"depth": 0, "capacity": 128}

        bad = client.post(
            "/compute/jobs",
            json={"job_id": "not-a-uuid", "action": "train", "parameters": {}},
        )
        assert bad.status_code == 422


def test_compute_job_lifecycle_and_events_are_available_before_execution():
    compute = ComputeService()
    app = create_app(InferenceModelRegistry(), compute=compute, preload=False)
    job_id = str(uuid.uuid4())
    with TestClient(app) as client:
        accepted = client.post(
            "/compute/jobs",
            json={
                "job_id": job_id,
                "action": "run_validate",
                "parameters": {"run": "/path/that-does-not-exist"},
            },
        )
        assert accepted.status_code == 202
        assert accepted.json()["status"] in {"queued", "running"}

        job = client.get(f"/compute/jobs/{job_id}")
        assert job.status_code == 200
        assert job.json()["job_id"] == job_id

        events = client.get(f"/compute/jobs/{job_id}/events")
        assert events.status_code == 200
        assert any(event["type"] == "queued" for event in events.json()["events"])


def test_compute_completion_reports_the_resolved_output_path():
    compute = ComputeService()
    job_id = str(uuid.uuid4())
    job = compute.submit(
        job_id=job_id,
        action="train",
        parameters={
            "config": "/tmp/config.toml",
            "input": "/tmp/dataset.sqlite",
            "runs_dir": "/tmp/oracle-runs",
            "output": "run-id",
        },
    )
    try:
        assert job["result"] is None
        assert compute._jobs[job_id].output_path == "/tmp/oracle-runs/run-id"
    finally:
        compute.close()
