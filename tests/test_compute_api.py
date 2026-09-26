from __future__ import annotations

import uuid
import json
import threading
import time

from fastapi.testclient import TestClient

from oracle_builder.api.app import create_app
from oracle_builder.api.compute import ComputeService, Job
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


def test_compute_preflight_seals_explicit_gpu_ids_and_rejects_unknown_devices(monkeypatch):
    compute = ComputeService()
    monkeypatch.setattr(compute, "_discover_gpus", lambda: [
        {"id": "0", "free_memory_mib": 8000, "total_memory_mib": 10000, "telemetry": "nvidia-smi"},
        {"id": "1", "free_memory_mib": 12000, "total_memory_mib": 16000, "telemetry": "nvidia-smi"},
    ])
    try:
        report = compute.preflight(
            action="run_validate", parameters={"run": "/tmp/run"},
            resources={"gpu_count": 1, "gpu_ids": ["1"]},
        )
        assert report["ready"] is True
        assert report["allocation"] == {"mode": "explicit", "gpu_ids": ["1"], "gpu_count": 1}
        assert report["vram"]["verified"] is True

        unavailable = compute.preflight(
            action="run_validate", parameters={"run": "/tmp/run"},
            resources={"gpu_count": 1, "gpu_ids": ["7"]},
        )
        assert unavailable["ready"] is False
        assert "not advertised" in unavailable["reasons"][0]
    finally:
        compute.close()


def test_batch_tune_passes_vram_inventory_and_target_band_to_isolated_probe(monkeypatch):
    compute = ComputeService()
    commands: list[list[str]] = []
    monkeypatch.setattr(compute, "_discover_gpus", lambda: [
        {"id": "1", "free_memory_mib": 8000, "total_memory_mib": 12288, "telemetry": "nvidia-smi"},
    ])

    class Completed:
        returncode = 0
        stdout = 'ORACLE_BATCH_TUNE_RESULT={"ready": true, "recommended_batch_size": 8}\n'

    def run(command, **_kwargs):
        commands.append(command)
        return Completed()

    monkeypatch.setattr("oracle_builder.api.compute.subprocess.run", run)
    try:
        result = compute.tune_batch_size(
            parameters={"config": "/tmp/config.toml", "input": "/tmp/frozen.sqlite", "minimum_batch_size": 1, "maximum_batch_size": 64},
            resources={"gpu_count": 1, "gpu_ids": ["1"]},
        )
        assert result["ready"] is True
        assert "--vram-total-mib" in commands[0]
        assert commands[0][commands[0].index("--vram-total-mib") + 1] == "12288"
        assert commands[0][commands[0].index("--target-vram-min") + 1] == "0.3"
        assert commands[0][commands[0].index("--target-vram-max") + 1] == "0.8"
    finally:
        compute.close()


def test_compute_training_status_is_job_scoped_and_available_before_artifacts(tmp_path):
    compute = ComputeService()
    app = create_app(InferenceModelRegistry(), compute=compute, preload=False)
    job_id = str(uuid.uuid4())
    run_dir = tmp_path / "runs" / "run-1"
    with TestClient(app) as client:
        accepted = client.post(
            "/compute/jobs",
            json={
                "job_id": job_id,
                "action": "train",
                "parameters": {
                    "config": "/tmp/config.toml", "input": "/tmp/input.sqlite",
                    "runs_dir": str(tmp_path / "runs"), "output": "run-1",
                },
            },
        )
        assert accepted.status_code == 202
        pending = client.get(f"/compute/jobs/{job_id}/training-status")
        assert pending.status_code == 200
        assert pending.json()["available"] is False

        run_dir.mkdir(parents=True)
        (run_dir / "training-status.json").write_text(json.dumps({
            "schema_version": 1, "phase": "Training", "metrics": {},
        }))
        live = client.get(f"/compute/jobs/{job_id}/training-status")
        assert live.status_code == 200
        assert live.json()["available"] is True
        assert live.json()["snapshot"]["phase"] == "Training"

        unknown = client.get(f"/compute/jobs/{uuid.uuid4()}/training-status")
        assert unknown.status_code == 404


def test_compute_running_job_can_pause_resume_and_cancel_without_stranding_process():
    class Process:
        def __init__(self): self.signals = []; self.terminated = False
        def send_signal(self, value): self.signals.append(value)
        def terminate(self): self.terminated = True

    compute = ComputeService()
    job_id, process = str(uuid.uuid4()), Process()
    try:
        job = Job(job_id=job_id, action="train", parameters={}, resources={}, status="running", process=process)  # type: ignore[arg-type]
        compute._jobs[job_id] = job
        compute._worker.status = "busy"

        paused = compute.pause(job_id)
        assert paused["status"] == "paused"
        assert process.signals
        assert compute.workers()[0]["status"] == "paused"

        resumed = compute.resume(job_id)
        assert resumed["status"] == "running"
        assert len(process.signals) == 2
        assert compute.workers()[0]["status"] == "busy"

        compute.pause(job_id)
        compute.cancel(job_id)
        assert process.terminated is True
        # Cancel wakes a paused process before terminating it.
        assert len(process.signals) == 4
    finally:
        compute.close()


def test_compute_scheduler_runs_multiple_slots_without_gpu_collisions(monkeypatch):
    """Two slots can progress concurrently, but each receives a distinct GPU."""
    compute = ComputeService(worker_slots=2, cpu_capacity=2)
    started = threading.Event()
    release = threading.Event()
    executions: list[tuple[str, dict]] = []

    monkeypatch.setattr(compute, "_discover_gpus", lambda: [
        {"id": "0", "free_memory_mib": 10000},
        {"id": "1", "free_memory_mib": 9000},
    ])

    def execute(job):
        executions.append((job.job_id, job.allocation or {}))
        if len(executions) == 2:
            started.set()
        release.wait(timeout=2)
        job.status = "succeeded"
        job.finished_at = time.time()

    monkeypatch.setattr(compute, "_execute", execute)
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    try:
        for job_id in ids:
            compute.submit(
                job_id=job_id, action="run_validate", parameters={"run": "/tmp/run"},
                resources={"gpu_count": 1, "cpu_count": 1},
            )
        assert started.wait(timeout=2)
        assert {item[1]["gpu_ids"][0] for item in executions} == {"0", "1"}
        status = compute.status()
        assert status["queue"]["depth"] == 0
        assert status["resources"]["cpu_in_use"] == 2
        assert len(status["resources"]["gpu_leases"]) == 2
    finally:
        release.set()
        compute.close()


def test_compute_scheduler_keeps_conflicting_gpu_job_queued_and_can_cancel(monkeypatch):
    compute = ComputeService(worker_slots=2, cpu_capacity=2)
    release = threading.Event()
    started = threading.Event()
    monkeypatch.setattr(compute, "_discover_gpus", lambda: [{"id": "0"}])

    def execute(job):
        started.set()
        release.wait(timeout=2)
        job.status = "succeeded"
        job.finished_at = time.time()

    monkeypatch.setattr(compute, "_execute", execute)
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    try:
        for job_id in (first, second):
            compute.submit(
                job_id=job_id, action="run_validate", parameters={"run": "/tmp/run"},
                resources={"gpu_count": 1, "gpu_ids": ["0"]},
            )
        assert started.wait(timeout=2)
        deadline = time.time() + 2
        while compute.get(second)["status"] != "queued" and time.time() < deadline:
            time.sleep(.01)
        assert compute.get(second)["status"] == "queued"
        assert compute.cancel(second)["status"] == "cancelled"
    finally:
        release.set()
        compute.close()
