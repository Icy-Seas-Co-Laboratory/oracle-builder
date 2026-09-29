from __future__ import annotations

from oracle_builder.orchestration.lifecycle import PullWorkerSpec, WorkerHealth, WorkerStatus
from oracle_builder.orchestration.service import Orchestrator


class FakeProvider:
    def __init__(self) -> None:
        self.specs = {}
        self.running: set[str] = set()

    def start(self, spec):
        self.specs[spec.worker_id] = spec
        self.running.add(spec.worker_id)
        return WorkerStatus(spec.worker_id, "running", 4242, spec.health_url.removesuffix("/health/ready"), "now")

    def stop(self, worker_id, *, timeout_seconds=10.0):
        self.running.discard(worker_id)
        spec = self.specs[worker_id]
        return WorkerStatus(worker_id, "stopped", 4242, spec.health_url.removesuffix("/health/ready"), "now", 0)

    def status(self, worker_id):
        spec = self.specs.get(worker_id)
        if spec is None:
            return None
        state = "running" if worker_id in self.running else "stopped"
        return WorkerStatus(worker_id, state, 4242, spec.health_url.removesuffix("/health/ready"), "now", None if state == "running" else 0)

    def health(self, worker_id, *, timeout_seconds=2.0):
        status = self.status(worker_id)
        if status is None:
            return WorkerHealth(worker_id, "unknown", False, None, "missing")
        return WorkerHealth(worker_id, status.state, status.state == "running", 200 if status.state == "running" else None, None if status.state == "running" else "stopped")


def test_orchestrator_manages_desired_local_worker_and_compute_endpoint(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = FakeProvider()
    orchestrator = Orchestrator(
        tmp_path / "orchestrator.sqlite", workspace_root=workspace,
        worker_providers={"local-process": provider},
    )

    worker = orchestrator.create_managed_worker(name="GPU 0", port=8123, compute_worker_slots=2)
    assert worker["desired_state"] == "stopped"
    assert worker["state"] == "stopped"

    started = orchestrator.start_managed_worker(worker["worker_id"])
    assert started["desired_state"] == "running"
    assert started["state"] == "running"
    assert started["endpoint_id"]
    endpoint = orchestrator.compute_endpoint(started["endpoint_id"])
    assert endpoint["base_url"] == "http://127.0.0.1:8123"

    reconciled = orchestrator.reconcile_managed_worker(worker["worker_id"])
    assert reconciled["state"] == "running"
    stopped = orchestrator.stop_managed_worker(worker["worker_id"])
    assert stopped["desired_state"] == "stopped"
    assert orchestrator.compute_endpoint(started["endpoint_id"])["enabled"] == 0
    assert [event["event_type"] for event in orchestrator.managed_worker_events(worker["worker_id"])] == [
        "created", "started", "reconciled", "stopped",
    ]


def test_managed_worker_api_is_retired(tmp_path):
    from fastapi.testclient import TestClient
    from oracle_builder.orchestration.api import create_app

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    orchestrator = Orchestrator(
        tmp_path / "orchestrator.sqlite", workspace_root=workspace,
        worker_providers={"local-process": FakeProvider()},
    )
    with TestClient(create_app(orchestrator)) as client:
        assert client.post("/v1/managed-workers", json={"name": "CPU", "port": 8124}).status_code == 404
        assert client.get("/v1/managed-workers").status_code == 404


def test_orchestrator_controls_fixed_pull_worker_provider_from_typed_spec(tmp_path):
    class PullProvider:
        def __init__(self):
            self.started = []
            self.stopped = []

        def start(self, spec):
            self.started.append(spec)
            return WorkerStatus(spec.worker_id, "running", 55, spec.orchestrator_url, "now")

        def stop(self, worker_id, *, timeout_seconds=10.0):
            self.stopped.append((worker_id, timeout_seconds))
            return WorkerStatus(worker_id, "stopped", 55, "http://control", "now", 0)

        def status(self, worker_id):
            return WorkerStatus(worker_id, "running", 55, "http://control", "now")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = PullProvider()
    orchestrator = Orchestrator(
        tmp_path / "orchestrator.sqlite", workspace_root=workspace, pull_worker_provider=provider,
    )
    pool = orchestrator.create_worker_pool(name="GPU", allowed_actions=["train"])
    spec = PullWorkerSpec(
        worker_id="gpu-01", orchestrator_url="https://control.example", pool_id=pool["pool_id"],
        credentials_file=str(tmp_path / "credentials"), scratch_root=str(tmp_path / "scratch"), executors=("train",),
    )
    assert orchestrator.start_pull_worker(spec)["state"] == "running"
    assert provider.started == [spec]
    assert orchestrator.pull_worker_status("gpu-01")["pid"] == 55
    assert orchestrator.stop_pull_worker("gpu-01")["state"] == "stopped"
    assert provider.stopped == [("gpu-01", 10.0)]
