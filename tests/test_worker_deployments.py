from __future__ import annotations

import subprocess
import json

from fastapi.testclient import TestClient

from oracle_builder.orchestration.api import create_app
from oracle_builder.orchestration.lifecycle import (
    DockerPullWorkerProfile,
    DockerPullWorkerProvider,
    PullWorkerDeploymentProfile,
    WorkerStatus,
)
from oracle_builder.orchestration.service import Orchestrator
from oracle_builder.orchestration.cli import load_deployment_profiles


class Provider:
    def __init__(self) -> None:
        self.started = []
        self.stopped = []

    def start(self, spec):
        self.started.append(spec)
        return WorkerStatus(spec.worker_id, "running", None, "test://worker")

    def stop(self, worker_id, *, timeout_seconds=10.0):
        self.stopped.append((worker_id, timeout_seconds))
        return WorkerStatus(worker_id, "stopped", None, "test://worker")

    def status(self, worker_id):
        return None


def _orchestrator(tmp_path):
    profile = PullWorkerDeploymentProfile(
        profile_id="gpu", provider="test", orchestrator_url="https://control.example",
        credentials_root=str(tmp_path / "credentials"), scratch_root=str(tmp_path / "scratch"),
        executors=("train",), registration_token_file=str(tmp_path / "pool-token"), capabilities={"gpu": "1"},
    )
    provider = Provider()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return Orchestrator(tmp_path / "control.sqlite", workspace_root=workspace,
                        deployment_profiles={"gpu": profile}, deployment_providers={"test": provider}), provider


def test_deployment_persists_only_profile_reference_and_reconciles_unknown(tmp_path):
    orchestrator, provider = _orchestrator(tmp_path)
    pool = orchestrator.create_worker_pool(name="GPU", allowed_actions=["train"])
    deployment = orchestrator.create_worker_deployment(name="GPU one", pool_id=pool["pool_id"], profile_id="gpu")
    assert deployment["profile_id"] == "gpu"
    assert "credentials_root" not in deployment
    started = orchestrator.start_worker_deployment(deployment["deployment_id"])
    assert started["desired_state"] == "running"
    assert provider.started[0].credentials_file.endswith(f"{deployment['deployment_id']}.json")
    assert orchestrator.reconcile_worker_deployments()[0]["state"] == "unknown"
    stopped = orchestrator.stop_worker_deployment(deployment["deployment_id"])
    assert stopped["desired_state"] == "stopped"


def test_deployment_routes_are_operator_protected_and_accept_no_paths(tmp_path):
    orchestrator, _provider = _orchestrator(tmp_path)
    pool = orchestrator.create_worker_pool(name="GPU", allowed_actions=["train"])
    token = "operator-token"
    import hashlib
    app = create_app(orchestrator, role_tokens={"operator": hashlib.sha256(token.encode()).hexdigest()})
    with TestClient(app) as client:
        assert client.post("/v1/worker-deployments", json={"name": "gpu", "pool_id": pool["pool_id"], "profile_id": "gpu"}).status_code == 401
        response = client.post("/v1/worker-deployments", headers={"Authorization": f"Bearer {token}"}, json={"name": "gpu", "pool_id": pool["pool_id"], "profile_id": "gpu", "command": "bad"})
        assert response.status_code == 422


def test_docker_provider_mounts_token_file_without_token_value(tmp_path):
    credential_root, scratch_root = tmp_path / "credentials", tmp_path / "scratch"
    credential_root.mkdir(); scratch_root.mkdir()
    token = tmp_path / "pool-token"; token.write_text("secret", encoding="utf-8")
    seen = []
    def runner(command, **kwargs):
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, "container-id\n", "")
    profile = DockerPullWorkerProfile(image="oracle-worker:test", credentials_root=str(credential_root), scratch_root=str(scratch_root), registration_token_file=str(token))
    provider = DockerPullWorkerProvider(profile, runner=runner)
    spec = PullWorkerDeploymentProfile("gpu", "docker", "https://control.example", str(credential_root), str(scratch_root), ("train",), str(token)).spec_for("deploy-1", "GPU", "pool")
    provider.start(spec)
    command = seen[0]
    assert "pool-token" not in " ".join(command)
    assert "--mount" in command and "registration-token" in " ".join(command)


def test_cli_loads_fixed_docker_profiles_without_persisting_unsafe_fields(tmp_path):
    document = {
        "profiles": [{"profile_id": "gpu", "provider": "docker", "orchestrator_url": "https://control.example",
                      "credentials_root": str(tmp_path / "credentials"), "scratch_root": str(tmp_path / "scratch"),
                      "executors": ["train"], "docker": {"image": "oracle-worker:test"}}]
    }
    source = tmp_path / "profiles.json"; source.write_text(json.dumps(document), encoding="utf-8")
    profiles, providers = load_deployment_profiles(str(source))
    assert profiles["gpu"].provider == "docker:gpu"
    assert "docker:gpu" in providers
