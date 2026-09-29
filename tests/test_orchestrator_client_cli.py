from __future__ import annotations

import json
from pathlib import Path

from oracle_builder.orchestration.client_cli import main


class FakeClient:
    instances: list["FakeClient"] = []

    def __init__(self, base_url: str, *, timeout: float):
        self.base_url = base_url
        self.timeout = timeout
        self.calls: list[tuple] = []
        self.instances.append(self)

    def request(self, method, path, *, body=None, query=None):
        self.calls.append((method, path, body, query))
        return {"method": method, "path": path, "body": body, "query": query}

    def upload(self, kind: str, source: Path):
        self.calls.append(("UPLOAD", kind, source, None))
        return {"path": "/owned/uploads/datasets/training.sqlite"}


def invoke(capsys, args: list[str]) -> tuple[int, dict, FakeClient]:
    FakeClient.instances.clear()
    result = main(args, client_factory=FakeClient)
    captured = capsys.readouterr()
    return result, json.loads(captured.out), FakeClient.instances[0]


def test_health_and_pool_enqueue_are_control_plane_calls(capsys):
    code, output, client = invoke(capsys, ["--url", "http://control:8110/", "health"])
    assert code == 0
    assert output["path"] == "/health/ready"
    assert client.base_url == "http://control:8110/"

    code, output, client = invoke(capsys, ["spec", "enqueue", "spec id", "--worker-pool", "remote-gpu"])
    assert code == 0
    assert output["method"] == "POST"
    assert client.calls == [("POST", "/v1/specifications/spec%20id:enqueue", {"worker_pool_id": "remote-gpu"}, None)]


def test_definition_queue_seals_work_for_a_worker_pool(capsys):
    code, output, client = invoke(capsys, [
        "definition", "queue", "definition id", "--name", "baseline", "--dataset", "dataset-1",
        "--worker-pool", "gpu-pool", "--gpu-count", "1", "--batch-size", "32", "--epochs", "12",
    ])
    assert code == 0
    assert output["path"] == "/v1/model-definitions/definition%20id:queue"
    assert client.calls == [("POST", "/v1/model-definitions/definition%20id:queue", {
        "name": "baseline", "dataset_id": "dataset-1", "worker_pool_id": "gpu-pool", "description": "",
        "resources": {"gpu_count": 1}, "batch_size": 32, "epochs": 12,
    }, None)]


def test_dataset_upload_only_streams_then_uses_server_returned_path(tmp_path, capsys):
    source = tmp_path / "training.sqlite"
    source.write_bytes(b"sqlite")
    code, output, client = invoke(capsys, ["dataset", "upload", str(source), "--ingest"])
    assert code == 0
    assert output["dataset"]["body"] == {"path": "/owned/uploads/datasets/training.sqlite"}
    assert client.calls[0] == ("UPLOAD", "datasets", source, None)
    assert client.calls[1] == ("POST", "/v1/datasets:ingest", {"path": "/owned/uploads/datasets/training.sqlite"}, None)


def test_recipe_upload_can_register_only_the_server_returned_path(tmp_path, capsys):
    source = tmp_path / "baseline.toml"
    source.write_text("[run]\n")
    code, output, client = invoke(capsys, ["recipe", "upload", str(source), "--create", "--name", "baseline"])
    assert code == 0
    assert output["recipe"]["body"] == {"name": "baseline", "config_path": "/owned/uploads/datasets/training.sqlite", "description": ""}
    assert client.calls[0] == ("UPLOAD", "configs", source, None)


def test_worker_pool_surface_does_not_expose_managed_compute_lifecycle(capsys):
    code, output, client = invoke(capsys, ["worker", "pool-create", "--name", "remote", "--action", "train", "--max-workers", "2"])
    assert code == 0
    assert output["path"] == "/v1/worker-pools"
    assert client.calls[0][2] == {"name": "remote", "allowed_actions": ["train"], "max_workers": 2}


def test_inference_submission_is_an_orchestrator_request(capsys):
    code, output, client = invoke(capsys, [
        "inference", "submit", "--name", "batch", "--model", "model-1",
        "--dataset", "dataset-1", "--worker-pool", "cpu", "--split", "test",
        "--resources", '{"cpu_count": 2}',
    ])
    assert code == 0
    assert output["path"] == "/v1/inference-runs"
    assert client.calls == [("POST", "/v1/inference-runs", {
        "name": "batch", "model_artifact_id": "model-1", "dataset_id": "dataset-1",
        "worker_pool_id": "cpu", "split": "test", "prediction_set": None,
        "resources": {"cpu_count": 2},
    }, None)]


def test_queue_verification_requires_explicit_start_flag(capsys):
    _, output, _ = invoke(capsys, ['queue', 'verify', 'run-1', 'run-2'])
    assert output['path'] == '/v1/queued-runs:verify'
    assert output['body'] == {'queued_run_ids': ['run-1', 'run-2'], 'start_after_verification': False}
    _, output, _ = invoke(capsys, ['queue', 'verify', 'run-1', '--start'])
    assert output['body']['start_after_verification'] is True
    _, output, _ = invoke(capsys, ['queue', 'start', 'run-1', '--worker-pool', 'pool'])
    assert output['path'] == '/v1/queued-runs:start'
    assert output['body'] == {'queued_run_ids': ['run-1'], 'worker_pool_id': 'pool'}
