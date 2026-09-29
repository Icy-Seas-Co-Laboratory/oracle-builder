from __future__ import annotations

import subprocess

import pytest

from oracle_builder.orchestration.lifecycle import (
    LocalProcessWorkerProvider,
    LocalWorkerSpec,
    PullWorkerProcessProvider,
    PullWorkerSpec,
)


class FakeProcess:
    def __init__(self, *, pid: int = 1234, exit_code: int | None = None, timeout_on_wait: bool = False) -> None:
        self.pid = pid
        self.exit_code = exit_code
        self.timeout_on_wait = timeout_on_wait
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.exit_code = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.timeout_on_wait and not self.killed:
            raise subprocess.TimeoutExpired("worker", timeout)
        self.exit_code = -15 if self.terminated and not self.killed else self.exit_code
        return self.exit_code or 0


def test_local_provider_builds_fixed_shell_free_command() -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    process = FakeProcess()

    def spawn(command: tuple[str, ...], **kwargs: object) -> FakeProcess:
        calls.append((command, kwargs))
        return process

    provider = LocalProcessWorkerProvider(executable=("oracle-worker",), process_factory=spawn)
    status = provider.start(LocalWorkerSpec("gpu-local", port=8123, compute_cpu_capacity=4))

    assert status.state == "running"
    assert calls[0][0] == (
        "oracle-worker", "--role", "compute", "--host", "127.0.0.1", "--port", "8123", "--no-preload",
        "--worker-id", "gpu-local", "--compute-queue-size", "128",
        "--compute-worker-slots", "1", "--compute-cpu-capacity", "4",
    )
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["stdin"] is subprocess.DEVNULL


def test_local_provider_reports_health_and_dead_process_without_network() -> None:
    process = FakeProcess()
    provider = LocalProcessWorkerProvider(
        executable=("worker",), process_factory=lambda *_args, **_kwargs: process,
        http_get=lambda url, timeout: 200,
    )
    provider.start(LocalWorkerSpec("one", host="localhost", port=8124))

    health = provider.health("one")
    assert health.ready is True
    assert health.status_code == 200
    process.exit_code = 2
    health = provider.health("one")
    assert health.ready is False
    assert health.state == "stopped"
    assert provider.status("missing") is None


def test_stop_escalates_to_kill_after_timeout() -> None:
    process = FakeProcess(timeout_on_wait=True)
    provider = LocalProcessWorkerProvider(
        executable=("worker",), process_factory=lambda *_args, **_kwargs: process,
    )
    provider.start(LocalWorkerSpec("one"))

    stopped = provider.stop("one", timeout_seconds=0.01)
    assert process.terminated is True
    assert process.killed is True
    assert stopped.state == "stopped"
    assert stopped.exit_code == -9


def test_specs_reject_unsafe_or_invalid_values() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        LocalWorkerSpec("bad id")
    with pytest.raises(ValueError, match="option prefixes"):
        LocalWorkerSpec("--role")
    with pytest.raises(ValueError, match="between"):
        LocalWorkerSpec("worker", port=0)
    with pytest.raises(ValueError, match="positive"):
        LocalWorkerSpec("worker", compute_worker_slots=0)


def test_pull_provider_launches_current_worker_without_token_in_argv(tmp_path) -> None:
    token_file = tmp_path / "pool-token"
    token_file.write_text("bootstrap-secret\n", encoding="utf-8")
    token_file.chmod(0o600)
    process = FakeProcess()
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def spawn(command: tuple[str, ...], **kwargs: object) -> FakeProcess:
        calls.append((command, kwargs))
        return process

    provider = PullWorkerProcessProvider(executable=("oracle-worker",), process_factory=spawn)
    spec = PullWorkerSpec(
        worker_id="gpu-local",
        orchestrator_url="https://orchestrator.example",
        pool_id="gpu-pool",
        credentials_file=str(tmp_path / "worker-credentials.json"),
        scratch_root=str(tmp_path / "scratch"),
        executors=("train",),
        registration_token_file=str(token_file),
        capabilities={"gpu_ids": "0"},
    )
    status = provider.start(spec)

    assert status.state == "running"
    command, kwargs = calls[0]
    assert command == (
        "oracle-worker", "--orchestrator-url", "https://orchestrator.example",
        "--pool-id", "gpu-pool", "--name", "gpu-local",
        "--credentials-file", str(tmp_path / "worker-credentials.json"),
        "--scratch-root", str(tmp_path / "scratch"), "--poll-interval-seconds", "2.0",
        "--executor", "train", "--capability", "gpu_ids=0",
    )
    assert "bootstrap-secret" not in " ".join(command)
    assert kwargs["env"]["ORACLE_BUILDER_WORKER_REGISTRATION_TOKEN"] == "bootstrap-secret"


def test_pull_provider_rejects_insecure_bootstrap_token_file(tmp_path) -> None:
    token_file = tmp_path / "pool-token"
    token_file.write_text("bootstrap-secret\n", encoding="utf-8")
    token_file.chmod(0o644)
    provider = PullWorkerProcessProvider(executable=("oracle-worker",), process_factory=lambda *_args, **_kwargs: FakeProcess())
    spec = PullWorkerSpec(
        worker_id="local", orchestrator_url="http://127.0.0.1:8110", pool_id="pool",
        credentials_file=str(tmp_path / "credentials"), scratch_root=str(tmp_path / "scratch"),
        executors=("package",), registration_token_file=str(token_file),
    )
    with pytest.raises(ValueError, match="group/world"):
        provider.start(spec)


def test_pull_worker_spec_accepts_inference_executor(tmp_path) -> None:
    spec = PullWorkerSpec(
        worker_id="inference-worker", orchestrator_url="https://orchestrator.example",
        pool_id="pool", credentials_file=str(tmp_path / "credentials.json"),
        scratch_root=str(tmp_path / "scratch"), executors=("infer",),
    )
    assert spec.executors == ("infer",)
