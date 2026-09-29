"""Local lifecycle supervision for Oracle compute workers.

This module deliberately owns only processes started by this instance.  It is
not a remote shell abstraction: remote workers register themselves (or are
managed by a deployment-specific provider) in a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Mapping, Protocol
from urllib.error import URLError
from urllib.request import urlopen


class WorkerProvider(Protocol):
    """Narrow lifecycle boundary used by the orchestration control plane."""

    def start(self, spec: "LocalWorkerSpec") -> "WorkerStatus": ...

    def stop(self, worker_id: str, *, timeout_seconds: float = 10.0) -> "WorkerStatus": ...

    def status(self, worker_id: str) -> "WorkerStatus | None": ...

    def health(self, worker_id: str, *, timeout_seconds: float = 2.0) -> "WorkerHealth": ...


@dataclass(frozen=True)
class LocalWorkerSpec:
    """The intentionally small, safe launch surface for a local worker."""

    worker_id: str
    host: str = "127.0.0.1"
    port: int = 8100
    compute_queue_size: int = 128
    compute_worker_slots: int = 1
    compute_cpu_capacity: int | None = None
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (not self.worker_id or self.worker_id.startswith("-")
                or any(character.isspace() or character == "\x00" for character in self.worker_id)):
            raise ValueError("worker_id must be a non-empty identifier without whitespace or option prefixes")
        if (not self.host or self.host.startswith("-")
                or any(character.isspace() or character == "\x00" for character in self.host)):
            raise ValueError("host must be non-empty and contain no whitespace or option prefixes")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.compute_queue_size < 1 or self.compute_worker_slots < 1:
            raise ValueError("compute capacities must be positive")
        if self.compute_cpu_capacity is not None and self.compute_cpu_capacity < 1:
            raise ValueError("compute_cpu_capacity must be positive when supplied")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.environment.items()):
            raise ValueError("environment keys and values must be strings")

    @property
    def health_url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        return f"http://{host}:{self.port}/health/ready"


@dataclass(frozen=True)
class WorkerStatus:
    worker_id: str
    state: str
    pid: int | None
    endpoint: str
    started_at: str | None = None
    exit_code: int | None = None


@dataclass(frozen=True)
class WorkerHealth:
    worker_id: str
    state: str
    ready: bool
    status_code: int | None
    detail: str | None = None


@dataclass
class _ManagedWorker:
    spec: LocalWorkerSpec
    process: subprocess.Popen[object]
    started_at: str


ProcessFactory = Callable[..., subprocess.Popen[object]]
HttpGet = Callable[[str, float], int]


def _default_http_get(url: str, timeout_seconds: float) -> int:
    with urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310 - URL is built from an explicit local spec.
        return int(response.status)


class LocalProcessWorkerProvider:
    """Safely supervise local ``oracle-worker`` processes.

    Commands are always an argument vector, never a shell string.  The worker
    executable is fixed at construction time and specifications cannot append
    arbitrary flags.  Process and HTTP dependencies are injectable so this
    provider can be tested without spawning a server.
    """

    def __init__(
        self,
        *,
        executable: tuple[str, ...] | None = None,
        process_factory: ProcessFactory = subprocess.Popen,
        http_get: HttpGet = _default_http_get,
    ) -> None:
        self._executable = executable or (sys.executable, "-m", "oracle_builder.worker.cli")
        if not self._executable or any(not value for value in self._executable):
            raise ValueError("executable must contain one or more non-empty arguments")
        self._process_factory = process_factory
        self._http_get = http_get
        self._workers: dict[str, _ManagedWorker] = {}

    def command_for(self, spec: LocalWorkerSpec) -> tuple[str, ...]:
        """Return the complete fixed argv used for a worker launch."""
        command = [*self._executable, "--role", "compute", "--host", spec.host, "--port", str(spec.port), "--no-preload",
                   "--worker-id", spec.worker_id, "--compute-queue-size", str(spec.compute_queue_size),
                   "--compute-worker-slots", str(spec.compute_worker_slots)]
        if spec.compute_cpu_capacity is not None:
            command.extend(("--compute-cpu-capacity", str(spec.compute_cpu_capacity)))
        return tuple(command)

    def start(self, spec: LocalWorkerSpec) -> WorkerStatus:
        existing = self._workers.get(spec.worker_id)
        if existing is not None and existing.process.poll() is None:
            raise RuntimeError(f"worker {spec.worker_id!r} is already running")
        environment = os.environ.copy()
        environment.update(spec.environment)
        process = self._process_factory(
            self.command_for(spec), shell=False, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=environment,
        )
        managed = _ManagedWorker(spec=spec, process=process, started_at=datetime.now(UTC).isoformat())
        self._workers[spec.worker_id] = managed
        return self._status(managed)

    def stop(self, worker_id: str, *, timeout_seconds: float = 10.0) -> WorkerStatus:
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds cannot be negative")
        managed = self._workers.get(worker_id)
        if managed is None:
            raise KeyError(f"unknown local worker {worker_id!r}")
        process = managed.process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout_seconds)
        return self._status(managed)

    def status(self, worker_id: str) -> WorkerStatus | None:
        managed = self._workers.get(worker_id)
        return None if managed is None else self._status(managed)

    def health(self, worker_id: str, *, timeout_seconds: float = 2.0) -> WorkerHealth:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        managed = self._workers.get(worker_id)
        if managed is None:
            return WorkerHealth(worker_id, "unknown", False, None, "not managed by this provider")
        status = self._status(managed)
        if status.state != "running":
            return WorkerHealth(worker_id, status.state, False, None, "process is not running")
        try:
            code = self._http_get(managed.spec.health_url, timeout_seconds)
        except (OSError, URLError, TimeoutError) as exc:
            return WorkerHealth(worker_id, "running", False, None, str(exc))
        return WorkerHealth(worker_id, "running", code == 200, code, None if code == 200 else "readiness check failed")

    @staticmethod
    def _status(managed: _ManagedWorker) -> WorkerStatus:
        exit_code = managed.process.poll()
        return WorkerStatus(
            worker_id=managed.spec.worker_id,
            state="running" if exit_code is None else "stopped",
            pid=getattr(managed.process, "pid", None),
            endpoint=managed.spec.health_url.rsplit("/health/ready", 1)[0],
            started_at=managed.started_at,
            exit_code=exit_code,
        )


# Pull-worker lifecycle ----------------------------------------------------
#
# These classes replace the retired HTTP compute-server launch shape for all
# new deployments.  The older classes above remain only so pre-v5 database
# records can be inspected during the removal window; they are no longer
# exposed by the orchestration API.


@dataclass(frozen=True)
class PullWorkerSpec:
    """A fixed local launch declaration for an outbound Oracle worker.

    Registration data is read from an operator-created ``0600`` file and is
    passed through the child environment, never written into argv, SQLite, or
    logs.  Once the worker has registered, its private credentials file lets
    later starts omit the bootstrap token entirely.
    """

    worker_id: str
    orchestrator_url: str
    pool_id: str
    credentials_file: str
    scratch_root: str
    executors: tuple[str, ...]
    name: str | None = None
    registration_token_file: str | None = None
    capabilities: Mapping[str, str] = field(default_factory=dict)
    poll_interval_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not self.worker_id or self.worker_id.startswith("-") or any(char.isspace() or char == "\x00" for char in self.worker_id):
            raise ValueError("worker_id must be a non-empty identifier without whitespace or option prefixes")
        if not self.orchestrator_url.startswith(("http://", "https://")) or any(char.isspace() for char in self.orchestrator_url):
            raise ValueError("orchestrator_url must be an HTTP(S) URL without whitespace")
        if not self.pool_id or any(char.isspace() or char == "\x00" for char in self.pool_id):
            raise ValueError("pool_id must be a non-empty token without whitespace")
        if not self.credentials_file or not self.scratch_root:
            raise ValueError("credentials_file and scratch_root are required")
        if not self.executors or any(action not in {"package", "train", "infer"} for action in self.executors):
            raise ValueError("executors must contain one or more supported fixed actions")
        if len(set(self.executors)) != len(self.executors):
            raise ValueError("executors must not contain duplicates")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if any(not key or not value or any(char.isspace() for char in key) for key, value in self.capabilities.items()):
            raise ValueError("capabilities must contain non-empty whitespace-free keys and values")


@dataclass
class _ManagedPullWorker:
    spec: PullWorkerSpec
    process: subprocess.Popen[object]
    started_at: str


class PullWorkerProcessProvider:
    """Lifecycle provider for the current stateless pull-worker runtime.

    This provider is intentionally local-process only.  Docker, Kubernetes,
    and VM providers should translate the same :class:`PullWorkerSpec` into
    their native, preconfigured deployment primitives rather than accepting a
    generic command supplied by the API.
    """

    def __init__(self, *, executable: tuple[str, ...] | None = None,
                 process_factory: ProcessFactory = subprocess.Popen) -> None:
        self._executable = executable or (sys.executable, "-m", "oracle_builder.worker.cli")
        if not self._executable or any(not value for value in self._executable):
            raise ValueError("executable must contain one or more non-empty arguments")
        self._process_factory = process_factory
        self._workers: dict[str, _ManagedPullWorker] = {}

    @staticmethod
    def _bootstrap_token(spec: PullWorkerSpec) -> str | None:
        if not spec.registration_token_file:
            return None
        token_path = Path(spec.registration_token_file).expanduser()
        if not token_path.is_file() or token_path.is_symlink():
            raise ValueError("registration_token_file must be a regular file")
        if token_path.stat().st_mode & 0o077:
            raise ValueError("registration_token_file must not be group/world accessible")
        token = token_path.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("registration_token_file is empty")
        return token

    def command_for(self, spec: PullWorkerSpec) -> tuple[str, ...]:
        command = [
            *self._executable,
            "--orchestrator-url", spec.orchestrator_url,
            "--pool-id", spec.pool_id,
            "--name", spec.name or spec.worker_id,
            "--credentials-file", spec.credentials_file,
            "--scratch-root", spec.scratch_root,
            "--poll-interval-seconds", str(spec.poll_interval_seconds),
        ]
        for action in spec.executors:
            command.extend(("--executor", action))
        for key, value in sorted(spec.capabilities.items()):
            command.extend(("--capability", f"{key}={value}"))
        return tuple(command)

    def start(self, spec: PullWorkerSpec) -> WorkerStatus:
        existing = self._workers.get(spec.worker_id)
        if existing is not None and existing.process.poll() is None:
            raise RuntimeError(f"worker {spec.worker_id!r} is already running")
        environment = os.environ.copy()
        token = self._bootstrap_token(spec)
        if token is not None:
            environment["ORACLE_BUILDER_WORKER_REGISTRATION_TOKEN"] = token
        process = self._process_factory(
            self.command_for(spec), shell=False, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=environment,
        )
        managed = _ManagedPullWorker(spec, process, datetime.now(UTC).isoformat())
        self._workers[spec.worker_id] = managed
        return self._status(managed)

    def stop(self, worker_id: str, *, timeout_seconds: float = 10.0) -> WorkerStatus:
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds cannot be negative")
        managed = self._workers.get(worker_id)
        if managed is None:
            raise KeyError(f"unknown pull worker {worker_id!r}")
        if managed.process.poll() is None:
            managed.process.terminate()
            try:
                managed.process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                managed.process.kill()
                managed.process.wait(timeout=timeout_seconds)
        return self._status(managed)

    def status(self, worker_id: str) -> WorkerStatus | None:
        managed = self._workers.get(worker_id)
        return None if managed is None else self._status(managed)

    def health(self, worker_id: str, *, timeout_seconds: float = 2.0) -> WorkerHealth:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        status = self.status(worker_id)
        if status is None:
            return WorkerHealth(worker_id, "unknown", False, None, "not managed by this provider")
        if status.state != "running":
            return WorkerHealth(worker_id, status.state, False, None, "process is not running")
        # Pull workers intentionally expose no listener.  Registration and
        # heartbeats are reconciled from the orchestrator's durable records.
        return WorkerHealth(worker_id, "running", True, None, "outbound pull worker process is running")

    @staticmethod
    def _status(managed: _ManagedPullWorker) -> WorkerStatus:
        exit_code = managed.process.poll()
        return WorkerStatus(
            worker_id=managed.spec.worker_id,
            state="running" if exit_code is None else "stopped",
            pid=getattr(managed.process, "pid", None),
            endpoint=managed.spec.orchestrator_url,
            started_at=managed.started_at,
            exit_code=exit_code,
        )


# Declarative deployment providers ---------------------------------------

class PullWorkerLifecycleProvider(Protocol):
    """Provider boundary for a pre-approved pull-worker deployment profile."""

    def start(self, spec: PullWorkerSpec) -> WorkerStatus: ...
    def stop(self, worker_id: str, *, timeout_seconds: float = 10.0) -> WorkerStatus: ...
    def status(self, worker_id: str) -> WorkerStatus | None: ...


@dataclass(frozen=True)
class PullWorkerDeploymentProfile:
    """Operator-owned recipe for one class of pull-worker deployment.

    It is intentionally loaded at process startup.  The API stores only its
    opaque ``profile_id`` and cannot replace paths, bootstrap tokens, image,
    network, or executable actions.
    """

    profile_id: str
    provider: str
    orchestrator_url: str
    credentials_root: str
    scratch_root: str
    executors: tuple[str, ...]
    registration_token_file: str | None = None
    capabilities: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.profile_id or any(char.isspace() or char == "\x00" for char in self.profile_id):
            raise ValueError("profile_id must be a non-empty token")
        if not self.provider or any(char.isspace() or char == "\x00" for char in self.provider):
            raise ValueError("provider must be a non-empty token")
        if not Path(self.credentials_root).is_absolute() or not Path(self.scratch_root).is_absolute():
            raise ValueError("deployment profile roots must be absolute")

    def spec_for(self, deployment_id: str, name: str, pool_id: str) -> PullWorkerSpec:
        return PullWorkerSpec(
            worker_id=deployment_id, name=name, orchestrator_url=self.orchestrator_url, pool_id=pool_id,
            credentials_file=str(Path(self.credentials_root) / f"{deployment_id}.json"),
            scratch_root=str(Path(self.scratch_root) / deployment_id), executors=self.executors,
            registration_token_file=self.registration_token_file, capabilities=self.capabilities,
        )


@dataclass(frozen=True)
class DockerPullWorkerProfile:
    """Fixed Docker policy.  It is operator configuration, not API input.

    The profile deliberately contains host paths and secret-file references.
    They never enter the deployment database, response payloads, or browser
    requests.  The worker bootstrap token is read inside the container from a
    read-only mount, rather than being placed in Docker environment metadata.
    """

    image: str
    network: str = "bridge"
    credentials_root: str = "/var/lib/oracle-worker/credentials"
    scratch_root: str = "/var/lib/oracle-worker/scratch"
    registration_token_file: str | None = None
    docker_executable: str = "docker"
    resource_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.image or any(char.isspace() or char == "\x00" for char in self.image):
            raise ValueError("Docker image must be non-empty and contain no whitespace")
        if not self.network or any(char.isspace() or char == "\x00" for char in self.network):
            raise ValueError("Docker network must be non-empty and contain no whitespace")
        if not self.docker_executable or any(char.isspace() or char == "\x00" for char in self.docker_executable):
            raise ValueError("docker_executable must be a fixed executable name")
        for value in (self.credentials_root, self.scratch_root, self.registration_token_file):
            if value is not None and (not Path(value).is_absolute() or "\x00" in value):
                raise ValueError("Docker profile paths must be absolute")
        if any(not argument or "\x00" in argument for argument in self.resource_args):
            raise ValueError("Docker resource arguments must be non-empty")


DockerRunner = Callable[..., subprocess.CompletedProcess[str]]


class DockerPullWorkerProvider:
    """Start fixed-profile pull workers with Docker, never arbitrary commands.

    Docker itself owns the process after creation, so state is queried by the
    deterministic deployment container name.  This avoids retaining a PID or
    a raw secret in the control plane.  The class accepts only a typed
    :class:`PullWorkerSpec` constructed by the Orchestrator from a configured
    profile.
    """

    def __init__(self, profile: DockerPullWorkerProfile, *, runner: DockerRunner = subprocess.run) -> None:
        self.profile = profile
        self._runner = runner

    @staticmethod
    def _container_name(worker_id: str) -> str:
        # PullWorkerSpec already rejects whitespace; keep Docker's name set
        # narrow and deterministic so status/stop cannot target another unit.
        if not worker_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("worker_id must contain only letters, digits, dashes, or underscores for Docker")
        return f"oracle-worker-{worker_id}"

    def _container_paths(self, spec: PullWorkerSpec) -> tuple[str, str, str | None]:
        credentials = Path(spec.credentials_file).resolve()
        scratch = Path(spec.scratch_root).resolve()
        credentials_root = Path(self.profile.credentials_root).resolve()
        scratch_root = Path(self.profile.scratch_root).resolve()
        if not credentials.is_relative_to(credentials_root) or not scratch.is_relative_to(scratch_root):
            raise ValueError("pull-worker spec paths are outside this Docker profile")
        token: str | None = None
        if spec.registration_token_file:
            configured = self.profile.registration_token_file
            if not configured or Path(spec.registration_token_file).resolve() != Path(configured).resolve():
                raise ValueError("registration token file is not approved by this Docker profile")
            token = "/oracle/secrets/registration-token"
        return f"/oracle/credentials/{credentials.name}", f"/oracle/scratch/{scratch.name}", token

    def command_for(self, spec: PullWorkerSpec) -> tuple[str, ...]:
        name = self._container_name(spec.worker_id)
        credentials_path, scratch_path, token_path = self._container_paths(spec)
        credential_root = str(Path(self.profile.credentials_root).resolve())
        scratch_root = str(Path(self.profile.scratch_root).resolve())
        command = [self.profile.docker_executable, "run", "--detach", "--name", name, "--network", self.profile.network]
        command.extend(self.profile.resource_args)
        command.extend(("--mount", f"type=bind,src={credential_root},dst=/oracle/credentials",
                        "--mount", f"type=bind,src={scratch_root},dst=/oracle/scratch"))
        if token_path:
            token_parent = str(Path(self.profile.registration_token_file or "").resolve().parent)
            command.extend(("--mount", f"type=bind,src={token_parent},dst=/oracle/secrets,readonly"))
        worker = ["oracle-worker", "--orchestrator-url", spec.orchestrator_url, "--pool-id", spec.pool_id,
                  "--name", spec.name or spec.worker_id, "--credentials-file", credentials_path,
                  "--scratch-root", scratch_path, "--poll-interval-seconds", str(spec.poll_interval_seconds)]
        for action in spec.executors:
            worker.extend(("--executor", action))
        for key, value in sorted(spec.capabilities.items()):
            worker.extend(("--capability", f"{key}={value}"))
        # This script is constant; no user/profile values are interpolated.
        bootstrap = "if [ -f /oracle/secrets/registration-token ]; then export ORACLE_BUILDER_WORKER_REGISTRATION_TOKEN=\"$(cat /oracle/secrets/registration-token)\"; fi; exec \"$@\""
        command.extend((self.profile.image, "sh", "-ec", bootstrap, "oracle-worker-entry", *worker))
        return tuple(command)

    def _run(self, command: tuple[str, ...], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return self._runner(command, check=check, capture_output=True, text=True)

    def start(self, spec: PullWorkerSpec) -> WorkerStatus:
        result = self._run(self.command_for(spec))
        container_id = result.stdout.strip() or None
        return WorkerStatus(spec.worker_id, "running", None, f"docker://{self._container_name(spec.worker_id)}", datetime.now(UTC).isoformat(), None)

    def stop(self, worker_id: str, *, timeout_seconds: float = 10.0) -> WorkerStatus:
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds cannot be negative")
        name = self._container_name(worker_id)
        self._run((self.profile.docker_executable, "stop", "--time", str(int(timeout_seconds)), name))
        return WorkerStatus(worker_id, "stopped", None, f"docker://{name}", exit_code=0)

    def status(self, worker_id: str) -> WorkerStatus | None:
        name = self._container_name(worker_id)
        result = self._run((self.profile.docker_executable, "inspect", "--format", "{{.State.Status}}", name), check=False)
        if result.returncode:
            return None
        state = result.stdout.strip()
        return WorkerStatus(worker_id, "running" if state == "running" else "stopped", None, f"docker://{name}")
