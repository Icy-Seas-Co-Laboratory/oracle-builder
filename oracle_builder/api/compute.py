"""Compute execution API for Oracle Builder workers.

This module intentionally owns *execution* only.  Experiment planning,
artifact naming, and durable catalog state belong to an orchestration client.
Clients submit an immutable, already-resolved job specification and receive
worker capability, lifecycle, and structured execution events in return.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
JobAction = Literal["train", "evaluate", "model_ingest", "run_validate", "run_pack"]


class ComputeRequestError(ValueError):
    """A client supplied a job that this worker cannot execute."""


@dataclass
class Worker:
    worker_id: str
    name: str
    capabilities: dict[str, Any]
    status: str = "idle"
    current_job_id: str | None = None
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["updated_at"] = _timestamp(self.updated_at)
        return result


@dataclass
class Job:
    job_id: str
    action: JobAction
    parameters: dict[str, Any]
    resources: dict[str, Any]
    submitted_at: float = field(default_factory=time.time)
    status: JobStatus = "queued"
    worker_id: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    error: str | None = None
    output_path: str | None = None
    cancel_requested: bool = False
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        terminal = self.status in {"succeeded", "failed", "cancelled"}
        return {
            "job_id": self.job_id,
            "action": self.action,
            "parameters": self.parameters,
            "resources": self.resources,
            "status": self.status,
            "worker_id": self.worker_id,
            "submitted_at": _timestamp(self.submitted_at),
            "started_at": _timestamp(self.started_at),
            "finished_at": _timestamp(self.finished_at),
            "exit_code": self.exit_code,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
            "result": {
                "output_path": self.output_path,
                "exit_code": self.exit_code,
            } if terminal else None,
        }


def _timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))


def _required(parameters: dict[str, Any], *keys: str) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = parameters.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ComputeRequestError(f"{key!r} is required for this compute action")
        values.append(value)
    return values


class ComputeService:
    """Threaded local compute worker with a bounded, inspectable queue.

    The queue is deliberately in-memory: the orchestrator owns durable job
    records and can reconcile or resubmit jobs after a service restart.
    """

    def __init__(self, *, max_queue_size: int = 128, worker_id: str = "local"):
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive")
        self._max_queue_size = max_queue_size
        self._jobs: dict[str, Job] = {}
        self._queue: deque[str] = deque()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._closed = False
        self._event_sequence = 0
        self._worker = Worker(
            worker_id=worker_id,
            name=platform.node() or worker_id,
            capabilities={
                "actions": ["train", "evaluate", "model_ingest", "run_validate", "run_pack"],
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpu_count": os.cpu_count(),
                "gpus": self._discover_gpus(),
            },
        )
        self._thread = threading.Thread(target=self._run, name="oracle-builder-compute", daemon=True)
        self._thread.start()

    @staticmethod
    def _discover_gpus() -> list[dict[str, Any]]:
        # Avoid importing TensorFlow merely to report health/capabilities.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible in {None, "", "-1"}:
            return []
        return [{"id": item.strip()} for item in visible.split(",") if item.strip()]

    def close(self) -> None:
        with self._condition:
            self._closed = True
            for job in self._jobs.values():
                if job.status == "running" and job.process is not None:
                    job.cancel_requested = True
                    job.process.terminate()
            self._condition.notify_all()
        self._thread.join(timeout=10)

    def workers(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._worker.to_dict()]

    def status(self) -> dict[str, Any]:
        with self._lock:
            counts = {status: 0 for status in ("queued", "running", "succeeded", "failed", "cancelled")}
            for job in self._jobs.values():
                counts[job.status] += 1
            return {
                "status": "ready" if not self._closed else "stopping",
                "queue": {"depth": len(self._queue), "capacity": self._max_queue_size},
                "jobs": counts,
                "workers": [self._worker.to_dict()],
            }

    def submit(self, *, job_id: str, action: JobAction, parameters: dict[str, Any], resources: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            uuid.UUID(job_id)
        except (ValueError, AttributeError) as exc:
            raise ComputeRequestError("job_id must be a UUID supplied by the orchestrator") from exc
        self._command(action, parameters)  # validate before accepting work
        with self._condition:
            if job_id in self._jobs:
                raise ComputeRequestError(f"job_id is already known: {job_id}")
            if len(self._queue) >= self._max_queue_size:
                raise ComputeRequestError("compute queue is full")
            job = Job(
                job_id=job_id,
                action=action,
                parameters=dict(parameters),
                resources=dict(resources or {}),
                output_path=self._expected_output(action, parameters),
            )
            self._jobs[job_id] = job
            self._event(job, "queued", "Job accepted by compute service")
            self._queue.append(job_id)
            self._condition.notify_all()
            return job.to_dict()

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return self._job(job_id).to_dict()

    def events(self, job_id: str, *, after: int = 0) -> dict[str, Any]:
        with self._lock:
            job = self._job(job_id)
            return {
                "job_id": job_id,
                "events": [event for event in job.events if event["sequence"] > after],
                "next_after": job.events[-1]["sequence"] if job.events else after,
                "status": job.status,
            }

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            job = self._job(job_id)
            if job.status in {"succeeded", "failed", "cancelled"}:
                return job.to_dict()
            job.cancel_requested = True
            if job.status == "queued":
                self._queue.remove(job_id)
                job.status = "cancelled"
                job.finished_at = time.time()
                self._event(job, "cancelled", "Job cancelled before execution")
            elif job.process is not None:
                job.process.terminate()
                self._event(job, "cancellation_requested", "Termination signal sent to compute process")
            self._condition.notify_all()
            return job.to_dict()

    def _job(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise KeyError(job_id) from exc

    def _event(self, job: Job, event_type: str, message: str, data: dict[str, Any] | None = None) -> None:
        self._event_sequence += 1
        job.events.append(
            {
                "sequence": self._event_sequence,
                "timestamp": _timestamp(time.time()),
                "type": event_type,
                "message": message,
                "data": data or {},
            }
        )

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                job = self._jobs[self._queue.popleft()]
                if job.status == "cancelled":
                    continue
                job.status = "running"
                job.started_at = time.time()
                job.worker_id = self._worker.worker_id
                self._worker.status = "busy"
                self._worker.current_job_id = job.job_id
                self._worker.updated_at = time.time()
                self._event(job, "started", "Job started", {"worker_id": job.worker_id})
            self._execute(job)
            with self._condition:
                self._worker.status = "idle"
                self._worker.current_job_id = None
                self._worker.updated_at = time.time()
                self._condition.notify_all()

    def _execute(self, job: Job) -> None:
        try:
            command = self._command(job.action, job.parameters)
            cwd = job.parameters.get("working_directory")
            if cwd is not None and not isinstance(cwd, str):
                raise ComputeRequestError("working_directory must be a path string")
            self._event(job, "command", "Launching Oracle Builder command", {"command": command})
            job.process = subprocess.Popen(
                command,
                cwd=cwd or None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert job.process.stdout is not None
            for line in job.process.stdout:
                self._event(job, "log", line.rstrip("\n"))
            job.exit_code = job.process.wait()
            job.finished_at = time.time()
            if job.cancel_requested:
                job.status = "cancelled"
                self._event(job, "cancelled", "Compute process stopped")
            elif job.exit_code == 0:
                job.status = "succeeded"
                self._event(job, "completed", "Job completed successfully", {
                    "exit_code": 0,
                    "output_path": job.output_path,
                })
            else:
                job.status = "failed"
                job.error = f"Compute process exited with status {job.exit_code}"
                self._event(job, "failed", job.error, {"exit_code": job.exit_code})
        except Exception as exc:
            job.finished_at = time.time()
            job.status = "cancelled" if job.cancel_requested else "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            self._event(job, job.status, job.error)
        finally:
            job.process = None

    @staticmethod
    def _expected_output(action: JobAction, parameters: dict[str, Any]) -> str | None:
        output = parameters.get("output")
        if not isinstance(output, str) or not output.strip():
            return None
        if action == "train" and isinstance(parameters.get("runs_dir"), str):
            return str(Path(parameters["runs_dir"]) / output)
        return output

    @staticmethod
    def _command(action: JobAction, parameters: dict[str, Any]) -> list[str]:
        if action == "train":
            if parameters.get("resume"):
                (resume,) = _required(parameters, "resume")
                command = [sys.executable, str(_PROJECT_ROOT / "model_training.py"), "--resume", resume]
            else:
                config, dataset, output = _required(parameters, "config", "input", "output")
                command = [sys.executable, str(_PROJECT_ROOT / "model_training.py"), "--config", config, "--input", dataset, "--output", output]
            if parameters.get("runs_dir"):
                command += ["--runs-dir", str(parameters["runs_dir"])]
            if parameters.get("overwrite"):
                command.append("--overwrite")
            return command
        if action == "evaluate":
            run, dataset, split, output = _required(parameters, "run", "input", "split", "output")
            return [sys.executable, str(_PROJECT_ROOT / "model_evaluate.py"), "--run", run, "--input", dataset, "--split", split, "--output", output]
        if action == "model_ingest":
            model, info, output = _required(parameters, "model", "info", "output")
            command = [sys.executable, "-m", "oracle_builder.products.cli", "ingest", "--model", model, "--info", info, "--output", output]
            if parameters.get("dataset"):
                command += ["--dataset", str(parameters["dataset"])]
            if parameters.get("no_promote"):
                command.append("--no-promote")
            return command
        if action == "run_validate":
            (run,) = _required(parameters, "run")
            return [sys.executable, "-m", "oracle_builder.artifacts.cli", "validate", run]
        if action == "run_pack":
            run, output = _required(parameters, "run", "output")
            return [sys.executable, "-m", "oracle_builder.artifacts.cli", "pack", run, output]
        raise ComputeRequestError(f"Unsupported compute action: {action}")
