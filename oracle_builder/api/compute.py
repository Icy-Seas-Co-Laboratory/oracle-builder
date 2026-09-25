"""Compute execution API for Oracle Builder workers.

This module intentionally owns *execution* only.  Experiment planning,
artifact naming, and durable catalog state belong to an orchestration client.
Clients submit an immutable, already-resolved job specification and receive
worker capability, lifecycle, and structured execution events in return.
"""
from __future__ import annotations

import os
import platform
import signal
import subprocess
import sys
import threading
import time
import uuid
import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_STATUS_FILENAME = "training-status.json"
_MAX_TRAINING_STATUS_BYTES = 1_000_000
_BATCH_TUNE_PREFIX = "ORACLE_BATCH_TUNE_RESULT="

JobStatus = Literal["queued", "running", "paused", "succeeded", "failed", "cancelled"]
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
        # Avoid importing TensorFlow merely to report health/capabilities. When
        # NVIDIA telemetry is available, report it so the control plane can
        # make an honest admission decision before it submits work.
        if os.environ.get("ORACLE_ACCELERATOR_BACKEND", "").lower() == "cpu":
            return []
        if os.environ.get("ORACLE_ACCELERATOR_BACKEND", "").lower() == "metal":
            # Metal has no nvidia-smi equivalent.  Query TensorFlow only for
            # this explicitly selected backend so normal Linux health checks
            # remain dependency-light and do not initialize CUDA early.
            try:
                import tensorflow as tf

                devices = tf.config.list_physical_devices("GPU")
            except Exception:
                devices = []
            return ([{"id": "metal", "name": str(devices[0]), "backend": "metal", "telemetry": "tensorflow-metal"}]
                    if devices else [])
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible in {None, "", "-1"}:
            return []
        # Keep the configured visibility order stable.  It is both meaningful
        # to CUDA and makes an automatic allocation reproducible.
        selected = [item.strip() for item in visible.split(",") if item.strip()]
        selected_ids = set(selected)
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.total,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return [{"id": item, "telemetry": "unavailable"} for item in selected]
        devices: list[dict[str, Any]] = []
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4 or parts[0] not in selected_ids:
                continue
            try:
                devices.append(
                    {
                        "id": parts[0],
                        "total_memory_mib": int(parts[1]),
                        "free_memory_mib": int(parts[2]),
                        "utilization_percent": int(parts[3]),
                        "telemetry": "nvidia-smi",
                    }
                )
            except ValueError:
                continue
        return devices or [{"id": item, "telemetry": "unavailable"} for item in selected]

    @staticmethod
    def _gpu_allocation(
        resources: dict[str, Any] | None, gpus: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], list[str]]:
        """Resolve a sealed GPU allocation without changing host state.

        ``gpu_ids`` is an exact, user-selected allocation.  An explicitly
        empty list is a CPU allocation and causes the launched process to hide
        CUDA devices.  Older callers that provide only ``gpu_count`` retain
        automatic allocation, selected from the current inventory.
        """
        request = dict(resources or {})
        has_explicit_ids = "gpu_ids" in request
        raw_ids = request.get("gpu_ids", [])
        reasons: list[str] = []
        if not isinstance(raw_ids, list):
            reasons.append("GPU allocation must be a list of GPU IDs")
            raw_ids = []
        gpu_ids = [str(item).strip() for item in raw_ids]
        if any(not item for item in gpu_ids):
            reasons.append("GPU allocation cannot contain an empty GPU ID")
        if len(gpu_ids) != len(set(gpu_ids)):
            reasons.append("GPU allocation cannot contain the same GPU more than once")

        raw_count = request.get("gpu_count", len(gpu_ids) if has_explicit_ids else 0)
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            reasons.append("GPU request must be a non-negative integer")
            gpu_count = 0
        else:
            gpu_count = raw_count
        if has_explicit_ids and gpu_count != len(gpu_ids):
            reasons.append("gpu_count must equal the number of selected gpu_ids")

        inventory = {str(gpu.get("id")): gpu for gpu in gpus if gpu.get("id") is not None}
        if has_explicit_ids:
            missing = [item for item in gpu_ids if item not in inventory]
            if missing:
                reasons.append(f"Selected GPU(s) are not advertised by this worker: {', '.join(missing)}")
            allocation = {
                "mode": "explicit" if gpu_ids else "cpu",
                "gpu_ids": gpu_ids,
                "gpu_count": len(gpu_ids),
            }
            return allocation, reasons

        if gpu_count > len(gpus):
            reasons.append(f"Run requests {gpu_count} GPU(s), but this worker advertises {len(gpus)}")
        # Automatic legacy/resource-count allocations prefer the most free
        # memory when telemetry exists, then preserve worker visibility order.
        ranked = sorted(
            gpus,
            key=lambda gpu: int(gpu.get("free_memory_mib", -1)),
            reverse=True,
        )
        return {
            "mode": "automatic" if gpu_count else "unconstrained",
            "gpu_ids": [str(gpu.get("id")) for gpu in ranked[:gpu_count]],
            "gpu_count": gpu_count,
        }, reasons

    def preflight(
        self,
        *,
        action: JobAction,
        parameters: dict[str, Any],
        resources: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate admission inputs without enqueuing work.

        This does not claim a train-step memory probe when the compute host
        cannot provide one.  It returns telemetry provenance so callers can
        distinguish verified capacity from an advisory inventory and recheck
        it immediately before launch.
        """
        command = self._command(action, parameters)
        gpus = self._discover_gpus()
        allocation, reasons = self._gpu_allocation(resources, gpus)
        allocated_ids = set(allocation["gpu_ids"])
        allocated_gpus = [gpu for gpu in gpus if str(gpu.get("id")) in allocated_ids]
        telemetry_available = not allocated_gpus or all(
            gpu.get("telemetry") == "nvidia-smi" for gpu in allocated_gpus
        )
        return {
            "ready": not reasons,
            "reasons": reasons,
            "action": action,
            "requested_resources": dict(resources or {}),
            "command": command,
            "worker": self._worker.to_dict(),
            "gpus": gpus,
            "allocation": allocation,
            "vram": {
                "verified": telemetry_available,
                "kind": "inventory_only",
                "message": (
                    "Selected GPU inventory captured; launch will recheck capacity before submission."
                    if telemetry_available and allocation["gpu_ids"]
                    else "CPU allocation is sealed for this run."
                    if allocation["mode"] == "cpu"
                    else "GPU telemetry is unavailable; capacity is advisory until launch."
                ),
            },
        }

    def tune_batch_size(
        self,
        *,
        parameters: dict[str, Any],
        resources: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a short-lived, selected-device forward/backward calibration.

        This does not become a queued training job: it is a validation-time
        measurement, and its evidence is returned to the orchestrator before
        the final run TOML is sealed.
        """
        config, input_path = _required(parameters, "config", "input")
        maximum = parameters.get("maximum_batch_size", 256)
        minimum = parameters.get("minimum_batch_size", 1)
        safety_factor = parameters.get("safety_factor", 0.8)
        if (isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1
                or isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1
                or minimum > maximum
                or isinstance(safety_factor, bool) or not isinstance(safety_factor, (int, float))
                or not 0 < float(safety_factor) <= 1):
            raise ComputeRequestError("Invalid batch-size calibration bounds")
        allocation, reasons = self._gpu_allocation(resources, self._discover_gpus())
        if reasons:
            raise ComputeRequestError("; ".join(reasons))
        environment = os.environ.copy()
        if allocation["mode"] in {"explicit", "automatic"} and os.environ.get("ORACLE_ACCELERATOR_BACKEND", "").lower() != "metal":
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(allocation["gpu_ids"])
        elif allocation["mode"] == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
        command = [
            sys.executable, "-m", "oracle_builder.training.batch_tune",
            "--config", config, "--input", input_path,
            "--minimum", str(minimum), "--maximum", str(maximum),
            "--safety-factor", str(float(safety_factor)),
        ]
        with self._condition:
            if self._worker.status != "idle" or self._queue:
                raise ComputeRequestError("Compute worker is busy; wait before calibrating batch size")
            self._worker.status = "calibrating"
            self._worker.updated_at = time.time()
        try:
            completed = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=environment, timeout=900, check=False,
            )
            output = completed.stdout[-20_000:]
            line = next((item for item in reversed(output.splitlines()) if item.startswith(_BATCH_TUNE_PREFIX)), None)
            if line is None:
                return {
                    "ready": False,
                    "reasons": ["Batch calibration returned no structured result"],
                    "allocation": allocation,
                    "output": output,
                }
            try:
                result = json.loads(line.removeprefix(_BATCH_TUNE_PREFIX))
            except json.JSONDecodeError:
                result = {"ready": False, "reasons": ["Batch calibration returned malformed JSON"]}
            if not isinstance(result, dict):
                result = {"ready": False, "reasons": ["Batch calibration returned an invalid result"]}
            result["ready"] = bool(result.get("ready")) and completed.returncode == 0
            if completed.returncode and not result["ready"]:
                result.setdefault("reasons", [f"Batch calibration exited with status {completed.returncode}"])
            result["allocation"] = allocation
            result["output"] = output
            return result
        except subprocess.TimeoutExpired:
            return {"ready": False, "reasons": ["Batch calibration exceeded its 15-minute limit"], "allocation": allocation}
        finally:
            with self._condition:
                self._worker.status = "idle"
                self._worker.updated_at = time.time()
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            for job in self._jobs.values():
                if job.status in {"running", "paused"} and job.process is not None:
                    job.cancel_requested = True
                    if job.status == "paused":
                        job.process.send_signal(signal.SIGCONT)
                    job.process.terminate()
            self._condition.notify_all()
        self._thread.join(timeout=10)

    def workers(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._worker.to_dict()]

    def status(self) -> dict[str, Any]:
        with self._lock:
            counts = {status: 0 for status in ("queued", "running", "paused", "succeeded", "failed", "cancelled")}
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
        _allocation, reasons = self._gpu_allocation(resources, self._discover_gpus())
        if reasons:
            raise ComputeRequestError("; ".join(reasons))
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

    def training_status(self, job_id: str) -> dict[str, Any]:
        """Return the live snapshot only from this job's resolved output path.

        The browser supplies a job identifier, never a filesystem path. Missing
        snapshots are normal while a job is queued or during early startup.
        """
        with self._lock:
            job = self._job(job_id)
            output_path = job.output_path
            response: dict[str, Any] = {
                "job_id": job.job_id,
                "job_status": job.status,
                "available": False,
                "snapshot": None,
            }
        if job.action != "train" or not output_path:
            return response
        snapshot_path = Path(output_path) / _TRAINING_STATUS_FILENAME
        try:
            stat = snapshot_path.stat()
            if not snapshot_path.is_file() or stat.st_size > _MAX_TRAINING_STATUS_BYTES:
                return response
            loaded = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            # Atomic replacement means malformed data should be transient at
            # worst; do not turn an in-progress training run into an API error.
            return response
        if not isinstance(loaded, dict) or loaded.get("schema_version") != 1:
            return response
        response["available"] = True
        response["snapshot"] = loaded
        return response

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
                # A stopped process does not act on SIGTERM until it resumes.
                # Continue it first so cancellation is prompt and deterministic.
                if job.status == "paused":
                    job.process.send_signal(signal.SIGCONT)
                job.process.terminate()
                self._event(job, "cancellation_requested", "Termination signal sent to compute process")
            self._condition.notify_all()
            return job.to_dict()

    def pause(self, job_id: str) -> dict[str, Any]:
        """Suspend a running process without discarding its in-memory state."""
        with self._condition:
            job = self._job(job_id)
            if job.status == "paused":
                return job.to_dict()
            if job.status != "running" or job.process is None:
                raise ComputeRequestError("Only a running compute job can be paused")
            if os.name == "nt":
                raise ComputeRequestError("Pausing jobs is not supported on this compute host")
            job.process.send_signal(signal.SIGSTOP)
            job.status = "paused"
            self._worker.status = "paused"
            self._worker.updated_at = time.time()
            self._event(job, "paused", "Compute process paused")
            self._condition.notify_all()
            return job.to_dict()

    def resume(self, job_id: str) -> dict[str, Any]:
        """Continue a process previously suspended by :meth:`pause`."""
        with self._condition:
            job = self._job(job_id)
            if job.status == "running":
                return job.to_dict()
            if job.status != "paused" or job.process is None:
                raise ComputeRequestError("Only a paused compute job can be resumed")
            if os.name == "nt":
                raise ComputeRequestError("Resuming jobs is not supported on this compute host")
            job.process.send_signal(signal.SIGCONT)
            job.status = "running"
            self._worker.status = "busy"
            self._worker.updated_at = time.time()
            self._event(job, "resumed", "Compute process resumed")
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
                while (not self._queue or self._worker.status != "idle") and not self._closed:
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
            allocation, reasons = self._gpu_allocation(job.resources, self._discover_gpus())
            if reasons:
                raise ComputeRequestError("; ".join(reasons))
            environment = os.environ.copy()
            if allocation["mode"] in {"explicit", "automatic"} and os.environ.get("ORACLE_ACCELERATOR_BACKEND", "").lower() != "metal":
                environment["CUDA_VISIBLE_DEVICES"] = ",".join(allocation["gpu_ids"])
            elif allocation["mode"] == "cpu":
                environment["CUDA_VISIBLE_DEVICES"] = "-1"
            self._event(job, "allocation", "Compute allocation sealed for job", allocation)
            self._event(job, "command", "Launching Oracle Builder command", {"command": command})
            job.process = subprocess.Popen(
                command,
                cwd=cwd or None,
                env=environment,
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
