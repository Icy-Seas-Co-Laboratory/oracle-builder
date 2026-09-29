"""Safe pull-worker client and execution boundary.

The control plane supplies portable artifact references and short-lived
delivery grants, never paths or commands.  This module materializes those
artifacts below a worker-owned scratch directory and gives a deliberately
narrow executor that local view.  It is intentionally stdlib-only so a
remote worker does not need the orchestrator's storage implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import Callable
import urllib.error
import urllib.request
import uuid
from typing import Any, Mapping, Protocol

from oracle_data_contracts.work_units import WorkUnit, WorkUnitError, parse_work_unit
from oracle_builder.worker.control import ExecutionInterrupted


class PullProtocolError(RuntimeError):
    """The worker could not safely complete a pull-protocol request."""


class OutputPublicationError(PullProtocolError):
    """A staged output was retained locally after resumable publication failed."""


class WorkUnitExecutor(Protocol):
    """Worker-owned implementation for one sealed work unit.

    The executor receives only locations created by :class:`PullWorkerRuntime`.
    It must not reinterpret a control-plane value as a command or path.
    """

    def execute(
        self,
        *,
        work_unit: Mapping[str, Any],
        inputs: Mapping[str, Path],
        configuration: Path | None,
        staging: Path,
        emit_event: Callable[[str, str, Mapping[str, Any] | None], None],
    ) -> Mapping[str, Any] | None: ...


class ExecutorRegistry:
    """Fixed, worker-owned action registry.

    Work units select an action name, never an executable or a command line.
    Deployments can register audited executors during process startup; they
    cannot cause this registry to import or execute a value supplied by the
    control plane.
    """

    def __init__(self, executors: Mapping[str, WorkUnitExecutor] | None = None):
        self._executors: dict[str, WorkUnitExecutor] = {}
        for action, executor in (executors or {}).items():
            self.register(action, executor)

    def register(self, action: str, executor: WorkUnitExecutor) -> None:
        if not isinstance(action, str) or not action or any(char.isspace() for char in action):
            raise ValueError("executor action must be a non-empty whitespace-free name")
        if not callable(getattr(executor, "execute", None)):
            raise TypeError("executor must implement execute")
        if action in self._executors:
            raise ValueError(f"executor already registered for action {action!r}")
        self._executors[action] = executor

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(sorted(self._executors))

    def execute(self, **kwargs: Any) -> Mapping[str, Any] | None:
        work_unit = kwargs.get("work_unit")
        action = work_unit.get("action") if isinstance(work_unit, Mapping) else None
        executor = self._executors.get(action) if isinstance(action, str) else None
        if executor is None:
            raise PullProtocolError(f"worker has no registered executor for action {action!r}")
        return executor.execute(**kwargs)


class DeterministicPackagingExecutor:
    """Built-in contract executor used for deployment checks and smoke tests.

    It deliberately does not train or invoke a subprocess.  It proves that a
    portable work unit can be materialized and published, and records only
    stable work-unit metadata in the output archive.
    """

    def execute(self, *, work_unit: Mapping[str, Any], inputs: Mapping[str, Path],
                configuration: Path | None, staging: Path,
                emit_event: Callable[[str, str, Mapping[str, Any] | None], None]) -> Mapping[str, Any]:
        manifest = {
            "work_unit_id": work_unit.get("work_unit_id"),
            "attempt_id": work_unit.get("attempt_id"),
            "action": work_unit.get("action"),
            "input_names": sorted(inputs),
            "has_configuration": configuration is not None,
        }
        (staging / "work-unit-manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8",
        )
        emit_event("packaged", "Worker produced deterministic work-unit manifest", {"inputs": len(inputs)})
        return {"output": "work-unit-manifest.json"}


class TrainingExecutor:
    """Fixed in-process adapter for the packaged Oracle training workflow.

    All paths below are worker-created materializations or staging locations;
    the WorkUnit only selects the fixed ``train`` action.  This preserves the
    existing scientific training implementation without restoring a generic
    shell-command boundary.
    """

    _BATCH_TUNE_PREFIX = "ORACLE_BATCH_TUNE_RESULT="

    @staticmethod
    def _auto_execution(work_unit: Mapping[str, Any]) -> Mapping[str, Any] | None:
        parameters = work_unit.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise PullProtocolError("train work-unit parameters must be an object")
        execution = parameters.get("queue_execution", {})
        if not isinstance(execution, Mapping):
            raise PullProtocolError("train queue execution policy must be an object")
        return execution if execution.get("mode") == "auto" else None

    def _calibrate_batch_size(
        self,
        *,
        work_unit: Mapping[str, Any],
        config: Path,
        dataset: Path,
        scratch: Path,
        emit_event: Callable[[str, str, Mapping[str, Any] | None], None],
    ) -> Path:
        """Resolve a sealed auto policy in a fresh process on this worker.

        TensorFlow can retain allocations after an OOM.  Running the existing
        bounded tuner in a child process makes the following training process
        start clean, and keeps calibration local to the worker's actual GPU.
        """
        execution = self._auto_execution(work_unit)
        if execution is None:
            return config
        minimum, maximum = execution.get("minimum_batch_size", 1), execution.get("maximum_batch_size", 256)
        if (isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1
                or isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < minimum):
            raise PullProtocolError("automatic batch calibration has invalid sealed bounds")
        target = execution.get("target_vram_fraction", {})
        if not isinstance(target, Mapping):
            raise PullProtocolError("automatic batch calibration has invalid VRAM target")
        target_min, target_max = target.get("minimum", 0.30), target.get("maximum", 0.80)
        if (isinstance(target_min, bool) or not isinstance(target_min, (int, float))
                or isinstance(target_max, bool) or not isinstance(target_max, (int, float))
                or not 0 < float(target_min) <= float(target_max) <= 1):
            raise PullProtocolError("automatic batch calibration has invalid VRAM target")

        execution_config = scratch / "resolved-training.toml"
        shutil.copyfile(config, execution_config)
        emit_event("batch_calibration_started", "Calibrating a safe batch size on this worker", {
            "minimum_batch_size": minimum, "maximum_batch_size": maximum,
        })
        try:
            completed = subprocess.run(
                [
                    sys.executable, "-m", "oracle_builder.training.batch_tune",
                    "--config", str(execution_config), "--input", str(dataset),
                    "--minimum", str(minimum), "--maximum", str(maximum),
                    "--target-vram-min", str(float(target_min)), "--target-vram-max", str(float(target_max)),
                ],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PullProtocolError("automatic batch calibration exceeded its 15-minute limit") from exc
        output = completed.stdout[-20_000:]
        result_line = next((line for line in reversed(output.splitlines()) if line.startswith(self._BATCH_TUNE_PREFIX)), None)
        if result_line is None:
            raise PullProtocolError("automatic batch calibration returned no structured result")
        try:
            result = json.loads(result_line.removeprefix(self._BATCH_TUNE_PREFIX))
        except json.JSONDecodeError as exc:
            raise PullProtocolError("automatic batch calibration returned malformed JSON") from exc
        if not isinstance(result, Mapping):
            raise PullProtocolError("automatic batch calibration returned an invalid result")
        recommended = result.get("recommended_batch_size")
        if completed.returncode != 0 or not result.get("ready") or isinstance(recommended, bool) or not isinstance(recommended, int):
            reasons = result.get("reasons")
            detail = "; ".join(str(item) for item in reasons) if isinstance(reasons, list) else "no safe batch size was found"
            raise PullProtocolError(f"automatic batch calibration failed: {detail}")
        if recommended < minimum or recommended > maximum:
            raise PullProtocolError("automatic batch calibration returned a batch size outside its sealed bounds")

        from oracle_builder.config import load_toml
        import tomli_w

        tuned = load_toml(execution_config)
        tuned.setdefault("data", {})["batch_size"] = recommended
        execution_config.write_text(tomli_w.dumps(tuned), encoding="utf-8")
        # Keep the event compact and durable: detailed raw probe logs remain
        # worker diagnostics, while the decision evidence is visible in the
        # control plane and artifact provenance.
        emit_event("batch_size_resolved", "Worker selected a safe batch size", {
            "batch_size": recommended,
            "largest_verified_batch_size": result.get("largest_verified_batch_size"),
            "tuning_strategy": result.get("tuning_strategy"),
            "probe_kind": result.get("probe_kind"),
        })
        return execution_config

    def execute(self, *, work_unit: Mapping[str, Any], inputs: Mapping[str, Path],
                configuration: Path | None, staging: Path,
                emit_event: Callable[[str, str, Mapping[str, Any] | None], None]) -> Mapping[str, Any]:
        dataset = inputs.get("input")
        if dataset is None or not (dataset / "payload").is_file():
            raise PullProtocolError("train executor requires a materialized dataset payload")
        if configuration is None or not (configuration / "payload").is_file():
            raise PullProtocolError("train executor requires a materialized configuration payload")
        execution = dict(work_unit.get("parameters", {}).get("queue_execution", {}))
        if execution.get("phase") == "verify":
            # Resolve the dataset/configuration and exercise a forward/backward
            # probe in an isolated child process. Never enter run_training.
            from oracle_builder.config import load_toml
            requested = load_toml(configuration / "payload").get("data", {}).get("batch_size", 16)
            probe = dict(execution)
            if probe.get("mode") != "auto":
                probe.update(mode="auto", minimum_batch_size=requested, maximum_batch_size=requested)
            emit_event("preflight_started", "Checking dataset, model, and batch capacity without training", {})
            resolved = self._calibrate_batch_size(
                work_unit={**work_unit, "parameters": {"queue_execution": probe}},
                config=configuration / "payload", dataset=dataset / "payload",
                scratch=staging.parent, emit_event=emit_event,
            )
            batch_size = load_toml(resolved)["data"]["batch_size"]
            from oracle_builder.config import resolve_config
            from oracle_data_contracts.artifacts.splits import create_split_manifest
            config = resolve_config(resolved, dataset / 'payload', staging)
            from oracle_builder.config import self_supervised_settings
            if self_supervised_settings(config).get('enabled', False):
                raise PullProtocolError('Resumable queued execution does not yet support self-supervised training; use the explicit whole-phase training workflow')
            split = create_split_manifest(staging, dataset / 'payload', config)
            stratified = config.get('classification', {}).get('stratification', {})
            unit_epochs = int(stratified.get('supra_epochs', 1)) if stratified.get('enabled') else 1
            training_policy = config.get('training', {})
            unit_steps = int(training_policy.get('work_unit_steps', 0)) if training_policy.get('step_cursor_policy') == 'ordered_batches_v1' else 0
            report = {"ready": True, "batch_size": batch_size,
                      "checks": ["configuration_resolution", "dataset_loading", "model_forward_backward", "split_coverage"],
                      "execution_contract_version": 2, "unit_epochs": unit_epochs,
                      "unit_steps": unit_steps,
                      "split_manifest": {key: value for key,value in split.items() if key != 'assignments'},
                      "work_unit_id": work_unit.get("work_unit_id")}
            (staging / "verification.json").write_text(json.dumps(report), encoding="utf-8")
            emit_event("preflight_passed", "Verification passed; training has not started", report)
            return {"artifact_root": ".", "verification": True}
        if execution.get("mode") == "verified":
            from oracle_builder.config import load_toml
            import tomli_w
            effective = load_toml(configuration / "payload")
            effective.setdefault("data", {})["batch_size"] = execution["batch_size"]
            local_configuration = staging.parent / "verified-configuration"
            local_configuration.mkdir()
            (local_configuration / "payload").write_text(tomli_w.dumps(effective), encoding="utf-8")
            configuration = local_configuration
        if execution.get('revalidate_fixed_batch') is True and work_unit.get('phase') == 'train':
            # A portable attempt must fit the already verified scientific batch
            # on its actual worker. Never retune or change the training batch.
            batch_size = execution['batch_size']
            probe = {**execution, 'mode': 'auto', 'minimum_batch_size': batch_size, 'maximum_batch_size': batch_size}
            self._calibrate_batch_size(
                work_unit={**work_unit, 'parameters': {'queue_execution': probe}},
                config=configuration / 'payload', dataset=dataset / 'payload',
                scratch=staging.parent, emit_event=emit_event,
            )
            emit_event('resume_capacity_verified', 'Worker verified the frozen training batch', {'batch_size': batch_size})
        run_name = "run"
        emit_event("training_starting", "Starting packaged training action", {"action": "train"})
        # This is an in-process typed boundary: the worker never mutates
        # process-global argv and the lease cannot select an executable or a
        # destination.  ``TrainingRequest`` is also what future API callers
        # use when they need the scientific workflow without a CLI.
        from oracle_builder.training.api import TrainingRequest
        from oracle_builder.training.workflow import run_training

        # The training callback writes a rich local snapshot.  Project a
        # bounded subset through durable lease events while it runs; the
        # dashboard then works for a remote pull worker without reaching into
        # worker storage or restoring the retired push-service status API.
        execution_config = self._calibrate_batch_size(
            work_unit=work_unit, config=configuration / "payload", dataset=dataset / "payload",
            scratch=staging.parent, emit_event=emit_event,
        )
        reporter = _TrainingProgressReporter(staging / run_name / "training-status.json", emit_event)
        reporter.start()
        workflow_started = time.perf_counter()
        segment = work_unit.get('parameters', {}).get('execution_segment')
        checkpoint = inputs.get('checkpoint')
        request_options = {}
        if segment:
            if checkpoint:
                shutil.copytree(checkpoint, staging / run_name)
                # A prior segment's result is provenance, not this attempt's
                # completion report. The workflow writes a fresh one.
                request_options['resume'] = str(staging / run_name)
            else:
                request_options.update(config=str(execution_config), output=run_name)
                pinned = inputs.get('split_manifest')
                if pinned:
                    request_options['split_manifest'] = str(pinned / 'protocol' / 'splits.json')
            if segment['phase'] == 'finalize':
                request_options['finalize_only'] = True
            else:
                request_options.update(segment_start_epoch=segment['start_epoch'], segment_stop_epoch=segment['stop_epoch'])
                if segment.get('max_steps'):
                    request_options.update(segment_max_steps=segment['max_steps'], segment_start_step=segment.get('start_step', 0))
            request_options['control_file'] = getattr(self, 'control_file', None)
        else:
            request_options.update(config=str(execution_config), output=run_name)
        try:
            result = run_training(
                TrainingRequest(
                    input=str(dataset / "payload"),
                    runs_dir=str(staging),
                    **request_options,
                )
            )
        except BaseException:
            emit_event("stage_timing", "Training workflow failed", {
                "schema": "oracle_timing_v1", "stage": "training_workflow",
                "duration_seconds": round(time.perf_counter() - workflow_started, 3), "outcome": "failed",
            })
            raise
        finally:
            reporter.stop()
        emit_event("stage_timing", "Training workflow completed", {
            "schema": "oracle_timing_v1", "stage": "training_workflow",
            "duration_seconds": round(time.perf_counter() - workflow_started, 3), "outcome": "succeeded",
        })
        if result not in {None, 0}:
            raise PullProtocolError(f"training action exited with status {result}")
        run_dir = staging / run_name
        if not run_dir.is_dir():
            raise PullProtocolError("training action did not create a run artifact")
        # A staging archive represents one artifact root, not a wrapper folder.
        for child in run_dir.iterdir():
            destination = staging / child.name
            if destination.exists():
                raise PullProtocolError(f"training output conflicts with staging entry {child.name!r}")
            child.replace(destination)
        run_dir.rmdir()
        # Runtime paths are worker-local recovery implementation details. Do
        # not publish them inside the portable artifact.
        runtime = staging / "provenance" / "runtime.json"
        if runtime.is_file():
            value = json.loads(runtime.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                from oracle_builder.artifacts import reopen_run_artifact, seal_run_artifact
                # The training workflow already sealed its local artifact.
                # Sanitizing provenance changes covered bytes, so complete
                # this final packaging transition before immutable publication.
                reopen_run_artifact(staging, reason="Remove worker-local recovery paths before publication")
                value["paths"] = {"input_artifact": "materialized"}
                runtime.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                seal_run_artifact(staging)
        emit_event("training_complete", "Packaged training action completed", {"action": "train"})
        return {"artifact_root": "."}


def _finite_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


def _numeric_fields(value: Any, *, limit: int = 24) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int | float] = {}
    for key, item in value.items():
        numeric = _finite_number(item)
        if numeric is not None and isinstance(key, str) and len(key) <= 100:
            result[key] = numeric
        if len(result) >= limit:
            break
    return result


def training_progress_payload(status_path: Path) -> dict[str, Any] | None:
    """Return a small, path-free projection of a worker-local status file.

    The full snapshot is useful to the worker but can contain a long batch
    trace.  Lease events are durable control-plane records, so cap both the
    dimensions and number of points here.
    """
    try:
        raw = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, Mapping):
        return None
    progress = raw.get("progress")
    timing = raw.get("timing")
    metrics = raw.get("metrics")
    current = metrics.get("current_batch") if isinstance(metrics, Mapping) else None
    completed = metrics.get("last_completed_epoch") if isinstance(metrics, Mapping) else None
    history = raw.get("history")
    trace = metrics.get("current_epoch_history") if isinstance(metrics, Mapping) else None
    bounded_trace = [_numeric_fields(item, limit=8) for item in trace[-60:]] if isinstance(trace, list) else []
    bounded_history = [_numeric_fields(item, limit=24) for item in history[-50:]] if isinstance(history, list) else []
    return {
        "updated_at": raw.get("updated_at") if isinstance(raw.get("updated_at"), str) else None,
        "phase": raw.get("phase") if isinstance(raw.get("phase"), str) else None,
        "message": raw.get("message") if isinstance(raw.get("message"), str) else None,
        "progress": _numeric_fields(progress, limit=12),
        "timing": _numeric_fields(timing, limit=12),
        "metrics": _numeric_fields(current, limit=24),
        "completed_metrics": _numeric_fields(completed, limit=24),
        "trace": bounded_trace,
        "history": bounded_history,
    }


class _TrainingProgressReporter:
    """Best-effort, bounded status projection for one training work unit."""

    def __init__(self, status_path: Path, emit_event: Callable[[str, str, Mapping[str, Any] | None], None]):
        self.status_path, self.emit_event = status_path, emit_event
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="oracle-training-progress", daemon=True)
        self.last_signature = ""

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3)
        self._emit_if_changed()

    def _emit_if_changed(self) -> None:
        snapshot = training_progress_payload(self.status_path)
        if snapshot is None:
            return
        signature = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        if signature == self.last_signature:
            return
        self.last_signature = signature
        try:
            self.emit_event("training_progress", snapshot.get("message") or "Training progress updated", snapshot)
        except Exception:
            # Progress is observational. The lease renewal/completion protocol
            # remains authoritative when a transient event delivery fails.
            pass

    def _run(self) -> None:
        # A ten-second cadence keeps the dashboard responsive without turning
        # a multi-hour job into an unbounded high-frequency event stream.
        while not self.stop_event.wait(10.0):
            self._emit_if_changed()


class InferenceExecutor:
    """Fixed adapter for a sealed-model, frozen-dataset batch inference lease.

    The executor accepts exactly the portable roles selected by the infer
    contract.  It does not accept an output path or an executable from the
    control plane: every output is sealed below the worker-owned staging
    directory and published by :class:`PullWorkerRuntime`.
    """

    _SPLITS = {"all", "train", "validation", "test"}

    def execute(self, *, work_unit: Mapping[str, Any], inputs: Mapping[str, Path],
                configuration: Path | None, staging: Path,
                emit_event: Callable[[str, str, Mapping[str, Any] | None], None]) -> Mapping[str, Any]:
        parameters = work_unit.get('parameters', {})
        if parameters.get('merge_item_ids'):
            from oracle_builder.inference.workflow import merge_inference_shards
            expected_shards = parameters.get('shard_ids', [])
            if not expected_shards or set(inputs) - {'model', 'input'} != {'shard_' + name for name in expected_shards}:
                raise PullProtocolError('Inference merge requires its exact sealed shard set')
            manifest = merge_inference_shards([inputs['shard_' + name] for name in expected_shards], staging,
                expected_item_ids=parameters['merge_item_ids'])
            return {'artifact_root': '.', 'artifact_id': manifest['artifact_id']}
        model = inputs.get("model")
        dataset = inputs.get("input")
        if model is None or not (model / "artifact.json").is_file():
            raise PullProtocolError("infer executor requires a materialized sealed model artifact")
        if dataset is None or not (dataset / "payload").is_file():
            raise PullProtocolError("infer executor requires a materialized frozen dataset payload")
        if configuration is not None:
            raise PullProtocolError("infer executor does not accept a separate configuration artifact")
        parameters = work_unit.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise PullProtocolError("infer work-unit parameters must be an object")
        unknown = set(parameters) - {"split", "prediction_set", "shard_size", "shard_id", "shard_item_ids"}
        if unknown:
            raise PullProtocolError("infer work-unit contains unsupported parameters")
        split = parameters.get("split", "all")
        prediction_set = parameters.get("prediction_set")
        if not isinstance(split, str) or split not in self._SPLITS:
            raise PullProtocolError("infer split must be all, train, validation, or test")
        if prediction_set is not None and (not isinstance(prediction_set, str) or not prediction_set.strip() or len(prediction_set) > 200):
            raise PullProtocolError("infer prediction_set must be a non-empty label of at most 200 characters")
        raw_inputs = work_unit.get("inputs")
        if not isinstance(raw_inputs, Mapping):
            raise PullProtocolError("infer work-unit inputs must be an object")
        from oracle_builder.inference.workflow import InferenceRequest, run_inference
        emit_event("inference_starting", "Starting sealed batch inference", {"action": "infer", "split": split})
        manifest = run_inference(InferenceRequest(
            model_run=str(model), input=str(dataset / "payload"), output_dir=str(staging),
            split=split, prediction_set=prediction_set,
            shard_id=parameters.get('shard_id'), shard_item_ids=tuple(parameters['shard_item_ids']) if parameters.get('shard_item_ids') is not None else None,
            lineage={"work_unit_id": work_unit.get("work_unit_id"), "attempt_id": work_unit.get("attempt_id")},
            model_reference=raw_inputs.get("model") if isinstance(raw_inputs.get("model"), Mapping) else None,
            input_reference=raw_inputs.get("input") if isinstance(raw_inputs.get("input"), Mapping) else None,
        ), progress=emit_event)
        emit_event("inference_complete", "Sealed batch inference completed", {"artifact_id": manifest["artifact_id"], "records": manifest["outputs"]["records"]})
        return {"artifact_root": ".", "artifact_id": manifest["artifact_id"], "records": manifest["outputs"]["records"]}


@dataclass(frozen=True, slots=True)
class MaterializedWorkUnit:
    """The sole filesystem view an executor receives for a leased unit."""

    scratch_dir: Path
    inputs: Mapping[str, Path]
    configuration: Path | None
    staging: Path


class OrchestratorPullClient:
    """HTTP client for a registered worker, using only the Python stdlib.

    Control-plane requests should fail quickly so a worker can recover a
    lease.  Artifact delivery is deliberately different: the API validates
    and prepares an archive before it can send response headers, and a frozen
    dataset may be many gigabytes.  Keep those two deadlines separate.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 15.0,
        artifact_timeout_seconds: float = 900.0,
    ):
        base = base_url.strip().rstrip("/")
        if not base.startswith(("http://", "https://")):
            raise ValueError("orchestrator URL must start with http:// or https://")
        if timeout_seconds <= 0 or artifact_timeout_seconds <= 0:
            raise ValueError("worker request timeouts must be positive")
        self.base_url = base
        self.timeout_seconds = timeout_seconds
        self.artifact_timeout_seconds = artifact_timeout_seconds

    def _request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
        *,
        worker_token: str | None = None,
        allow_empty: bool = False,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        request = urllib.request.Request(self.base_url + path, method=method)
        request.add_header("Accept", "application/json")
        if worker_token:
            request.add_header("Authorization", f"Bearer {worker_token}")
        if body is not None:
            request.data = json.dumps(dict(body), sort_keys=True, separators=(",", ":")).encode("utf-8")
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds or self.timeout_seconds) as response:
                content = response.read()
                if not content and allow_empty:
                    return None
                decoded = json.loads(content.decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise PullProtocolError("orchestrator returned a non-object response")
                return decoded
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            error = PullProtocolError(f"orchestrator returned {exc.code}: {detail}")
            error.status_code = exc.code
            raise error from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise PullProtocolError(f"could not communicate with orchestrator: {exc}") from exc

    def _binary_request(
        self, method: str, path: str, *, worker_token: str, lease_token: str,
        artifact_token: str | None = None, body: bytes | None = None,
        content_type: str | None = None,
    ) -> bytes:
        request = urllib.request.Request(self.base_url + path, data=body, method=method)
        request.add_header("Accept", "application/x-tar, application/octet-stream")
        request.add_header("Authorization", f"Bearer {worker_token}")
        request.add_header("X-Oracle-Lease-Token", lease_token)
        if artifact_token:
            request.add_header("X-Oracle-Artifact-Grant", artifact_token)
        if content_type:
            request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(request, timeout=self.artifact_timeout_seconds) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise PullProtocolError(f"orchestrator returned {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise PullProtocolError(f"could not communicate with orchestrator: {exc}") from exc

    def register(
        self,
        *,
        pool_id: str,
        name: str,
        registration_token: str,
        endpoint: str | None = None,
        capabilities: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pool_id": pool_id,
            "name": name,
            "registration_token": registration_token,
            "capabilities": dict(capabilities or {}),
        }
        if endpoint:
            payload["endpoint"] = endpoint
        return self._request("POST", "/v1/workers:register", payload) or {}

    def poll(self, *, worker_id: str, worker_token: str, ttl_seconds: int = 60, capabilities: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        return self._request(
            "POST", f"/v1/workers/{worker_id}:lease", {"ttl_seconds": ttl_seconds, **({"capabilities": dict(capabilities)} if capabilities is not None else {})},
            worker_token=worker_token, allow_empty=True,
        )

    def renew(self, *, lease_id: str, worker_token: str, lease_token: str, ttl_seconds: int = 60) -> dict[str, Any]:
        return self._request(
            "POST", f"/v1/worker-leases/{lease_id}:renew",
            {"lease_token": lease_token, "ttl_seconds": ttl_seconds}, worker_token=worker_token,
        ) or {}

    def heartbeat(self, *, lease_id, worker_token, lease_token, ttl_seconds=30, telemetry=None):
        return self._request('POST', f'/v1/worker-leases/{lease_id}:heartbeat',
            {'lease_token': lease_token, 'ttl_seconds': ttl_seconds, 'telemetry': telemetry or {}}, worker_token=worker_token,
            timeout_seconds=min(5, ttl_seconds / 4)) or {}

    def commands(self, *, lease_id, worker_token, lease_token, wait_seconds=25):
        return self._request('POST', f'/v1/worker-leases/{lease_id}/commands:poll',
            {'lease_token': lease_token, 'wait_seconds': wait_seconds}, worker_token=worker_token,
            timeout_seconds=wait_seconds + 5) or {}

    def acknowledge_command(self, *, lease_id, worker_token, lease_token, command_id, status, result=None):
        return self._request('POST', f'/v1/worker-leases/{lease_id}/commands:acknowledge',
            {'lease_token': lease_token, 'command_id': command_id, 'status': status, 'result': result or {}}, worker_token=worker_token) or {}

    def release(self, *, lease_id: str, worker_token: str, lease_token: str, outcome: str = "released") -> dict[str, Any]:
        return self._request(
            "POST", f"/v1/worker-leases/{lease_id}:release",
            {"lease_token": lease_token, "outcome": outcome}, worker_token=worker_token,
        ) or {}

    def acknowledge(self, *, lease_id: str, worker_token: str, lease_token: str) -> dict[str, Any]:
        return self._request(
            "POST", f"/v1/worker-leases/{lease_id}:acknowledge", {"lease_token": lease_token},
            worker_token=worker_token,
        ) or {}

    def events(
        self, *, lease_id: str, worker_token: str, lease_token: str,
        events: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        # Sending a bounded batch lets a disconnected worker retry a single
        # durable request without exposing a streaming control channel.
        normalized: list[dict[str, Any]] = []
        for event in events:
            item = dict(event)
            # Event IDs make retries idempotent at the durable receiver.  An
            # executor may bring its own ID only when retrying the same event.
            item.setdefault("event_id", str(uuid.uuid4()))
            normalized.append(item)
        return self._request(
            "POST", f"/v1/worker-leases/{lease_id}:events",
            {"lease_token": lease_token, "events": normalized},
            worker_token=worker_token,
        ) or {}

    def complete(
        self, *, lease_id: str, worker_token: str, lease_token: str,
        outcome: str, error: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"lease_token": lease_token, "outcome": outcome}
        if error:
            body["error"] = error
        return self._request(
            "POST", f"/v1/worker-leases/{lease_id}:complete", body, worker_token=worker_token,
        ) or {}

    def defer_output_publication(self, *, lease_id: str, worker_token: str, lease_token: str) -> dict[str, Any]:
        return self._request(
            "POST", f"/v1/worker-leases/{lease_id}:defer-output", {"lease_token": lease_token},
            worker_token=worker_token,
        ) or {}

    def resume_output_publication(self, *, lease_id: str, worker_id: str, worker_token: str,
                                  ttl_seconds: int = 60) -> dict[str, Any]:
        return self._request(
            "POST", f"/v1/worker-leases/{lease_id}:resume-output",
            {"worker_id": worker_id, "ttl_seconds": ttl_seconds}, worker_token=worker_token,
        ) or {}

    def cancellation(self, *, lease_id: str, worker_token: str, lease_token: str) -> bool:
        response = self._request(
            "POST", f"/v1/worker-leases/{lease_id}/cancellation", {"lease_token": lease_token},
            worker_token=worker_token,
        ) or {}
        requested = response.get("cancel_requested")
        if not isinstance(requested, bool):
            raise PullProtocolError("cancellation response is invalid")
        return requested

    def cancelled(self, *, lease_id: str, worker_token: str, lease_token: str,
                  message: str = "Worker acknowledged cancellation") -> dict[str, Any]:
        return self._request(
            "POST", f"/v1/worker-leases/{lease_id}:cancelled",
            {"lease_token": lease_token, "message": message[:2000]}, worker_token=worker_token,
        ) or {}

    def artifact_grant(
        self, *, lease_id: str, worker_token: str, lease_token: str,
        ref: Mapping[str, Any], ttl_seconds: int = 600,
    ) -> dict[str, Any]:
        """Ask for a one-time delivery grant for a portable artifact ref."""
        return self._request(
            "POST", f"/v1/worker-leases/{lease_id}:artifact-grants",
            {"lease_token": lease_token, "ref": dict(ref), "ttl_seconds": ttl_seconds},
            worker_token=worker_token,
        ) or {}

    def download_artifact(
        self, *, worker_token: str, lease_token: str,
        download_path: str, artifact_token: str,
    ) -> bytes:
        """Fetch using the server-provided path; never derive it from a ref."""
        if not isinstance(download_path, str) or not download_path.startswith("/v1/") or "?" in download_path or "#" in download_path:
            raise PullProtocolError("artifact delivery response has an unsafe download path")
        return self._binary_request(
            "GET", download_path,
            worker_token=worker_token, lease_token=lease_token, artifact_token=artifact_token,
        )

    def publish_staging(
        self, *, lease_id: str, worker_token: str, lease_token: str,
        archive: bytes,
    ) -> None:
        self._binary_request(
            "PUT", f"/v1/worker-leases/{lease_id}/staging", worker_token=worker_token,
            lease_token=lease_token, body=archive, content_type="application/x-tar",
        )

    def start_output_upload(
        self, *, lease_id: str, worker_token: str, lease_token: str,
        archive_size: int, archive_sha256: str, part_size: int,
    ) -> dict[str, Any]:
        return self._request(
            "POST", f"/v1/worker-leases/{lease_id}/output-uploads",
            {"lease_token": lease_token, "archive_size": archive_size,
             "archive_sha256": archive_sha256, "part_size": part_size},
            worker_token=worker_token,
        ) or {}

    def upload_output_part(
        self, *, upload_id: str, part_number: int, data: bytes, archive_size: int,
        offset: int, part_sha256: str, worker_token: str, lease_token: str,
    ) -> dict[str, Any]:
        if not upload_id or part_number < 0 or offset < 0 or not data:
            raise PullProtocolError("output upload part arguments are invalid")
        request = urllib.request.Request(
            self.base_url + f"/v1/worker-output-uploads/{upload_id}/parts/{part_number}",
            data=data, method="PUT",
        )
        request.add_header("Authorization", f"Bearer {worker_token}")
        request.add_header("X-Oracle-Lease-Token", lease_token)
        request.add_header("X-Oracle-Part-SHA256", part_sha256)
        request.add_header("Content-Type", "application/octet-stream")
        request.add_header("Content-Range", f"bytes {offset}-{offset + len(data) - 1}/{archive_size}")
        try:
            with urllib.request.urlopen(request, timeout=self.artifact_timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise PullProtocolError("output upload response is not an object")
                return decoded
        except urllib.error.HTTPError as exc:
            raise PullProtocolError(f"orchestrator returned {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise PullProtocolError(f"could not upload output part: {exc}") from exc

    def finalize_output_upload(self, *, upload_id: str, worker_token: str, lease_token: str) -> dict[str, Any]:
        return self._request(
            "POST", f"/v1/worker-output-uploads/{upload_id}:finalize", {"lease_token": lease_token},
            worker_token=worker_token,
        ) or {}


def _grant_value(grant: Mapping[str, Any], key: str) -> str:
    value = grant.get(key)
    if not isinstance(value, str) or not value:
        raise PullProtocolError(f"artifact delivery grant is missing {key}")
    return value


def _safe_member_path(destination: Path, member_name: str) -> Path:
    # Tar paths are POSIX paths even on Windows.  Resolve both sides to reject
    # absolute and parent traversal before tarfile writes anything.
    candidate = (destination / member_name).resolve()
    try:
        candidate.relative_to(destination.resolve())
    except ValueError as exc:
        raise PullProtocolError("artifact archive contains a path outside its destination") from exc
    return candidate


def extract_tar_safely(archive: bytes, destination: Path) -> None:
    """Extract a normal-file/directory tar without links or device entries."""
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tar:
            for member in tar.getmembers():
                _safe_member_path(destination, member.name)
                if not (member.isdir() or member.isfile()):
                    raise PullProtocolError("artifact archive contains a link or special file")
            # Members were fully validated before extraction (not use
            # extractall's path filtering, which differs by Python version).
            for member in tar.getmembers():
                target = _safe_member_path(destination, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = tar.extractfile(member)
                    if source is None:
                        raise PullProtocolError("artifact archive member could not be read")
                    with source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
    except PullProtocolError:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    except (tarfile.TarError, OSError) as exc:
        shutil.rmtree(destination, ignore_errors=True)
        raise PullProtocolError(f"artifact archive could not be extracted: {exc}") from exc


def archive_directory_to_file(source: Path, destination: Path) -> tuple[int, str]:
    """Write a normal-file-only tar to disk and return ``(size, sha256)``.

    This is the production publication path.  Unlike :func:`archive_directory`
    (a compatibility helper for old test clients), it never holds an archive
    in memory.
    """
    if not source.is_dir():
        raise PullProtocolError("staging directory does not exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, mode="w", dereference=True) as tar:
        for path in sorted(source.rglob("*")):
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise PullProtocolError("staging output contains a symlink or special file")
            tar.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)
    digest = hashlib.sha256()
    size = 0
    with destination.open("rb") as archive:
        while chunk := archive.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    if not size:
        raise PullProtocolError("staging output archive is empty")
    return size, digest.hexdigest()


def _accepted_parts(value: Mapping[str, Any]) -> set[int]:
    raw = value.get("uploaded_parts", [])
    if not isinstance(raw, list) or any(not isinstance(part, int) or part < 0 for part in raw):
        raise PullProtocolError("output upload session has invalid uploaded_parts")
    return set(raw)


def upload_output_archive(
    client: Any, *, archive_path: Path, lease_id: str, worker_token: str, lease_token: str,
    part_size: int = 8 * 1024 * 1024, retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    check_cancel: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Resume a bounded multipart upload from a disk archive.

    Each request owns at most ``part_size`` bytes.  Re-starting this function
    asks the control plane for the existing session and skips accepted parts.
    """
    if part_size < 64 * 1024 or part_size > 64 * 1024 * 1024:
        raise ValueError("part_size must be between 64 KiB and 64 MiB")
    if retries < 0 or not archive_path.is_file():
        raise ValueError("retries must be non-negative and archive_path must be a file")
    size = archive_path.stat().st_size
    digest = hashlib.sha256()
    with archive_path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            if check_cancel:
                check_cancel()
            digest.update(chunk)
    try:
        session = client.start_output_upload(
            lease_id=lease_id, worker_token=worker_token, lease_token=lease_token,
            archive_size=size, archive_sha256=digest.hexdigest(), part_size=part_size,
        )
    except Exception as exc:
        raise OutputPublicationError(f"output upload session could not be created: {exc}") from exc
    try:
        upload_id = session.get("upload_id") if isinstance(session, Mapping) else None
        server_part_size = session.get("part_size") if isinstance(session, Mapping) else None
        if not isinstance(upload_id, str) or not upload_id or not isinstance(server_part_size, int) or server_part_size < 64 * 1024 or server_part_size > 64 * 1024 * 1024:
            raise PullProtocolError("output upload session is invalid")
        accepted = _accepted_parts(session)
    except PullProtocolError as exc:
        raise OutputPublicationError(str(exc)) from exc
    with archive_path.open("rb") as source:
        part_number = 0
        offset = 0
        while True:
            if check_cancel:
                check_cancel()
            data = source.read(server_part_size)
            if not data:
                break
            if part_number not in accepted:
                part_digest = hashlib.sha256(data).hexdigest()
                for attempt in range(retries + 1):
                    if check_cancel:
                        check_cancel()
                    try:
                        client.upload_output_part(
                            upload_id=upload_id, part_number=part_number, data=data, archive_size=size,
                            offset=offset, part_sha256=part_digest, worker_token=worker_token, lease_token=lease_token,
                        )
                        break
                    except Exception as exc:
                        if attempt >= retries:
                            raise OutputPublicationError(f"output upload part {part_number} failed after {retries + 1} attempts: {exc}") from exc
                        sleep(min(2.0, 0.1 * (2 ** attempt)))
            offset += len(data)
            part_number += 1
    try:
        if check_cancel:
            check_cancel()
        return client.finalize_output_upload(upload_id=upload_id, worker_token=worker_token, lease_token=lease_token)
    except ExecutionInterrupted:
        raise
    except Exception as exc:
        raise OutputPublicationError(f"output upload finalization failed: {exc}") from exc


def recover_output_upload(
    client: Any, *, recovery_dir: str | Path, worker_token: str, lease_token: str,
    retries: int = 3, sleep: Callable[[float], None] = time.sleep,
    remove_on_success: bool = True,
) -> dict[str, Any]:
    """Resume a retained output upload; callers supply fresh worker/lease credentials.

    Recovery metadata deliberately contains only a lease id, checksums and
    filenames relative to ``recovery_dir``—never tokens or external paths.
    """
    root = Path(recovery_dir).resolve()
    metadata_path = root / "recovery.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise PullProtocolError("output recovery metadata is missing")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        lease_id, archive_name, part_size = metadata["lease_id"], metadata["archive"], metadata["part_size"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise PullProtocolError("output recovery metadata is invalid") from exc
    if not isinstance(lease_id, str) or not isinstance(archive_name, str) or Path(archive_name).name != archive_name or not isinstance(part_size, int):
        raise PullProtocolError("output recovery metadata contains unsafe values")
    result = upload_output_archive(client, archive_path=root / archive_name, lease_id=lease_id,
                                   worker_token=worker_token, lease_token=lease_token,
                                   part_size=part_size, retries=retries, sleep=sleep)
    if remove_on_success:
        shutil.rmtree(root)
    return result


class PullWorkerRuntime:
    """One acknowledged lease from materialization through staged publication."""

    def __init__(self, client: OrchestratorPullClient, *, scratch_root: str | Path, executor: WorkUnitExecutor, supervised: bool = False):
        self.client = client
        self.scratch_root = Path(scratch_root).expanduser().resolve()
        self.executor = executor
        self.supervised = supervised
        self.artifact_cache = None
        if supervised:
            from oracle_builder.worker.cache import WorkerArtifactCache
            self.artifact_cache = WorkerArtifactCache(self.scratch_root / 'artifact-cache')

    def _retain_output_recovery(self, *, scratch: Path, lease_id: str, archive_name: str,
                                archive_size: int, archive_sha256: str, part_size: int) -> Path:
        """Move failed publication material below a worker-owned recovery root."""
        recovery_root = self.scratch_root / "recovery"
        recovery_root.mkdir(parents=True, exist_ok=True)
        destination = recovery_root / uuid.uuid4().hex
        metadata = {
            "version": 1, "lease_id": lease_id, "archive": archive_name,
            "archive_size": archive_size, "archive_sha256": archive_sha256, "part_size": part_size,
        }
        metadata_path = scratch / "recovery.json"
        fd = os.open(metadata_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(metadata, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
        os.chmod(metadata_path, 0o600)
        shutil.move(str(scratch), str(destination))
        return destination

    def _materialize_ref(self, ref: Mapping[str, Any], *, lease_id: str, worker_token: str, lease_token: str, destination: Path) -> Path:
        delivery = self.client.artifact_grant(
            lease_id=lease_id, worker_token=worker_token, lease_token=lease_token, ref=ref,
        )
        grant = delivery.get("grant")
        if not isinstance(grant, Mapping):
            raise PullProtocolError("artifact delivery response is missing a grant")
        if self.artifact_cache and grant.get('content_manifest_sha256'):
            from oracle_data_contracts.artifacts import ArtifactRef
            def fetch(temporary):
                archive = self.client.download_artifact(worker_token=worker_token, lease_token=lease_token,
                    download_path=_grant_value(delivery, 'download_path'), artifact_token=_grant_value(grant, 'token'))
                extracted = temporary / 'download'
                extract_tar_safely(archive, extracted)
                root = extracted / 'artifact'
                if not root.is_dir():
                    raise PullProtocolError('Artifact archive is missing its required root')
                wrapper = temporary / 'entry'
                wrapper.mkdir()
                root.replace(wrapper / 'payload')
                return wrapper
            return self.artifact_cache.materialize_to(ArtifactRef.from_dict(ref), grant, fetch, destination / 'artifact')
        archive = self.client.download_artifact(
            worker_token=worker_token, lease_token=lease_token,
            download_path=_grant_value(delivery, "download_path"), artifact_token=_grant_value(grant, "token"),
        )
        extract_tar_safely(archive, destination)
        # Delivery uses a mandatory top-level directory so a hostile archive
        # can never lay files beside our selected destination.  It is a
        # transport wrapper, not part of the ArtifactStore directory contract
        # exposed to action adapters.
        artifact_root = destination / "artifact"
        if not artifact_root.is_dir():
            shutil.rmtree(destination, ignore_errors=True)
            raise PullProtocolError("artifact archive is missing its required artifact root")
        expected_digest = grant.get('content_manifest_sha256')
        if expected_digest is not None:
            from oracle_data_contracts.artifacts import directory_content_digest
            if directory_content_digest(artifact_root) != expected_digest:
                shutil.rmtree(destination, ignore_errors=True)
                raise PullProtocolError('Materialized artifact content does not match its authenticated grant')
        return artifact_root

    def _cancellation_requested(self, *, lease_id: str, worker_token: str, lease_token: str) -> bool:
        # Test/deployment clients from before cancellation support remain
        # usable; production client always implements the check.
        check = getattr(self.client, "cancellation", None)
        return bool(check(lease_id=lease_id, worker_token=worker_token, lease_token=lease_token)) if callable(check) else False

    def _complete_cancellation(self, *, lease_id: str, worker_token: str, lease_token: str) -> None:
        complete = getattr(self.client, "cancelled", None)
        if not callable(complete):
            raise PullProtocolError("worker client does not support cancellation completion")
        complete(lease_id=lease_id, worker_token=worker_token, lease_token=lease_token)

    def run_lease(
        self, *, worker_id: str, worker_token: str, lease_response: Mapping[str, Any],
        lease_ttl_seconds: int | None = None, renew_interval_seconds: float | None = None,
    ) -> Mapping[str, Any]:
        lease = lease_response.get("lease")
        work_unit = lease_response.get("work_unit")
        if not isinstance(lease, Mapping) or not isinstance(work_unit, Mapping):
            raise PullProtocolError("lease response must contain lease and work_unit objects")
        try:
            # Validate the sealed wire contract before using its references.
            # This rejects accidental future path-bearing additions as well.
            parsed_unit = parse_work_unit(work_unit)
            if lease.get('work_unit_sha256') and parsed_unit.sha256 != lease['work_unit_sha256']:
                raise PullProtocolError('Work unit does not match the leased contract digest')
            work_unit = parsed_unit.to_dict()
        except WorkUnitError as exc:
            raise PullProtocolError(f"lease included an invalid work unit: {exc}") from exc
        lease_id, lease_token = _grant_value(lease, "lease_id"), _grant_value(lease, "lease_token")
        raw_inputs = work_unit.get("inputs")
        if not isinstance(raw_inputs, Mapping):
            raise PullProtocolError("work unit must include portable input references")
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix=f"oracle-{lease_id}-", dir=self.scratch_root))
        staging = scratch / "staging"
        staging.mkdir()
        inputs: dict[str, Path] = {}
        keepalive: _LeaseKeepalive | None = None
        control = None
        retained_recovery = False
        try:
            def emit(event_type: str, message: str, data: Mapping[str, Any] | None = None) -> None:
                self.client.events(lease_id=lease_id, worker_token=worker_token, lease_token=lease_token, events=[{
                    "type": event_type, "message": message, "data": dict(data or {}),
                }])

            def stage_timing(stage: str, started: float, message: str, **details: Any) -> None:
                """Persist a compact, comparable timing measurement.

                These events deliberately cover worker/control-plane boundaries
                rather than micro-benchmarks.  They survive worker restarts,
                are queryable with normal job events, and do not disclose
                worker-local paths.
                """
                emit("stage_timing", message, {
                    "schema": "oracle_timing_v1", "stage": stage,
                    "duration_seconds": round(max(0.0, time.perf_counter() - started), 3),
                    **details,
                })

            lease_started = time.perf_counter()
            self.client.acknowledge(lease_id=lease_id, worker_token=worker_token, lease_token=lease_token)
            stage_timing("lease_acknowledgement", lease_started, "Worker lease acknowledged")
            if self._cancellation_requested(lease_id=lease_id, worker_token=worker_token, lease_token=lease_token):
                self._complete_cancellation(lease_id=lease_id, worker_token=worker_token, lease_token=lease_token)
                return {"status": "cancelled"}
            if self.supervised:
                from oracle_builder.worker.control import WorkerControl
                control = WorkerControl(self.client, credentials={'lease_id': lease_id, 'worker_token': worker_token, 'lease_token': lease_token},
                    control_file=scratch / 'execution-control.json', ttl_seconds=lease_ttl_seconds or 30, generation=lease.get('generation', 1), worker_boot_id=lease.get('worker_boot_id'))
                control.start()
            elif lease_ttl_seconds is not None:
                if lease_ttl_seconds < 2:
                    raise ValueError("lease_ttl_seconds must be at least 2 when renewal is enabled")
                keepalive = _LeaseKeepalive(
                    self.client, lease_id=lease_id, worker_token=worker_token, lease_token=lease_token,
                    ttl_seconds=lease_ttl_seconds,
                    interval_seconds=renew_interval_seconds or max(1.0, lease_ttl_seconds / 2),
                )
                keepalive.start()
            materialization_started = time.perf_counter()
            emit("materializing", "Worker is materializing leased artifacts", {"worker_id": worker_id})
            for name, ref in raw_inputs.items():
                if not isinstance(name, str) or not name or not isinstance(ref, Mapping):
                    raise PullProtocolError("work-unit input artifact reference is invalid")
                inputs[name] = self._materialize_ref(ref, lease_id=lease_id, worker_token=worker_token, lease_token=lease_token, destination=scratch / "inputs" / name)
                if control:
                    control.check()
                if self._cancellation_requested(lease_id=lease_id, worker_token=worker_token, lease_token=lease_token):
                    self._complete_cancellation(lease_id=lease_id, worker_token=worker_token, lease_token=lease_token)
                    return {"status": "cancelled"}
            configuration = None
            if isinstance(work_unit.get("configuration"), Mapping):
                configuration = self._materialize_ref(work_unit["configuration"], lease_id=lease_id, worker_token=worker_token, lease_token=lease_token, destination=scratch / "configuration")
            stage_timing(
                "artifact_materialization", materialization_started, "Worker materialized input artifacts",
                input_count=len(inputs), has_configuration=configuration is not None,
            )
            execution_started = time.perf_counter()
            emit("executing", "Worker started execution", {})
            if control:
                from oracle_builder.worker.supervisor import execute_supervised
                result = execute_supervised(work_unit=work_unit, inputs=inputs, configuration=configuration, staging=staging, emit_event=emit, control=control)
            else:
                result = self.executor.execute(work_unit=work_unit, inputs=inputs, configuration=configuration, staging=staging, emit_event=emit)
            stage_timing("execution", execution_started, "Worker execution completed")
            if keepalive is not None:
                keepalive.raise_if_failed()
            if self._cancellation_requested(lease_id=lease_id, worker_token=worker_token, lease_token=lease_token):
                self._complete_cancellation(lease_id=lease_id, worker_token=worker_token, lease_token=lease_token)
                return {"status": "cancelled"}
            archive_started = time.perf_counter()
            if control:
                control.check()
                control.progress('publishing')
            archive_path = scratch / "output.tar"
            archive_size, archive_sha256 = archive_directory_to_file(staging, archive_path)
            stage_timing("output_archiving", archive_started, "Worker archived staged output", bytes=archive_size)
            emit("publishing_output", "Uploading staged output in bounded parts", {"bytes": archive_size})
            publication_started = time.perf_counter()
            try:
                upload_output_archive(
                    self.client, archive_path=archive_path, lease_id=lease_id,
                    worker_token=worker_token, lease_token=lease_token,
                    check_cancel=control.check if control else None,
                )
            except OutputPublicationError:
                recovery = self._retain_output_recovery(
                    scratch=scratch, lease_id=lease_id, archive_name=archive_path.name,
                    archive_size=archive_size, archive_sha256=archive_sha256, part_size=8 * 1024 * 1024,
                )
                retained_recovery = True
                try:
                    emit("output_publication_deferred", "Output publication failed; retained for recovery", {"recovery_id": recovery.name})
                except Exception:
                    pass
                raise
            stage_timing("output_publication", publication_started, "Worker output upload finalized", bytes=archive_size)
            emit("output_published", "Staged output upload finalized", {"bytes": archive_size})
            stage_timing("worker_total", lease_started, "Worker execution and output publication completed", outcome="succeeded")
            if control:
                control.check()
            self.client.complete(lease_id=lease_id, worker_token=worker_token, lease_token=lease_token, outcome="succeeded")
            return {"status": "succeeded", "result": dict(result or {})}
        except ExecutionInterrupted as exc:
            if exc.command and exc.action in {'stop_now', 'restart'}:
                self.client.acknowledge_command(lease_id=lease_id, worker_token=worker_token, lease_token=lease_token,
                    command_id=exc.command['command_id'], status='applied', result={'executor_stopped': True})
            elif exc.action == 'stop_now':
                self._complete_cancellation(lease_id=lease_id, worker_token=worker_token, lease_token=lease_token)
            # Expired/revoked leases have no authority to report completion.
            return {'status': 'interrupted', 'action': exc.action}
        except OutputPublicationError:
            # Do not mark the lease failed: its retained output can be resumed
            # while the lease is still valid using fresh credentials.
            defer = getattr(self.client, "defer_output_publication", None)
            if callable(defer):
                try:
                    defer(lease_id=lease_id, worker_token=worker_token, lease_token=lease_token)
                except Exception:
                    # A network loss can also prevent the transition. Retained
                    # bytes are still safer than rerunning scientific work.
                    pass
            raise
        except Exception as exc:
            # Best effort only: the lease expiry remains the recovery authority
            # if the network itself is unavailable.
            try:
                self.client.events(lease_id=lease_id, worker_token=worker_token, lease_token=lease_token, events=[{
                    "type": "failed", "message": "Worker execution failed", "data": {"error": str(exc)[:2000]},
                }])
                self.client.complete(lease_id=lease_id, worker_token=worker_token, lease_token=lease_token, outcome="failed", error=str(exc)[:2000])
            except PullProtocolError:
                pass
            raise
        finally:
            if control:
                control.stop()
            if keepalive is not None:
                keepalive.stop()
            # Scratch may contain user-provided data and trained-model output;
            # it is never a durable artifact store and must not accumulate.
            if not retained_recovery:
                shutil.rmtree(scratch, ignore_errors=True)


class _LeaseKeepalive:
    """Renew a lease while an executor is running, without owning execution."""

    def __init__(self, client: OrchestratorPullClient, *, lease_id: str, worker_token: str,
                 lease_token: str, ttl_seconds: int, interval_seconds: float):
        self.client = client
        self.lease_id = lease_id
        self.worker_token = worker_token
        self.lease_token = lease_token
        self.ttl_seconds = ttl_seconds
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._failure: Exception | None = None
        self._thread = threading.Thread(target=self._run, name=f"oracle-lease-{lease_id}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.client.renew(
                    lease_id=self.lease_id, worker_token=self.worker_token,
                    lease_token=self.lease_token, ttl_seconds=self.ttl_seconds,
                )
            except Exception as exc:  # the foreground code reports/finishes safely
                self._failure = exc
                self._stop.set()
                return

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise PullProtocolError(f"worker lease renewal failed: {self._failure}") from self._failure

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))


@dataclass(frozen=True, slots=True)
class WorkerCredentials:
    worker_id: str
    worker_token: str
    pool_id: str
    orchestrator_url: str


def load_worker_credentials(path: str | Path) -> WorkerCredentials | None:
    """Read a local worker credential file only when it is private."""
    candidate = Path(path).expanduser()
    if not candidate.exists():
        return None
    if not candidate.is_file() or candidate.is_symlink():
        raise PullProtocolError("worker credentials path must be a regular file")
    if os.stat(candidate).st_mode & 0o077:
        raise PullProtocolError("worker credentials file must not be group/world accessible")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("not an object")
        return WorkerCredentials(**{key: value[key] for key in WorkerCredentials.__annotations__})
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise PullProtocolError(f"worker credentials file is invalid: {exc}") from exc


def save_worker_credentials(path: str | Path, credentials: WorkerCredentials) -> None:
    """Atomically persist only the worker's own opaque credential."""
    candidate = Path(path).expanduser()
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if candidate.exists() and (not candidate.is_file() or candidate.is_symlink()):
        raise PullProtocolError("worker credentials path must be a regular file")
    temporary = candidate.with_name(candidate.name + ".tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(credentials.__dict__ if hasattr(credentials, "__dict__") else {
                "worker_id": credentials.worker_id, "worker_token": credentials.worker_token,
                "pool_id": credentials.pool_id, "orchestrator_url": credentials.orchestrator_url,
            }, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
        os.replace(temporary, candidate)
        os.chmod(candidate, 0o600)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise PullProtocolError(f"could not save worker credentials: {exc}") from exc


class PullWorkerLoop:
    """Operational polling loop with reusable identity and bounded shutdown."""

    def __init__(self, client: OrchestratorPullClient, *, runtime: PullWorkerRuntime, pool_id: str,
                 name: str, credentials_path: str | Path, registration_token: str | None,
                 capabilities: Mapping[str, Any], endpoint: str | None = None,
                 lease_ttl_seconds: int = 60, poll_interval_seconds: float = 2.0):
        self.client, self.runtime = client, runtime
        self.pool_id, self.name = pool_id, name
        self.credentials_path, self.registration_token = Path(credentials_path), registration_token
        self.capabilities, self.endpoint = dict(capabilities), endpoint
        self.lease_ttl_seconds, self.poll_interval_seconds = lease_ttl_seconds, poll_interval_seconds

    def credentials(self) -> WorkerCredentials:
        current = load_worker_credentials(self.credentials_path)
        if current is not None:
            if current.pool_id != self.pool_id or current.orchestrator_url.rstrip("/") != self.client.base_url:
                raise PullProtocolError("worker credentials belong to a different pool or orchestrator")
            return current
        if not self.registration_token:
            raise PullProtocolError("registration token is required when no worker credentials exist")
        response = self.client.register(pool_id=self.pool_id, name=self.name,
                                        registration_token=self.registration_token,
                                        endpoint=self.endpoint, capabilities=self.capabilities)
        worker = response.get("worker") or response
        worker_id, worker_token = worker.get("worker_id"), response.get("worker_token")
        if not isinstance(worker_id, str) or not isinstance(worker_token, str):
            raise PullProtocolError("registration response did not contain worker credentials")
        current = WorkerCredentials(worker_id, worker_token, self.pool_id, self.client.base_url)
        save_worker_credentials(self.credentials_path, current)
        return current

    def run(self, *, max_jobs: int | None = None, once: bool = False,
            stop_event: threading.Event | None = None) -> int:
        if max_jobs is not None and max_jobs < 1:
            raise ValueError("max_jobs must be positive")
        if self.lease_ttl_seconds < 2:
            raise ValueError("lease_ttl_seconds must be at least 2")
        stop = stop_event or threading.Event()
        credentials = self.credentials()
        completed = 0
        while not stop.is_set() and (max_jobs is None or completed < max_jobs):
            lease = self.client.poll(worker_id=credentials.worker_id, worker_token=credentials.worker_token,
                                     ttl_seconds=self.lease_ttl_seconds, capabilities=self.capabilities)
            if lease is None:
                if once:
                    break
                stop.wait(self.poll_interval_seconds)
                continue
            try:
                self.runtime.run_lease(worker_id=credentials.worker_id, worker_token=credentials.worker_token,
                                       lease_response=lease, lease_ttl_seconds=self.lease_ttl_seconds)
            except OutputPublicationError:
                # The control plane now owns a publication-pending lease for
                # this worker. Do not poll again and accidentally overwrite
                # its visible ``publishing`` worker state with ``idle``; a
                # supervisor/operator resumes the retained archive explicitly.
                break
            except Exception:
                # run_lease reports a failed completion when connectivity
                # permits.  Continue so a single bad unit cannot kill a fleet.
                pass
            completed += 1
            if once:
                break
        return completed


__all__ = [
    "DeterministicPackagingExecutor", "ExecutorRegistry", "InferenceExecutor", "MaterializedWorkUnit", "TrainingExecutor",
    "OrchestratorPullClient", "OutputPublicationError", "PullProtocolError", "PullWorkerLoop", "PullWorkerRuntime",
    "WorkerCredentials", "WorkUnitExecutor", "archive_directory_to_file", "extract_tar_safely",
    "load_worker_credentials", "save_worker_credentials",
    "recover_output_upload", "upload_output_archive",
]
