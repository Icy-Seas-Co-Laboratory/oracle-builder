"""Bounded, warm resident inference runtime for the V2 serving API.

The runtime deliberately owns no HTTP concerns.  A process may keep one of
these objects alive and let an API adapter submit ordinary ``InferenceItem``
objects to it.  Models are identified by the fingerprint of their sealed
artifact, never by a mutable filesystem path or display name.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np

from oracle_builder.artifacts import validate_run_artifact
from oracle_builder.evaluation.segmentation_targets import CANDIDATE_DELTA, segmentation_target_mode
from oracle_builder.inference.contracts import InferenceItem, InferenceResultSet

if TYPE_CHECKING:
    from oracle_builder.inference.bundle import InferenceBundle


def _load_bundle(path: Path) -> "InferenceBundle":
    """Defer the ML import until an artifact is actually loaded."""
    from oracle_builder.inference.bundle import InferenceBundle

    return InferenceBundle.load(path)


class ResidentRuntimeError(RuntimeError):
    """Base exception for resident serving failures."""


class ResidentModelNotFoundError(ResidentRuntimeError, KeyError):
    """The requested immutable model identity is not registered."""


class ResidentRuntimeCapacityError(ResidentRuntimeError):
    """Every loaded model is busy, so no model can safely be evicted."""


class ResidentArtifactChangedError(ResidentRuntimeError):
    """A registered artifact no longer verifies as the sealed identity."""


@dataclass(frozen=True)
class ResidentModel:
    """Immutable identity and source location for one registered artifact."""

    artifact_fingerprint: str
    artifact_id: str
    run_id: str
    task: str
    architecture: str
    run_dir: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "artifact_fingerprint": self.artifact_fingerprint,
            "artifact_id": self.artifact_id,
            "run_id": self.run_id,
            "task": self.task,
            "architecture": self.architecture,
        }


@dataclass
class _Entry:
    model: ResidentModel
    bundle: InferenceBundle | None = None
    lane: "_PredictionLane | None" = None
    state: str = "registered"
    load_error: str | None = None
    in_flight: int = 0
    last_used: float = field(default_factory=time.monotonic)
    warmup: dict[str, Any] | None = None


@dataclass
class _PendingRequest:
    items: list[InferenceItem]
    future: Future
    enqueued_at: float


class _PredictionLane:
    """Serialize accelerator use for one model and micro-batch classifiers."""

    def __init__(
        self,
        bundle: InferenceBundle,
        *,
        max_batch_size: int,
        max_wait_ms: int,
        queue_capacity: int,
    ):
        self.bundle = bundle
        self.task = str(bundle.model_reference.task)
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_ms = max(0, int(max_wait_ms))
        self._queue: queue.Queue[_PendingRequest | None] = queue.Queue(
            maxsize=max(1, int(queue_capacity))
        )
        self._closed = threading.Event()
        self._stats_lock = threading.Lock()
        self._stats: dict[str, float | int] = {
            "requests": 0,
            "items": 0,
            "batches": 0,
            "queue_wait_ms": 0.0,
            "execution_ms": 0.0,
        }
        self._thread = threading.Thread(
            target=self._run,
            name="oracle-resident-inference",
            daemon=True,
        )
        self._thread.start()

    @property
    def classification_batched(self) -> bool:
        return self.task in {"classification", "clustering", "embedding"}

    def submit(self, items: Iterable[InferenceItem]) -> InferenceResultSet:
        materialized = list(items)
        if not materialized:
            raise ValueError("Inference request must contain at least one item")
        if self.classification_batched and len(materialized) > self.max_batch_size:
            raise ValueError(
                f"Inference request exceeds the {self.max_batch_size}-item limit"
            )
        if self._closed.is_set():
            raise ResidentRuntimeError("Inference runtime is closed")
        pending = _PendingRequest(materialized, Future(), time.perf_counter())
        try:
            self._queue.put_nowait(pending)
        except queue.Full as exc:
            raise ResidentRuntimeCapacityError("Inference queue is full") from exc
        return pending.future.result()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        # A full queue means this lane is busy.  Its daemon thread will drain
        # after the active request; leaving it to do so is safer than dropping
        # accepted inference requests during cache eviction.
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5)

    def diagnostics(self) -> dict[str, Any]:
        with self._stats_lock:
            stats = dict(self._stats)
        batches = max(int(stats["batches"]), 1)
        return {
            "mode": "classification_microbatch" if self.classification_batched else "segmentation_item",
            "max_batch_size": self.max_batch_size if self.classification_batched else 1,
            "max_wait_ms": self.max_wait_ms if self.classification_batched else 0,
            "queue_capacity": self._queue.maxsize,
            "queue_depth": self._queue.qsize(),
            "requests": stats["requests"],
            "items": stats["items"],
            "batches": stats["batches"],
            "mean_items_per_batch": stats["items"] / batches,
            "mean_queue_wait_ms": stats["queue_wait_ms"] / max(int(stats["requests"]), 1),
            "mean_execution_ms": stats["execution_ms"] / batches,
        }

    def _run(self) -> None:
        carry: _PendingRequest | None = None
        while True:
            if carry is not None:
                first, carry = carry, None
            else:
                try:
                    first = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if self._closed.is_set():
                        return
                    continue
            if first is None:
                return
            pending = [first]
            item_count = len(first.items)
            if self.classification_batched:
                deadline = time.perf_counter() + self.max_wait_ms / 1000.0
                while item_count < self.max_batch_size:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        candidate = self._queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if candidate is None:
                        self._closed.set()
                        break
                    if item_count + len(candidate.items) > self.max_batch_size:
                        # Keep the first request for the next batch locally.
                        # Putting it back can deadlock if another producer
                        # fills the bounded queue before this lane does so.
                        carry = candidate
                        break
                    pending.append(candidate)
                    item_count += len(candidate.items)
            self._execute(pending)
            if self._closed.is_set() and self._queue.empty() and carry is None:
                return

    def _execute(self, pending: list[_PendingRequest]) -> None:
        all_items = [item for request in pending for item in request.items]
        started = time.perf_counter()
        try:
            # For segmentation this intentionally executes one accepted request
            # at a time.  ``InferenceBundle`` expands its ROIs into tiles and
            # reassembles them, so cross-request tile batching would require a
            # separate geometry scheduler to preserve result correlation.
            combined = self.bundle.predict_batch(all_items)
            duration_ms = (time.perf_counter() - started) * 1000.0
            offset = 0
            for request in pending:
                result_set = InferenceResultSet(
                    model=combined.model,
                    execution=(
                        dict(combined.execution)
                        if combined.execution
                        else dict(getattr(self.bundle, "execution_diagnostics", {}))
                    ),
                )
                result_set.parameters.update(
                    {
                        "resident_batch_item_count": len(all_items),
                        "resident_batch_request_count": len(pending),
                        "resident_scheduling": (
                            "classification_microbatch"
                            if self.classification_batched
                            else "segmentation_item"
                        ),
                    }
                )
                for sequence, result in enumerate(
                    combined.results[offset : offset + len(request.items)]
                ):
                    result.result_set_id = result_set.result_set_id
                    result.sequence_number = sequence
                    result_set.append(result)
                offset += len(request.items)
                request.future.set_result(result_set.complete())
            with self._stats_lock:
                self._stats["requests"] += len(pending)
                self._stats["items"] += len(all_items)
                self._stats["batches"] += 1
                self._stats["queue_wait_ms"] += sum(
                    (started - request.enqueued_at) * 1000.0 for request in pending
                )
                self._stats["execution_ms"] += duration_ms
        except Exception as exc:
            for request in pending:
                request.future.set_exception(exc)


class ResidentInferenceRuntime:
    """Thread-safe warm cache for sealed, immutable model artifacts.

    ``register`` and every subsequent model load validate the artifact.  This
    makes a path mutation visible as an error instead of silently serving a
    different model under a previously trusted fingerprint.
    """

    def __init__(
        self,
        *,
        max_loaded_models: int = 2,
        classification_max_batch_size: int = 64,
        classification_max_wait_ms: int = 8,
        queue_capacity: int = 1024,
        loader: Callable[[Path], InferenceBundle] | None = None,
        validator: Callable[[str | Path], dict[str, Any]] | None = None,
    ):
        if max_loaded_models < 1:
            raise ValueError("max_loaded_models must be at least one")
        self.max_loaded_models = int(max_loaded_models)
        self.classification_max_batch_size = max(1, int(classification_max_batch_size))
        self.classification_max_wait_ms = max(0, int(classification_max_wait_ms))
        self.queue_capacity = max(1, int(queue_capacity))
        self._loader = loader or _load_bundle
        self._validator = validator or validate_run_artifact
        self._entries: dict[str, _Entry] = {}
        self._aliases: dict[str, str] = {}
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._closed = False

    def register(
        self,
        run_dir: str | Path,
        *,
        artifact_fingerprint: str | None = None,
    ) -> ResidentModel:
        """Register a sealed model and return its immutable serving identity."""
        path = Path(run_dir).expanduser().resolve()
        model = self._read_verified_model(path, expected_fingerprint=artifact_fingerprint)
        with self._lock:
            existing = self._entries.get(model.artifact_fingerprint)
            if existing is not None:
                if existing.model.run_dir != model.run_dir:
                    raise ValueError(
                        "Artifact fingerprint is already registered from a different path"
                    )
                return existing.model
            for alias in (model.artifact_fingerprint, model.artifact_id, model.run_id):
                known = self._aliases.get(alias)
                if known is not None and known != model.artifact_fingerprint:
                    raise ValueError(f"Model identity alias is already registered: {alias}")
            self._entries[model.artifact_fingerprint] = _Entry(model=model)
            for alias in (model.artifact_fingerprint, model.artifact_id, model.run_id):
                self._aliases[alias] = model.artifact_fingerprint
        return model

    def warm(self, selector: str) -> dict[str, Any]:
        """Load, compile, and retain a model before its first request."""
        entry = self._acquire(selector)
        try:
            return {
                **entry.model.to_dict(),
                "state": entry.state,
                "warmup": dict(entry.warmup or {}),
                "capabilities": self._capabilities(entry.bundle),
                "runtime": entry.lane.diagnostics() if entry.lane else None,
            }
        finally:
            self._release(entry)

    def predict(
        self, selector: str, items: Iterable[InferenceItem]
    ) -> InferenceResultSet:
        """Serve a request through the selected resident model's work lane."""
        entry = self._acquire(selector)
        try:
            if entry.lane is None:  # defensive: acquire only returns ready entries
                raise ResidentRuntimeError("Model prediction lane is unavailable")
            return entry.lane.submit(items)
        finally:
            self._release(entry)

    def describe(self) -> list[dict[str, Any]]:
        with self._lock:
            entries = list(self._entries.values())
            rows = []
            for entry in entries:
                rows.append(
                    {
                        **entry.model.to_dict(),
                        "state": entry.state,
                        "loaded": entry.bundle is not None,
                        "in_flight": entry.in_flight,
                        "load_error": entry.load_error,
                        "warmup": dict(entry.warmup or {}),
                        "runtime": entry.lane.diagnostics() if entry.lane else None,
                    }
                )
        return sorted(rows, key=lambda row: str(row["artifact_fingerprint"]))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            lanes = [entry.lane for entry in self._entries.values() if entry.lane]
            for entry in self._entries.values():
                entry.state = "closed"
                entry.bundle = None
                entry.lane = None
            self._changed.notify_all()
        for lane in lanes:
            lane.close()

    def _resolve_entry(self, selector: str) -> _Entry:
        fingerprint = self._aliases.get(str(selector), str(selector))
        entry = self._entries.get(fingerprint)
        if entry is None:
            raise ResidentModelNotFoundError(selector)
        return entry

    def _acquire(self, selector: str) -> _Entry:
        with self._changed:
            if self._closed:
                raise ResidentRuntimeError("Inference runtime is closed")
            entry = self._resolve_entry(selector)
            while entry.state == "loading":
                self._changed.wait()
            if entry.bundle is not None and entry.lane is not None:
                entry.in_flight += 1
                entry.last_used = time.monotonic()
                return entry
            self._evict_if_needed_locked(excluding=entry.model.artifact_fingerprint)
            entry.state = "loading"
        try:
            self._revalidate(entry.model)
            bundle = self._loader(entry.model.run_dir)
            warmup = self._warm_bundle(bundle)
            resolved_batch = self._resolved_batch_size(warmup) if str(bundle.model_reference.task) in {"classification", "clustering", "embedding"} else 1
            lane = _PredictionLane(
                bundle,
                max_batch_size=resolved_batch,
                max_wait_ms=self.classification_max_wait_ms,
                queue_capacity=self.queue_capacity,
            )
        except Exception as exc:
            with self._changed:
                entry.state = "failed"
                entry.load_error = f"{type(exc).__name__}: {exc}"
                self._changed.notify_all()
            raise
        with self._changed:
            if self._closed:
                lane.close()
                raise ResidentRuntimeError("Inference runtime is closed")
            entry.bundle = bundle
            entry.lane = lane
            entry.warmup = warmup
            entry.state = "ready"
            entry.load_error = None
            entry.in_flight += 1
            entry.last_used = time.monotonic()
            self._changed.notify_all()
            return entry

    def _resolved_batch_size(self, report: dict[str, Any]) -> int:
        """Respect OOM backoff, including the weakest stratified child."""
        children = report.get("strata")
        if isinstance(children, dict) and children:
            return min(self._resolved_batch_size(child) if isinstance(child, dict) else 1 for child in children.values())
        value = report.get("resolved_max_batch_size")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return 1
        return min(self.classification_max_batch_size, value)

    @staticmethod
    def _candidate_required(config: dict[str, Any]) -> bool:
        data = config.get("data", {})
        shape = list(data.get("input_shape") or [])
        return (
            segmentation_target_mode(config) == CANDIDATE_DELTA
            or bool(data.get("candidate_sdf"))
            or str(data.get("candidate_distance") or "none").strip().lower() != "none"
            or (len(shape) >= 3 and int(shape[-1]) == 2)
        )

    def _capabilities(self, bundle: InferenceBundle | None) -> dict[str, Any]:
        if bundle is None:
            return {}
        task = str(bundle.model_reference.task)
        if task == "segmentation":
            return {
                "task": "mask_refinement",
                "required_inputs": ["image", "candidate_mask"] if self._candidate_required(bundle.config) else ["image"],
                "target_mode": segmentation_target_mode(bundle.config),
                "default_outputs": ["mask"],
                "possible_outputs": [
                    "probability_map", "logits",
                    *(["delta_mask", "delta_probability_map"] if segmentation_target_mode(bundle.config) == CANDIDATE_DELTA else []),
                ],
                "input_shape": list(bundle.config.get("data", {}).get("input_shape") or []),
            }
        return {
            "model_task": task,
            "request_task": "classification",
            "required_inputs": ["image"],
            "default_outputs": (
                ["class_probabilities", "primary_decision", "diagnostics"]
                if task == "classification" else ["embedding", "embedding_normalized"]
            ),
            "possible_outputs": [
                *(["class_probabilities", "primary_decision", "diagnostics"] if task == "classification" else []),
                "embedding", "embedding_normalized", "knn", "prototype_similarity",
                "cluster_evidence",
            ],
            "input_shape": list(bundle.config.get("data", {}).get("input_shape") or []),
        }

    def _release(self, entry: _Entry) -> None:
        with self._changed:
            entry.in_flight = max(0, entry.in_flight - 1)
            entry.last_used = time.monotonic()
            self._changed.notify_all()

    def _evict_if_needed_locked(self, *, excluding: str) -> None:
        # A loading entry reserves a slot too.  Otherwise two callers can both
        # observe an empty cache and load large models concurrently.
        occupied = [
            entry
            for entry in self._entries.values()
            if entry.bundle is not None or entry.state == "loading"
        ]
        while len(occupied) >= self.max_loaded_models:
            candidates = [
                entry
                for entry in occupied
                if entry.model.artifact_fingerprint != excluding and entry.in_flight == 0
                and entry.bundle is not None
            ]
            if not candidates:
                raise ResidentRuntimeCapacityError(
                    "All resident model slots are in use; retry after an active request completes"
                )
            victim = min(candidates, key=lambda entry: entry.last_used)
            lane = victim.lane
            victim.bundle = None
            victim.lane = None
            victim.warmup = None
            victim.state = "registered"
            if lane is not None:
                lane.close()
            occupied = [
                entry
                for entry in self._entries.values()
                if entry.bundle is not None or entry.state == "loading"
            ]

    def _read_verified_model(
        self, path: Path, *, expected_fingerprint: str | None
    ) -> ResidentModel:
        verification = self._validator(path)
        if not verification.get("valid"):
            errors = "; ".join(str(value) for value in verification.get("errors", []))
            raise ResidentArtifactChangedError(
                f"Artifact verification failed for {path}: {errors or 'unknown error'}"
            )
        if verification.get("lifecycle") != "sealed" or verification.get("status") != "complete":
            raise ValueError("Resident inference requires a sealed, complete model artifact")
        manifest_path = path / "artifact.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResidentArtifactChangedError(f"Cannot read artifact manifest: {exc}") from exc
        fingerprint = str(manifest.get("fingerprint_sha256") or "")
        if not fingerprint:
            raise ValueError("Sealed model artifact has no fingerprint_sha256")
        if verification.get("fingerprint_sha256") != fingerprint:
            raise ResidentArtifactChangedError("Artifact fingerprint changed during verification")
        if expected_fingerprint is not None and expected_fingerprint != fingerprint:
            raise ResidentArtifactChangedError("Artifact fingerprint does not match requested identity")
        model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
        task = str(model.get("task") or manifest.get("task") or "")
        architecture = str(model.get("architecture") or manifest.get("architecture") or "")
        artifact_id = str(manifest.get("artifact_id") or "")
        run_id = str(manifest.get("run_id") or "")
        if not all((task, architecture, artifact_id, run_id)):
            raise ValueError("Sealed model artifact has incomplete serving identity")
        return ResidentModel(fingerprint, artifact_id, run_id, task, architecture, path)

    def _revalidate(self, model: ResidentModel) -> None:
        observed = self._read_verified_model(
            model.run_dir, expected_fingerprint=model.artifact_fingerprint
        )
        if observed != model:
            raise ResidentArtifactChangedError(
                "Registered artifact identity changed after registration"
            )

    def _warm_bundle(self, bundle: InferenceBundle) -> dict[str, Any]:
        task = str(bundle.model_reference.task)
        if task in {"classification", "clustering", "embedding"}:
            return dict(
                bundle.warm_for_serving(
                    (1, self.classification_max_batch_size)
                )
            )
        if task != "segmentation":
            raise ValueError(f"Unsupported resident inference task: {task}")
        # Segmentation's normal warm helper cannot invoke a model because each
        # ROI may need tiling.  A single native-size zero ROI exercises that
        # exact path and is discarded immediately.
        shape = list(bundle.config.get("data", {}).get("input_shape") or [])
        if len(shape) < 2:
            raise ValueError("Segmentation artifact has no input_shape")
        requires_candidate = self._candidate_required(bundle.config)
        image_shape = shape[:2] if requires_candidate or len(shape) < 3 else shape
        image = np.zeros(tuple(int(value) for value in image_shape), dtype="float32")
        candidate = (
            np.zeros(tuple(int(value) for value in shape[:2]), dtype="uint8")
            if requires_candidate
            else None
        )
        warmed = bundle.predict(InferenceItem.from_array(image, candidate_mask=candidate))
        if warmed.status != "ok":
            message = (warmed.error or {}).get("message", "segmentation warmup failed")
            raise ResidentRuntimeError(str(message))
        return {
            "runtime": "segmentation",
            "execution": dict(getattr(bundle, "execution_diagnostics", {})),
            "warmup": [{"batch_size": 1, "tile_count": (warmed.output or {}).get("transform", {}).get("tile_count", 1)}],
        }
