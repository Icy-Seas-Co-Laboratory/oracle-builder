"""Bounded per-model micro-batching for the HTTP inference service."""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from oracle_builder.inference.contracts import InferenceResultSet


@dataclass
class _PendingBatch:
    items: list[Any]
    future: Future
    enqueued_at: float


class MicroBatchExecutor:
    """One GPU-facing worker that combines short-lived concurrent requests."""

    def __init__(self, bundle: Any, *, max_batch_size: int = 256, max_wait_ms: int = 8, queue_capacity: int = 1024):
        self.bundle = bundle
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_ms = max(0, int(max_wait_ms))
        self._queue: queue.Queue[_PendingBatch | None] = queue.Queue(maxsize=max(1, int(queue_capacity)))
        self._closed = threading.Event()
        self._stats_lock = threading.Lock()
        self._stats = {"requests": 0, "items": 0, "batches": 0, "queue_wait_ms": 0.0, "execution_ms": 0.0}
        self._thread = threading.Thread(target=self._run, name="oracle-inference-microbatch", daemon=True)
        self._thread.start()

    def submit(self, items: list[Any]) -> InferenceResultSet:
        if self._closed.is_set():
            raise RuntimeError("Inference executor is closed")
        pending = _PendingBatch(list(items), Future(), time.perf_counter())
        try:
            self._queue.put_nowait(pending)
        except queue.Full as exc:
            raise RuntimeError("Inference queue is full") from exc
        return pending.future.result()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(None)
        self._thread.join(timeout=5)

    def diagnostics(self) -> dict[str, Any]:
        with self._stats_lock:
            stats = dict(self._stats)
        batches = max(stats["batches"], 1)
        return {
            "enabled": True,
            "max_batch_size": self.max_batch_size,
            "max_wait_ms": self.max_wait_ms,
            "queue_capacity": self._queue.maxsize,
            "queue_depth": self._queue.qsize(),
            "requests": stats["requests"],
            "items": stats["items"],
            "batches": stats["batches"],
            "mean_items_per_batch": stats["items"] / batches,
            "mean_queue_wait_ms": stats["queue_wait_ms"] / max(stats["requests"], 1),
            "mean_execution_ms": stats["execution_ms"] / batches,
        }

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                return
            pending = [first]
            item_count = len(first.items)
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
                    # Preserve the request intact; it will begin the next batch.
                    self._queue.put(candidate)
                    break
                pending.append(candidate)
                item_count += len(candidate.items)
            self._execute(pending)
            if self._closed.is_set() and self._queue.empty():
                return

    def _execute(self, pending: list[_PendingBatch]) -> None:
        all_items = [item for request in pending for item in request.items]
        started = time.perf_counter()
        try:
            combined = self.bundle.predict_batch(all_items)
            execution_ms = (time.perf_counter() - started) * 1000.0
            offset = 0
            for request in pending:
                result_set = InferenceResultSet(model=combined.model)
                result_set.parameters.update({
                    "microbatch_item_count": len(all_items),
                    "microbatch_request_count": len(pending),
                })
                for sequence, result in enumerate(combined.results[offset:offset + len(request.items)]):
                    result.result_set_id = result_set.result_set_id
                    result.sequence_number = sequence
                    result_set.append(result)
                offset += len(request.items)
                request.future.set_result(result_set.complete())
            with self._stats_lock:
                self._stats["requests"] += len(pending)
                self._stats["items"] += len(all_items)
                self._stats["batches"] += 1
                self._stats["queue_wait_ms"] += sum((started - value.enqueued_at) * 1000.0 for value in pending)
                self._stats["execution_ms"] += execution_ms
        except Exception as exc:
            for request in pending:
                request.future.set_exception(exc)
