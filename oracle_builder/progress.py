from __future__ import annotations

import sys
import time
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class PostTrainingProgress:
    total: int
    current: int = 0
    status_path: str | Path | None = None
    completed_epochs: int | None = None
    planned_epochs: int | None = None
    stopped_early: bool = False

    def __post_init__(self) -> None:
        self._started = time.perf_counter()
        self._snapshot = {}
        self._writer = None
        if self.status_path is not None:
            from oracle_builder.training.status import TrainingStatusSnapshotWriter
            path = Path(self.status_path)
            try:
                self._snapshot = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            if not isinstance(self._snapshot, dict):
                self._snapshot = {}
            self._writer = TrainingStatusSnapshotWriter(path, min_interval_seconds=1.0)
        if self.completed_epochs is None:
            progress = self._snapshot.get("progress")
            self.completed_epochs = progress.get("epoch") if isinstance(progress, dict) else None
        if isinstance(self.completed_epochs, int) and isinstance(self.planned_epochs, int):
            self.stopped_early = self.stopped_early or self.completed_epochs < self.planned_epochs

    def _publish(self, label: str, status: str, *, completed_items: int | None = None,
                 total_items: int | None = None, force: bool = True) -> None:
        if self._writer is None:
            return
        detail = {
            "stage": label, "stage_index": self.current, "total_stages": self.total,
            "status": status, "completed_epochs": self.completed_epochs,
            "planned_epochs": self.planned_epochs, "stopped_early": self.stopped_early,
            "elapsed_seconds": round(time.perf_counter() - self._started, 1),
            "completed_items": completed_items, "total_items": total_items,
        }
        self._snapshot.update(state="running", phase="Post-training", message=label,
                              post_training=detail)
        self._writer.write(self._snapshot, force=force)

    def update_items(self, completed: int, total: int) -> None:
        if self.current:
            self._publish(self._snapshot.get("post_training", {}).get("stage", "Generating artifacts"),
                          "running", completed_items=completed, total_items=total, force=False)

    def complete_current(self) -> None:
        if self.current:
            self._publish(self._snapshot.get("post_training", {}).get("stage", "Finalizing artifacts"), "completed")

    @contextmanager
    def stage(self, label: str, *, publish_completion: bool = True) -> Iterator[None]:
        self.current += 1
        started = time.perf_counter()
        self._publish(label, "running")
        print(
            f"[post-training {self.current}/{self.total}] {label}...",
            flush=True,
        )
        try:
            yield
        except Exception:
            elapsed = time.perf_counter() - started
            print(f"  failed after {elapsed:.1f}s", flush=True)
            self._publish(label, "failed")
            raise
        elapsed = time.perf_counter() - started
        print(f"  completed in {elapsed:.1f}s", flush=True)
        if publish_completion:
            self._publish(label, "completed")


class BatchProgress:
    """Small dependency-free progress display suitable for TTYs and logs."""

    def __init__(self, label: str, total: int, *, enabled: bool = True):
        self.label = label
        self.total = max(0, int(total))
        self.enabled = enabled
        self.completed = 0
        self._last_percent = -1
        self._tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._started = time.perf_counter()

    def update(self, count: int) -> None:
        if not self.enabled:
            return
        self.completed = min(self.total, self.completed + int(count))
        percent = int(100 * self.completed / self.total) if self.total else 100
        if self._tty:
            print(
                f"\r{self.label}: {self.completed}/{self.total} ({percent:3d}%)",
                end="",
                flush=True,
            )
        elif percent == 100 or percent // 10 > self._last_percent // 10:
            print(
                f"{self.label}: {self.completed}/{self.total} ({percent}%)",
                flush=True,
            )
        self._last_percent = percent

    def close(self) -> None:
        if not self.enabled:
            return
        if self._tty:
            print()
        elapsed = time.perf_counter() - self._started
        print(f"{self.label}: complete in {elapsed:.1f}s", flush=True)
