from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class PostTrainingProgress:
    total: int
    current: int = 0

    @contextmanager
    def stage(self, label: str) -> Iterator[None]:
        self.current += 1
        started = time.perf_counter()
        print(
            f"[post-training {self.current}/{self.total}] {label}...",
            flush=True,
        )
        try:
            yield
        except Exception:
            elapsed = time.perf_counter() - started
            print(f"  failed after {elapsed:.1f}s", flush=True)
            raise
        elapsed = time.perf_counter() - started
        print(f"  completed in {elapsed:.1f}s", flush=True)


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
