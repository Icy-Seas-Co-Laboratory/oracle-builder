"""Reusable, terminal-safe training status display."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from tensorflow import keras

from oracle_builder.training.logging_callbacks import log_event


def _numeric_metrics(logs: dict[str, Any] | None) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, value in (logs or {}).items():
        try:
            values[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def _ordered_metrics(metrics: dict[str, float]) -> list[tuple[str, float]]:
    preferred = ("loss", "val_loss", "accuracy", "val_accuracy")
    names = [name for name in preferred if name in metrics]
    names.extend(sorted(name for name in metrics if name not in names))
    return [(name, metrics[name]) for name in names]


def _rendered_metrics(metrics: dict[str, float], *, limit: int) -> str:
    values = _ordered_metrics(metrics)
    rendered = ", ".join(f"{name}={value:.5g}" for name, value in values[:limit])
    if len(values) > limit:
        rendered = f"{rendered}, +{len(values) - limit} more"
    return rendered


class RichTrainingStatusCallback(keras.callbacks.Callback):
    """A compact live board for Keras training, with log-safe text fallback.

    The board is deliberately metric-agnostic: it formats whichever scalar
    metrics a model reports, so custom SSL diagnostics and normal validation
    metrics share exactly the same presentation.
    """

    def __init__(
        self,
        *,
        phase: str,
        epochs: int | None = None,
        display: str = "rich",
        training_log: str | Path | None = None,
        run_id: str | None = None,
        stream=None,
    ):
        super().__init__()
        if display not in {"rich", "text", "off"}:
            raise ValueError("training.display must be 'rich', 'text', or 'off'")
        self.phase = str(phase)
        self.epochs = int(epochs) if epochs is not None else None
        self.display = display
        self.training_log = training_log
        self.run_id = run_id
        self.stream = stream if stream is not None else sys.stdout
        self._interactive = display == "rich" and bool(
            getattr(self.stream, "isatty", lambda: False)()
        )
        self.console = Console(file=self.stream, force_terminal=self._interactive)
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._epoch_task: int | None = None
        self._batch_task: int | None = None
        self._epoch = 0
        self._steps: int | None = None
        self._started_at: float | None = None
        self._epoch_started_at: float | None = None
        self._metrics: dict[str, float] = {}

    def _event(self, message: str, details: dict[str, Any]) -> None:
        if self.training_log is not None and self.run_id is not None:
            log_event(self.training_log, self.run_id, "INFO", message, details)

    def _learning_rate(self) -> float | None:
        try:
            value = self.model.optimizer.learning_rate
            return float(keras.backend.get_value(value))
        except (AttributeError, TypeError, ValueError):
            return None

    def _board(self):
        total_epochs = self.epochs or self.params.get("epochs") or "?"
        metrics = Table.grid(expand=True, padding=(0, 2))
        for name, value in _ordered_metrics(self._metrics)[:12]:
            metrics.add_row(f"[bold]{name}[/bold]", f"{value:.5g}")
        if len(self._metrics) > 12:
            metrics.add_row("metrics", f"+{len(self._metrics) - 12} more")
        learning_rate = self._learning_rate()
        if learning_rate is not None:
            metrics.add_row("[bold]learning rate[/bold]", f"{learning_rate:.3g}")
        if not self._metrics:
            metrics.add_row("status", "Waiting for first batch")
        subtitle = f"Epoch {self._epoch}/{total_epochs}"
        run_label = f" · run {self.run_id[:8]}" if self.run_id else ""
        return Panel(
            Group(self._progress, metrics),
            title=f"[bold cyan]{self.phase}{run_label}[/bold cyan]",
            subtitle=subtitle,
            border_style="cyan",
        )

    def on_train_begin(self, logs=None):
        del logs
        self._started_at = time.perf_counter()
        if not self._interactive:
            return
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self._epoch_task = self._progress.add_task("Epochs", total=self.epochs or self.params.get("epochs"))
        self._batch_task = self._progress.add_task("Batches", total=None)
        self._live = Live(self._board(), console=self.console, refresh_per_second=8, transient=True)
        self._live.start()

    def on_epoch_begin(self, epoch: int, logs=None):
        del logs
        self._epoch = int(epoch) + 1
        self._steps = self.params.get("steps")
        self._metrics = {}
        self._epoch_started_at = time.perf_counter()
        details = {"phase": self.phase, "epoch": self._epoch, "epochs": self.epochs}
        self._event("Training epoch started", details)
        if self._interactive and self._progress is not None and self._batch_task is not None:
            total = self._steps if self._steps not in (None, -1) else None
            self._progress.update(self._batch_task, completed=0, total=total, description="Batches")
            if self._live is not None:
                self._live.update(self._board())
        elif self.display == "text":
            print(f"[{self.phase}] epoch {self._epoch}/{self.epochs or '?'} started", file=self.stream, flush=True)

    def on_train_batch_end(self, batch: int, logs=None):
        self._metrics = _numeric_metrics(logs)
        if self._interactive and self._progress is not None and self._batch_task is not None:
            self._progress.update(self._batch_task, completed=int(batch) + 1)
            if self._live is not None:
                self._live.update(self._board())

    def on_epoch_end(self, epoch: int, logs=None):
        self._metrics = _numeric_metrics(logs)
        elapsed = time.perf_counter() - (self._epoch_started_at or time.perf_counter())
        details = {
            "phase": self.phase,
            "epoch": int(epoch) + 1,
            "epochs": self.epochs,
            "elapsed_seconds": elapsed,
            "metrics": self._metrics,
        }
        self._event("Training epoch completed", details)
        if self._interactive and self._live is not None:
            self._live.update(self._board())
        elif self.display != "off":
            rendered = _rendered_metrics(self._metrics, limit=8)
            lr = self._learning_rate()
            if lr is not None:
                rendered = f"{rendered}, lr={lr:.3g}" if rendered else f"lr={lr:.3g}"
            print(
                f"[{self.phase}] epoch {int(epoch) + 1}/{self.epochs or '?'} completed "
                f"in {elapsed:.1f}s" + (f" — {rendered}" if rendered else ""),
                file=self.stream,
                flush=True,
            )

    def on_train_end(self, logs=None):
        del logs
        if self._live is not None:
            self._live.stop()
            self._live = None
        if self.display != "off" and self._interactive:
            total = time.perf_counter() - (self._started_at or time.perf_counter())
            self.console.print(f"[green]Completed {self.phase} in {total:.1f}s[/green]")
