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
    preferred = (
        "loss",
        "val_loss",
        "accuracy",
        "val_accuracy",
        "macro_f1",
        "val_macro_f1",
    )
    names = [name for name in preferred if name in metrics]
    names.extend(sorted(name for name in metrics if name not in names))
    return [(name, metrics[name]) for name in names]


def _rendered_metrics(metrics: dict[str, float], *, limit: int) -> str:
    values = _ordered_metrics(metrics)
    rendered = ", ".join(f"{name}={value:.5g}" for name, value in values[:limit])
    if len(values) > limit:
        rendered = f"{rendered}, +{len(values) - limit} more"
    return rendered


def _sparkline(values: list[float], width: int = 16) -> str:
    """Render a compact, scale-local history suitable for a terminal board."""
    points = values[-max(1, int(width)):]
    if not points:
        return ""
    if len(points) == 1:
        return "▅"
    low, high = min(points), max(points)
    if high == low:
        return "▅" * len(points)
    glyphs = "▁▂▃▄▅▆▇█"
    return "".join(
        glyphs[min(len(glyphs) - 1, int((point - low) / (high - low) * len(glyphs)))]
        for point in points
    )


def _trend(values: list[float]) -> str:
    if len(values) < 2:
        return ""
    delta = values[-1] - values[-2]
    tolerance = max(abs(values[-2]), 1.0) * 1e-6
    if abs(delta) <= tolerance:
        return "→"
    return "↗" if delta > 0 else "↘"


class RichTrainingStatusCallback(keras.callbacks.Callback):
    """A compact live board for Keras training, with log-safe text fallback.

    Batch callbacks advance progress only. Metrics are refreshed at epoch
    boundaries, where Keras has aggregated every batch and, when configured,
    completed held-out validation. This deliberately avoids presenting a
    transient mini-batch loss, accuracy, or F1 score as a model result.
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
        self._latest_validation: dict[str, float] = {}
        self._history: dict[str, list[float]] = {}
        self._batch_metrics: dict[str, float] = {}
        self._completed_batches = 0
        self._external_phase: str | None = None
        self._external_completed: int | None = None
        self._external_total: int | None = None
        self._external_started_at: float | None = None
        self._batch_status = "Preparing first batch"

    def _event(self, message: str, details: dict[str, Any]) -> None:
        if self.training_log is not None and self.run_id is not None:
            log_event(self.training_log, self.run_id, "INFO", message, details)

    def _learning_rate(self) -> float | None:
        try:
            value = self.model.optimizer.learning_rate
            return float(keras.backend.get_value(value))
        except (AttributeError, TypeError, ValueError):
            return None

    def _timing_summary(self) -> str:
        """Return useful live throughput without pretending batch logs are epoch metrics."""
        elapsed = time.perf_counter() - (self._epoch_started_at or time.perf_counter())
        if not self._completed_batches or elapsed <= 0:
            return f"epoch elapsed {elapsed:.0f}s"
        seconds_per_batch = elapsed / self._completed_batches
        rate = self._completed_batches / elapsed
        remaining = (
            max(self._steps - self._completed_batches, 0) * seconds_per_batch
            if self._steps not in (None, -1)
            else None
        )
        rendered = f"{rate:.2f} batches/s · {seconds_per_batch:.2f}s/batch"
        if remaining is not None:
            rendered += f" · epoch ETA {remaining / 60:.1f}m"
        return rendered

    def _external_summary(self) -> str | None:
        if self._external_phase is None:
            return None
        elapsed = time.perf_counter() - (self._external_started_at or time.perf_counter())
        progress = ""
        if self._external_completed is not None:
            if self._external_total:
                percent = self._external_completed / self._external_total * 100
                progress = (
                    f" · {self._external_completed:,}/{self._external_total:,} batches "
                    f"({percent:.0f}%)"
                )
            else:
                progress = f" · {self._external_completed:,} batches"
        return (
            "[bold yellow]Post-epoch analysis (no optimizer updates)[/bold yellow] "
            f"· {self._external_phase}{progress} · {elapsed / 60:.1f}m elapsed"
        )

    def _refresh(self) -> None:
        if self._interactive and self._live is not None:
            self._live.update(self._board())

    def begin_post_epoch_analysis(self, split: str, total_batches: int | None = None) -> None:
        """Expose expensive non-training work performed by another callback."""
        self._external_phase = f"rich metrics: {split}"
        self._external_completed = 0
        self._external_total = total_batches if total_batches and total_batches > 0 else None
        self._external_started_at = time.perf_counter()
        self._refresh()

    def update_post_epoch_analysis(self, completed_batches: int) -> None:
        if self._external_phase is None:
            return
        self._external_completed = max(0, int(completed_batches))
        self._refresh()

    def end_post_epoch_analysis(self) -> None:
        self._external_phase = None
        self._external_completed = None
        self._external_total = None
        self._external_started_at = None
        self._refresh()

    def _board(self):
        total_epochs = self.epochs or self.params.get("epochs") or "?"
        run_state = Table.grid(expand=True, padding=(0, 2))
        run_state.add_column(ratio=1)
        run_state.add_column(justify="right", ratio=1)
        steps = (
            f"{self._completed_batches:,}/{self._steps:,} batches"
            if self._steps not in (None, -1)
            else f"{self._completed_batches:,} batches"
        )
        run_state.add_row(
            f"[bold]Epoch {self._epoch}/{total_epochs}[/bold] · {steps}",
            f"[dim]{self._timing_summary()}[/dim]",
        )

        metrics = Table.grid(expand=True, padding=(0, 2))
        metrics.add_column(style="bold", width=23)
        metrics.add_column(ratio=1)
        metrics.row_styles = ["", "on grey15"]
        live = _rendered_metrics(self._batch_metrics, limit=5) or "waiting for the first completed batch"
        validation = _rendered_metrics(self._latest_validation, limit=5) or "not available until epoch 1 completes"
        metrics.add_row("Train (running)", live)
        metrics.add_row("Last validation", validation)
        if any(not name.startswith("val_") for name in self._metrics):
            completed = _rendered_metrics(self._metrics, limit=5)
            metrics.add_row("Last completed epoch", completed)
        learning_rate = self._learning_rate()
        if learning_rate is not None:
            metrics.add_row("Learning rate", f"{learning_rate:.3g}")
        status = Table.grid(expand=True)
        status.add_row(f"[dim]{self._batch_status}[/dim]")
        external = self._external_summary()
        if external:
            status.add_row(external)
        run_label = f" · run {self.run_id[:8]}" if self.run_id else ""
        return Panel(
            Group(run_state, self._progress, metrics, status),
            title=f"[bold cyan]{self.phase}{run_label}[/bold cyan]",
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
        # Keras reports validation metrics only at an epoch boundary. Keep the
        # latest values visible while the next epoch's training batches arrive.
        self._metrics = dict(self._latest_validation)
        self._batch_metrics = {}
        self._completed_batches = 0
        self.end_post_epoch_analysis()
        self._batch_status = "Preparing first batch"
        self._epoch_started_at = time.perf_counter()
        details = {"phase": self.phase, "epoch": self._epoch, "epochs": self.epochs}
        self._event("Training epoch started", details)
        if self._interactive and self._progress is not None and self._batch_task is not None:
            total = self._steps if self._steps not in (None, -1) else None
            self._progress.update(self._batch_task, completed=0, total=total, description="Batches")
            self._refresh()
        elif self.display == "text":
            print(f"[{self.phase}] epoch {self._epoch}/{self.epochs or '?'} started", file=self.stream, flush=True)

    def on_train_batch_begin(self, batch: int, logs=None):
        del logs
        batch_number = int(batch) + 1
        self._batch_status = (
            "Computing first optimizer update (initial shape may compile)"
            if batch_number == 1
            else f"Computing batch {batch_number}"
        )
        if batch_number == 1:
            self._event(
                "Training first optimizer update started",
                {"phase": self.phase, "epoch": self._epoch, "batch": batch_number},
            )
        self._refresh()

    def on_input_batch_loading(self, batch: int):
        """Report input-pipeline work before a manual training update starts."""
        batch_number = int(batch) + 1
        self._batch_status = f"Loading batch {batch_number} from input pipeline"
        if batch_number == 1:
            self._event(
                "Training first batch loading",
                {"phase": self.phase, "epoch": self._epoch, "batch": batch_number},
            )
        self._refresh()

    def on_train_batch_end(self, batch: int, logs=None):
        batch_number = int(batch) + 1
        self._completed_batches = batch_number
        self._batch_metrics = {
            name: value
            for name, value in _numeric_metrics(logs).items()
            if not name.startswith("val_") and name not in {"batch", "size"}
        }
        self._batch_status = f"Completed batch {batch_number}"
        if batch_number == 1 or batch_number % 100 == 0:
            self._event(
                "Training batch progress",
                {"phase": self.phase, "epoch": self._epoch, "batch": batch_number},
            )
        if self._interactive and self._progress is not None and self._batch_task is not None:
            self._progress.update(self._batch_task, completed=int(batch) + 1)
            self._refresh()

    def on_epoch_end(self, epoch: int, logs=None):
        self._metrics = _numeric_metrics(logs)
        self._latest_validation = {
            name: value for name, value in self._metrics.items() if name.startswith("val_")
        }
        for name, value in self._metrics.items():
            self._history.setdefault(name, []).append(value)
        elapsed = time.perf_counter() - (self._epoch_started_at or time.perf_counter())
        details = {
            "phase": self.phase,
            "epoch": int(epoch) + 1,
            "epochs": self.epochs,
            "elapsed_seconds": elapsed,
            "metrics": self._metrics,
        }
        self._event("Training epoch completed", details)
        if self._interactive:
            self._refresh()
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
