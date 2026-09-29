from __future__ import annotations

from pathlib import Path
from typing import Any

import math

from tensorflow import keras

from oracle_builder.training.logging_callbacks import (
    ClassificationEpochMetricsLogger,
    SQLiteMetricLogger,
)
from oracle_builder.training.status import RichTrainingStatusCallback


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


class _CallbackStateRestorer(keras.callbacks.Callback):
    """Restore Keras callback counters after their normal train-begin reset."""

    def __init__(self, state: dict[str, Any], early=None, reduce_lr=None):
        super().__init__()
        self.state = state
        self.early = early
        self.reduce_lr = reduce_lr

    def on_train_begin(self, logs=None):
        if self.early is not None:
            saved = self.state.get("early_stopping", {})
            if isinstance(saved, dict):
                for name in ("wait", "stopped_epoch", "best_epoch"):
                    if name in saved:
                        setattr(self.early, name, int(saved[name]))
                if saved.get("best") is not None:
                    self.early.best = float(saved["best"])
                if saved.get("best_weights") is not None:
                    self.early.best_weights = saved["best_weights"]
        if self.reduce_lr is not None:
            saved = self.state.get("reduce_lr_on_plateau", {})
            if isinstance(saved, dict):
                for name in ("wait", "cooldown_counter"):
                    if name in saved:
                        setattr(self.reduce_lr, name, int(saved[name]))
                if saved.get("best") is not None:
                    self.reduce_lr.best = float(saved["best"])


def build_callbacks(
    config: dict[str, Any],
    run_dir: str | Path,
    training_log: str | Path,
    run_id: str,
    *,
    artifact_id: str | None = None,
    classification_metric_datasets: dict[str, Any] | None = None,
    segment_control=None,
    segment_stop_epoch: int | None = None,
    require_recovery: bool = False,
    resume_callback_state: dict[str, Any] | None = None,
):
    status_callback = RichTrainingStatusCallback(
        phase="Supervised training",
        epochs=int(config.get("training", {}).get("epochs", 10)),
        display=str(config.get("training", {}).get("display", "rich")),
        training_log=training_log,
        run_id=run_id,
        status_path=Path(run_dir) / "training-status.json",
        monitoring=config.get("monitoring") if isinstance(config.get("monitoring"), dict) else None,
    )
    callbacks: list[keras.callbacks.Callback] = [
        SQLiteMetricLogger(training_log, run_id),
        status_callback,
    ]
    if classification_metric_datasets is not None:
        labels = {
            int(row["class_index"]): str(row.get("name") or row["class_index"])
            for row in config.get("dataset", {}).get("labels", [])
        }
        callbacks.insert(
            0,
            ClassificationEpochMetricsLogger(
                training_log,
                run_id,
                classification_metric_datasets,
                labels,
                status_callback,
            ),
        )
    callback_config = config.get("callbacks", {})
    output_config = config.get("output", {})
    monitor = callback_config.get("checkpoint_monitor", "val_loss")

    if output_config.get("save_checkpoints", False):
        checkpoint_path = Path(run_dir) / "model" / "checkpoints" / "epoch_{epoch:03d}.keras"
        callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_path, monitor=monitor, save_best_only=False))
    early_stopping = None
    reduce_lr = None
    if callback_config.get("early_stopping", False):
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=int(callback_config.get("early_stopping_patience", 5)),
            baseline=callback_config.get("early_stopping_baseline"),
            restore_best_weights=True,
        )
        callbacks.append(early_stopping)
    if callback_config.get("reduce_lr_on_plateau", False):
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=monitor, patience=3, factor=0.5)
        callbacks.append(reduce_lr)
    callback_state = resume_callback_state or {}
    if callback_state and (early_stopping is not None or reduce_lr is not None):
        callbacks.append(_CallbackStateRestorer(callback_state, early_stopping, reduce_lr))

    def callback_state_provider() -> dict[str, Any]:
        result: dict[str, Any] = {}
        if early_stopping is not None:
            result["early_stopping"] = {
                "wait": int(early_stopping.wait),
                "stopped_epoch": int(early_stopping.stopped_epoch),
                "best_epoch": int(getattr(early_stopping, "best_epoch", 0)),
                "best": _finite_or_none(getattr(early_stopping, "best", None)),
                "best_weights": list(early_stopping.best_weights or []),
            }
        if reduce_lr is not None:
            result["reduce_lr_on_plateau"] = {
                "wait": int(reduce_lr.wait),
                "cooldown_counter": int(reduce_lr.cooldown_counter),
                "best": _finite_or_none(getattr(reduce_lr, "best", None)),
            }
        return result

    if config.get("recovery", {}).get("enabled", True) or require_recovery:
        if not artifact_id:
            raise ValueError("Recovery requires the run artifact_id")
        from oracle_builder.training.recovery import RollingRecoveryCallback

        callbacks.append(
            RollingRecoveryCallback(
                run_dir,
                config,
                training_log,
                run_id,
                artifact_id,
                force_every_epoch=require_recovery,
                callback_state_provider=callback_state_provider,
            )
        )
    if segment_control is not None:
        # This callback is deliberately after rolling recovery: a boundary
        # stop is reported only after its epoch has a committed snapshot.
        from oracle_builder.training.control import SegmentBoundaryCallback
        callbacks.append(
            SegmentBoundaryCallback(segment_control, stop_epoch=segment_stop_epoch)
        )
    return callbacks
