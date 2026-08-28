from __future__ import annotations

from pathlib import Path
from typing import Any

from tensorflow import keras

from oracle_builder.training.logging_callbacks import SQLiteMetricLogger
from oracle_builder.training.status import RichTrainingStatusCallback


def build_callbacks(
    config: dict[str, Any],
    run_dir: str | Path,
    training_log: str | Path,
    run_id: str,
    *,
    artifact_id: str | None = None,
):
    callbacks: list[keras.callbacks.Callback] = [
        SQLiteMetricLogger(training_log, run_id),
        RichTrainingStatusCallback(
            phase="Supervised training",
            epochs=int(config.get("training", {}).get("epochs", 10)),
            display=str(config.get("training", {}).get("display", "rich")),
            training_log=training_log,
            run_id=run_id,
        ),
    ]
    callback_config = config.get("callbacks", {})
    output_config = config.get("output", {})
    monitor = callback_config.get("checkpoint_monitor", "val_loss")

    if output_config.get("save_checkpoints", False):
        checkpoint_path = Path(run_dir) / "model" / "checkpoints" / "epoch_{epoch:03d}.keras"
        callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_path, monitor=monitor, save_best_only=False))
    if config.get("recovery", {}).get("enabled", True):
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
            )
        )
    if callback_config.get("early_stopping", False):
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=int(callback_config.get("early_stopping_patience", 5)),
                restore_best_weights=True,
            )
        )
    if callback_config.get("reduce_lr_on_plateau", False):
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor=monitor, patience=3, factor=0.5))
    return callbacks
