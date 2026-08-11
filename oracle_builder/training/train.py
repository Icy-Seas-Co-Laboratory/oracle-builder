from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from oracle_builder.registry import get_model_builder
from oracle_builder.training.callbacks import build_callbacks
from oracle_builder.training.losses import (
    BinaryCrossentropySoftDice,
    BinaryCrossentropySoftTversky,
    WeightedSparseCategoricalCrossentropy,
)
from oracle_builder.training.class_weights import WEIGHTED_CROSS_ENTROPY_NAMES
from oracle_builder.training.metrics import BinaryDice
from oracle_builder.training.distribution import (
    select_distribution_strategy,
    write_distribution_info,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compile_model(model: keras.Model, config: dict[str, Any]) -> keras.Model:
    training = config["training"]
    learning_rate = float(training.get("learning_rate", 0.001))
    optimizer_name = training.get("optimizer", "adam").lower()
    if optimizer_name == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.get(optimizer_name)
        optimizer.learning_rate = learning_rate

    loss_name = training["loss"]
    if str(loss_name).lower() in {"bce_soft_dice", "bce+soft_dice", "binary_crossentropy_soft_dice"}:
        loss = BinaryCrossentropySoftDice(
            bce_weight=float(training.get("bce_weight", 1.0)),
            dice_weight=float(training.get("soft_dice_weight", 1.0)),
            smooth=float(training.get("soft_dice_smooth", 1e-6)),
        )
    elif str(loss_name).lower() in {"bce_soft_tversky", "bce+soft_tversky", "binary_crossentropy_soft_tversky"}:
        loss = BinaryCrossentropySoftTversky(
            bce_weight=float(training.get("bce_weight", 1.0)),
            tversky_weight=float(training.get("soft_tversky_weight", 1.0)),
            alpha=float(training.get("tversky_alpha", 0.3)),
            beta=float(training.get("tversky_beta", 0.7)),
            smooth=float(training.get("soft_tversky_smooth", 1e-6)),
        )
    elif str(loss_name).lower() in WEIGHTED_CROSS_ENTROPY_NAMES:
        class_weights = training.get("class_weights", {}).get("values")
        if not class_weights:
            raise ValueError(
                "Weighted cross entropy requires resolved "
                "training.class_weights.values"
            )
        loss = WeightedSparseCategoricalCrossentropy(class_weights)
    else:
        loss = loss_name

    metrics = []
    for metric in training.get("metrics", []):
        if str(metric).lower() == "dice":
            metrics.append(BinaryDice())
            continue
        if str(metric).lower() == "iou":
            continue
        metrics.append(metric)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def build_and_compile_model(config: dict[str, Any]) -> keras.Model:
    builder = get_model_builder(config["run"]["model"])
    model = builder(config)
    return compile_model(model, config)


def write_model_summary(model: keras.Model, path: str | Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    Path(path).write_text("\n".join(lines) + "\n")


def train_model(
    config: dict[str, Any],
    datasets: dict[str, Any],
    run_dir: str | Path,
    training_log: str | Path,
    run_id: str,
    pretraining_dataset=None,
    resume_state: dict[str, Any] | None = None,
):
    set_seed(int(config["run"].get("seed", 123)))
    strategy, distribution_info = select_distribution_strategy(config)
    with strategy.scope():
        if resume_state is None:
            model = build_and_compile_model(config)
        else:
            recovery_path = Path(run_dir) / resume_state["model_path"]
            model = keras.models.load_model(recovery_path)
    write_distribution_info(distribution_info, run_dir)
    write_model_summary(model, Path(run_dir) / "model" / "model_summary.txt")
    from oracle_builder.training.logging_callbacks import log_event

    log_event(
        training_log,
        run_id,
        "INFO",
        "Configured TensorFlow distribution strategy",
        {
            "requested_strategy": distribution_info.requested_strategy,
            "resolved_strategy": distribution_info.resolved_strategy,
            "replicas": distribution_info.replicas,
            "devices": distribution_info.devices,
            "global_batch_size": distribution_info.global_batch_size,
            "per_replica_batch_size": distribution_info.per_replica_batch_size,
            "cross_device_ops": distribution_info.cross_device_ops,
        },
    )
    if config.get("pretraining", {}).get("enabled", False) and resume_state is None:
        if pretraining_dataset is None:
            raise ValueError("Enabled self-supervised pretraining requires a pretraining dataset")
        from oracle_builder.training.student_teacher import run_grayscale_reconstruction_pretraining, run_student_teacher_pretraining

        log_event(
            training_log,
            run_id,
            "INFO",
            "Started self-supervised pretraining",
            config["pretraining"],
        )
        pretraining_history = (run_grayscale_reconstruction_pretraining(model, pretraining_dataset, config, run_dir, strategy=strategy)
            if str(config["pretraining"].get("method", "byol")).lower() == "grayscale_reconstruction"
            else run_student_teacher_pretraining(model, pretraining_dataset, config, run_dir, strategy=strategy))
        log_event(
            training_log,
            run_id,
            "INFO",
            "Completed self-supervised pretraining",
            {
                key: float(values[-1])
                for key, values in pretraining_history.history.items()
                if values
            },
        )
    elif config.get("pretraining", {}).get("enabled", False):
        log_event(
            training_log,
            run_id,
            "INFO",
            "Skipped completed self-supervised pretraining while resuming supervised training",
        )
    callbacks = build_callbacks(
        config,
        run_dir,
        training_log,
        run_id,
        artifact_id=config.get("artifact", {}).get("artifact_id"),
    )
    validation_data = datasets.get("validation")
    initial_epoch = int(resume_state.get("completed_epoch", 0)) if resume_state else 0
    total_epochs = int(config["training"].get("epochs", 10))
    if initial_epoch > total_epochs:
        raise ValueError("Recovery snapshot is beyond configured training.epochs")
    if initial_epoch < total_epochs:
        print(f"Training supervised epochs {initial_epoch + 1} through {total_epochs}", flush=True)
        model.fit(
            datasets["train"],
            validation_data=validation_data,
            initial_epoch=initial_epoch,
            epochs=total_epochs,
            callbacks=callbacks,
            verbose=2 if config.get("debug") else 1,
        )
    else:
        print("Supervised epochs already complete; continuing finalization.", flush=True)
    from oracle_builder.artifacts.layout import RunLayout
    from oracle_builder.training.logging_callbacks import history_from_training_log

    history_data = history_from_training_log(training_log, run_id)
    history_rows = [
        {"epoch": epoch, **{name: values[epoch] for name, values in history_data.items()}}
        for epoch in range(max((len(values) for values in history_data.values()), default=0))
    ]
    metrics_df = pd.DataFrame(history_rows)
    layout = RunLayout(run_dir)
    metrics_df.to_csv(layout.metrics_csv, index=False)
    layout.metrics_json.write_text(
        json.dumps(history_data, indent=2, default=float) + "\n"
    )
    history = keras.callbacks.History()
    history.history = history_data
    return model, history
