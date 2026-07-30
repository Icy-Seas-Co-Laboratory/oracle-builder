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
from oracle_builder.training.losses import BinaryCrossentropySoftDice
from oracle_builder.training.metrics import BinaryDice


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
):
    set_seed(int(config["run"].get("seed", 123)))
    model = build_and_compile_model(config)
    write_model_summary(model, Path(run_dir) / "model" / "model_summary.txt")
    if config.get("pretraining", {}).get("enabled", False):
        if pretraining_dataset is None:
            raise ValueError("Enabled self-supervised pretraining requires a pretraining dataset")
        from oracle_builder.training.logging_callbacks import log_event
        from oracle_builder.training.student_teacher import run_student_teacher_pretraining

        log_event(
            training_log,
            run_id,
            "INFO",
            "Started student-teacher self-supervised pretraining",
            config["pretraining"],
        )
        pretraining_history = run_student_teacher_pretraining(
            model,
            pretraining_dataset,
            config,
            run_dir,
        )
        log_event(
            training_log,
            run_id,
            "INFO",
            "Completed student-teacher self-supervised pretraining",
            {
                key: float(values[-1])
                for key, values in pretraining_history.history.items()
                if values
            },
        )
    callbacks = build_callbacks(config, run_dir, training_log, run_id)
    validation_data = datasets.get("validation")
    history = model.fit(
        datasets["train"],
        validation_data=validation_data,
        epochs=int(config["training"].get("epochs", 10)),
        callbacks=callbacks,
        verbose=2 if config.get("debug") else 1,
    )
    history_rows = []
    for epoch, metrics in enumerate(zip(*history.history.values(), strict=False)):
        row = {"epoch": epoch}
        for name, value in zip(history.history.keys(), metrics, strict=False):
            row[name] = float(value)
        history_rows.append(row)
    metrics_df = pd.DataFrame(history_rows)
    metrics_df.to_csv(Path(run_dir) / "metrics.csv", index=False)
    Path(run_dir, "metrics.json").write_text(json.dumps(history.history, indent=2, default=float) + "\n")
    return model, history
