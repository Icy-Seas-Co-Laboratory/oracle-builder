from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from oracle_builder.config import self_supervised_settings
from oracle_builder.registry import get_model_builder
from oracle_builder.training.callbacks import build_callbacks
from oracle_builder.training.losses import (
    BinaryCrossentropySoftDice,
    BinaryCrossentropySoftTversky,
    WeightedSparseCategoricalCrossentropy,
)
from oracle_builder.training.class_weights import WEIGHTED_CROSS_ENTROPY_NAMES
from oracle_builder.training.metrics import BinaryDice, SparseCategoricalMacroF1
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
    weight_decay = float(training.get("weight_decay", 0.0))
    optimizer_name = training.get("optimizer", "adam").lower()
    if optimizer_name == "adam":
        optimizer = (
            keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
            if weight_decay
            else keras.optimizers.Adam(learning_rate=learning_rate)
        )
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
        if str(metric).lower() in {"macro_f1", "macro-f1"}:
            if config["run"]["task"] not in {"classification", "embedding"}:
                raise ValueError("macro_f1 is available only for classification tasks")
            metrics.append(SparseCategoricalMacroF1(config["data"]["num_classes"]))
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
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")


def model_summary_path(run_dir: str | Path, config: dict[str, Any]) -> Path:
    """Return the canonical architecture-summary location for this model.

    Stratified orchestration sets the private active dimension on each child
    config. It is intentionally runtime-only: the parent configuration retains
    the complete stratum list while each child writes next to its own artifacts.
    """
    stratification = config.get("classification", {}).get("stratification", {})
    dimension = stratification.get("_active_dimension") if isinstance(stratification, dict) else None
    if dimension is not None:
        return Path(run_dir) / "model" / "strata" / str(int(dimension)) / "model_summary.txt"
    return Path(run_dir) / "model" / "model_summary.txt"


def train_model(
    config: dict[str, Any],
    datasets: dict[str, Any],
    run_dir: str | Path,
    training_log: str | Path,
    run_id: str,
    pretraining_dataset=None,
    resume_state: dict[str, Any] | None = None,
    classification_metric_datasets: dict[str, Any] | None = None,
    segment_stop_epoch: int | None = None,
    segment_control=None,
    require_recovery: bool = False,
    segment_max_steps: int | None = None,
    segment_start_step: int | None = None,
):
    set_seed(int(config["run"].get("seed", 123)))
    strategy, distribution_info = select_distribution_strategy(config)
    with strategy.scope():
        if resume_state is None:
            model = build_and_compile_model(config)
        else:
            recovery_path = Path(run_dir) / resume_state["model_path"]
            # Import at use time to avoid a module-import cycle: the shared
            # loader also supports artifact load tests and imports this
            # module's builder. It allowlists project components while keeping
            # Keras safe deserialization enabled.
            from oracle_builder.saving.load_test import _load_keras_model
            model = _load_keras_model(recovery_path, compile=True)
    write_distribution_info(distribution_info, run_dir)
    write_model_summary(model, model_summary_path(run_dir, config))
    from oracle_builder.training.logging_callbacks import log_event
    from oracle_builder.training.recovery import read_recovery_callback_state

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
    self_supervised = self_supervised_settings(config)
    if (
        segment_max_steps is None
        and resume_state is not None
        and isinstance(resume_state.get("cursor"), dict)
        and int(resume_state.get("completed_epoch", 0)) < int(config["training"].get("epochs", 10))
    ):
        raise ValueError(
            "This recovery snapshot has a partial ordered-batch cursor; "
            "continue it with segment_max_steps rather than model.fit"
        )
    if segment_max_steps is not None:
        return _train_ordered_batch_segment(
            model, config, datasets, run_dir, training_log, run_id,
            resume_state=resume_state,
            max_steps=int(segment_max_steps),
            requested_start_step=segment_start_step,
        )
    if self_supervised.get("enabled", False) and resume_state is None:
        if pretraining_dataset is None:
            raise ValueError("Enabled self-supervised training requires a self-supervised dataset")
        from oracle_builder.training.student_teacher import (
            run_grayscale_reconstruction_self_supervised,
            run_self_supervised_training,
        )

        log_event(
            training_log,
            run_id,
            "INFO",
            "Started self-supervised training",
            self_supervised,
        )
        self_supervised_history = (
            run_grayscale_reconstruction_self_supervised(
                model,
                pretraining_dataset,
                config,
                run_dir,
                strategy=strategy,
                training_log=training_log,
                run_id=run_id,
            )
            if str(self_supervised.get("method", "byol")).lower()
            == "grayscale_reconstruction"
            else run_self_supervised_training(
                model,
                pretraining_dataset,
                config,
                run_dir,
                strategy=strategy,
                training_log=training_log,
                run_id=run_id,
            )
        )
        log_event(
            training_log,
            run_id,
            "INFO",
            "Completed self-supervised training",
            {
                key: float(values[-1])
                for key, values in self_supervised_history.history.items()
                if values
            },
        )
    elif self_supervised.get("enabled", False):
        log_event(
            training_log,
            run_id,
            "INFO",
            "Skipped completed self-supervised training while resuming supervised training",
        )
    callbacks = build_callbacks(
        config,
        run_dir,
        training_log,
        run_id,
        artifact_id=config.get("artifact", {}).get("artifact_id"),
        classification_metric_datasets=classification_metric_datasets,
        segment_control=segment_control,
        segment_stop_epoch=segment_stop_epoch,
        require_recovery=require_recovery,
        resume_callback_state=(
            read_recovery_callback_state(run_dir, resume_state)
            if resume_state is not None else None
        ),
    )
    validation_data = datasets.get("validation")
    initial_epoch = int(resume_state.get("completed_epoch", 0)) if resume_state else 0
    total_epochs = int(config["training"].get("epochs", 10))
    if initial_epoch > total_epochs:
        raise ValueError("Recovery snapshot is beyond configured training.epochs")
    target_epoch = min(total_epochs, int(segment_stop_epoch)) if segment_stop_epoch is not None else total_epochs
    if target_epoch < initial_epoch:
        raise ValueError("Segment stop epoch precedes the committed recovery cursor")
    if initial_epoch < target_epoch:
        print(f"Training supervised epochs {initial_epoch + 1} through {target_epoch}", flush=True)
        model.fit(
            datasets["train"],
            validation_data=validation_data,
            initial_epoch=initial_epoch,
            epochs=target_epoch,
            callbacks=callbacks,
            # RichTrainingStatusCallback owns console rendering. Keeping Keras
            # quiet prevents its batch progress output from competing with it.
            verbose=0,
        )
    else:
        print("Supervised epochs already complete; continuing finalization.", flush=True)
    from oracle_builder.artifacts.layout import RunLayout
    from oracle_builder.training.logging_callbacks import (
        history_from_training_log,
    )

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


def _train_ordered_batch_segment(
    model: keras.Model,
    config: dict[str, Any],
    datasets: dict[str, Any],
    run_dir: str | Path,
    training_log: str | Path,
    run_id: str,
    *,
    resume_state: dict[str, Any] | None,
    max_steps: int,
    requested_start_step: int | None,
):
    """Advance a portable cursor through one deterministic, finite epoch.

    This is intentionally a small supported surface. ``model.fit`` callback
    semantics are epoch-oriented, so partial epochs use direct batch updates
    and report sample-weighted *partial* metrics.  Callers must opt into the
    sealed ``ordered_batches_v1`` policy; streaming, random augmentation and
    callback-controlled schedules are not silently treated as resumable.
    """
    if self_supervised_settings(config).get("enabled", False):
        raise ValueError("Step-bounded segments do not support self-supervised training")
    if str(config.get("distribution", {}).get("strategy", "cpu")).lower() != "cpu":
        raise ValueError("Step-bounded segments currently require distribution.strategy = 'cpu'")
    if str(config.get("training", {}).get("step_cursor_policy", "")).lower() != "ordered_batches_v1":
        raise ValueError("Step-bounded segments require training.step_cursor_policy = 'ordered_batches_v1'")
    if config.get("data", {}).get("streaming", {}).get("enabled", False):
        raise ValueError("Step-bounded segments do not support streaming datasets yet")
    if config.get("augmentation", {}).get("enabled", False):
        raise ValueError("Step-bounded segments do not support random augmentation yet")
    callbacks = config.get("callbacks", {})
    if callbacks.get("early_stopping") or callbacks.get("reduce_lr_on_plateau"):
        raise ValueError("Step-bounded segments do not support epoch callback schedules yet")
    train = datasets.get("train")
    if train is None:
        raise ValueError("Step-bounded segments require a train dataset")
    cardinality = int(tf.data.experimental.cardinality(train).numpy())
    if cardinality < 0:
        raise ValueError("Step-bounded segments require a finite train dataset")
    cursor = dict((resume_state or {}).get("cursor", {}))
    if cursor and cursor.get("kind") != "ordered_batches_v1":
        raise ValueError("Recovery snapshot does not contain an ordered batch cursor")
    epoch = int(cursor.get("epoch", (resume_state or {}).get("completed_epoch", 0)))
    offset = int(cursor.get("batch_offset", 0))
    global_step = int(cursor.get("global_step", epoch * cardinality + offset))
    if requested_start_step is not None and int(requested_start_step) != offset:
        raise ValueError("Segment batch offset does not match the committed checkpoint cursor")
    if offset < 0 or offset >= cardinality:
        raise ValueError("Recovery batch offset is outside the deterministic epoch")
    options = tf.data.Options()
    options.experimental_deterministic = True
    batches = train.with_options(options).skip(offset).take(min(max_steps, cardinality - offset))
    totals: dict[str, float] = {}
    samples = 0
    for batch in batches:
        model.reset_metrics()
        values = model.train_on_batch(*batch, return_dict=True)
        inputs = batch[0]
        first = next(iter(inputs.values())) if isinstance(inputs, dict) else inputs
        count = int(tf.shape(first)[0])
        samples += count
        for name, value in values.items():
            totals[str(name)] = totals.get(str(name), 0.0) + float(value) * count
        offset += 1
        global_step += 1
    if samples == 0:
        raise ValueError("Step-bounded segment did not consume any deterministic batches")
    completed_epoch = epoch
    if offset == cardinality:
        completed_epoch = epoch + 1
        epoch += 1
        offset = 0
    partial_metrics = {
        "sample_count": samples,
        "metrics": {name: value / samples for name, value in totals.items()},
        "complete_epoch": offset == 0,
    }
    from oracle_builder.training.logging_callbacks import log_event
    from oracle_builder.training.recovery import save_step_recovery_snapshot

    state = save_step_recovery_snapshot(
        model, run_dir, config, training_log, run_id,
        config.get("artifact", {}).get("artifact_id", ""),
        completed_epoch=completed_epoch,
        cursor={"epoch": epoch, "batch_offset": offset, "global_step": global_step},
        partial_metrics=partial_metrics,
    )
    log_event(training_log, run_id, "INFO", "Committed deterministic batch segment", {
        "cursor": state["cursor"], "partial_metrics": partial_metrics,
    })
    history = keras.callbacks.History()
    history.history = {}
    return model, history
