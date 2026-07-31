from __future__ import annotations

import sqlite3
import uuid

import numpy as np
import pytest

from oracle_builder.artifacts.layout import RunLayout
from oracle_builder.training.callbacks import build_callbacks
from oracle_builder.training.logging_callbacks import (
    history_from_training_log,
    init_training_log,
)
from oracle_builder.training.recovery import validate_recovery_state
from oracle_builder.training.train import build_and_compile_model, train_model


def recovery_config() -> dict:
    return {
        "run": {
            "task": "classification",
            "model": "simple_cnn",
            "seed": 123,
            "run_id": "e5712425-99af-498d-853a-f0fb9a177233",
        },
        "artifact": {"artifact_id": "6d6254e3-ae66-4671-a4d6-0ee0e7b9b0aa"},
        "data": {"input_shape": [16, 16, 1], "num_classes": 2, "batch_size": 2},
        "model": {"base_filters": 2, "dropout": 0.0, "embedding_dim": 8},
        "training": {
            "epochs": 2,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"],
        },
        "recovery": {"enabled": True, "save_every_epochs": 1},
        "output": {"save_checkpoints": False},
    }


def test_rolling_recovery_restores_optimizer_model_and_continues_history(tmp_path):
    config = recovery_config()
    run_id = config["run"]["run_id"]
    artifact_id = config["artifact"]["artifact_id"]
    layout = RunLayout(tmp_path / "run")
    layout.create_directories()
    environment = {"test": True}
    init_training_log(layout.training_log, run_id, "recovery", config, environment)
    x = np.random.default_rng(12).random((4, 16, 16, 1), dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype="int64")
    dataset = __import__("tensorflow").data.Dataset.from_tensor_slices((x, y)).batch(2)
    model = build_and_compile_model(config)
    model.fit(
        dataset,
        epochs=1,
        callbacks=build_callbacks(
            config, layout.root, layout.training_log, run_id, artifact_id=artifact_id
        ),
        verbose=0,
    )

    state = validate_recovery_state(
        layout.root, config, artifact_id=artifact_id, run_id=run_id
    )
    assert state["completed_epoch"] == 1
    assert layout.recovery_model.exists()
    assert layout.recovery_state.exists()

    resumed_model, history = train_model(
        config,
        {"train": dataset, "validation": dataset},
        layout.root,
        layout.training_log,
        run_id,
        resume_state=state,
    )

    assert resumed_model.optimizer is not None
    assert len(history.history["loss"]) == 2
    with sqlite3.connect(layout.training_log) as connection:
        epochs = connection.execute(
            "SELECT DISTINCT epoch FROM epoch_metrics WHERE run_id = ? ORDER BY epoch",
            (run_id,),
        ).fetchall()
    assert epochs == [(0,), (1,)]
    assert len(history_from_training_log(layout.training_log, run_id)["loss"]) == 2


def test_recovery_rejects_changed_training_contract(tmp_path):
    config = recovery_config()
    run_id = config["run"]["run_id"]
    artifact_id = config["artifact"]["artifact_id"]
    layout = RunLayout(tmp_path / "run")
    layout.create_directories()
    init_training_log(layout.training_log, run_id, "recovery", config, {})
    x = np.zeros((2, 16, 16, 1), dtype="float32")
    y = np.array([0, 1], dtype="int64")
    model = build_and_compile_model(config)
    model.fit(
        x,
        y,
        epochs=1,
        callbacks=build_callbacks(
            config, layout.root, layout.training_log, run_id, artifact_id=artifact_id
        ),
        verbose=0,
    )
    changed = {**config, "training": {**config["training"], "learning_rate": 0.1}}

    with pytest.raises(ValueError, match="resolved training contract"):
        validate_recovery_state(
            layout.root, changed, artifact_id=artifact_id, run_id=run_id
        )
