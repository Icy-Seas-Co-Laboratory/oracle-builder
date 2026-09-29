from __future__ import annotations

import sqlite3
import uuid
import copy
import subprocess
import sys

import numpy as np
import pytest

from oracle_builder.artifacts.layout import RunLayout
from oracle_builder.training.callbacks import build_callbacks
from oracle_builder.training.logging_callbacks import (
    history_from_training_log,
    init_training_log,
)
from oracle_builder.training.recovery import (
    read_recovery_callback_state,
    validate_recovery_state,
)
from oracle_builder.training.train import build_and_compile_model, train_model
from oracle_builder.training.losses import WeightedSparseCategoricalCrossentropy
from oracle_builder.training.metrics import SparseCategoricalMacroF1
from oracle_builder.models.components import L2Normalize


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
        "distribution": {"strategy": "cpu"},
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

    # Model construction normalizes a component configuration in place. The
    # recovery assertion is CPU-only and must not acquire a live GPU selected
    # by an ambient auto strategy.
    config["distribution"] = {"strategy": "cpu"}
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


def test_fresh_interpreter_loads_a_compiled_recovery_checkpoint_with_project_components(tmp_path):
    """Supervisor children resume in a fresh interpreter, not this test's registry."""
    keras = __import__("tensorflow").keras
    inputs = keras.Input(shape=(2,))
    model = keras.Model(inputs, L2Normalize(name="features")(inputs))
    model.compile(
        optimizer="adam",
        loss=WeightedSparseCategoricalCrossentropy([1.0, 1.0]),
        metrics=[SparseCategoricalMacroF1(2)],
    )
    checkpoint = tmp_path / "recovery.keras"
    model.save(checkpoint)
    script = (
        "import sys; "
        "from oracle_builder.saving.load_test import _load_keras_model; "
        "_load_keras_model(sys.argv[1], compile=True)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(checkpoint)],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_segment_stops_on_an_atomic_generation_and_can_resume(tmp_path):
    config = recovery_config()
    config["distribution"] = {"strategy": "cpu"}
    run_id = config["run"]["run_id"]
    artifact_id = config["artifact"]["artifact_id"]
    layout = RunLayout(tmp_path / "run")
    layout.create_directories()
    init_training_log(layout.training_log, run_id, "recovery", config, {})
    x = np.random.default_rng(19).random((4, 16, 16, 1), dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype="int64")
    dataset = __import__("tensorflow").data.Dataset.from_tensor_slices((x, y)).batch(2)

    train_model(
        config,
        {"train": dataset, "validation": dataset},
        layout.root,
        layout.training_log,
        run_id,
        segment_stop_epoch=1,
        require_recovery=True,
    )
    state = validate_recovery_state(
        layout.root, config, artifact_id=artifact_id, run_id=run_id
    )
    assert state["completed_epoch"] == 1
    generation = layout.root / state["generation"]["path"]
    assert (generation / "model.keras").exists()
    assert (generation / "state.json").exists()
    assert state["model_path"].startswith("model/recovery/generations/")
    assert state["continuation"]["rng_state"] == "not_portable"

    _model, history = train_model(
        config,
        {"train": dataset, "validation": dataset},
        layout.root,
        layout.training_log,
        run_id,
        resume_state=state,
    )
    assert len(history.history["loss"]) == 2


def test_segment_restores_stateful_callback_continuation(tmp_path):
    config = recovery_config()
    config["distribution"] = {"strategy": "cpu"}
    config["callbacks"] = {
        "early_stopping": True,
        "early_stopping_patience": 20,
        "reduce_lr_on_plateau": True,
        "checkpoint_monitor": "val_loss",
    }
    run_id = config["run"]["run_id"]
    artifact_id = config["artifact"]["artifact_id"]
    layout = RunLayout(tmp_path / "callback-run")
    layout.create_directories()
    init_training_log(layout.training_log, run_id, "callback", config, {})
    x = np.zeros((4, 16, 16, 1), dtype="float32")
    y = np.array([0, 1, 0, 1], dtype="int64")
    dataset = __import__("tensorflow").data.Dataset.from_tensor_slices((x, y)).batch(2)
    train_model(
        config, {"train": dataset, "validation": dataset}, layout.root,
        layout.training_log, run_id, segment_stop_epoch=1, require_recovery=True,
    )
    state = validate_recovery_state(
        layout.root, config, artifact_id=artifact_id, run_id=run_id
    )
    callback_state = read_recovery_callback_state(layout.root, state)
    assert callback_state["early_stopping"]["best_weights"]
    assert "reduce_lr_on_plateau" in callback_state
    # A second segment loads the counters and weight snapshot rather than
    # silently restarting callback-managed behavior.
    _model, history = train_model(
        config, {"train": dataset, "validation": dataset}, layout.root,
        layout.training_log, run_id, resume_state=state,
    )
    assert len(history.history["loss"]) == 2


def test_cpu_epoch_resume_matches_uninterrupted_training(tmp_path):
    config = recovery_config()
    config["distribution"] = {"strategy": "cpu"}
    config["callbacks"] = {"early_stopping": False, "reduce_lr_on_plateau": False}
    x = np.random.default_rng(41).random((4, 16, 16, 1), dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype="int64")
    dataset = __import__("tensorflow").data.Dataset.from_tensor_slices((x, y)).batch(2)

    full = RunLayout(tmp_path / "full")
    full.create_directories()
    init_training_log(full.training_log, config["run"]["run_id"], "full", config, {})
    uninterrupted, _history = train_model(
        config, {"train": dataset, "validation": dataset}, full.root,
        full.training_log, config["run"]["run_id"],
    )

    segmented_config = copy.deepcopy(config)
    segmented_config["run"]["run_id"] = str(uuid.uuid4())
    segmented_config["artifact"]["artifact_id"] = str(uuid.uuid4())
    segmented = RunLayout(tmp_path / "segmented")
    segmented.create_directories()
    init_training_log(
        segmented.training_log, segmented_config["run"]["run_id"],
        "segmented", segmented_config, {},
    )
    train_model(
        segmented_config, {"train": dataset, "validation": dataset}, segmented.root,
        segmented.training_log, segmented_config["run"]["run_id"],
        segment_stop_epoch=1, require_recovery=True,
    )
    state = validate_recovery_state(
        segmented.root, segmented_config,
        artifact_id=segmented_config["artifact"]["artifact_id"],
        run_id=segmented_config["run"]["run_id"],
    )
    resumed, _history = train_model(
        segmented_config, {"train": dataset, "validation": dataset}, segmented.root,
        segmented.training_log, segmented_config["run"]["run_id"], resume_state=state,
    )
    for expected, observed in zip(uninterrupted.get_weights(), resumed.get_weights(), strict=True):
        np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7)


def test_deterministic_batch_cursor_resumes_mid_epoch_on_cpu(tmp_path):
    config = recovery_config()
    config["distribution"] = {"strategy": "cpu"}
    config["training"]["step_cursor_policy"] = "ordered_batches_v1"
    config["augmentation"] = {"enabled": False}
    config["callbacks"] = {"early_stopping": False, "reduce_lr_on_plateau": False}
    x = np.random.default_rng(99).random((4, 16, 16, 1), dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype="int64")
    dataset = __import__("tensorflow").data.Dataset.from_tensor_slices((x, y)).batch(2)

    full = RunLayout(tmp_path / "step-full")
    full.create_directories()
    init_training_log(full.training_log, config["run"]["run_id"], "step-full", config, {})
    uninterrupted, _ = train_model(
        config, {"train": dataset}, full.root, full.training_log,
        config["run"]["run_id"], segment_max_steps=2,
    )

    resumed_config = copy.deepcopy(config)
    resumed_config["run"]["run_id"] = str(uuid.uuid4())
    resumed_config["artifact"]["artifact_id"] = str(uuid.uuid4())
    segmented = RunLayout(tmp_path / "step-segmented")
    segmented.create_directories()
    init_training_log(
        segmented.training_log, resumed_config["run"]["run_id"],
        "step-segmented", resumed_config, {},
    )
    train_model(
        resumed_config, {"train": dataset}, segmented.root, segmented.training_log,
        resumed_config["run"]["run_id"], segment_max_steps=1,
    )
    state = validate_recovery_state(
        segmented.root, resumed_config,
        artifact_id=resumed_config["artifact"]["artifact_id"],
        run_id=resumed_config["run"]["run_id"],
    )
    assert state["cursor"]["batch_offset"] == 1
    assert state["continuation"]["data_order"] == "pinned_ordered_batches_v1"
    resumed, _ = train_model(
        resumed_config, {"train": dataset}, segmented.root, segmented.training_log,
        resumed_config["run"]["run_id"], resume_state=state,
        segment_max_steps=1, segment_start_step=1,
    )
    final_state = validate_recovery_state(
        segmented.root, resumed_config,
        artifact_id=resumed_config["artifact"]["artifact_id"],
        run_id=resumed_config["run"]["run_id"],
    )
    assert final_state["cursor"] == {
        "kind": "ordered_batches_v1", "epoch": 1, "batch_offset": 0, "global_step": 2,
    }
    for expected, observed in zip(uninterrupted.get_weights(), resumed.get_weights(), strict=True):
        np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7)
