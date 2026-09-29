from __future__ import annotations

import json
import sqlite3

import numpy as np

from oracle_builder.artifacts import read_run_config, read_run_manifest, validate_run_artifact
from oracle_builder.data.sqlite_dataset import create_synthetic_classification
from oracle_builder.datasets.schema import set_dataset_lifecycle
from oracle_builder.saving.load_test import _load_keras_model
from oracle_builder.training import TrainingRequest, run_training
from oracle_builder.training.recovery import validate_recovery_state


def test_real_workflow_commits_and_resumes_an_ordered_batch_segment(tmp_path):
    database = tmp_path / "data.sqlite"
    create_synthetic_classification(database, n=4, shape=(16, 16, 1), classes=2)
    with sqlite3.connect(database) as connection:
        set_dataset_lifecycle(connection, "frozen")
        connection.commit()
    recipe = tmp_path / "step.toml"
    recipe.write_text(
        """
[run]
task = "classification"
model = "simple_cnn"

[data]
input_shape = [16, 16, 1]
batch_size = 2
validation_split = 0.0
test_split = 0.0

[data.streaming]
enabled = false

[training]
epochs = 1
loss = "sparse_categorical_crossentropy"
display = "off"
step_cursor_policy = "ordered_batches_v1"
work_unit_steps = 1

[distribution]
strategy = "cpu"

[augmentation]
enabled = false
repeats_per_epoch = 1

[callbacks]
early_stopping = false
reduce_lr_on_plateau = false

[evidence]
enabled = false

[output]
save_predictions = false
"""
    )
    runs = tmp_path / "runs"
    result = tmp_path / "first-result.json"
    assert run_training(TrainingRequest(
        config=str(recipe), input=str(database), output="step", runs_dir=str(runs),
        segment_start_epoch=0, segment_max_steps=1, segment_result_path=str(result),
    )) == 0
    run_dir = runs / "step"
    first = json.loads(result.read_text())
    assert first["cursor"]["batch_offset"] == 1
    assert first["next_phase"] == "train"
    assert read_run_manifest(run_dir)["status"] == "interrupted"
    assert validate_run_artifact(run_dir)["valid"]

    second_result = tmp_path / "second-result.json"
    assert run_training(TrainingRequest(
        resume=str(run_dir), input=str(database), segment_start_epoch=0,
        segment_start_step=1, segment_max_steps=1,
        segment_result_path=str(second_result),
    )) == 0
    second = json.loads(second_result.read_text())
    assert second["cursor"]["completed_epoch"] == 1
    assert second["cursor"]["batch_offset"] == 0
    assert second["next_phase"] == "finalize"
    assert validate_run_artifact(run_dir)["valid"]


def test_real_workflow_finalizes_an_early_stopped_epoch_segment(tmp_path):
    database = tmp_path / "early.sqlite"
    create_synthetic_classification(database, n=12, shape=(16, 16, 1), classes=2)
    with sqlite3.connect(database) as connection:
        set_dataset_lifecycle(connection, "frozen")
        connection.commit()
    recipe = tmp_path / "early.toml"
    recipe.write_text(
        """
[run]
task = "classification"
model = "simple_cnn"

[data]
input_shape = [16, 16, 1]
batch_size = 2
validation_split = 0.5
test_split = 0.0
split_minimum_per_class = 0

[data.streaming]
enabled = false

[training]
epochs = 3
loss = "sparse_categorical_crossentropy"
optimizer = "sgd"
learning_rate = 0.00000001
display = "off"

[distribution]
strategy = "cpu"

[augmentation]
enabled = false

[callbacks]
early_stopping = true
early_stopping_patience = 1
early_stopping_baseline = -1.0
reduce_lr_on_plateau = false

[evidence]
enabled = false

[output]
save_predictions = false
"""
    )
    runs = tmp_path / "runs"
    segment_result = tmp_path / "early-segment.json"
    assert run_training(TrainingRequest(
        config=str(recipe), input=str(database), output="early", runs_dir=str(runs),
        segment_start_epoch=0, segment_stop_epoch=3,
        segment_result_path=str(segment_result),
    )) == 0
    handoff = json.loads(segment_result.read_text())
    assert handoff["stopped_early"] is True
    assert handoff["next_phase"] == "finalize"
    run_dir = runs / "early"
    assert read_run_manifest(run_dir)["status"] == "interrupted"
    assert run_training(TrainingRequest(
        resume=str(run_dir), input=str(database), finalize_only=True,
    )) == 0
    assert read_run_manifest(run_dir)["status"] == "complete"
    assert validate_run_artifact(run_dir)["valid"]


def test_cpu_ordered_batch_segments_match_uninterrupted_model_and_optimizer(tmp_path):
    """The supported step cursor must preserve the complete optimizer state.

    This deliberately uses the narrow deterministic execution surface: CPU,
    fixed ordered batches, no augmentation or callbacks, and no dropout.
    The reference still goes through ``run_training`` and its ordinary
    ``model.fit`` epoch path; only the comparison run is split into one-batch
    work units and resumed through durable artifacts.
    """
    database = tmp_path / "equivalence.sqlite"
    create_synthetic_classification(database, n=4, shape=(16, 16, 1), classes=2)
    with sqlite3.connect(database) as connection:
        set_dataset_lifecycle(connection, "frozen")
        connection.commit()
    recipe = tmp_path / "equivalence.toml"
    recipe.write_text(
        """
[run]
task = "classification"
model = "simple_cnn"
seed = 817

[data]
input_shape = [16, 16, 1]
batch_size = 2
validation_split = 0.0
test_split = 0.0

[data.streaming]
enabled = false

[training]
epochs = 2
loss = "sparse_categorical_crossentropy"
optimizer = "adam"
learning_rate = 0.001
display = "off"
step_cursor_policy = "ordered_batches_v1"
work_unit_steps = 1

[distribution]
strategy = "cpu"

[augmentation]
enabled = false
repeats_per_epoch = 1

[callbacks]
early_stopping = false
reduce_lr_on_plateau = false

[model]
dropout = 0.0

[evidence]
enabled = false

[output]
save_predictions = false
"""
    )
    runs = tmp_path / "runs"

    reference_result = tmp_path / "reference-segment.json"
    assert run_training(TrainingRequest(
        config=str(recipe), input=str(database), output="reference", runs_dir=str(runs),
        segment_start_epoch=0, segment_stop_epoch=2,
        segment_result_path=str(reference_result),
    )) == 0
    reference_dir = runs / "reference"
    reference_manifest = read_run_manifest(reference_dir)
    reference_config = read_run_config(reference_dir)
    reference_state = validate_recovery_state(
        reference_dir, reference_config,
        artifact_id=reference_manifest["artifact_id"], run_id=reference_manifest["run_id"],
    )
    reference_model = _load_keras_model(reference_dir / reference_state["model_path"], compile=True)

    result_path = tmp_path / "segmented-result.json"
    assert run_training(TrainingRequest(
        config=str(recipe), input=str(database), output="segmented", runs_dir=str(runs),
        segment_start_epoch=0, segment_max_steps=1, segment_result_path=str(result_path),
    )) == 0
    segmented_dir = runs / "segmented"
    result = json.loads(result_path.read_text())
    while result["next_phase"] == "train":
        cursor = result["cursor"]
        assert run_training(TrainingRequest(
            resume=str(segmented_dir), input=str(database),
            segment_start_epoch=int(cursor["completed_epoch"]),
            segment_start_step=int(cursor["batch_offset"]),
            segment_max_steps=1, segment_result_path=str(result_path),
        )) == 0
        result = json.loads(result_path.read_text())
    assert result["next_phase"] == "finalize"

    segmented_manifest = read_run_manifest(segmented_dir)
    segmented_config = read_run_config(segmented_dir)
    segmented_state = validate_recovery_state(
        segmented_dir, segmented_config,
        artifact_id=segmented_manifest["artifact_id"], run_id=segmented_manifest["run_id"],
    )
    segmented_model = _load_keras_model(segmented_dir / segmented_state["model_path"], compile=True)

    assert int(reference_model.optimizer.iterations.numpy()) == 4
    assert int(segmented_model.optimizer.iterations.numpy()) == 4
    for expected, observed in zip(reference_model.get_weights(), segmented_model.get_weights(), strict=True):
        np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7)
    for expected, observed in zip(reference_model.optimizer.variables, segmented_model.optimizer.variables, strict=True):
        np.testing.assert_allclose(observed.numpy(), expected.numpy(), rtol=1e-6, atol=1e-7)
