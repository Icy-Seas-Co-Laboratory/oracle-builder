from __future__ import annotations

import json
import sqlite3
from oracle_builder.artifacts import read_run_config, read_run_manifest
from oracle_builder.data.sqlite_dataset import create_synthetic_classification
from oracle_builder.datasets.schema import set_dataset_lifecycle
from oracle_builder.training import TrainingRequest, run_training
from oracle_builder.training.recovery import validate_recovery_state


def test_imported_pretraining_initializes_and_resumes(tmp_path):
    database = tmp_path / "ssl.sqlite"
    create_synthetic_classification(database, n=4, shape=(16, 16, 1), classes=2)
    with sqlite3.connect(database) as connection:
        set_dataset_lifecycle(connection, "frozen")
        connection.commit()
    recipe = tmp_path / "ssl.toml"
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

[training]
epochs = 2
loss = "sparse_categorical_crossentropy"
display = "off"

[self_supervised]
enabled = true
method = "byol"
epochs = 1
projection_dim = 8
projection_hidden_dim = 16

[distribution]
strategy = "cpu"

[augmentation]
enabled = false

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
    handoff_path = tmp_path / "handoff.json"
    assert run_training(TrainingRequest(
        config=str(recipe), input=str(database), output="ssl", runs_dir=str(runs),
        pretrain_only=True, segment_result_path=str(handoff_path),
    )) == 0
    run_dir = runs / "ssl"
    handoff = json.loads(handoff_path.read_text())
    assert handoff["phase"] == "initialize"
    assert handoff["next_phase"] == "train"
    state = validate_recovery_state(run_dir, read_run_config(run_dir),
                                    artifact_id=read_run_manifest(run_dir)["artifact_id"],
                                    run_id=read_run_manifest(run_dir)["run_id"])
    assert state["pretraining_completed"] is True

    source_id = read_run_manifest(run_dir)["artifact_id"]
    assert run_training(TrainingRequest(
        config=str(recipe), input=str(database), output="imported", runs_dir=str(runs),
        pretrain_only=True, pretraining_source=str(run_dir),
        pretraining_source_artifact_id=source_id,
    )) == 0
    imported = runs / "imported"
    manifest = read_run_manifest(imported)
    state = validate_recovery_state(imported, read_run_config(imported),
                                    artifact_id=manifest["artifact_id"], run_id=manifest["run_id"])
    assert state["pretraining_handoff"]["source_artifact_id"] == source_id
    assert run_training(TrainingRequest(resume=str(imported), input=str(database),
                                        segment_start_epoch=0, segment_stop_epoch=1)) == 0
    state = validate_recovery_state(imported, read_run_config(imported),
                                    artifact_id=manifest["artifact_id"], run_id=manifest["run_id"])
    assert state["pretraining_completed"] is True
    assert state["pretraining_handoff"]["source_artifact_id"] == source_id
