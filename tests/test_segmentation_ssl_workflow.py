from __future__ import annotations

import json
import sqlite3

from oracle_builder.artifacts import read_run_config, read_run_manifest
from oracle_builder.data.sqlite_dataset import create_synthetic_segmentation
from oracle_builder.datasets.schema import set_dataset_lifecycle
from oracle_builder.training import TrainingRequest, run_training
from oracle_builder.training.recovery import validate_recovery_state


def test_segmentation_ssl_initialization_resumes_into_supervised_training(tmp_path):
    database = tmp_path / "segmentation.sqlite"
    create_synthetic_segmentation(database, n=4, shape=(16, 16, 1))
    with sqlite3.connect(database) as connection:
        set_dataset_lifecycle(connection, "frozen")
        connection.commit()

    recipe = tmp_path / "segmentation-ssl.toml"
    recipe.write_text(
        """
[run]
task = "segmentation"
model = "unet"

[data]
input_shape = [16, 16, 1]
output_shape = [16, 16, 1]
batch_size = 2
validation_split = 0.0
test_split = 0.0

[data.streaming]
enabled = false

[model]
base_filters = 2
depth = 2
dropout = 0.0
final_activation = "sigmoid"

[training]
epochs = 1
optimizer = "adam"
learning_rate = 0.001
loss = "bce_soft_dice"
segmentation_target = "validated_mask"
display = "off"

[self_supervised]
enabled = true
method = "grayscale_reconstruction"
epochs = 1
learning_rate = 0.001
reconstruction_foreground_weight = 2.0

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
save_figures = false
export_savedmodel = false
"""
    )
    runs = tmp_path / "runs"
    initialization_result = tmp_path / "initialization-result.json"

    assert run_training(
        TrainingRequest(
            config=str(recipe),
            input=str(database),
            output="segmentation-ssl",
            runs_dir=str(runs),
            pretrain_only=True,
            segment_result_path=str(initialization_result),
        )
    ) == 0

    run_dir = runs / "segmentation-ssl"
    initialization = json.loads(initialization_result.read_text())
    assert initialization["phase"] == "initialize"
    assert initialization["next_phase"] == "train"
    manifest = read_run_manifest(run_dir)
    recovery = validate_recovery_state(
        run_dir,
        read_run_config(run_dir),
        artifact_id=manifest["artifact_id"],
        run_id=manifest["run_id"],
    )
    assert recovery["pretraining_completed"] is True

    supervised_result = tmp_path / "supervised-result.json"
    assert run_training(
        TrainingRequest(
            resume=str(run_dir),
            input=str(database),
            segment_start_epoch=0,
            segment_stop_epoch=1,
            segment_result_path=str(supervised_result),
        )
    ) == 0
    supervised = json.loads(supervised_result.read_text())
    assert supervised["phase"] == "train"
    assert supervised["next_phase"] == "finalize"

