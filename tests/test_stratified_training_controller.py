from __future__ import annotations

import json
import sqlite3

import numpy as np

from oracle_builder.classification.stratified_training import (
    RECOVERY_SCHEMA,
    _sha256,
    child_config,
    epoch_stratum_schedule,
    recovery_config_hash,
    routed_index,
    supra_epoch_schedule,
    train_stratified_models,
    validate_recovery_state,
)
from oracle_builder.artifacts.layout import RunLayout
from oracle_builder.data.decoders import encode_npy
from oracle_builder.data.sqlite_dataset import create_synthetic_classification
from oracle_builder.data.sqlite_stream import SQLiteSampleRef, SQLiteSplitIndex
from oracle_builder.training.logging_callbacks import init_training_log


def _config():
    return {
        "run": {"task": "classification", "model": "simple_cnn", "seed": 7},
        "data": {
            "input_shape": [32, 32, 1],
            "batch_size": 16,
            "num_classes": 2,
        },
        "training": {"epochs": 3},
        "classification": {
            "stratification": {
                "enabled": True,
                "dimensions": [32, 64],
                "basis": "max_original_dimension",
                "batch_size_policy": "constant_input_tensor",
                "training_routing": {
                    "enabled": True,
                    "adjacent_lower_probability": 1.0,
                    "seed": 7,
                },
            }
        },
        "artifact": {"artifact_id": "artifact-1"},
    }


def _ref(item: str, shape: list[int]) -> SQLiteSampleRef:
    return SQLiteSampleRef(
        item_id=item,
        uuid=item,
        split="train",
        target=0,
        input_encoding="npy",
        input_dimensions=json.dumps(shape),
        metadata_json="{}",
    )


def test_child_configs_are_isolated_and_use_area_scaled_batches():
    config = _config()
    child = child_config(config, 64)
    assert child["data"]["input_shape"] == [64, 64, 1]
    assert child["data"]["batch_size"] == 4
    assert child["classification"]["stratification"]["_active_dimension"] == 64
    assert config["data"]["input_shape"] == [32, 32, 1]


def test_schedule_completes_every_stratum_before_advancing_parent_epoch():
    assert list(epoch_stratum_schedule(_config(), 2)) == [
        (0, 32), (0, 64), (1, 32), (1, 64),
    ]


def test_supra_epoch_schedule_keeps_each_child_active_for_the_configured_block():
    config = _config()
    config["classification"]["stratification"]["supra_epochs"] = 2
    assert list(supra_epoch_schedule(config, 5)) == [
        (0, 2, 32), (0, 2, 64),
        (2, 4, 32), (2, 4, 64),
        (4, 5, 32), (4, 5, 64),
    ]


def test_epoch_routing_assigns_every_reference_exactly_once(tmp_path):
    config = _config()
    base = SQLiteSplitIndex(
        tmp_path / "unused.sqlite",
        "train",
        [_ref("small", [20, 20, 1]), _ref("medium", [20, 36, 1])],
    )
    canonical = {
        ref.uuid
        for dimension in (32, 64)
        for ref in routed_index(base, config, dimension).refs
    }
    stochastic_lists = [
        routed_index(base, config, dimension, epoch=0).refs
        for dimension in (32, 64)
    ]
    stochastic = [ref.uuid for refs in stochastic_lists for ref in refs]
    assert set(stochastic) == canonical == {"small", "medium"}
    assert len(stochastic) == len(set(stochastic))
    assert [ref.uuid for ref in stochastic_lists[0]] == ["small", "medium"]
    assert stochastic_lists[1] == []


def test_stratified_recovery_validates_every_child_checksum(tmp_path):
    config = _config()
    target = tmp_path / "model" / "strata" / "32" / "recovery" / "latest.keras"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"model")
    state = {
        "schema": RECOVERY_SCHEMA,
        "artifact_id": "artifact-1",
        "run_id": "run-1",
        "config_sha256": recovery_config_hash(config),
        "dimensions": [32, 64],
        "weight_sharing": "shared",
        "children": {
            "32": {
                "completed_epochs": 1,
                "model_path": target.relative_to(tmp_path).as_posix(),
                "model_sha256": _sha256(target),
            }
        },
    }
    state["shared"] = dict(state["children"]["32"])
    recovery = tmp_path / "model" / "recovery" / "stratified_state.json"
    recovery.parent.mkdir(parents=True)
    recovery.write_text(json.dumps(state))
    assert validate_recovery_state(
        tmp_path,
        config,
        artifact_id="artifact-1",
        run_id="run-1",
    )["children"]["32"]["completed_epochs"] == 1


def test_tiny_end_to_end_run_persists_both_children(tmp_path):
    database = tmp_path / "classification.sqlite"
    create_synthetic_classification(database, n=4, shape=(8, 8, 1), classes=2)
    with sqlite3.connect(database) as connection:
        assets = connection.execute(
            "SELECT image_asset_id FROM classification_items ORDER BY item_id"
        ).fetchall()
        for position, (asset_id,) in enumerate(assets):
            side = 8 if position < 2 else 12
            array = np.full((side, side, 1), position / 4, dtype="float32")
            connection.execute(
                "UPDATE assets SET payload = ?, encoding = 'npy', shape_json = ? "
                "WHERE asset_id = ?",
                (encode_npy(array), json.dumps(list(array.shape)), asset_id),
            )
    config = _config()
    config["classification"]["stratification"]["dimensions"] = [8, 16]
    config["classification"]["stratification"]["training_routing"][
        "adjacent_lower_probability"
    ] = 0.0
    config["data"].update(
        {
            "input_shape": [8, 8, 1],
            "batch_size": 2,
            "validation_split": 0.0,
            "test_split": 0.0,
            "streaming": {
                "reader_workers": 1,
                "prefetch_batches": 1,
                "deterministic": True,
            },
        }
    )
    config["model"] = {"base_filters": 2, "embedding_dim": 4, "dropout": 0.0}
    config["training"].update(
        {
            "epochs": 1,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"],
        }
    )
    config["distribution"] = {"strategy": "cpu"}
    config["preprocessing"] = {
        "resize_mode": "fit_pad",
        "normalization": "dtype",
        "rescale": True,
        "channel_mode": "grayscale",
    }
    config["augmentation"] = {"enabled": False, "repeats_per_epoch": 1}
    config["callbacks"] = {"early_stopping": False}
    config["recovery"] = {"enabled": True, "save_every_epochs": 1}
    config["output"] = {"save_checkpoints": False}
    run_dir = tmp_path / "run"
    layout = RunLayout(run_dir)
    layout.create_directories()
    init_training_log(layout.training_log, "run-1", "tiny", config, {})

    result = train_stratified_models(
        config, database, run_dir, layout.training_log, "run-1"
    )

    assert set(result.children) == {8, 16}
    assert all(result.model_path(dimension).exists() for dimension in (8, 16))
    assert all(result.children[dimension].completed_epochs == 1 for dimension in (8, 16))
    assert result.manifest_path.exists()
    state = validate_recovery_state(
        run_dir, config, artifact_id="artifact-1", run_id="run-1"
    )
    assert state["shared"]["model_path"] == "model/recovery/latest.keras"
    assert (run_dir / state["shared"]["model_path"]).exists()

    # A process can be interrupted after its final rolling snapshot but before
    # outer run finalization.  Resuming then skips the already-completed
    # epochs, so it must still rebuild the public child manifest entries.
    resumed = train_stratified_models(
        config,
        database,
        run_dir,
        layout.training_log,
        "run-1",
        resume_state=state,
    )
    assert set(resumed.children) == {8, 16}
    assert resumed.manifest_path.exists()
