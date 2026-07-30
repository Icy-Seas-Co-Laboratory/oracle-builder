from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from oracle_builder.config import load_toml, resolve_config
from oracle_builder.data.sqlite_dataset import create_synthetic_classification
from oracle_builder.datasets.schema import set_dataset_lifecycle


def freeze(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        set_dataset_lifecycle(connection, "frozen")
        connection.commit()


def test_resolve_config_adds_defaults_and_paths(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    input_path = tmp_path / "data.sqlite"
    run_dir = tmp_path / "runs" / "test"
    config_path.write_text(
        """
[run]
task = "classification"
model = "simple_cnn"

[data]
input_shape = [16, 16, 1]
num_classes = 2

[training]
loss = "sparse_categorical_crossentropy"
"""
    )
    create_synthetic_classification(input_path, n=4, shape=(16, 16, 1), classes=2)
    freeze(input_path)
    config = resolve_config(config_path, input_path, run_dir)
    assert config["data"]["batch_size"] == 16
    assert config["inference"]["batch_size"] == "auto"
    assert config["inference"]["maximum_batch_size"] == 64
    assert config["run"]["seed"] == 123
    assert config["paths"]["run_dir"] == str(run_dir.resolve())


def test_resolve_config_infers_class_count_from_sqlite(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    input_path = tmp_path / "data.sqlite"
    run_dir = tmp_path / "runs" / "test"
    create_synthetic_classification(input_path, n=6, shape=(16, 16, 1), classes=3)
    freeze(input_path)
    config_path.write_text(
        """
[run]
task = "classification"
model = "simple_cnn"

[data]
input_shape = [16, 16, 1]

[training]
loss = "sparse_categorical_crossentropy"
"""
    )

    config = resolve_config(config_path, input_path, run_dir)

    assert config["data"]["num_classes"] == 3


def test_classification_defaults_to_weighted_loss_grayscale_and_no_checkpoints(
    tmp_path: Path,
):
    config_path = tmp_path / "minimal.toml"
    input_path = tmp_path / "data.sqlite"
    create_synthetic_classification(input_path, n=6, shape=(16, 16, 1), classes=3)
    freeze(input_path)
    config_path.write_text(
        """
[run]
task = "classification"
model = "simple_cnn"

[data]
input_shape = [16, 16, 1]

[training]
epochs = 1
"""
    )

    config = resolve_config(config_path, input_path, tmp_path / "runs" / "test")

    assert config["training"]["loss"] == (
        "weighted_sparse_categorical_crossentropy"
    )
    assert config["training"]["class_weights"]["mode"] == "effective_number"
    assert config["preprocessing"]["channel_mode"] == "grayscale"
    assert config["output"]["save_checkpoints"] is False


@pytest.mark.parametrize(
    ("filename", "model"),
    [
        ("simple_cnn.toml", "simple_cnn"),
        ("resnet_like.toml", "resnet_like"),
        ("densenet_like.toml", "densenet_like"),
        ("resnet.toml", "resnet"),
        ("densenet.toml", "densenet"),
        ("efficientnet.toml", "efficientnet"),
    ],
)
def test_classification_default_configs_resolve(
    tmp_path: Path,
    filename: str,
    model: str,
):
    input_path = tmp_path / "data.sqlite"
    create_synthetic_classification(
        input_path,
        n=9,
        shape=(16, 16, 3),
        classes=3,
    )
    freeze(input_path)
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "classification_defaults"
        / filename
    )

    config = resolve_config(
        config_path,
        input_path,
        tmp_path / "runs" / model,
    )

    assert config["run"]["model"] == model
    assert config["data"]["num_classes"] == 3


def test_classification_default_configs_share_augmentation():
    directory = (
        Path(__file__).parents[1]
        / "configs"
        / "classification_defaults"
    )
    augmentations = [
        load_toml(path)["augmentation"]
        for path in sorted(directory.glob("*.toml"))
    ]

    assert len(augmentations) == 6
    assert all(value == augmentations[0] for value in augmentations[1:])
