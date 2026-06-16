from __future__ import annotations

from pathlib import Path

from oracle_builder.data.sqlite_dataset import (
    create_synthetic_classification,
    create_synthetic_segmentation,
    load_arrays,
    make_tf_datasets,
    synthetic_splits,
)


def test_load_synthetic_classification_arrays(tmp_path: Path):
    db_path = tmp_path / "classification.sqlite"
    create_synthetic_classification(db_path, n=12, shape=(16, 16, 3), classes=3)
    config = {
        "run": {"task": "classification", "seed": 123},
        "data": {"input_shape": [16, 16, 3], "num_classes": 3, "validation_split": 0.2, "test_split": 0.1},
    }
    x, y, records = load_arrays(db_path, config, split="train")
    assert x.shape[1:] == (16, 16, 3)
    assert y.ndim == 1
    assert len(records) == len(y)


def test_synthetic_splits_always_match_requested_count():
    assert len(synthetic_splits(48)) == 48
    assert len(synthetic_splits(24)) == 24
    assert len(synthetic_splits(5)) == 5


def test_create_default_synthetic_segmentation_dataset(tmp_path: Path):
    db_path = tmp_path / "segmentation.sqlite"
    create_synthetic_segmentation(db_path)
    config = {
        "run": {"task": "segmentation", "seed": 123},
        "data": {
            "input_shape": [256, 256, 3],
            "output_shape": [256, 256, 1],
            "validation_split": 0.2,
            "test_split": 0.1,
        },
    }
    x, y, records = load_arrays(db_path, config, split="train")
    assert x.shape[1:] == (256, 256, 3)
    assert y.shape[1:] == (256, 256, 1)
    assert len(records) == len(y)


def test_training_dataset_can_repeat_augmented_samples_per_epoch(tmp_path: Path):
    db_path = tmp_path / "classification.sqlite"
    create_synthetic_classification(db_path, n=10, shape=(8, 8, 3), classes=2)
    config = {
        "run": {"task": "classification", "seed": 123},
        "data": {
            "input_shape": [8, 8, 3],
            "num_classes": 2,
            "batch_size": 4,
            "shuffle_buffer": 16,
            "validation_split": 0.0,
            "test_split": 0.0,
        },
        "augmentation": {
            "enabled": True,
            "repeats_per_epoch": 3,
            "invert": True,
        },
    }

    datasets, records_by_split = make_tf_datasets(db_path, config)
    sample_count = len(records_by_split["train"])
    observed_count = sum(int(batch_x.shape[0]) for batch_x, _ in datasets["train"])

    assert sample_count == 7
    assert observed_count == sample_count * 3
