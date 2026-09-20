from __future__ import annotations

import json
import sqlite3

import numpy as np

from oracle_builder.classification.stratified_data import (
    build_stratum_dataset,
    build_stratum_datasets,
    child_config,
    routed_index,
)
from oracle_builder.data.decoders import encode_npy
from oracle_builder.data.sqlite_dataset import create_synthetic_classification


def _config():
    return {
        "run": {"task": "classification", "seed": 123},
        "data": {
            "input_shape": [32, 32, 1], "num_classes": 2, "batch_size": 16,
            "validation_split": 0.0, "test_split": 0.0,
            "streaming": {"enabled": True, "reader_workers": 1, "prefetch_batches": 1},
        },
        "classification": {"stratification": {
            "enabled": True, "dimensions": [32, 64],
            "training_routing": {"enabled": True, "adjacent_lower_probability": 1.0, "seed": 5},
        }},
        "preprocessing": {"resize_mode": "fit_pad", "channel_mode": "grayscale"},
        "augmentation": {"enabled": False},
    }


def _write_native_shapes(path):
    shapes = [(20, 30, 1), (20, 36, 1), (45, 60, 1), (75, 20, 1)]
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT image_asset_id FROM classification_items ORDER BY item_id"
        ).fetchall()
        for asset_id, shape in zip(rows, shapes, strict=True):
            array = np.full(shape, 127, dtype=np.uint8)
            connection.execute(
                "UPDATE assets SET payload = ?, encoding = 'npy', shape_json = ? WHERE asset_id = ?",
                (encode_npy(array), json.dumps(list(shape)), asset_id[0]),
            )


def test_child_config_uses_resolution_batch_plan_without_parent_mutation():
    config = _config()
    child = child_config(config, 64)
    assert child["data"]["input_shape"] == [64, 64, 1]
    assert child["data"]["batch_size"] == 4
    assert child["classification"]["stratification"]["_active_dimension"] == 64
    assert config["data"]["input_shape"] == [32, 32, 1]


def test_canonical_and_stochastic_indices_route_each_item_once(tmp_path):
    path = tmp_path / "classification.sqlite"
    create_synthetic_classification(path, n=4, shape=(8, 8, 1), classes=2)
    _write_native_shapes(path)
    config = _config()
    canonical = [routed_index(path, config, "train", dimension) for dimension in (32, 64)]
    stochastic = [
        routed_index(path, config, "train", dimension, routing_mode="training_stochastic", epoch=0)
        for dimension in (32, 64)
    ]
    assert [len(index) for index in canonical] == [1, 3]
    assert [len(index) for index in stochastic] == [4, 0]
    assert {ref.uuid for index in canonical for ref in index.refs} == {
        ref.uuid for index in stochastic for ref in index.refs
    }


def test_stratum_dataset_resizes_and_keeps_metadata_named_input(tmp_path):
    path = tmp_path / "classification.sqlite"
    create_synthetic_classification(path, n=4, shape=(8, 8, 1), classes=2)
    _write_native_shapes(path)
    config = _config()
    config["model"] = {"auxiliary_features": [{
        "name": "log_area", "source": "roi.bounding_box_area_px", "transform": "log",
        "standardize": True,
    }], "auxiliary_features_fitted": [{
        "name": "log_area", "source": "roi.bounding_box_area_px", "transform": "log",
        "standardize": True, "mean": 0.0, "scale": 1.0,
    }]}
    bundle = build_stratum_dataset(path, config, "train", 64, augment=False, shuffle=False)
    inputs, labels = next(iter(bundle.dataset))
    assert bundle.count == 3
    assert bundle.config["data"]["batch_size"] == 4
    assert inputs["image"].shape == (3, 64, 64, 1)
    assert inputs["metadata"].shape == (3, 1)
    assert labels.shape == (3,)
    all_bundles = build_stratum_datasets(path, config, "train", include_empty=False)
    assert set(all_bundles) == {32, 64}


def test_stochastic_routing_rejects_evaluation_split(tmp_path):
    path = tmp_path / "classification.sqlite"
    create_synthetic_classification(path, n=4, shape=(8, 8, 1), classes=2)
    _write_native_shapes(path)
    try:
        routed_index(path, _config(), "validation", 32, routing_mode="training_stochastic", epoch=0)
    except ValueError as exc:
        assert "only valid for the train split" in str(exc)
    else:
        raise AssertionError("Expected validation stochastic routing to be rejected")
