from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from oracle_builder.clustering.training import (
    _frozen_classification_index,
    _validate_pretraining_history,
    _validate_serving_embeddings,
    fit_clustering_evidence_from_encoder,
    resolve_clustering_config,
)
from oracle_builder.config import DEFAULT_CONFIG, deep_merge, validate_config
from oracle_builder.data.sqlite_dataset import create_synthetic_classification
from oracle_builder.datasets.schema import set_dataset_lifecycle


def _write_config(path, clustering: str) -> None:
    path.write_text(
        """
[run]
model = "simple_cnn"

[data]
input_shape = [8, 8, 1]

[clustering]
""".strip()
        + "\n"
        + clustering
        + "\n"
    )


@pytest.mark.parametrize(
    ("clustering", "message"),
    [
        ('method = "hdbscan"', "clustering.method"),
        ("n_clusters = 1", "clustering.n_clusters"),
        ("top_k = 0", "clustering.top_k"),
        ("novelty_percentile = 101", "clustering.novelty_percentile"),
    ],
)
def test_clustering_config_rejects_invalid_options_before_training(
    tmp_path, clustering, message
):
    config_path = tmp_path / "clustering.toml"
    _write_config(config_path, clustering)

    with pytest.raises(ValueError, match=message):
        resolve_clustering_config(config_path, tmp_path / "unused.sqlite", tmp_path / "run")


def test_cluster_count_is_checked_against_frozen_roi_count(tmp_path):
    database = tmp_path / "rois.sqlite"
    create_synthetic_classification(database, n=3, shape=(8, 8, 1), classes=2)
    with sqlite3.connect(database) as connection:
        set_dataset_lifecycle(connection, "frozen")
        connection.commit()
    config_path = tmp_path / "clustering.toml"
    _write_config(config_path, "n_clusters = 4")
    config = resolve_clustering_config(config_path, database, tmp_path / "run")

    with pytest.raises(ValueError, match="must not exceed"):
        _frozen_classification_index(database, config)


def test_clustering_index_ignores_an_encoder_split_manifest(tmp_path):
    database = tmp_path / "rois.sqlite"
    create_synthetic_classification(database, n=3, shape=(8, 8, 1), classes=2)
    with sqlite3.connect(database) as connection:
        set_dataset_lifecycle(connection, "frozen")
        connection.commit()
    config_path = tmp_path / "clustering.toml"
    _write_config(config_path, "n_clusters = 2")
    config = resolve_clustering_config(config_path, database, tmp_path / "run")
    config["_split_manifest"] = {"assignments": {"an-unrelated-item": "train"}}

    index = _frozen_classification_index(database, config)

    assert len(index.refs) == 3
    assert {ref.split for ref in index.refs} == {"inference"}


def test_existing_encoder_attachment_requires_explicit_reopen_and_reseal(tmp_path):
    with pytest.raises(ValueError, match="reopen_and_reseal=True"):
        fit_clustering_evidence_from_encoder(
            tmp_path / "clustering.toml",
            tmp_path / "rois.sqlite",
            tmp_path / "encoder-run",
        )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("minimum_global_batch_size", 1, "minimum_global_batch_size"),
        ("collapse_std_threshold", -1, "collapse_std_threshold"),
    ],
)
def test_ssl_config_rejects_unsafe_robustness_settings(key, value, message):
    config = deep_merge(DEFAULT_CONFIG, {"run": {"task": "classification", "model": "simple_cnn"}, "data": {"input_shape": [8, 8, 1], "num_classes": 2}, "preprocessing": {"invert": False}, "pretraining": {"enabled": True, key: value}})
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_clustering_rejects_missing_or_collapsed_ssl_diagnostics():
    config = deep_merge(
        DEFAULT_CONFIG,
        {"pretraining": {"collapse_std_threshold": 1e-3}},
    )

    class History:
        history = {"loss": [1.0]}

    with pytest.raises(ValueError, match="representation_std"):
        _validate_pretraining_history(History(), config)

    History.history = {"representation_std": [0.0]}
    with pytest.raises(ValueError, match="collapsed"):
        _validate_pretraining_history(History(), config)


def test_clustering_rejects_collapsed_serving_embeddings():
    config = deep_merge(
        DEFAULT_CONFIG,
        {"pretraining": {"collapse_embedding_spread_threshold": 0.005}},
    )
    collapsed = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"), (8, 1))

    with pytest.raises(ValueError, match="collapsed serving embeddings"):
        _validate_serving_embeddings(collapsed, config)
