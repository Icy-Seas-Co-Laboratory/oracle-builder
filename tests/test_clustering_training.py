from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from oracle_builder.clustering.training import (
    _frozen_classification_index,
    _serving_embedding_health,
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


def test_serving_embedding_health_reports_healthy_geometry():
    config = deep_merge(DEFAULT_CONFIG, {"pretraining": {"embedding_health": {"enabled": True}}})
    embeddings = np.eye(16, dtype="float32")

    diagnostics = _serving_embedding_health(embeddings, config)

    assert diagnostics["embedding_dimension"] == 16
    assert diagnostics["sample_count"] == 16
    # Centering removes the all-ones direction from this orthogonal set.
    assert diagnostics["effective_rank"] == pytest.approx(15.0)
    assert diagnostics["mean_direction_norm"] == pytest.approx(0.25)
    assert diagnostics["mean_pairwise_cosine"] == pytest.approx(0.0)
    _validate_serving_embeddings(embeddings, config)


def test_serving_embedding_health_rejects_a_narrow_high_cosine_cone():
    rng = np.random.default_rng(123)
    embeddings = np.column_stack(
        [np.ones(64), 0.03 * rng.normal(size=(64, 15))]
    ).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "pretraining": {
                "collapse_embedding_spread_threshold": 0.005,
                "embedding_health": {
                    "enabled": True,
                    "minimum_effective_rank": 1.0,
                    "maximum_mean_pairwise_cosine": 0.90,
                    "maximum_p95_pairwise_cosine": 0.99999,
                    "maximum_mean_direction_norm": 1.0,
                },
            }
        },
    )

    diagnostics = _serving_embedding_health(embeddings, config)

    assert diagnostics["centered_rms"] > 0.005
    assert diagnostics["mean_pairwise_cosine"] > 0.90
    with pytest.raises(ValueError, match="mean_pairwise_cosine"):
        _validate_serving_embeddings(embeddings, config)
