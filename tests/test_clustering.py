from __future__ import annotations

import numpy as np

from oracle_builder.clustering.evidence import ClusterEvidenceIndex
from oracle_builder.clustering.training import train_clustering_run
from oracle_builder.data.sqlite_dataset import create_synthetic_classification
from oracle_builder.datasets.schema import set_dataset_lifecycle
from oracle_builder.inference.bundle import InferenceBundle
from oracle_builder.inference.contracts import InferenceItem, ModelReference


def test_cluster_index_fits_persists_and_returns_structure(tmp_path):
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.2, 0.8, 0.0],
        ],
        dtype="float32",
    )
    uuids = [f"roi-{index}" for index in range(len(embeddings))]
    index = ClusterEvidenceIndex.fit(embeddings, uuids, n_clusters=2)

    assert index.cluster_count == 2
    assert sum(index.cluster_sizes) == len(embeddings)
    packet = index.packet(np.asarray([1.0, 0.0, 0.0], dtype="float32"), top_k=2)
    assert packet["type"] == "roi_clustering"
    assert packet["decision"]["cluster_id"] == "cluster-0000" or packet["decision"]["cluster_id"] == "cluster-0001"
    assert len(packet["clusters"]) == 2
    assert packet["structure"]["cluster_count"] == 2

    path = tmp_path / "clustering_evidence"
    index.save(path)
    loaded = ClusterEvidenceIndex.load(path)
    np.testing.assert_allclose(loaded.centroids, index.centroids)
    assert loaded.exemplar("roi-0")["cluster_id"] == index.exemplar("roi-0")["cluster_id"]


class _EmbeddingModel:
    def predict_outputs(self, values, verbose=0):
        del verbose
        features = np.asarray(values, dtype="float32").reshape(len(values), -1)
        features = features[:, :2]
        return {
            "logits": np.zeros((len(values), 1), dtype="float32"),
            "probabilities": np.ones((len(values), 1), dtype="float32"),
            "features": features,
            "logits_source": "model",
        }


def test_clustering_bundle_returns_ml_evidence_packet():
    index = ClusterEvidenceIndex.fit(
        np.asarray([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype="float32"),
        ["a", "b", "c", "d"],
        n_clusters=2,
    )
    bundle = InferenceBundle(
        _EmbeddingModel(),
        {
            "run": {"task": "clustering", "model": "test"},
            "data": {"input_shape": [1, 2, 1]},
            "model": {"normalize_embeddings": True},
            "clustering": {"top_k": 2},
        },
        ModelReference("artifact", "run", "clustering", "test"),
        cluster_index=index,
    )
    result = bundle.predict(InferenceItem.from_array(np.asarray([[[1.0], [0.0]]], dtype="float32")))
    assert result.status == "ok"
    assert result.output["type"] == "clustering"
    assert result.output["evidence"]["type"] == "roi_clustering"
    assert "decision" in result.output["evidence"]


def test_classifier_can_return_secondary_clustering_evidence():
    index = ClusterEvidenceIndex.fit(
        np.asarray([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype="float32"),
        ["a", "b", "c", "d"],
        n_clusters=2,
    )
    bundle = InferenceBundle(
        _EmbeddingModel(),
        {
            "run": {"task": "classification", "model": "test"},
            "data": {"input_shape": [1, 2, 1]},
            "model": {"normalize_embeddings": True},
            "dataset": {"labels": []},
            "clustering": {"enabled": True, "top_k": 2},
        },
        ModelReference("artifact", "run", "classification", "test"),
        cluster_index=index,
    )
    result = bundle.predict(
        InferenceItem.from_array(np.asarray([[[1.0], [0.0]]], dtype="float32"))
    )
    assert result.status == "ok"
    assert result.output["type"] == "classification"
    assert result.output["clustering_evidence"]["type"] == "roi_clustering"
    batch = bundle.predict_batch(
        [
            InferenceItem.from_array(np.asarray([[[1.0], [0.0]]], dtype="float32")),
            InferenceItem.from_array(np.asarray([[[0.0], [1.0]]], dtype="float32")),
        ]
    )
    assert all(item.status == "ok" for item in batch.results)
    assert all(
        item.output["clustering_evidence"]["type"] == "roi_clustering"
        for item in batch.results
    )


def test_clustering_training_packages_a_servable_artifact(tmp_path):
    database = tmp_path / "rois.sqlite"
    create_synthetic_classification(database, n=6, shape=(8, 8, 1), classes=2)
    import sqlite3

    with sqlite3.connect(database) as connection:
        set_dataset_lifecycle(connection, "frozen")
        connection.commit()
    config = tmp_path / "clustering.toml"
    config.write_text(
        """
[run]
task = "clustering"
model = "simple_cnn"
seed = 7

[data]
input_shape = [8, 8, 1]
batch_size = 2

[model]
base_filters = 4
embedding_dim = 4
normalize_embeddings = true

[pretraining]
method = "byol"
epochs = 1
learning_rate = 0.001
projection_dim = 4
projection_hidden_dim = 8

[pretraining.embedding_health]
# This smoke test intentionally trains only three tiny batches; it exercises
# packaging rather than representation quality.
enabled = false

[clustering]
n_clusters = 2
top_k = 2

[augmentation]
enabled = false
""".strip()
        + "\n"
    )
    output = tmp_path / "cluster-run"
    result = train_clustering_run(config, database, output)

    assert result["clustering"]["cluster_count"] == 2
    assert (output / "artifact.json").exists()
    assert (output / "model" / "clustering_evidence" / "metadata.json").exists()
    assert (output / "model" / "export_savedmodel").exists()
    bundle = InferenceBundle.load(output)
    result = bundle.predict(
        InferenceItem.from_array(np.zeros((8, 8, 1), dtype="float32"))
    )
    assert result.status == "ok"
    assert result.output["type"] == "clustering"
