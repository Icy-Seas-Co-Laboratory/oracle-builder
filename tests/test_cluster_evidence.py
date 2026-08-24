from __future__ import annotations

import json

import numpy as np
import pytest

from oracle_builder.clustering.evidence import ClusterEvidenceIndex


def _embeddings() -> np.ndarray:
    return np.asarray(
        [[1, 0], [.98, .02], [.95, .05], [0, 1], [.02, .98], [.05, .95]],
        dtype="float32",
    )


def test_fit_rejects_degenerate_embeddings():
    with pytest.raises(ValueError, match="non-empty clusters"):
        ClusterEvidenceIndex.fit(
            np.ones((4, 2), dtype="float32"), ["a", "b", "c", "d"], n_clusters=2
        )


def test_bounded_references_and_batch_assignment():
    index = ClusterEvidenceIndex.fit(
        _embeddings(),
        [f"roi-{i}" for i in range(6)],
        n_clusters=2,
        reference_neighbors_per_cluster=1,
        silhouette_max_samples=2,
    )
    assert len(index.embeddings) == 2
    assert sum(index.cluster_sizes) == 6
    assert index.summary()["reference_retention"]["mode"] == "per_cluster_top_similarity"
    packets = index.packet_many(_embeddings()[:2], include_neighbors=False)
    assert len(packets) == 2
    assert packets[0]["nearest_neighbors"] == []
    assert packets[0]["nearest_neighbors_retained"] is False


def test_large_fit_uses_bounded_minibatch_kmeans():
    index = ClusterEvidenceIndex.fit(
        _embeddings(),
        [f"roi-{i}" for i in range(6)],
        n_clusters=2,
        fit_batch_size=2,
    )
    assert index.method in {"spherical_minibatch_kmeans", "spherical_kmeans_numpy"}


def test_held_out_calibration_is_persisted(tmp_path):
    index = ClusterEvidenceIndex.fit(
        _embeddings(),
        [f"roi-{i}" for i in range(6)],
        n_clusters=2,
        calibration_embeddings=_embeddings()[:2],
        novelty_percentile=10,
    )
    assert index.packet(np.asarray([1, 0], dtype="float32"))["decision"]["novelty_status"] == "held_out_similarity_threshold"
    index.save(tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["schema_version"] == 2
    assert ClusterEvidenceIndex.load(tmp_path).summary()["novelty_calibration"]["status"] == "held_out"
