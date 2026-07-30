from __future__ import annotations

import numpy as np

from oracle_builder.classification.evidence import IdentityEvidenceIndex


def test_cosine_knn_packet_reports_requested_evidence_and_excludes_self():
    index = IdentityEvidenceIndex.build(
        embeddings=np.asarray(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype="float32",
        ),
        labels=np.asarray([0, 0, 1, 2]),
        uuids=["self", "same-class", "other-class", "opposite"],
    )

    packet = index.packet(
        np.asarray([1.0, 0.0]),
        np.asarray([0.7, 0.2, 0.1]),
        query_uuid="self",
        k=3,
    )

    assert packet["softmax"]["predicted_class"] == 0
    assert packet["prototype"]["predicted_class"] == 0
    assert packet["knn"]["k_used"] == 3
    assert packet["knn"]["nearest_neighbor_uuid"] == "same-class"
    assert np.isclose(packet["knn"]["nearest_neighbor_similarity"], 0.8)
    assert np.isclose(packet["knn"]["label_agreement"], 1 / 3)
    assert packet["knn"]["strongest_label"] == 0
    assert packet["knn"]["weighted_label_support"]["0"] > packet["knn"][
        "weighted_label_support"
    ]["1"]
    assert packet["knn"]["label_support_margin"] > 0
    assert all(neighbor["uuid"] != "self" for neighbor in packet["knn"]["neighbors"])


def test_normalized_cosine_and_euclidean_neighbor_rankings_match():
    rng = np.random.default_rng(123)
    references = rng.normal(size=(20, 8)).astype("float32")
    query = rng.normal(size=8).astype("float32")
    index = IdentityEvidenceIndex.build(
        references,
        np.arange(20) % 3,
        [f"roi-{index}" for index in range(20)],
    )
    normalized_query = query / np.linalg.norm(query)

    cosine_order = np.argsort(-(index.embeddings @ normalized_query))
    euclidean_order = np.argsort(
        np.sum((index.embeddings - normalized_query[None, :]) ** 2, axis=1)
    )

    np.testing.assert_array_equal(cosine_order, euclidean_order)


def test_evidence_index_round_trips_without_pickle(tmp_path):
    index = IdentityEvidenceIndex.build(
        np.eye(3, dtype="float32"),
        np.asarray([0, 1, 2]),
        ["a", "b", "c"],
    )
    path = tmp_path / "evidence.npz"

    index.save(path)
    loaded = IdentityEvidenceIndex.load(path)

    np.testing.assert_allclose(loaded.embeddings, index.embeddings)
    np.testing.assert_allclose(loaded.prototypes, index.prototypes)
    np.testing.assert_array_equal(loaded.uuids, index.uuids)
    assert path.with_suffix(".json").exists()
