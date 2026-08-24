from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# Version 1 artifacts remain readable. Version 2 records novelty provenance
# and optional bounded reference-neighbor storage.
CLUSTER_EVIDENCE_SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA_VERSIONS = {1, CLUSTER_EVIDENCE_SCHEMA_VERSION}


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float32")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def _numpy_spherical_kmeans(values: np.ndarray, n_clusters: int, *, seed: int, iterations: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Dependency-free spherical k-means fallback for lightweight hosts."""
    generator = np.random.default_rng(seed)
    centers = values[generator.choice(len(values), size=n_clusters, replace=False)].copy()
    assignments = np.full(len(values), -1, dtype="int64")
    for _ in range(iterations):
        updated = np.argmax(values @ centers.T, axis=1).astype("int64")
        next_centers = centers.copy()
        for cluster in range(n_clusters):
            selected = values[updated == cluster]
            if len(selected):
                next_centers[cluster] = _normalize(selected.mean(axis=0, keepdims=True))[0]
        centers = next_centers
        if np.array_equal(assignments, updated):
            assignments = updated
            break
        assignments = updated
    return assignments, centers


def _compact_clusters(assignments: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove empty clusters and re-number assignments contiguously."""
    sizes = np.bincount(assignments, minlength=len(centroids)).astype("int64")
    kept = np.flatnonzero(sizes > 0)
    if len(kept) < 2:
        raise ValueError("Clustering produced fewer than two non-empty clusters; reduce n_clusters or provide embeddings with more distinct structure")
    remap = np.full(len(centroids), -1, dtype="int64")
    remap[kept] = np.arange(len(kept), dtype="int64")
    return remap[assignments], centroids[kept], sizes[kept]


@dataclass
class ClusterEvidenceIndex:
    """Persistent cosine-space cluster evidence.

    ``embeddings`` can be a bounded set of representative references, while
    ``cluster_sizes`` always describes the full data used for fitting.
    """

    embeddings: np.ndarray
    uuids: np.ndarray
    assignments: np.ndarray
    centroids: np.ndarray
    cluster_sizes: np.ndarray
    cluster_similarity_floors: np.ndarray
    medoid_uuids: np.ndarray
    method: str = "spherical_kmeans"
    metric: str = "cosine_similarity"
    novelty_similarity_threshold: float = 0.0
    silhouette: float | None = None
    novelty_calibration: dict[str, Any] = field(default_factory=dict)
    source_reference_count: int | None = None
    reference_retention: dict[str, Any] = field(default_factory=dict)
    silhouette_sample_count: int | None = None

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        uuids: list[str] | np.ndarray,
        *,
        n_clusters: int,
        seed: int = 123,
        novelty_percentile: float = 5.0,
        calibration_embeddings: np.ndarray | None = None,
        silhouette_max_samples: int = 2_000,
        reference_neighbors_per_cluster: int | None = None,
        fit_batch_size: int = 4_096,
    ) -> "ClusterEvidenceIndex":
        """Fit a spherical index with bounded validation and neighbor options.

        A supplied ``calibration_embeddings`` set should be held out. Otherwise
        novelty is explicitly an uncalibrated distance flag, not an OOD claim.
        """
        values = _normalize(np.asarray(embeddings, dtype="float32"))
        if values.ndim != 2 or len(values) < 2:
            raise ValueError("At least two two-dimensional embeddings are required")
        if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) == 0):
            raise ValueError("Embeddings must be finite, non-zero vectors")
        if len(uuids) != len(values):
            raise ValueError("Embedding and UUID counts must match")
        if not 2 <= int(n_clusters) <= len(values):
            raise ValueError("n_clusters must be between 2 and the item count")
        if not 0 <= float(novelty_percentile) <= 100:
            raise ValueError("novelty_percentile must be between 0 and 100")
        if int(silhouette_max_samples) < 2:
            raise ValueError("silhouette_max_samples must be at least 2")
        if reference_neighbors_per_cluster is not None and int(reference_neighbors_per_cluster) < 1:
            raise ValueError("reference_neighbors_per_cluster must be at least 1")
        if int(fit_batch_size) < 2:
            raise ValueError("fit_batch_size must be at least 2")

        try:
            if len(values) > int(fit_batch_size):
                from sklearn.cluster import MiniBatchKMeans

                estimator = MiniBatchKMeans(
                    n_clusters=int(n_clusters),
                    batch_size=int(fit_batch_size),
                    n_init=10,
                    random_state=int(seed),
                )
                method = "spherical_minibatch_kmeans"
            else:
                from sklearn.cluster import KMeans

                estimator = KMeans(
                    n_clusters=int(n_clusters), n_init=10, random_state=int(seed)
                )
                method = "spherical_kmeans"
            assignments = np.asarray(estimator.fit_predict(values), dtype="int64")
            centroids = _normalize(np.asarray(estimator.cluster_centers_, dtype="float32"))
        except ModuleNotFoundError:
            assignments, centroids = _numpy_spherical_kmeans(values, int(n_clusters), seed=int(seed))
            method = "spherical_kmeans_numpy"

        assignments, centroids, cluster_sizes = _compact_clusters(assignments, centroids)
        similarities = np.sum(values * centroids[assignments], axis=1)
        uuid_values = np.asarray(uuids, dtype=str)
        floors = np.empty(len(centroids), dtype="float32")
        medoids: list[str] = []
        for cluster in range(len(centroids)):
            positions = np.flatnonzero(assignments == cluster)
            floors[cluster] = float(np.percentile(similarities[positions], novelty_percentile))
            medoids.append(str(uuid_values[positions[np.argmax(similarities[positions])]]))

        calibration_values = values
        calibration_source, calibration_status = "fit_embeddings", "uncalibrated"
        if calibration_embeddings is not None:
            calibration_values = _normalize(np.asarray(calibration_embeddings, dtype="float32"))
            if calibration_values.ndim != 2 or calibration_values.shape[1] != values.shape[1]:
                raise ValueError("calibration_embeddings must match the embedding dimension")
            if not len(calibration_values) or not np.isfinite(calibration_values).all() or np.any(np.linalg.norm(calibration_values, axis=1) == 0):
                raise ValueError("calibration_embeddings must be finite, non-empty, non-zero vectors")
            calibration_source, calibration_status = "held_out_embeddings", "held_out"
        calibration_assignments = np.argmax(calibration_values @ centroids.T, axis=1)
        calibration_similarities = np.sum(calibration_values * centroids[calibration_assignments], axis=1)
        for cluster in range(len(centroids)):
            selected = calibration_similarities[calibration_assignments == cluster]
            if len(selected):
                floors[cluster] = float(np.percentile(selected, novelty_percentile))
        novelty_threshold = float(np.percentile(calibration_similarities, novelty_percentile))

        score = None
        sample_count = min(len(values), int(silhouette_max_samples))
        if len(np.unique(assignments)) > 1 and sample_count > len(centroids):
            try:
                from sklearn.metrics import silhouette_score

                if sample_count < len(values):
                    sample = np.random.default_rng(seed).choice(len(values), sample_count, replace=False)
                    score = float(silhouette_score(values[sample], assignments[sample], metric="cosine"))
                else:
                    score = float(silhouette_score(values, assignments, metric="cosine"))
            except (ModuleNotFoundError, ValueError):
                score = None

        retained = np.arange(len(values), dtype="int64")
        retention: dict[str, Any] = {"mode": "all", "retained_count": int(len(values))}
        if reference_neighbors_per_cluster is not None:
            chosen = []
            for cluster in range(len(centroids)):
                positions = np.flatnonzero(assignments == cluster)
                chosen.append(positions[np.argsort(-similarities[positions])][:int(reference_neighbors_per_cluster)])
            retained = np.sort(np.concatenate(chosen))
            retention = {"mode": "per_cluster_top_similarity", "neighbors_per_cluster": int(reference_neighbors_per_cluster), "retained_count": int(len(retained))}

        return cls(
            embeddings=values[retained], uuids=uuid_values[retained], assignments=assignments[retained],
            centroids=centroids, cluster_sizes=cluster_sizes, cluster_similarity_floors=floors,
            medoid_uuids=np.asarray(medoids, dtype=str), method=method,
            novelty_similarity_threshold=novelty_threshold, silhouette=score,
            novelty_calibration={
                "status": calibration_status, "source": calibration_source,
                "percentile": float(novelty_percentile), "sample_count": int(len(calibration_values)),
                "decision_semantics": "held_out_similarity_threshold" if calibration_status == "held_out" else "uncalibrated_distance_flag",
            },
            source_reference_count=int(len(values)), reference_retention=retention,
            silhouette_sample_count=sample_count,
        )

    @property
    def cluster_count(self) -> int:
        return int(len(self.centroids))

    @property
    def embedding_dim(self) -> int:
        return int(self.centroids.shape[1])

    def _calibration(self) -> dict[str, Any]:
        return self.novelty_calibration or {
            "status": "uncalibrated", "source": "fit_embeddings", "percentile": None,
            "sample_count": int(self.source_reference_count or len(self.uuids)),
            "decision_semantics": "uncalibrated_distance_flag",
        }

    def _retention(self) -> dict[str, Any]:
        return self.reference_retention or {"mode": "all", "retained_count": int(len(self.uuids))}

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": CLUSTER_EVIDENCE_SCHEMA_VERSION, "method": self.method, "metric": self.metric,
            "reference_count": int(len(self.uuids)), "source_reference_count": int(self.source_reference_count or len(self.uuids)),
            "reference_retention": self._retention(), "embedding_dim": self.embedding_dim,
            "cluster_count": self.cluster_count, "cluster_sizes": [int(value) for value in self.cluster_sizes],
            "silhouette": self.silhouette,
            "silhouette_sample_count": int(self.silhouette_sample_count or min(int(self.source_reference_count or len(self.uuids)), 10_000)),
            "novelty_similarity_threshold": float(self.novelty_similarity_threshold),
            "novelty_calibration": self._calibration(),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name, values, dtype in (
            ("embeddings", self.embeddings, "float32"), ("uuids", self.uuids, str),
            ("assignments", self.assignments, "int64"), ("centroids", self.centroids, "float32"),
            ("cluster_sizes", self.cluster_sizes, "int64"), ("cluster_similarity_floors", self.cluster_similarity_floors, "float32"),
            ("medoid_uuids", self.medoid_uuids, str),
        ):
            np.save(path / f"{name}.npy", np.asarray(values, dtype=dtype), allow_pickle=False)
        (path / "metadata.json").write_text(json.dumps({**self.summary(), "storage": "npy_directory"}, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ClusterEvidenceIndex":
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text())
        if int(metadata.get("schema_version", 0)) not in _SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError("Unsupported cluster evidence schema version")
        index = cls(
            embeddings=np.load(path / "embeddings.npy", mmap_mode="r"), uuids=np.load(path / "uuids.npy", mmap_mode="r"),
            assignments=np.load(path / "assignments.npy", mmap_mode="r"), centroids=np.load(path / "centroids.npy", mmap_mode="r"),
            cluster_sizes=np.load(path / "cluster_sizes.npy", mmap_mode="r"), cluster_similarity_floors=np.load(path / "cluster_similarity_floors.npy", mmap_mode="r"),
            medoid_uuids=np.load(path / "medoid_uuids.npy", mmap_mode="r"), method=str(metadata.get("method", "spherical_kmeans")),
            metric=str(metadata.get("metric", "cosine_similarity")), novelty_similarity_threshold=float(metadata.get("novelty_similarity_threshold", 0.0)),
            silhouette=float(metadata["silhouette"]) if metadata.get("silhouette") is not None else None,
            novelty_calibration=dict(metadata.get("novelty_calibration") or {}),
            source_reference_count=int(metadata.get("source_reference_count", metadata.get("reference_count", 0))) or None,
            reference_retention=dict(metadata.get("reference_retention") or {}),
            silhouette_sample_count=(int(metadata["silhouette_sample_count"]) if metadata.get("silhouette_sample_count") is not None else None),
        )
        index._validate_loaded()
        return index

    def _validate_loaded(self) -> None:
        if self.centroids.ndim != 2 or self.cluster_count < 2:
            raise ValueError("Cluster evidence requires at least two valid centroids")
        if len(self.cluster_sizes) != self.cluster_count or len(self.cluster_similarity_floors) != self.cluster_count or len(self.medoid_uuids) != self.cluster_count:
            raise ValueError("Cluster evidence arrays must agree on cluster count")
        if self.embeddings.ndim != 2 or self.embeddings.shape[1] != self.embedding_dim:
            raise ValueError("Cluster reference embeddings do not match centroid dimension")
        if not (len(self.uuids) == len(self.assignments) == len(self.embeddings)):
            raise ValueError("Cluster reference arrays must have matching lengths")
        if np.any(self.cluster_sizes <= 0) or np.any(self.assignments < 0) or np.any(self.assignments >= self.cluster_count):
            raise ValueError("Cluster evidence contains empty or invalid cluster assignments")

    def packet_many(self, embeddings: np.ndarray, *, query_uuids: Sequence[str | None] | None = None, top_k: int = 5, include_neighbors: bool = True) -> list[dict[str, Any]]:
        """Batch cluster assignment; disable neighbors for O(batch * clusters) serving."""
        queries = _normalize(np.asarray(embeddings, dtype="float32"))
        if queries.ndim != 2 or queries.shape[1] != self.embedding_dim:
            raise ValueError(f"Embeddings must have shape (n, {self.embedding_dim})")
        if np.any(np.linalg.norm(queries, axis=1) == 0):
            raise ValueError("Embeddings must be non-zero vectors")
        if query_uuids is not None and len(query_uuids) != len(queries):
            raise ValueError("query_uuids must match the embedding count")
        centroid_similarities = np.asarray(queries @ self.centroids.T, dtype="float32")
        return [self._packet_from_normalized_query(query, centroid_similarities[position], query_uuid=None if query_uuids is None else query_uuids[position], top_k=top_k, include_neighbors=include_neighbors) for position, query in enumerate(queries)]

    def packet(self, embedding: np.ndarray, *, query_uuid: str | None = None, top_k: int = 5, include_neighbors: bool = True) -> dict[str, Any]:
        return self.packet_many(np.asarray(embedding, dtype="float32")[None, :], query_uuids=[query_uuid], top_k=top_k, include_neighbors=include_neighbors)[0]

    def _packet_from_normalized_query(self, query: np.ndarray, similarities: np.ndarray, *, query_uuid: str | None, top_k: int, include_neighbors: bool) -> dict[str, Any]:
        top_k = max(1, int(top_k))
        order = np.argsort(-similarities)
        best = int(order[0])
        floor, best_similarity = float(self.cluster_similarity_floors[best]), float(similarities[best])
        threshold = max(float(self.novelty_similarity_threshold), floor)
        novel = best_similarity < threshold
        neighbors: list[dict[str, Any]] = []
        if include_neighbors and len(self.embeddings):
            neighbor_similarities = np.asarray(self.embeddings @ query, dtype="float32")
            eligible = np.ones(len(neighbor_similarities), dtype=bool)
            if query_uuid is not None:
                eligible &= self.uuids != str(query_uuid)
            positions = np.flatnonzero(eligible)
            positions = positions[np.argsort(-neighbor_similarities[positions])][:top_k]
            neighbors = [{"uuid": str(self.uuids[position]), "cluster_index": int(self.assignments[position]), "cluster_id": f"cluster-{int(self.assignments[position]):04d}", "similarity": float(neighbor_similarities[position])} for position in positions]
        calibration = self._calibration()
        return {
            "schema_version": CLUSTER_EVIDENCE_SCHEMA_VERSION, "type": "roi_clustering", "metric": self.metric,
            "embedding": {"dimension": self.embedding_dim, "normalized": True},
            "decision": {"cluster_index": None if novel else best, "cluster_id": None if novel else f"cluster-{best:04d}", "similarity": best_similarity, "similarity_floor": floor, "novelty_similarity_threshold": float(self.novelty_similarity_threshold), "novel": bool(novel), "abstained": bool(novel), "novelty_status": calibration["decision_semantics"], "novelty_calibration": calibration},
            "clusters": [{"cluster_index": int(cluster), "cluster_id": f"cluster-{int(cluster):04d}", "similarity": float(similarities[cluster]), "size": int(self.cluster_sizes[cluster]), "similarity_floor": float(self.cluster_similarity_floors[cluster]), "representative_uuid": str(self.medoid_uuids[cluster])} for cluster in order[:top_k]],
            "nearest_neighbors": neighbors, "nearest_neighbors_retained": bool(include_neighbors), "structure": self.summary(),
        }

    def exemplar(self, item_id: str) -> dict[str, Any] | None:
        matches = np.flatnonzero(self.uuids == str(item_id))
        if not len(matches):
            return None
        position = int(matches[0])
        cluster = int(self.assignments[position])
        return {"item_id": str(item_id), "cluster_index": cluster, "cluster_id": f"cluster-{cluster:04d}", "representative": str(self.medoid_uuids[cluster]), "image_available": False}
