from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oracle_builder.classification.features import predict_with_features
from oracle_builder.classification.features import build_feature_model
from oracle_builder.progress import BatchProgress


EVIDENCE_SCHEMA_VERSION = 1


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float32")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


@dataclass
class IdentityEvidenceIndex:
    embeddings: np.ndarray
    labels: np.ndarray
    uuids: np.ndarray
    prototype_labels: np.ndarray
    prototypes: np.ndarray

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        labels: np.ndarray,
        uuids: list[str],
    ) -> "IdentityEvidenceIndex":
        normalized = _normalize(embeddings)
        labels = np.asarray(labels, dtype="int64")
        prototype_labels = np.unique(labels)
        prototypes = np.stack(
            [_normalize(normalized[labels == label].mean(axis=0, keepdims=True))[0] for label in prototype_labels]
        )
        return cls(
            embeddings=normalized,
            labels=labels,
            uuids=np.asarray(uuids, dtype=str),
            prototype_labels=prototype_labels,
            prototypes=prototypes,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix != ".npz":
            path.mkdir(parents=True, exist_ok=True)
            np.save(path / "embeddings.npy", self.embeddings, allow_pickle=False)
            np.save(path / "labels.npy", self.labels, allow_pickle=False)
            np.save(path / "uuids.npy", self.uuids, allow_pickle=False)
            np.save(path / "prototype_labels.npy", self.prototype_labels, allow_pickle=False)
            np.save(path / "prototypes.npy", self.prototypes, allow_pickle=False)
            _write_metadata(path / "metadata.json", self)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            labels=self.labels,
            uuids=self.uuids,
            prototype_labels=self.prototype_labels,
            prototypes=self.prototypes,
        )
        _write_metadata(path.with_suffix(".json"), self)

    @classmethod
    def load(cls, path: str | Path) -> "IdentityEvidenceIndex":
        path = Path(path)
        if path.is_dir():
            return cls(
                embeddings=np.load(path / "embeddings.npy", mmap_mode="r"),
                labels=np.load(path / "labels.npy", mmap_mode="r"),
                uuids=np.load(path / "uuids.npy", mmap_mode="r"),
                prototype_labels=np.load(path / "prototype_labels.npy", mmap_mode="r"),
                prototypes=np.load(path / "prototypes.npy", mmap_mode="r"),
            )
        with np.load(path, allow_pickle=False) as values:
            return cls(
                embeddings=values["embeddings"],
                labels=values["labels"],
                uuids=values["uuids"],
                prototype_labels=values["prototype_labels"],
                prototypes=values["prototypes"],
            )

    def packet(
        self,
        embedding: np.ndarray,
        probabilities: np.ndarray,
        *,
        query_uuid: str | None = None,
        k: int = 5,
    ) -> dict[str, Any]:
        query = _normalize(np.asarray(embedding, dtype="float32")[None, :])[0]
        prototype_similarities = self.prototypes @ query
        prototype_order = np.argsort(-prototype_similarities)
        prototype_best = int(prototype_order[0])
        prototype_second = (
            float(prototype_similarities[prototype_order[1]]) if len(prototype_order) > 1 else 0.0
        )

        similarities = self.embeddings @ query
        eligible = np.ones(len(similarities), dtype=bool)
        if query_uuid is not None:
            eligible &= self.uuids != str(query_uuid)
        eligible_indices = np.flatnonzero(eligible)
        k_used = min(max(int(k), 1), len(eligible_indices))
        if k_used:
            candidate_similarities = similarities[eligible_indices]
            if k_used < len(eligible_indices):
                selected = np.argpartition(-candidate_similarities, k_used - 1)[:k_used]
                neighbor_indices = eligible_indices[selected]
            else:
                neighbor_indices = eligible_indices
            neighbor_indices = neighbor_indices[np.argsort(-similarities[neighbor_indices])]
        else:
            neighbor_indices = np.asarray([], dtype=int)
        neighbor_labels = self.labels[neighbor_indices]
        neighbor_similarities = similarities[neighbor_indices]
        label_counts: dict[int, int] = {}
        weighted_support_raw: dict[int, float] = {}
        for label, similarity in zip(neighbor_labels, neighbor_similarities, strict=True):
            label = int(label)
            label_counts[label] = label_counts.get(label, 0) + 1
            weight = (float(similarity) + 1.0) / 2.0
            weighted_support_raw[label] = weighted_support_raw.get(label, 0.0) + max(weight, 0.0)
        support_total = sum(weighted_support_raw.values())
        weighted_support = {
            label: value / support_total if support_total else 0.0
            for label, value in weighted_support_raw.items()
        }
        support_order = sorted(weighted_support, key=lambda label: (-weighted_support[label], label))
        strongest_label = support_order[0] if support_order else None
        strongest_support = weighted_support.get(strongest_label, 0.0)
        second_support = weighted_support.get(support_order[1], 0.0) if len(support_order) > 1 else 0.0

        predicted_class = int(np.argmax(probabilities))
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "softmax": {
                "probabilities": [float(value) for value in probabilities],
                "predicted_class": predicted_class,
                "confidence": float(np.max(probabilities)),
            },
            "embedding": {"dimension": int(query.shape[0]), "normalized": True},
            "prototype": {
                "similarities": {
                    str(int(label)): float(similarity)
                    for label, similarity in zip(
                        self.prototype_labels, prototype_similarities, strict=True
                    )
                },
                "predicted_class": int(self.prototype_labels[prototype_best]),
                "nearest_similarity": float(prototype_similarities[prototype_best]),
                "similarity_margin": float(
                    prototype_similarities[prototype_best] - prototype_second
                ),
            },
            "knn": {
                "k_requested": int(k),
                "k_used": int(k_used),
                "nearest_neighbor_similarity": (
                    float(neighbor_similarities[0]) if k_used else None
                ),
                "nearest_neighbor_uuid": str(self.uuids[neighbor_indices[0]]) if k_used else None,
                "strongest_label": strongest_label,
                "label_agreement": (
                    float(label_counts.get(strongest_label, 0) / k_used) if k_used else 0.0
                ),
                "label_counts": {str(label): count for label, count in label_counts.items()},
                "weighted_label_support": {
                    str(label): float(value) for label, value in weighted_support.items()
                },
                "label_support_margin": float(strongest_support - second_support),
                "neighbors": [
                    {
                        "uuid": str(self.uuids[index]),
                        "label": int(self.labels[index]),
                        "similarity": float(similarities[index]),
                    }
                    for index in neighbor_indices
                ],
            },
        }


def build_evidence_index(
    model,
    x: np.ndarray,
    y: np.ndarray,
    records: list[dict[str, Any]],
    path: str | Path,
) -> IdentityEvidenceIndex:
    _, features = predict_with_features(model, x)
    if features is None:
        raise ValueError("Classification model does not expose identity embeddings")
    index = IdentityEvidenceIndex.build(
        features,
        np.asarray(y, dtype="int64"),
        [record["uuid"] for record in records],
    )
    index.save(path)
    return index


def build_evidence_index_streaming(
    model,
    dataset,
    sample_index,
    path: str | Path,
    *,
    progress: bool = True,
) -> IdentityEvidenceIndex:
    """Build an evidence index without retaining decoded source images."""
    feature_model = build_feature_model(model)
    embedding_dim = int(model.get_layer("features").output.shape[-1])
    path = Path(path)
    if path.suffix == ".npz":
        embeddings = np.empty((len(sample_index), embedding_dim), dtype="float32")
    else:
        path.mkdir(parents=True, exist_ok=True)
        embeddings = np.lib.format.open_memmap(
            path / "embeddings.npy",
            mode="w+",
            dtype="float32",
            shape=(len(sample_index), embedding_dim),
        )
    labels = np.asarray([ref.target for ref in sample_index.refs], dtype="int64")
    prototype_labels = np.unique(labels)
    prototype_sums = {
        int(label): np.zeros(embedding_dim, dtype="float64")
        for label in prototype_labels
    }
    prototype_counts = {int(label): 0 for label in prototype_labels}
    display = BatchProgress(
        "Building classification evidence",
        len(sample_index),
        enabled=progress,
    )
    for images, positions in dataset:
        outputs = feature_model(images, training=False)
        batch_features = np.asarray(outputs["features"], dtype="float32")
        position_values = np.asarray(positions, dtype="int64")
        normalized_batch = _normalize(batch_features)
        embeddings[position_values] = normalized_batch
        batch_labels = labels[position_values]
        for label in np.unique(batch_labels):
            selected = normalized_batch[batch_labels == label]
            prototype_sums[int(label)] += selected.sum(axis=0)
            prototype_counts[int(label)] += len(selected)
        display.update(len(position_values))
    display.close()
    prototypes = np.stack(
        [
            _normalize(
                (
                    prototype_sums[int(label)] / max(prototype_counts[int(label)], 1)
                )[None, :]
            )[0]
            for label in prototype_labels
        ]
    )
    index = IdentityEvidenceIndex(
        embeddings=embeddings,
        labels=labels,
        uuids=np.asarray([ref.uuid for ref in sample_index.refs], dtype=str),
        prototype_labels=prototype_labels,
        prototypes=prototypes,
    )
    if path.suffix == ".npz":
        index.save(path)
    else:
        np.save(path / "labels.npy", labels, allow_pickle=False)
        np.save(path / "uuids.npy", index.uuids, allow_pickle=False)
        np.save(path / "prototype_labels.npy", prototype_labels, allow_pickle=False)
        np.save(path / "prototypes.npy", prototypes, allow_pickle=False)
        embeddings.flush()
        _write_metadata(path / "metadata.json", index)
    return index


def _write_metadata(path: Path, index: IdentityEvidenceIndex) -> None:
    metadata = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "metric": "cosine_similarity",
        "normalized": True,
        "storage": "memory_mapped_npy" if path.name == "metadata.json" else "npz",
        "reference_count": int(len(index.labels)),
        "embedding_dim": int(index.embeddings.shape[1]),
        "classes": [int(label) for label in index.prototype_labels],
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n")
