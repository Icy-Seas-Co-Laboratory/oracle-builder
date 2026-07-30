from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oracle_builder.classification.features import predict_with_features


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
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            labels=self.labels,
            uuids=self.uuids,
            prototype_labels=self.prototype_labels,
            prototypes=self.prototypes,
        )
        metadata = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "metric": "cosine_similarity",
            "normalized": True,
            "reference_count": int(len(self.labels)),
            "embedding_dim": int(self.embeddings.shape[1]),
            "classes": [int(label) for label in self.prototype_labels],
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "IdentityEvidenceIndex":
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
