"""Small, reproducible representation diagnostics and supervised probes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .pipelines import PosthocConfig, build_posthoc_pipeline


@dataclass(frozen=True)
class ProbeResult:
    name: str
    accuracy: float
    balanced_accuracy: float
    sample_count: int
    classes: tuple[Any, ...]


def representation_diagnostics(embeddings: np.ndarray, *, epsilon: float = 1e-12) -> dict[str, float | int]:
    """Return label-free health checks without materialising an NxN matrix."""
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("embeddings must be a non-empty rank-2 array")
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular * singular
    total = float(energy.sum())
    proportions = energy / total if total > epsilon else np.zeros_like(energy)
    entropy = -float(np.sum(proportions[proportions > 0] * np.log(proportions[proportions > 0])))
    norms = np.linalg.norm(values, axis=1)
    return {
        "sample_count": int(values.shape[0]), "dimension": int(values.shape[1]),
        "mean_feature_std": float(centered.std(axis=0).mean()),
        "effective_rank": float(np.exp(entropy)),
        "rank": int(np.count_nonzero(singular > epsilon)),
        "mean_l2_norm": float(norms.mean()), "zero_norm_fraction": float(np.mean(norms <= epsilon)),
    }


def run_probe_suite(
    train_embeddings: np.ndarray, train_labels: Sequence[Any], test_embeddings: np.ndarray,
    test_labels: Sequence[Any], *, configs: Mapping[str, PosthocConfig | Mapping[str, Any]] | None = None,
) -> dict[str, ProbeResult]:
    """Fit inexpensive linear/kNN/centroid probes and score held-out vectors."""
    try:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
    except ImportError as exc:
        raise RuntimeError("representation probes require scikit-learn") from exc
    if configs is None:
        configs = {
            "linear": PosthocConfig(("standard",), "logistic_regression"),
            "knn": PosthocConfig(("standard", "l2"), "knn", classifier_options={"n_neighbors": 10}),
            "centroid": PosthocConfig(("standard",), "nearest_centroid"),
        }
    x_train, x_test = np.asarray(train_embeddings), np.asarray(test_embeddings)
    y_train, y_test = np.asarray(train_labels), np.asarray(test_labels)
    if x_train.ndim != 2 or x_test.ndim != 2 or len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("embedding/label dimensions are inconsistent")
    if x_train.shape[1] != x_test.shape[1] or len(np.unique(y_train)) < 2:
        raise ValueError("probes require matching dimensions and at least two training classes")
    result = {}
    for name, config in configs.items():
        estimator = build_posthoc_pipeline(config)
        estimator.fit(x_train, y_train)
        predicted = estimator.predict(x_test)
        result[name] = ProbeResult(name, float(accuracy_score(y_test, predicted)), float(balanced_accuracy_score(y_test, predicted)), len(y_test), tuple(np.unique(y_train).tolist()))
    return result
