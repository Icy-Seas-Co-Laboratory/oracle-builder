"""Configurable sklearn baselines trained over exported embeddings."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PosthocConfig:
    """Serializable post-hoc estimator configuration.

    Transform names execute in order: ``identity``, ``standard``, ``robust``,
    ``l2``, and ``pca``.  Parameters are passed through only to their named
    sklearn object (for example ``pca={"n_components": 32}``).
    """
    transforms: tuple[str, ...] = ("identity",)
    classifier: str = "logistic_regression"
    transform_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    classifier_options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PosthocConfig":
        transforms = value.get("transforms", ("identity",))
        if isinstance(transforms, str):
            transforms = (transforms,)
        return cls(
            transforms=tuple(transforms),
            classifier=value.get("classifier", "logistic_regression"),
            transform_options=value.get("transform_options", {}),
            classifier_options=value.get("classifier_options", {}),
        )


def _sklearn():
    try:
        from sklearn.base import BaseEstimator, TransformerMixin
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
        from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer, Normalizer, RobustScaler, StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:
        raise RuntimeError("post-hoc estimators require scikit-learn; install oracle-builder dependencies") from exc
    return locals()


def build_posthoc_pipeline(config: PosthocConfig | Mapping[str, Any]):
    """Build an unfitted sklearn pipeline from a portable configuration."""
    if not isinstance(config, PosthocConfig):
        config = PosthocConfig.from_mapping(config)
    sk = _sklearn()
    transforms = {
        "identity": sk["FunctionTransformer"](validate=False),
        "standard": sk["StandardScaler"],
        "robust": sk["RobustScaler"],
        "l2": sk["Normalizer"],
        "pca": sk["PCA"],
    }
    steps = []
    for index, name in enumerate(config.transforms):
        if name not in transforms:
            raise ValueError(f"unknown post-hoc transform: {name}")
        builder = transforms[name]
        options = dict(config.transform_options.get(name, {}))
        step = builder if name == "identity" else builder(**options)
        steps.append((f"{name}_{index}", step))
    classifiers = {
        "logistic_regression": sk["LogisticRegression"], "linear_svm": sk["SVC"],
        "kernel_svm": sk["SVC"], "knn": sk["KNeighborsClassifier"],
        "nearest_centroid": sk["NearestCentroid"], "random_forest": sk["RandomForestClassifier"],
        "extra_trees": sk["ExtraTreesClassifier"], "gradient_boosting": sk["GradientBoostingClassifier"],
        "lda": sk["LinearDiscriminantAnalysis"], "qda": sk["QuadraticDiscriminantAnalysis"],
    }
    if config.classifier not in classifiers:
        raise ValueError(f"unknown post-hoc classifier: {config.classifier}")
    options = dict(config.classifier_options)
    if config.classifier == "logistic_regression":
        options.setdefault("max_iter", 1000)
    if config.classifier == "linear_svm":
        options.setdefault("kernel", "linear")
    if config.classifier == "kernel_svm":
        options.setdefault("kernel", "rbf")
    steps.append(("classifier", classifiers[config.classifier](**options)))
    return sk["Pipeline"](steps)
