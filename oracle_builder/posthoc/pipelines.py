"""Portable sklearn pipelines and validation for post-hoc classifiers."""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import math
from typing import Any, Mapping

_ESTIMATORS = {"logistic_regression", "linear_svm", "gradient_boosting", "shallow_mlp"}
_SCALING = {"standard", "robust", "none"}
_OPTION_KEYS = {
    "logistic_regression": {"C", "max_iter", "class_weight", "tol"},
    "linear_svm": {"C", "max_iter", "class_weight", "tol"},
    "gradient_boosting": {"n_estimators", "learning_rate", "max_depth", "min_samples_leaf", "subsample"},
    "shallow_mlp": {"hidden_layer_sizes", "activation", "alpha", "learning_rate_init", "batch_size", "max_iter", "early_stopping", "validation_fraction"},
}


def _positive_number(value: Any, name: str, *, allow_zero: bool = False) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"posthoc.classifier_options.{name} must be {'non-negative' if allow_zero else 'positive'}")


def normalize_posthoc_settings(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the canonical, JSON-serializable post-hoc settings mapping."""
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("posthoc must be an object")
    value = dict(value or {})
    allowed = {"enabled", "representation", "estimator", "scaling", "pca_enabled", "pca_components", "classifier_options", "tune", "seed", "calibration_folds", "training_dataset_id", "training_dataset_fingerprint_sha256"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown posthoc settings: {', '.join(sorted(unknown))}")
    for key in ("enabled", "pca_enabled", "tune"):
        if key in value and not isinstance(value[key], bool):
            raise ValueError(f"posthoc.{key} must be boolean")
    raw_options = value.get("classifier_options", {})
    if not isinstance(raw_options, Mapping):
        raise ValueError("posthoc.classifier_options must be an object")
    settings = {
        "enabled": value.get("enabled", False), "representation": value.get("representation", "fused_embedding"),
        "estimator": value.get("estimator", "logistic_regression"), "scaling": value.get("scaling", "standard"),
        "pca_enabled": value.get("pca_enabled", False), "pca_components": value.get("pca_components", 32),
        "classifier_options": dict(raw_options), "tune": value.get("tune", True),
        "seed": value.get("seed", 42), "calibration_folds": value.get("calibration_folds", 3),
    }
    for key in ("training_dataset_id", "training_dataset_fingerprint_sha256"):
        if key in value:
            settings[key] = value[key]
    if settings["representation"] != "fused_embedding":
        raise ValueError("posthoc.representation must be fused_embedding")
    if not isinstance(settings["estimator"], str) or settings["estimator"] not in _ESTIMATORS:
        raise ValueError(f"posthoc.estimator must be one of: {', '.join(sorted(_ESTIMATORS))}")
    if not isinstance(settings["scaling"], str) or settings["scaling"] not in _SCALING:
        raise ValueError("posthoc.scaling must be one of: standard, robust, none")
    if not isinstance(settings["pca_components"], int) or isinstance(settings["pca_components"], bool) or settings["pca_components"] < 1:
        raise ValueError("posthoc.pca_components must be a positive integer")
    if not isinstance(settings["seed"], int) or isinstance(settings["seed"], bool) or not 0 <= settings["seed"] <= 2**32 - 1:
        raise ValueError("posthoc.seed must be an integer from 0 through 4294967295")
    if not isinstance(settings["calibration_folds"], int) or isinstance(settings["calibration_folds"], bool) or not 2 <= settings["calibration_folds"] <= 10:
        raise ValueError("posthoc.calibration_folds must be an integer from 2 through 10")
    options = settings["classifier_options"]
    unknown_options = set(options) - _OPTION_KEYS[settings["estimator"]]
    if unknown_options:
        raise ValueError(f"Unsupported options for {settings['estimator']}: {', '.join(sorted(unknown_options))}")
    for name in {"C", "tol", "learning_rate", "learning_rate_init", "subsample", "validation_fraction"} & set(options):
        _positive_number(options[name], name)
    if "alpha" in options:
        _positive_number(options["alpha"], "alpha", allow_zero=True)
    for name in {"max_iter", "n_estimators", "max_depth", "min_samples_leaf", "batch_size"} & set(options):
        if not isinstance(options[name], int) or isinstance(options[name], bool) or options[name] < 1:
            raise ValueError(f"posthoc.classifier_options.{name} must be a positive integer")
    if "class_weight" in options and options["class_weight"] not in (None, "balanced"):
        raise ValueError("posthoc.classifier_options.class_weight must be null or 'balanced'")
    if "early_stopping" in options and not isinstance(options["early_stopping"], bool):
        raise ValueError("posthoc.classifier_options.early_stopping must be boolean")
    if "activation" in options and (not isinstance(options["activation"], str) or options["activation"] not in {"identity", "logistic", "relu", "tanh"}):
        raise ValueError("posthoc.classifier_options.activation is invalid")
    if "hidden_layer_sizes" in options:
        layers = options["hidden_layer_sizes"]
        if not isinstance(layers, (tuple, list)) or not 1 <= len(layers) <= 2 or any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in layers):
            raise ValueError("posthoc.classifier_options.hidden_layer_sizes must be one or two positive integers")
        options["hidden_layer_sizes"] = list(layers)
    for name in ("subsample", "validation_fraction"):
        if name in options and options[name] > 1:
            raise ValueError(f"posthoc.classifier_options.{name} must be at most 1")
    if options.get("validation_fraction") == 1:
        raise ValueError("posthoc.classifier_options.validation_fraction must be less than 1")
    return settings


@dataclass(frozen=True)
class PosthocConfig:
    """Legacy portable pipeline configuration used by probes and integrations."""
    transforms: tuple[str, ...] = ("identity",)
    classifier: str = "logistic_regression"
    transform_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    classifier_options: Mapping[str, Any] = field(default_factory=dict)
    calibration_folds: int = 3
    seed: int = 42

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PosthocConfig":
        transforms = value.get("transforms", ("identity",))
        if isinstance(transforms, str): transforms = (transforms,)
        return cls(tuple(transforms), value.get("classifier", "logistic_regression"), value.get("transform_options", {}), value.get("classifier_options", {}), value.get("calibration_folds", 3), value.get("seed", 42))


def _sklearn():
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.decomposition import PCA
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer, Normalizer, RobustScaler, StandardScaler
        from sklearn.svm import LinearSVC, SVC
    except ImportError as exc:
        raise RuntimeError("post-hoc estimators require scikit-learn; install oracle-builder dependencies") from exc
    return locals()


def _calibrator(sk: Mapping[str, Any], estimator: Any, folds: int):
    kwargs = {"cv": folds, "method": "sigmoid"}
    kwargs["estimator" if "estimator" in inspect.signature(sk["CalibratedClassifierCV"]).parameters else "base_estimator"] = estimator
    return sk["CalibratedClassifierCV"](**kwargs)


def _steps_for_transforms(sk: Mapping[str, Any], transforms: tuple[str, ...], options: Mapping[str, Mapping[str, Any]]):
    builders = {"identity": sk["FunctionTransformer"], "standard": sk["StandardScaler"], "robust": sk["RobustScaler"], "l2": sk["Normalizer"], "pca": sk["PCA"]}
    steps = []
    for index, name in enumerate(transforms):
        if name not in builders: raise ValueError(f"unknown post-hoc transform: {name}")
        kwargs = dict(options.get(name, {}))
        steps.append((f"{name}_{index}", builders[name](validate=False) if name == "identity" else builders[name](**kwargs)))
    return steps


def build_posthoc_pipeline(config: PosthocConfig | Mapping[str, Any]):
    """Build an unfitted sklearn estimator with class-aligned probabilities."""
    if not isinstance(config, PosthocConfig): config = PosthocConfig.from_mapping(config)
    sk = _sklearn()
    steps = _steps_for_transforms(sk, config.transforms, config.transform_options)
    options = dict(config.classifier_options)
    if config.classifier == "logistic_regression":
        options.setdefault("max_iter", 1000); options.setdefault("random_state", config.seed)
        steps.append(("classifier", sk["LogisticRegression"](**options))); return sk["Pipeline"](steps)
    if config.classifier == "gradient_boosting":
        options.setdefault("random_state", config.seed)
        steps.append(("classifier", sk["GradientBoostingClassifier"](**options))); return sk["Pipeline"](steps)
    if config.classifier == "shallow_mlp":
        options.setdefault("max_iter", 300); options.setdefault("random_state", config.seed)
        steps.append(("classifier", sk["MLPClassifier"](**options))); return sk["Pipeline"](steps)
    if config.classifier == "linear_svm":
        options.setdefault("random_state", config.seed)
        base = sk["Pipeline"]([*steps, ("classifier", sk["LinearSVC"](**options))])
        return _calibrator(sk, base, config.calibration_folds)
    # Legacy probe configurations remain available as uncalibrated diagnostic
    # estimators.  The user-facing fitting path only accepts _ESTIMATORS.
    legacy = {
        "kernel_svm": sk["SVC"], "knn": sk["KNeighborsClassifier"],
        "nearest_centroid": sk["NearestCentroid"], "random_forest": sk["RandomForestClassifier"],
        "extra_trees": sk["ExtraTreesClassifier"], "lda": sk["LinearDiscriminantAnalysis"],
        "qda": sk["QuadraticDiscriminantAnalysis"],
    }
    if config.classifier in legacy:
        if config.classifier == "kernel_svm": options.setdefault("kernel", "rbf")
        if config.classifier in {"random_forest", "extra_trees"}: options.setdefault("random_state", config.seed)
        steps.append(("classifier", legacy[config.classifier](**options)))
        return sk["Pipeline"](steps)
    raise ValueError(f"unknown post-hoc classifier: {config.classifier}")


def pipeline_config_from_settings(settings: Mapping[str, Any]) -> PosthocConfig:
    """Translate normalized UI/API settings to the reusable sklearn builder."""
    normalized = normalize_posthoc_settings(settings)
    transforms = [] if normalized["scaling"] == "none" else [normalized["scaling"]]
    options: dict[str, Mapping[str, Any]] = {}
    if normalized["pca_enabled"]:
        transforms.append("pca"); options["pca"] = {"n_components": normalized["pca_components"], "random_state": normalized["seed"]}
    if not transforms: transforms.append("identity")
    return PosthocConfig(tuple(transforms), normalized["estimator"], options, normalized["classifier_options"], normalized["calibration_folds"], normalized["seed"])
