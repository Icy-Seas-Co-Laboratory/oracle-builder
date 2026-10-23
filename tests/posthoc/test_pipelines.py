import numpy as np
import pytest

pytest.importorskip("sklearn")

from oracle_builder.posthoc.pipelines import PosthocConfig, build_posthoc_pipeline, normalize_posthoc_settings, pipeline_config_from_settings
from oracle_builder.posthoc.probes import representation_diagnostics, run_probe_suite


def test_pipeline_builds_configured_transform_and_classifier():
    pipeline = build_posthoc_pipeline({
        "transforms": ["standard", "pca", "l2"],
        "transform_options": {"pca": {"n_components": 2}},
        "classifier": "linear_svm",
    })
    # LinearSVC is calibrated for a softmax-like probability contract.  Its
    # preprocessing lives inside each calibration fold to prevent leakage.
    assert pipeline.method == "sigmoid"
    assert pipeline.cv == 3


def test_settings_normalize_and_translate_optional_pca():
    settings = normalize_posthoc_settings({"estimator": "gradient_boosting", "scaling": "robust", "pca_enabled": True, "pca_components": 2})
    assert settings["tune"] is True
    config = pipeline_config_from_settings(settings)
    assert config.transforms == ("robust", "pca")
    assert config.transform_options["pca"]["n_components"] == 2


def test_settings_reject_unknown_estimator_options():
    with pytest.raises(ValueError, match="Unsupported options"):
        normalize_posthoc_settings({"estimator": "linear_svm", "classifier_options": {"kernel": "rbf"}})


@pytest.mark.parametrize("settings", [
    {"enabled": "false"}, {"classifier_options": []}, {"seed": -1},
    {"estimator": "linear_svm", "classifier_options": {"tol": 0}},
    {"classifier_options": {"C": float("nan")}},
    {"estimator": "gradient_boosting", "classifier_options": {"subsample": 1.1}},
])
def test_settings_are_strictly_validated(settings):
    with pytest.raises(ValueError):
        normalize_posthoc_settings(settings)


def test_mlp_settings_remain_json_portable():
    settings = normalize_posthoc_settings({"estimator": "shallow_mlp", "classifier_options": {"hidden_layer_sizes": [16, 8]}})
    assert settings["classifier_options"]["hidden_layer_sizes"] == [16, 8]


def test_probes_and_diagnostics_are_useful_for_separable_data():
    rng = np.random.default_rng(7)
    negative = rng.normal(-3, 0.2, (30, 4))
    positive = rng.normal(3, 0.2, (30, 4))
    train = np.concatenate([negative[:20], positive[:20]])
    test = np.concatenate([negative[20:], positive[20:]])
    labels_train = [0] * 20 + [1] * 20
    labels_test = [0] * 10 + [1] * 10
    probes = run_probe_suite(train, labels_train, test, labels_test)
    assert all(item.accuracy == 1.0 for item in probes.values())
    diagnostic = representation_diagnostics(train)
    assert diagnostic["effective_rank"] > 1
    assert diagnostic["zero_norm_fraction"] == 0.0


def test_unknown_component_is_actionable():
    with pytest.raises(ValueError, match="unknown post-hoc transform"):
        build_posthoc_pipeline(PosthocConfig(("wrong",)))
