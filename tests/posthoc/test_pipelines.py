import numpy as np
import pytest

pytest.importorskip("sklearn")

from oracle_builder.posthoc.pipelines import PosthocConfig, build_posthoc_pipeline
from oracle_builder.posthoc.probes import representation_diagnostics, run_probe_suite


def test_pipeline_builds_configured_transform_and_classifier():
    pipeline = build_posthoc_pipeline({
        "transforms": ["standard", "pca", "l2"],
        "transform_options": {"pca": {"n_components": 2}},
        "classifier": "linear_svm",
    })
    assert list(pipeline.named_steps) == ["standard_0", "pca_1", "l2_2", "classifier"]
    assert pipeline.named_steps["classifier"].kernel == "linear"


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
