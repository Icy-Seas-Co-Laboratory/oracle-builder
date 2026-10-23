"""Contract tests for the V2 configuration catalog and generated presets."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from oracle_builder.config import (
    DEFAULT_CONFIG,
    STANDARD_DISTRIBUTION_STRATEGIES,
    STANDARD_TRAINING_LOSSES_BY_TASK,
    STANDARD_TRAINING_METRICS_BY_TASK,
    STANDARD_TRAINING_OPTIMIZERS,
    load_toml,
    deep_merge,
    normalize_component_config,
    validate_config,
    resolve_v2_config,
    v2_validation_errors,
)
from oracle_builder.config_schema import SCHEMA_VERSION, configuration_schema, field_exposure, v2_defaults


PROJECT_ROOT = Path(__file__).parents[1]
REFERENCE_V2 = PROJECT_ROOT / "configs" / "reference_v2_exhaustive.toml"


def _leaf_paths(value: Any, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        return {
            path
            for key, item in value.items()
            for path in _leaf_paths(item, f"{prefix}.{key}" if prefix else key)
        }
    return {prefix}


def test_catalog_covers_every_runtime_default_and_exposes_v2_metadata():
    schema = configuration_schema()
    fields = {field["path"]: field for field in schema["fields"]}

    assert schema["schema_version"] == SCHEMA_VERSION
    assert len(schema["fingerprint"]) == 64
    assert schema["defaults"]["architecture"]["version"] == 2
    assert _leaf_paths(v2_defaults()) <= fields.keys()
    assert not any(path == "run.model" or path.startswith("model.") for path in fields)
    assert not any(path.startswith("pretraining.") for path in fields)
    assert fields["preprocessing.resize_mode"]["choices"] == [
        "fit_pad", "fit_pad_max_2x", "fit_pad_max_3x", "center_pad",
        "center_roi_pad", "fill_crop", "center_crop", "stretch", "none", "fit",
    ]
    assert fields["preprocessing.resize_mode"]["exposure"] == "internal"
    assert fields["data.num_classes"]["editable"] is False
    assert fields["data.num_classes"]["exposure"] == "internal"
    assert fields["data.batch_size"]["exposure"] == "internal"
    assert fields["stem.kernel_size"]["minimum"] == 1
    assert fields["encoder.num_layers"]["visible_when"] == {"encoder.family": "simple_cnn"}
    assert fields["encoder.num_layers"]["minimum"] == 1
    assert fields["encoder.base_filters"]["default"] == 32
    assert fields["encoder.filter_multiplier"]["default"] == 2.0
    assert {"none", "log", "sqrt", "inverse"} <= set(fields["metadata.fields[].transform"]["choices"])
    assert fields["training.learning_rate"]["exposure"] == "standard"
    assert fields["training.epochs"]["control"] == "number"
    assert fields["training.epochs"]["minimum"] == 1
    assert fields["training.epochs"]["step"] == 1
    assert fields["training.epochs"]["exposure"] == "standard"
    assert fields["training.optimizer"]["choices"] == list(STANDARD_TRAINING_OPTIMIZERS)
    assert fields["training.optimizer"]["control"] == "select"
    assert fields["training.learning_rate"]["control"] == "number"
    assert fields["training.learning_rate"]["minimum"] == 1e-8
    assert fields["training.loss"]["allowed_values_by"] == {
        key: list(values) for key, values in STANDARD_TRAINING_LOSSES_BY_TASK.items()
    }
    assert fields["training.loss"]["allowed_values_path"] == "run.task"
    assert fields["training.class_weights.mode"]["choices"] == [
        "explicit", "inverse_frequency", "effective_number", "power_law"
    ]
    assert fields["training.class_weights.alpha"]["default"] == 0.5
    assert fields["training.class_weights.alpha"]["minimum"] == 0
    assert fields["training.class_weights.alpha"]["maximum"] == 1
    assert fields["training.metrics"]["control"] == "multi_select"
    assert fields["training.metrics"]["allowed_values_by"] == {
        key: list(values) for key, values in STANDARD_TRAINING_METRICS_BY_TASK.items()
    }
    assert fields["training.metrics"]["allowed_values_path"] == "run.task"
    assert fields["augmentation.enabled"]["exposure"] == "standard"
    assert fields["augmentation.enabled"]["control"] == "toggle"
    assert fields["augmentation.gaussian_blur_max_sigma"]["default"] == 1.0
    assert fields["augmentation.gaussian_blur_max_sigma"]["minimum"] == 0
    assert fields["distribution.strategy"]["exposure"] == "standard"
    assert fields["distribution.strategy"]["control"] == "select"
    assert fields["distribution.strategy"]["choices"] == list(STANDARD_DISTRIBUTION_STRATEGIES)
    assert fields["distribution.devices"]["exposure"] == "advanced"
    assert fields["data.materialization.mode"]["exposure"] == "advanced"
    assert fields["data.streaming.reader_workers"]["exposure"] == "expert"
    assert {item["id"] for item in schema["exposures"]} == {
        "standard", "advanced", "expert", "internal"
    }
    assert fields["preprocessing.invert"]["type"] == "boolean_or_auto"
    assert fields["preprocessing.invert"]["control"] == "segmented"
    assert fields["preprocessing.invert"]["choice_options"] == [
        {"value": True, "label": "Yes"},
        {"value": False, "label": "No"},
        {"value": "auto", "label": "Auto"},
    ]
    assert fields["inference.batch_size"]["type"] == "integer_or_auto"
    assert fields["inference.batch_size"]["control"] == "auto_number"
    assert fields["input.channels"]["control"] == "multi_select"
    assert fields["input.channels"]["required_choices"] == ["intensity"]
    assert fields["input.channels"]["choice_options"][:2] == [
        {"value": "intensity", "label": "Intensity"},
        {"value": "gradient_magnitude", "label": "Sobel gradient"},
    ]
    assert fields["preprocessing.derived_channels.gradient_magnitude"]["exposure"] == "internal"


def test_catalog_covers_every_active_field_in_the_exhaustive_v2_reference():
    documented = {
        path for path in _leaf_paths(load_toml(REFERENCE_V2))
        if path != "run.model" and not path.startswith(("model.", "pretraining."))
    }
    catalogued = {field["path"] for field in configuration_schema()["fields"]}
    assert documented <= catalogued


def test_v2_resolver_rejects_runtime_compatibility_paths_and_derives_dataset_facts():
    authoring = {
        "architecture": {"version": 2},
        "run": {"task": "classification"},
        "encoder": {"family": "resnet", "variant": "resnet18"},
        "data": {"input_shape": [128, 128]},
    }
    resolved = resolve_v2_config(authoring, dataset_facts={"num_classes": 3})
    assert resolved["data"]["num_classes"] == 3
    assert "model" not in resolved
    assert "pretraining" not in resolved

    errors = v2_validation_errors({**authoring, "run": {"task": "classification", "model": "resnet"}})
    assert errors == [{
        "path": "run.model",
        "code": "legacy_field",
        "message": "run.model is not authorable in Architecture V2.",
    }]
    with pytest.raises(ValueError, match="data.num_classes is not authorable"):
        resolve_v2_config({**authoring, "data": {"input_shape": [128, 128], "num_classes": 3}})


def test_simple_cnn_v2_topology_maps_to_runtime_and_rejects_invalid_schedule():
    authored = {"architecture": {"version": 2}, "encoder": {
        "family": "simple_cnn", "variant": "simple_cnn",
        "num_layers": 4, "base_filters": 16, "filter_multiplier": 1.5,
    }}
    resolved = deep_merge(DEFAULT_CONFIG, authored)
    normalize_component_config(resolved, authored)
    assert resolved["model"]["conv_layers"] == 4
    assert resolved["model"]["base_filters"] == 16
    assert resolved["model"]["filter_multiplier"] == 1.5
    resolved["run"]["task"] = "classification"
    resolved["data"]["num_classes"] = 2
    resolved["data"]["input_shape"] = [32, 32]
    resolved["training"]["loss"] = "sparse_categorical_crossentropy"
    resolved["preprocessing"]["invert"] = False
    resolved["model"]["filter_multiplier"] = 0
    with pytest.raises(ValueError, match="filter_multiplier"):
        validate_config(resolved)


def test_ssl_classification_accepts_roi_context_but_embedding_rejects_it():
    authored = {
        "architecture": {"version": 2},
        "run": {"task": "classification"},
        "data": {"input_shape": [32, 32], "num_classes": 2},
        "encoder": {"family": "simple_cnn", "variant": "simple_cnn"},
        "metadata": {"fields": [{"name": "roi_area", "source": "roi.bounding_box_area_px", "transform": "log"}]},
        "self_supervised": {"enabled": True, "method": "simclr"},
        "training": {"loss": "sparse_categorical_crossentropy"},
        "preprocessing": {"invert": False},
    }
    resolved = deep_merge(DEFAULT_CONFIG, authored)
    normalize_component_config(resolved, authored)
    validate_config(resolved)
    resolved["classification"] = {"stratification": {"enabled": True, "dimensions": [32]}}
    with pytest.raises(ValueError, match="not yet compatible with self-supervised|does not support resolution stratification"):
        validate_config(resolved)
    resolved.pop("classification")
    resolved["run"]["task"] = "embedding"
    with pytest.raises(ValueError, match="Embedding artifacts do not yet support"):
        validate_config(resolved)


def test_v2_preset_generator_is_idempotent_and_recipes_parse(tmp_path: Path):
    output = tmp_path / "classification_defaults"
    script = PROJECT_ROOT / "scripts" / "generate_classification_default_configs.py"
    generated = subprocess.run(
        [sys.executable, str(script), "--output", str(output)],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "Wrote" in generated.stdout
    first = {path.name: path.read_bytes() for path in output.glob("*.toml")}
    verified = subprocess.run(
        [sys.executable, str(script), "--output", str(output), "--check"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "Verified" in verified.stdout
    assert first == {path.name: path.read_bytes() for path in output.glob("*.toml")}
    for path in output.glob("*.toml"):
        document = load_toml(path)
        assert document["architecture"]["version"] == 2
        assert document["encoder"]["variant"]
        assert "model" not in document
        assert "model" not in document["run"]


def test_checked_in_classification_defaults_match_the_generator():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "generate_classification_default_configs.py"), "--check"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "Verified 36" in result.stdout


def test_posthoc_catalog_exposes_models_and_dimensionality_controls():
    fields = {field['path']: field for field in configuration_schema()['fields']}
    assert fields['posthoc.estimator']['choices'] == [
        'logistic_regression', 'linear_svm', 'gradient_boosting', 'shallow_mlp',
    ]
    assert fields['posthoc.scaling']['choices'] == ['standard', 'robust', 'none']
    assert fields['posthoc.pca_components']['minimum'] == 1
    assert fields['posthoc.pca_components']['visible_when'] == {'posthoc.pca_enabled': True}
    assert fields['posthoc.calibration_folds']['minimum'] == 2


@pytest.mark.parametrize('settings', [
    {'estimator': 'unknown'}, {'pca_enabled': True, 'pca_components': 0},
    {'estimator': 'linear_svm', 'classifier_options': {'kernel': 'rbf'}},
    {'estimator': 'shallow_mlp', 'classifier_options': {'hidden_layer_sizes': [32, 16, 8]}},
])
def test_v2_authoring_rejects_invalid_posthoc_settings(settings):
    config = v2_defaults()
    config['posthoc'] = settings
    assert any(error['path'] == 'posthoc' for error in v2_validation_errors(config))


@pytest.mark.parametrize('estimator', ['logistic_regression', 'linear_svm', 'gradient_boosting', 'shallow_mlp'])
def test_v2_authoring_accepts_posthoc_models(estimator):
    config = v2_defaults()
    config['posthoc'] = {'enabled': True, 'estimator': estimator, 'pca_enabled': True, 'pca_components': 8}
    assert not any(error['path'] == 'posthoc' for error in v2_validation_errors(config))


@pytest.mark.parametrize('settings', [False, [], 'invalid', {'estimator': []}, {'scaling': {}}, {'estimator': 'shallow_mlp', 'classifier_options': {'early_stopping': 'false'}}])
def test_posthoc_invalid_json_types_return_authoring_errors(settings):
    config = v2_defaults()
    config['posthoc'] = settings
    assert any(error['path'] == 'posthoc' for error in v2_validation_errors(config))
