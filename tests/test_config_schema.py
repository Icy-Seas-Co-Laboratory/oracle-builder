"""Contract tests for the V2 configuration catalog and generated presets."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from oracle_builder.config import (
    STANDARD_DISTRIBUTION_STRATEGIES,
    STANDARD_TRAINING_LOSSES_BY_TASK,
    STANDARD_TRAINING_METRICS_BY_TASK,
    STANDARD_TRAINING_OPTIMIZERS,
    load_toml,
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
    assert fields["training.metrics"]["control"] == "multi_select"
    assert fields["training.metrics"]["allowed_values_by"] == {
        key: list(values) for key, values in STANDARD_TRAINING_METRICS_BY_TASK.items()
    }
    assert fields["training.metrics"]["allowed_values_path"] == "run.task"
    assert fields["augmentation.enabled"]["exposure"] == "standard"
    assert fields["augmentation.enabled"]["control"] == "toggle"
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
