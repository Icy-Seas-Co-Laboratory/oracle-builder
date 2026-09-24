from pathlib import Path

import pytest

from oracle_builder.config import DEFAULT_CONFIG, deep_merge, load_toml, validate_config
from oracle_builder.registry import MODEL_REGISTRY


CONFIG_ROOT = Path(__file__).parents[1] / "configs"
CONFIG_DIR = CONFIG_ROOT / "classification_defaults"
BASELINE_PRESETS = {
    "simple_cnn",
    "resnet_like",
    "densenet_like",
    "resnet",
    "densenet",
    "efficientnet",
}
VARIANT_PRESETS = {
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "densenet121", "densenet169", "densenet201",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    "efficientnet_v2", "efficientnet_v2_b0", "efficientnet_v2_b1",
    "efficientnet_v2_b2", "efficientnet_v2_b3", "efficientnet_v2_s",
    "efficientnet_v2_m", "efficientnet_v2_l",
    "convnext", "convnext_tiny", "convnext_small",
    "mobilenet", "mobilenet_v3_small", "mobilenet_v3_large",
}
PRESETS = BASELINE_PRESETS | VARIANT_PRESETS
STANDARD_AUGMENTATION = {
    "enabled": True,
    "repeats_per_epoch": 1,
    "invert": False,
    "rotation": 0.5,
    "zoom": 0.20,
    "translation": [0.15, 0.15],
    "skew": 0.20,
    "flip_horizontal": True,
    "flip_vertical": True,
    "brightness": 0.20,
    "contrast": 0.20,
    "gaussian_noise": 0.05,
    "fill_value": 0.0,
}


def classification_config_paths():
    paths = sorted(CONFIG_DIR.glob("*.toml"))
    paths.extend(sorted(CONFIG_ROOT.glob("example_classification*.toml")))
    return paths


def validate_dataset_independent_recipe(user_config):
    """Validate a recipe after substituting facts normally resolved from its dataset."""
    resolved = deep_merge(DEFAULT_CONFIG, user_config)
    resolved["data"]["num_classes"] = 3
    # ``auto`` polarity is intentionally resolved from frozen dataset metadata
    # by the training planner; use a deterministic stand-in for static tests.
    if resolved.get("preprocessing", {}).get("invert") == "auto":
        resolved["preprocessing"]["invert"] = False
    validate_config(resolved)


def test_documented_default_exists_for_every_classification_preset():
    assert {path.stem for path in CONFIG_DIR.glob("*.toml")} == PRESETS


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_documented_classification_presets_are_explicit_v2_and_dataset_independent(preset):
    user_config = load_toml(CONFIG_DIR / f"{preset}.toml")

    assert user_config["run"]["model"] in MODEL_REGISTRY
    assert user_config["architecture"]["version"] == 2
    for section in ("input", "encoder", "normalization", "pooling", "image_embedding", "metadata", "fusion", "classifier"):
        assert section in user_config
    assert "num_classes" not in user_config["data"]

    validate_dataset_independent_recipe(user_config)


def test_classification_presets_have_a_safe_explicit_augmentation_policy():
    for path in sorted(CONFIG_DIR.glob("*.toml")):
        policy = load_toml(path)["augmentation"]
        assert policy["enabled"] is True
        assert 0 <= policy["rotation"] <= 0.5
        assert 0 <= policy["zoom"] <= 0.95
        assert policy["fill_value"] == 0.0


def test_resnet_default_preserves_roi_detail_and_uses_raw_classifier_embeddings():
    config = load_toml(CONFIG_DIR / "resnet.toml")

    assert config["data"]["input_shape"] == [128, 128]
    assert config["model"]["variant"] == "resnet18"
    assert config["model"]["stem_kernel_size"] == 3
    assert config["model"]["stem_stride"] == 1
    assert config["model"]["stem_pool"] is False
    assert config["model"]["normalize_embeddings"] is False


def test_resnet_like_default_uses_roi_input_and_raw_classifier_embeddings():
    config = load_toml(CONFIG_DIR / "resnet_like.toml")

    assert config["data"]["input_shape"] == [128, 128]
    assert config["model"]["normalize_embeddings"] is False


@pytest.mark.parametrize("family", sorted(BASELINE_PRESETS))
def test_family_defaults_include_the_safe_shared_stratified_recipe(family):
    config = load_toml(CONFIG_DIR / f"{family}.toml")
    settings = config["classification"]["stratification"]

    assert settings["enabled"] is False
    assert settings["dimensions"] == [32, 64, 128]
    assert settings["schedule"] == "interleaved_steps"
    assert settings["supra_epochs"] == 1
    assert settings["normalization"] == "group"
    assert settings["group_norm_groups"] == 8
    assert settings["conditioning"] == {"enabled": True, "embedding_dim": 16}
    assert settings["training_routing"]["enabled"] is False
    assert settings["training_routing"]["adjacent_lower_probability"] == 0.0
    assert settings["cycle_scheduler"]["aggregation"] == "equal_strata"
    assert settings["cycle_scheduler"]["guardrail_metric"] == "macro_f1"
    assert settings["cycle_scheduler"]["max_stratum_drop"] == 0.03
    assert config["training"]["learning_rate"] == 0.0003
    assert config["training"]["weight_decay"] == 0.0001


@pytest.mark.parametrize(
    "path",
    classification_config_paths(),
    ids=lambda path: path.name,
)
def test_all_classification_examples_share_high_level_defaults(path):
    user_config = load_toml(path)

    assert user_config["run"]["task"] == "classification"
    assert "num_classes" not in user_config["data"]
    assert len(user_config["data"]["input_shape"]) == 2
    assert user_config["preprocessing"]["channel_mode"] == "grayscale"
    assert user_config["training"]["loss"] == (
        "weighted_sparse_categorical_crossentropy"
    )
    assert user_config["architecture"]["version"] == 2
    assert user_config["training"]["class_weights"]["mode"] == "effective_number"
    assert user_config["training"]["metrics"] == ["accuracy", "macro_f1"]
    if path.parent != CONFIG_DIR:
        assert user_config["augmentation"] == STANDARD_AUGMENTATION
        assert user_config["output"]["save_checkpoints"] is False
        assert user_config["recovery"]["save_every_epochs"] == 1
    else:
        assert user_config["augmentation"]["enabled"] is True
        assert user_config["output"]["save_predictions"] is True

    # The maintained family defaults are the supported shared-stratification
    # recipes. Older top-level examples intentionally preserve their historic
    # preprocessing/SSL demonstrations and are covered by their own tests.
    if path.parent == CONFIG_DIR:
        validate_dataset_independent_recipe(user_config)
