from pathlib import Path

import pytest

from oracle_builder.config import DEFAULT_CONFIG, deep_merge, load_toml, validate_config
from oracle_builder.registry import MODEL_REGISTRY


CONFIG_ROOT = Path(__file__).parents[1] / "configs"
CONFIG_DIR = CONFIG_ROOT / "classification_defaults"
FAMILIES = {
    "simple_cnn",
    "resnet_like",
    "densenet_like",
    "resnet",
    "densenet",
    "efficientnet",
}
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


def test_documented_default_exists_for_each_classification_family():
    assert {path.stem for path in CONFIG_DIR.glob("*.toml")} == FAMILIES


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_documented_classification_defaults_are_valid_and_dataset_independent(family):
    user_config = load_toml(CONFIG_DIR / f"{family}.toml")

    assert user_config["run"]["model"] == family
    assert family in MODEL_REGISTRY
    assert "num_classes" not in user_config["data"]

    resolved = deep_merge(DEFAULT_CONFIG, user_config)
    resolved["data"]["num_classes"] = 3
    validate_config(resolved)


def test_family_defaults_share_the_same_augmentation_policy():
    policies = [
        load_toml(path)["augmentation"]
        for path in sorted(CONFIG_DIR.glob("*.toml"))
    ]
    assert policies
    assert all(policy == policies[0] for policy in policies[1:])


def test_resnet_default_preserves_roi_detail_and_uses_raw_classifier_embeddings():
    config = load_toml(CONFIG_DIR / "resnet.toml")

    assert config["data"]["input_shape"] == [128, 128, 1]
    assert config["model"]["variant"] == "resnet18"
    assert config["model"]["stem_kernel_size"] == 3
    assert config["model"]["stem_stride"] == 1
    assert config["model"]["stem_pool"] is False
    assert config["model"]["normalize_embeddings"] is False


def test_resnet_like_default_uses_roi_input_and_raw_classifier_embeddings():
    config = load_toml(CONFIG_DIR / "resnet_like.toml")

    assert config["data"]["input_shape"] == [128, 128, 1]
    assert config["model"]["normalize_embeddings"] is False


@pytest.mark.parametrize(
    "path",
    classification_config_paths(),
    ids=lambda path: path.name,
)
def test_all_classification_examples_share_high_level_defaults(path):
    user_config = load_toml(path)

    assert user_config["run"]["task"] == "classification"
    assert "num_classes" not in user_config["data"]
    assert user_config["data"]["input_shape"][-1] == 1
    assert user_config["preprocessing"]["channel_mode"] == "grayscale"
    assert user_config["training"]["loss"] == (
        "weighted_sparse_categorical_crossentropy"
    )
    assert user_config["training"]["class_weights"]["mode"] == "effective_number"
    assert user_config["augmentation"] == STANDARD_AUGMENTATION
    assert user_config["output"]["save_checkpoints"] is False
    assert user_config["recovery"] == {"enabled": True, "save_every_epochs": 1}

    resolved = deep_merge(DEFAULT_CONFIG, user_config)
    resolved["data"]["num_classes"] = 3
    validate_config(resolved)
