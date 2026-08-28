from __future__ import annotations

import numpy as np
import pytest
from tensorflow import keras

from oracle_builder.classification.features import build_feature_model
from oracle_builder.registry import get_model_builder
from oracle_builder.training.train import build_and_compile_model


def classification_config(model_name: str, model_options: dict | None = None):
    return {
        "run": {"task": "classification", "model": model_name, "seed": 123},
        "data": {"input_shape": [32, 32, 3], "num_classes": 4},
        "model": model_options or {},
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"],
        },
    }


@pytest.mark.parametrize(
    ("family", "variant", "options"),
    [
        (
            "resnet",
            "resnet18",
            {"block_counts": [1, 1, 1, 1], "base_filters": 8, "stem_pool": False},
        ),
        (
            "densenet",
            "densenet121",
            {
                "block_config": [1, 1, 1, 1],
                "growth_rate": 4,
                "initial_filters": 8,
                "stem_pool": False,
            },
        ),
        (
            "efficientnet",
            "efficientnet_b0",
            {
                "width_coefficient": 0.25,
                "depth_coefficient": 0.25,
                "stem_filters": 8,
                "top_filters": 16,
            },
        ),
    ],
)
def test_classification_families_build_and_train(family, variant, options):
    options.update(
        {
            "variant": variant,
            "stem_kernel_size": 5,
            "stem_stride": 1,
            "embedding_dim": 24,
            "normalize_embeddings": True,
        }
    )
    model = build_and_compile_model(classification_config(family, options))

    assert model.name == variant
    assert model.output_shape == (None, 4)
    assert model.get_layer("stem_conv").kernel_size == (5, 5)
    assert model.get_layer("stem_conv").strides == (1, 1)

    x = np.ones((2, 32, 32, 3), dtype="float32")
    prediction = model.predict(x, verbose=0)
    inference = build_feature_model(model).predict(x, verbose=0)
    assert prediction.shape == (2, 4)
    assert inference["features"].shape == (2, 24)
    np.testing.assert_allclose(inference["probabilities"], prediction, atol=1e-6)
    np.testing.assert_allclose(
        np.linalg.norm(inference["features"], axis=1),
        1.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(prediction.sum(axis=1), 1.0, atol=1e-5)
    loss = model.train_on_batch(x, np.array([0, 1], dtype="int64"))
    assert np.all(np.isfinite(loss))


@pytest.mark.parametrize(
    "model_name",
    ["resnet18", "resnet50", "densenet121", "densenet201", "efficientnet_b0", "efficientnet_b7"],
)
def test_named_variants_select_their_architecture(model_name):
    options = {
        "block_counts": [1, 1, 1, 1],
        "block_config": [1, 1, 1, 1],
        "base_filters": 4,
        "growth_rate": 2,
        "initial_filters": 4,
        "width_coefficient": 0.25,
        "depth_coefficient": 0.25,
        "stem_filters": 8,
        "top_filters": 16,
        "stem_pool": False,
    }
    model = get_model_builder(model_name)(classification_config(model_name, options))
    assert model.name == model_name


@pytest.mark.parametrize("family", ["resnet", "densenet", "efficientnet"])
def test_unknown_classification_variant_is_rejected(family):
    with pytest.raises(ValueError, match="Unknown"):
        get_model_builder(family)(
            classification_config(family, {"variant": "not_a_real_architecture"})
        )


@pytest.mark.parametrize(
    "model_name",
    ["simple_cnn", "resnet_like", "densenet_like"],
)
def test_existing_classifiers_follow_feature_contract(model_name):
    config = classification_config(
        model_name,
        {"base_filters": 4, "dropout": 0.0, "embedding_dim": 12},
    )
    model = get_model_builder(model_name)(config)
    outputs = build_feature_model(model).predict(
        np.ones((2, 32, 32, 3), dtype="float32"),
        verbose=0,
    )

    assert model.output_shape == (None, 4)
    assert outputs["features"].shape == (2, 12)
    np.testing.assert_allclose(np.linalg.norm(outputs["features"], axis=1), 1.0, atol=1e-5)


def test_embedding_normalization_can_be_disabled():
    config = classification_config(
        "simple_cnn",
        {
            "base_filters": 4,
            "dropout": 0.0,
            "embedding_dim": 7,
            "normalize_embeddings": False,
        },
    )
    model = get_model_builder("simple_cnn")(config)

    assert model.get_layer("features").output.shape[-1] == 7


def test_resnet_programmatic_defaults_are_roi_friendly():
    model = get_model_builder("resnet")(classification_config("resnet"))

    assert model.name == "resnet18"
    assert model.get_layer("stem_conv").kernel_size == (3, 3)
    assert model.get_layer("stem_conv").strides == (1, 1)
    assert "stem_pool" not in {layer.name for layer in model.layers}
    assert model.get_layer("features").__class__.__name__ == "Activation"


@pytest.mark.parametrize(
    ("variant", "block_type", "stage_depths"),
    [
        ("resnet18", "basic", [2, 2, 2, 2]),
        ("resnet34", "basic", [3, 4, 6, 3]),
        ("resnet50", "bottleneck", [3, 4, 6, 3]),
        ("resnet101", "bottleneck", [3, 4, 23, 3]),
        ("resnet152", "bottleneck", [3, 8, 36, 3]),
    ],
)
def test_canonical_resnet_variants_have_expected_stages_and_shortcuts(
    variant, block_type, stage_depths
):
    base_filters = 2
    model = get_model_builder(variant)(
        classification_config(
            variant,
            {"base_filters": base_filters, "stem_stride": 1, "stem_pool": False},
        )
    )
    expansion = 1 if block_type == "basic" else 4
    downsampling_conv = "conv1" if block_type == "basic" else "conv2"

    for stage, depth in enumerate(stage_depths, start=1):
        stage_outputs = [
            layer
            for layer in model.layers
            if layer.name.startswith(f"stage{stage}_block") and layer.name.endswith("_out")
        ]
        assert len(stage_outputs) == depth
        assert stage_outputs[-1].output.shape[-1] == base_filters * (2 ** (stage - 1)) * expansion

        first_block = f"stage{stage}_block1"
        expected_stride = (2, 2) if stage > 1 else (1, 1)
        assert model.get_layer(f"{first_block}_{downsampling_conv}").strides == expected_stride

        projection_layers = [
            layer
            for layer in model.layers
            if layer.name.endswith("_projection") and layer.name.startswith(f"stage{stage}_")
        ]
        assert len(projection_layers) == (1 if stage > 1 or block_type == "bottleneck" else 0)
        if projection_layers:
            assert projection_layers[0].strides == expected_stride

        for block in range(2, depth + 1):
            assert model.get_layer(f"stage{stage}_block{block}_{downsampling_conv}").strides == (1, 1)

    conv_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Conv2D)]
    batch_norm_layers = [
        layer for layer in model.layers if isinstance(layer, keras.layers.BatchNormalization)
    ]
    assert conv_layers
    assert batch_norm_layers
    assert all(isinstance(layer.kernel_initializer, keras.initializers.HeNormal) for layer in conv_layers)
    assert all(layer.momentum == pytest.approx(0.99) for layer in batch_norm_layers)
    assert all(layer.epsilon == pytest.approx(1e-3) for layer in batch_norm_layers)
