from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest
import tensorflow as tf

from oracle_builder.config import validate_config
from oracle_builder.classification.features import (
    build_embedding_model,
    build_pretraining_embedding_model,
)
from oracle_builder.data.sqlite_dataset import (
    create_synthetic_classification,
    load_arrays,
    load_prediction_arrays,
)
from oracle_builder.training.augmentation import augment_batch
from oracle_builder.training.student_teacher import (
    SimCLRPretrainer,
    StudentTeacherPretrainer,
    _view_config,
    foreground_weighted_reconstruction_mse,
    grayscale_reconstruction_config,
    load_grayscale_reconstruction_dataset,
    make_pretraining_dataset,
    _pretraining_optimizer,
    run_student_teacher_pretraining,
    vicreg_regularization,
)
from oracle_builder.training.train import build_and_compile_model
from oracle_builder.registry import get_model_builder


def pretraining_config():
    return {
        "run": {"task": "classification", "model": "simple_cnn", "seed": 123},
        "data": {
            "input_shape": [16, 16, 1],
            "num_classes": 3,
            "batch_size": 2,
            "shuffle_buffer": 8,
        },
        "model": {
            "base_filters": 2,
            "dropout": 0.0,
            "embedding_dim": 8,
            "normalize_embeddings": True,
        },
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"],
        },
        "pretraining": {
            "enabled": True,
            "method": "byol",
            "epochs": 1,
            "learning_rate": 0.001,
            "teacher_momentum": 0.9,
            "projection_dim": 4,
            "projection_hidden_dim": 8,
            "augmentation": {
                "rotation": 0.0,
                "zoom": 0.0,
                "translation": 0.0,
                "skew": 0.0,
                "flip_horizontal": True,
                "brightness": 0.1,
                "contrast": 0.1,
                "gaussian_noise": 0.01,
            },
        },
    }


def test_ssl_view_augmentation_can_be_disabled():
    config = pretraining_config()
    config["pretraining"]["use_training_augmentation"] = True
    config.setdefault("augmentation", {}).update(
        {
            "enabled": True,
            "rotation": 0.5,
            "zoom": 0.2,
            "translation": 0.15,
        }
    )
    config["pretraining"]["augmentation"] = {
        "enabled": False,
        "rotation": 0.5,
        "zoom": 0.2,
        "translation": 0.15,
    }

    view_config = _view_config(config)
    assert view_config["augmentation"]["enabled"] is False

    values = tf.reshape(tf.range(16, dtype=tf.float32), [1, 4, 4, 1])
    labels = tf.zeros([1], dtype=tf.int32)
    unchanged, unchanged_labels = augment_batch(values, labels, view_config)

    np.testing.assert_array_equal(unchanged.numpy(), values.numpy())
    np.testing.assert_array_equal(unchanged_labels.numpy(), labels.numpy())


def test_self_supervised_views_do_not_inherit_training_augmentation():
    config = pretraining_config()
    config["pretraining"]["use_training_augmentation"] = True
    config.setdefault("augmentation", {}).update(
        {"enabled": False, "rotation": 0.5, "zoom": 0.2}
    )

    view_config = _view_config(config)

    assert view_config["augmentation"]["enabled"] is True
    assert view_config["augmentation"]["rotation"] == 0.0
    assert view_config["augmentation"]["zoom"] == 0.0


def test_self_supervised_config_section_is_supported():
    config = pretraining_config()
    config["self_supervised"] = config.pop("pretraining")
    config["self_supervised"]["augmentation"] = {"enabled": False}

    view_config = _view_config(config)

    assert view_config["augmentation"]["enabled"] is False


@pytest.mark.parametrize("method", ["byol", "simclr"])
def test_student_teacher_pretraining_updates_shared_classifier_encoder(
    tmp_path, method, capsys
):
    config = pretraining_config()
    config["pretraining"]["method"] = method
    classifier = build_and_compile_model(config)
    x = np.random.default_rng(123).random((4, 16, 16, 1), dtype=np.float32)
    dataset = make_pretraining_dataset(x, config)
    before = classifier.get_layer("embedding_projection").get_weights()[0].copy()

    history = run_student_teacher_pretraining(classifier, dataset, config, tmp_path)

    output = capsys.readouterr().out
    assert f"[self-supervised {method} 1/1] started" in output
    assert f"[self-supervised {method} 1/1] completed" in output

    after = classifier.get_layer("embedding_projection").get_weights()[0]
    assert "loss" in history.history
    assert np.isfinite(history.history["loss"][-1])
    assert not np.allclose(before, after)
    supervised_loss = classifier.train_on_batch(
        x[:2],
        np.array([0, 1], dtype="int64"),
    )
    assert np.all(np.isfinite(supervised_loss))
    assert (tmp_path / "metrics" / "pretraining" / "metrics.csv").exists()
    assert (
        tmp_path
        / "model"
        / "pretraining"
        / "student_pretrained.weights.h5"
    ).exists()
    metrics = json.loads(
        (tmp_path / "metrics" / "pretraining" / "metrics.json").read_text()
    )
    assert len(metrics["loss"]) == 1
    if method == "byol":
        for metric in (
            "variance_loss",
            "covariance_loss",
            "representation_std",
            "collapse_indicator",
        ):
            assert metric in history.history
            assert np.isfinite(history.history[metric][-1])


def test_teacher_starts_with_student_encoder_weights():
    config = pretraining_config()
    classifier = build_and_compile_model(config)
    pretrainer = StudentTeacherPretrainer(classifier, config)

    for student, teacher in zip(
        pretrainer.student_encoder.get_weights(),
        pretrainer.teacher_encoder.get_weights(),
        strict=True,
    ):
        np.testing.assert_allclose(student, teacher)
    assert not pretrainer.teacher_encoder.trainable
    assert not pretrainer.teacher_projector.trainable


def test_ssl_encoder_uses_unnormalized_embedding_projection():
    config = pretraining_config()
    classifier = build_and_compile_model(config)
    projection = classifier.get_layer("embedding_projection")
    kernel, bias = projection.get_weights()
    projection.set_weights([np.zeros_like(kernel), np.full_like(bias, 2.0)])
    inputs = np.zeros((2, 16, 16, 1), dtype="float32")

    serving_embeddings = build_embedding_model(classifier)(inputs, training=False)
    ssl_embeddings = build_pretraining_embedding_model(classifier)(inputs, training=False)
    pretrainer = StudentTeacherPretrainer(classifier, config)
    pretrainer_embeddings = pretrainer.student_encoder(inputs, training=False)

    np.testing.assert_allclose(np.linalg.norm(serving_embeddings.numpy(), axis=1), 1.0)
    np.testing.assert_allclose(ssl_embeddings.numpy(), 2.0)
    np.testing.assert_allclose(pretrainer_embeddings.numpy(), ssl_embeddings.numpy())


def test_vicreg_regularization_detects_collapsed_representations():
    collapsed = tf.ones((4, 3), dtype=tf.float32)
    varied = tf.constant(
        [[-2.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [2.0, 1.0, 0.0]],
        dtype=tf.float32,
    )

    collapsed_variance, collapsed_covariance, collapsed_std = vicreg_regularization(collapsed)
    varied_variance, varied_covariance, varied_std = vicreg_regularization(varied)

    assert collapsed_variance.numpy() > varied_variance.numpy()
    assert collapsed_std.numpy() < varied_std.numpy()
    assert np.isfinite(collapsed_covariance.numpy())
    assert np.isfinite(varied_covariance.numpy())


def test_ssl_optimizer_supports_adamw_without_changing_adam_default():
    adam = _pretraining_optimizer(pretraining_config()["pretraining"])
    adamw = _pretraining_optimizer(
        {**pretraining_config()["pretraining"], "ssl_optimizer": "adamw", "weight_decay": 0.01}
    )

    assert isinstance(adam, tf.keras.optimizers.Adam)
    assert isinstance(adamw, tf.keras.optimizers.AdamW)
    assert float(adamw.weight_decay) == pytest.approx(0.01)


@pytest.mark.parametrize("pretrainer_type", [StudentTeacherPretrainer, SimCLRPretrainer])
def test_pretraining_wrappers_expose_a_buildable_call_path(pretrainer_type):
    config = pretraining_config()
    classifier = build_and_compile_model(config)
    pretrainer = pretrainer_type(classifier, config)

    output = pretrainer(np.zeros((1, 16, 16, 1), dtype="float32"), training=False)

    assert output.shape == (1, 4)
    assert pretrainer.built


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("epochs", 0, "epochs"),
        ("teacher_momentum", 1.0, "momentum"),
        ("projection_dim", 0, "projection_dim"),
        ("reconstruction_foreground_weight", 0.5, "foreground_weight"),
    ],
)
def test_invalid_pretraining_configuration_is_rejected(field, value, message):
    config = pretraining_config()
    config["pretraining"][field] = value

    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_pretraining_is_rejected_for_segmentation():
    config = pretraining_config()
    config["run"]["task"] = "segmentation"
    config["run"]["model"] = "unet"
    config["data"]["output_shape"] = [16, 16, 1]

    with pytest.raises(ValueError, match="requires method='grayscale_reconstruction'"):
        validate_config(config)


def test_grayscale_reconstruction_requires_existing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_grayscale_reconstruction_dataset(tmp_path / "missing.sqlite", pretraining_config())


def test_grayscale_reconstruction_uses_linear_disposable_head():
    config = pretraining_config()
    config["run"].update({"task": "segmentation", "model": "unet"})
    config["data"]["output_shape"] = [16, 16, 1]
    config["model"]["final_activation"] = "sigmoid"

    reconstruction = grayscale_reconstruction_config(config)

    assert reconstruction["data"]["input_shape"] == [16, 16, 1]
    assert reconstruction["model"]["final_activation"] == "linear"
    assert config["model"]["final_activation"] == "sigmoid"


def test_foreground_weighted_reconstruction_mse_emphasizes_bright_pixels():
    targets = np.array([[[[0.0], [1.0]]]], dtype="float32")
    prediction = np.zeros_like(targets)

    loss = foreground_weighted_reconstruction_mse(targets, prediction, 4.0)

    np.testing.assert_allclose(loss.numpy(), [2.0])


def test_grayscale_reconstruction_pretraining_runs_with_linear_head(tmp_path):
    from oracle_builder.training.student_teacher import (
        run_grayscale_reconstruction_pretraining,
    )

    config = pretraining_config()
    config["run"].update({"task": "segmentation", "model": "unet"})
    config["data"]["output_shape"] = [16, 16, 1]
    config["pretraining"].update(
        {
            "method": "grayscale_reconstruction",
            "reconstruction_foreground_weight": 4.0,
        }
    )
    images = np.zeros((2, 16, 16, 1), dtype="float32")
    images[:, 4:12, 4:12] = 1.0
    dataset = tf.data.Dataset.from_tensor_slices((images, images)).batch(2)
    model = get_model_builder("unet")(config)

    history = run_grayscale_reconstruction_pretraining(model, dataset, config, tmp_path)

    assert np.isfinite(history.history["loss"][-1])
    for metric in (
        "reconstruction_mse",
        "foreground_mse",
        "prediction_mean",
        "prediction_std",
    ):
        assert metric in history.history
        assert np.isfinite(history.history[metric][-1])
    assert (tmp_path / "model" / "pretraining" / "grayscale_reconstruction.weights.h5").exists()


def test_unlabeled_train_rois_are_available_only_to_pretraining(tmp_path):
    database = tmp_path / "classification.sqlite"
    create_synthetic_classification(database, n=6, shape=(16, 16, 1), classes=3)
    config = pretraining_config()
    _, _, initial_train_records = load_prediction_arrays(
        database,
        config,
        split="train",
    )
    unlabeled_uuid = initial_train_records[0]["uuid"]
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE classification_annotations
            SET is_current = 0
            WHERE item_id = ?
            """,
            (unlabeled_uuid,),
        )
    _, supervised_y, supervised_records = load_arrays(database, config, split="train")
    _, pretraining_y, pretraining_records = load_prediction_arrays(
        database,
        config,
        split="train",
    )

    assert unlabeled_uuid not in {record["uuid"] for record in supervised_records}
    assert unlabeled_uuid in {record["uuid"] for record in pretraining_records}
    assert len(pretraining_records) == len(supervised_records) + 1
    unlabeled_index = next(
        index
        for index, record in enumerate(pretraining_records)
        if record["uuid"] == unlabeled_uuid
    )
    assert pretraining_y[unlabeled_index] is None
    assert len(supervised_y) == len(supervised_records)
