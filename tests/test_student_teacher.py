from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from oracle_builder.config import validate_config
from oracle_builder.data.sqlite_dataset import (
    create_synthetic_classification,
    load_arrays,
    load_prediction_arrays,
)
from oracle_builder.training.student_teacher import (
    SimCLRPretrainer,
    StudentTeacherPretrainer,
    make_pretraining_dataset,
    run_student_teacher_pretraining,
)
from oracle_builder.training.train import build_and_compile_model


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


@pytest.mark.parametrize("method", ["byol", "simclr"])
def test_student_teacher_pretraining_updates_shared_classifier_encoder(
    tmp_path, method
):
    config = pretraining_config()
    config["pretraining"]["method"] = method
    classifier = build_and_compile_model(config)
    x = np.random.default_rng(123).random((4, 16, 16, 1), dtype=np.float32)
    dataset = make_pretraining_dataset(x, config)
    before = classifier.get_layer("embedding_projection").get_weights()[0].copy()

    history = run_student_teacher_pretraining(classifier, dataset, config, tmp_path)

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

    with pytest.raises(ValueError, match="only supported for classification"):
        validate_config(config)


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
