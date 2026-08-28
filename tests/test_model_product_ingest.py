from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

import tensorflow as tf
from tensorflow import keras

from oracle_builder.artifacts import validate_run_artifact
from oracle_builder.artifacts import read_run_config
from oracle_builder.classification.features import predict_classification_outputs
from oracle_builder.datasets.schema import initialize_database
from oracle_builder.products.ingest import ingest_keras_model, ingest_savedmodel
from oracle_builder.saving.load_test import load_model_for_run


def _write_info(path) -> None:
    path.write_text(
        """
[product]
name = "tiny-external-model"
task = "generic"
version = "1.0.0"
description = "A test model."
tags = ["test"]

[preprocessing]
rescale = true

[outputs]
primary = "scores"
""".lstrip(),
        encoding="utf-8",
    )


def test_ingest_keras_model_creates_a_sealed_portable_product(tmp_path):
    source = tmp_path / "external.keras"
    info = tmp_path / "product.toml"
    output = tmp_path / "product"
    inputs = keras.Input(shape=(4,), name="input")
    outputs = keras.layers.Dense(2, activation="softmax", name="scores")(inputs)
    keras.Model(inputs, outputs, name="tiny_model").save(source)
    _write_info(info)

    result = ingest_keras_model(source, info, output)

    manifest = json.loads((output / "artifact.json").read_text(encoding="utf-8"))
    inspection = json.loads(
        (output / "model" / "inspection.json").read_text(encoding="utf-8")
    )
    model_manifest = json.loads(
        (output / "model" / "model_manifest.json").read_text(encoding="utf-8")
    )
    assert validate_run_artifact(output)["valid"]
    assert manifest["artifact_type"] == "model_product"
    assert manifest["product"]["name"] == "tiny-external-model"
    assert result["artifact_id"] == manifest["artifact_id"]
    assert (output / "model" / "source" / "original.keras").exists()
    assert (output / "model" / "final.keras").exists()
    assert inspection["reload_test"]["keras_reloaded"] is True
    assert inspection["reload_test"]["attempted"] is True
    assert model_manifest["formats"][0]["format"] == "keras_v3"


def test_ingest_records_optional_dataset_provenance(tmp_path):
    database = tmp_path / "dataset.sqlite"
    with sqlite3.connect(database) as connection:
        initialize_database(connection, "classification", name="test-dataset")
        connection.commit()
    source = tmp_path / "external.keras"
    info = tmp_path / "product.toml"
    output = tmp_path / "product"
    inputs = keras.Input(shape=(4,))
    keras.Model(inputs, keras.layers.Dense(2)(inputs)).save(source)
    info.write_text('[product]\nname = "with-dataset"\ntask = "classification"\n')

    ingest_keras_model(source, info, output, dataset=database)

    manifest = json.loads((output / "artifact.json").read_text(encoding="utf-8"))
    assert manifest["dataset"]["dataset_type"] == "classification"
    assert len(manifest["dataset"]["fingerprint_sha256"]) == 64


def test_ingest_promotes_a_softmax_classifier_to_standard_outputs(tmp_path):
    source = tmp_path / "external.keras"
    info = tmp_path / "product.toml"
    output = tmp_path / "product"
    inputs = keras.Input(shape=(4,))
    representation = keras.layers.Dense(6, activation="relu", name="representation")(inputs)
    probabilities = keras.layers.Dense(3, activation="softmax", name="head")(representation)
    keras.Model(inputs, probabilities).save(source)
    info.write_text(
        '[product]\nname = "promoted"\ntask = "classification"\n\n[promotion]\nenabled = true\n',
        encoding="utf-8",
    )

    ingest_keras_model(source, info, output)

    loaded = load_model_for_run(output, read_run_config(output))
    values = predict_classification_outputs(loaded, np.zeros((2, 4), dtype="float32"))
    inspection = json.loads(
        (output / "model" / "inspection.json").read_text(encoding="utf-8")
    )
    assert inspection["promotion"]["promoted"] is True
    assert values["logits"].shape == (2, 3)
    assert values["probabilities"].shape == (2, 3)
    assert values["features"].shape == (2, 6)
    assert (output / "model" / "imported.keras").exists()


def test_ingest_preserves_a_generic_representation_contract_without_clustering(
    tmp_path,
):
    source = tmp_path / "encoder.keras"
    info = tmp_path / "product.toml"
    output = tmp_path / "product"
    inputs = keras.Input(shape=(4,), name="input")
    features = keras.layers.Dense(6, name="features")(inputs)
    keras.Model(inputs, features, name="encoder").save(source)
    info.write_text(
        """
[product]
name = "representation"
task = "generic"

[outputs]
primary = "representation"
representation = "features"
dimension = 6
normalized = true
""".lstrip(),
        encoding="utf-8",
    )

    ingest_keras_model(source, info, output)

    contract = json.loads((output / "model" / "contract.json").read_text())
    manifest = json.loads((output / "artifact.json").read_text())
    assert contract["task"] == "generic"
    assert contract["outputs"] == {
        "primary": "representation",
        "representation": "features",
        "dimension": 6,
        "normalized": True,
    }
    assert "cluster_evidence" not in contract["outputs"]
    assert manifest["model"]["outputs"] == contract["outputs"]


def test_ingest_rejects_clustering_as_a_model_product_task(tmp_path):
    source = tmp_path / "encoder.keras"
    info = tmp_path / "product.toml"
    inputs = keras.Input(shape=(4,), name="input")
    keras.Model(inputs, keras.layers.Dense(6, name="features")(inputs)).save(source)
    info.write_text('[product]\nname = "clusterer"\ntask = "clustering"\n')

    with pytest.raises(
        ValueError,
        match="product.task must be generic, classification, or segmentation",
    ):
        ingest_keras_model(source, info, tmp_path / "product")


def test_ingest_savedmodel_creates_named_classification_contract(tmp_path):
    class LegacyClassifier(tf.Module):
        @tf.function
        def serve(self, values):
            flattened = tf.reshape(values, [tf.shape(values)[0], 16])
            return {"output_0": tf.nn.softmax(flattened[:, :3])}

    source = tmp_path / "legacy_savedmodel"
    module = LegacyClassifier()
    tf.saved_model.save(
        module,
        str(source),
        signatures={"serving_default": module.serve.get_concrete_function(
            tf.TensorSpec([None, 4, 4, 1], tf.float32, name="input_layer")
        )},
    )
    info = tmp_path / "product.toml"
    info.write_text(
        '[product]\nname = "legacy"\ntask = "classification"\n\n'
        '[[labels]]\nname = "a"\n[[labels]]\nname = "b"\n[[labels]]\nname = "c"\n\n'
        '[promotion]\nactivation = "softmax"\n', encoding="utf-8"
    )
    output = tmp_path / "product"
    ingest_savedmodel(source, info, output)

    loaded = load_model_for_run(output, read_run_config(output), prefer_savedmodel=True)
    values = predict_classification_outputs(loaded, np.zeros((2, 4, 4, 1), dtype="float32"))
    assert validate_run_artifact(output)["valid"]
    assert values["logits"].shape == (2, 3)
    assert values["probabilities"].shape == (2, 3)
    assert values["features"] is None
