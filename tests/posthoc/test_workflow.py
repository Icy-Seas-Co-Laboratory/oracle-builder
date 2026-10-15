from __future__ import annotations

import json
import sqlite3
import uuid

from oracle_builder.artifacts import create_run_artifact, read_run_manifest, seal_run_artifact, update_run_artifact, validate_run_artifact
from oracle_builder.data.sqlite_dataset import create_synthetic_classification
from oracle_builder.datasets.schema import read_dataset_info, set_dataset_lifecycle
from oracle_builder.environment import write_environment
from oracle_builder.inference.bundle import InferenceBundle
from oracle_builder.posthoc.workflow import fit_existing_model_posthoc
from oracle_builder.saving.load_test import run_load_tests
from oracle_builder.saving.save_model import save_model_artifacts, write_load_test_report
from oracle_builder.training.train import build_and_compile_model


def test_existing_model_can_fit_separate_posthoc_head_on_new_classes(tmp_path):
    source = tmp_path / "source"
    config_file = tmp_path / "source.toml"
    config_file.write_text("[run]\ntask = 'classification'\nmodel = 'simple_cnn'\n")
    config = {
        "run": {"task": "classification", "model": "simple_cnn", "seed": 123, "run_id": str(uuid.uuid4())},
        "data": {"input_shape": [12, 12, 1], "num_classes": 2, "batch_size": 8,
                 "validation_split": 0.2, "test_split": 0.1},
        "model": {"base_filters": 4, "dropout": 0.0},
        "training": {"optimizer": "adam", "learning_rate": 0.001,
                     "loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"]},
        "output": {"export_savedmodel": False},
        "dataset": {"labels": [{"class_index": 0, "name": "old-a"}, {"class_index": 1, "name": "old-b"}]},
    }
    source_manifest = create_run_artifact(source, run_id=config["run"]["run_id"], name="source",
                                          config=config, source_config=config_file, artifact_type="model_product")
    model = build_and_compile_model(config)
    save_report = save_model_artifacts(model, source, config)
    write_load_test_report(source, run_load_tests(source, config, save_report))
    write_environment(source)
    (source / "metrics").mkdir(exist_ok=True)
    (source / "metrics" / "history.csv").write_text("epoch,loss,val_loss\n1,0.7,0.8\n")
    (source / "evaluation").mkdir(exist_ok=True)
    (source / "evaluation" / "evaluation_summary.json").write_text('{"accuracy": 0.01}\n')
    (source / "model" / "posthoc").mkdir(exist_ok=True)
    (source / "model" / "posthoc" / "report.json").write_text('{"metrics": {"accuracy": 0.02}}\n')
    update_run_artifact(source, status="complete")
    source_fingerprint = seal_run_artifact(source)["fingerprint_sha256"]

    dataset = tmp_path / "new.sqlite"
    create_synthetic_classification(dataset, n=45, shape=(12, 12, 1), classes=3)
    with sqlite3.connect(dataset) as connection:
        set_dataset_lifecycle(connection, "frozen")
        dataset_id = read_dataset_info(connection)["dataset_id"]

    product = tmp_path / "fitted"
    result = fit_existing_model_posthoc(source, dataset, product, split_strategy="class_stratified")
    assert result["artifact_id"] != source_manifest["artifact_id"]
    assert read_run_manifest(source)["fingerprint_sha256"] == source_fingerprint
    validation = validate_run_artifact(product)
    assert validation["valid"], validation
    report = json.loads((product / "model" / "posthoc" / "report.json").read_text())
    assert report["training_dataset"]["dataset_id"] == dataset_id
    assert len(report["class_names"]) == 3
    assert (product / "model" / "posthoc" / "estimator.joblib").is_file()
    assert (product / "model" / "weights.weights.h5").read_bytes() == (source / "model" / "weights.weights.h5").read_bytes()
    assert (product / "metrics" / "history.csv").read_text() == "epoch,loss,val_loss\n1,0.7,0.8\n"
    assert json.loads((product / "metrics" / "provenance.json").read_text())["source_model_artifact_id"] == source_manifest["artifact_id"]
    assert json.loads((product / "provenance" / "source_model" / "evaluation" / "evaluation_summary.json").read_text())["accuracy"] == 0.01
    assert json.loads((product / "provenance" / "source_model" / "posthoc" / "report.json").read_text())["metrics"]["accuracy"] == 0.02
    assert json.loads((product / "evaluation" / "evaluation_summary.json").read_text())["accuracy"] != 0.01
    assert result["summary"]["evaluation_head"] == "posthoc"
    bundle = InferenceBundle.load(product)
    assert bundle.posthoc_head is not None
