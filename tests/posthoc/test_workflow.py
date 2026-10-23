from __future__ import annotations

import json
import sqlite3
import uuid

from oracle_builder.artifacts import create_run_artifact, read_run_manifest, seal_run_artifact, update_run_artifact, validate_run_artifact
from oracle_builder.data.sqlite_dataset import create_synthetic_classification
from oracle_builder.datasets.schema import read_dataset_info, set_dataset_lifecycle
from oracle_builder.environment import write_environment
from oracle_builder.inference.bundle import InferenceBundle
from oracle_builder.orchestration.service import Orchestrator
from oracle_builder.posthoc.workflow import fit_existing_model_posthoc
from oracle_builder.saving.load_test import run_load_tests
from oracle_builder.saving.save_model import save_model_artifacts, write_load_test_report
from oracle_builder.training.train import build_and_compile_model


def test_existing_model_can_fit_separate_posthoc_head_on_new_classes(tmp_path, monkeypatch):
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
        "dataset": {"labels": [{"class_index": 0, "name": "0"}, {"class_index": 1, "name": "1"}]},
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
    events = []
    result = fit_existing_model_posthoc(
        source, dataset, product, split_strategy="class_stratified",
        progress=lambda event, message, data: events.append((event, message, data)),
    )
    extraction = [data for event, _, data in events if event == "posthoc_embedding_progress"]
    assert {data["split"] for data in extraction} == {"train", "validation", "test"}
    assert all(data["completed"] <= data["total"] for data in extraction)
    assert any(event == "posthoc_candidate_completed" for event, _, _ in events)
    assert result["artifact_id"] != source_manifest["artifact_id"]
    assert read_run_manifest(source)["fingerprint_sha256"] == source_fingerprint
    validation = validate_run_artifact(product)
    assert validation["valid"], validation
    report = json.loads((product / "model" / "posthoc" / "report.json").read_text())
    assert report["training_dataset"]["dataset_id"] == dataset_id
    assert len(report["class_names"]) == 3
    assert report["neural_baseline"]["status"] == "incompatible"
    assert "vocabulary" in report["neural_baseline"]["reason"]
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
    orchestrator = Orchestrator(tmp_path / "orchestrator.sqlite", workspace_root=tmp_path, artifact_root=tmp_path)
    orchestrator.scan(source)
    orchestrator.scan(product)
    source_detail = orchestrator.artifact_detail(source_manifest["artifact_id"])
    derived_detail = orchestrator.artifact_detail(result["artifact_id"])
    assert source_detail["artifact"]["evaluation_head"] == "neural"
    assert source_detail["artifact"]["metrics"]["accuracy"] == 0.01
    assert derived_detail["artifact"]["evaluation_head"] == "posthoc"
    assert derived_detail["artifact"]["metrics"]["accuracy"] == report["metrics"]["accuracy"]
    assert derived_detail["source_neural_evaluation"]["accuracy"] == 0.01
    assert derived_detail["neural_evaluation"]["accuracy"] == 0.01
    assert derived_detail["neural_artifact_id"] == source_manifest["artifact_id"]
    assert derived_detail["posthoc"]["metrics"]["accuracy"] == report["metrics"]["accuracy"]
    catalog = {row["artifact_id"]: row for row in orchestrator.artifact_catalog_query()["artifacts"]}
    assert catalog[source_manifest["artifact_id"]]["evaluation_head"] == "neural"
    assert catalog[result["artifact_id"]]["evaluation_head"] == "posthoc"
    assert catalog[result["artifact_id"]]["loss"] is None
    with sqlite3.connect(tmp_path / "orchestrator.sqlite") as connection:
        stored = json.loads(connection.execute(
            "SELECT facts_json FROM artifact_facts WHERE artifact_id=?", (result["artifact_id"],)
        ).fetchone()[0])
        stored["metrics"]["loss"] = 0.8
        stored["loss"] = 0.8
        stored["field_sources"]["loss"] = {"source": "training_history", "status": "inferred"}
        connection.execute(
            "UPDATE artifact_facts SET facts_json=?,loss=? WHERE artifact_id=?",
            (json.dumps(stored), 0.8, result["artifact_id"]),
        )
    assert orchestrator.artifact_detail(result["artifact_id"])["facts"]["facts"]["loss"] is None
    assert next(row for row in orchestrator.artifact_catalog_query()["artifacts"]
                if row["artifact_id"] == result["artifact_id"])["loss"] is None

    compatible_dataset = tmp_path / "compatible.sqlite"
    create_synthetic_classification(compatible_dataset, n=30, shape=(12, 12, 1), classes=2)
    with sqlite3.connect(compatible_dataset) as connection:
        set_dataset_lifecycle(connection, "frozen")
    compatible_product = tmp_path / "compatible-fit"
    shared_embeddings = tmp_path / "shared-embeddings"
    fit_existing_model_posthoc(source, compatible_dataset, compatible_product, split_strategy="class_stratified",
                               embedding_cache_root=shared_embeddings)
    compatible_report = json.loads((compatible_product / "model" / "posthoc" / "report.json").read_text())
    assert compatible_report["neural_baseline"]["status"] == "complete"
    assert compatible_report["neural_baseline"]["evaluation_signature"] == compatible_report["evaluation_signature"]
    assert (compatible_product / "model" / "posthoc" / "neural_baseline" / "evaluation" / "evaluation_summary.json").is_file()
    # Same source, frozen dataset, and exact deterministic assignments can
    # reuse only raw representations even when the estimator changes.
    monkeypatch.setattr("oracle_builder.posthoc.training.write_representation_artifacts",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("embeddings were recomputed")))
    cached_product = tmp_path / "cached-fit"
    fit_existing_model_posthoc(source, compatible_dataset, cached_product, split_strategy="class_stratified",
                               posthoc_settings={"estimator": "gradient_boosting"}, embedding_cache_root=shared_embeddings)
    assert json.loads((cached_product / "model" / "posthoc" / "report.json").read_text())["estimator"] == "gradient_boosting"
    monkeypatch.undo()

    second_product = tmp_path / "second-fit"
    second = fit_existing_model_posthoc(product, dataset, second_product, split_strategy="class_stratified")
    orchestrator.scan(second_product)
    second_detail = orchestrator.artifact_detail(second["artifact_id"])
    assert second_detail["neural_artifact_id"] == source_manifest["artifact_id"]
    assert second_detail["neural_evaluation"]["accuracy"] == 0.01
    assert second_detail["source_model_evaluation"]["accuracy"] == report["metrics"]["accuracy"]
    assert second_detail["artifact"]["evaluation_head"] == "posthoc"
    assert (second_product / "provenance" / "neural" / "evaluation" / "evaluation_summary.json").is_file()
    portable_catalog = Orchestrator(tmp_path / "portable.sqlite", workspace_root=tmp_path, artifact_root=tmp_path)
    portable_catalog.scan(second_product)
    portable_detail = portable_catalog.artifact_detail(second["artifact_id"])
    assert portable_detail["neural_evaluation"]["accuracy"] == 0.01
    assert portable_detail["neural_artifact_id"] == source_manifest["artifact_id"]
