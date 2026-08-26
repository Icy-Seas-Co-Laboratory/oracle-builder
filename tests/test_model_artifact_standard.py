from __future__ import annotations

import json
from pathlib import Path

from oracle_builder.artifacts.deployment import publish_deployment_asset
from oracle_builder.artifacts.training import materialize_training_record
from oracle_builder.artifacts import (
    RunLayout,
    create_run_artifact,
    seal_run_artifact,
    update_run_artifact,
    validate_run_artifact,
)
from oracle_builder.training.logging_callbacks import (
    append_jsonl_event,
    write_history_jsonl,
)


def _config() -> dict:
    return {
        "run": {
            "run_id": "334a9730-d2e9-4802-8a8a-a7a660a3949d",
            "run_name": "training",
            "task": "classification",
            "model": "simple_cnn",
        },
        "data": {"input_shape": [8, 8, 1], "num_classes": 2},
        "model": {"embedding_dim": 4},
        "dataset": {
            "dataset_id": "d4f550b2-9f8b-42eb-a3aa-6d6f20fc8aee",
            "dataset_type": "classification",
            "fingerprint_sha256": "a" * 64,
        },
        "paths": {"input_path": "/tmp/input.sqlite"},
    }


def _complete_training_record(tmp_path: Path) -> Path:
    source_config = tmp_path / "source.toml"
    source_config.write_text('[run]\ntask = "classification"\n', encoding="utf-8")
    run = tmp_path / "training"
    config = _config()
    create_run_artifact(run, run_id=config["run"]["run_id"], name="training", config=config, source_config=source_config)
    layout = RunLayout(run)
    from oracle_builder.artifacts import create_unavailable_split_manifest

    create_unavailable_split_manifest(run, config, reason="fixture")
    layout.environment.write_text("{}\n", encoding="utf-8")
    layout.requirements.write_text("oracle-builder==0.1.0\n", encoding="utf-8")
    layout.training_log.write_bytes(b"sqlite-placeholder")
    layout.metrics_json.write_text("{}\n", encoding="utf-8")
    layout.metrics_jsonl.write_text("\n", encoding="utf-8")
    (layout.model / "final.keras").write_bytes(b"model")
    (layout.model / "load_test_report.json").write_text("{}\n", encoding="utf-8")
    (layout.model / "model_manifest.json").write_text(
        json.dumps({
            "schema_name": "oracle_builder_inference_bundle",
            "schema_version": "1.0.0",
            "model_asset_id": "00000000-0000-0000-0000-000000000002",
            "artifact_id": "old",
            "run_id": config["run"]["run_id"],
            "task": "classification",
            "architecture": "simple_cnn",
            "input": {"shape": [None, 8, 8, 1]},
            "outputs": {"logits": True, "probabilities": True},
            "inference_contract": {"version": "1.0.0"},
        })
        + "\n",
        encoding="utf-8",
    )
    update_run_artifact(run, status="complete")
    seal_run_artifact(run)
    return run


def test_training_and_deployment_profiles_are_distinct(tmp_path, monkeypatch):
    training = _complete_training_record(tmp_path)
    deployment = tmp_path / "deployment"

    def fake_environment(path):
        layout = RunLayout(path)
        layout.environment.write_text("{}\n", encoding="utf-8")
        layout.requirements.write_text("oracle-builder==0.1.0\n", encoding="utf-8")
        return {}

    monkeypatch.setattr("oracle_builder.artifacts.deployment.write_environment", fake_environment)
    result = publish_deployment_asset(training, deployment, include_weights=False)

    manifest = json.loads((deployment / "artifact.json").read_text(encoding="utf-8"))
    assert result["artifact_id"] == manifest["artifact_id"]
    assert manifest["standard"]["profile"] == "deployment_asset"
    assert manifest["lineage"]["training_record_id"]
    assert (deployment / "model" / "contract.json").exists()
    assert (deployment / "logs" / "events.jsonl").exists()
    assert not (deployment / "logs" / "training.sqlite").exists()
    assert not (deployment / "protocol" / "splits.json").exists()
    assert validate_run_artifact(deployment)["valid"]
    assert validate_run_artifact(training)["valid"]


def test_jsonl_events_and_metrics_are_self_describing(tmp_path):
    events = tmp_path / "events.jsonl"
    metrics = tmp_path / "metrics.jsonl"
    append_jsonl_event(events, "run-1", "INFO", "epoch_complete", {"epoch": 1})
    write_history_jsonl({"loss": [0.5], "val_loss": [0.6]}, metrics, run_id="run-1")

    event = json.loads(events.read_text(encoding="utf-8").splitlines()[0])
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
    assert event["schema"] == "oracle_training_event"
    assert event["event"] == "epoch_complete"
    assert {row["schema"] for row in rows} == {"oracle_training_metric"}
    assert {row["split"] for row in rows} == {"train", "validation"}


def test_training_record_materialization_embeds_retraining_library(tmp_path):
    training = _complete_training_record(tmp_path)
    dataset = tmp_path / "dataset.sqlite"
    dataset.write_bytes(b"dataset")
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "train.py").write_text("print('train')\n", encoding="utf-8")
    output = tmp_path / "portable-training"

    result = materialize_training_record(
        training,
        output,
        dataset=dataset,
        source_root=source_root,
    )

    library = json.loads((output / "library" / "manifest.json").read_text())
    assert result["library_entries"] == 2
    assert len(library["entries"]) == 2
    assert (output / "library" / "dataset" / "dataset.sqlite").exists()
    assert (output / "library" / "source" / "train.py").exists()
    assert validate_run_artifact(output)["valid"]
    assert validate_run_artifact(training)["valid"]
