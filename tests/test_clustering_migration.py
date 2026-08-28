from __future__ import annotations

import json

import numpy as np
import pytest

from oracle_builder.artifacts import RunLayout, create_run_artifact, seal_run_artifact, update_run_artifact
from oracle_builder.artifacts import create_unavailable_split_manifest
from oracle_builder.artifacts.cli import main as artifact_main
from oracle_builder.clustering.evidence import ClusterEvidenceIndex
from oracle_builder.clustering.migration import migrate_sealed_clustering_package


def _sealed_legacy_cluster_run(tmp_path):
    source_config = tmp_path / "source.toml"
    source_config.write_text('[run]\ntask = "clustering"\n', encoding="utf-8")
    run = tmp_path / "legacy-clusters"
    config = {
        "run": {"run_id": "334a9730-d2e9-4802-8a8a-a7a660a3949d", "run_name": run.name, "task": "clustering", "model": "simple_cnn"},
        "data": {"input_shape": [8, 8, 1], "num_classes": 2},
        "model": {"embedding_dim": 2},
        "dataset": {"labels": []},
        "clustering": {"top_k": 2, "structure": {"cluster_count": 2}},
    }
    create_run_artifact(run, run_id=config["run"]["run_id"], name=run.name, config=config, source_config=source_config)
    create_unavailable_split_manifest(run, config, reason="legacy clustering fixture")
    layout = RunLayout(run)
    layout.environment.write_text("{}\n", encoding="utf-8")
    layout.requirements.write_text("oracle-builder==0.1.0\n", encoding="utf-8")
    layout.training_log.write_bytes(b"sqlite-placeholder")
    layout.metrics_json.write_text("{}\n", encoding="utf-8")
    layout.metrics_jsonl.write_text("\n", encoding="utf-8")
    (layout.model / "final.keras").write_bytes(b"model")
    (layout.model / "load_test_report.json").write_text("{}\n", encoding="utf-8")
    (layout.model / "model_manifest.json").write_text(json.dumps({
        "artifact_id": "old", "run_id": config["run"]["run_id"], "task": "clustering", "architecture": "simple_cnn",
        "input": {"shape": [None, 8, 8, 1]}, "outputs": {"cluster_evidence": True}, "inference_contract": {"version": "1.0.0"},
    }) + "\n", encoding="utf-8")
    ClusterEvidenceIndex.fit(
        np.asarray([[1, 0], [.9, .1], [0, 1], [.1, .9]], dtype="float32"),
        ["a", "b", "c", "d"], n_clusters=2,
    ).save(layout.model / "clustering_evidence")
    update_run_artifact(run, status="complete")
    seal_run_artifact(run)
    return run


def test_migrates_sealed_legacy_cluster_package_without_mutating_source(tmp_path, monkeypatch):
    source = _sealed_legacy_cluster_run(tmp_path)
    product = tmp_path / "cluster-product"
    source_manifest = (source / "artifact.json").read_bytes()

    def fake_environment(path):
        layout = RunLayout(path)
        layout.environment.write_text("{}\n", encoding="utf-8")
        layout.requirements.write_text("oracle-builder==0.1.0\n", encoding="utf-8")
        return {}

    monkeypatch.setattr("oracle_builder.artifacts.deployment.write_environment", fake_environment)
    result = migrate_sealed_clustering_package(source, product)

    assert result["migration"]["record_type"] == "downstream_cluster_definition"
    assert result["migration"]["cluster_count"] == 2
    assert (source / "artifact.json").read_bytes() == source_manifest
    assert (product / "evidence" / "metadata.json").exists()
    manifest = json.loads((product / "cluster_manifest.json").read_text(encoding="utf-8"))
    assert manifest["record_type"] == "downstream_cluster_definition"
    assert manifest["source_model"]["artifact_id"] == json.loads(
        (source / "artifact.json").read_text(encoding="utf-8")
    )["artifact_id"]


def test_migration_rejects_non_clustering_source(tmp_path):
    source = _sealed_legacy_cluster_run(tmp_path)
    config_path = source / "config" / "resolved.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["run"]["task"] = "classification"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="not a legacy clustering package"):
        migrate_sealed_clustering_package(source, tmp_path / "product")


def test_artifact_cli_dispatches_clustering_migration(monkeypatch, capsys):
    monkeypatch.setattr(
        "oracle_builder.clustering.migration.migrate_sealed_clustering_package",
        lambda source, output: {"output": output, "source": source},
    )

    assert artifact_main(["migrate-clustering", "legacy", "product"]) == 0
    assert json.loads(capsys.readouterr().out) == {"output": "product", "source": "legacy"}
