from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest

import model_training
from oracle_builder.artifacts import (
    RunLayout,
    create_run_artifact,
    create_unavailable_split_manifest,
    migrate_legacy_run,
    pack_run_artifact,
    read_run_config,
    reopen_run_artifact,
    seal_run_artifact,
    unpack_run_artifact,
    update_run_artifact,
    validate_run_artifact,
)
from oracle_builder.data.sqlite_dataset import create_synthetic_classification
from oracle_builder.datasets.schema import set_dataset_lifecycle


def example_config(tmp_path: Path) -> dict:
    return {
        "run": {
            "run_id": "334a9730-d2e9-4802-8a8a-a7a660a3949d",
            "run_name": "example",
            "task": "classification",
            "model": "resnet",
        },
        "data": {"input_shape": [32, 32, 1], "num_classes": 3},
        "model": {"variant": "resnet18", "embedding_dim": 64},
        "training": {"loss": "sparse_categorical_crossentropy"},
        "preprocessing": {"resize_mode": "fit_pad", "rescale": True},
        "evaluation": {},
        "dataset": {
            "dataset_id": "d4f550b2-9f8b-42eb-a3aa-6d6f20fc8aee",
            "dataset_type": "classification",
            "schema_name": "oracle_builder_dataset",
            "schema_version": "1.1.0",
            "version": "1",
            "lifecycle": "frozen",
            "fingerprint_sha256": "a" * 64,
        },
        "paths": {
            "input_path": str((tmp_path / "source.sqlite").resolve()),
            "run_dir": str((tmp_path / "run").resolve()),
        },
    }


def create_example_run(tmp_path: Path) -> tuple[Path, dict]:
    run = tmp_path / "run"
    source_config = tmp_path / "source.toml"
    source_config.write_text(
        '[run]\ntask = "classification"\nmodel = "resnet"\n',
        encoding="utf-8",
    )
    config = example_config(tmp_path)
    create_run_artifact(
        run,
        run_id=config["run"]["run_id"],
        name="example",
        config=config,
        source_config=source_config,
    )
    return run, config


def add_completed_run_files(run: Path) -> None:
    layout = RunLayout(run)
    create_unavailable_split_manifest(
        run,
        read_run_config(run),
        reason="Synthetic completed-run fixture has no backing dataset.",
    )
    layout.environment.write_text("{}\n", encoding="utf-8")
    layout.requirements.write_text("oracle-builder==0.1.0\n", encoding="utf-8")
    layout.training_log.write_bytes(b"sqlite-placeholder")
    layout.metrics_json.write_text('{"loss": [1.0]}\n', encoding="utf-8")
    (layout.model / "model_manifest.json").write_text("{}\n", encoding="utf-8")
    (layout.model / "load_test_report.json").write_text("{}\n", encoding="utf-8")


def test_run_artifact_separates_portable_config_from_runtime_paths(tmp_path):
    run, config = create_example_run(tmp_path)
    layout = RunLayout(run)

    portable = read_run_config(run)
    runtime = json.loads(layout.runtime.read_text())

    assert "paths" not in portable
    assert runtime["paths"] == config["paths"]
    assert portable["dataset"]["fingerprint_sha256"] == "a" * 64
    assert layout.source_config.exists()
    assert layout.model_card.exists()
    assert validate_run_artifact(run)["warnings"] == [
        "Run artifact is working and has not been sealed"
    ]


def test_seal_detects_changes_and_explicit_reopen_allows_reseal(tmp_path):
    run, _ = create_example_run(tmp_path)
    layout = RunLayout(run)
    add_completed_run_files(run)
    update_run_artifact(run, status="complete", summary={"accuracy": 0.9})

    sealed = seal_run_artifact(run)

    assert sealed["lifecycle"] == "sealed"
    assert len(sealed["fingerprint_sha256"]) == 64
    assert validate_run_artifact(run)["valid"]
    layout.metrics_json.write_text('{"loss": [0.5]}\n', encoding="utf-8")
    report = validate_run_artifact(run)
    assert not report["valid"]
    assert "Inventory mismatch: metrics/history.json" in report["errors"]

    reopened = reopen_run_artifact(run, reason="correct metrics")
    assert reopened["lifecycle"] == "working"
    assert not layout.checksums.exists()
    resealed = seal_run_artifact(run)
    assert resealed["fingerprint_sha256"] != sealed["fingerprint_sha256"]
    assert validate_run_artifact(run)["valid"]


def test_pack_is_deterministic_and_unpack_validates(tmp_path):
    run, _ = create_example_run(tmp_path)
    add_completed_run_files(run)
    update_run_artifact(run, status="complete")
    seal_run_artifact(run)
    first = tmp_path / "run-one.oracle-run.zip"
    second = tmp_path / "run-two.oracle-run.zip"

    first_report = pack_run_artifact(run, first)
    second_report = pack_run_artifact(run, second)
    restored = tmp_path / "restored"
    unpacked = unpack_run_artifact(first, restored)

    assert first.read_bytes() == second.read_bytes()
    assert first_report["package_sha256"] == second_report["package_sha256"]
    assert unpacked["valid"]
    assert validate_run_artifact(restored)["valid"]


def test_running_artifact_cannot_be_sealed(tmp_path):
    run, _ = create_example_run(tmp_path)

    with pytest.raises(ValueError, match="status is still running"):
        seal_run_artifact(run)


def test_legacy_run_migration_is_non_destructive(tmp_path):
    source = tmp_path / "legacy"
    output = tmp_path / "v1"
    (source / "model").mkdir(parents=True)
    (source / "run_config.toml").write_text(
        '[run]\ntask = "classification"\nmodel = "simple_cnn"\n'
    )
    config = example_config(tmp_path)
    (source / "resolved_config.json").write_text(json.dumps(config))
    (source / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": config["run"]["run_id"],
                "run_name": "legacy",
                "status": "complete",
                "evaluation_summary": {"accuracy": 0.8},
            }
        )
    )
    (source / "environment.json").write_text("{}\n")
    (source / "requirements_freeze.txt").write_text("keras\n")
    (source / "training_log.sqlite").write_bytes(b"log")
    (source / "metrics.json").write_text('{"loss": [1.0]}\n')
    (source / "metrics.csv").write_text("epoch,loss\n0,1.0\n")
    (source / "model" / "final.keras").write_bytes(b"model")
    (source / "model" / "load_test_report.json").write_text("{}\n")
    (source / "mask_comparison.png").write_bytes(b"figure")

    result = migrate_legacy_run(source, output)

    assert result["valid"]
    assert result["lifecycle"] == "sealed"
    assert (output / "attachments" / "mask_comparison.png").exists()
    assert (output / "model" / "model_manifest.json").exists()
    assert (source / "resolved_config.json").exists()


def test_training_writes_and_seals_v1_run_artifact(monkeypatch, tmp_path):
    pytest.importorskip("tensorflow")
    database = tmp_path / "training.sqlite"
    config_path = tmp_path / "config.toml"
    runs = tmp_path / "runs"
    create_synthetic_classification(
        database, n=10, shape=(8, 8, 1), classes=2
    )
    with sqlite3.connect(database) as connection:
        set_dataset_lifecycle(connection, "frozen")
        connection.commit()
    config_path.write_text(
        """
[run]
task = "classification"
model = "simple_cnn"
seed = 123

[data]
input_shape = [8, 8, 1]
batch_size = 2

[data.streaming]
enabled = false

[model]
base_filters = 2
dropout = 0.0
embedding_dim = 4

[training]
epochs = 1
loss = "sparse_categorical_crossentropy"
metrics = ["accuracy"]

[augmentation]
enabled = false

[evidence]
enabled = false

[output]
export_savedmodel = false
save_predictions = false
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "model_training.py",
            "--config",
            str(config_path),
            "--input",
            str(database),
            "--output",
            "portable-run",
            "--runs-dir",
            str(runs),
        ],
    )
    reports = types.ModuleType("oracle_builder.evaluation.reports")
    reports.evaluate_run_model = lambda *_args, **_kwargs: {
        "summary": {"accuracy": 1.0}
    }
    monkeypatch.setitem(sys.modules, "oracle_builder.evaluation.reports", reports)

    assert model_training.main() == 0

    run = runs / "portable-run"
    layout = RunLayout(run)
    report = validate_run_artifact(run)
    manifest = json.loads(layout.manifest.read_text())
    portable_config = json.loads(layout.resolved_config.read_text())
    split_manifest = json.loads(layout.split_manifest.read_text())
    assert report["valid"]
    assert report["lifecycle"] == "sealed"
    assert report["status"] == "complete"
    assert manifest["dataset"]["dataset_id"]
    assert manifest["dataset"]["fingerprint_sha256"]
    assert manifest["model"]["outputs"]["embedding_dimension"] == 4
    assert portable_config["artifact"]["artifact_id"] == manifest["artifact_id"]
    assert "_split_manifest" not in portable_config
    assert split_manifest["dataset"]["dataset_id"] == manifest["dataset"]["dataset_id"]
    assert split_manifest["dataset"]["fingerprint_sha256"] == manifest["dataset"]["fingerprint_sha256"]
    assert sum(split_manifest["counts"].values()) == 10
    assert len(split_manifest["assignments"]) == 10
    assert layout.training_log.exists()
    assert layout.metrics_json.exists()
    assert (layout.model / "model_manifest.json").exists()
    assert not (run / "resolved_config.json").exists()
