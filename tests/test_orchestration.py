from __future__ import annotations

import json
from pathlib import Path
import pytest

from oracle_builder.orchestration.service import Orchestrator


def _create_sealed_product(path, source_config, *, artifact_type="model_product", metrics=None, dataset_fingerprint=None, task="classification", detailed_evidence=False):
    from oracle_builder.artifacts import (
        RunLayout, create_run_artifact, create_unavailable_split_manifest,
        seal_run_artifact, update_run_artifact,
    )

    path = Path(path)
    run_id = "334a9730-d2e9-4802-8a8a-a7a660a3949d"
    config = {
        "run": {"run_id": run_id, "run_name": "external", "task": task, "model": "external"},
        "data": {"input_shape": [4], "num_classes": 2},
        "model": {"variant": "imported"},
        "training": {"loss": "sparse_categorical_crossentropy"},
        "preprocessing": {}, "evaluation": {},
        "dataset": {"dataset_id": "d4f550b2-9f8b-42eb-a3aa-6d6f20fc8aee" if dataset_fingerprint else None, "revision_id": None, "fingerprint_sha256": dataset_fingerprint},
        "paths": {"run_dir": str(path)},
    }
    create_run_artifact(path, run_id=run_id, name=path.name, config=config, source_config=source_config, artifact_type=artifact_type)
    layout = RunLayout(path)
    create_unavailable_split_manifest(path, config, reason="Imported product fixture")
    layout.environment.write_text("{}\n")
    layout.requirements.write_text("oracle-builder==0.1.0\n")
    layout.training_log.write_bytes(b"sqlite-placeholder")
    layout.metrics_json.write_text("{}\n")
    (layout.model / "model_manifest.json").write_text("{}\n")
    (layout.model / "load_test_report.json").write_text("{}\n")
    if metrics is not None:
        (layout.evaluation / "evaluation_summary.json").write_text(json.dumps({"task": task, **metrics}) + "\n")
    if detailed_evidence:
        (layout.evaluation / "confusion_matrix.json").write_text(json.dumps({
            "labels": [0, 1], "class_names": ["copepod", "other"], "matrix": [[8, 2], [1, 9]],
        }) + "\n")
        (layout.evaluation / "per_class_metrics.csv").write_text(
            "class_index,class_name,support,precision,recall,f1_score,average_precision\n"
            "0,copepod,10,0.88,0.8,0.84,0.9\n1,other,10,0.82,0.9,0.86,0.91\n"
        )
        (path / "figures").mkdir(exist_ok=True)
        (path / "figures" / "confusion_matrix.png").write_bytes(b"png-evidence")
    update_run_artifact(path, status="complete", summary={"evaluation": {"task": task, **metrics}} if metrics is not None else {"product": {"name": "external"}})
    return seal_run_artifact(path)


def test_dispatch_and_reconcile_keeps_orchestrator_job_identity(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model = workspace / "external.keras"
    model.write_bytes(b"placeholder")
    info = workspace / "external.toml"
    info.write_text("[product]\nname = 'External'\n")
    orchestrator = Orchestrator(tmp_path / "orchestrator.sqlite", workspace_root=workspace)
    endpoint = orchestrator.register_compute_endpoint(name="local", base_url="http://oracle-serve:8100")
    experiment = orchestrator.create_model_import(name="external", model_path=model, info_path=info)
    specification = orchestrator.specifications(experiment["experiment_id"])[0]
    requests = []

    def request(base, method, endpoint, body=None):
        requests.append((base, method, endpoint, body))
        if endpoint == "/health/ready":
            return {"status": "ready", "compute_enabled": True}
        if endpoint == "/compute/status":
            return {"status": "ready", "queue": {"depth": 0, "capacity": 10}, "workers": [{"worker_id": "gpu-a", "status": "idle", "capabilities": {"actions": ["model_ingest"], "gpus": []}}]}
        if method == "POST":
            return {"status": "queued", "worker_id": None}
        return {"status": "succeeded", "worker_id": "gpu-a", "finished_at": "2026-08-11T00:00:00Z", "error": None}

    monkeypatch.setattr(orchestrator, "_request", request)
    job = orchestrator.dispatch(specification["specification_id"], endpoint["endpoint_id"])
    submitted = next(item for item in requests if item[1] == "POST")
    assert submitted[3]["job_id"] == job["job_id"]
    assert job["status"] == "submitted"

    reconciled = orchestrator.reconcile_job(job["job_id"])
    assert reconciled["status"] == "artifact_invalid"
    assert reconciled["worker_id"] == "gpu-a"


def test_orchestrator_api_creates_a_model_import_specification(tmp_path):
    from fastapi.testclient import TestClient
    from oracle_builder.orchestration.api import create_app

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model = workspace / "external.keras"
    model.write_bytes(b"placeholder")
    info = workspace / "external.toml"
    info.write_text("[product]\nname = 'External'\n")
    with TestClient(create_app(Orchestrator(tmp_path / "orchestrator.sqlite", workspace_root=workspace))) as client:
        created = client.post("/v1/model-imports", json={
            "name": "api-import", "model_path": str(model), "info_path": str(info),
        })
        assert created.status_code == 201
        specifications = client.get("/v1/specifications").json()["specifications"]
        assert specifications[0]["status"] == "planned"
        assert specifications[0]["action"] == "model_ingest"


def test_successful_compute_is_validated_indexed_and_linked(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model = workspace / "external.keras"
    model.write_bytes(b"placeholder")
    info = workspace / "external.toml"
    info.write_text("[product]\nname = 'External'\n")
    orchestrator = Orchestrator(
        tmp_path / "orchestrator.sqlite",
        workspace_root=workspace,
        artifact_root=tmp_path / "artifacts",
    )
    endpoint = orchestrator.register_compute_endpoint(name="local", base_url="http://oracle-serve:8100")
    experiment = orchestrator.create_model_import(name="external", model_path=model, info_path=info)
    specification = orchestrator.specifications(experiment["experiment_id"])[0]
    manifest = _create_sealed_product(specification["parameters"]["output"], info)

    def request(base, method, endpoint, body=None):
        if endpoint == "/health/ready":
            return {"status": "ready", "compute_enabled": True}
        if endpoint == "/compute/status":
            return {"status": "ready", "queue": {"depth": 0, "capacity": 10}, "workers": [{"worker_id": "gpu-a", "status": "idle", "capabilities": {"actions": ["model_ingest"], "gpus": []}}]}
        if method == "POST":
            return {"status": "queued", "worker_id": None}
        if endpoint.endswith("/events?after=0"):
            return {"events": [{
                "sequence": 1, "timestamp": "2026-08-11T00:00:00Z",
                "type": "completed", "message": "Job completed successfully",
                "data": {"output_path": specification["parameters"]["output"]},
            }]}
        return {
            "status": "succeeded", "worker_id": "gpu-a",
            "started_at": "2026-08-10T23:59:00Z",
            "finished_at": "2026-08-11T00:00:00Z", "error": None,
            "result": {"output_path": specification["parameters"]["output"], "exit_code": 0},
        }

    monkeypatch.setattr(orchestrator, "_request", request)
    job = orchestrator.dispatch(specification["specification_id"], endpoint["endpoint_id"])
    completed = orchestrator.reconcile_job(job["job_id"])

    assert completed["status"] == "indexed"
    assert completed["validation_status"] == "valid"
    assert completed["validation_report"]["artifact_id"] == manifest["artifact_id"]
    assert completed["artifact_id"] == manifest["artifact_id"]
    assert completed["artifact_path"] == specification["parameters"]["output"]
    assert orchestrator.specification(specification["specification_id"])["status"] == "indexed"
    assert [event["event_type"] for event in orchestrator.job_events(job["job_id"])] == [
        "completed", "validating", "artifact_indexed",
    ]


def test_dispatch_preflight_explains_missing_gpu_capacity(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model = workspace / "external.keras"
    model.write_bytes(b"placeholder")
    info = workspace / "external.toml"
    info.write_text("[product]\nname = 'External'\n")
    orchestrator = Orchestrator(tmp_path / "orchestrator.sqlite", workspace_root=workspace)
    endpoint = orchestrator.register_compute_endpoint(name="cpu", base_url="http://compute:8100")
    experiment = orchestrator.create_model_import(name="external", model_path=model, info_path=info, resources={"gpu_count": 1})
    specification = orchestrator.specifications(experiment["experiment_id"])[0]

    def request(base, method, path, body=None):
        if path == "/health/ready":
            return {"status": "ready", "compute_enabled": True}
        return {"status": "ready", "queue": {"depth": 0, "capacity": 4}, "workers": [{"worker_id": "cpu", "status": "idle", "capabilities": {"actions": ["model_ingest"], "gpus": []}}]}

    monkeypatch.setattr(orchestrator, "_request", request)
    report = orchestrator.preflight(specification["specification_id"], endpoint["endpoint_id"])
    assert not report["ready"]
    assert "requests 1 GPU" in report["reasons"][0]


def test_file_explorer_stays_inside_allow_listed_root(tmp_path):
    workspace = tmp_path / "workspace"
    configs = workspace / "configs"
    configs.mkdir(parents=True)
    (configs / "baseline.toml").write_text("[run]\n")
    orchestrator = Orchestrator(tmp_path / "orchestrator.sqlite", browse_roots=[workspace])

    listing = orchestrator.files("root-1", "configs")
    assert listing["path"] == "configs"
    assert listing["entries"] == [{"name": "baseline.toml", "path": "configs/baseline.toml", "kind": "file", "size_bytes": 6}]
    try:
        orchestrator.files("root-1", "../")
    except ValueError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("Explorer accepted a path outside its root")

    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        orchestrator.scan(outside)
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("Catalog scan accepted a root outside its allow list")

    invalid = workspace / "invalid-run"
    invalid.mkdir()
    (invalid / "artifact.json").write_text('{"artifact_schema":{"name":"oracle_builder_model_run","version":"1.0.0"},"artifact_id":"334a9730-d2e9-4802-8a8a-a7a660a3949d"}')
    report = orchestrator.scan(workspace)
    assert not report["artifacts"]
    assert "validation failed" in report["skipped"][0]["reason"]


def test_upload_api_stages_allowed_files_in_owned_artifact_root(tmp_path):
    from fastapi.testclient import TestClient
    from oracle_builder.orchestration.api import create_app

    artifact_root = tmp_path / "artifacts"
    with TestClient(create_app(Orchestrator(tmp_path / "orchestrator.sqlite", artifact_root=artifact_root))) as client:
        response = client.post("/v1/uploads/configs/baseline.toml", content=b"[run]\n")
        assert response.status_code == 201
        uploaded = response.json()
        assert uploaded["size_bytes"] == 6
        assert (artifact_root / "uploads" / "configs" / "baseline.toml").read_bytes() == b"[run]\n"
        assert client.post("/v1/uploads/configs/baseline.toml", content=b"[run]\n").status_code == 409
        assert client.post("/v1/uploads/configs/not-a-config.txt", content=b"x").status_code == 422


def test_typed_training_experiment_owns_dataset_path_and_materializes_seed_configs(tmp_path):
    import sqlite3
    from oracle_data_contracts.datasets.schema import initialize_database

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    dataset_path = workspace / "frozen.sqlite"
    with sqlite3.connect(dataset_path) as connection:
        initialize_database(connection, "classification")
        connection.execute("UPDATE dataset SET lifecycle='frozen', frozen_at='now'")
        connection.commit()
    config_path = workspace / "baseline.toml"
    config_path.write_text("""[run]
task = "classification"
model = "simple_cnn"

[data]
input_shape = [32, 32, 1]
""")
    orchestrator = Orchestrator(tmp_path / "orchestrator.sqlite", artifact_root=tmp_path / "artifacts", workspace_root=workspace)
    dataset = orchestrator.ingest_dataset(dataset_path)
    recipe = orchestrator.create_recipe(name="baseline", config_path=config_path)

    experiment = orchestrator.create_training_experiment(
        name="seed study", dataset_id=dataset["dataset_id"], recipe_ids=[recipe["recipe_id"]], seeds=[11, 12]
    )
    specifications = orchestrator.specifications(experiment["experiment_id"])
    assert [item["parameters"]["seed"] for item in specifications] == [11, 12]
    assert all(item["parameters"]["input"] == str(dataset_path) for item in specifications)
    assert all("runs_dir" in item["parameters"] for item in specifications)


def test_experiment_results_and_persisted_comparison_use_artifact_metrics(tmp_path):
    import sqlite3
    from oracle_data_contracts.datasets.schema import initialize_database

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    dataset_path = workspace / "frozen.sqlite"
    with sqlite3.connect(dataset_path) as connection:
        initialize_database(connection, "classification")
        connection.execute("UPDATE dataset SET lifecycle='frozen', frozen_at='now'")
        connection.commit()
    config_path = workspace / "baseline.toml"
    config_path.write_text('[run]\ntask="classification"\nmodel="simple_cnn"\n[data]\ninput_shape=[4]\n')
    orchestrator = Orchestrator(tmp_path / "orchestrator.sqlite", workspace_root=workspace, artifact_root=tmp_path / "artifacts")
    dataset = orchestrator.ingest_dataset(dataset_path)
    recipe = orchestrator.create_recipe(name="baseline", config_path=config_path)
    experiment = orchestrator.create_training_experiment(name="comparison study", dataset_id=dataset["dataset_id"], recipe_ids=[recipe["recipe_id"]], seeds=[1, 2])
    specifications = orchestrator.specifications(experiment["experiment_id"])
    artifact_ids = []
    for index, specification in enumerate(specifications):
        path = specification["parameters"]["runs_dir"] + "/" + specification["parameters"]["output"]
        manifest = _create_sealed_product(
            Path(path), specification["parameters"]["config"],
            artifact_type="model_run", dataset_fingerprint=dataset["fingerprint_sha256"],
            metrics={"accuracy": 0.8 + index * 0.1, "balanced_accuracy": 0.75 + index * 0.1, "macro_f1": 0.7 + index * 0.1},
            detailed_evidence=index == 0,
        )
        orchestrator.scan(path)
        artifact_ids.append(manifest["artifact_id"])
        with orchestrator._connection() as db:
            db.execute("UPDATE run_specifications SET artifact_id=?, status='indexed' WHERE specification_id=?", (manifest["artifact_id"], specification["specification_id"]))

    results = orchestrator.experiment_results(experiment["experiment_id"])
    assert results["summary"]["indexed"] == 2
    assert results["comparison"]["compatible"]
    assert results["candidates"][1]["artifact"]["primary_metrics"]["accuracy"] == 0.9

    evidence = orchestrator.artifact_evidence(artifact_ids[0])
    assert evidence["classification"]["confusion_matrix"]["matrix"] == [[8, 2], [1, 9]]
    assert evidence["classification"]["per_class_metrics"][0]["class_name"] == "copepod"
    assert evidence["availability"]["confusion_matrix"]
    assert evidence["media"][0]["kind"] == "figure"

    comparison = orchestrator.create_comparison(name="seed comparison", artifact_ids=artifact_ids)
    assert comparison["selection"]["artifact_ids"] == artifact_ids
    assert "accuracy" in comparison["protocol"]["common_metrics"]

    incompatible = _create_sealed_product(
        tmp_path / "artifacts" / "runs" / "other-dataset", config_path,
        artifact_type="model_run", dataset_fingerprint="b" * 64,
        metrics={"accuracy": 0.95, "balanced_accuracy": 0.9, "macro_f1": 0.9},
    )
    orchestrator.scan(tmp_path / "artifacts" / "runs" / "other-dataset")
    with pytest.raises(ValueError, match="different dataset revision"):
        orchestrator.create_comparison(name="invalid", artifact_ids=[artifact_ids[0], incompatible["artifact_id"]])

    from fastapi.testclient import TestClient
    from oracle_builder.orchestration.api import create_app
    with TestClient(create_app(orchestrator)) as client:
        assert client.get(f"/v1/experiments/{experiment['experiment_id']}/results").json()["summary"]["indexed"] == 2
        detail = client.get(f"/v1/artifacts/{artifact_ids[0]}/evidence")
        assert detail.status_code == 200
        media = client.get(f"/v1/artifacts/{artifact_ids[0]}/evidence/files/figures/confusion_matrix.png")
        assert media.status_code == 200
        assert media.content == b"png-evidence"
        assert client.get(f"/v1/artifacts/{artifact_ids[0]}/evidence/files/../artifact.json").status_code in {404, 422}
        saved = client.post("/v1/comparisons", json={"name": "API comparison", "artifact_ids": artifact_ids})
        assert saved.status_code == 201
        assert len(client.get("/v1/comparisons").json()["comparisons"]) == 2
