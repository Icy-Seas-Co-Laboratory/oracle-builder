from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from oracle_builder.orchestration.api import create_app
from oracle_builder.orchestration.service import Orchestrator


def _orchestrator(tmp_path: Path) -> Orchestrator:
    return Orchestrator(
        tmp_path / "control.sqlite", artifact_root=tmp_path / "artifacts",
        workspace_root=tmp_path, runs_root=tmp_path / "runs",
    )


def test_resumable_upload_records_progress_and_atomically_completes(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    # Keep the test small while exercising a resumed multi-part transfer.
    orchestrator._UPLOAD_CHUNK_BYTES = 4
    with TestClient(create_app(orchestrator)) as client:
        created = client.post("/v1/uploads/sessions", json={
            "kind": "configs", "filename": "baseline.toml", "size_bytes": 9,
        })
        assert created.status_code == 201
        upload_id = created.json()["upload_id"]
        assert client.put(
            f"/v1/uploads/sessions/{upload_id}/part", content=b"[run",
            headers={"content-range": "bytes 0-3/9"},
        ).status_code == 200
        progress = client.get(f"/v1/uploads/sessions/{upload_id}").json()
        assert progress["received_bytes"] == 4
        assert "path" not in progress
        assert client.put(
            f"/v1/uploads/sessions/{upload_id}/part", content=b"]\na=1",
            headers={"content-range": "bytes 4-8/9"},
        ).status_code == 200
        completed = client.post(f"/v1/uploads/sessions/{upload_id}:complete")
    assert completed.status_code == 200
    assert Path(completed.json()["path"]).read_bytes() == b"[run]\na=1"


def test_dataset_and_artifact_downloads_do_not_expose_storage_paths(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    model = tmp_path / "runs" / "model-a"
    model.mkdir(parents=True)
    (model / "weights.keras").write_bytes(b"weights")
    dataset = tmp_path / "dataset.sqlite"
    dataset.write_bytes(b"SQLite format 3\000")
    now = datetime.now(timezone.utc).isoformat()
    with orchestrator._connection() as db:
        db.execute(
            "INSERT INTO artifacts(artifact_id,artifact_type,name,path,manifest_json,discovered_at,updated_at) VALUES(?,?,?,?,?,?,?)",
            ("artifact-1", "model_run", "Model A", str(model), "{}", now, now),
        )
        db.execute(
            "INSERT INTO datasets VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("dataset-1", None, "Dataset", "classification", "frozen", None, str(dataset), "{}", now, now),
        )
    with TestClient(create_app(orchestrator)) as client:
        archive = client.get("/v1/artifacts/artifact-1:download")
        downloaded_dataset = client.get("/v1/datasets/dataset-1:download")
    assert archive.status_code == 200
    assert archive.headers["content-type"].startswith("application/gzip")
    assert downloaded_dataset.status_code == 200
    assert downloaded_dataset.content == b"SQLite format 3\000"
