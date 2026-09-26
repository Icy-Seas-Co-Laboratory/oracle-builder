from __future__ import annotations

import sqlite3

from oracle_builder.orchestration.training_catalog import scan_training_catalog


def _sqlite_dataset(path, *, name="ISIISNet", frozen=False):
    from oracle_builder.data.sqlite_dataset import create_synthetic_classification

    create_synthetic_classification(path, n=4, shape=(12, 8, 1), classes=2)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE dataset SET name=?, lifecycle=?", (name, "frozen" if frozen else "working"))
        connection.commit()


def test_catalog_ignores_image_directories_until_explicitly_converted(tmp_path):
    from PIL import Image

    source = tmp_path / "catalog" / "marine" / "copepod"
    source.mkdir(parents=True)
    Image.new("L", (12, 8), color=42).save(source / "one.png")

    assert scan_training_catalog(tmp_path / "catalog")["entries"] == []
    assert (source / "one.png").exists()


def test_catalog_requires_existing_root(tmp_path):
    try:
        scan_training_catalog(tmp_path / "missing")
    except NotADirectoryError:
        pass
    else:
        raise AssertionError("a missing catalog root was accepted")


def test_catalog_scans_oracle_sqlite_datasets_and_derives_a_version_bundle(tmp_path):
    dataset = tmp_path / "datasets" / "ISIISNet.pre-registry-20260807T090213.sqlite"
    _sqlite_dataset(dataset, frozen=True)

    report = scan_training_catalog(dataset.parent)

    assert len(report["entries"]) == 1
    entry = report["entries"][0]
    assert entry["source_type"] == "oracle_sqlite"
    assert entry["status"] == "frozen"
    assert entry["item_count"] == 4
    assert entry["class_count"] == 2
    assert entry["training_set_family"] == "ISIISNet"
    assert entry["training_set_version"] == "pre-registry-20260807T090213"


def test_control_plane_catalog_serves_sqlite_previews_freezes_and_groups_revisions(tmp_path):
    from fastapi.testclient import TestClient

    from oracle_builder.orchestration.api import create_app
    from oracle_builder.orchestration.service import Orchestrator

    source_root = tmp_path / "datasets"
    source_root.mkdir()
    working = source_root / "ISIISNet.pre-registry-20260807T090213.sqlite"
    frozen = source_root / "ISIISNet.registry-20260808T090213.sqlite"
    _sqlite_dataset(working)
    _sqlite_dataset(frozen, frozen=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    orchestrator = Orchestrator(tmp_path / "orchestrator.sqlite", workspace_root=workspace, training_catalog_roots=[source_root])

    report = orchestrator.scan_training_catalog()
    working_entry = next(entry for entry in report["entries"] if entry["path"] == str(working))
    assert len(report["registration"]["registered"]) == 1
    assert [dataset["name"] for dataset in orchestrator.datasets()] == ["ISIISNet"]
    catalog_entry = next(entry for entry in orchestrator.training_catalog() if entry["catalog_id"] == working_entry["catalog_id"])
    assert catalog_entry["item_count"] == 4
    assert len(orchestrator.training_catalog_bundles()) == 1
    previews = orchestrator.training_catalog_previews(working_entry["catalog_id"])
    assert previews["items"][0]["preview_url"].startswith("/api/v1/training-catalog/")
    assert orchestrator.training_catalog_preview_image(working_entry["catalog_id"], previews["items"][0]["item_id"]).startswith(b"\xff\xd8")

    with TestClient(create_app(orchestrator)) as client:
        detail = client.get(f"/v1/training-catalog/{working_entry['catalog_id']}")
        assert detail.status_code == 200
        assert detail.json()["classes"][1]["name"] == "1"
        image = client.get(f"/v1/training-catalog/{working_entry['catalog_id']}/previews/{previews['items'][0]['item_id']}")
        assert image.status_code == 200
        frozen_response = client.post(f"/v1/training-catalog/{working_entry['catalog_id']}:freeze")
        assert frozen_response.status_code == 200
        assert frozen_response.json()["entry"]["status"] == "frozen"
        assert len(client.get("/v1/datasets").json()["datasets"]) == 2
