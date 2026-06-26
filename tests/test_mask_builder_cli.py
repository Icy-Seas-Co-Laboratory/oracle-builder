from __future__ import annotations

import sqlite3
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import mask_builder
from oracle_builder.data.decoders import decode_blob
from oracle_builder.masking.api_io import ApiMaskSample


def test_input_path_initializes_missing_database_before_listing(monkeypatch, tmp_path):
    db_path = tmp_path / "new" / "dataset.sqlite"
    monkeypatch.setattr(
        "sys.argv",
        ["mask_builder.py", "--input", str(db_path), "--missing-masks-only", "--read-only"],
    )

    with pytest.raises(SystemExit, match="No matching samples found"):
        mask_builder.main()

    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
    assert {"samples", "mask_annotations"}.issubset(tables)


def test_database_path_initializes_missing_database_before_listing(monkeypatch, tmp_path):
    db_path = tmp_path / "new" / "dataset.sqlite"
    monkeypatch.setattr(
        "sys.argv",
        ["mask_builder.py", "--database", str(db_path), "--missing-masks-only", "--read-only"],
    )

    with pytest.raises(SystemExit, match="No matching samples found"):
        mask_builder.main()

    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
    assert {"samples", "mask_annotations"}.issubset(tables)


def test_input_image_path_is_treated_as_image_import(monkeypatch, tmp_path):
    image_path = tmp_path / "000094a2-bcad-4de0-b4e1-37b836463909.png"
    db_path = tmp_path / "datasets" / "data1.sqlite"
    Image.fromarray(np.zeros((5, 6), dtype="uint8")).save(image_path)
    captured = {}

    fake_module = types.ModuleType("oracle_builder.masking.napari_app")

    def fake_launch(**kwargs):
        captured.update(kwargs)

    fake_module.launch_mask_builder_app = fake_launch
    monkeypatch.setitem(sys.modules, "oracle_builder.masking.napari_app", fake_module)
    monkeypatch.setattr(
        "sys.argv",
        ["mask_builder.py", "--input", str(image_path), "--output", str(db_path)],
    )

    assert mask_builder.main() == 0
    assert captured["sample_uuid"] == "000094a2-bcad-4de0-b4e1-37b836463909"
    assert captured["output_db_path"] == str(db_path)
    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT uuid, input_blob_encoding FROM samples").fetchone()
    assert row == ("000094a2-bcad-4de0-b4e1-37b836463909", "png")


def test_image_import_uses_database_argument(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    db_path = tmp_path / "datasets" / "data1.sqlite"
    Image.fromarray(np.zeros((5, 6), dtype="uint8")).save(image_path)
    captured = {}

    fake_module = types.ModuleType("oracle_builder.masking.napari_app")

    def fake_launch(**kwargs):
        captured.update(kwargs)

    fake_module.launch_mask_builder_app = fake_launch
    monkeypatch.setitem(sys.modules, "oracle_builder.masking.napari_app", fake_module)
    monkeypatch.setattr(
        "sys.argv",
        ["mask_builder.py", "--image", str(image_path), "--database", str(db_path)],
    )

    assert mask_builder.main() == 0
    assert captured["sample_uuid"] == "sample"
    assert captured["output_db_path"] == str(db_path)
    assert db_path.exists()


def test_image_folder_builds_save_next_queue(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    db_path = tmp_path / "datasets" / "data1.sqlite"
    Image.fromarray(np.zeros((5, 6), dtype="uint8")).save(image_dir / "a.png")
    Image.fromarray(np.ones((4, 3), dtype="uint8")).save(image_dir / "b.png")
    captured = {}

    fake_module = types.ModuleType("oracle_builder.masking.napari_app")

    def fake_launch(**kwargs):
        captured.update(kwargs)

    fake_module.launch_mask_builder_app = fake_launch
    monkeypatch.setitem(sys.modules, "oracle_builder.masking.napari_app", fake_module)
    monkeypatch.setattr(
        "sys.argv",
        ["mask_builder.py", "--image", str(image_dir), "--database", str(db_path)],
    )

    assert mask_builder.main() == 0
    assert captured["sample_uuid"] == "a"
    assert len(captured["sample_queue"]) == 2
    next_sample = captured["sample_loader"](captured["sample_queue"][1])
    assert next_sample["uuid"] == "b"
    assert next_sample["image"].shape == (4, 3)


def test_api_roi_id_uses_pelagia_detection_loader_by_default(monkeypatch, tmp_path):
    db_path = tmp_path / "api.sqlite"
    image = np.zeros((5, 6), dtype="uint8")
    mask = np.zeros((5, 6), dtype="uint8")
    mask[1:3, 2:4] = 1
    captured = {}

    def fake_load_pelagia_detection(base_url, detection_id, token=None):
        assert base_url == "http://localhost:8000"
        assert detection_id == "roi-7"
        assert token is None
        return ApiMaskSample(
            uuid="roi-7",
            image=image,
            mask=mask,
            metadata={"source": "api"},
            raw={},
        )

    fake_module = types.ModuleType("oracle_builder.masking.napari_app")

    def fake_launch(**kwargs):
        captured.update(kwargs)

    fake_module.launch_mask_builder_app = fake_launch
    monkeypatch.setattr(mask_builder, "load_pelagia_detection", fake_load_pelagia_detection)
    monkeypatch.setitem(sys.modules, "oracle_builder.masking.napari_app", fake_module)
    monkeypatch.setattr(
        "sys.argv",
        ["mask_builder.py", "--api-roi-id", "roi-7", "--database", str(db_path)],
    )

    assert mask_builder.main() == 0
    assert captured["sample_uuid"] == "roi-7"
    assert np.array_equal(captured["initial_mask"], mask)
    assert np.array_equal(captured["initial_candidate_mask"], mask)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT uuid, input_blob, input_blob_encoding, input_blob_dimensions,
                   input_aux_blob, input_aux_blob_encoding, input_aux_blob_dimensions, metadata_json
            FROM samples
            """
        ).fetchone()
    assert row[0] == "roi-7"
    assert row[2] == "npy"
    training_input = decode_blob(row[1], row[2], row[3])
    candidate_mask = decode_blob(row[4], row[5], row[6])
    assert training_input.shape == (5, 6, 2)
    assert np.array_equal(training_input[..., 1] > 0, mask > 0)
    assert np.array_equal(candidate_mask > 0, mask[..., None] > 0)
    assert "api" in row[7]


def test_api_token_is_passed_to_pelagia_loader(monkeypatch, tmp_path):
    db_path = tmp_path / "api.sqlite"
    image = np.zeros((5, 6), dtype="uint8")
    captured = {}

    def fake_load_pelagia_detection(base_url, detection_id, token=None):
        assert detection_id == "roi-token"
        assert token == "cli-token"
        return ApiMaskSample(uuid=detection_id, image=image, mask=None, metadata={}, raw={})

    fake_module = types.ModuleType("oracle_builder.masking.napari_app")

    def fake_launch(**kwargs):
        captured.update(kwargs)

    fake_module.launch_mask_builder_app = fake_launch
    monkeypatch.setattr(mask_builder, "load_pelagia_detection", fake_load_pelagia_detection)
    monkeypatch.setitem(sys.modules, "oracle_builder.masking.napari_app", fake_module)
    monkeypatch.setattr(
        "sys.argv",
        ["mask_builder.py", "--api-roi-id", "roi-token", "--api-token", "cli-token", "--database", str(db_path)],
    )

    assert mask_builder.main() == 0
    assert captured["sample_uuid"] == "roi-token"


def test_api_env_token_is_passed_to_pelagia_list(monkeypatch, capsys):
    def fake_list(base_url, **filters):
        assert base_url == "http://localhost:8000"
        assert filters["token"] == "env-token"
        return [{"id": "d1"}]

    monkeypatch.setenv("PELAGIA_API_TOKEN", "env-token")
    monkeypatch.setattr(mask_builder, "list_pelagia_detections", fake_list)
    monkeypatch.setattr("sys.argv", ["mask_builder.py", "--list-api-rois"])

    assert mask_builder.main() == 0
    assert "d1" in capsys.readouterr().out


def test_list_api_rois_uses_username_password_login(monkeypatch, capsys):
    def fake_login(base_url, username, password, project_key="default"):
        assert base_url == "http://localhost:8000"
        assert username == "ada"
        assert password == "secret"
        assert project_key == "default"
        return SimpleNamespace(token="login-token")

    def fake_list(base_url, **filters):
        assert base_url == "http://localhost:8000"
        assert filters["token"] == "login-token"
        return [{"id": "d1"}]

    monkeypatch.setattr(mask_builder, "login_pelagia", fake_login)
    monkeypatch.setattr(mask_builder, "list_pelagia_detections", fake_list)
    monkeypatch.setattr(
        "sys.argv",
        ["mask_builder.py", "--list-api-rois", "--api-username", "ada", "--api-password", "secret"],
    )

    assert mask_builder.main() == 0
    assert "d1" in capsys.readouterr().out


def test_api_username_password_login_supplies_token(monkeypatch, tmp_path):
    db_path = tmp_path / "api.sqlite"
    image = np.zeros((5, 6), dtype="uint8")
    captured = {}

    def fake_login(base_url, username, password, project_key="default"):
        assert base_url == "http://localhost:8000"
        assert username == "ada"
        assert password == "secret"
        assert project_key == "default"
        return SimpleNamespace(token="login-token")

    def fake_load_pelagia_detection(base_url, detection_id, token=None):
        assert detection_id == "roi-login"
        assert token == "login-token"
        return ApiMaskSample(uuid=detection_id, image=image, mask=None, metadata={}, raw={})

    fake_module = types.ModuleType("oracle_builder.masking.napari_app")

    def fake_launch(**kwargs):
        captured.update(kwargs)

    fake_module.launch_mask_builder_app = fake_launch
    monkeypatch.setattr(mask_builder, "login_pelagia", fake_login)
    monkeypatch.setattr(mask_builder, "load_pelagia_detection", fake_load_pelagia_detection)
    monkeypatch.setitem(sys.modules, "oracle_builder.masking.napari_app", fake_module)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mask_builder.py",
            "--api-roi-id",
            "roi-login",
            "--api-username",
            "ada",
            "--api-password",
            "secret",
            "--database",
            str(db_path),
        ],
    )

    assert mask_builder.main() == 0
    assert captured["sample_uuid"] == "roi-login"


def test_api_browse_rois_builds_save_next_queue(monkeypatch, tmp_path):
    db_path = tmp_path / "api.sqlite"
    captured = {}

    monkeypatch.setattr(
        mask_builder,
        "list_pelagia_detections",
        lambda base_url, **filters: [{"id": "d1"}, {"id": "d2"}],
    )

    def fake_load_pelagia_detection(base_url, detection_id, token=None):
        value = 1 if detection_id == "d1" else 2
        return ApiMaskSample(
            uuid=detection_id,
            image=np.ones((5, 6), dtype="uint8") * value,
            mask=None,
            metadata={"pelagia_detection_id": detection_id},
            raw={},
        )

    fake_module = types.ModuleType("oracle_builder.masking.napari_app")

    def fake_launch(**kwargs):
        captured.update(kwargs)

    fake_module.launch_mask_builder_app = fake_launch
    monkeypatch.setattr(mask_builder, "load_pelagia_detection", fake_load_pelagia_detection)
    monkeypatch.setitem(sys.modules, "oracle_builder.masking.napari_app", fake_module)
    monkeypatch.setattr(
        "sys.argv",
        ["mask_builder.py", "--api-browse-rois", "--database", str(db_path)],
    )

    assert mask_builder.main() == 0
    assert captured["sample_uuid"] == "d1"
    assert len(captured["sample_queue"]) == 2
    next_sample = captured["sample_loader"](captured["sample_queue"][1])
    assert next_sample["uuid"] == "d2"
    assert next_sample["metadata"]["pelagia_detection_id"] == "d2"


def test_api_endpoint_template_uses_generic_loader(monkeypatch, tmp_path):
    db_path = tmp_path / "api.sqlite"
    image = np.zeros((5, 6), dtype="uint8")
    captured = {}

    def fake_load_api_roi(base_url, roi_id, endpoint_template, token=None):
        assert base_url == "http://localhost:8000"
        assert roi_id == "roi-8"
        assert endpoint_template == "/legacy/{roi_id}"
        assert token is None
        return ApiMaskSample(uuid="roi-8", image=image, mask=None, metadata={"source": "generic"}, raw={})

    fake_module = types.ModuleType("oracle_builder.masking.napari_app")

    def fake_launch(**kwargs):
        captured.update(kwargs)

    fake_module.launch_mask_builder_app = fake_launch
    monkeypatch.setattr(mask_builder, "load_api_roi", fake_load_api_roi)
    monkeypatch.setitem(sys.modules, "oracle_builder.masking.napari_app", fake_module)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mask_builder.py",
            "--api-roi-id",
            "roi-8",
            "--api-endpoint-template",
            "/legacy/{roi_id}",
            "--database",
            str(db_path),
        ],
    )

    assert mask_builder.main() == 0
    assert captured["sample_uuid"] == "roi-8"


def test_list_api_rois_prints_detection_ids(monkeypatch, capsys):
    def fake_list(base_url, **filters):
        assert base_url == "http://localhost:8000"
        assert filters["asset_id"] == "asset-1"
        assert filters["min_area"] == 500
        assert filters["max_bbox_w"] == 120
        return [
            {"id": "d1", "asset_id": "asset-1", "frame_id": "f1", "frame_index": 1, "roi_index": 0, "area": 12.5},
            {"id": "d2", "asset_id": "asset-1", "frame_id": "f2", "frame_index": 2, "roi_index": 1, "area": 9.0},
        ]

    monkeypatch.setattr(mask_builder, "list_pelagia_detections", fake_list)
    monkeypatch.setattr(
        "sys.argv",
        ["mask_builder.py", "--list-api-rois", "--asset-id", "asset-1", "--min-area", "500", "--max-bbox-w", "120"],
    )

    assert mask_builder.main() == 0
    output = capsys.readouterr().out
    assert "id\tasset_id" in output
    assert "d1\tasset-1" in output
    assert "d2\tasset-1" in output


def test_list_api_rois_accepts_width_height_aliases(monkeypatch, capsys):
    def fake_list(base_url, **filters):
        assert filters["min_bbox_w"] == 64
        assert filters["min_bbox_h"] == 32
        return [{"id": "d1"}]

    monkeypatch.setattr(mask_builder, "list_pelagia_detections", fake_list)
    monkeypatch.setattr(
        "sys.argv",
        ["mask_builder.py", "--list-api-rois", "--min-width", "64", "--min-height", "32"],
    )

    assert mask_builder.main() == 0
    assert "d1" in capsys.readouterr().out


def test_random_api_roi_selects_detection_before_loading(monkeypatch, tmp_path):
    db_path = tmp_path / "api.sqlite"
    image = np.zeros((5, 6), dtype="uint8")
    captured = {}

    def fake_list(base_url, **filters):
        assert filters["min_area"] == 500
        assert filters["sort_by"] == "id"
        return [{"id": "random-detection"}]

    monkeypatch.setattr(mask_builder, "list_pelagia_detections", fake_list)

    def fake_load_pelagia_detection(base_url, detection_id, token=None):
        assert detection_id == "random-detection"
        assert token is None
        return ApiMaskSample(uuid=detection_id, image=image, mask=None, metadata={"source": "api"}, raw={})

    fake_module = types.ModuleType("oracle_builder.masking.napari_app")

    def fake_launch(**kwargs):
        captured.update(kwargs)

    fake_module.launch_mask_builder_app = fake_launch
    monkeypatch.setattr(mask_builder, "load_pelagia_detection", fake_load_pelagia_detection)
    monkeypatch.setitem(sys.modules, "oracle_builder.masking.napari_app", fake_module)
    monkeypatch.setattr(
        "sys.argv",
        ["mask_builder.py", "--random-api-roi", "--min-area", "500", "--database", str(db_path)],
    )

    assert mask_builder.main() == 0
    assert captured["sample_uuid"] == "random-detection"


def test_random_api_roi_respects_explicit_sort_by(monkeypatch, tmp_path):
    db_path = tmp_path / "api.sqlite"
    image = np.zeros((5, 6), dtype="uint8")
    captured = {}

    def fake_list(base_url, **filters):
        assert filters["sort_by"] == "asset_frame"
        return [{"id": "random-detection"}]

    def fake_load_pelagia_detection(base_url, detection_id, token=None):
        return ApiMaskSample(uuid=detection_id, image=image, mask=None, metadata={"source": "api"}, raw={})

    fake_module = types.ModuleType("oracle_builder.masking.napari_app")

    def fake_launch(**kwargs):
        captured.update(kwargs)

    fake_module.launch_mask_builder_app = fake_launch
    monkeypatch.setattr(mask_builder, "list_pelagia_detections", fake_list)
    monkeypatch.setattr(mask_builder, "load_pelagia_detection", fake_load_pelagia_detection)
    monkeypatch.setitem(sys.modules, "oracle_builder.masking.napari_app", fake_module)
    monkeypatch.setattr(
        "sys.argv",
        ["mask_builder.py", "--random-api-roi", "--sort-by", "asset_frame", "--database", str(db_path)],
    )

    assert mask_builder.main() == 0
    assert captured["sample_uuid"] == "random-detection"


def test_random_api_roi_conflicts_with_explicit_roi(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["mask_builder.py", "--random-api-roi", "--api-roi-id", "d1", "--read-only"],
    )

    with pytest.raises(SystemExit, match="cannot be combined"):
        mask_builder.main()
