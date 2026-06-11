from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from oracle_builder.masking.api_io import (
    build_api_url,
    choose_random_pelagia_detection_id,
    decode_pelagia_image_response,
    detection_id_from_summary,
    list_pelagia_detections,
    parse_api_roi_payload,
)


def _png_base64(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_build_api_url_uses_roi_template():
    assert (
        build_api_url("http://localhost:8000/", "/rois/{roi_id}", "abc 123")
        == "http://localhost:8000/rois/abc%20123"
    )


def test_parse_api_roi_payload_decodes_image_mask_and_metadata():
    image = np.zeros((5, 6), dtype="uint8")
    image[:, 3:] = 128
    mask = np.zeros((5, 6), dtype="uint8")
    mask[1:4, 2:5] = 255
    payload = {
        "roi_id": "roi-1",
        "roi": {"data": _png_base64(image), "encoding": "png", "dimensions": [5, 6]},
        "mask": {"data": _png_base64(mask), "encoding": "png", "dimensions": [5, 6]},
        "metadata": {"case": "demo"},
        "score": 0.91,
    }

    sample = parse_api_roi_payload(payload, fallback_uuid="fallback", source_url="http://localhost:8000/rois/roi-1")

    assert sample.uuid == "roi-1"
    assert sample.image.shape == (5, 6)
    assert sample.mask is not None
    assert set(np.unique(sample.mask).tolist()) == {0, 1}
    assert sample.metadata["case"] == "demo"
    assert sample.metadata["api_import"]["extra_fields"]["score"] == 0.91


def test_parse_api_roi_payload_accepts_standardized_blob_field_names():
    image = np.ones((3, 4), dtype="uint8") * 200
    payload = {
        "uuid": "sample-1",
        "input_blob": _png_base64(image),
        "input_blob_encoding": "png",
        "metadata_json": '{"source": "api"}',
    }

    sample = parse_api_roi_payload(payload, fallback_uuid="fallback")

    assert sample.uuid == "sample-1"
    assert sample.image.shape == (3, 4)
    assert sample.mask is None
    assert sample.metadata["source"] == "api"


def test_decode_pelagia_png_response():
    image = np.ones((3, 4), dtype="uint8") * 100
    payload = base64.b64decode(_png_base64(image))
    decoded, metadata = decode_pelagia_image_response(payload, "image/png", source_url="http://api/detections/d1/roi")
    assert decoded.shape == (3, 4)
    assert metadata["encoding"] == "png"


def test_decode_pelagia_json_matrix_response_as_mask():
    payload = {
        "dtype": "uint8",
        "shape": [2, 3],
        "data": [0, 1, 2, 0, 0, 5],
        "detection_id": "d1",
        "payload_kind": "mask",
    }
    decoded, metadata = decode_pelagia_image_response(
        str(payload).replace("'", '"').encode("utf-8"),
        "application/json",
        source_url="http://api/detections/d1/mask",
        mask=True,
    )
    assert decoded.tolist() == [[0, 1, 1], [0, 0, 1]]
    assert metadata["response"]["detection_id"] == "d1"


def test_list_pelagia_detections_parses_detection_list(monkeypatch):
    captured = {}

    def fake_fetch_json(url, timeout=30.0):
        captured["url"] = url
        return {"detections": [{"id": "d1"}, {"id": "d2"}]}

    monkeypatch.setattr("oracle_builder.masking.api_io.fetch_json", fake_fetch_json)

    detections = list_pelagia_detections("http://api", asset_id="asset-1", min_area=500, max_bbox_w=80, limit=2)

    assert [detection_id_from_summary(item) for item in detections] == ["d1", "d2"]
    assert "asset_id=asset-1" in captured["url"]
    assert "min_area=500" in captured["url"]
    assert "max_bbox_w=80" in captured["url"]
    assert "limit=2" in captured["url"]


def test_choose_random_pelagia_detection_id(monkeypatch):
    monkeypatch.setattr(
        "oracle_builder.masking.api_io.list_pelagia_detections",
        lambda *args, **kwargs: [{"id": "d1"}],
    )
    assert choose_random_pelagia_detection_id("http://api") == "d1"
