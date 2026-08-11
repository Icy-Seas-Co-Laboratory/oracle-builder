from __future__ import annotations

import base64
import json
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_builder.data.decoders import decode_blob
from oracle_builder.masking.sqlite_io import decode_mask


@dataclass
class ApiMaskSample:
    uuid: str
    image: np.ndarray
    mask: np.ndarray | None
    metadata: dict[str, Any]
    raw: dict[str, Any]


@dataclass
class PelagiaSession:
    token: str
    user: dict[str, Any]
    project: dict[str, Any]
    session: dict[str, Any]


def build_api_url(base_url: str, endpoint_template: str, roi_id: str) -> str:
    base = base_url.rstrip("/")
    endpoint = endpoint_template.format(roi_id=urllib.parse.quote(roi_id, safe=""))
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    return f"{base}/{endpoint.lstrip('/')}"


def build_url(base_url: str, path: str, query: dict[str, Any] | None = None) -> str:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    clean_query = {key: value for key, value in (query or {}).items() if value is not None}
    if clean_query:
        url = f"{url}?{urllib.parse.urlencode(clean_query)}"
    return url


def request_headers(accept: str, token: str | None = None, content_type: str | None = None) -> dict[str, str]:
    headers = {"Accept": accept}
    if content_type:
        headers["Content-Type"] = content_type
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float = 30.0,
    token: str | None = None,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers("application/json", token=token, content_type="application/json"),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload = response.read()
    except urllib.error.HTTPError as exc:
        _raise_pelagia_http_error(exc)
    data = json.loads(response_payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"API response must be a JSON object, got {type(data).__name__}")
    return data


def login_pelagia(
    base_url: str,
    username: str,
    password: str,
    project_key: str = "default",
    timeout: float = 30.0,
) -> PelagiaSession:
    payload = post_json(
        build_url(base_url, "/auth/login"),
        {"username": username, "password": password, "project_key": project_key},
        timeout=timeout,
    )
    token = payload.get("token")
    if not isinstance(token, str) or not token:
        raise ValueError("Pelagia login response did not include a token.")
    return PelagiaSession(
        token=token,
        user=payload.get("user") if isinstance(payload.get("user"), dict) else {},
        project=payload.get("project") if isinstance(payload.get("project"), dict) else {},
        session=payload.get("session") if isinstance(payload.get("session"), dict) else {},
    )


def fetch_json(url: str, timeout: float = 30.0, token: str | None = None) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=request_headers("application/json", token=token))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        _raise_pelagia_http_error(exc)
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"API response must be a JSON object, got {type(data).__name__}")
    return data


def fetch_bytes(
    url: str,
    timeout: float = 30.0,
    accept: str = "image/png, application/json",
    token: str | None = None,
) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers=request_headers(accept, token=token))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        _raise_pelagia_http_error(exc)
    return payload, content_type


def _raise_pelagia_http_error(exc: urllib.error.HTTPError) -> None:
    if exc.code in {401, 403}:
        raise RuntimeError(
            "Pelagia rejected the request. Pass --api-token or log in with "
            "--api-username, --api-password, and --api-project-key."
        ) from exc
    raise exc


def load_api_roi(
    base_url: str,
    roi_id: str,
    endpoint_template: str = "/rois/{roi_id}",
    timeout: float = 30.0,
    token: str | None = None,
) -> ApiMaskSample:
    url = build_api_url(base_url, endpoint_template, roi_id)
    payload = fetch_json(url, timeout=timeout, token=token)
    return parse_api_roi_payload(payload, fallback_uuid=roi_id, source_url=url)


def load_pelagia_detection(
    base_url: str,
    detection_id: str,
    image_format: str = "png",
    mask_format: str = "png",
    include_mask: bool = True,
    timeout: float = 30.0,
    token: str | None = None,
) -> ApiMaskSample:
    detail_url = build_url(base_url, f"/detections/{urllib.parse.quote(detection_id, safe='')}")
    detail_payload = fetch_json(detail_url, timeout=timeout, token=token)
    detection = detail_payload.get("detection", detail_payload)
    if not isinstance(detection, dict):
        detection = {}

    roi_url = build_url(
        base_url,
        f"/detections/{urllib.parse.quote(detection_id, safe='')}/roi",
        {"format": image_format},
    )
    roi_bytes, roi_content_type = fetch_bytes(roi_url, timeout=timeout, accept=f"image/{image_format}, application/json", token=token)
    image, roi_metadata = decode_pelagia_image_response(roi_bytes, roi_content_type, source_url=roi_url, mask=False)

    mask = None
    mask_metadata: dict[str, Any] = {}
    mask_url = build_url(
        base_url,
        f"/detections/{urllib.parse.quote(detection_id, safe='')}/mask",
        {"format": mask_format},
    )
    if include_mask:
        try:
            mask_bytes, mask_content_type = fetch_bytes(mask_url, timeout=timeout, accept=f"image/{mask_format}, application/json", token=token)
            decoded_mask, mask_metadata = decode_pelagia_image_response(mask_bytes, mask_content_type, source_url=mask_url, mask=True)
            mask = (np.asarray(decoded_mask) > 0).astype("uint8")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            mask_metadata = {"missing_mask": True, "status_code": 404, "source_url": mask_url}

    uuid = str(detection.get("id") or detection_id)
    metadata = {
        **(detection.get("metadata") if isinstance(detection.get("metadata"), dict) else {}),
        "pelagia_detection_id": uuid,
        "pelagia": {
            "detection": detection,
            "detail_url": detail_url,
            "roi": roi_metadata,
            "mask": mask_metadata,
        },
    }
    return ApiMaskSample(
        uuid=uuid,
        image=np.asarray(image),
        mask=mask,
        metadata=metadata,
        raw={"detection": detection, "roi_metadata": roi_metadata, "mask_metadata": mask_metadata},
    )


def list_pelagia_detections(
    base_url: str,
    run_id: str | None = None,
    asset_id: str | None = None,
    collection: str | None = None,
    frame_id: str | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    roi_index: int | None = None,
    min_bbox_x: int | None = None,
    max_bbox_x: int | None = None,
    min_bbox_y: int | None = None,
    max_bbox_y: int | None = None,
    min_bbox_w: int | None = None,
    max_bbox_w: int | None = None,
    min_bbox_h: int | None = None,
    max_bbox_h: int | None = None,
    min_area: float | None = None,
    max_area: float | None = None,
    min_perimeter: float | None = None,
    max_perimeter: float | None = None,
    roi_encoding: str | None = None,
    roi_format: str | None = None,
    mask_encoding: str | None = None,
    mask_format: str | None = None,
    has_roi_payload: bool = True,
    limit: int | None = 25,
    offset: int = 0,
    sort_by: str = "asset_frame",
    sort_dir: str = "desc",
    timeout: float = 30.0,
    token: str | None = None,
) -> list[dict[str, Any]]:
    url = build_url(
        base_url,
        "/detections",
        {
            "run_id": run_id,
            "asset_id": asset_id,
            "collection": collection,
            "frame_id": frame_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "roi_index": roi_index,
            "min_bbox_x": min_bbox_x,
            "max_bbox_x": max_bbox_x,
            "min_bbox_y": min_bbox_y,
            "max_bbox_y": max_bbox_y,
            "min_bbox_w": min_bbox_w,
            "max_bbox_w": max_bbox_w,
            "min_bbox_h": min_bbox_h,
            "max_bbox_h": max_bbox_h,
            "min_area": min_area,
            "max_area": max_area,
            "min_perimeter": min_perimeter,
            "max_perimeter": max_perimeter,
            "roi_encoding": roi_encoding,
            "roi_format": roi_format,
            "mask_encoding": mask_encoding,
            "mask_format": mask_format,
            "has_roi_payload": has_roi_payload,
            "limit": limit,
            "offset": offset,
            "sort_by": sort_by,
            "sort_dir": sort_dir,
        },
    )
    payload = fetch_json(url, timeout=timeout, token=token)
    detections = payload.get("detections", [])
    if not isinstance(detections, list):
        raise ValueError("Pelagia /detections response did not contain a detections list.")
    return [detection for detection in detections if isinstance(detection, dict)]


def detection_id_from_summary(detection: dict[str, Any]) -> str:
    detection_id = detection.get("id") or detection.get("detection_id") or detection.get("uuid")
    if not detection_id:
        raise ValueError(f"Detection summary does not include an id: {detection}")
    return str(detection_id)


def choose_random_pelagia_detection_id(
    base_url: str,
    run_id: str | None = None,
    asset_id: str | None = None,
    collection: str | None = None,
    frame_id: str | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    roi_index: int | None = None,
    min_bbox_x: int | None = None,
    max_bbox_x: int | None = None,
    min_bbox_y: int | None = None,
    max_bbox_y: int | None = None,
    min_bbox_w: int | None = None,
    max_bbox_w: int | None = None,
    min_bbox_h: int | None = None,
    max_bbox_h: int | None = None,
    min_area: float | None = None,
    max_area: float | None = None,
    min_perimeter: float | None = None,
    max_perimeter: float | None = None,
    roi_encoding: str | None = None,
    roi_format: str | None = None,
    mask_encoding: str | None = None,
    mask_format: str | None = None,
    limit: int | None = 100,
    offset: int = 0,
    sort_by: str = "random",
    sort_dir: str = "desc",
    timeout: float = 30.0,
    token: str | None = None,
) -> str:
    detections = list_pelagia_detections(
        base_url,
        run_id=run_id,
        asset_id=asset_id,
        collection=collection,
        frame_id=frame_id,
        start_frame=start_frame,
        end_frame=end_frame,
        roi_index=roi_index,
        min_bbox_x=min_bbox_x,
        max_bbox_x=max_bbox_x,
        min_bbox_y=min_bbox_y,
        max_bbox_y=max_bbox_y,
        min_bbox_w=min_bbox_w,
        max_bbox_w=max_bbox_w,
        min_bbox_h=min_bbox_h,
        max_bbox_h=max_bbox_h,
        min_area=min_area,
        max_area=max_area,
        min_perimeter=min_perimeter,
        max_perimeter=max_perimeter,
        roi_encoding=roi_encoding,
        roi_format=roi_format,
        mask_encoding=mask_encoding,
        mask_format=mask_format,
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        sort_dir=sort_dir,
        timeout=timeout,
        token=token,
    )
    if not detections:
        raise ValueError("Pelagia returned no detections for the requested filters.")
    return detection_id_from_summary(detections[0])


def decode_pelagia_image_response(
    payload: bytes,
    content_type: str,
    source_url: str | None = None,
    mask: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type == "application/json" or (not media_type and payload.strip().startswith(b"{")):
        data = json.loads(payload.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Pelagia image JSON response must be an object.")
        array = _decode_pelagia_matrix(data, mask=mask)
        metadata = {
            "source_url": source_url,
            "content_type": content_type,
            "response": {key: value for key, value in data.items() if key not in {"data"}},
        }
        return array, metadata
    encoding = _normalize_encoding(media_type or "png")
    array = _decode_blob_payload(payload, encoding, None, mask=mask)
    metadata = {"source_url": source_url, "content_type": content_type, "encoding": encoding}
    return np.asarray(array), metadata


def _decode_pelagia_matrix(payload: dict[str, Any], mask: bool = False) -> np.ndarray:
    if "data" not in payload or "shape" not in payload:
        return _decode_array_payload(payload, default_encoding=payload.get("encoding"), mask=mask)
    dtype = payload.get("dtype") or "float32"
    array = np.asarray(payload["data"], dtype=dtype)
    shape = payload["shape"]
    if shape:
        array = array.reshape(shape)
    if mask:
        return (array > 0).astype("uint8")
    return array


def parse_api_roi_payload(payload: dict[str, Any], fallback_uuid: str, source_url: str | None = None) -> ApiMaskSample:
    uuid = str(payload.get("uuid") or payload.get("id") or payload.get("roi_id") or fallback_uuid)
    image_payload = _first_present(payload, ("image", "roi", "input", "input_image", "input_blob", "roi_image"))
    if image_payload is None:
        raise ValueError("API response did not include an image/ROI payload.")
    image = _decode_array_payload(image_payload, default_encoding=payload.get("input_blob_encoding"))
    mask_payload = _first_present(payload, ("mask", "output", "output_mask", "output_blob", "segmentation"))
    mask = None
    if mask_payload not in (None, ""):
        mask = (_decode_array_payload(mask_payload, default_encoding=payload.get("output_blob_encoding"), mask=True) > 0).astype("uint8")
    metadata = _metadata_from_payload(payload, uuid, source_url)
    return ApiMaskSample(uuid=uuid, image=np.asarray(image), mask=mask, metadata=metadata, raw=payload)


def _metadata_from_payload(payload: dict[str, Any], uuid: str, source_url: str | None) -> dict[str, Any]:
    metadata = payload.get("metadata") or payload.get("metadata_json") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {"metadata_text": metadata}
    if not isinstance(metadata, dict):
        metadata = {"metadata": metadata}
    reserved = {
        "image",
        "roi",
        "input",
        "input_image",
        "input_blob",
        "roi_image",
        "mask",
        "output",
        "output_mask",
        "output_blob",
        "segmentation",
    }
    extras = {key: value for key, value in payload.items() if key not in reserved and key not in {"metadata", "metadata_json"}}
    metadata = {
        **metadata,
        "api_import": {
            "uuid": uuid,
            "source_url": source_url,
            "extra_fields": extras,
        },
    }
    return metadata


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _decode_array_payload(value: Any, default_encoding: str | None = None, mask: bool = False) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, (list, dict)):
            return _decode_array_payload(parsed, default_encoding=default_encoding, mask=mask)
        blob = _decode_base64(value)
        encoding = default_encoding or "png"
        return _decode_blob_payload(blob, encoding, None, mask=mask)
    if not isinstance(value, dict):
        raise ValueError(f"Unsupported API array payload type: {type(value).__name__}")

    if "array" in value:
        array = np.asarray(value["array"])
        dimensions = value.get("dimensions") or value.get("shape")
        return array.reshape(dimensions) if dimensions else array
    if "values" in value:
        array = np.asarray(value["values"])
        dimensions = value.get("dimensions") or value.get("shape")
        return array.reshape(dimensions) if dimensions else array

    nested = _first_present(value, ("image", "roi", "mask", "data", "blob", "bytes", "content"))
    if nested is None:
        raise ValueError("API array payload did not include array, values, data, blob, bytes, or content.")
    if isinstance(nested, (list, dict)):
        return _decode_array_payload(nested, default_encoding=value.get("encoding") or default_encoding, mask=mask)

    blob = _decode_base64(str(nested))
    encoding = value.get("encoding") or value.get("blob_encoding") or value.get("mime_type") or default_encoding or "png"
    dimensions = value.get("dimensions") or value.get("shape")
    return _decode_blob_payload(blob, encoding, dimensions, mask=mask)


def _decode_base64(value: str) -> bytes:
    text = value.strip()
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    return base64.b64decode(text)


def _decode_blob_payload(blob: bytes, encoding: str, dimensions: Any, mask: bool = False) -> np.ndarray:
    normalized_encoding = _normalize_encoding(encoding)
    dimensions_json = json.dumps(dimensions) if dimensions is not None and not isinstance(dimensions, str) else dimensions
    if mask and normalized_encoding in {"png", "npy"}:
        decoded = decode_mask(blob, normalized_encoding, dimensions_json)
        if decoded is not None:
            return decoded
    return np.asarray(decode_blob(blob, normalized_encoding, dimensions_json))


def _normalize_encoding(encoding: str) -> str:
    value = encoding.lower()
    if value in {"image/png", "png"}:
        return "png"
    if value in {"image/jpeg", "image/jpg", "jpg", "jpeg"}:
        return "jpg"
    if value in {"image/tiff", "tif", "tiff"}:
        return "tif"
    if value in {"application/x-npy", "application/octet-stream", "npy", "nparray"}:
        return "npy"
    return value
