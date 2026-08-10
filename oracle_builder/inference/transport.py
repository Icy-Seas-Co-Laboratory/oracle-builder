from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle_builder.inference.contracts import (
    ArrayPayload,
    InferenceItem,
    InferenceResultSet,
    SourceReference,
)


INFERENCE_REQUEST_SCHEMA = "oracle_builder.inference_request"
INFERENCE_TRANSPORT_VERSION = "1.0.0"
NPZ_MEDIA_TYPE = "application/vnd.oracle-builder.inference+npz"


class InferenceTransportError(ValueError):
    """Raised when an inference transport payload violates the wire contract."""


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    items: list[InferenceItem]


def _manifest_array(value: dict[str, Any]) -> np.ndarray:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return np.frombuffer(encoded, dtype=np.uint8)


def _read_manifest(archive: Any) -> dict[str, Any]:
    if "manifest" not in archive.files:
        raise InferenceTransportError("NPZ payload is missing the manifest array")
    try:
        value = json.loads(np.asarray(archive["manifest"], dtype=np.uint8).tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise InferenceTransportError("NPZ manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise InferenceTransportError("NPZ manifest must be a JSON object")
    return value


def _load_npz(payload: bytes, *, max_payload_bytes: int) -> Any:
    if not payload:
        raise InferenceTransportError("Inference payload is empty")
    if len(payload) > max_payload_bytes:
        raise InferenceTransportError(
            f"Inference payload exceeds the {max_payload_bytes}-byte limit"
        )
    try:
        return np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc:
        raise InferenceTransportError("Inference payload is not a valid safe NPZ archive") from exc


def encode_inference_request(request_id: str, items: list[InferenceItem]) -> bytes:
    arrays: dict[str, np.ndarray] = {}
    manifest_items: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        inputs: dict[str, dict[str, Any]] = {}
        for role, array_payload in sorted(item.inputs.items()):
            key = f"item_{index}_{role}"
            arrays[key] = np.asarray(array_payload.values)
            inputs[role] = {
                "transport_key": key,
                "asset_id": array_payload.asset_id,
                "sha256": array_payload.sha256,
                "shape": array_payload.shape,
                "dtype": array_payload.dtype,
            }
        manifest_items.append(
            {
                "item_id": item.item_id,
                "request_id": item.request_id,
                "source": item.source.to_dict() if item.source else None,
                "metadata": item.metadata,
                "inputs": inputs,
            }
        )
    manifest = {
        "schema_name": INFERENCE_REQUEST_SCHEMA,
        "schema_version": INFERENCE_TRANSPORT_VERSION,
        "request_id": request_id,
        "items": manifest_items,
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, manifest=_manifest_array(manifest), **arrays)
    return buffer.getvalue()


def decode_inference_request(
    payload: bytes,
    *,
    max_payload_bytes: int = 256 * 1024 * 1024,
    max_items: int | None = None,
) -> InferenceRequest:
    with _load_npz(payload, max_payload_bytes=max_payload_bytes) as archive:
        manifest = _read_manifest(archive)
        if manifest.get("schema_name") != INFERENCE_REQUEST_SCHEMA:
            raise InferenceTransportError("Unsupported inference request schema")
        if manifest.get("schema_version") != INFERENCE_TRANSPORT_VERSION:
            raise InferenceTransportError("Unsupported inference request schema version")
        rows = manifest.get("items")
        if not isinstance(rows, list) or not rows:
            raise InferenceTransportError("Inference request must contain at least one item")
        if max_items is not None and len(rows) > max_items:
            raise InferenceTransportError(f"Inference request exceeds the {max_items}-item limit")
        items: list[InferenceItem] = []
        for row in rows:
            if not isinstance(row, dict):
                raise InferenceTransportError("Inference item manifest must be an object")
            input_rows = row.get("inputs")
            if not isinstance(input_rows, dict):
                raise InferenceTransportError("Inference item inputs must be an object")
            inputs: dict[str, ArrayPayload] = {}
            for role, description in input_rows.items():
                if not isinstance(description, dict) or not description.get("transport_key"):
                    raise InferenceTransportError(f"Input {role!r} is missing transport_key")
                key = str(description["transport_key"])
                if key not in archive.files:
                    raise InferenceTransportError(f"NPZ payload is missing input array {key!r}")
                inputs[str(role)] = ArrayPayload(
                    np.asarray(archive[key]),
                    asset_id=str(description["asset_id"]),
                    sha256=str(description["sha256"]),
                )
            source_row = row.get("source")
            source = None
            if source_row is not None:
                if not isinstance(source_row, dict):
                    raise InferenceTransportError("Inference item source must be an object")
                source = SourceReference(**source_row)
            items.append(
                InferenceItem(
                    inputs=inputs,
                    item_id=str(row["item_id"]),
                    request_id=str(row["request_id"]),
                    source=source,
                    metadata=dict(row.get("metadata") or {}),
                )
            )
        return InferenceRequest(request_id=str(manifest["request_id"]), items=items)


def _export_arrays(value: Any, arrays: dict[str, np.ndarray], prefix: str) -> Any:
    if isinstance(value, ArrayPayload):
        key = f"output_{len(arrays)}_{prefix}"
        arrays[key] = np.asarray(value.values)
        description = value.to_dict(include_data=False)
        description["transport_key"] = key
        return description
    if isinstance(value, np.ndarray):
        return _export_arrays(ArrayPayload(value), arrays, prefix)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _export_arrays(item, arrays, f"{prefix}_{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _export_arrays(item, arrays, f"{prefix}_{index}")
            for index, item in enumerate(value)
        ]
    return value


def encode_inference_result_set(result_set: InferenceResultSet) -> bytes:
    arrays: dict[str, np.ndarray] = {}
    results: list[dict[str, Any]] = []
    for index, result in enumerate(result_set.results):
        row = result.to_dict(include_array_data=False)
        row["output"] = _export_arrays(result.output, arrays, f"result_{index}")
        results.append(row)
    manifest = result_set.to_dict(include_array_data=False)
    manifest["results"] = results
    manifest["transport"] = {
        "media_type": NPZ_MEDIA_TYPE,
        "version": INFERENCE_TRANSPORT_VERSION,
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, manifest=_manifest_array(manifest), **arrays)
    return buffer.getvalue()


def decode_inference_result(payload: bytes, *, max_payload_bytes: int = 256 * 1024 * 1024) -> dict[str, Any]:
    """Decode a result archive into a JSON-like manifest with NumPy arrays restored."""
    with _load_npz(payload, max_payload_bytes=max_payload_bytes) as archive:
        manifest = _read_manifest(archive)

        def restore(value: Any) -> Any:
            if isinstance(value, dict):
                key = value.get("transport_key")
                if key is not None:
                    if key not in archive.files:
                        raise InferenceTransportError(f"NPZ result is missing output array {key!r}")
                    return np.asarray(archive[key])
                return {str(k): restore(v) for k, v in value.items()}
            if isinstance(value, list):
                return [restore(item) for item in value]
            return value

        return restore(manifest)
