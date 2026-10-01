"""Dependency-light V2 interactive-inference NPZ wire contract.

This module deliberately contains only NumPy and standard-library dependencies.
It is the client-facing contract for services that submit Pelagia-style batches
without importing the Oracle Builder runtime or its ML dependencies.
"""
from __future__ import annotations

import hashlib
import io
import json
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

V2_INFERENCE_REQUEST_SCHEMA = "oracle_builder.inference.v2.request"
V2_INFERENCE_RESULT_SET_SCHEMA = "oracle_builder.inference.v2.result_set"
V2_INFERENCE_TRANSPORT_VERSION = "2.0.0"
V2_NPZ_MEDIA_TYPE = "application/vnd.oracle-builder.inference.v2+npz"
IMAGE_OUTPUT_SCHEMA = "oracle_builder.inference.v2.image_output"
IMAGE_OUTPUT_VERSION = "1.0.0"

InferenceTask = Literal["classification", "mask_refinement"]
DEFAULT_MAX_ARRAY_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_TOTAL_ARRAY_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_CLIENT_OUTPUT_BYTES = 32 * 1024 * 1024
_INPUTS = {"classification": frozenset({"image"}), "mask_refinement": frozenset({"image", "candidate_mask"})}
_OUTPUTS = {
    "classification": frozenset({"class_probabilities", "primary_decision", "embedding", "embedding_normalized", "knn", "prototype_similarity", "secondary_heads", "cluster_evidence", "diagnostics"}),
    "mask_refinement": frozenset({"mask", "probability_map", "logits", "delta_mask", "delta_probability_map"}),
}
_DEFAULT_OUTPUTS = {"classification": _OUTPUTS["classification"], "mask_refinement": frozenset({"mask"})}
_IMAGE_REQUIRED_FIELDS = frozenset({"class_probabilities", "primary_decision", "embedding", "embedding_normalized", "knn", "prototype_similarity", "secondary_heads", "cluster_evidence", "schema_name", "schema_version", "type", "model_task", "diagnostics"})

class V2InferenceTransportError(ValueError):
    """A V2 payload or requested field is outside the published contract."""

class V2InferenceOutputLimitError(V2InferenceTransportError):
    """A requested response would exceed its declared output budget."""

def _array_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO(); np.save(buffer, np.asarray(value), allow_pickle=False); return buffer.getvalue()

def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

@dataclass(frozen=True)
class ArrayPayload:
    """A NumPy array with a stable asset identifier and verified content hash."""
    values: np.ndarray
    asset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    media_type: str = "application/x-npy"
    sha256: str | None = None
    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "asset_id", str(uuid.UUID(self.asset_id)))
        actual = _sha256(_array_bytes(values))
        if self.sha256 is not None and self.sha256 != actual:
            raise ValueError("Array payload SHA-256 does not match its values")
        object.__setattr__(self, "sha256", actual)
    @property
    def shape(self) -> list[int]: return [int(v) for v in self.values.shape]
    @property
    def dtype(self) -> str: return str(self.values.dtype)

@dataclass(frozen=True)
class SourceReference:
    system: str
    resource_type: str
    resource_id: str
    revision: str | None = None
    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in {"system": self.system, "resource_type": self.resource_type, "resource_id": self.resource_id, "revision": self.revision}.items() if value is not None}

@dataclass(frozen=True)
class InferenceItem:
    inputs: dict[str, ArrayPayload]
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: SourceReference | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", str(uuid.UUID(self.item_id)))
        if not self.request_id: raise ValueError("request_id cannot be empty")
        if "image" not in self.inputs: raise ValueError("InferenceItem.inputs must contain an 'image' payload")
    @classmethod
    def from_array(cls, image: Any, *, candidate_mask: Any | None = None, item_id: str | None = None, request_id: str | None = None, source: SourceReference | None = None, metadata: dict[str, Any] | None = None) -> "InferenceItem":
        inputs = {"image": ArrayPayload(np.asarray(image))}
        if candidate_mask is not None: inputs["candidate_mask"] = ArrayPayload(np.asarray(candidate_mask))
        return cls(inputs=inputs, item_id=item_id or str(uuid.uuid4()), request_id=request_id or str(uuid.uuid4()), source=source, metadata=dict(metadata or {}))
    @property
    def input_sha256(self) -> str:
        digest = hashlib.sha256()
        for role, payload in sorted(self.inputs.items()):
            digest.update(role.encode("utf-8")); digest.update(bytes.fromhex(payload.sha256 or ""))
        return digest.hexdigest()

@dataclass(frozen=True)
class V2ResponseOptions:
    outputs: frozenset[str] | None = None
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    def resolved_outputs(self, task: InferenceTask) -> frozenset[str]:
        _validate_task(task); requested = self.outputs if self.outputs is not None else _DEFAULT_OUTPUTS[task]
        unsupported = requested - _OUTPUTS[task]
        if unsupported: raise V2InferenceTransportError(f"Unsupported {task} outputs: {', '.join(sorted(unsupported))}")
        if not 1 <= self.max_output_bytes <= MAX_CLIENT_OUTPUT_BYTES: raise V2InferenceTransportError(f"max_output_bytes must be between 1 and {MAX_CLIENT_OUTPUT_BYTES}")
        return requested
    def to_dict(self, task: InferenceTask) -> dict[str, Any]:
        return {"outputs": sorted(self.resolved_outputs(task)) if self.outputs is not None or task == "mask_refinement" else None, "max_output_bytes": self.max_output_bytes}

@dataclass(frozen=True)
class V2InferenceRequest:
    request_id: str
    task: InferenceTask
    items: list[InferenceItem]
    response_options: V2ResponseOptions = field(default_factory=V2ResponseOptions)
    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip(): raise V2InferenceTransportError("request_id must be a non-empty string")
        _validate_task(self.task)
        if not self.items: raise V2InferenceTransportError("V2 inference request must contain at least one item")
        if len({item.item_id for item in self.items}) != len(self.items): raise V2InferenceTransportError("item_id values must be unique within a request")
        for item in self.items:
            if item.request_id != self.request_id: raise V2InferenceTransportError("Every item.request_id must equal request_id")
            extra = set(item.inputs) - _INPUTS[self.task]
            if extra: raise V2InferenceTransportError(f"Unsupported {self.task} inputs: {', '.join(sorted(extra))}")
            if "image" not in item.inputs: raise V2InferenceTransportError("Every inference item requires an image input")
            _json_object(item.metadata, "item metadata")
        self.response_options.resolved_outputs(self.task)

def _validate_task(task: str) -> None:
    if task not in _INPUTS: raise V2InferenceTransportError("task must be one of: classification, mask_refinement")
def _json_object(value: Any, label: str) -> None:
    try: json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc: raise V2InferenceTransportError(f"{label} must be JSON serializable") from exc
def _manifest_array(value: dict[str, Any]) -> np.ndarray:
    _json_object(value, "manifest"); return np.frombuffer(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(), dtype=np.uint8)
def _read_manifest(archive: Any) -> dict[str, Any]:
    if "manifest" not in archive.files: raise V2InferenceTransportError("NPZ payload is missing its manifest")
    try: value = json.loads(np.asarray(archive["manifest"], dtype=np.uint8).tobytes().decode())
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc: raise V2InferenceTransportError("NPZ manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict): raise V2InferenceTransportError("NPZ manifest must be a JSON object")
    return value
def _load_npz(payload: bytes, *, max_payload_bytes: int, max_decompressed_bytes: int) -> Any:
    if not payload: raise V2InferenceTransportError("Inference payload is empty")
    if len(payload) > max_payload_bytes: raise V2InferenceTransportError(f"Inference payload exceeds the {max_payload_bytes}-byte limit")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
            entries = bundle.infolist(); names = [entry.filename for entry in entries]
            if len(names) != len(set(names)) or any("/" in name or "\\" in name for name in names): raise V2InferenceTransportError("NPZ payload has invalid member names")
            if sum(entry.file_size for entry in entries) > max_decompressed_bytes: raise V2InferenceTransportError("NPZ payload exceeds the decompressed byte limit")
    except V2InferenceTransportError: raise
    except (OSError, zipfile.BadZipFile) as exc: raise V2InferenceTransportError("Inference payload is not a valid NPZ archive") from exc
    try: return np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc: raise V2InferenceTransportError("Inference payload is not a valid safe NPZ archive") from exc
def _array_description(payload: Any, key: str) -> dict[str, Any]:
    values = np.asarray(payload.values)
    return {"transport_key": key, "asset_id": payload.asset_id, "sha256": payload.sha256, "shape": [int(v) for v in values.shape], "dtype": str(values.dtype), "media_type": getattr(payload, "media_type", "application/x-npy")}
def _checked_array(archive: Any, key: str, desc: dict[str, Any], *, max_array_bytes: int, remaining_total_bytes: int) -> tuple[ArrayPayload, int]:
    if key not in archive.files: raise V2InferenceTransportError(f"NPZ payload is missing array {key!r}")
    try: values = np.asarray(archive[key])
    except Exception as exc: raise V2InferenceTransportError(f"Unable to read NPZ array {key!r}") from exc
    if values.dtype.hasobject or values.dtype.fields: raise V2InferenceTransportError(f"NPZ array {key!r} has an unsafe dtype")
    if values.nbytes > max_array_bytes: raise V2InferenceTransportError(f"NPZ array {key!r} exceeds the per-array byte limit")
    if values.nbytes > remaining_total_bytes: raise V2InferenceTransportError("NPZ arrays exceed the decompressed byte limit")
    if desc.get("shape") != [int(v) for v in values.shape] or desc.get("dtype") != str(values.dtype): raise V2InferenceTransportError(f"NPZ array {key!r} shape or dtype does not match its manifest")
    try: value = ArrayPayload(values, asset_id=str(desc["asset_id"]), media_type=str(desc.get("media_type", "application/x-npy")), sha256=str(desc["sha256"]))
    except (KeyError, TypeError, ValueError) as exc: raise V2InferenceTransportError(f"NPZ array {key!r} failed identity verification") from exc
    return value, remaining_total_bytes - values.nbytes
def _encode_npz(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO(); np.savez_compressed(buffer, manifest=_manifest_array(manifest), **arrays); return buffer.getvalue()
def _validate_limits(*limits: int) -> None:
    if any(not isinstance(value, int) or value < 1 for value in limits): raise ValueError("Transport byte limits must be positive integers")

def encode_v2_inference_request(request: V2InferenceRequest) -> bytes:
    arrays: dict[str, np.ndarray] = {}; rows = []
    for index, item in enumerate(request.items):
        inputs = {}
        for role, payload in sorted(item.inputs.items()):
            key = f"item_{index}_{role}"; arrays[key] = np.asarray(payload.values); inputs[role] = _array_description(payload, key)
        rows.append({"item_id": item.item_id, "request_id": item.request_id, "source": item.source.to_dict() if item.source else None, "metadata": item.metadata, "input_sha256": item.input_sha256, "inputs": inputs})
    return _encode_npz({"schema_name": V2_INFERENCE_REQUEST_SCHEMA, "schema_version": V2_INFERENCE_TRANSPORT_VERSION, "request_id": request.request_id, "task": request.task, "response_options": request.response_options.to_dict(request.task), "items": rows}, arrays)

def decode_v2_inference_request(payload: bytes, *, max_payload_bytes: int = 256 * 1024 * 1024, max_items: int | None = None, max_array_bytes: int = DEFAULT_MAX_ARRAY_BYTES, max_total_array_bytes: int = DEFAULT_MAX_TOTAL_ARRAY_BYTES) -> V2InferenceRequest:
    _validate_limits(max_payload_bytes, max_array_bytes, max_total_array_bytes)
    with _load_npz(payload, max_payload_bytes=max_payload_bytes, max_decompressed_bytes=max_total_array_bytes + 1024 * 1024) as archive:
        manifest = _read_manifest(archive)
        if manifest.get("schema_name") != V2_INFERENCE_REQUEST_SCHEMA or manifest.get("schema_version") != V2_INFERENCE_TRANSPORT_VERSION: raise V2InferenceTransportError("Unsupported V2 inference request schema or version")
        request_id, task = manifest.get("request_id"), manifest.get("task")
        if not isinstance(request_id, str) or not isinstance(task, str): raise V2InferenceTransportError("V2 request must include string request_id and task")
        _validate_task(task); raw_options = manifest.get("response_options") or {}
        if not isinstance(raw_options, dict): raise V2InferenceTransportError("response_options must be an object")
        raw_outputs = raw_options.get("outputs"); raw_budget = raw_options.get("max_output_bytes", DEFAULT_MAX_OUTPUT_BYTES)
        if raw_outputs is not None and (not isinstance(raw_outputs, list) or not all(isinstance(value, str) for value in raw_outputs)): raise V2InferenceTransportError("response_options.outputs must be an array of strings")
        if isinstance(raw_budget, bool) or not isinstance(raw_budget, int): raise V2InferenceTransportError("max_output_bytes must be an integer")
        rows = manifest.get("items")
        if not isinstance(rows, list) or not rows: raise V2InferenceTransportError("V2 inference request must contain at least one item")
        if max_items is not None and len(rows) > max_items: raise V2InferenceTransportError(f"Inference request exceeds the {max_items}-item limit")
        remaining = max_total_array_bytes; expected = {"manifest"}; items = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("inputs"), dict): raise V2InferenceTransportError("V2 item must contain an inputs object")
            inputs = {}
            for role, description in row["inputs"].items():
                if not isinstance(role, str) or not isinstance(description, dict): raise V2InferenceTransportError("V2 input descriptions must be objects")
                key = description.get("transport_key")
                if not isinstance(key, str) or not key: raise V2InferenceTransportError(f"Input {role!r} is missing transport_key")
                if key in expected: raise V2InferenceTransportError(f"NPZ array {key!r} is referenced more than once")
                expected.add(key); inputs[role], remaining = _checked_array(archive, key, description, max_array_bytes=max_array_bytes, remaining_total_bytes=remaining)
            source = row.get("source")
            if source is not None and not isinstance(source, dict): raise V2InferenceTransportError("V2 item source must be an object")
            try: item = InferenceItem(inputs=inputs, item_id=str(row["item_id"]), request_id=str(row["request_id"]), source=SourceReference(**source) if source else None, metadata=dict(row.get("metadata") or {}))
            except (KeyError, TypeError, ValueError) as exc: raise V2InferenceTransportError("V2 item is invalid") from exc
            if row.get("input_sha256") != item.input_sha256: raise V2InferenceTransportError("V2 item input_sha256 does not match its input assets")
            items.append(item)
        if set(archive.files) != expected: raise V2InferenceTransportError("NPZ payload contains unreferenced arrays")
        return V2InferenceRequest(request_id, task, items, V2ResponseOptions(frozenset(raw_outputs) if raw_outputs is not None else None, raw_budget))

def _validate_image_output(output: Any) -> None:
    if not isinstance(output, dict): raise V2InferenceTransportError("Image result output must be an object")
    missing = _IMAGE_REQUIRED_FIELDS - set(output)
    if missing: raise V2InferenceTransportError(f"Image result is missing required fields: {', '.join(sorted(missing))}")
    if output["schema_name"] != IMAGE_OUTPUT_SCHEMA or output["schema_version"] != IMAGE_OUTPUT_VERSION: raise V2InferenceTransportError("Unsupported image-result schema or version")
    if output["type"] != "image_analysis" or output["model_task"] not in {"classification", "embedding", "clustering"}: raise V2InferenceTransportError("Image result has an invalid model task")
    if output["secondary_heads"] is not None and not isinstance(output["secondary_heads"], list): raise V2InferenceTransportError("secondary_heads must be an array or null")

def decode_v2_inference_result_set(payload: bytes, *, max_payload_bytes: int = 256 * 1024 * 1024, max_array_bytes: int = DEFAULT_MAX_ARRAY_BYTES, max_total_array_bytes: int = DEFAULT_MAX_TOTAL_ARRAY_BYTES) -> dict[str, Any]:
    """Decode and integrity-check a V2 result archive, restoring arrays as ndarrays."""
    _validate_limits(max_payload_bytes, max_array_bytes, max_total_array_bytes)
    with _load_npz(payload, max_payload_bytes=max_payload_bytes, max_decompressed_bytes=max_total_array_bytes + 1024 * 1024) as archive:
        manifest = _read_manifest(archive)
        if manifest.get("schema_name") != V2_INFERENCE_RESULT_SET_SCHEMA or manifest.get("schema_version") != V2_INFERENCE_TRANSPORT_VERSION: raise V2InferenceTransportError("Unsupported V2 inference result schema or version")
        remaining = max_total_array_bytes; expected = {"manifest"}
        def restore(value: Any) -> Any:
            nonlocal remaining
            if isinstance(value, dict):
                key = value.get("transport_key")
                if key is not None:
                    if not isinstance(key, str) or key in expected: raise V2InferenceTransportError("V2 result array key is invalid or duplicated")
                    expected.add(key); payload_value, remaining = _checked_array(archive, key, value, max_array_bytes=max_array_bytes, remaining_total_bytes=remaining); return payload_value.values
                return {str(key): restore(item) for key, item in value.items()}
            if isinstance(value, list): return [restore(item) for item in value]
            return value
        restored = restore(manifest)
        if set(archive.files) != expected: raise V2InferenceTransportError("NPZ result contains unreferenced arrays")
        for row in restored.get("results", []):
            if not isinstance(row, dict): raise V2InferenceTransportError("V2 result entry must be an object")
            if row.get("status") == "ok" and (row.get("model") or {}).get("task") in {"classification", "embedding", "clustering"}: _validate_image_output(row.get("output"))
        return restored

__all__ = ["ArrayPayload", "DEFAULT_MAX_ARRAY_BYTES", "DEFAULT_MAX_OUTPUT_BYTES", "DEFAULT_MAX_TOTAL_ARRAY_BYTES", "InferenceItem", "SourceReference", "V2_INFERENCE_REQUEST_SCHEMA", "V2_INFERENCE_RESULT_SET_SCHEMA", "V2_INFERENCE_TRANSPORT_VERSION", "V2_NPZ_MEDIA_TYPE", "V2InferenceOutputLimitError", "V2InferenceRequest", "V2InferenceTransportError", "V2ResponseOptions", "decode_v2_inference_request", "decode_v2_inference_result_set", "encode_v2_inference_request"]
