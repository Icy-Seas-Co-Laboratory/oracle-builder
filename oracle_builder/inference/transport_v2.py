"""Version 2 binary transport for interactive resident inference.

The contract deliberately keeps the transport independent from HTTP and WebSocket
framing.  A unary request body and each frame on a stream therefore use exactly
the same NPZ envelope, preserving item correlation and content verification.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import numpy as np
from oracle_data_contracts.inference_v2 import (
    V2InferenceTransportError as _SharedV2InferenceTransportError,
    decode_v2_inference_request as _decode_shared_v2_inference_request,
    encode_v2_inference_request as _encode_shared_v2_inference_request,
)

from oracle_builder.inference.contracts import (
    ArrayPayload,
    InferenceItem,
    InferenceResult,
    InferenceResultSet,
    SourceReference,
)
from oracle_builder.inference.image_contract_v2 import IMAGE_SELECTABLE_FIELDS, image_output, present_image_fields, validate_image_output_shape


V2_INFERENCE_REQUEST_SCHEMA = "oracle_builder.inference.v2.request"
V2_INFERENCE_RESULT_SET_SCHEMA = "oracle_builder.inference.v2.result_set"
V2_INFERENCE_TRANSPORT_VERSION = "2.0.0"
V2_NPZ_MEDIA_TYPE = "application/vnd.oracle-builder.inference.v2+npz"

InferenceTask = Literal["classification", "mask_refinement"]
DEFAULT_MAX_ARRAY_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_TOTAL_ARRAY_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_CLIENT_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_RESULT_ARRAY_BYTES = 256 * 1024 * 1024

_INPUTS: dict[str, frozenset[str]] = {
    "classification": frozenset({"image"}),
    "mask_refinement": frozenset({"image", "candidate_mask"}),
}
_OUTPUTS: dict[str, frozenset[str]] = {
    "classification": IMAGE_SELECTABLE_FIELDS,
    "mask_refinement": frozenset({
        "mask", "probability_map", "logits", "delta_mask", "delta_probability_map",
    }),
}
_DEFAULT_OUTPUTS: dict[str, frozenset[str]] = {
    "classification": IMAGE_SELECTABLE_FIELDS,
    "mask_refinement": frozenset({"mask"}),
}


class V2InferenceTransportError(ValueError):
    """A payload or selection does not meet the V2 inference contract."""


class V2InferenceOutputLimitError(V2InferenceTransportError):
    """A requested response would exceed its declared output budget."""


class V2InferenceContractError(RuntimeError):
    """A served model violated an advertised V2 output contract."""


@dataclass(frozen=True)
class V2ResponseOptions:
    """Requested result fields and a per-item binary-output budget.

    Image results always retain their eight stable scientific keys. Selection
    controls which supported fields are populated. Dense mask maps are opt-in.
    """

    outputs: frozenset[str] | None = None
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES

    def resolved_outputs(self, task: InferenceTask) -> frozenset[str]:
        _validate_task(task)
        requested = self.outputs if self.outputs is not None else _DEFAULT_OUTPUTS[task]
        unsupported = requested - _OUTPUTS[task]
        if unsupported:
            raise V2InferenceTransportError(
                f"Unsupported {task} outputs: {', '.join(sorted(unsupported))}"
            )
        if not 1 <= self.max_output_bytes <= MAX_CLIENT_OUTPUT_BYTES:
            raise V2InferenceTransportError(f"max_output_bytes must be between 1 and {MAX_CLIENT_OUTPUT_BYTES}")
        return requested

    def to_dict(self, task: InferenceTask) -> dict[str, Any]:
        return {
            "outputs": sorted(self.resolved_outputs(task)) if self.outputs is not None or task == "mask_refinement" else None,
            "max_output_bytes": self.max_output_bytes,
        }


@dataclass(frozen=True)
class V2InferenceRequest:
    """One bounded batch or stream message for a single task and model."""

    request_id: str
    task: InferenceTask
    items: list[InferenceItem]
    response_options: V2ResponseOptions = field(default_factory=V2ResponseOptions)

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise V2InferenceTransportError("request_id must be a non-empty string")
        _validate_task(self.task)
        if not self.items:
            raise V2InferenceTransportError("V2 inference request must contain at least one item")
        if len({item.item_id for item in self.items}) != len(self.items):
            raise V2InferenceTransportError("item_id values must be unique within a request")
        for item in self.items:
            if item.request_id != self.request_id:
                raise V2InferenceTransportError("Every item.request_id must equal request_id")
            extra = set(item.inputs) - _INPUTS[self.task]
            if extra:
                raise V2InferenceTransportError(
                    f"Unsupported {self.task} inputs: {', '.join(sorted(extra))}"
                )
            if "image" not in item.inputs:
                raise V2InferenceTransportError("Every inference item requires an image input")
            _json_object(item.metadata, "item metadata")
        self.response_options.resolved_outputs(self.task)


def _validate_task(task: str) -> None:
    if task not in _INPUTS:
        raise V2InferenceTransportError(
            "task must be one of: classification, mask_refinement"
        )


def _json_object(value: Any, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise V2InferenceTransportError(f"{label} must be JSON serializable") from exc


def _manifest_array(value: dict[str, Any]) -> np.ndarray:
    _json_object(value, "manifest")
    return np.frombuffer(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"),
        dtype=np.uint8,
    )


def _read_manifest(archive: Any) -> dict[str, Any]:
    if "manifest" not in archive.files:
        raise V2InferenceTransportError("NPZ payload is missing its manifest")
    try:
        value = json.loads(np.asarray(archive["manifest"], dtype=np.uint8).tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise V2InferenceTransportError("NPZ manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise V2InferenceTransportError("NPZ manifest must be a JSON object")
    return value


def _load_npz(payload: bytes, *, max_payload_bytes: int, max_decompressed_bytes: int) -> Any:
    if not payload:
        raise V2InferenceTransportError("Inference payload is empty")
    if len(payload) > max_payload_bytes:
        raise V2InferenceTransportError(f"Inference payload exceeds the {max_payload_bytes}-byte limit")
    # Check ZIP member sizes before NumPy materializes an array.  Without this
    # guard, a small compressed request could expand to exhaust worker memory.
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
            names = [entry.filename for entry in bundle.infolist()]
            if len(names) != len(set(names)) or any("/" in name or "\\" in name for name in names):
                raise V2InferenceTransportError("NPZ payload has invalid member names")
            if sum(entry.file_size for entry in bundle.infolist()) > max_decompressed_bytes:
                raise V2InferenceTransportError("NPZ payload exceeds the decompressed byte limit")
    except V2InferenceTransportError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise V2InferenceTransportError("Inference payload is not a valid NPZ archive") from exc
    try:
        return np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc:
        raise V2InferenceTransportError("Inference payload is not a valid safe NPZ archive") from exc


def _checked_array(
    archive: Any,
    key: str,
    description: dict[str, Any],
    *,
    max_array_bytes: int,
    remaining_total_bytes: int,
) -> tuple[ArrayPayload, int]:
    if key not in archive.files:
        raise V2InferenceTransportError(f"NPZ payload is missing array {key!r}")
    try:
        values = np.asarray(archive[key])
    except Exception as exc:
        raise V2InferenceTransportError(f"Unable to read NPZ array {key!r}") from exc
    if values.dtype.hasobject or values.dtype.fields:
        raise V2InferenceTransportError(f"NPZ array {key!r} has an unsafe dtype")
    if values.nbytes > max_array_bytes:
        raise V2InferenceTransportError(f"NPZ array {key!r} exceeds the per-array byte limit")
    if values.nbytes > remaining_total_bytes:
        raise V2InferenceTransportError("NPZ arrays exceed the decompressed byte limit")
    if description.get("shape") != [int(value) for value in values.shape]:
        raise V2InferenceTransportError(f"NPZ array {key!r} shape does not match its manifest")
    if description.get("dtype") != str(values.dtype):
        raise V2InferenceTransportError(f"NPZ array {key!r} dtype does not match its manifest")
    try:
        payload = ArrayPayload(
            values,
            asset_id=str(description["asset_id"]),
            sha256=str(description["sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise V2InferenceTransportError(f"NPZ array {key!r} failed identity verification") from exc
    return payload, remaining_total_bytes - values.nbytes


def _array_description(payload: ArrayPayload, key: str) -> dict[str, Any]:
    return {
        "transport_key": key,
        "asset_id": payload.asset_id,
        "sha256": payload.sha256,
        "shape": payload.shape,
        "dtype": payload.dtype,
        "media_type": payload.media_type,
    }


def encode_v2_inference_request(request: V2InferenceRequest) -> bytes:
    """Encode through the dependency-light public transport implementation."""
    try:
        return _encode_shared_v2_inference_request(request)  # type: ignore[arg-type]
    except _SharedV2InferenceTransportError as exc:
        raise V2InferenceTransportError(str(exc)) from exc


def decode_v2_inference_request(
    payload: bytes,
    *,
    max_payload_bytes: int = 256 * 1024 * 1024,
    max_items: int | None = None,
    max_array_bytes: int = DEFAULT_MAX_ARRAY_BYTES,
    max_total_array_bytes: int = DEFAULT_MAX_TOTAL_ARRAY_BYTES,
) -> V2InferenceRequest:
    """Decode through the dependency-light public transport implementation."""
    try:
        return _decode_shared_v2_inference_request(
            payload,
            max_payload_bytes=max_payload_bytes,
            max_items=max_items,
            max_array_bytes=max_array_bytes,
            max_total_array_bytes=max_total_array_bytes,
        )  # type: ignore[return-value]
    except _SharedV2InferenceTransportError as exc:
        raise V2InferenceTransportError(str(exc)) from exc


def _output_array_bytes(value: Any) -> int:
    if isinstance(value, ArrayPayload):
        return int(value.values.nbytes)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_output_array_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_output_array_bytes(item) for item in value)
    return 0


def project_v2_output(
    output: dict[str, Any] | None, options: V2ResponseOptions, task: InferenceTask,
    *, model_task: str | None = None, artifact_id: str | None = None,
    supported_outputs: frozenset[str] | None = None,
    default_outputs: frozenset[str] | None = None,
) -> dict[str, Any] | None:
    """Return only requested result fields and enforce the per-item array budget."""
    if output is None:
        return None
    requested = options.resolved_outputs(task)
    if task == "classification":
        supported = supported_outputs if supported_outputs is not None else present_image_fields(output, model_task or str(output.get("type") or "classification"))
        selected = options.outputs if options.outputs is not None else (default_outputs or supported)
        invalid = selected - supported
        if invalid:
            raise V2InferenceTransportError(f"Requested output is unsupported by this artifact: {', '.join(sorted(invalid))}")
        try:
            projected_image = image_output(
                output, model_task=model_task or str(output.get("type") or "classification"),
                artifact_id=artifact_id or "unknown", selected=selected, supported=supported,
            )
        except ValueError as exc:
            raise V2InferenceContractError(str(exc)) from exc
        actual_bytes = _output_array_bytes(projected_image)
        if actual_bytes > options.max_output_bytes:
            raise V2InferenceOutputLimitError(
                f"Selected output for one item is {actual_bytes} bytes, exceeding max_output_bytes={options.max_output_bytes}"
            )
        try:
            validate_image_output_shape(projected_image)
        except ValueError as exc:
            raise V2InferenceContractError(str(exc)) from exc
        return projected_image
    projected: dict[str, Any] = {"type": output.get("type", task)}
    # These are small interpretation fields required to use a refinement mask.
    if task == "mask_refinement":
        for key in ("target_mode", "threshold", "logits_source", "transform"):
            if key in output:
                projected[key] = output[key]
    for key in requested:
        source = key
        if task == "mask_refinement":
            candidate_delta = output.get("target_mode") == "candidate_delta"
            source = {
                "mask": "reconstructed_mask" if candidate_delta else "mask",
                "probability_map": "reconstructed_probability_map" if candidate_delta else "probability_map",
                "delta_mask": "mask" if candidate_delta else "delta_mask",
                "delta_probability_map": "probability_map" if candidate_delta else "delta_probability_map",
            }.get(key, key)
        if source in output:
            projected[key] = output[source]
        else:
            raise V2InferenceTransportError(f"Requested output {key!r} is unavailable for this model")
    actual_bytes = _output_array_bytes(projected)
    if actual_bytes > options.max_output_bytes:
        raise V2InferenceOutputLimitError(
            f"Selected output for one item is {actual_bytes} bytes, exceeding max_output_bytes={options.max_output_bytes}"
        )
    return projected


def project_v2_result_set(
    result_set: InferenceResultSet,
    response_options: V2ResponseOptions,
    task: InferenceTask,
    *, supported_outputs: frozenset[str] | None = None,
    default_outputs: frozenset[str] | None = None,
) -> InferenceResultSet:
    """Copy a result set with task-specific response projection applied."""
    projected = InferenceResultSet(
        model=result_set.model,
        result_set_id=result_set.result_set_id,
        source_dataset=result_set.source_dataset,
        parameters=dict(result_set.parameters),
        started_at=result_set.started_at,
        completed_at=result_set.completed_at,
        execution=dict(result_set.execution),
    )
    for result in result_set.results:
        projected.append(InferenceResult(
            request_id=result.request_id,
            item_id=result.item_id,
            model=result.model,
            output=project_v2_output(
                result.output, response_options, task,
                model_task=result_set.model.task,
                artifact_id=result_set.model.artifact_id,
                supported_outputs=supported_outputs,
                default_outputs=default_outputs,
            ) if result.status == "ok" else None,
            input_sha256=result.input_sha256,
            result_set_id=projected.result_set_id,
            source=result.source,
            result_id=result.result_id,
            sequence_number=result.sequence_number,
            status=result.status,
            received_at=result.received_at,
            completed_at=result.completed_at,
            duration_ms=result.duration_ms,
            warnings=list(result.warnings),
            error=dict(result.error) if result.error else None,
        ))
    total_bytes = sum(_output_array_bytes(result.output) for result in projected.results)
    if total_bytes > MAX_RESULT_ARRAY_BYTES:
        raise V2InferenceOutputLimitError(
            f"Selected response arrays exceed the {MAX_RESULT_ARRAY_BYTES}-byte aggregate limit"
        )
    return projected


def _export_arrays(value: Any, arrays: dict[str, np.ndarray], prefix: str) -> Any:
    if isinstance(value, ArrayPayload):
        key = f"output_{len(arrays)}_{prefix}"
        arrays[key] = np.asarray(value.values)
        return _array_description(value, key)
    if isinstance(value, np.ndarray):
        return _export_arrays(ArrayPayload(value), arrays, prefix)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _export_arrays(item, arrays, f"{prefix}_{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_export_arrays(item, arrays, f"{prefix}_{index}") for index, item in enumerate(value)]
    return value


def encode_v2_inference_result_set(result_set: InferenceResultSet) -> bytes:
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for result in result_set.results:
        row = result.to_dict(include_array_data=False)
        row["output"] = _export_arrays(result.output, arrays, f"item_{result.item_id}")
        rows.append(row)
    manifest = result_set.to_dict(include_array_data=False)
    manifest.update({
        "schema_name": V2_INFERENCE_RESULT_SET_SCHEMA,
        "schema_version": V2_INFERENCE_TRANSPORT_VERSION,
        "results": rows,
    })
    return _encode_npz(manifest, arrays)


def decode_v2_inference_result_set(
    payload: bytes,
    *,
    max_payload_bytes: int = 256 * 1024 * 1024,
    max_array_bytes: int = DEFAULT_MAX_ARRAY_BYTES,
    max_total_array_bytes: int = DEFAULT_MAX_TOTAL_ARRAY_BYTES,
) -> dict[str, Any]:
    """Decode a V2 result archive, including validation of every array hash."""
    _validate_limits(max_payload_bytes, max_array_bytes, max_total_array_bytes)
    with _load_npz(
        payload, max_payload_bytes=max_payload_bytes,
        max_decompressed_bytes=max_total_array_bytes + 1024 * 1024,
    ) as archive:
        manifest = _read_manifest(archive)
        if manifest.get("schema_name") != V2_INFERENCE_RESULT_SET_SCHEMA:
            raise V2InferenceTransportError("Unsupported V2 inference result schema")
        if manifest.get("schema_version") != V2_INFERENCE_TRANSPORT_VERSION:
            raise V2InferenceTransportError("Unsupported V2 inference result schema version")
        remaining = max_total_array_bytes
        expected_keys = {"manifest"}

        def restore(value: Any) -> Any:
            nonlocal remaining
            if isinstance(value, dict):
                key = value.get("transport_key")
                if key is not None:
                    if not isinstance(key, str) or key in expected_keys:
                        raise V2InferenceTransportError("V2 result array key is invalid or duplicated")
                    expected_keys.add(key)
                    payload_value, remaining = _checked_array(
                        archive, key, value, max_array_bytes=max_array_bytes,
                        remaining_total_bytes=remaining,
                    )
                    return payload_value.values
                return {str(key): restore(item) for key, item in value.items()}
            if isinstance(value, list):
                return [restore(item) for item in value]
            return value

        restored = restore(manifest)
        if set(archive.files) != expected_keys:
            raise V2InferenceTransportError("NPZ result contains unreferenced arrays")
        for row in restored.get("results", []):
            if not isinstance(row, dict):
                raise V2InferenceTransportError("V2 result entry must be an object")
            model = row.get("model") or {}
            if row.get("status") == "ok" and model.get("task") in {"classification", "embedding", "clustering"}:
                try:
                    validate_image_output_shape(row.get("output"))
                except ValueError as exc:
                    raise V2InferenceTransportError(str(exc)) from exc
        return restored


def _encode_npz(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, manifest=_manifest_array(manifest), **arrays)
    return buffer.getvalue()


def _validate_limits(*limits: int) -> None:
    if any(not isinstance(value, int) or value < 1 for value in limits):
        raise ValueError("Transport byte limits must be positive integers")


__all__ = [
    "DEFAULT_MAX_ARRAY_BYTES",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_MAX_TOTAL_ARRAY_BYTES",
    "V2_INFERENCE_REQUEST_SCHEMA",
    "V2_INFERENCE_RESULT_SET_SCHEMA",
    "V2_INFERENCE_TRANSPORT_VERSION",
    "V2_NPZ_MEDIA_TYPE",
    "V2InferenceOutputLimitError",
    "V2InferenceContractError",
    "V2InferenceRequest",
    "V2InferenceTransportError",
    "V2ResponseOptions",
    "decode_v2_inference_request",
    "decode_v2_inference_result_set",
    "encode_v2_inference_request",
    "encode_v2_inference_result_set",
    "project_v2_output",
    "project_v2_result_set",
]
