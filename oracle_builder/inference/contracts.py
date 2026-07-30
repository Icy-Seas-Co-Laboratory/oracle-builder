from __future__ import annotations

import base64
import hashlib
import io
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np


INFERENCE_ITEM_SCHEMA = "oracle_builder.inference_item"
INFERENCE_RESULT_SCHEMA = "oracle_builder.inference_result"
INFERENCE_RESULT_SET_SCHEMA = "oracle_builder.inference_result_set"
INFERENCE_SCHEMA_VERSION = "1.0.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_uuid() -> str:
    return str(uuid.uuid4())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


@dataclass(frozen=True)
class ArrayPayload:
    """A typed in-memory array asset with stable identity and content hash."""

    values: np.ndarray
    asset_id: str = field(default_factory=new_uuid)
    media_type: str = "application/x-npy"
    sha256: str | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "asset_id", str(uuid.UUID(self.asset_id)))
        actual_hash = sha256_bytes(_array_bytes(values))
        if self.sha256 is not None and self.sha256 != actual_hash:
            raise ValueError("Array payload SHA-256 does not match its values")
        object.__setattr__(self, "sha256", actual_hash)

    @property
    def shape(self) -> list[int]:
        return [int(value) for value in self.values.shape]

    @property
    def dtype(self) -> str:
        return str(self.values.dtype)

    def to_dict(self, *, include_data: bool = True) -> dict[str, Any]:
        result = {
            "asset_id": self.asset_id,
            "media_type": self.media_type,
            "shape": self.shape,
            "dtype": self.dtype,
            "sha256": self.sha256,
            "encoding": "npy",
        }
        if include_data:
            result["data_base64"] = base64.b64encode(
                _array_bytes(self.values)
            ).decode("ascii")
        return result


@dataclass(frozen=True)
class SourceReference:
    system: str
    resource_type: str
    resource_id: str
    revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "system": self.system,
                "resource_type": self.resource_type,
                "resource_id": self.resource_id,
                "revision": self.revision,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class InferenceItem:
    """Storage-neutral input accepted by every Oracle Builder connector."""

    inputs: dict[str, ArrayPayload]
    item_id: str = field(default_factory=new_uuid)
    request_id: str = field(default_factory=new_uuid)
    source: SourceReference | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", str(uuid.UUID(self.item_id)))
        if not self.request_id:
            raise ValueError("request_id cannot be empty")
        if "image" not in self.inputs:
            raise ValueError("InferenceItem.inputs must contain an 'image' payload")

    @classmethod
    def from_array(
        cls,
        image: Any,
        *,
        candidate_mask: Any | None = None,
        item_id: str | None = None,
        request_id: str | None = None,
        source: SourceReference | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "InferenceItem":
        inputs = {"image": ArrayPayload(np.asarray(image))}
        if candidate_mask is not None:
            inputs["candidate_mask"] = ArrayPayload(np.asarray(candidate_mask))
        return cls(
            inputs=inputs,
            item_id=item_id or new_uuid(),
            request_id=request_id or new_uuid(),
            source=source,
            metadata=dict(metadata or {}),
        )

    @property
    def input_sha256(self) -> str:
        digest = hashlib.sha256()
        for role, payload in sorted(self.inputs.items()):
            digest.update(role.encode("utf-8"))
            digest.update(bytes.fromhex(payload.sha256 or ""))
        return digest.hexdigest()

    def to_dict(self, *, include_data: bool = True) -> dict[str, Any]:
        return {
            "schema_name": INFERENCE_ITEM_SCHEMA,
            "schema_version": INFERENCE_SCHEMA_VERSION,
            "item_id": self.item_id,
            "request_id": self.request_id,
            "source": self.source.to_dict() if self.source else None,
            "input_sha256": self.input_sha256,
            "inputs": {
                role: value.to_dict(include_data=include_data)
                for role, value in self.inputs.items()
            },
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ModelReference:
    artifact_id: str
    run_id: str
    task: str
    architecture: str
    artifact_fingerprint: str | None = None
    contract_version: str = INFERENCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "artifact_id": self.artifact_id,
                "artifact_fingerprint": self.artifact_fingerprint,
                "run_id": self.run_id,
                "task": self.task,
                "architecture": self.architecture,
                "contract_version": self.contract_version,
            }.items()
            if value is not None
        }


def _json_value(value: Any, *, include_array_data: bool) -> Any:
    if isinstance(value, ArrayPayload):
        return value.to_dict(include_data=include_array_data)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return ArrayPayload(value).to_dict(include_data=include_array_data)
    if isinstance(value, dict):
        return {
            str(key): _json_value(item, include_array_data=include_array_data)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, include_array_data=include_array_data)
            for item in value
        ]
    return value


@dataclass
class InferenceResult:
    request_id: str
    item_id: str
    model: ModelReference
    output: dict[str, Any] | None
    input_sha256: str
    result_set_id: str
    source: SourceReference | None = None
    result_id: str = field(default_factory=new_uuid)
    sequence_number: int | None = None
    status: str = "ok"
    received_at: str = field(default_factory=utc_now)
    completed_at: str = field(default_factory=utc_now)
    duration_ms: float | None = None
    warnings: list[dict[str, Any] | str] = field(default_factory=list)
    error: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.result_id = str(uuid.UUID(self.result_id))
        self.result_set_id = str(uuid.UUID(self.result_set_id))
        if self.status not in {"ok", "rejected", "failed"}:
            raise ValueError(f"Unsupported inference status: {self.status}")
        if self.status == "ok" and self.output is None:
            raise ValueError("Successful inference results require an output")

    def to_dict(self, *, include_array_data: bool = True) -> dict[str, Any]:
        return {
            "schema_name": INFERENCE_RESULT_SCHEMA,
            "schema_version": INFERENCE_SCHEMA_VERSION,
            "result_id": self.result_id,
            "result_set_id": self.result_set_id,
            "request_id": self.request_id,
            "item_id": self.item_id,
            "sequence_number": self.sequence_number,
            "source": self.source.to_dict() if self.source else None,
            "input_sha256": self.input_sha256,
            "model": self.model.to_dict(),
            "status": self.status,
            "output": _json_value(
                self.output, include_array_data=include_array_data
            ),
            "execution": {
                "received_at": self.received_at,
                "completed_at": self.completed_at,
                "duration_ms": self.duration_ms,
            },
            "warnings": self.warnings,
            "error": self.error,
        }

    def to_json(self, *, include_array_data: bool = True) -> str:
        return json.dumps(
            self.to_dict(include_array_data=include_array_data),
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass
class InferenceResultSet:
    model: ModelReference
    result_set_id: str = field(default_factory=new_uuid)
    results: list[InferenceResult] = field(default_factory=list)
    source_dataset: dict[str, Any] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=utc_now)
    completed_at: str | None = None

    def __post_init__(self) -> None:
        self.result_set_id = str(uuid.UUID(self.result_set_id))

    def append(self, result: InferenceResult) -> None:
        if result.result_set_id != self.result_set_id:
            raise ValueError("Result belongs to a different result set")
        if result.sequence_number is None:
            result.sequence_number = len(self.results)
        self.results.append(result)

    def complete(self) -> "InferenceResultSet":
        self.completed_at = utc_now()
        return self

    @property
    def counts(self) -> dict[str, int]:
        return {
            "requested": len(self.results),
            "succeeded": sum(row.status == "ok" for row in self.results),
            "rejected": sum(row.status == "rejected" for row in self.results),
            "failed": sum(row.status == "failed" for row in self.results),
        }

    def to_dict(self, *, include_array_data: bool = True) -> dict[str, Any]:
        return {
            "schema_name": INFERENCE_RESULT_SET_SCHEMA,
            "schema_version": INFERENCE_SCHEMA_VERSION,
            "result_set_id": self.result_set_id,
            "model": self.model.to_dict(),
            "source_dataset": self.source_dataset,
            "parameters": self.parameters,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "counts": self.counts,
            "results": [
                result.to_dict(include_array_data=include_array_data)
                for result in self.results
            ],
        }

    def to_json_lines(self, *, include_array_data: bool = True) -> Iterable[str]:
        yield json.dumps(
            {
                "event": "result_set_start",
                "schema_version": INFERENCE_SCHEMA_VERSION,
                "result_set_id": self.result_set_id,
                "model": self.model.to_dict(),
                "started_at": self.started_at,
            },
            sort_keys=True,
        )
        for result in self.results:
            yield json.dumps(
                {
                    "event": "result",
                    "value": result.to_dict(
                        include_array_data=include_array_data
                    ),
                },
                sort_keys=True,
            )
        yield json.dumps(
            {
                "event": "result_set_complete",
                "result_set_id": self.result_set_id,
                "completed_at": self.completed_at,
                "counts": self.counts,
            },
            sort_keys=True,
        )
