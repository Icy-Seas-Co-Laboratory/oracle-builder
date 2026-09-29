"""Versioned, storage-neutral execution contracts for Oracle workers.

Work units deliberately identify inputs and staging destinations by immutable
artifact references.  A host path is an implementation detail of a local
artifact-store adapter and must not be placed in this durable contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
import json
import math
import re
import uuid
from typing import Any, Mapping

from oracle_data_contracts.artifacts.ref import ArtifactRef


WORK_UNIT_SCHEMA_NAME = "oracle_work_unit"
WORK_UNIT_SCHEMA_VERSION = 1
WORK_UNIT_V2_SCHEMA_VERSION = 2
WORK_UNIT_ACTIONS = frozenset({"train", "infer", "evaluate", "model_ingest", "run_validate", "run_pack"})
WORK_UNIT_V2_PHASES = frozenset({"verify", "initialize", "train", "evaluate", "finalize", "infer"})
EXECUTION_ENVELOPE_SCHEMA_NAME = "oracle_work_unit_execution"
EXECUTION_ENVELOPE_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class WorkUnitError(ValueError):
    """A work unit is malformed or incompatible with this worker contract."""


def _require_uuid(value: str, label: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise WorkUnitError(f"{label} must be a UUID") from exc


def _json_object(value: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise WorkUnitError(f"{label} must be an object")
    result = dict(value)
    try:
        json.dumps(result, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise WorkUnitError(f"{label} must be JSON serializable") from exc
    return result


def _strict_json_value(value: Any, label: str) -> Any:
    """Return strict JSON data, refusing non-finite or implicit values.

    ``json.dumps`` normally writes NaN/Infinity even though they are not JSON.
    V2 descriptors are cross-language content identities, so accepting either
    would give different consumers different meanings for the same hash.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WorkUnitError(f"{label} cannot contain NaN or Infinity")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise WorkUnitError(f"{label} object keys must be strings")
            result[key] = _strict_json_value(item, f"{label}.{key}")
        return result
    if isinstance(value, list):
        return [_strict_json_value(item, f"{label}[]") for item in value]
    raise WorkUnitError(f"{label} must contain only JSON values")


def _strict_json_object(value: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise WorkUnitError(f"{label} must be an object")
    result = _strict_json_value(value, label)
    assert isinstance(result, dict)
    # This is deliberately retained after recursive validation as a defence
    # against a future change accidentally enabling non-standard JSON output.
    try:
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise WorkUnitError(f"{label} must be strict JSON serializable") from exc
    return result


def _require_exact_fields(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise WorkUnitError(f"{label} contains unknown fields: {', '.join(unknown)}")
    if missing:
        raise WorkUnitError(f"{label} is missing fields: {', '.join(missing)}")


def _require_pinned_ref(value: Any, label: str) -> ArtifactRef:
    if not isinstance(value, ArtifactRef):
        raise WorkUnitError(f"{label} must be an ArtifactRef")
    if value.fingerprint_sha256 is None:
        raise WorkUnitError(f"{label} must include a content fingerprint_sha256")
    return value


def _parse_pinned_ref(value: Any, label: str) -> ArtifactRef:
    if not isinstance(value, Mapping):
        raise WorkUnitError(f"{label} must be an artifact reference object")
    try:
        return _require_pinned_ref(ArtifactRef.from_dict(value), label)
    except (TypeError, ValueError) as exc:
        raise WorkUnitError(f"Invalid {label}: {exc}") from exc


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkUnitError(f"{label} must be a non-negative integer")
    return value


def _require_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise WorkUnitError(f"{label} must be an ISO-8601 timestamp")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise WorkUnitError(f"{label} must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None:
        raise WorkUnitError(f"{label} must include a timezone")
    return value


@dataclass(frozen=True)
class WorkUnit:
    """A sealed, storage-neutral instruction for exactly one worker attempt."""

    work_unit_id: str
    attempt_id: str
    specification_id: str
    action: str
    inputs: dict[str, ArtifactRef]
    configuration: ArtifactRef | None
    staging: ArtifactRef
    resources: dict[str, Any]
    # Parameters are intentionally a bounded JSON object selected by the
    # action contract.  They are not paths, commands, or environment values.
    # Keeping the field optional on the v1 wire format preserves existing
    # sealed training work units while allowing infer to select a split and a
    # stable prediction-set label.
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "work_unit_id", _require_uuid(self.work_unit_id, "work_unit_id"))
        object.__setattr__(self, "attempt_id", _require_uuid(self.attempt_id, "attempt_id"))
        object.__setattr__(self, "specification_id", _require_uuid(self.specification_id, "specification_id"))
        if self.action not in WORK_UNIT_ACTIONS:
            raise WorkUnitError(f"Unsupported work-unit action: {self.action}")
        if not isinstance(self.inputs, dict) or not self.inputs:
            raise WorkUnitError("inputs must contain at least one named artifact reference")
        if any(not isinstance(name, str) or not name.strip() or not isinstance(ref, ArtifactRef) for name, ref in self.inputs.items()):
            raise WorkUnitError("inputs must map non-empty names to ArtifactRef values")
        if self.configuration is not None and not isinstance(self.configuration, ArtifactRef):
            raise WorkUnitError("configuration must be an ArtifactRef or null")
        if not isinstance(self.staging, ArtifactRef) or self.staging.kind != "staging":
            raise WorkUnitError("staging must be a staging ArtifactRef")
        object.__setattr__(self, "resources", _json_object(self.resources, "resources"))
        object.__setattr__(self, "parameters", _json_object(self.parameters, "parameters"))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema": {"name": WORK_UNIT_SCHEMA_NAME, "version": WORK_UNIT_SCHEMA_VERSION},
            "work_unit_id": self.work_unit_id,
            "attempt_id": self.attempt_id,
            "specification_id": self.specification_id,
            "action": self.action,
            "inputs": {name: ref.to_dict() for name, ref in sorted(self.inputs.items())},
            "configuration": self.configuration.to_dict() if self.configuration else None,
            "staging": self.staging.to_dict(),
            "resources": self.resources,
        }
        # Preserve the original v1 wire representation for existing training
        # units; inference is the first action that needs this bounded field.
        if self.parameters:
            result["parameters"] = self.parameters
        return result

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkUnit":
        if not isinstance(value, Mapping):
            raise WorkUnitError("work unit must be an object")
        schema = value.get("schema")
        if not isinstance(schema, Mapping) or schema.get("name") != WORK_UNIT_SCHEMA_NAME or schema.get("version") != WORK_UNIT_SCHEMA_VERSION:
            raise WorkUnitError(f"Expected {WORK_UNIT_SCHEMA_NAME} schema version {WORK_UNIT_SCHEMA_VERSION}")
        raw_inputs = value.get("inputs")
        if not isinstance(raw_inputs, Mapping):
            raise WorkUnitError("inputs must be an object")
        try:
            inputs = {str(name): ArtifactRef.from_dict(ref) for name, ref in raw_inputs.items()}
            configuration = ArtifactRef.from_dict(value["configuration"]) if value.get("configuration") is not None else None
            staging = ArtifactRef.from_dict(value["staging"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkUnitError(f"Invalid artifact reference: {exc}") from exc
        return cls(
            work_unit_id=str(value.get("work_unit_id", "")),
            attempt_id=str(value.get("attempt_id", "")),
            specification_id=str(value.get("specification_id", "")),
            action=str(value.get("action", "")),
            inputs=inputs,
            configuration=configuration,
            staging=staging,
            resources=_json_object(value.get("resources"), "resources"),
            parameters=_json_object(value.get("parameters"), "parameters"),
        )


@dataclass(frozen=True)
class WorkUnitV2:
    """Immutable, attempt-free description of one bounded run advancement.

    A V2 unit is deliberately reusable after a worker failure.  Attempt and
    lease authority belong to :class:`ExecutionEnvelope`, which is attached
    only at dispatch and is never included in this unit's digest.
    """

    run_id: str
    work_unit_id: str
    sequence: int
    phase: str
    predecessor_work_unit_id: str | None
    inputs: dict[str, ArtifactRef]
    configuration: ArtifactRef
    resources: dict[str, Any]
    parameters: dict[str, Any]
    start_cursor: dict[str, Any]
    stop_boundary: dict[str, Any]
    output_contract: dict[str, Any]
    compatibility: dict[str, Any]

    @property
    def action(self) -> str:
        """Compatibility action used by the existing worker admission lane.

        V2 phase owns the scientific lifecycle.  Until the scheduler gains
        phase-native admission, all non-inference phases execute through the
        existing ``train`` capability and inference uses ``infer``.
        """
        return "infer" if self.phase == "infer" else "train"

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_uuid(self.run_id, "run_id"))
        object.__setattr__(self, "work_unit_id", _require_uuid(self.work_unit_id, "work_unit_id"))
        if self.predecessor_work_unit_id is not None:
            object.__setattr__(
                self, "predecessor_work_unit_id",
                _require_uuid(self.predecessor_work_unit_id, "predecessor_work_unit_id"),
            )
        object.__setattr__(self, "sequence", _require_nonnegative_int(self.sequence, "sequence"))
        if self.phase not in WORK_UNIT_V2_PHASES:
            raise WorkUnitError(f"Unsupported V2 work-unit phase: {self.phase}")
        if not isinstance(self.inputs, dict) or not self.inputs:
            raise WorkUnitError("inputs must contain at least one named pinned ArtifactRef")
        inputs: dict[str, ArtifactRef] = {}
        for name, ref in self.inputs.items():
            if not isinstance(name, str) or not name.strip():
                raise WorkUnitError("inputs must map non-empty names to ArtifactRef values")
            inputs[name] = _require_pinned_ref(ref, f"inputs.{name}")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "configuration", _require_pinned_ref(self.configuration, "configuration"))
        for field_name in (
            "resources", "parameters", "start_cursor", "stop_boundary", "output_contract", "compatibility",
        ):
            object.__setattr__(self, field_name, _strict_json_object(getattr(self, field_name), field_name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": {"name": WORK_UNIT_SCHEMA_NAME, "version": WORK_UNIT_V2_SCHEMA_VERSION},
            "run_id": self.run_id,
            "work_unit_id": self.work_unit_id,
            "sequence": self.sequence,
            "phase": self.phase,
            "action": self.action,
            "predecessor_work_unit_id": self.predecessor_work_unit_id,
            "inputs": {name: ref.to_dict() for name, ref in sorted(self.inputs.items())},
            "configuration": self.configuration.to_dict(),
            "resources": self.resources,
            "parameters": self.parameters,
            "start_cursor": self.start_cursor,
            "stop_boundary": self.stop_boundary,
            "output_contract": self.output_contract,
            "compatibility": self.compatibility,
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def sha256(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkUnitV2":
        if not isinstance(value, Mapping):
            raise WorkUnitError("work unit must be an object")
        fields = frozenset({
            "schema", "run_id", "work_unit_id", "sequence", "phase", "predecessor_work_unit_id",
            "action", "inputs", "configuration", "resources", "parameters", "start_cursor", "stop_boundary",
            "output_contract", "compatibility",
        })
        _require_exact_fields(value, fields, "V2 work unit")
        schema = value["schema"]
        if not isinstance(schema, Mapping) or dict(schema) != {"name": WORK_UNIT_SCHEMA_NAME, "version": WORK_UNIT_V2_SCHEMA_VERSION}:
            raise WorkUnitError(f"Expected {WORK_UNIT_SCHEMA_NAME} schema version {WORK_UNIT_V2_SCHEMA_VERSION}")
        raw_inputs = value["inputs"]
        if not isinstance(raw_inputs, Mapping):
            raise WorkUnitError("inputs must be an object")
        inputs = {name: _parse_pinned_ref(ref, f"inputs.{name}") for name, ref in raw_inputs.items() if isinstance(name, str)}
        if len(inputs) != len(raw_inputs):
            raise WorkUnitError("inputs must use string names")
        unit = cls(
            run_id=value["run_id"],
            work_unit_id=value["work_unit_id"],
            sequence=value["sequence"],
            phase=value["phase"],
            predecessor_work_unit_id=value["predecessor_work_unit_id"],
            inputs=inputs,
            configuration=_parse_pinned_ref(value["configuration"], "configuration"),
            resources=_strict_json_object(value["resources"], "resources"),
            parameters=_strict_json_object(value["parameters"], "parameters"),
            start_cursor=_strict_json_object(value["start_cursor"], "start_cursor"),
            stop_boundary=_strict_json_object(value["stop_boundary"], "stop_boundary"),
            output_contract=_strict_json_object(value["output_contract"], "output_contract"),
            compatibility=_strict_json_object(value["compatibility"], "compatibility"),
        )
        if value["action"] != unit.action:
            raise WorkUnitError(f"V2 action must be {unit.action!r} for phase {unit.phase!r}")
        return unit


@dataclass(frozen=True)
class ExecutionEnvelope:
    """Ephemeral authority to execute one V2 unit attempt.

    Bearer lease tokens, staging paths, and output locations are intentionally
    absent.  They are transport secrets/control-plane state, not durable
    scientific execution input.
    """

    work_unit_id: str
    work_unit_sha256: str
    attempt_id: str
    generation: int
    worker_id: str
    worker_boot_id: str
    lease_id: str
    lease_expires_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "work_unit_id", _require_uuid(self.work_unit_id, "work_unit_id"))
        if not isinstance(self.work_unit_sha256, str) or not _SHA256.fullmatch(self.work_unit_sha256):
            raise WorkUnitError("work_unit_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "attempt_id", _require_uuid(self.attempt_id, "attempt_id"))
        object.__setattr__(self, "generation", _require_nonnegative_int(self.generation, "generation"))
        if not isinstance(self.worker_id, str) or not self.worker_id.strip() or "/" in self.worker_id or "\\" in self.worker_id:
            raise WorkUnitError("worker_id must be a non-path identifier")
        object.__setattr__(self, "worker_boot_id", _require_uuid(self.worker_boot_id, "worker_boot_id"))
        object.__setattr__(self, "lease_id", _require_uuid(self.lease_id, "lease_id"))
        object.__setattr__(self, "lease_expires_at", _require_timestamp(self.lease_expires_at, "lease_expires_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": {"name": EXECUTION_ENVELOPE_SCHEMA_NAME, "version": EXECUTION_ENVELOPE_SCHEMA_VERSION},
            "work_unit_id": self.work_unit_id,
            "work_unit_sha256": self.work_unit_sha256,
            "attempt_id": self.attempt_id,
            "generation": self.generation,
            "worker_id": self.worker_id,
            "worker_boot_id": self.worker_boot_id,
            "lease_id": self.lease_id,
            "lease_expires_at": self.lease_expires_at,
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionEnvelope":
        if not isinstance(value, Mapping):
            raise WorkUnitError("execution envelope must be an object")
        fields = frozenset({
            "schema", "work_unit_id", "work_unit_sha256", "attempt_id", "generation", "worker_id",
            "worker_boot_id", "lease_id", "lease_expires_at",
        })
        _require_exact_fields(value, fields, "execution envelope")
        schema = value["schema"]
        if not isinstance(schema, Mapping) or dict(schema) != {"name": EXECUTION_ENVELOPE_SCHEMA_NAME, "version": EXECUTION_ENVELOPE_SCHEMA_VERSION}:
            raise WorkUnitError(f"Expected {EXECUTION_ENVELOPE_SCHEMA_NAME} schema version {EXECUTION_ENVELOPE_SCHEMA_VERSION}")
        return cls(
            work_unit_id=value["work_unit_id"], work_unit_sha256=value["work_unit_sha256"],
            attempt_id=value["attempt_id"], generation=value["generation"], worker_id=value["worker_id"],
            worker_boot_id=value["worker_boot_id"], lease_id=value["lease_id"],
            lease_expires_at=value["lease_expires_at"],
        )


def parse_work_unit(value: Mapping[str, Any]) -> WorkUnit | WorkUnitV2:
    """Parse either supported immutable work-unit wire contract."""
    if not isinstance(value, Mapping):
        raise WorkUnitError("work unit must be an object")
    schema = value.get("schema")
    if not isinstance(schema, Mapping) or schema.get("name") != WORK_UNIT_SCHEMA_NAME:
        raise WorkUnitError("work unit has an unsupported schema")
    version = schema.get("version")
    if version == WORK_UNIT_SCHEMA_VERSION:
        return WorkUnit.from_dict(value)
    if version == WORK_UNIT_V2_SCHEMA_VERSION:
        return WorkUnitV2.from_dict(value)
    raise WorkUnitError(f"Unsupported {WORK_UNIT_SCHEMA_NAME} schema version {version}")


__all__ = [
    "EXECUTION_ENVELOPE_SCHEMA_NAME", "EXECUTION_ENVELOPE_SCHEMA_VERSION", "WORK_UNIT_ACTIONS",
    "WORK_UNIT_SCHEMA_NAME", "WORK_UNIT_SCHEMA_VERSION", "WORK_UNIT_V2_PHASES", "WORK_UNIT_V2_SCHEMA_VERSION",
    "ExecutionEnvelope", "WorkUnit", "WorkUnitError", "WorkUnitV2", "parse_work_unit",
]
