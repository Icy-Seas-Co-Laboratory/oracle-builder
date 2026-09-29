"""Orchestrator helpers for building and locally adapting WorkUnit v1.

The durable :mod:`oracle_data_contracts.work_units` object never receives a
host path.  This module is the explicit, temporary bridge used by the current
local compute API, whose command implementation still requires paths.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from oracle_data_contracts.artifacts import ArtifactRef
from oracle_data_contracts.work_units import WorkUnit


def _legacy_file_ref(role: str, value: str | Path) -> ArtifactRef:
    """Reference a legacy local input without serializing its path.

    The digest identifies the local adapter binding, not the content bytes.
    It is intentionally marked as ``legacy_file`` so a remote worker cannot
    mistake it for a store-resolvable artifact.
    """
    identifier = sha256(f"{role}:{Path(value).expanduser().resolve()}".encode("utf-8")).hexdigest()
    return ArtifactRef("legacy_file", identifier)


def _artifact_ref(value: Any, field: str) -> ArtifactRef:
    """Parse a portable ref accepted in a control-plane plan parameter."""
    try:
        if isinstance(value, ArtifactRef):
            return value
        if isinstance(value, str):
            return ArtifactRef.parse(value)
        if isinstance(value, Mapping):
            return ArtifactRef.from_dict(value)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a valid portable artifact reference") from error
    raise ValueError(f"{field} must be a valid portable artifact reference")


def _portable_inputs(parameters: Mapping[str, Any]) -> dict[str, ArtifactRef]:
    raw = parameters.get("artifact_inputs", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("artifact_inputs must be a mapping of input role to artifact reference")
    parsed = {str(role): _artifact_ref(value, f"artifact_inputs.{role}") for role, value in raw.items()}
    # The individual aliases make JSON form submissions less verbose while the
    # mapping stays the canonical persisted shape.
    for role in ("input", "model", "info", "run"):
        alias = f"{role}_artifact"
        if alias in parameters:
            ref = _artifact_ref(parameters[alias], alias)
            existing = parsed.get(role)
            if existing is not None and existing != ref:
                raise ValueError(f"conflicting portable references for {role}")
            parsed[role] = ref
    return parsed


def build_work_unit(
    *,
    work_unit_id: str,
    attempt_id: str,
    specification_id: str,
    action: str,
    parameters: Mapping[str, Any],
    resources: Mapping[str, Any] | None,
    dataset: Mapping[str, Any] | None = None,
    require_portable: bool = False,
) -> WorkUnit:
    """Build the sealed transport contract from an already-planned spec.

    All current specification creators flow through dispatch, so centralizing
    this conversion means they remain compatible while their path payloads
    are progressively replaced with resolvable artifact-store references.
    """
    inputs = _portable_inputs(parameters)
    input_path = parameters.get("input")
    if input_path is not None and "input" not in inputs:
        if require_portable:
            raise ValueError("portable work units require artifact_inputs.input instead of input")
        if dataset is not None:
            inputs["input"] = ArtifactRef(
                "dataset", str(dataset["dataset_id"]),
                revision=str(dataset["revision_id"]) if dataset.get("revision_id") else None,
                fingerprint_sha256=str(dataset["fingerprint_sha256"]) if dataset.get("fingerprint_sha256") else None,
            )
        else:
            inputs["input"] = _legacy_file_ref("input", str(input_path))
    for role in ("model", "info", "run"):
        value = parameters.get(role)
        if value is not None and role not in inputs:
            if require_portable:
                raise ValueError(f"portable work units require artifact_inputs.{role} instead of {role}")
            inputs[role] = _legacy_file_ref(role, str(value))
    # Model ingestion has no ``input`` field; it still needs one or more
    # references, while a run validation has only the run reference.
    if not inputs:
        raise ValueError(f"Cannot build a work unit without inputs for {action}")
    config_artifact = parameters.get("configuration_artifact", parameters.get("config_artifact"))
    if config_artifact is not None:
        configuration = _artifact_ref(config_artifact, "configuration_artifact")
    else:
        config = parameters.get("config")
        if config is not None and require_portable:
            raise ValueError("portable work units require configuration_artifact instead of config")
        configuration = _legacy_file_ref("config", str(config)) if config is not None else None
    if require_portable and any(ref.kind == "legacy_file" for ref in [*inputs.values(), *([configuration] if configuration else [])]):
        raise ValueError("portable work units cannot contain legacy_file references")
    action_parameters: dict[str, Any] = {}
    if action == "infer":
        raw_inference = parameters.get("inference", {})
        if not isinstance(raw_inference, Mapping):
            raise ValueError("inference parameters must be an object")
        action_parameters = dict(raw_inference)
    elif action == "train":
        # Train work needs no user-controlled executable or path, but it can
        # carry the sealed batch execution policy.  In particular, an auto
        # policy is resolved only by the worker that owns the accelerator.
        raw_execution = parameters.get("queue_execution", {})
        if not isinstance(raw_execution, Mapping):
            raise ValueError("queue_execution must be an object")
        if raw_execution:
            action_parameters = {"queue_execution": dict(raw_execution)}
    return WorkUnit(
        work_unit_id=work_unit_id,
        attempt_id=attempt_id,
        specification_id=specification_id,
        action=action,
        inputs=inputs,
        configuration=configuration,
        staging=ArtifactRef("staging", work_unit_id, revision=attempt_id),
        resources=dict(resources or {}),
        parameters=action_parameters,
    )


class LocalPathWorkUnitAdapter:
    """Compatibility adapter from a sealed work unit to today's local API.

    This is the only Phase-2 boundary that intentionally handles paths.  A
    future worker will materialize the ``ArtifactRef`` values through an
    ``ArtifactStore`` instead of receiving these parameters.
    """

    @staticmethod
    def request_envelope(work_unit: WorkUnit, parameters: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "job_id": work_unit.work_unit_id,
            "action": work_unit.action,
            "parameters": dict(parameters),
            "resources": dict(work_unit.resources),
            "work_unit": work_unit.to_dict(),
        }


__all__ = ["LocalPathWorkUnitAdapter", "build_work_unit"]
