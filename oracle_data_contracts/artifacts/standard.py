"""Common model-artifact standard helpers."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


MODEL_ARTIFACT_SCHEMA_NAME = "oracle_model_artifact"
MODEL_ARTIFACT_SCHEMA_VERSION = "2.0.0"
MODEL_ARTIFACT_PROFILES = {"deployment_asset", "training_record"}


def profile_for_artifact_type(artifact_type: str) -> str:
    if artifact_type == "model_product":
        return "deployment_asset"
    if artifact_type == "model_run":
        return "training_record"
    raise ValueError(f"Unsupported model artifact type: {artifact_type}")


def standard_descriptor(artifact_type: str) -> dict[str, str]:
    return {
        "name": MODEL_ARTIFACT_SCHEMA_NAME,
        "version": MODEL_ARTIFACT_SCHEMA_VERSION,
        "profile": profile_for_artifact_type(artifact_type),
    }


def write_model_contract(run_dir: str | Path, contract: dict[str, Any]) -> Path:
    """Write the single authoritative model contract for an artifact."""
    path = Path(run_dir).expanduser().resolve() / "model" / "contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = dict(contract)
    # V1 external-product contracts used a singular ``input`` field.  V2 is
    # role-oriented and always exposes a mapping of named inputs.
    if "input" in payload and "inputs" not in payload:
        payload["inputs"] = {"image": payload.pop("input")}
    payload.setdefault(
        "schema",
        {"name": "oracle_model_contract", "version": MODEL_ARTIFACT_SCHEMA_VERSION},
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path
