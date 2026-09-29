"""Reusable, worker-safe batch inference workflow.

This module deliberately knows nothing about HTTP, worker leases, or command
line parsing.  Callers provide local paths that they own: a sealed model
artifact, a frozen SQLite input, and an empty output directory.  The output is
an immutable inference-result artifact suitable for staged publication.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sqlite3
import uuid
from typing import Any, Callable, Mapping

from oracle_builder.artifacts import (
    read_run_config,
    read_run_manifest,
    split_manifest_matches_dataset,
)
from oracle_builder.classification.evidence import IdentityEvidenceIndex
from oracle_builder.data.sqlite_dataset import load_prediction_arrays
from oracle_builder.evaluation.predictions import write_predictions_db
from oracle_builder.inference.batching import resolve_inference_batch_size
from oracle_builder.saving.load_test import load_model_for_run


INFERENCE_RESULT_SCHEMA_NAME = "oracle_builder_inference_result"
INFERENCE_RESULT_SCHEMA_VERSION = 1
INFERENCE_SHARD_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Explicit local inputs for one immutable batch-inference artifact."""

    model_run: str
    input: str
    output_dir: str
    split: str = "all"
    prediction_set: str | None = None
    artifact_id: str | None = None
    lineage: Mapping[str, Any] | None = None
    model_reference: Mapping[str, Any] | None = None
    input_reference: Mapping[str, Any] | None = None
    shard_id: str | None = None
    shard_item_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.split not in {"all", "train", "validation", "test"}:
            raise ValueError("split must be one of all, train, validation, or test")
        if not self.model_run or not self.input or not self.output_dir:
            raise ValueError("model_run, input, and output_dir are required")
        if self.prediction_set is not None and (not self.prediction_set.strip() or len(self.prediction_set) > 200):
            raise ValueError("prediction_set must be a non-empty label of at most 200 characters")
        if (self.shard_id is None) != (self.shard_item_ids is None):
            raise ValueError("shard_id and shard_item_ids must be supplied together")
        if self.shard_item_ids is not None:
            ids = tuple(sorted(str(item_id) for item_id in self.shard_item_ids))
            if not self.shard_id or not ids or len(ids) != len(set(ids)):
                raise ValueError("a shard requires a non-empty id and unique item ids")
            object.__setattr__(self, "shard_item_ids", ids)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"artifact.json", "checksums.sha256"}:
            continue
        digest = sha256(path.read_bytes()).hexdigest()
        rows.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest})
    return rows


def _item_ids_digest(item_ids: tuple[str, ...] | list[str]) -> str:
    return sha256(json.dumps(sorted(item_ids), separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_shard_manifest(output: Path, request: InferenceRequest, item_ids: list[str]) -> dict[str, Any] | None:
    if request.shard_id is None:
        return None
    expected = list(request.shard_item_ids or ())
    if sorted(item_ids) != expected:
        raise ValueError("inference shard wrote a different item population than its sealed item-id contract")
    value = {
        "schema": {"name": "oracle_builder_inference_shard", "version": INFERENCE_SHARD_SCHEMA_VERSION},
        "shard_id": request.shard_id,
        "item_ids": expected,
        "item_ids_sha256": _item_ids_digest(expected),
    }
    (output / "inference_shard.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _seal_result(*, output: Path, model_manifest: Mapping[str, Any], request: InferenceRequest,
                 written: int, model_ref: Mapping[str, Any] | None, input_ref: Mapping[str, Any] | None) -> dict[str, Any]:
    artifact_id = request.artifact_id or str(uuid.uuid4())
    try:
        artifact_id = str(uuid.UUID(artifact_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("artifact_id must be a UUID") from exc
    # Write the human-facing summary before building the inventory so every
    # result file except the self-describing manifest/checksum list is sealed.
    (output / "README.md").write_text(
        f"# Inference result {artifact_id}\n\n"
        f"- Model artifact: `{model_manifest.get('artifact_id', '')}`\n"
        f"- Predictions: `predictions.sqlite` ({written} records)\n"
        f"- Split: `{request.split}`\n",
        encoding="utf-8",
    )
    inventory = _file_inventory(output)
    manifest: dict[str, Any] = {
        "schema": {"name": INFERENCE_RESULT_SCHEMA_NAME, "version": INFERENCE_RESULT_SCHEMA_VERSION},
        "artifact_type": "inference_result",
        "artifact_id": artifact_id,
        "status": "complete",
        "lifecycle": "sealed",
        "created_at": _utc_now(),
        "model": {
            "artifact_id": model_manifest.get("artifact_id"),
            "fingerprint_sha256": model_manifest.get("fingerprint_sha256"),
            "reference": dict(model_ref or {}),
        },
        "input": {"reference": dict(input_ref or {})},
        "parameters": {"split": request.split, "prediction_set": request.prediction_set,
                       "shard_id": request.shard_id},
        "outputs": {"predictions": "predictions.sqlite", "records": written},
        "lineage": dict(request.lineage or {}),
        "inventory": inventory,
        "fingerprint_sha256": None,
    }
    fingerprint_payload = dict(manifest)
    fingerprint_payload["fingerprint_sha256"] = None
    manifest["fingerprint_sha256"] = sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (output / "checksums.sha256").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in inventory), encoding="utf-8"
    )
    (output / "artifact.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def run_inference(request: InferenceRequest, *, progress: Callable[[str, str, Mapping[str, Any] | None], None] | None = None) -> dict[str, Any]:
    """Run batch inference and seal its portable result in ``output_dir``."""
    if not isinstance(request, InferenceRequest):
        raise TypeError("request must be an InferenceRequest")
    report = progress or (lambda _kind, _message, _data=None: None)
    run_dir = Path(request.model_run).expanduser().resolve()
    input_path = Path(request.input).expanduser().resolve()
    output = Path(request.output_dir).expanduser().resolve()
    if not run_dir.is_dir() or not input_path.is_file():
        raise ValueError("model_run must be a directory and input must be a SQLite file")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output_dir must be empty")
    output.mkdir(parents=True, exist_ok=True)
    model_manifest = read_run_manifest(run_dir)
    if model_manifest.get("lifecycle") != "sealed" or model_manifest.get("status") != "complete":
        raise ValueError("inference requires a sealed, completed model artifact")
    report("inference_loading", "Loading sealed model artifact", {"model_artifact_id": model_manifest.get("artifact_id")})
    config = read_run_config(run_dir)
    model = load_model_for_run(run_dir, config)
    evidence_path = run_dir / "model" / "classification_evidence"
    if not evidence_path.exists():
        evidence_path = run_dir / "model" / "classification_evidence.npz"
    evidence_index = IdentityEvidenceIndex.load(evidence_path) if config["run"]["task"] == "classification" and evidence_path.exists() else None
    batch_size = resolve_inference_batch_size(model, config).batch_size
    selected_split = None if request.split == "all" else request.split
    if not split_manifest_matches_dataset(config, input_path):
        if selected_split is not None:
            raise ValueError("a named split requires the exact dataset revision recorded by the model artifact")
        config.pop("_split_manifest", None)
        config["_external_inference"] = True
    prediction_path = output / "predictions.sqlite"
    prediction_set = request.prediction_set or str(model_manifest.get("name") or run_dir.name)
    report("inference_processing", "Running model inference", {"batch_size": batch_size, "split": request.split})
    if config["run"]["task"] == "classification" and config.get("data", {}).get("streaming", {}).get("enabled", True):
        from oracle_builder.data.sqlite_stream import SQLiteClassificationSource, build_all_classification_index, build_classification_index
        from oracle_builder.evaluation.predictions import write_classification_predictions_streaming
        index = build_all_classification_index(input_path, config, labeled_only=False) if selected_split is None else build_classification_index(input_path, config, selected_split, labeled_only=False)
        if request.shard_item_ids is not None:
            selected_ids = set(request.shard_item_ids)
            available = {ref.item_id for ref in index.refs}
            missing = selected_ids.difference(available)
            if missing:
                raise ValueError(f"inference shard item ids are unavailable for this sealed input: {sorted(missing)[:3]}")
            index.refs = [ref for ref in index.refs if ref.item_id in selected_ids]
        source = SQLiteClassificationSource(input_path, config)
        written = write_classification_predictions_streaming(model, source.indexed_image_dataset(index, batch_size=batch_size), index, config, prediction_path, source_sqlite=input_path, prediction_set=prediction_set, evidence_index=evidence_index, progress=bool(config.get("inference", {}).get("progress", True)))
    else:
        x, targets, records = load_prediction_arrays(input_path, config, split=selected_split)
        if request.shard_item_ids is not None:
            selected_ids = set(request.shard_item_ids)
            positions = [index for index, record in enumerate(records) if str(record["uuid"]) in selected_ids]
            if {str(records[index]["uuid"]) for index in positions} != selected_ids:
                raise ValueError("inference shard item ids are unavailable for this sealed input")
            x = x[positions]
            targets = [targets[index] for index in positions]
            records = [records[index] for index in positions]
        write_predictions_db(model, x, targets, records, config, prediction_path, source_sqlite=input_path, prediction_set=prediction_set, evidence_index=evidence_index, inference_batch_size=batch_size)
        written = len(records)
    actual_ids = _prediction_item_ids(prediction_path, prediction_set)
    _write_shard_manifest(output, request, actual_ids)
    report("inference_sealing", "Sealing inference-result artifact", {"records": written})
    # Persist the resolved default too: shard merge must compare the actual
    # prediction-set identity, never a nullable caller hint.
    sealed_request = replace(request, prediction_set=prediction_set)
    manifest = _seal_result(
        output=output, model_manifest=model_manifest, request=sealed_request, written=written,
        model_ref=request.model_reference, input_ref=request.input_reference,
    )
    report("inference_complete", "Inference result artifact sealed", {"artifact_id": manifest["artifact_id"], "records": written})
    return manifest


def _prediction_item_ids(path: Path, prediction_set: str) -> list[str]:
    with sqlite3.connect(path) as connection:
        return [str(row[0]) for row in connection.execute(
            "SELECT uuid FROM predictions WHERE prediction_set = ? ORDER BY uuid", (prediction_set,)
        )]


def merge_inference_shards(
    shard_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    expected_item_ids: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Merge sealed shard artifacts after proving their identities and coverage.

    This intentionally accepts duplicate retry artifacts only when their output
    bytes match. A second prediction for an item with different content is a
    protocol failure, never last-writer-wins.
    """
    expected = tuple(sorted(str(item) for item in expected_item_ids))
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected_item_ids must be a non-empty unique item-id set")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output_dir must be empty")
    manifests: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for root_value in shard_dirs:
        root = Path(root_value).expanduser().resolve()
        artifact = json.loads((root / "artifact.json").read_text(encoding="utf-8"))
        shard = json.loads((root / "inference_shard.json").read_text(encoding="utf-8"))
        if shard.get("schema") != {"name": "oracle_builder_inference_shard", "version": INFERENCE_SHARD_SCHEMA_VERSION}:
            raise ValueError(f"unsupported inference shard manifest: {root}")
        ids = [str(item) for item in shard.get("item_ids", [])]
        if shard.get("item_ids_sha256") != _item_ids_digest(ids):
            raise ValueError(f"inference shard item-id digest mismatch: {root}")
        if _prediction_item_ids(root / "predictions.sqlite", str(artifact.get("parameters", {}).get("prediction_set") or "")) != sorted(ids):
            raise ValueError(f"inference shard prediction rows do not match its declared item ids: {root}")
        manifests.append((root, artifact, shard))
    if not manifests:
        raise ValueError("at least one shard artifact is required")
    identity = (manifests[0][1].get("model"), manifests[0][1].get("input"), manifests[0][1].get("parameters", {}).get("split"), manifests[0][1].get("parameters", {}).get("prediction_set"))
    if any((artifact.get("model"), artifact.get("input"), artifact.get("parameters", {}).get("split"), artifact.get("parameters", {}).get("prediction_set")) != identity for _, artifact, _ in manifests):
        raise ValueError("all inference shards must have the same model, input, split, and prediction-set identity")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifests[0][0] / "predictions.sqlite", output / "predictions.sqlite")
    prediction_set = str(identity[3] or "")
    seen: dict[str, tuple[str | None, str | None]] = {}
    with sqlite3.connect(output / "predictions.sqlite") as destination:
        destination.execute("DELETE FROM predictions WHERE prediction_set = ?", (prediction_set,))
        for root, artifact, _shard in manifests:
            candidate_set = str(artifact.get("parameters", {}).get("prediction_set") or "")
            with sqlite3.connect(root / "predictions.sqlite") as source:
                rows = source.execute("SELECT * FROM predictions WHERE prediction_set = ? ORDER BY uuid", (candidate_set,)).fetchall()
                columns = [row[1] for row in source.execute("PRAGMA table_info(predictions)")]
                for row in rows:
                    mapped = dict(zip(columns, row, strict=True)); item_id = str(mapped["uuid"])
                    content = (mapped.get("output_sha256"), mapped.get("inference_result_json"))
                    if item_id in seen:
                        if seen[item_id] != content:
                            raise ValueError(f"conflicting duplicate inference output for item {item_id}")
                        continue
                    seen[item_id] = content
                    placeholders = ", ".join("?" for _ in columns)
                    destination.execute(f"INSERT INTO predictions ({', '.join(columns)}) VALUES ({placeholders})", row)
        if sorted(seen) != list(expected):
            missing = sorted(set(expected).difference(seen)); extra = sorted(set(seen).difference(expected))
            raise ValueError(f"inference shard merge coverage mismatch: missing={missing[:3]}, extra={extra[:3]}")
        destination.commit()
    request = InferenceRequest(model_run="merge", input="merge", output_dir=str(output), split=str(identity[2]), prediction_set=prediction_set)
    manifest = _seal_result(output=output, model_manifest=identity[0] or {}, request=request, written=len(seen), model_ref=(identity[0] or {}).get("reference"), input_ref=(identity[1] or {}).get("reference"))
    return manifest


__all__ = ["INFERENCE_RESULT_SCHEMA_NAME", "INFERENCE_RESULT_SCHEMA_VERSION", "INFERENCE_SHARD_SCHEMA_VERSION", "InferenceRequest", "run_inference", "merge_inference_shards"]
