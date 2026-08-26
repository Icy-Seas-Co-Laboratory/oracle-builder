"""Publish a lean deployment asset from a sealed training record."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from oracle_data_contracts.artifacts.layout import RunLayout
from oracle_data_contracts.artifacts.run import (
    create_run_artifact,
    read_run_config,
    read_run_manifest,
    seal_run_artifact,
    update_run_artifact,
    write_run_config,
)
from oracle_data_contracts.artifacts.standard import write_model_contract
from oracle_builder.environment import write_environment
from oracle_builder.training.logging_callbacks import append_jsonl_event


def publish_deployment_asset(
    training_run: str | Path,
    output: str | Path,
    *,
    include_weights: bool = False,
    include_evidence: bool = True,
) -> dict[str, Any]:
    """Create a new sealed deployment asset from a sealed training record.

    The source training record is never modified.  Checkpoints, recovery
    state, pretraining material, logs, predictions, and evaluation products
    are intentionally not copied.
    """
    source = Path(training_run).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    source_manifest = read_run_manifest(source)
    if source_manifest.get("lifecycle") != "sealed":
        raise ValueError("A deployment asset can only be published from a sealed training record")
    if source_manifest.get("standard", {}).get("profile") not in {None, "training_record"}:
        raise ValueError("Source artifact is not a training record")

    source_layout = RunLayout(source)
    source_config = read_run_config(source)
    deployment_config = json.loads(json.dumps(source_config, default=str))
    deployment_run_id = str(uuid.uuid4())
    deployment_config.setdefault("run", {})["run_id"] = deployment_run_id
    deployment_config["run"]["run_name"] = destination.name
    deployment_config["lineage"] = {
        "training_record_id": source_manifest["artifact_id"],
        "training_run_id": source_manifest.get("run_id"),
        "source_fingerprint_sha256": source_manifest.get("fingerprint_sha256"),
    }

    source_config_path = source_layout.source_config
    if not source_config_path.exists():
        raise FileNotFoundError(source_config_path)
    manifest = create_run_artifact(
        destination,
        run_id=deployment_run_id,
        name=destination.name,
        config=deployment_config,
        source_config=source_config_path,
        artifact_type="model_product",
        lineage=deployment_config["lineage"],
    )
    layout = RunLayout(destination)
    environment = write_environment(destination)
    append_jsonl_event(
        layout.events_jsonl,
        deployment_run_id,
        "INFO",
        "deployment_asset_started",
        {"source_training_record_id": source_manifest["artifact_id"]},
    )

    try:
        model_source = source_layout.model
        required_files = ("model_manifest.json", "final.keras", "load_test_report.json")
        for name in required_files:
            source_path = model_source / name
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            shutil.copy2(source_path, layout.model / name)
        for name in ("export_savedmodel", "inspection.json", "model_summary.txt"):
            source_path = model_source / name
            if source_path.is_dir():
                shutil.copytree(source_path, layout.model / name)
            elif source_path.exists():
                shutil.copy2(source_path, layout.model / name)
        if include_weights and (model_source / "weights.weights.h5").exists():
            shutil.copy2(model_source / "weights.weights.h5", layout.model / "weights.weights.h5")
        if include_evidence:
            for name in ("classification_evidence", "clustering_evidence"):
                source_path = model_source / name
                if source_path.is_dir():
                    shutil.copytree(source_path, layout.model / name)

        model_manifest_path = layout.model / "model_manifest.json"
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        model_manifest["artifact_id"] = manifest["artifact_id"]
        model_manifest["run_id"] = deployment_run_id
        model_manifest["training_record_id"] = source_manifest["artifact_id"]
        model_manifest_path.write_text(
            json.dumps(model_manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        source_contract = model_source / "contract.json"
        if source_contract.exists():
            shutil.copy2(source_contract, layout.model / "contract.json")
        else:
            write_model_contract(
                destination,
                {
                    "task": model_manifest.get("task"),
                    "architecture": model_manifest.get("architecture"),
                    "inputs": {"image": model_manifest.get("input", {})},
                    "outputs": model_manifest.get("outputs", {}),
                    "preprocessing": model_manifest.get("input", {}).get("preprocessing", {}),
                    "postprocessing": model_manifest.get("postprocessing", {}),
                    "inference": model_manifest.get("inference_contract", {}),
                },
            )
        deployment_config["artifact"] = {
            "artifact_id": manifest["artifact_id"],
            "schema_name": manifest["artifact_schema"]["name"],
            "schema_version": manifest["artifact_schema"]["version"],
        }
        write_run_config(destination, deployment_config)
        append_jsonl_event(
            layout.events_jsonl,
            deployment_run_id,
            "INFO",
            "deployment_asset_created",
            {"include_weights": include_weights, "include_evidence": include_evidence},
        )
        update_run_artifact(
            destination,
            status="complete",
            summary={
                "deployment_asset": {
                    "source_training_record_id": source_manifest["artifact_id"],
                    "include_weights": include_weights,
                    "include_evidence": include_evidence,
                }
            },
        )
        sealed = seal_run_artifact(destination)
        return {
            "output": str(destination),
            "artifact_id": sealed["artifact_id"],
            "run_id": deployment_run_id,
            "fingerprint_sha256": sealed["fingerprint_sha256"],
            "environment_written": bool(environment),
        }
    except Exception as exc:
        append_jsonl_event(layout.events_jsonl, deployment_run_id, "ERROR", "deployment_asset_failed", {"error": str(exc)})
        update_run_artifact(destination, status="failed", error=str(exc))
        seal_run_artifact(destination)
        raise
