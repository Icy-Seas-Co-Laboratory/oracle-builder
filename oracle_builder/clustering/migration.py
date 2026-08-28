"""Migrate legacy clustering runs into standalone downstream records."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from oracle_builder.artifacts import read_run_config, read_run_manifest, validate_run_artifact
from oracle_builder.clustering.evidence import ClusterEvidenceIndex


CLUSTER_RECORD_SCHEMA = "oracle_cluster_record"
CLUSTER_RECORD_VERSION = "1.0.0"


def _write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path.name == "checksums.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def migrate_sealed_clustering_package(
    source: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Extract legacy cluster evidence into a standalone downstream record.

    The source model and source artifact remain untouched. The resulting
    record contains only the dataset-conditioned cluster definition and a
    lineage reference to the embedding model that produced it.
    """
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    manifest = read_run_manifest(source_path)
    if manifest.get("lifecycle") != "sealed":
        raise ValueError("Legacy clustering package must be sealed before migration")
    config = read_run_config(source_path)
    if config.get("run", {}).get("task") != "clustering":
        raise ValueError("Source artifact is not a legacy clustering package")
    validation = validate_run_artifact(source_path)
    if not validation["valid"]:
        raise ValueError(
            "Legacy clustering package failed validation: "
            + "; ".join(validation["errors"])
        )
    evidence_path = source_path / "model" / "clustering_evidence"
    if not evidence_path.is_dir():
        raise FileNotFoundError(
            "Legacy clustering package is missing model/clustering_evidence"
        )
    evidence = ClusterEvidenceIndex.load(evidence_path)

    output_path.mkdir(parents=True)
    try:
        destination_evidence = output_path / "evidence"
        shutil.copytree(evidence_path, destination_evidence)
        record = {
            "schema": {"name": CLUSTER_RECORD_SCHEMA, "version": CLUSTER_RECORD_VERSION},
            "record_id": str(uuid.uuid4()),
            "record_type": "downstream_cluster_definition",
            "lifecycle": "sealed",
            "source_model": {
                "artifact_id": manifest.get("artifact_id"),
                "run_id": manifest.get("run_id"),
                "fingerprint_sha256": manifest.get("fingerprint_sha256"),
                "architecture": manifest.get("model", {}).get("architecture"),
                "embedding_dimension": evidence.embedding_dim,
                "embedding_normalized": True,
            },
            "source_dataset": manifest.get("dataset", {}),
            "cluster_structure": evidence.summary(),
            "evidence_path": "evidence",
            "semantics": {
                "cluster_ids": "run_local",
                "labels": "not_provided",
                "novelty": evidence.summary().get("novelty_calibration"),
            },
        }
        (output_path / "cluster_manifest.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        (output_path / "README.md").write_text(
            "# Downstream cluster record\n\n"
            "This record was migrated from a legacy Oracle Builder clustering run. "
            "It defines dataset-conditioned cluster evidence; it is not a model "
            "and must be used with the referenced embedding model.\n",
            encoding="utf-8",
        )
        _write_checksums(output_path)
    except Exception:
        shutil.rmtree(output_path)
        raise
    return {
        "output": str(output_path),
        "migration": {
            "source_artifact_id": manifest.get("artifact_id"),
            "source_run_id": manifest.get("run_id"),
            "record_type": "downstream_cluster_definition",
            "cluster_count": evidence.cluster_count,
        },
    }
