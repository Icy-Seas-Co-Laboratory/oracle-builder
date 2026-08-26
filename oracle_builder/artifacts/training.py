"""Training-record packaging and materialization helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from oracle_data_contracts.artifacts.layout import RunLayout
from oracle_data_contracts.artifacts.run import (
    read_run_manifest,
    reopen_run_artifact,
    seal_run_artifact,
)
from oracle_builder.training.logging_callbacks import append_jsonl_event


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_training_record(
    training_record: str | Path,
    output: str | Path,
    *,
    dataset: str | Path | None = None,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Copy a sealed training record and embed optional retraining resources."""
    source = Path(training_record).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    manifest = read_run_manifest(source)
    if manifest.get("lifecycle") != "sealed":
        raise ValueError("Only a sealed training record can be materialized")
    if manifest.get("standard", {}).get("profile") not in {None, "training_record"}:
        raise ValueError("The source artifact is not a training record")

    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("checksums.sha256"))
    layout = RunLayout(destination)
    reopened = reopen_run_artifact(destination, reason="Materialized retraining library into a new package")
    library = layout.root / "library"
    library.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    if dataset is not None:
        dataset_path = Path(dataset).expanduser().resolve()
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        target = library / "dataset" / dataset_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset_path, target)
        entries.append({
            "role": "dataset",
            "source": str(dataset_path),
            "path": target.relative_to(destination).as_posix(),
            "size_bytes": target.stat().st_size,
            "sha256": _sha256(target),
        })

    if source_root is not None:
        source_path = Path(source_root).expanduser().resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(source_path)
        target = library / "source"
        shutil.copytree(source_path, target)
        for path in sorted(value for value in target.rglob("*") if value.is_file()):
            entries.append({
                "role": "source",
                "source": str(source_path),
                "path": path.relative_to(destination).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })

    (library / "manifest.json").write_text(
        json.dumps({
            "schema": {"name": "oracle_training_library", "version": "1.0.0"},
            "training_record_id": manifest["artifact_id"],
            "materialized_from": str(source),
            "entries": entries,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    current = read_run_manifest(destination)
    current.setdefault("paths", {})["training_library"] = "library"
    current.setdefault("lineage", {})["materialized_from"] = manifest["artifact_id"]
    layout.manifest.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl_event(layout.events_jsonl, manifest["run_id"], "INFO", "training_library_materialized", {"entry_count": len(entries)})
    sealed = seal_run_artifact(destination)
    return {
        "output": str(destination),
        "artifact_id": sealed["artifact_id"],
        "fingerprint_sha256": sealed["fingerprint_sha256"],
        "library_entries": len(entries),
        "reopened_from": reopened.get("fingerprint_sha256"),
    }

