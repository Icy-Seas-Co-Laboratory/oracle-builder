from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oracle_builder.artifacts.layout import RunLayout
from oracle_builder.datasets.schema import dataset_fingerprint, read_dataset_info


SPLIT_MANIFEST_SCHEMA_NAME = "oracle_builder_split_manifest"
SPLIT_MANIFEST_SCHEMA_VERSION = "1.0.0"
SPLIT_NAMES = ("train", "validation", "test")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manifest_fingerprint(manifest: dict[str, Any]) -> str:
    semantic = {key: value for key, value in manifest.items() if key != "fingerprint_sha256"}
    encoded = json.dumps(
        semantic, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _assignment_rows(
    item_ids: list[str], *, seed: int, validation: float, test: float
) -> list[dict[str, str]]:
    """Return deterministic assignments with exact rounded split counts."""
    ranked = sorted(
        item_ids,
        key=lambda item_id: (
            hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).digest(),
            item_id,
        ),
    )
    n_test = int(round(len(ranked) * test))
    n_validation = int(round(len(ranked) * validation))
    if n_test + n_validation >= len(ranked):
        overflow = n_test + n_validation - len(ranked) + 1
        reduce_validation = min(n_validation, overflow)
        n_validation -= reduce_validation
        n_test -= overflow - reduce_validation
    assigned: dict[str, str] = {}
    for index, item_id in enumerate(ranked):
        if index < n_test:
            assigned[item_id] = "test"
        elif index < n_test + n_validation:
            assigned[item_id] = "validation"
        else:
            assigned[item_id] = "train"
    return [
        {"item_id": item_id, "split": assigned[item_id]}
        for item_id in sorted(item_ids)
    ]


def create_split_manifest(
    run_dir: str | Path,
    sqlite_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Create the immutable dataset-item assignment protocol for one model run."""
    validation = float(config["data"].get("validation_split", 0.2))
    test = float(config["data"].get("test_split", 0.1))
    if validation < 0 or test < 0 or validation + test >= 1:
        raise ValueError("Split fractions must be non-negative and sum to less than 1")
    seed = int(config["run"].get("seed", 123))
    layout = RunLayout(run_dir)
    if layout.split_manifest.exists():
        raise FileExistsError(layout.split_manifest)

    with sqlite3.connect(Path(sqlite_path).expanduser().resolve()) as connection:
        info = read_dataset_info(connection)
        fingerprint = dataset_fingerprint(connection)
        item_ids = [
            str(row[0])
            for row in connection.execute(
                "SELECT item_id FROM dataset_items ORDER BY item_id"
            )
        ]
    expected_dataset = config.get("dataset", {})
    if (
        expected_dataset.get("dataset_id")
        and expected_dataset["dataset_id"] != info["dataset_id"]
    ):
        raise ValueError("Resolved config dataset_id does not match the input dataset")
    if (
        expected_dataset.get("fingerprint_sha256")
        and expected_dataset["fingerprint_sha256"] != fingerprint
    ):
        raise ValueError(
            "Resolved config dataset fingerprint does not match the input dataset"
        )
    if not item_ids:
        raise ValueError("Cannot create a split manifest for an empty dataset")
    assignments = _assignment_rows(
        item_ids, seed=seed, validation=validation, test=test
    )
    counts = {
        name: sum(row["split"] == name for row in assignments)
        for name in SPLIT_NAMES
    }
    manifest: dict[str, Any] = {
        "schema": {
            "name": SPLIT_MANIFEST_SCHEMA_NAME,
            "version": SPLIT_MANIFEST_SCHEMA_VERSION,
        },
        "split_manifest_id": str(uuid.uuid4()),
        "created_at": _utc_now(),
        "dataset": {
            "dataset_id": info["dataset_id"],
            "revision_id": info["revision_id"],
            "fingerprint_sha256": fingerprint,
        },
        "policy": {
            "method": "stable_sha256_rank",
            "seed": seed,
            "validation_fraction": validation,
            "test_fraction": test,
        },
        "counts": counts,
        "assignments": assignments,
        "fingerprint_sha256": None,
    }
    manifest["fingerprint_sha256"] = _manifest_fingerprint(manifest)
    _write_json(layout.split_manifest, manifest)
    return manifest


def create_unavailable_split_manifest(
    run_dir: str | Path,
    config: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Record that historical per-item split assignments are unavailable."""
    layout = RunLayout(run_dir)
    if layout.split_manifest.exists():
        raise FileExistsError(layout.split_manifest)
    dataset = config.get("dataset", {})
    manifest: dict[str, Any] = {
        "schema": {
            "name": SPLIT_MANIFEST_SCHEMA_NAME,
            "version": SPLIT_MANIFEST_SCHEMA_VERSION,
        },
        "split_manifest_id": str(uuid.uuid4()),
        "created_at": _utc_now(),
        "dataset": {
            "dataset_id": dataset.get("dataset_id"),
            "revision_id": dataset.get("revision_id"),
            "fingerprint_sha256": dataset.get("fingerprint_sha256"),
        },
        "policy": {
            "method": "unavailable",
            "reason": str(reason),
        },
        "counts": {name: None for name in SPLIT_NAMES},
        "assignments": [],
        "fingerprint_sha256": None,
    }
    manifest["fingerprint_sha256"] = _manifest_fingerprint(manifest)
    _write_json(layout.split_manifest, manifest)
    return manifest


def read_split_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = RunLayout(run_dir).split_manifest
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != {
        "name": SPLIT_MANIFEST_SCHEMA_NAME,
        "version": SPLIT_MANIFEST_SCHEMA_VERSION,
    }:
        raise ValueError(f"Unsupported split manifest schema in {path}")
    expected = manifest.get("fingerprint_sha256")
    if not expected or expected != _manifest_fingerprint(manifest):
        raise ValueError(f"Split manifest fingerprint mismatch: {path}")
    return manifest


def split_assignments(manifest: dict[str, Any]) -> dict[str, str]:
    result = {
        str(row["item_id"]): str(row["split"])
        for row in manifest.get("assignments", [])
    }
    invalid = set(result.values()).difference(SPLIT_NAMES)
    if invalid:
        raise ValueError(f"Invalid split names in manifest: {sorted(invalid)}")
    return result


def attach_split_manifest(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Attach runtime-only assignments without embedding them in resolved config."""
    config["_split_manifest"] = {
        "split_manifest_id": manifest["split_manifest_id"],
        "fingerprint_sha256": manifest["fingerprint_sha256"],
        "dataset": dict(manifest["dataset"]),
        "assignments": split_assignments(manifest),
    }


def split_manifest_matches_dataset(
    config: dict[str, Any], sqlite_path: str | Path
) -> bool:
    # Evaluation and inference validate dataset identity before their ordinary
    # task loaders run. Ensure legacy ROI databases cross the same automatic
    # migration boundary as training, analysis, the editor, and visualization.
    if config.get("run", {}).get("task") == "segmentation":
        from oracle_builder.datasets.legacy_roi import (
            ensure_mask_refinement_database,
        )

        ensure_mask_refinement_database(sqlite_path)
    runtime = config.get("_split_manifest")
    if runtime is None:
        return False
    with sqlite3.connect(Path(sqlite_path).expanduser().resolve()) as connection:
        info = read_dataset_info(connection)
        fingerprint = dataset_fingerprint(connection)
    dataset = runtime.get("dataset", {})
    return (
        dataset.get("dataset_id") == info["dataset_id"]
        and dataset.get("revision_id") == info["revision_id"]
        and dataset.get("fingerprint_sha256") == fingerprint
    )
