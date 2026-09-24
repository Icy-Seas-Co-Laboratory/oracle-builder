"""Read-only discovery for Oracle Builder SQLite training-set revisions.

The catalog deliberately accepts only the versioned Oracle SQLite contract.
Image folders are source material, not training sets: they must first pass
through an explicit dataset conversion/import workflow.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oracle_data_contracts.datasets import read_dataset_info


def _utc_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _safe_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _family_and_version(metadata: dict[str, Any], path: Path) -> tuple[str, str]:
    """Derive a stable display bundle while honoring explicit dataset metadata."""
    custom = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
    name = str(metadata.get("name") or path.stem).strip() or path.stem
    family = next((str(values[key]).strip() for values in (custom, metadata)
                   for key in ("training_set_family", "dataset_family", "family")
                   if values.get(key)), name)
    filename_tail = path.stem[len(name):].strip("._-") if path.stem.lower().startswith(name.lower()) else ""
    version = next((str(values[key]).strip() for values in (custom, metadata)
                    for key in ("training_set_version", "dataset_version") if values.get(key)), "")
    version = version or filename_tail or str(metadata.get("version") or metadata.get("revision_id") or "unversioned")
    return family, version


def _sqlite_entry(path: Path, root: Path) -> dict[str, Any] | None:
    try:
        with sqlite3.connect(path) as connection:
            connection.row_factory = sqlite3.Row
            info = read_dataset_info(connection)
            labels: list[dict[str, Any]] = []
            if info.get("dataset_type") == "classification":
                rows = connection.execute(
                    """SELECT l.name, count(ca.item_id) AS item_count FROM classification_labels l
                    LEFT JOIN classification_annotations ca ON ca.label_id=l.label_id
                    AND ca.is_current=1 AND ca.status='accepted'
                    GROUP BY l.label_id, l.name ORDER BY l.class_index"""
                ).fetchall()
                labels = [{"name": row["name"], "item_count": row["item_count"]} for row in rows]
            count_row = connection.execute("SELECT count(*) AS count FROM dataset_items").fetchone()
    except (sqlite3.DatabaseError, KeyError, OSError):
        return None
    metadata = dict(info)
    family, version = _family_and_version(metadata, path)
    return {
        "catalog_id": hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:24],
        "name": metadata.get("name") or path.stem,
        "path": str(path),
        "relative_path": _safe_relative(path, root),
        "source_type": "oracle_sqlite",
        "status": metadata.get("lifecycle", "available"),
        "task": metadata.get("dataset_type"),
        "item_count": int(count_row["count"] if count_row else 0),
        "class_count": len(labels),
        "classes": labels,
        "dimensions": {},
        "fingerprint_sha256": metadata.get("fingerprint_sha256"),
        "modified_at": _utc_timestamp(path),
        "size_bytes": path.stat().st_size,
        "training_set_family": family,
        "training_set_version": version,
        "family_id": hashlib.sha256(f"{metadata.get('dataset_type')}:{family.casefold()}".encode()).hexdigest()[:24],
        "warnings": [],
        "dataset_info": metadata,
    }


def scan_training_catalog(root: str | Path) -> dict[str, Any]:
    """Return discoverable sources below ``root`` without changing them."""
    location = Path(root).expanduser().resolve()
    if not location.is_dir():
        raise NotADirectoryError(location)
    entries: list[dict[str, Any]] = []
    for path in location.rglob("*.sqlite"):
        if not path.is_file():
            continue
        entry = _sqlite_entry(path, location)
        if entry:
            entries.append(entry)
    return {
        "root": str(location),
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "entries": sorted(entries, key=lambda entry: (str(entry["name"]).lower(), entry["path"])),
    }
