from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from oracle_builder.datasets.schema import (
    DatasetSchemaError,
    read_dataset_info,
    utc_now,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


METADATA_SUFFIXES = {".json", ".toml", ".yaml", ".yml"}


def parse_metadata_document(
    document: str | Path,
    *,
    name: str | None = None,
) -> dict[str, Any]:
    """Parse a supported metadata document while preserving its source text."""
    path = Path(document).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Metadata document does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix not in METADATA_SUFFIXES:
        supported = ", ".join(sorted(METADATA_SUFFIXES))
        raise ValueError(
            f"Unsupported metadata format {suffix or '<none>'!r}; "
            f"expected one of: {supported}"
        )
    raw_text = path.read_text(encoding="utf-8")
    try:
        if suffix == ".json":
            parsed = json.loads(raw_text)
        elif suffix == ".toml":
            with path.open("rb") as handle:
                parsed = tomllib.load(handle)
        else:
            if yaml is None:
                raise RuntimeError(
                    "YAML metadata requires PyYAML; install the project dependencies"
                )
            parsed = yaml.safe_load(raw_text)
    except Exception as exc:
        raise ValueError(f"Invalid metadata document {path}: {exc}") from exc

    resolved_name = str(name).strip() if name is not None else path.name
    if not resolved_name:
        raise ValueError("Metadata document name cannot be empty")
    return {
        "metadata_name": resolved_name,
        "source_filename": path.name,
        "source_format": suffix.lstrip("."),
        "metadata_json": json.dumps(parsed, sort_keys=True, default=str),
        "parsed": parsed,
        "raw_text": raw_text,
        "sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "source_path": str(path),
    }


def discover_metadata_documents(root: str | Path) -> list[dict[str, Any]]:
    """Parse supported metadata files directly beneath a folder."""
    directory = Path(root).expanduser().resolve()
    return [
        parse_metadata_document(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in METADATA_SUFFIXES
    ]


def add_metadata_document(
    database: str | Path,
    document: str | Path,
    *,
    name: str | None = None,
    actor: str | None = None,
) -> dict[str, Any]:
    """Add or replace one metadata document in an editable V1 dataset."""
    database_path = Path(database).expanduser().resolve()
    parsed = parse_metadata_document(document, name=name)
    with sqlite3.connect(database_path) as connection:
        info = read_dataset_info(connection)
        if info["lifecycle"] != "working":
            raise DatasetSchemaError(
                f"Dataset is {info['lifecycle']!r}; thaw it before attaching metadata"
            )
        existing = connection.execute(
            """
            SELECT document_id, sha256
            FROM metadata_documents
            WHERE dataset_id = ? AND name = ?
            """,
            (info["dataset_id"], parsed["metadata_name"]),
        ).fetchone()
        document_id = (
            str(existing[0])
            if existing
            else str(
                uuid.uuid5(
                    uuid.UUID(info["dataset_id"]),
                    f"metadata:{parsed['metadata_name']}",
                )
            )
        )
        now = utc_now()
        connection.execute(
            """
            INSERT INTO metadata_documents (
                document_id, dataset_id, name, source_filename, source_format,
                parsed_json, raw_text, sha256, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id, name) DO UPDATE SET
                source_filename = excluded.source_filename,
                source_format = excluded.source_format,
                parsed_json = excluded.parsed_json,
                raw_text = excluded.raw_text,
                sha256 = excluded.sha256,
                created_at = excluded.created_at
            """,
            (
                document_id,
                info["dataset_id"],
                parsed["metadata_name"],
                parsed["source_filename"],
                parsed["source_format"],
                parsed["metadata_json"],
                parsed["raw_text"],
                parsed["sha256"],
                now,
            ),
        )
        action = "updated" if existing else "added"
        details = {
            "document_id": document_id,
            "name": parsed["metadata_name"],
            "source_filename": parsed["source_filename"],
            "source_format": parsed["source_format"],
            "sha256": parsed["sha256"],
        }
        if existing:
            details["previous_sha256"] = str(existing[1])
        connection.execute(
            """
            INSERT INTO dataset_events (
                event_id, dataset_id, revision_id, event_type, created_at,
                actor, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                info["dataset_id"],
                info["revision_id"],
                f"metadata_document.{action}",
                now,
                actor,
                json.dumps(details, sort_keys=True),
            ),
        )
        connection.commit()
        return {
            "action": action,
            "actor": actor,
            "database": str(database_path),
            "dataset_id": info["dataset_id"],
            **details,
        }
