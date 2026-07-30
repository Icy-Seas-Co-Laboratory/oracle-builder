from __future__ import annotations

import os
import json
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oracle_builder.datasets.schema import (
    dataset_fingerprint,
    read_dataset_info,
    set_dataset_lifecycle,
    validate_database,
)


def checkpoint_path(source: str | Path, timestamp: datetime | None = None) -> Path:
    path = Path(source).expanduser().resolve()
    instant = timestamp or datetime.now(timezone.utc)
    stamp = instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return path.with_name(f"{path.stem}.checkpoint-{stamp}{path.suffix or '.sqlite'}")


def save_checkpoint(
    source: str | Path,
    destination: str | Path | None = None,
    *,
    actor: str | None = None,
) -> dict[str, Any]:
    """Create a consistent frozen copy while leaving the source unchanged."""
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    from oracle_builder.datasets.legacy_roi import migrate_legacy_roi_if_needed

    migrate_legacy_roi_if_needed(source_path)
    destination_path = (
        Path(destination).expanduser().resolve()
        if destination is not None
        else checkpoint_path(source_path)
    )
    if destination_path == source_path:
        raise ValueError("Checkpoint destination must differ from its source")
    if destination_path.exists():
        raise FileExistsError(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(source_path) as source_connection:
        source_report = validate_database(source_connection)
        if not source_report["valid"]:
            raise ValueError(
                "Cannot checkpoint an invalid dataset: "
                + "; ".join(source_report["errors"])
            )
        source_info = read_dataset_info(source_connection)
        if source_info["lifecycle"] == "frozen":
            raise ValueError(
                "Dataset is already frozen; thaw it explicitly before creating "
                "a new checkpoint revision"
            )
        source_fingerprint = dataset_fingerprint(source_connection)
        checkpoint_revision_id = str(uuid.uuid4())
        temporary = tempfile.NamedTemporaryFile(
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            dir=destination_path.parent,
            delete=False,
        )
        temporary_path = Path(temporary.name)
        temporary.close()
        try:
            with sqlite3.connect(temporary_path) as destination_connection:
                source_connection.backup(destination_connection)
                destination_connection.execute(
                    """
                    UPDATE dataset
                    SET revision_id = ?, parent_revision_id = ?
                    WHERE singleton = 1
                    """,
                    (checkpoint_revision_id, source_info["revision_id"]),
                )
                set_dataset_lifecycle(
                    destination_connection,
                    "frozen",
                    actor=actor,
                    details={
                        "operation": "checkpoint",
                        "source_filename": source_path.name,
                        "source_fingerprint": source_fingerprint,
                        "source_revision_id": source_info["revision_id"],
                    },
                )
                destination_connection.commit()
                destination_report = validate_database(destination_connection)
                if not destination_report["valid"]:
                    raise ValueError(
                        "Checkpoint validation failed: "
                        + "; ".join(destination_report["errors"])
                    )
            os.replace(temporary_path, destination_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    return {
        "source": str(source_path),
        "checkpoint": str(destination_path),
        "dataset_id": source_info["dataset_id"],
        "dataset_type": source_info["dataset_type"],
        "revision_id": checkpoint_revision_id,
        "parent_revision_id": source_info["revision_id"],
        "fingerprint": source_fingerprint,
        "lifecycle": "frozen",
    }


def thaw_database(
    path: str | Path,
    *,
    actor: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    database = Path(path).expanduser().resolve()
    from oracle_builder.datasets.legacy_roi import migrate_legacy_roi_if_needed

    migrate_legacy_roi_if_needed(database)
    with sqlite3.connect(database) as connection:
        previous = read_dataset_info(connection)
        if previous["lifecycle"] == "working":
            return {
                "path": str(database),
                "dataset_id": previous["dataset_id"],
                "revision_id": previous["revision_id"],
                "parent_revision_id": previous.get("parent_revision_id"),
                "lifecycle": previous["lifecycle"],
                "fingerprint": dataset_fingerprint(connection),
            }
        info = set_dataset_lifecycle(
            connection,
            "working",
            actor=actor,
            details={"operation": "explicit_thaw", "reason": reason},
        )
        branched_revision_id = str(uuid.uuid4())
        connection.execute(
            """
            UPDATE dataset
            SET revision_id = ?, parent_revision_id = ?
            WHERE singleton = 1
            """,
            (branched_revision_id, previous["revision_id"]),
        )
        connection.execute(
            """
            INSERT INTO dataset_events (
                event_id, dataset_id, revision_id, event_type, created_at,
                actor, details_json
            ) VALUES (?, ?, ?, 'revision.branched', ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                info["dataset_id"],
                branched_revision_id,
                datetime.now(timezone.utc).isoformat(),
                actor,
                json.dumps(
                    {
                        "parent_revision_id": previous["revision_id"],
                        "reason": reason,
                    },
                    sort_keys=True,
                ),
            ),
        )
        connection.commit()
        info = read_dataset_info(connection)
        return {
            "path": str(database),
            "dataset_id": info["dataset_id"],
            "revision_id": info["revision_id"],
            "parent_revision_id": info.get("parent_revision_id"),
            "lifecycle": info["lifecycle"],
            "fingerprint": dataset_fingerprint(connection),
        }
