from __future__ import annotations

import os
import json
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oracle_data_contracts.datasets.schema import (
    dataset_fingerprint,
    read_dataset_info,
    set_dataset_lifecycle,
    validate_database,
    workspace_fingerprint,
)


def checkpoint_path(source: str | Path, timestamp: datetime | None = None) -> Path:
    path = Path(source).expanduser().resolve()
    instant = timestamp or datetime.now(timezone.utc)
    stamp = instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return path.with_name(f"{path.stem}.checkpoint-{stamp}{path.suffix or '.sqlite'}")


def workspace_snapshot_path(
    source: str | Path, timestamp: datetime | None = None
) -> Path:
    path = Path(source).expanduser().resolve()
    instant = timestamp or datetime.now(timezone.utc)
    stamp = instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return path.with_name(f"{path.stem}.snapshot-{stamp}{path.suffix or '.sqlite'}")


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


def save_workspace_snapshot(
    source: str | Path,
    destination: str | Path | None = None,
    *,
    actor: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Create a frozen full-workspace snapshot, including derived evidence."""
    source_path = Path(source).expanduser().resolve()
    output = (
        Path(destination).expanduser().resolve()
        if destination is not None
        else workspace_snapshot_path(source_path)
    )
    with sqlite3.connect(source_path) as connection:
        before = workspace_fingerprint(connection)
    result = save_checkpoint(source_path, output, actor=actor)
    with sqlite3.connect(output) as connection:
        info = read_dataset_info(connection)
        connection.execute(
            """
            INSERT INTO dataset_events (
                event_id, dataset_id, revision_id, event_type, created_at,
                actor, details_json
            ) VALUES (?, ?, ?, 'workspace.snapshot.created', ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                info["dataset_id"],
                info["revision_id"],
                datetime.now(timezone.utc).isoformat(),
                actor,
                json.dumps(
                    {"note": note, "workspace_fingerprint_sha256": before},
                    sort_keys=True,
                ),
            ),
        )
        connection.commit()
        result["workspace_fingerprint"] = workspace_fingerprint(connection)
    result["snapshot"] = result.pop("checkpoint")
    result["snapshot_note"] = note
    return result


def restore_workspace_snapshot(
    snapshot: str | Path,
    destination: str | Path,
    *,
    actor: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Fork a frozen snapshot into a new editable workspace file."""
    snapshot_path = Path(snapshot).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if not snapshot_path.exists():
        raise FileNotFoundError(snapshot_path)
    if destination_path.exists():
        raise FileExistsError(destination_path)
    if snapshot_path == destination_path:
        raise ValueError("Workspace restore destination must differ from snapshot")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(snapshot_path) as source, sqlite3.connect(destination_path) as target:
        report = validate_database(source)
        if not report["valid"]:
            raise ValueError("Cannot restore invalid snapshot: " + "; ".join(report["errors"]))
        source.backup(target)
    result = thaw_database(destination_path, actor=actor, reason=reason or "restore workspace snapshot")
    with sqlite3.connect(destination_path) as connection:
        result["workspace_fingerprint"] = workspace_fingerprint(connection)
    result["snapshot"] = str(snapshot_path)
    return result


def release_training_dataset(
    workspace: str | Path,
    destination: str | Path,
    *,
    name: str | None = None,
    actor: str | None = None,
) -> dict[str, Any]:
    """Materialize a frozen training dataset from a mutable annotation workspace.

    Derived model evidence is intentionally removed. Human annotations, masks,
    reviews, assets, and metadata remain available to the training release.
    """
    workspace_path = Path(workspace).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if not workspace_path.exists():
        raise FileNotFoundError(workspace_path)
    if destination_path.exists():
        raise FileExistsError(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(workspace_path) as source:
        report = validate_database(source)
        if not report["valid"]:
            raise ValueError("Cannot release invalid workspace: " + "; ".join(report["errors"]))
        source_info = read_dataset_info(source)
        if source_info["lifecycle"] != "working":
            raise ValueError("Training releases must be created from an editable workspace")
        source_workspace_fingerprint = workspace_fingerprint(source)
        with sqlite3.connect(destination_path) as target:
            source.backup(target)
    try:
        with sqlite3.connect(destination_path) as connection:
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM model_evidence")
            connection.execute("DELETE FROM evidence_arrays")
            connection.execute("DELETE FROM inference_runs")
            new_dataset_id, new_revision_id = str(uuid.uuid4()), str(uuid.uuid4())
            metadata = json.loads(
                connection.execute("SELECT metadata_json FROM dataset WHERE singleton = 1").fetchone()[0]
                or "{}"
            )
            metadata["training_release"] = {
                "source_workspace_dataset_id": source_info["dataset_id"],
                "source_workspace_revision_id": source_info["revision_id"],
                "source_workspace_fingerprint_sha256": source_workspace_fingerprint,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            for table in (
                "assets", "dataset_items", "metadata_documents", "import_events",
                "dataset_events", "classification_labels", "annotation_labels",
            ):
                if connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (table,)
                ).fetchone():
                    connection.execute(f"UPDATE {table} SET dataset_id = ?", (new_dataset_id,))
            connection.execute(
                """
                UPDATE dataset
                SET dataset_id = ?, revision_id = ?, parent_revision_id = NULL,
                    name = ?, metadata_json = ?
                WHERE singleton = 1
                """,
                (new_dataset_id, new_revision_id, name or source_info["name"], json.dumps(metadata, sort_keys=True)),
            )
            set_dataset_lifecycle(
                connection,
                "frozen",
                actor=actor,
                details={
                    "operation": "training_release",
                    "source_workspace_dataset_id": source_info["dataset_id"],
                    "source_workspace_fingerprint_sha256": source_workspace_fingerprint,
                    "excluded": ["inference_runs", "model_evidence", "evidence_arrays"],
                },
            )
            violations = connection.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise ValueError(f"Training release has {len(violations)} foreign-key violation(s)")
            connection.commit()
            connection.execute("PRAGMA foreign_keys = ON")
            output_info = read_dataset_info(connection)
            fingerprint = dataset_fingerprint(connection)
    except Exception:
        destination_path.unlink(missing_ok=True)
        raise
    return {
        "workspace": str(workspace_path),
        "release": str(destination_path),
        "dataset_id": output_info["dataset_id"],
        "revision_id": output_info["revision_id"],
        "fingerprint": fingerprint,
        "source_workspace_fingerprint": source_workspace_fingerprint,
        "lifecycle": output_info["lifecycle"],
    }
