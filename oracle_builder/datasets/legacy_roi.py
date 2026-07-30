from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from oracle_builder.data.decoders import decode_blob, encode_npy
from oracle_builder.datasets.repository import SQLiteDatasetRepository
from oracle_builder.datasets.schema import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    DatasetSchemaError,
    initialize_database,
    read_dataset_info,
    utc_now,
    validate_database,
)


LEGACY_ROI_SCHEMA = "oracle_builder_legacy_roi_samples"


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def is_legacy_roi_database(connection: sqlite3.Connection) -> bool:
    tables = _table_names(connection)
    if "ob_schema" in tables or "samples" not in tables:
        return False
    if "mask_annotations" in tables:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(mask_annotations)")
        }
        return "sample_uuid" in columns and "mask_blob" in columns
    sample_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(samples)")
    }
    return "input_aux_blob" in sample_columns and "output_blob" in sample_columns


def inspect_dataset_kind(path: str | Path) -> str:
    database = Path(path).expanduser().resolve()
    if not database.exists():
        return "missing"
    with sqlite3.connect(database) as connection:
        tables = _table_names(connection)
        if "ob_schema" in tables:
            try:
                return str(read_dataset_info(connection)["dataset_type"])
            except DatasetSchemaError:
                return "unsupported"
        if is_legacy_roi_database(connection):
            return "legacy_mask_refinement"
        if "samples" in tables:
            return "legacy_generic"
        return "unknown"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _backup_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.pre-v1-{_timestamp()}{path.suffix}")


def _shape(value: str | None) -> list[int] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, list):
        return None
    return [int(part) for part in parsed]


def _metadata(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"legacy_metadata_text": value}
    return parsed if isinstance(parsed, dict) else {"legacy_metadata": parsed}


def _row_get(row: sqlite3.Row | dict[str, Any], key: str, default: Any = None) -> Any:
    return row[key] if key in row.keys() else default


def _json_text(value: str | None) -> str:
    if not value:
        return "{}"
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = {"legacy_raw_text": value}
    return json.dumps(parsed, sort_keys=True, default=str)


def _media_type(encoding: str) -> str:
    normalized = encoding.lower()
    if normalized == "png":
        return "image/png"
    if normalized in {"jpg", "jpeg"}:
        return "image/jpeg"
    if normalized == "npy":
        return "application/x-npy"
    return "application/octet-stream"


def _add_asset(
    repository: SQLiteDatasetRepository,
    blob: bytes,
    encoding: str,
    dimensions: str | None,
    *,
    dtype: str | None = None,
):
    return repository.add_asset(
        blob,
        encoding=encoding,
        media_type=_media_type(encoding),
        shape=_shape(dimensions),
        dtype=dtype,
    )


def _decode_legacy_input(row: sqlite3.Row) -> tuple[
    tuple[bytes, str, str | None, str | None],
    tuple[bytes, str, str | None, str | None] | None,
]:
    input_blob = row["input_blob"]
    input_encoding = row["input_blob_encoding"]
    input_dimensions = row["input_blob_dimensions"]
    if input_blob is None or not input_encoding:
        raise ValueError(f"Legacy ROI {row['uuid']!r} has no decodable input")
    decoded = np.asarray(
        decode_blob(input_blob, input_encoding, input_dimensions)
    )
    embedded_candidate = None
    if decoded.ndim == 3 and decoded.shape[-1] == 2:
        roi = decoded[..., 0]
        embedded_candidate = decoded[..., 1]
        image = (
            encode_npy(roi),
            "npy",
            json.dumps(list(roi.shape)),
            str(roi.dtype),
        )
    else:
        image = (
            bytes(input_blob),
            str(input_encoding),
            input_dimensions,
            str(decoded.dtype),
        )

    keys = set(row.keys())
    aux_blob = row["input_aux_blob"] if "input_aux_blob" in keys else None
    aux_encoding = (
        row["input_aux_blob_encoding"]
        if "input_aux_blob_encoding" in keys
        else None
    )
    aux_dimensions = (
        row["input_aux_blob_dimensions"]
        if "input_aux_blob_dimensions" in keys
        else None
    )
    if aux_blob is not None and aux_encoding:
        candidate = (
            bytes(aux_blob),
            str(aux_encoding),
            aux_dimensions,
            None,
        )
    elif embedded_candidate is not None:
        candidate = (
            encode_npy(embedded_candidate),
            "npy",
            json.dumps(list(embedded_candidate.shape)),
            str(embedded_candidate.dtype),
        )
    else:
        candidate = None
    return image, candidate


def _copy_unknown_tables(
    source: sqlite3.Connection, destination: sqlite3.Connection
) -> list[str]:
    reserved = {
        "samples",
        "mask_annotations",
        "sqlite_sequence",
        *_table_names(destination),
    }
    copied = []
    objects = source.execute(
        """
        SELECT type, name, sql
        FROM sqlite_master
        WHERE type IN ('table', 'index', 'trigger') AND sql IS NOT NULL
        ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 1 ELSE 2 END, name
        """
    ).fetchall()
    copied_tables: set[str] = set()
    for object_type, name, sql in objects:
        if object_type != "table" or name in reserved:
            continue
        destination.execute(sql)
        columns = [
            str(row[1])
            for row in source.execute(f'PRAGMA table_info("{name}")')
        ]
        if columns:
            placeholders = ", ".join("?" for _ in columns)
            quoted = ", ".join(f'"{column}"' for column in columns)
            cursor = source.execute(f'SELECT {quoted} FROM "{name}"')
            while rows := cursor.fetchmany(1000):
                destination.executemany(
                    f'INSERT INTO "{name}" ({quoted}) VALUES ({placeholders})',
                    rows,
                )
        copied_tables.add(str(name))
        copied.append(str(name))
    for object_type, name, sql in objects:
        if object_type == "table":
            continue
        table_name = source.execute(
            "SELECT tbl_name FROM sqlite_master WHERE name = ?", (name,)
        ).fetchone()[0]
        if table_name in copied_tables:
            destination.execute(sql)
    return copied


def _migrate_contents(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    *,
    dataset_id: str,
    revision_id: str,
    name: str,
    source_sha256: str,
) -> dict[str, Any]:
    source.row_factory = sqlite3.Row
    initialize_database(
        destination,
        "mask_refinement",
        dataset_id=dataset_id,
        revision_id=revision_id,
        name=name,
        metadata={
            "migration": {
                "source_schema": LEGACY_ROI_SCHEMA,
                "source_sha256": source_sha256,
            }
        },
    )
    repository = SQLiteDatasetRepository(destination)
    tables = _table_names(source)
    sample_count = int(
        source.execute("SELECT count(*) FROM samples").fetchone()[0]
    )
    annotation_count = (
        int(source.execute("SELECT count(*) FROM mask_annotations").fetchone()[0])
        if "mask_annotations" in tables
        else 0
    )
    orphan_ids = (
        [
            str(row[0])
            for row in source.execute(
                """
                SELECT DISTINCT ma.sample_uuid
                FROM mask_annotations ma
                LEFT JOIN samples s ON s.uuid = ma.sample_uuid
                WHERE s.uuid IS NULL
                ORDER BY ma.sample_uuid
                LIMIT 5
                """
            )
        ]
        if "mask_annotations" in tables
        else []
    )
    if orphan_ids:
        raise ValueError(
            "Legacy mask annotations reference missing samples: "
            + ", ".join(orphan_ids)
        )

    synthetic_annotations = 0
    dropped_split_assignments = 0
    sample_cursor = source.execute("SELECT * FROM samples ORDER BY uuid")
    for row in sample_cursor:
        item_id = str(row["uuid"])
        image, candidate = _decode_legacy_input(row)
        image_asset = _add_asset(
            repository, image[0], image[1], image[2], dtype=image[3]
        )
        candidate_asset_id = None
        if candidate is not None:
            candidate_asset_id = _add_asset(
                repository,
                candidate[0],
                candidate[1],
                candidate[2],
                dtype=candidate[3],
            ).asset_id
        metadata = _metadata(row["metadata_json"])
        legacy_split = _row_get(row, "split")
        if legacy_split is not None and str(legacy_split).strip():
            dropped_split_assignments += 1
        source_key = str(
            metadata.get("pelagia_detection_id")
            or metadata.get("source_path")
            or item_id
        )
        repository.add_item(
            item_id=item_id,
            source_key=source_key,
            sample_weight=_row_get(row, "sample_weight"),
            metadata=metadata,
        )
        repository.add_mask_item(
            item_id=item_id,
            image_asset_id=image_asset.asset_id,
            candidate_mask_asset_id=candidate_asset_id,
        )
        historical = (
            source.execute(
                """
                SELECT * FROM mask_annotations
                WHERE sample_uuid = ?
                ORDER BY created_at, annotation_id
                """,
                (item_id,),
            ).fetchall()
            if "mask_annotations" in tables
            else []
        )
        output_is_historical = (
            row["output_blob"] is not None
            and any(
                bytes(annotation["mask_blob"]) == bytes(row["output_blob"])
                and str(annotation["mask_blob_encoding"]).lower()
                == str(row["output_blob_encoding"]).lower()
                for annotation in historical
            )
        )
        if row["output_blob"] is not None and not output_is_historical:
            synthetic_annotations += 1
            synthetic_created_at = (
                metadata.get("mask_builder", {}).get("last_saved_at")
                if isinstance(metadata.get("mask_builder"), dict)
                else None
            ) or "1970-01-01T00:00:00+00:00"
            historical = list(historical) + [
                {
                    "annotation_id": str(
                        uuid.uuid5(
                            uuid.UUID(dataset_id),
                            f"legacy-output:{item_id}",
                        )
                    ),
                    "created_at": synthetic_created_at,
                    "annotator": None,
                    "mask_blob": row["output_blob"],
                    "mask_blob_encoding": row["output_blob_encoding"],
                    "mask_blob_dimensions": row["output_blob_dimensions"],
                    "method": "legacy_samples_output",
                    "parameters_json": "{}",
                    "validation_json": "{}",
                    "accepted": 1,
                    "notes": "Synthesized from legacy samples.output_blob.",
                }
            ]
        accepted_positions = [
            index
            for index, annotation in enumerate(historical)
            if bool(_row_get(annotation, "accepted", 1))
        ]
        current_position = accepted_positions[-1] if accepted_positions else None
        parent_id = None
        for index, annotation in enumerate(historical):
            mask_asset = _add_asset(
                repository,
                bytes(annotation["mask_blob"]),
                str(annotation["mask_blob_encoding"]),
                _row_get(annotation, "mask_blob_dimensions"),
                dtype="uint8",
            )
            annotation_id = str(annotation["annotation_id"])
            destination.execute(
                """
                INSERT INTO mask_annotations (
                    annotation_id, item_id, mask_asset_id, created_at, annotator,
                    method, parameters_json, validation_json, status, is_current,
                    parent_annotation_id, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    annotation_id,
                    item_id,
                    mask_asset.asset_id,
                    str(_row_get(annotation, "created_at") or utc_now()),
                    _row_get(annotation, "annotator"),
                    _row_get(annotation, "method"),
                    _json_text(_row_get(annotation, "parameters_json")),
                    _json_text(_row_get(annotation, "validation_json")),
                    "accepted"
                    if bool(_row_get(annotation, "accepted", 1))
                    else "rejected",
                    1 if index == current_position else 0,
                    parent_id,
                    _row_get(annotation, "notes"),
                ),
            )
            parent_id = annotation_id

    copied_tables = _copy_unknown_tables(source, destination)
    info = read_dataset_info(destination)
    destination.execute(
        """
        INSERT INTO dataset_events (
            event_id, dataset_id, revision_id, event_type, created_at, actor,
            details_json
        ) VALUES (?, ?, ?, 'migrated', ?, 'oracle-builder', ?)
        """,
        (
            str(uuid.uuid4()),
            info["dataset_id"],
            info["revision_id"],
            utc_now(),
            json.dumps(
                {
                    "source_schema": LEGACY_ROI_SCHEMA,
                    "source_sha256": source_sha256,
                    "samples": sample_count,
                    "annotations": annotation_count,
                    "synthetic_annotations": synthetic_annotations,
                    "dropped_legacy_split_assignments": dropped_split_assignments,
                    "copied_additive_tables": copied_tables,
                },
                sort_keys=True,
            ),
        ),
    )
    return {
        "sample_count": sample_count,
        "legacy_annotation_count": annotation_count,
        "synthetic_annotation_count": synthetic_annotations,
        "dropped_legacy_split_assignments": dropped_split_assignments,
        "copied_additive_tables": copied_tables,
    }


def migrate_legacy_roi_database(
    path: str | Path,
    *,
    backup_path: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically replace a legacy ROI database after creating a safe backup."""
    database = Path(path).expanduser().resolve()
    if not database.exists():
        raise FileNotFoundError(database)
    backup = (
        Path(backup_path).expanduser().resolve()
        if backup_path is not None
        else _backup_path(database)
    )
    if backup.exists():
        raise FileExistsError(backup)
    with sqlite3.connect(database) as source:
        if not is_legacy_roi_database(source):
            raise DatasetSchemaError(f"Not a recognized legacy ROI database: {database}")
        with sqlite3.connect(backup) as backup_connection:
            source.backup(backup_connection)
    digest = hashlib.sha256()
    with backup.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    source_sha256 = digest.hexdigest()
    dataset_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"oracle-builder:{LEGACY_ROI_SCHEMA}:{source_sha256}",
        )
    )
    revision_id = str(uuid.uuid5(uuid.UUID(dataset_id), "schema-v1-migration"))
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{database.stem}.migration-", dir=database.parent))
    migrated = temp_dir / database.name
    try:
        with sqlite3.connect(backup) as source, sqlite3.connect(migrated) as destination:
            destination.execute("PRAGMA foreign_keys = ON")
            counts = _migrate_contents(
                source,
                destination,
                dataset_id=dataset_id,
                revision_id=revision_id,
                name=database.stem,
                source_sha256=source_sha256,
            )
            destination.commit()
            report = validate_database(destination)
            if not report["valid"]:
                raise ValueError(
                    "Migrated database failed validation: "
                    + "; ".join(report["errors"])
                )
            migrated_counts = {
                "samples": destination.execute(
                    "SELECT count(*) FROM dataset_items"
                ).fetchone()[0],
                "annotations": destination.execute(
                    "SELECT count(*) FROM mask_annotations"
                ).fetchone()[0],
            }
            expected_annotations = (
                counts["legacy_annotation_count"]
                + counts["synthetic_annotation_count"]
            )
            if migrated_counts != {
                "samples": counts["sample_count"],
                "annotations": expected_annotations,
            }:
                raise ValueError(
                    f"Migration count mismatch: expected "
                    f"{counts['sample_count']} samples/{expected_annotations} annotations, "
                    f"observed {migrated_counts['samples']}/"
                    f"{migrated_counts['annotations']}"
                )
        os.chmod(migrated, database.stat().st_mode)
        os.replace(migrated, database)
        for suffix in ("-wal", "-shm"):
            database.with_name(database.name + suffix).unlink(missing_ok=True)
    except Exception:
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    result = {
        "migrated": True,
        "database": str(database),
        "backup": str(backup),
        "source_schema": LEGACY_ROI_SCHEMA,
        "target_schema": SCHEMA_NAME,
        "target_schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "revision_id": revision_id,
        "source_sha256": source_sha256,
        **counts,
    }
    report_path = database.with_name(
        f"{database.stem}.migration-{_timestamp()}.json"
    )
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["report"] = str(report_path)
    return result


def ensure_mask_refinement_database(
    path: str | Path,
    *,
    auto_migrate: bool = True,
) -> dict[str, Any] | None:
    database = Path(path).expanduser().resolve()
    if not database.exists():
        return None
    kind = inspect_dataset_kind(database)
    if kind == "mask_refinement":
        return None
    if kind == "legacy_mask_refinement":
        if not auto_migrate:
            raise DatasetSchemaError(
                f"Legacy ROI database requires migration: {database}"
            )
        result = migrate_legacy_roi_database(database)
        warnings.warn(
            "Migrated legacy ROI database to Oracle Builder V1. "
            f"Original preserved at {result['backup']}",
            RuntimeWarning,
            stacklevel=2,
        )
        return result
    if kind == "legacy_generic":
        raise DatasetSchemaError(
            "Legacy samples-table database is not identifiable as mask refinement; "
            "use an explicit importer or migration command."
        )
    if kind not in {"missing", "unknown"}:
        raise DatasetSchemaError(
            f"Expected mask_refinement dataset, found {kind!r}: {database}"
        )
    return None


def migrate_legacy_roi_if_needed(path: str | Path) -> dict[str, Any] | None:
    """Migrate recognized legacy ROI input and ignore every other dataset kind."""
    database = Path(path).expanduser().resolve()
    if not database.exists() or inspect_dataset_kind(database) != "legacy_mask_refinement":
        return None
    result = migrate_legacy_roi_database(database)
    warnings.warn(
        "Migrated legacy ROI database to Oracle Builder V1. "
        f"Original preserved at {result['backup']}",
        RuntimeWarning,
        stacklevel=2,
    )
    return result
