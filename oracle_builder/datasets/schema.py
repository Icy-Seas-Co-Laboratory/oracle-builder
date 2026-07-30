from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any


SCHEMA_NAME = "oracle_builder_dataset"
SCHEMA_VERSION = "1.1.0"
DATASET_TYPES = {"classification", "mask_refinement"}
LIFECYCLE_STATES = {"working", "frozen", "deprecated"}


class DatasetSchemaError(ValueError):
    """Raised when a database does not satisfy the Oracle Builder dataset contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_dataset_type(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "segmentation": "mask_refinement",
        "mask_refining": "mask_refinement",
        "mask_refinement": "mask_refinement",
        "classification": "classification",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise DatasetSchemaError(
            f"dataset_type must be one of: {', '.join(sorted(DATASET_TYPES))}"
        ) from exc


COMMON_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ob_schema (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_name TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    dataset_id TEXT NOT NULL UNIQUE,
    revision_id TEXT NOT NULL UNIQUE,
    parent_revision_id TEXT,
    dataset_type TEXT NOT NULL CHECK (dataset_type IN ('classification', 'mask_refinement')),
    name TEXT NOT NULL,
    title TEXT,
    description TEXT,
    version TEXT,
    lifecycle TEXT NOT NULL DEFAULT 'working'
        CHECK (lifecycle IN ('working', 'frozen', 'deprecated')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    frozen_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    payload BLOB,
    external_uri TEXT,
    encoding TEXT NOT NULL,
    media_type TEXT,
    shape_json TEXT,
    dtype TEXT,
    original_filename TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES dataset(dataset_id),
    CHECK (payload IS NOT NULL OR external_uri IS NOT NULL),
    UNIQUE (dataset_id, content_sha256, encoding)
);

CREATE TABLE IF NOT EXISTS dataset_items (
    item_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    sample_weight REAL,
    source_key TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES dataset(dataset_id),
    UNIQUE (dataset_id, source_key)
);

CREATE TABLE IF NOT EXISTS metadata_documents (
    document_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    name TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    source_format TEXT NOT NULL,
    parsed_json TEXT NOT NULL,
    raw_text TEXT,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES dataset(dataset_id),
    UNIQUE (dataset_id, name)
);

CREATE TABLE IF NOT EXISTS import_events (
    import_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    importer TEXT NOT NULL,
    source_uri TEXT,
    options_json TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES dataset(dataset_id)
);

CREATE TABLE IF NOT EXISTS dataset_events (
    event_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    actor TEXT,
    details_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (dataset_id) REFERENCES dataset(dataset_id)
);

CREATE INDEX IF NOT EXISTS idx_assets_dataset_sha
    ON assets(dataset_id, content_sha256);
"""


CLASSIFICATION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS classification_labels (
    label_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    class_index INTEGER NOT NULL CHECK (class_index >= 0),
    name TEXT NOT NULL,
    parent_label_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (dataset_id) REFERENCES dataset(dataset_id),
    FOREIGN KEY (parent_label_id) REFERENCES classification_labels(label_id),
    UNIQUE (dataset_id, class_index),
    UNIQUE (dataset_id, name)
);

CREATE TABLE IF NOT EXISTS classification_items (
    item_id TEXT PRIMARY KEY,
    image_asset_id TEXT NOT NULL,
    FOREIGN KEY (item_id) REFERENCES dataset_items(item_id) ON DELETE CASCADE,
    FOREIGN KEY (image_asset_id) REFERENCES assets(asset_id)
);

CREATE TABLE IF NOT EXISTS classification_annotations (
    annotation_id TEXT PRIMARY KEY,
    item_id TEXT NOT NULL,
    label_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    annotator TEXT,
    source TEXT,
    confidence REAL,
    status TEXT NOT NULL DEFAULT 'accepted'
        CHECK (status IN ('candidate', 'accepted', 'rejected', 'deprecated')),
    is_current INTEGER NOT NULL DEFAULT 1 CHECK (is_current IN (0, 1)),
    parent_annotation_id TEXT,
    notes TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (item_id) REFERENCES classification_items(item_id) ON DELETE CASCADE,
    FOREIGN KEY (label_id) REFERENCES classification_labels(label_id),
    FOREIGN KEY (parent_annotation_id) REFERENCES classification_annotations(annotation_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_classification_current_annotation
    ON classification_annotations(item_id)
    WHERE is_current = 1;
CREATE INDEX IF NOT EXISTS idx_classification_annotation_label
    ON classification_annotations(label_id);
"""


MASK_REFINEMENT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mask_refinement_items (
    item_id TEXT PRIMARY KEY,
    image_asset_id TEXT NOT NULL,
    candidate_mask_asset_id TEXT,
    FOREIGN KEY (item_id) REFERENCES dataset_items(item_id) ON DELETE CASCADE,
    FOREIGN KEY (image_asset_id) REFERENCES assets(asset_id),
    FOREIGN KEY (candidate_mask_asset_id) REFERENCES assets(asset_id)
);

CREATE TABLE IF NOT EXISTS mask_annotations (
    annotation_id TEXT PRIMARY KEY,
    item_id TEXT NOT NULL,
    mask_asset_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    annotator TEXT,
    method TEXT,
    parameters_json TEXT NOT NULL DEFAULT '{}',
    validation_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'accepted'
        CHECK (status IN ('candidate', 'accepted', 'rejected', 'deprecated')),
    is_current INTEGER NOT NULL DEFAULT 1 CHECK (is_current IN (0, 1)),
    parent_annotation_id TEXT,
    notes TEXT,
    FOREIGN KEY (item_id) REFERENCES mask_refinement_items(item_id) ON DELETE CASCADE,
    FOREIGN KEY (mask_asset_id) REFERENCES assets(asset_id),
    FOREIGN KEY (parent_annotation_id) REFERENCES mask_annotations(annotation_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_mask_current_annotation
    ON mask_annotations(item_id)
    WHERE is_current = 1;
"""


MUTABLE_TABLES = (
    "ob_schema",
    "assets",
    "dataset_items",
    "metadata_documents",
    "classification_labels",
    "classification_items",
    "classification_annotations",
    "mask_refinement_items",
    "mask_annotations",
)


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        is not None
    )


def _install_freeze_triggers(connection: sqlite3.Connection) -> None:
    for table in MUTABLE_TABLES:
        if not _table_exists(connection, table):
            continue
        for operation in ("INSERT", "UPDATE", "DELETE"):
            trigger = f"freeze_{table}_{operation.lower()}"
            connection.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS {trigger}
                BEFORE {operation} ON {table}
                WHEN (SELECT lifecycle FROM dataset WHERE singleton = 1) = 'frozen'
                BEGIN
                    SELECT RAISE(ABORT, 'dataset is frozen; thaw it before editing');
                END
                """
            )
    connection.executescript(
        """
        CREATE TRIGGER IF NOT EXISTS freeze_dataset_semantic_update
        BEFORE UPDATE OF dataset_id, revision_id, parent_revision_id,
                         dataset_type, name, title, description, version,
                         metadata_json
        ON dataset
        WHEN OLD.lifecycle = 'frozen'
        BEGIN
            SELECT RAISE(ABORT, 'dataset is frozen; thaw it before editing');
        END;

        CREATE TRIGGER IF NOT EXISTS freeze_dataset_delete
        BEFORE DELETE ON dataset
        WHEN OLD.lifecycle = 'frozen'
        BEGIN
            SELECT RAISE(ABORT, 'dataset is frozen; thaw it before editing');
        END;
        """
    )


def _install_touch_triggers(connection: sqlite3.Connection) -> None:
    for table in MUTABLE_TABLES:
        if not _table_exists(connection, table):
            continue
        for operation in ("INSERT", "UPDATE", "DELETE"):
            trigger = f"touch_dataset_{table}_{operation.lower()}"
            connection.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS {trigger}
                AFTER {operation} ON {table}
                BEGIN
                    UPDATE dataset
                    SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE singleton = 1;
                END
                """
            )


def _migrate_schema_if_needed(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "ob_schema"):
        return
    schema_row = connection.execute(
        "SELECT schema_name, schema_version FROM ob_schema WHERE singleton = 1"
    ).fetchone()
    if schema_row is None or tuple(schema_row) == (SCHEMA_NAME, SCHEMA_VERSION):
        return
    if tuple(schema_row) != (SCHEMA_NAME, "1.0.0"):
        raise DatasetSchemaError(
            f"Unsupported dataset schema {schema_row[0]!r} version "
            f"{schema_row[1]!r}; expected {SCHEMA_NAME!r} {SCHEMA_VERSION!r}"
        )
    if connection.in_transaction:
        raise DatasetSchemaError(
            "Dataset schema migration requires a connection with no active transaction"
        )
    foreign_keys = int(connection.execute("PRAGMA foreign_keys").fetchone()[0])
    connection.execute("PRAGMA foreign_keys = OFF")
    try:
        connection.execute("BEGIN IMMEDIATE")
        trigger_names = [
            str(row[0])
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'trigger'
                  AND (name LIKE 'freeze_%' OR name LIKE 'touch_dataset_%')
                """
            )
        ]
        for trigger in trigger_names:
            escaped = trigger.replace('"', '""')
            connection.execute(f'DROP TRIGGER "{escaped}"')
        connection.execute("DROP INDEX IF EXISTS idx_items_dataset_split")
        removed_split_assignments = int(
            connection.execute(
                "SELECT count(*) FROM dataset_items WHERE split IS NOT NULL"
            ).fetchone()[0]
        )
        connection.execute(
            """
            CREATE TABLE dataset_items_v1_1 (
                item_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                sample_weight REAL,
                source_key TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES dataset(dataset_id),
                UNIQUE (dataset_id, source_key)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO dataset_items_v1_1 (
                item_id, dataset_id, sample_weight, source_key,
                metadata_json, created_at, updated_at
            )
            SELECT item_id, dataset_id, sample_weight, source_key,
                   metadata_json, created_at, updated_at
            FROM dataset_items
            """
        )
        removed_metadata_hints = 0
        for item_id, metadata_json in connection.execute(
            "SELECT item_id, metadata_json FROM dataset_items_v1_1"
        ).fetchall():
            try:
                item_metadata = json.loads(metadata_json or "{}")
            except json.JSONDecodeError:
                continue
            if not isinstance(item_metadata, dict):
                continue
            changed = False
            for key in ("legacy_split_hint", "import_split_hint"):
                if key in item_metadata:
                    item_metadata.pop(key)
                    removed_metadata_hints += 1
                    changed = True
            if changed:
                connection.execute(
                    """
                    UPDATE dataset_items_v1_1
                    SET metadata_json = ?
                    WHERE item_id = ?
                    """,
                    (json.dumps(item_metadata, sort_keys=True, default=str), item_id),
                )
        connection.execute("DROP TABLE dataset_items")
        connection.execute(
            "ALTER TABLE dataset_items_v1_1 RENAME TO dataset_items"
        )
        connection.execute(
            "UPDATE ob_schema SET schema_version = ? WHERE singleton = 1",
            (SCHEMA_VERSION,),
        )
        dataset_id, revision_id = connection.execute(
            "SELECT dataset_id, revision_id FROM dataset WHERE singleton = 1"
        ).fetchone()
        connection.execute(
            """
            INSERT INTO dataset_events (
                event_id, dataset_id, revision_id, event_type,
                created_at, actor, details_json
            ) VALUES (?, ?, ?, 'schema.migrated', ?, 'oracle-builder', ?)
            """,
            (
                str(uuid.uuid4()),
                dataset_id,
                revision_id,
                utc_now(),
                json.dumps(
                    {
                        "from_version": "1.0.0",
                        "to_version": SCHEMA_VERSION,
                        "removed_column": "dataset_items.split",
                        "removed_non_null_assignments": removed_split_assignments,
                        "removed_metadata_split_hints": removed_metadata_hints,
                    },
                    sort_keys=True,
                ),
            ),
        )
        _install_freeze_triggers(connection)
        _install_touch_triggers(connection)
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise DatasetSchemaError(
                f"Schema migration produced {len(violations)} foreign-key violation(s)"
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.execute(f"PRAGMA foreign_keys = {foreign_keys}")


def initialize_database(
    connection: sqlite3.Connection,
    dataset_type: str,
    *,
    dataset_id: str | None = None,
    name: str | None = None,
    title: str | None = None,
    description: str | None = None,
    version: str | None = None,
    metadata: dict[str, Any] | None = None,
    revision_id: str | None = None,
    parent_revision_id: str | None = None,
) -> dict[str, Any]:
    """Create or validate a V1 single-dataset SQLite database."""
    resolved_type = normalize_dataset_type(dataset_type)
    connection.execute("PRAGMA foreign_keys = ON")
    if _table_exists(connection, "samples") and not _table_exists(connection, "ob_schema"):
        raise DatasetSchemaError(
            "Legacy generic samples-table database detected. V1 intentionally requires "
            "an explicit migration or a fresh deterministic import."
        )
    connection.executescript(COMMON_SCHEMA_SQL)
    _migrate_schema_if_needed(connection)
    schema_row = connection.execute(
        "SELECT schema_name, schema_version FROM ob_schema WHERE singleton = 1"
    ).fetchone()
    if schema_row is None:
        connection.execute(
            "INSERT INTO ob_schema VALUES (1, ?, ?, ?)",
            (SCHEMA_NAME, SCHEMA_VERSION, utc_now()),
        )
    elif tuple(schema_row) != (SCHEMA_NAME, SCHEMA_VERSION):
        raise DatasetSchemaError(
            f"Unsupported dataset schema {schema_row[0]!r} version {schema_row[1]!r}; "
            f"expected {SCHEMA_NAME!r} {SCHEMA_VERSION!r}"
        )

    existing = connection.execute(
        """
        SELECT dataset_id, dataset_type, revision_id, parent_revision_id
        FROM dataset WHERE singleton = 1
        """
    ).fetchone()
    if existing is None:
        resolved_id = str(uuid.UUID(dataset_id)) if dataset_id else str(uuid.uuid4())
        now = utc_now()
        connection.execute(
            """
            INSERT INTO dataset (
                singleton, dataset_id, revision_id, parent_revision_id,
                dataset_type, name, title, description,
                version, lifecycle, created_at, updated_at, frozen_at, metadata_json
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, 'working', ?, ?, NULL, ?)
            """,
            (
                resolved_id,
                str(uuid.UUID(revision_id)) if revision_id else str(uuid.uuid4()),
                str(uuid.UUID(parent_revision_id)) if parent_revision_id else None,
                resolved_type,
                name or f"untitled-{resolved_type}-dataset",
                title,
                description,
                version,
                now,
                now,
                json.dumps(metadata or {}, sort_keys=True, default=str),
            ),
        )
        connection.execute(
            """
            INSERT INTO dataset_events
                (event_id, dataset_id, revision_id, event_type, created_at,
                 actor, details_json)
            VALUES (?, ?, ?, 'created', ?, NULL, ?)
            """,
            (
                str(uuid.uuid4()),
                resolved_id,
                str(
                    connection.execute(
                        "SELECT revision_id FROM dataset WHERE singleton = 1"
                    ).fetchone()[0]
                ),
                now,
                json.dumps({"schema_version": SCHEMA_VERSION}, sort_keys=True),
            ),
        )
    elif existing[1] != resolved_type:
        raise DatasetSchemaError(
            f"Database contains a {existing[1]!r} dataset, not {resolved_type!r}"
        )
    elif dataset_id is not None and existing[0] != str(uuid.UUID(dataset_id)):
        raise DatasetSchemaError(
            f"Database dataset_id is {existing[0]!r}, not requested {dataset_id!r}"
        )
    elif revision_id is not None and existing[2] != str(uuid.UUID(revision_id)):
        raise DatasetSchemaError(
            f"Database revision_id is {existing[2]!r}, not requested {revision_id!r}"
        )
    connection.executescript(
        CLASSIFICATION_SCHEMA_SQL
        if resolved_type == "classification"
        else MASK_REFINEMENT_SCHEMA_SQL
    )
    _install_freeze_triggers(connection)
    _install_touch_triggers(connection)
    return read_dataset_info(connection)


def read_dataset_info(connection: sqlite3.Connection) -> dict[str, Any]:
    _migrate_schema_if_needed(connection)
    previous_factory = connection.row_factory
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT d.*, s.schema_name, s.schema_version
            FROM dataset d CROSS JOIN ob_schema s
            WHERE d.singleton = 1 AND s.singleton = 1
            """
        ).fetchone()
    finally:
        connection.row_factory = previous_factory
    if row is None:
        raise DatasetSchemaError("Not an Oracle Builder V1 dataset database")
    result = dict(row)
    result.pop("singleton", None)
    result["metadata"] = json.loads(result.pop("metadata_json") or "{}")
    return result


def set_dataset_lifecycle(
    connection: sqlite3.Connection,
    lifecycle: str,
    *,
    actor: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = str(lifecycle).strip().lower()
    if state not in LIFECYCLE_STATES:
        raise DatasetSchemaError(
            f"lifecycle must be one of: {', '.join(sorted(LIFECYCLE_STATES))}"
        )
    current = read_dataset_info(connection)
    now = utc_now()
    connection.execute(
        """
        UPDATE dataset
        SET lifecycle = ?, updated_at = ?,
            frozen_at = CASE WHEN ? = 'frozen' THEN ? ELSE NULL END
        WHERE singleton = 1
        """,
        (state, now, state, now),
    )
    connection.execute(
        """
        INSERT INTO dataset_events
            (event_id, dataset_id, revision_id, event_type, created_at, actor,
             details_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            current["dataset_id"],
            current["revision_id"],
            f"lifecycle.{state}",
            now,
            actor,
            json.dumps(details or {}, sort_keys=True, default=str),
        ),
    )
    return read_dataset_info(connection)


def _canonical_rows(
    connection: sqlite3.Connection, query: str, parameters: tuple[Any, ...] = ()
) -> list[list[Any]]:
    values = []
    for row in connection.execute(query, parameters):
        values.append(
            [
                value.hex() if isinstance(value, bytes) else value
                for value in row
            ]
        )
    return values


def dataset_fingerprint(connection: sqlite3.Connection) -> str:
    """Hash the semantic dataset state independently of SQLite page layout."""
    info = read_dataset_info(connection)
    payload: dict[str, Any] = {
        "schema_name": info["schema_name"],
        "schema_version": info["schema_version"],
        "dataset_id": info["dataset_id"],
        "dataset_type": info["dataset_type"],
        "name": info["name"],
        "title": info["title"],
        "description": info["description"],
        "version": info["version"],
        "metadata": info["metadata"],
        "items": _canonical_rows(
            connection,
            """
            SELECT item_id, sample_weight, source_key, metadata_json
            FROM dataset_items ORDER BY item_id
            """,
        ),
        "metadata_documents": _canonical_rows(
            connection,
            """
            SELECT document_id, name, source_filename, source_format,
                   parsed_json, sha256
            FROM metadata_documents ORDER BY document_id
            """,
        ),
    }
    if info["dataset_type"] == "classification":
        payload["assets"] = _canonical_rows(
            connection,
            """
            SELECT DISTINCT a.asset_id, a.content_sha256, a.encoding, a.media_type,
                   a.shape_json, a.dtype, a.original_filename, a.external_uri,
                   a.metadata_json
            FROM assets a
            JOIN classification_items ci ON ci.image_asset_id = a.asset_id
            ORDER BY a.asset_id
            """,
        )
        payload["labels"] = _canonical_rows(
            connection,
            """
            SELECT label_id, class_index, name, parent_label_id, metadata_json
            FROM classification_labels ORDER BY class_index
            """,
        )
        payload["classification_items"] = _canonical_rows(
            connection,
            "SELECT item_id, image_asset_id FROM classification_items ORDER BY item_id",
        )
        payload["annotations"] = _canonical_rows(
            connection,
            """
            SELECT annotation_id, item_id, label_id, annotator, source,
                   confidence, status, is_current, parent_annotation_id, notes,
                   metadata_json
            FROM classification_annotations ORDER BY annotation_id
            """,
        )
    else:
        payload["assets"] = _canonical_rows(
            connection,
            """
            SELECT DISTINCT a.asset_id, a.content_sha256, a.encoding, a.media_type,
                   a.shape_json, a.dtype, a.original_filename, a.external_uri,
                   a.metadata_json
            FROM assets a
            WHERE a.asset_id IN (
                SELECT image_asset_id FROM mask_refinement_items
                UNION
                SELECT candidate_mask_asset_id FROM mask_refinement_items
                    WHERE candidate_mask_asset_id IS NOT NULL
                UNION
                SELECT mask_asset_id FROM mask_annotations
            )
            ORDER BY a.asset_id
            """,
        )
        payload["mask_items"] = _canonical_rows(
            connection,
            """
            SELECT item_id, image_asset_id, candidate_mask_asset_id
            FROM mask_refinement_items ORDER BY item_id
            """,
        )
        payload["annotations"] = _canonical_rows(
            connection,
            """
            SELECT annotation_id, item_id, mask_asset_id, annotator, method,
                   parameters_json, validation_json, status, is_current,
                   parent_annotation_id, notes
            FROM mask_annotations ORDER BY annotation_id
            """,
        )
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_database(connection: sqlite3.Connection) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        info = read_dataset_info(connection)
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": []}
    if info["schema_name"] != SCHEMA_NAME or info["schema_version"] != SCHEMA_VERSION:
        errors.append(
            f"Unsupported schema {info['schema_name']!r} {info['schema_version']!r}"
        )
    try:
        uuid.UUID(info["dataset_id"])
    except (TypeError, ValueError):
        errors.append("dataset.dataset_id must be a canonical UUID")
    try:
        uuid.UUID(info["revision_id"])
    except (TypeError, ValueError):
        errors.append("dataset.revision_id must be a canonical UUID")
    if info.get("parent_revision_id") is not None:
        try:
            uuid.UUID(info["parent_revision_id"])
        except (TypeError, ValueError):
            errors.append("dataset.parent_revision_id must be a canonical UUID or NULL")
    required_tables = {
        "ob_schema",
        "dataset",
        "assets",
        "dataset_items",
        "metadata_documents",
        "import_events",
        "dataset_events",
    }
    if info["dataset_type"] == "classification":
        required_tables.update(
            {
                "classification_labels",
                "classification_items",
                "classification_annotations",
            }
        )
    else:
        required_tables.update({"mask_refinement_items", "mask_annotations"})
    missing_tables = sorted(
        table for table in required_tables if not _table_exists(connection, table)
    )
    if missing_tables:
        errors.append("Missing required table(s): " + ", ".join(missing_tables))
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "dataset": info,
            "item_count": 0,
            "fingerprint": None,
        }
    foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_keys:
        errors.append(f"{len(foreign_keys)} foreign-key violation(s)")
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        errors.append(f"SQLite integrity check failed: {integrity}")
    for asset_id, expected, payload in connection.execute(
        "SELECT asset_id, content_sha256, payload FROM assets WHERE payload IS NOT NULL"
    ):
        observed = hashlib.sha256(payload).hexdigest()
        if observed != expected:
            errors.append(
                f"Asset {asset_id} checksum mismatch: expected {expected}, got {observed}"
            )
    for name, expected, raw_text in connection.execute(
        "SELECT name, sha256, raw_text FROM metadata_documents WHERE raw_text IS NOT NULL"
    ):
        observed = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        if observed != expected:
            errors.append(
                f"Metadata document {name!r} checksum mismatch: "
                f"expected {expected}, got {observed}"
            )
    item_count = connection.execute("SELECT count(*) FROM dataset_items").fetchone()[0]
    if item_count == 0:
        warnings.append("Dataset contains no items")
    if info["dataset_type"] == "classification":
        if _table_exists(connection, "mask_refinement_items"):
            errors.append("Classification database contains mask-refinement schema tables")
        missing_items = connection.execute(
            """
            SELECT count(*) FROM dataset_items di
            LEFT JOIN classification_items ci ON ci.item_id = di.item_id
            WHERE ci.item_id IS NULL
            """
        ).fetchone()[0]
        if missing_items:
            errors.append(
                f"{missing_items} dataset item(s) lack a classification item record"
            )
        indices = [
            row[0]
            for row in connection.execute(
                "SELECT class_index FROM classification_labels ORDER BY class_index"
            )
        ]
        if indices != list(range(len(indices))):
            errors.append("Classification class indices must be contiguous from zero")
        missing = connection.execute(
            """
            SELECT count(*) FROM classification_items ci
            LEFT JOIN classification_annotations ca
              ON ca.item_id = ci.item_id AND ca.is_current = 1
            WHERE ca.annotation_id IS NULL
            """
        ).fetchone()[0]
        if missing:
            warnings.append(f"{missing} classification item(s) lack a current annotation")
    else:
        if _table_exists(connection, "classification_items"):
            errors.append("Mask-refinement database contains classification schema tables")
        missing_items = connection.execute(
            """
            SELECT count(*) FROM dataset_items di
            LEFT JOIN mask_refinement_items mi ON mi.item_id = di.item_id
            WHERE mi.item_id IS NULL
            """
        ).fetchone()[0]
        if missing_items:
            errors.append(
                f"{missing_items} dataset item(s) lack a mask-refinement item record"
            )
        missing = connection.execute(
            """
            SELECT count(*) FROM mask_refinement_items mi
            LEFT JOIN mask_annotations ma
              ON ma.item_id = mi.item_id AND ma.is_current = 1
            WHERE ma.annotation_id IS NULL
            """
        ).fetchone()[0]
        if missing:
            warnings.append(f"{missing} mask-refinement item(s) lack a current validated mask")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "dataset": info,
        "item_count": int(item_count),
        "fingerprint": dataset_fingerprint(connection),
    }
