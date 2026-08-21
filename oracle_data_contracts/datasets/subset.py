"""Deterministic, non-destructive derivation of classification datasets."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import Any

from oracle_data_contracts.datasets.schema import (
    dataset_fingerprint,
    initialize_database,
    read_dataset_info,
    utc_now,
    validate_database,
)


def _ranked_item_ids(item_ids: list[str], *, seed: int) -> list[str]:
    return sorted(
        item_ids,
        key=lambda item_id: (
            hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).digest(),
            item_id,
        ),
    )


def _classification_selection(
    connection: sqlite3.Connection,
    *,
    minimum_class_count: int,
    max_items: int | None,
    seed: int,
    include_unlabeled: bool,
) -> tuple[list[sqlite3.Row], list[sqlite3.Row], dict[str, Any]]:
    labels = list(
        connection.execute(
            """
            SELECT l.label_id, l.class_index, l.name, l.parent_label_id,
                   l.metadata_json, COUNT(ca.annotation_id) AS current_count
            FROM classification_labels l
            LEFT JOIN classification_annotations ca
              ON ca.label_id = l.label_id AND ca.is_current = 1
            GROUP BY l.label_id
            ORDER BY l.class_index
            """
        )
    )
    retained_label_ids = {
        str(row["label_id"])
        for row in labels
        if int(row["current_count"]) >= minimum_class_count
        and int(row["current_count"]) > 0
    }
    excluded_labels = [
        {
            "class_index": int(row["class_index"]),
            "name": str(row["name"]),
            "current_count": int(row["current_count"]),
            "reason": f"fewer than minimum_class_count ({minimum_class_count})",
        }
        for row in labels
        if str(row["label_id"]) not in retained_label_ids
    ]
    if not retained_label_ids and not include_unlabeled:
        raise ValueError("Subset selection retained no labeled classes")

    placeholders = ", ".join("?" for _ in retained_label_ids)
    rows: list[sqlite3.Row] = []
    if retained_label_ids:
        rows.extend(
            connection.execute(
                f"""
                SELECT di.item_id, di.sample_weight, di.source_key,
                       di.metadata_json AS item_metadata_json, di.created_at,
                       di.updated_at, a.asset_id, a.content_sha256, a.payload,
                       a.external_uri, a.encoding, a.media_type, a.shape_json,
                       a.dtype, a.original_filename,
                       a.metadata_json AS asset_metadata_json, a.created_at AS asset_created_at,
                       g.coordinate_space, g.bbox_x, g.bbox_y, g.bbox_w, g.bbox_h,
                       g.crop_bbox_x, g.crop_bbox_y, g.crop_bbox_w, g.crop_bbox_h,
                       g.metadata_json AS geometry_metadata_json,
                       ca.annotation_id, ca.created_at AS annotation_created_at,
                       ca.annotator, ca.source, ca.confidence, ca.status,
                       ca.notes, ca.metadata_json AS annotation_metadata_json,
                       ca.label_id
                FROM dataset_items di
                JOIN classification_items ci ON ci.item_id = di.item_id
                JOIN assets a ON a.asset_id = ci.image_asset_id
                JOIN item_geometry g ON g.item_id = di.item_id
                JOIN classification_annotations ca
                  ON ca.item_id = di.item_id AND ca.is_current = 1
                WHERE ca.label_id IN ({placeholders})
                """,
                tuple(sorted(retained_label_ids)),
            )
        )
    if include_unlabeled:
        rows.extend(
            connection.execute(
                """
                SELECT di.item_id, di.sample_weight, di.source_key,
                       di.metadata_json AS item_metadata_json, di.created_at,
                       di.updated_at, a.asset_id, a.content_sha256, a.payload,
                       a.external_uri, a.encoding, a.media_type, a.shape_json,
                       a.dtype, a.original_filename,
                       a.metadata_json AS asset_metadata_json, a.created_at AS asset_created_at,
                       g.coordinate_space, g.bbox_x, g.bbox_y, g.bbox_w, g.bbox_h,
                       g.crop_bbox_x, g.crop_bbox_y, g.crop_bbox_w, g.crop_bbox_h,
                       g.metadata_json AS geometry_metadata_json,
                       NULL AS annotation_id, NULL AS annotation_created_at,
                       NULL AS annotator, NULL AS source, NULL AS confidence,
                       NULL AS status, NULL AS notes,
                       NULL AS annotation_metadata_json, NULL AS label_id
                FROM dataset_items di
                JOIN classification_items ci ON ci.item_id = di.item_id
                JOIN assets a ON a.asset_id = ci.image_asset_id
                JOIN item_geometry g ON g.item_id = di.item_id
                LEFT JOIN classification_annotations ca
                  ON ca.item_id = di.item_id AND ca.is_current = 1
                WHERE ca.annotation_id IS NULL
                """
            )
        )
    rows_by_id = {str(row["item_id"]): row for row in rows}
    selected_ids = _ranked_item_ids(list(rows_by_id), seed=seed)
    if max_items is not None:
        selected_ids = selected_ids[:max_items]
    selected_rows = [rows_by_id[item_id] for item_id in selected_ids]
    selected_label_ids = {
        str(row["label_id"]) for row in selected_rows if row["label_id"] is not None
    }
    selected_labels = [
        row for row in labels if str(row["label_id"]) in selected_label_ids
    ]
    counts_after = {
        str(label["name"]): sum(
            row["label_id"] == label["label_id"] for row in selected_rows
        )
        for label in selected_labels
    }
    report = {
        "minimum_class_count": minimum_class_count,
        "max_items": max_items,
        "seed": seed,
        "include_unlabeled": include_unlabeled,
        "source_class_counts": {
            str(row["name"]): int(row["current_count"]) for row in labels
        },
        "excluded_labels": excluded_labels,
        "selected_class_counts": counts_after,
        "selected_item_count": len(selected_rows),
        "selection_order": "sha256(seed:item_id)",
        "annotation_history": "current_annotations_only",
    }
    return selected_rows, selected_labels, report


def _copy_metadata_documents(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    dataset_id: str,
) -> None:
    for row in source.execute(
        """
        SELECT name, source_filename, source_format, parsed_json, raw_text, sha256, created_at
        FROM metadata_documents ORDER BY name
        """
    ):
        destination.execute(
            """
            INSERT INTO metadata_documents (
                document_id, dataset_id, name, source_filename, source_format,
                parsed_json, raw_text, sha256, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), dataset_id, *row),
        )


def _copy_selected_classification(
    destination: sqlite3.Connection,
    dataset_id: str,
    rows: list[sqlite3.Row],
    labels: list[sqlite3.Row],
) -> None:
    label_ids = {str(row["label_id"]): str(uuid.uuid4()) for row in labels}
    for index, label in enumerate(labels):
        parent_id = label_ids.get(str(label["parent_label_id"]))
        destination.execute(
            """
            INSERT INTO classification_labels
                (label_id, dataset_id, class_index, name, parent_label_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                label_ids[str(label["label_id"])], dataset_id, index,
                label["name"], parent_id, label["metadata_json"],
            ),
        )
    for row in rows:
        destination.execute(
            """
            INSERT INTO assets (
                asset_id, dataset_id, content_sha256, payload, external_uri, encoding,
                media_type, shape_json, dtype, original_filename, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["asset_id"], dataset_id, row["content_sha256"], row["payload"],
                row["external_uri"], row["encoding"], row["media_type"],
                row["shape_json"], row["dtype"], row["original_filename"],
                row["asset_metadata_json"], row["asset_created_at"],
            ),
        )
        destination.execute(
            """
            INSERT INTO dataset_items (
                item_id, dataset_id, sample_weight, source_key, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["item_id"], dataset_id, row["sample_weight"], row["source_key"],
                row["item_metadata_json"], row["created_at"], row["updated_at"],
            ),
        )
        destination.execute(
            "INSERT INTO classification_items (item_id, image_asset_id) VALUES (?, ?)",
            (row["item_id"], row["asset_id"]),
        )
        destination.execute(
            """INSERT INTO item_geometry (
            item_id,coordinate_space,bbox_x,bbox_y,bbox_w,bbox_h,
            crop_bbox_x,crop_bbox_y,crop_bbox_w,crop_bbox_h,metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row["item_id"], row["coordinate_space"], row["bbox_x"], row["bbox_y"],
                row["bbox_w"], row["bbox_h"], row["crop_bbox_x"], row["crop_bbox_y"],
                row["crop_bbox_w"], row["crop_bbox_h"], row["geometry_metadata_json"],
            ),
        )
        if row["annotation_id"] is not None:
            destination.execute(
                """
                INSERT INTO classification_annotations (
                    annotation_id, item_id, label_id, created_at, annotator, source,
                    confidence, status, is_current, parent_annotation_id, notes, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, NULL, ?, ?)
                """,
                (
                    str(uuid.uuid4()), row["item_id"], label_ids[str(row["label_id"])],
                    row["annotation_created_at"], row["annotator"], row["source"],
                    row["confidence"], row["status"], row["notes"],
                    row["annotation_metadata_json"],
                ),
            )


def subset_classification_dataset(
    source_path: str | Path,
    output_path: str | Path,
    *,
    minimum_class_count: int = 1,
    max_items: int | None = None,
    seed: int = 123,
    include_unlabeled: bool = False,
    name: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create an editable classification dataset containing a deterministic subset.

    The derived dataset preserves encoded assets and current labels, but starts a
    fresh annotation history.  It is therefore suitable for fast experiments or
    for a curated training release, while its metadata preserves full source
    identity and the exact selection policy.
    """
    if minimum_class_count < 0:
        raise ValueError("minimum_class_count must be non-negative")
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be at least one")
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if source == output:
        raise ValueError("Subset output must differ from source")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not dry_run:
        raise FileExistsError(output)
    with sqlite3.connect(source) as source_connection:
        source_connection.row_factory = sqlite3.Row
        validation = validate_database(source_connection)
        if not validation["valid"]:
            raise ValueError("Cannot subset an invalid dataset: " + "; ".join(validation["errors"]))
        source_info = read_dataset_info(source_connection)
        if source_info["dataset_type"] != "classification":
            raise ValueError("oracle-dataset subset currently supports classification datasets only")
        source_fingerprint = dataset_fingerprint(source_connection)
        rows, labels, selection = _classification_selection(
            source_connection,
            minimum_class_count=minimum_class_count,
            max_items=max_items,
            seed=seed,
            include_unlabeled=include_unlabeled,
        )
        if not rows:
            raise ValueError("Subset selection retained no dataset items")
        result = {
            "source": str(source),
            "output": str(output),
            "source_dataset_id": source_info["dataset_id"],
            "source_fingerprint": source_fingerprint,
            "dataset_type": "classification",
            "lifecycle": "working",
            "selection": selection,
        }
        if dry_run:
            result["dry_run"] = True
            return result
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        )
        temporary_path = Path(temporary.name)
        temporary.close()
        try:
            metadata = dict(source_info.get("metadata") or {})
            metadata["derived_from"] = {
                "operation": "subset",
                "source_dataset_id": source_info["dataset_id"],
                "source_revision_id": source_info["revision_id"],
                "source_fingerprint_sha256": source_fingerprint,
                "created_at": utc_now(),
                "selection": selection,
            }
            with sqlite3.connect(temporary_path) as destination:
                destination.execute("PRAGMA foreign_keys = ON")
                destination_info = initialize_database(
                    destination,
                    "classification",
                    name=name or f"{source_info['name']}-subset",
                    title=source_info.get("title"),
                    description=source_info.get("description"),
                    version=source_info.get("version"),
                    metadata=metadata,
                )
                _copy_selected_classification(
                    destination, destination_info["dataset_id"], rows, labels
                )
                _copy_metadata_documents(
                    source_connection, destination, destination_info["dataset_id"]
                )
                destination.execute(
                    """
                    INSERT INTO dataset_events
                        (event_id, dataset_id, revision_id, event_type, created_at, actor, details_json)
                    VALUES (?, ?, ?, 'subset_created', ?, NULL, ?)
                    """,
                    (
                        str(uuid.uuid4()), destination_info["dataset_id"],
                        destination_info["revision_id"], utc_now(),
                        json.dumps(metadata["derived_from"], sort_keys=True),
                    ),
                )
                destination.commit()
                report = validate_database(destination)
                if not report["valid"]:
                    raise ValueError("Subset database validation failed: " + "; ".join(report["errors"]))
                result["dataset_id"] = destination_info["dataset_id"]
                result["revision_id"] = destination_info["revision_id"]
                result["fingerprint"] = dataset_fingerprint(destination)
            os.replace(temporary_path, output)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    return result
