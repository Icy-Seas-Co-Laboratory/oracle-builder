from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import tomli_w

from oracle_data_contracts.datasets.schema import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    dataset_fingerprint,
    initialize_database,
    read_dataset_info,
    utc_now,
    validate_database,
)


EXPORT_FORMAT_VERSION = "1.0.0"


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return safe or "unnamed"


def _path_token(value: str) -> str:
    """Produce a readable filename token without trusting an opaque identifier."""
    return f"{_safe_name(value)}-{hashlib.sha256(str(value).encode()).hexdigest()[:12]}"


def _extension(encoding: str) -> str:
    return {
        "jpeg": "jpg",
        "jpg": "jpg",
        "tif": "tif",
        "tiff": "tif",
        "png": "png",
        "npy": "npy",
    }.get(str(encoding).lower(), _safe_name(encoding).lower())


def _write_payload(root: Path, relative_path: str, payload: bytes) -> dict[str, Any]:
    path = _contained_path(root, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _contained_path(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"Export manifest path must be relative: {relative_path!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Export manifest path escapes its root: {relative_path!r}")
    return resolved


def _metadata_toml(info: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    source_metadata = _toml_safe(info["metadata"].get("source_metadata", {}))
    return {
        "oracle_builder": {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "export_format_version": EXPORT_FORMAT_VERSION,
            "dataset_id": info["dataset_id"],
            "revision_id": info["revision_id"],
            "parent_revision_id": info["parent_revision_id"] or "",
            "dataset_type": info["dataset_type"],
            "lifecycle": info["lifecycle"],
            "fingerprint_sha256": fingerprint,
        },
        "dataset": {
            "name": info["name"],
            "title": info["title"] or "",
            "description": info["description"] or "",
            "version": info["version"] or "",
            "created_at": info["created_at"],
            "updated_at": info["updated_at"],
        },
        "source_metadata": source_metadata,
    }


def _toml_safe(value: Any) -> Any:
    """Convert JSON-compatible metadata into values accepted by TOML writers."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return {str(key): _toml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_toml_safe(item) for item in value]
        # TOML arrays must be homogeneous. Preserve mixed arrays losslessly as JSON.
        kinds = {type(item) for item in converted}
        return converted if len(kinds) <= 1 else json.dumps(value, sort_keys=True)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def export_dataset(
    sqlite_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    database = Path(sqlite_path).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Export destination already exists; choose a new directory: {output}"
        )
    output.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        report = validate_database(connection)
        if not report["valid"]:
            raise ValueError("Invalid dataset: " + "; ".join(report["errors"]))
        info = read_dataset_info(connection)
        fingerprint = dataset_fingerprint(connection)
        manifest: dict[str, Any] = {
            "export_format_version": EXPORT_FORMAT_VERSION,
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "dataset": {
                key: info[key]
                for key in (
                    "dataset_id",
                    "revision_id",
                    "parent_revision_id",
                    "dataset_type",
                    "name",
                    "title",
                    "description",
                    "version",
                    "lifecycle",
                    "created_at",
                    "updated_at",
                    "frozen_at",
                    "metadata",
                )
            },
            "fingerprint_sha256": fingerprint,
            "items": [],
            "metadata_documents": [],
            "import_events": [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT import_id, dataset_id, revision_id, created_at,
                           importer, source_uri, options_json, summary_json
                    FROM import_events ORDER BY created_at, import_id
                    """
                )
            ],
            "dataset_events": [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT event_id, dataset_id, revision_id, event_type,
                           created_at, actor, details_json
                    FROM dataset_events ORDER BY created_at, event_id
                    """
                )
            ],
            "annotation_labels": [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT label_id, dataset_id, name, display_name, parent_label_id,
                           rank, description, metadata_json, created_at, deprecated_at
                    FROM annotation_labels ORDER BY label_id
                    """
                )
            ],
            "item_label_annotations": [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT annotation_id, item_id, label_id, created_at, annotator,
                           method, confidence, status, is_current, parent_annotation_id,
                           parameters_json, notes, metadata_json
                    FROM item_label_annotations ORDER BY annotation_id
                    """
                )
            ],
            "annotation_reviews": [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT review_id, annotation_id, reviewer, decision, created_at,
                           notes, metadata_json
                    FROM annotation_reviews ORDER BY review_id
                    """
                )
            ],
            "taxonomy_concepts": [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT concept_id, vocabulary_id, vocabulary_version, vocabulary_node_id,
                           name, display_name, scientific_name, concept_type, rank,
                           parent_concept_id, selectable, metadata_json, created_at
                    FROM taxonomy_concepts ORDER BY concept_id
                    """
                )
            ],
            "taxonomy_concept_mappings": [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT mapping_id, concept_id, authority, scheme, identifier, uri,
                           relationship, metadata_json, created_at
                    FROM taxonomy_concept_mappings ORDER BY mapping_id
                    """
                )
            ],
            "classification_label_concepts": [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT label_id, concept_id, relationship, mapped_by, metadata_json, created_at
                    FROM classification_label_concepts ORDER BY label_id
                    """
                )
            ],
        }
        if info["dataset_type"] == "classification":
            manifest["labels"] = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT label_id, class_index, name, parent_label_id, metadata_json
                    FROM classification_labels ORDER BY class_index
                    """
                )
            ]
            rows = connection.execute(
                """
                SELECT di.item_id, di.sample_weight, di.source_key,
                       di.metadata_json, di.created_at, di.updated_at,
                       a.asset_id, a.content_sha256, a.payload, a.external_uri,
                       a.encoding, a.media_type, a.shape_json, a.dtype,
                       a.original_filename, a.metadata_json AS asset_metadata_json,
                       ca.annotation_id, ca.created_at AS annotation_created_at,
                       ca.annotator, ca.source, ca.confidence, ca.status, ca.is_current,
                       ca.parent_annotation_id, ca.notes,
                       ca.metadata_json AS annotation_metadata_json,
                       l.label_id, l.name AS class_name
                FROM dataset_items di
                JOIN classification_items ci ON ci.item_id = di.item_id
                JOIN assets a ON a.asset_id = ci.image_asset_id
                LEFT JOIN classification_annotations ca
                  ON ca.item_id = di.item_id AND ca.is_current = 1
                LEFT JOIN classification_labels l ON l.label_id = ca.label_id
                ORDER BY di.item_id
                """
            )
            for row in rows:
                if row["payload"] is None:
                    raise ValueError(
                        f"Folder export requires embedded payload for item {row['item_id']}"
                    )
                relative = (
                    f"images/{_safe_name(row['class_name'] or 'unlabeled')}/"
                    f"{_path_token(row['item_id'])}.{_extension(row['encoding'])}"
                )
                file_record = _write_payload(output, relative, row["payload"])
                manifest["items"].append(
                    {
                        "item_id": row["item_id"],
                        "sample_weight": row["sample_weight"],
                        "source_key": row["source_key"],
                        "metadata_json": row["metadata_json"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "geometry": _item_geometry(connection, row["item_id"]),
                        "asset": {
                            "asset_id": row["asset_id"],
                            "content_sha256": row["content_sha256"],
                            "external_uri": row["external_uri"],
                            "encoding": row["encoding"],
                            "media_type": row["media_type"],
                            "shape_json": row["shape_json"],
                            "dtype": row["dtype"],
                            "original_filename": row["original_filename"],
                            "metadata_json": row["asset_metadata_json"],
                            **file_record,
                        },
                        "annotations": [
                            dict(annotation)
                            for annotation in connection.execute(
                                """
                                SELECT annotation_id, label_id, created_at, annotator,
                                       source, confidence, status, is_current,
                                       parent_annotation_id, notes, metadata_json
                                FROM classification_annotations
                                WHERE item_id = ?
                                ORDER BY created_at, annotation_id
                                """,
                                (row["item_id"],),
                            )
                        ],
                    }
                )
        else:
            _export_mask_items(connection, output, manifest)

        for row in connection.execute(
            """
            SELECT document_id, name, source_filename, source_format, parsed_json,
                   raw_text, sha256, created_at
            FROM metadata_documents ORDER BY name
            """
        ):
            relative = (
                f"metadata/source/{_safe_name(row['document_id'])}-"
                f"{_safe_name(row['source_filename'])}"
            )
            raw = (row["raw_text"] or "").encode("utf-8")
            file_record = _write_payload(output, relative, raw)
            manifest["metadata_documents"].append(
                {
                    **dict(row),
                    "path": relative,
                    "export_sha256": file_record["sha256"],
                }
            )

    metadata_path = output / "metadata.toml"
    metadata_path.write_text(
        tomli_w.dumps(_metadata_toml(info, fingerprint)), encoding="utf-8"
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        relative = path.relative_to(output).as_posix()
        checksum_lines.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}"
        )
    (output / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    return {
        "database": str(database),
        "output": str(output),
        "dataset_id": info["dataset_id"],
        "dataset_type": info["dataset_type"],
        "fingerprint": fingerprint,
        "item_count": len(manifest["items"]),
    }


def _export_mask_items(
    connection: sqlite3.Connection,
    output: Path,
    manifest: dict[str, Any],
) -> None:
    rows = connection.execute(
        """
        SELECT di.item_id, di.sample_weight, di.source_key,
               di.metadata_json, di.created_at, di.updated_at,
               image.asset_id AS image_asset_id, image.content_sha256 AS image_sha256,
               image.payload AS image_payload, image.encoding AS image_encoding,
               image.media_type AS image_media_type, image.shape_json AS image_shape_json,
               image.dtype AS image_dtype, image.original_filename AS image_original_filename,
               image.external_uri AS image_external_uri,
               image.metadata_json AS image_metadata_json,
               candidate.asset_id AS candidate_asset_id,
               candidate.content_sha256 AS candidate_sha256,
               candidate.payload AS candidate_payload,
               candidate.encoding AS candidate_encoding,
               candidate.media_type AS candidate_media_type,
               candidate.shape_json AS candidate_shape_json,
               candidate.dtype AS candidate_dtype,
               candidate.original_filename AS candidate_original_filename,
               candidate.external_uri AS candidate_external_uri,
               candidate.metadata_json AS candidate_metadata_json
        FROM dataset_items di
        JOIN mask_refinement_items mi ON mi.item_id = di.item_id
        JOIN assets image ON image.asset_id = mi.image_asset_id
        LEFT JOIN assets candidate ON candidate.asset_id = mi.candidate_mask_asset_id
        ORDER BY di.item_id
        """
    )
    for row in rows:
        if row["image_payload"] is None:
            raise ValueError(
                f"Folder export requires embedded image payload for item {row['item_id']}"
            )
        image_relative = (
            f"images/{_path_token(row['item_id'])}."
            f"{_extension(row['image_encoding'])}"
        )
        image_file = _write_payload(output, image_relative, row["image_payload"])
        item = {
            "item_id": row["item_id"],
            "sample_weight": row["sample_weight"],
            "source_key": row["source_key"],
            "metadata_json": row["metadata_json"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "geometry": _item_geometry(connection, row["item_id"]),
            "image_asset": {
                "asset_id": row["image_asset_id"],
                "content_sha256": row["image_sha256"],
                "encoding": row["image_encoding"],
                "media_type": row["image_media_type"],
                "shape_json": row["image_shape_json"],
                "dtype": row["image_dtype"],
                "original_filename": row["image_original_filename"],
                "external_uri": row["image_external_uri"],
                "metadata_json": row["image_metadata_json"],
                **image_file,
            },
            "candidate_mask_asset": None,
            "annotations": [],
        }
        if row["candidate_asset_id"]:
            if row["candidate_payload"] is None:
                raise ValueError(
                    "Folder export requires embedded candidate-mask payload for "
                    f"item {row['item_id']}"
                )
            candidate_relative = (
                f"candidate_masks/{_path_token(row['item_id'])}."
                f"{_extension(row['candidate_encoding'])}"
            )
            candidate_file = _write_payload(
                output, candidate_relative, row["candidate_payload"]
            )
            item["candidate_mask_asset"] = {
                "asset_id": row["candidate_asset_id"],
                "content_sha256": row["candidate_sha256"],
                "encoding": row["candidate_encoding"],
                "media_type": row["candidate_media_type"],
                "shape_json": row["candidate_shape_json"],
                "dtype": row["candidate_dtype"],
                "original_filename": row["candidate_original_filename"],
                "external_uri": row["candidate_external_uri"],
                "metadata_json": row["candidate_metadata_json"],
                **candidate_file,
            }
        annotations = connection.execute(
            """
            SELECT ma.*, a.content_sha256, a.payload, a.external_uri,
                   a.encoding, a.media_type, a.shape_json, a.dtype,
                   a.original_filename, a.metadata_json
            FROM mask_annotations ma
            JOIN assets a ON a.asset_id = ma.mask_asset_id
            WHERE ma.item_id = ? ORDER BY ma.created_at, ma.annotation_id
            """,
            (row["item_id"],),
        )
        for annotation in annotations:
            if annotation["payload"] is None:
                raise ValueError(
                    "Folder export requires embedded annotation payload for "
                    f"annotation {annotation['annotation_id']}"
                )
            relative = (
                f"annotations/{_path_token(row['item_id'])}/"
                f"{_path_token(annotation['annotation_id'])}."
                f"{_extension(annotation['encoding'])}"
            )
            file_record = _write_payload(output, relative, annotation["payload"])
            item["annotations"].append(
                {
                    key: annotation[key]
                    for key in (
                        "annotation_id",
                        "mask_asset_id",
                        "created_at",
                        "annotator",
                        "method",
                        "parameters_json",
                        "validation_json",
                        "status",
                        "is_current",
                        "parent_annotation_id",
                        "notes",
                    )
                }
                | {
                    "asset": {
                        "asset_id": annotation["mask_asset_id"],
                        "content_sha256": annotation["content_sha256"],
                        "encoding": annotation["encoding"],
                        "media_type": annotation["media_type"],
                        "shape_json": annotation["shape_json"],
                        "dtype": annotation["dtype"],
                        "original_filename": annotation["original_filename"],
                        "external_uri": annotation["external_uri"],
                        "metadata_json": annotation["metadata_json"],
                        **file_record,
                    }
                }
            )
        manifest["items"].append(item)


def _verify_checksums(root: Path) -> None:
    checksum_path = root / "checksums.sha256"
    if not checksum_path.exists():
        raise ValueError("Export folder is missing checksums.sha256")
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = _contained_path(root, relative)
        if not path.is_file():
            raise ValueError(f"Export payload is missing: {relative}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"Checksum mismatch for {relative}")


def import_dataset_export(
    input_directory: str | Path,
    sqlite_path: str | Path,
) -> dict[str, Any]:
    root = Path(input_directory).expanduser().resolve()
    output = Path(sqlite_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    _verify_checksums(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("export_format_version") != EXPORT_FORMAT_VERSION:
        raise ValueError("Unsupported dataset folder export version")
    dataset = manifest["dataset"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output) as connection:
        initialize_database(
            connection,
            dataset["dataset_type"],
            dataset_id=dataset["dataset_id"],
            revision_id=dataset["revision_id"],
            parent_revision_id=dataset.get("parent_revision_id"),
            name=dataset["name"],
            title=dataset.get("title"),
            description=dataset.get("description"),
            version=dataset.get("version"),
            metadata=dataset.get("metadata") or {},
        )
        connection.execute("DELETE FROM dataset_events")
        for label in manifest.get("annotation_labels", []):
            connection.execute(
                """
                INSERT INTO annotation_labels (
                    label_id, dataset_id, name, display_name, parent_label_id, rank,
                    description, metadata_json, created_at, deprecated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    label["label_id"], dataset["dataset_id"], label["name"],
                    label.get("display_name"), label.get("parent_label_id"),
                    label.get("rank"), label.get("description"),
                    label.get("metadata_json") or "{}", label["created_at"],
                    label.get("deprecated_at"),
                ),
            )
        if dataset["dataset_type"] == "classification":
            for label in manifest.get("labels", []):
                connection.execute(
                    """
                    INSERT INTO classification_labels
                        (label_id, dataset_id, class_index, name, parent_label_id, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        label["label_id"],
                        dataset["dataset_id"],
                        label["class_index"],
                        label["name"],
                        label.get("parent_label_id"),
                        label.get("metadata_json") or "{}",
                    ),
                )
            for item in manifest["items"]:
                asset_id = _restore_asset(connection, dataset["dataset_id"], root, item["asset"])
                _restore_item(connection, dataset["dataset_id"], item)
                connection.execute(
                    "INSERT INTO classification_items VALUES (?, ?)",
                    (item["item_id"], asset_id),
                )
                for annotation in item.get("annotations", []):
                    connection.execute(
                        """
                        INSERT INTO classification_annotations (
                            annotation_id, item_id, label_id, created_at, annotator,
                            source, confidence, status, is_current, parent_annotation_id,
                            notes, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            annotation["annotation_id"],
                            item["item_id"],
                            annotation["label_id"],
                            annotation["created_at"],
                            annotation.get("annotator"),
                            annotation.get("source"),
                            annotation.get("confidence"),
                            annotation["status"],
                            annotation["is_current"],
                            annotation.get("parent_annotation_id"),
                            annotation.get("notes"),
                            annotation.get("metadata_json") or "{}",
                        ),
                    )
        else:
            for item in manifest["items"]:
                image_id = _restore_asset(
                    connection, dataset["dataset_id"], root, item["image_asset"]
                )
                candidate_id = (
                    _restore_asset(
                        connection,
                        dataset["dataset_id"],
                        root,
                        item["candidate_mask_asset"],
                    )
                    if item.get("candidate_mask_asset")
                    else None
                )
                _restore_item(connection, dataset["dataset_id"], item)
                connection.execute(
                    "INSERT INTO mask_refinement_items VALUES (?, ?, ?)",
                    (item["item_id"], image_id, candidate_id),
                )
                for annotation in item.get("annotations", []):
                    mask_id = _restore_asset(
                        connection, dataset["dataset_id"], root, annotation["asset"]
                    )
                    connection.execute(
                        """
                        INSERT INTO mask_annotations (
                            annotation_id, item_id, mask_asset_id, created_at, annotator,
                            method, parameters_json, validation_json, status, is_current,
                            parent_annotation_id, notes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            annotation["annotation_id"],
                            item["item_id"],
                            mask_id,
                            annotation["created_at"],
                            annotation.get("annotator"),
                            annotation.get("method"),
                            annotation.get("parameters_json") or "{}",
                            annotation.get("validation_json") or "{}",
                            annotation["status"],
                            annotation["is_current"],
                            annotation.get("parent_annotation_id"),
                            annotation.get("notes"),
                        ),
                    )
        for concept in manifest.get("taxonomy_concepts", []):
            connection.execute(
                """
                INSERT INTO taxonomy_concepts (
                    concept_id, vocabulary_id, vocabulary_version, vocabulary_node_id,
                    name, display_name, scientific_name, concept_type, rank,
                    parent_concept_id, selectable, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                """,
                (
                    concept["concept_id"], concept["vocabulary_id"], concept["vocabulary_version"],
                    concept["vocabulary_node_id"], concept["name"], concept.get("display_name"),
                    concept.get("scientific_name"), concept["concept_type"], concept.get("rank"),
                    concept.get("selectable", 1), concept.get("metadata_json") or "{}",
                    concept["created_at"],
                ),
            )
        for concept in manifest.get("taxonomy_concepts", []):
            if concept.get("parent_concept_id"):
                connection.execute(
                    "UPDATE taxonomy_concepts SET parent_concept_id = ? WHERE concept_id = ?",
                    (concept["parent_concept_id"], concept["concept_id"]),
                )
        for mapping in manifest.get("taxonomy_concept_mappings", []):
            connection.execute(
                """
                INSERT INTO taxonomy_concept_mappings (
                    mapping_id, concept_id, authority, scheme, identifier, uri,
                    relationship, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mapping["mapping_id"], mapping["concept_id"], mapping["authority"],
                    mapping["scheme"], mapping["identifier"], mapping.get("uri"),
                    mapping.get("relationship", "exact"), mapping.get("metadata_json") or "{}",
                    mapping["created_at"],
                ),
            )
        for mapping in manifest.get("classification_label_concepts", []):
            connection.execute(
                """
                INSERT INTO classification_label_concepts (
                    label_id, concept_id, relationship, mapped_by, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    mapping["label_id"], mapping["concept_id"],
                    mapping.get("relationship", "exact"), mapping.get("mapped_by"),
                    mapping.get("metadata_json") or "{}", mapping["created_at"],
                ),
            )
        for annotation in manifest.get("item_label_annotations", []):
            connection.execute(
                """
                INSERT INTO item_label_annotations (
                    annotation_id, item_id, label_id, created_at, annotator, method,
                    confidence, status, is_current, parent_annotation_id,
                    parameters_json, notes, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    annotation["annotation_id"], annotation["item_id"],
                    annotation["label_id"], annotation["created_at"],
                    annotation.get("annotator"), annotation.get("method", "manual"),
                    annotation.get("confidence"), annotation["status"],
                    annotation["is_current"], annotation.get("parent_annotation_id"),
                    annotation.get("parameters_json") or "{}", annotation.get("notes"),
                    annotation.get("metadata_json") or "{}",
                ),
            )
        for review in manifest.get("annotation_reviews", []):
            connection.execute(
                """
                INSERT INTO annotation_reviews (
                    review_id, annotation_id, reviewer, decision, created_at, notes, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review["review_id"], review["annotation_id"], review["reviewer"],
                    review["decision"], review["created_at"], review.get("notes"),
                    review.get("metadata_json") or "{}",
                ),
            )
        for document in manifest.get("metadata_documents", []):
            connection.execute(
                """
                INSERT INTO metadata_documents (
                    document_id, dataset_id, name, source_filename, source_format,
                    parsed_json, raw_text, sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document["document_id"],
                    dataset["dataset_id"],
                    document["name"],
                    document["source_filename"],
                    document["source_format"],
                    document["parsed_json"],
                    document.get("raw_text"),
                    document["sha256"],
                    document["created_at"],
                ),
            )
        for event in manifest.get("import_events", []):
            connection.execute(
                """
                INSERT INTO import_events (
                    import_id, dataset_id, revision_id, created_at, importer,
                    source_uri, options_json, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["import_id"],
                    event["dataset_id"],
                    event["revision_id"],
                    event["created_at"],
                    event["importer"],
                    event.get("source_uri"),
                    event["options_json"],
                    event["summary_json"],
                ),
            )
        for event in manifest.get("dataset_events", []):
            connection.execute(
                """
                INSERT INTO dataset_events (
                    event_id, dataset_id, revision_id, event_type, created_at,
                    actor, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["event_id"],
                    event["dataset_id"],
                    event["revision_id"],
                    event["event_type"],
                    event["created_at"],
                    event.get("actor"),
                    event["details_json"],
                ),
            )
        connection.execute(
            """
            UPDATE dataset SET lifecycle = ?, created_at = ?, updated_at = ?, frozen_at = ?
            WHERE singleton = 1
            """,
            (
                dataset["lifecycle"],
                dataset["created_at"],
                dataset["updated_at"],
                dataset.get("frozen_at"),
            ),
        )
        connection.commit()
        observed = dataset_fingerprint(connection)
        expected = manifest["fingerprint_sha256"]
        if observed != expected:
            raise ValueError(
                f"Round-trip fingerprint mismatch: expected {expected}, observed {observed}"
            )
    return {
        "input": str(root),
        "output": str(output),
        "dataset_id": dataset["dataset_id"],
        "dataset_type": dataset["dataset_type"],
        "fingerprint": expected,
        "item_count": len(manifest["items"]),
    }


def _restore_item(
    connection: sqlite3.Connection, dataset_id: str, item: dict[str, Any]
) -> None:
    connection.execute(
        """
        INSERT INTO dataset_items (
            item_id, dataset_id, sample_weight, source_key,
            metadata_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item["item_id"],
            dataset_id,
            item.get("sample_weight"),
            item.get("source_key"),
            item.get("metadata_json") or "{}",
            item["created_at"],
            item["updated_at"],
        ),
    )
    geometry = item.get("geometry")
    if geometry:
        connection.execute(
            """INSERT INTO item_geometry (
            item_id,coordinate_space,bbox_x,bbox_y,bbox_w,bbox_h,
            crop_bbox_x,crop_bbox_y,crop_bbox_w,crop_bbox_h,metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                item["item_id"], geometry["coordinate_space"],
                geometry["bbox_x"], geometry["bbox_y"], geometry["bbox_w"], geometry["bbox_h"],
                geometry["crop_bbox_x"], geometry["crop_bbox_y"],
                geometry["crop_bbox_w"], geometry["crop_bbox_h"],
                geometry.get("metadata_json") or "{}",
            ),
        )


def _item_geometry(connection: sqlite3.Connection, item_id: str) -> dict[str, Any] | None:
    row = connection.execute(
        """SELECT coordinate_space,bbox_x,bbox_y,bbox_w,bbox_h,
        crop_bbox_x,crop_bbox_y,crop_bbox_w,crop_bbox_h,metadata_json
        FROM item_geometry WHERE item_id=?""",
        (item_id,),
    ).fetchone()
    return dict(row) if row else None


def _restore_asset(
    connection: sqlite3.Connection,
    dataset_id: str,
    root: Path,
    asset: dict[str, Any],
) -> str:
    payload = _contained_path(root, asset["path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != asset["content_sha256"]:
        raise ValueError(f"Asset content checksum mismatch: {asset['path']}")
    connection.execute(
        """
        INSERT OR IGNORE INTO assets (
            asset_id, dataset_id, content_sha256, payload, external_uri, encoding,
            media_type, shape_json, dtype, original_filename, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            asset["asset_id"],
            dataset_id,
            asset["content_sha256"],
            payload,
            asset.get("external_uri"),
            asset["encoding"],
            asset.get("media_type"),
            asset.get("shape_json"),
            asset.get("dtype"),
            asset.get("original_filename"),
            asset.get("metadata_json") or "{}",
            utc_now(),
        ),
    )
    return asset["asset_id"]
