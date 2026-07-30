from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

from oracle_builder.datasets.schema import read_dataset_info, utc_now


@dataclass(frozen=True)
class StoredAsset:
    asset_id: str
    content_sha256: str
    encoding: str


def stable_uuid(namespace: str, value: str) -> str:
    return str(uuid.uuid5(uuid.UUID(namespace), value))


def content_uuid(dataset_id: str, content_sha256: str, encoding: str) -> str:
    """Return a stable, dataset-scoped identifier for an encoded asset."""
    return stable_uuid(
        dataset_id,
        f"asset:{content_sha256}:{str(encoding).strip().lower()}",
    )


class SQLiteDatasetRepository:
    """Storage adapter for the Oracle Builder V1 logical dataset contract."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.connection.row_factory = sqlite3.Row
        self.info = read_dataset_info(connection)
        self.dataset_id = self.info["dataset_id"]
        self.dataset_type = self.info["dataset_type"]

    def add_asset(
        self,
        payload: bytes | None,
        *,
        encoding: str,
        media_type: str | None = None,
        shape: Iterable[int] | None = None,
        dtype: str | None = None,
        original_filename: str | None = None,
        external_uri: str | None = None,
        metadata: dict[str, Any] | None = None,
        content_sha256: str | None = None,
    ) -> StoredAsset:
        if payload is None and external_uri is None:
            raise ValueError("An asset requires payload bytes or an external_uri")
        if payload is None and content_sha256 is None:
            raise ValueError(
                "An external-only asset requires content_sha256 so its identity "
                "does not depend on its storage location"
            )
        digest = content_sha256 or hashlib.sha256(payload or b"").hexdigest()
        normalized_encoding = encoding.lower()
        asset_id = content_uuid(self.dataset_id, digest, normalized_encoding)
        self.connection.execute(
            """
            INSERT INTO assets (
                asset_id, dataset_id, content_sha256, payload, external_uri, encoding,
                media_type, shape_json, dtype, original_filename, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id, content_sha256, encoding) DO UPDATE SET
                external_uri = COALESCE(assets.external_uri, excluded.external_uri),
                original_filename = COALESCE(assets.original_filename, excluded.original_filename)
            """,
            (
                asset_id,
                self.dataset_id,
                digest,
                payload,
                external_uri,
                normalized_encoding,
                media_type,
                json.dumps(list(shape)) if shape is not None else None,
                dtype,
                original_filename,
                json.dumps(metadata or {}, sort_keys=True, default=str),
                utc_now(),
            ),
        )
        row = self.connection.execute(
            """
            SELECT asset_id, content_sha256, encoding
            FROM assets
            WHERE dataset_id = ? AND content_sha256 = ? AND encoding = ?
            """,
            (self.dataset_id, digest, normalized_encoding),
        ).fetchone()
        return StoredAsset(str(row[0]), str(row[1]), str(row[2]))

    def item_id(self, source_key: str) -> str:
        return stable_uuid(self.dataset_id, f"item:{source_key}")

    def add_item(
        self,
        *,
        item_id: str | None = None,
        source_key: str,
        sample_weight: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        resolved_id = item_id or self.item_id(source_key)
        now = utc_now()
        self.connection.execute(
            """
            INSERT INTO dataset_items (
                item_id, dataset_id, sample_weight, source_key,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                sample_weight = excluded.sample_weight,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                resolved_id,
                self.dataset_id,
                sample_weight,
                source_key,
                json.dumps(metadata or {}, sort_keys=True, default=str),
                now,
                now,
            ),
        )
        return resolved_id

    def add_classification_label(
        self,
        class_index: int,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        label_id = stable_uuid(self.dataset_id, f"label:{int(class_index)}:{name}")
        self.connection.execute(
            """
            INSERT INTO classification_labels (
                label_id, dataset_id, class_index, name, parent_label_id, metadata_json
            ) VALUES (?, ?, ?, ?, NULL, ?)
            ON CONFLICT(dataset_id, class_index) DO UPDATE SET
                name = excluded.name,
                metadata_json = excluded.metadata_json
            """,
            (
                label_id,
                self.dataset_id,
                int(class_index),
                name,
                json.dumps(metadata or {}, sort_keys=True, default=str),
            ),
        )
        row = self.connection.execute(
            """
            SELECT label_id FROM classification_labels
            WHERE dataset_id = ? AND class_index = ?
            """,
            (self.dataset_id, int(class_index)),
        ).fetchone()
        return str(row[0])

    def add_classification_item(
        self,
        *,
        item_id: str,
        image_asset_id: str,
        label_id: str,
        source: str = "import",
        annotator: str | None = None,
        confidence: float | None = None,
        annotation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.connection.execute(
            """
            INSERT INTO classification_items (item_id, image_asset_id)
            VALUES (?, ?)
            ON CONFLICT(item_id) DO UPDATE SET image_asset_id = excluded.image_asset_id
            """,
            (item_id, image_asset_id),
        )
        current = self.connection.execute(
            """
            SELECT annotation_id, label_id FROM classification_annotations
            WHERE item_id = ? AND is_current = 1
            """,
            (item_id,),
        ).fetchone()
        if current and current["label_id"] == label_id:
            return str(current["annotation_id"])
        if current:
            self.connection.execute(
                "UPDATE classification_annotations SET is_current = 0 WHERE annotation_id = ?",
                (current["annotation_id"],),
            )
        resolved_annotation = annotation_id or str(uuid.uuid4())
        self.connection.execute(
            """
            INSERT INTO classification_annotations (
                annotation_id, item_id, label_id, created_at, annotator, source,
                confidence, status, is_current, parent_annotation_id, notes, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'accepted', 1, ?, NULL, ?)
            """,
            (
                resolved_annotation,
                item_id,
                label_id,
                utc_now(),
                annotator,
                source,
                confidence,
                current["annotation_id"] if current else None,
                json.dumps(metadata or {}, sort_keys=True, default=str),
            ),
        )
        return resolved_annotation

    def add_mask_item(
        self,
        *,
        item_id: str,
        image_asset_id: str,
        candidate_mask_asset_id: str | None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO mask_refinement_items (
                item_id, image_asset_id, candidate_mask_asset_id
            ) VALUES (?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                image_asset_id = excluded.image_asset_id,
                candidate_mask_asset_id = excluded.candidate_mask_asset_id
            """,
            (item_id, image_asset_id, candidate_mask_asset_id),
        )

    def classification_rows(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT di.item_id, di.sample_weight, di.source_key,
                   di.metadata_json, a.payload AS image_blob, a.encoding AS image_encoding,
                   a.shape_json AS image_dimensions, a.asset_id AS image_asset_id,
                   l.class_index, l.name AS class_name, ca.annotation_id
            FROM dataset_items di
            JOIN classification_items ci ON ci.item_id = di.item_id
            JOIN assets a ON a.asset_id = ci.image_asset_id
            LEFT JOIN classification_annotations ca
              ON ca.item_id = di.item_id AND ca.is_current = 1
            LEFT JOIN classification_labels l ON l.label_id = ca.label_id
            ORDER BY di.item_id
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def mask_rows(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT di.item_id, di.sample_weight, di.source_key,
                   di.metadata_json,
                   image.payload AS image_blob, image.encoding AS image_encoding,
                   image.shape_json AS image_dimensions,
                   candidate.payload AS candidate_blob,
                   candidate.encoding AS candidate_encoding,
                   candidate.shape_json AS candidate_dimensions,
                   mask.payload AS validated_blob,
                   mask.encoding AS validated_encoding,
                   mask.shape_json AS validated_dimensions,
                   ma.annotation_id AS current_annotation_id
            FROM dataset_items di
            JOIN mask_refinement_items mi ON mi.item_id = di.item_id
            JOIN assets image ON image.asset_id = mi.image_asset_id
            LEFT JOIN assets candidate ON candidate.asset_id = mi.candidate_mask_asset_id
            LEFT JOIN mask_annotations ma
              ON ma.item_id = di.item_id AND ma.is_current = 1
            LEFT JOIN assets mask ON mask.asset_id = ma.mask_asset_id
            ORDER BY di.item_id
            """
        ).fetchall()
        return [dict(row) for row in rows]
