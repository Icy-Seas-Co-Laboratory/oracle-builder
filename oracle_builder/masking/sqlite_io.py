from __future__ import annotations

import io
import json
import sqlite3
import uuid as uuid_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from oracle_builder.data.decoders import decode_blob, encode_npy
from oracle_builder.datasets.repository import SQLiteDatasetRepository
from oracle_builder.datasets.schema import initialize_database
from oracle_builder.datasets.legacy_roi import ensure_mask_refinement_database
from oracle_builder.data.sqlite_dataset import ensure_schema
from oracle_builder.masking.image_io import encode_image_png


def _json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_mask_annotation_table(conn: sqlite3.Connection) -> None:
    initialize_database(conn, "mask_refinement")


def ensure_mask_builder_columns(conn: sqlite3.Connection) -> None:
    initialize_database(conn, "mask_refinement")


def encode_mask(mask: np.ndarray, mask_encoding: str) -> tuple[bytes, str, str]:
    encoding = mask_encoding.lower()
    array = np.asarray(mask)
    if array.ndim == 2:
        dimensions = json.dumps([int(array.shape[0]), int(array.shape[1]), 1])
        mask_array = array
    elif array.ndim == 3 and array.shape[-1] == 1:
        dimensions = json.dumps([int(array.shape[0]), int(array.shape[1]), 1])
        mask_array = array[..., 0]
    else:
        dimensions = json.dumps(list(array.shape))
        mask_array = array
    if encoding == "png":
        if mask_array.ndim != 2:
            raise ValueError("PNG mask encoding only supports 2D masks")
        png_array = np.where(mask_array > 0, 255, 0).astype("uint8")
        buffer = io.BytesIO()
        Image.fromarray(png_array, mode="L").save(buffer, format="PNG")
        return buffer.getvalue(), "png", dimensions
    if encoding == "npy":
        if array.ndim == 2:
            array = array[..., None]
        return encode_npy(array), "npy", dimensions
    raise ValueError("mask_encoding must be 'png' or 'npy'")


def decode_mask(blob: bytes | None, encoding: str | None, dimensions: str | None = None) -> np.ndarray | None:
    if not blob:
        return None
    encoding = (encoding or "").lower()
    if encoding == "png":
        image = Image.open(io.BytesIO(blob))
        return (np.asarray(image) > 0).astype("uint8")
    if encoding == "npy":
        array = decode_blob(blob, "npy", dimensions)
        return (np.asarray(array) > 0).astype("uint8")
    raise ValueError(f"Unsupported mask encoding: {encoding}")


def encode_training_input(image: np.ndarray, candidate_mask: np.ndarray | None = None) -> tuple[bytes, str, str]:
    if candidate_mask is None:
        image_array = np.asarray(image)
        if image_array.ndim != 2:
            raise ValueError("Training input without a candidate mask must be a 2D ROI")
        return encode_image_png(image_array), "png", json.dumps(list(image_array.shape))
    tensor = make_training_input_tensor(image, candidate_mask)
    return encode_npy(tensor), "npy", json.dumps(list(tensor.shape))


def make_training_input_tensor(image: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    roi = _roi_channel(image)
    candidate = _mask_channel(candidate_mask, roi.shape)
    if roi.dtype.kind in {"u", "i"} and roi.max(initial=0) > 1:
        candidate = (candidate * 255).astype(roi.dtype)
    else:
        candidate = candidate.astype("float32")
        roi = roi.astype("float32")
    return np.stack([roi, candidate], axis=-1)


def split_training_input(array: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    value = np.asarray(array)
    if value.ndim == 3 and value.shape[-1] == 2:
        candidate = (value[..., 1] > 0).astype("uint8")
        return value[..., 0], candidate
    return value, None


def _roi_channel(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[..., 0]
    if array.ndim == 3 and array.shape[-1] >= 3:
        rgb = array[..., :3].astype("float32")
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        if array.dtype.kind in {"u", "i"}:
            return np.clip(np.rint(gray), np.iinfo(array.dtype).min, np.iinfo(array.dtype).max).astype(array.dtype)
        return gray.astype(array.dtype)
    raise ValueError(f"Unsupported ROI image shape: {array.shape}")


def _mask_channel(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.shape != shape:
        raise ValueError(f"Candidate mask shape {array.shape} does not match ROI shape {shape}")
    return (array > 0).astype("uint8")


def load_sample(conn: sqlite3.Connection, uuid: str) -> dict[str, Any]:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT di.item_id, di.sample_weight, di.metadata_json,
               image.payload AS image_blob, image.encoding AS image_encoding,
               image.shape_json AS image_dimensions,
               candidate.payload AS candidate_blob,
               candidate.encoding AS candidate_encoding,
               candidate.shape_json AS candidate_dimensions,
               mask.payload AS mask_blob, mask.encoding AS mask_encoding,
               mask.shape_json AS mask_dimensions
        FROM dataset_items di
        JOIN mask_refinement_items mi ON mi.item_id = di.item_id
        JOIN assets image ON image.asset_id = mi.image_asset_id
        LEFT JOIN assets candidate ON candidate.asset_id = mi.candidate_mask_asset_id
        LEFT JOIN mask_annotations ma
          ON ma.item_id = di.item_id AND ma.is_current = 1
        LEFT JOIN assets mask ON mask.asset_id = ma.mask_asset_id
        WHERE di.item_id = ?
        """,
        (uuid,),
    ).fetchone()
    if row is None:
        raise KeyError(f"No sample found for uuid={uuid!r}")
    sample = dict(row)
    image = decode_blob(
        sample["image_blob"], sample["image_encoding"], sample["image_dimensions"]
    )
    candidate_mask = None
    if sample.get("candidate_blob"):
        candidate_mask = decode_mask(
            sample["candidate_blob"],
            sample.get("candidate_encoding"),
            sample.get("candidate_dimensions"),
        )
    mask = decode_mask(
        sample.get("mask_blob"),
        sample.get("mask_encoding"),
        sample.get("mask_dimensions"),
    )
    return {
        "uuid": sample["item_id"],
        "image": np.asarray(image),
        "mask": mask,
        "candidate_mask": candidate_mask,
        "input_blob_encoding": sample.get("image_encoding"),
        "input_blob_dimensions": sample.get("image_dimensions"),
        "input_aux_blob_encoding": sample.get("candidate_encoding"),
        "input_aux_blob_dimensions": sample.get("candidate_dimensions"),
        "output_blob_encoding": sample.get("mask_encoding"),
        "output_blob_dimensions": sample.get("mask_dimensions"),
        "label_text": None,
        "sample_weight": sample.get("sample_weight"),
        "metadata": _json_loads(sample.get("metadata_json")),
    }


def list_samples(
    conn: sqlite3.Connection,
    missing_masks_only: bool = False,
) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    clauses = []
    values: list[Any] = []
    if missing_masks_only:
        clauses.append("ma.annotation_id IS NULL")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT di.item_id, image.encoding AS input_blob_encoding,
               image.shape_json AS input_blob_dimensions,
               ma.annotation_id IS NOT NULL AS has_mask, di.metadata_json
        FROM dataset_items di
        JOIN mask_refinement_items mi ON mi.item_id = di.item_id
        JOIN assets image ON image.asset_id = mi.image_asset_id
        LEFT JOIN mask_annotations ma
          ON ma.item_id = di.item_id AND ma.is_current = 1
        {where}
        ORDER BY di.item_id
        """,
        values,
    ).fetchall()
    return [
        {
            "uuid": row["item_id"],
            "label_text": None,
            "input_blob_encoding": row["input_blob_encoding"],
            "input_blob_dimensions": row["input_blob_dimensions"],
            "has_mask": bool(row["has_mask"]),
            "metadata": _json_loads(row["metadata_json"]),
        }
        for row in rows
    ]


def save_mask_annotation(
    conn: sqlite3.Connection,
    sample_uuid: str,
    mask: np.ndarray,
    mask_encoding: str,
    method: str,
    parameters: dict,
    validation: dict,
    accepted: bool = True,
    annotator: str | None = None,
    notes: str | None = None,
) -> str:
    ensure_mask_annotation_table(conn)
    ensure_mask_builder_columns(conn)
    blob, encoding, dimensions = encode_mask(mask, mask_encoding)
    annotation_id = str(uuid_module.uuid4())
    repository = SQLiteDatasetRepository(conn)
    with conn:
        existing = conn.execute(
            """
            SELECT image.shape_json, di.metadata_json
            FROM dataset_items di
            JOIN mask_refinement_items mi ON mi.item_id = di.item_id
            JOIN assets image ON image.asset_id = mi.image_asset_id
            WHERE di.item_id = ?
            """,
            (sample_uuid,),
        ).fetchone()
        if existing is None:
            raise KeyError(f"Cannot save mask annotation; sample does not exist: {sample_uuid}")
        mask_asset = repository.add_asset(
            blob,
            encoding=encoding,
            media_type="image/png" if encoding == "png" else "application/x-npy",
            shape=_shape_from_dimensions(dimensions),
            dtype="uint8",
        )
        current = conn.execute(
            """
            SELECT annotation_id FROM mask_annotations
            WHERE item_id = ? AND is_current = 1
            """,
            (sample_uuid,),
        ).fetchone()
        if accepted and current:
            conn.execute(
                "UPDATE mask_annotations SET is_current = 0 WHERE annotation_id = ?",
                (current[0],),
            )
        conn.execute(
            """
            INSERT INTO mask_annotations (
                annotation_id, item_id, mask_asset_id, created_at, annotator, method,
                parameters_json, validation_json, status, is_current,
                parent_annotation_id, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                annotation_id,
                sample_uuid,
                mask_asset.asset_id,
                _utc_now(),
                annotator,
                method,
                json.dumps(parameters, sort_keys=True, default=str),
                json.dumps(validation, sort_keys=True, default=str),
                "accepted" if accepted else "rejected",
                1 if accepted else 0,
                current[0] if current else None,
                notes,
            ),
        )
        metadata = _json_loads(existing["metadata_json"] if hasattr(existing, "keys") else existing[1])
        input_shape = _shape_from_dimensions(existing["shape_json"] if hasattr(existing, "keys") else existing[0])
        output_shape = _shape_from_dimensions(dimensions)
        if input_shape and len(input_shape) == 2:
            input_shape = [input_shape[0], input_shape[1], 1]
        if output_shape and len(output_shape) == 2:
            output_shape = [output_shape[0], output_shape[1], 1]
        metadata["mask_builder"] = {
            **metadata.get("mask_builder", {}),
            "last_annotation_id": annotation_id,
            "last_saved_at": _utc_now(),
            "last_mask_encoding": encoding,
            "accepted": bool(accepted),
            "training_task": "segmentation",
            "mask_type": "binary",
            "input_shape": input_shape,
            "output_shape": output_shape,
        }
        conn.execute(
            """
            UPDATE dataset_items
            SET metadata_json = ?, updated_at = ?
            WHERE item_id = ?
            """,
            (
                json.dumps(metadata, sort_keys=True, default=str),
                _utc_now(),
                sample_uuid,
            ),
        )
    return annotation_id


def _shape_from_dimensions(dimensions: str | None) -> list[int] | None:
    if not dimensions:
        return None
    try:
        shape = json.loads(dimensions)
    except json.JSONDecodeError:
        return None
    if not isinstance(shape, list):
        return None
    return [int(value) for value in shape]


def create_or_update_image_sample(
    conn: sqlite3.Connection,
    uuid: str,
    image: np.ndarray,
    image_encoding: str,
    image_metadata: dict,
    candidate_mask: np.ndarray | None = None,
) -> None:
    ensure_mask_builder_columns(conn)
    encoding = image_encoding.lower()
    if encoding != "png":
        raise ValueError("Initial image imports currently support image_encoding='png'")
    if candidate_mask is None:
        image_blob = encode_image_png(image)
        dimensions = json.dumps(list(np.asarray(image).shape))
        input_encoding = encoding
    else:
        image_blob, input_encoding, dimensions = encode_training_input(image, candidate_mask)
    candidate_blob = candidate_encoding = candidate_dimensions = None
    if candidate_mask is not None:
        candidate_blob, candidate_encoding, candidate_dimensions = encode_mask(candidate_mask, "png")
    with conn:
        repository = SQLiteDatasetRepository(conn)
        existing = conn.execute(
            "SELECT metadata_json FROM dataset_items WHERE item_id = ?", (uuid,)
        ).fetchone()
        metadata = _json_loads(existing[0]) if existing else {}
        metadata.update(image_metadata)
        metadata["mask_builder_import"] = {
            **metadata.get("mask_builder_import", {}),
            "imported_at": _utc_now(),
            "image_encoding": input_encoding,
            "input_shape": _shape_from_dimensions(dimensions),
            "candidate_mask_shape": _shape_from_dimensions(candidate_dimensions),
        }
        image_array = np.asarray(image)
        image_asset = repository.add_asset(
            encode_image_png(image_array),
            encoding="png",
            media_type="image/png",
            shape=image_array.shape,
            dtype=str(image_array.dtype),
        )
        candidate_asset_id = None
        if candidate_blob is not None:
            candidate_asset = repository.add_asset(
                candidate_blob,
                encoding=candidate_encoding,
                media_type="image/png",
                shape=_shape_from_dimensions(candidate_dimensions),
                dtype="uint8",
            )
            candidate_asset_id = candidate_asset.asset_id
        item_id = repository.add_item(
            item_id=uuid,
            source_key=str(
                image_metadata.get("pelagia_detection_id")
                or image_metadata.get("source_path")
                or uuid
            ),
            metadata=metadata,
        )
        repository.add_mask_item(
            item_id=item_id,
            image_asset_id=image_asset.asset_id,
            candidate_mask_asset_id=candidate_asset_id,
        )


def open_database(path: str | Path, create: bool = True) -> sqlite3.Connection:
    db_path = Path(path)
    if not create and not db_path.exists():
        raise FileNotFoundError(f"SQLite dataset does not exist: {db_path}")
    if create and db_path.parent != Path("."):
        db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        ensure_mask_refinement_database(db_path)
    conn = sqlite3.connect(db_path)
    initialize_database(conn, "mask_refinement", name=db_path.stem)
    conn.commit()
    return conn
