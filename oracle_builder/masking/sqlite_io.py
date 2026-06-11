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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mask_annotations (
            annotation_id TEXT PRIMARY KEY,
            sample_uuid TEXT NOT NULL,
            created_at TEXT NOT NULL,
            annotator TEXT,
            mask_blob BLOB NOT NULL,
            mask_blob_encoding TEXT NOT NULL,
            mask_blob_dimensions TEXT NOT NULL,
            method TEXT,
            parameters_json TEXT,
            validation_json TEXT,
            accepted INTEGER DEFAULT 1,
            notes TEXT
        )
        """
    )


def ensure_mask_builder_columns(conn: sqlite3.Connection) -> None:
    ensure_schema(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(samples)").fetchall()}
    additions = {
        "input_aux_blob": "BLOB",
        "input_aux_blob_encoding": "TEXT",
        "input_aux_blob_dimensions": "TEXT",
    }
    for column, column_type in additions.items():
        if column not in columns:
            conn.execute(f"ALTER TABLE samples ADD COLUMN {column} {column_type}")


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
    row = conn.execute("SELECT * FROM samples WHERE uuid = ?", (uuid,)).fetchone()
    if row is None:
        raise KeyError(f"No sample found for uuid={uuid!r}")
    sample = dict(row)
    training_input = decode_blob(sample["input_blob"], sample["input_blob_encoding"], sample["input_blob_dimensions"])
    image, candidate_mask = split_training_input(np.asarray(training_input))
    if sample.get("input_aux_blob"):
        candidate_mask = decode_mask(
            sample["input_aux_blob"],
            sample.get("input_aux_blob_encoding"),
            sample.get("input_aux_blob_dimensions"),
        )
    mask = decode_mask(sample["output_blob"], sample["output_blob_encoding"], sample["output_blob_dimensions"])
    return {
        "uuid": sample["uuid"],
        "split": sample.get("split"),
        "image": np.asarray(image),
        "mask": mask,
        "candidate_mask": candidate_mask,
        "input_blob_encoding": sample.get("input_blob_encoding"),
        "input_blob_dimensions": sample.get("input_blob_dimensions"),
        "input_aux_blob_encoding": sample.get("input_aux_blob_encoding"),
        "input_aux_blob_dimensions": sample.get("input_aux_blob_dimensions"),
        "output_blob_encoding": sample.get("output_blob_encoding"),
        "output_blob_dimensions": sample.get("output_blob_dimensions"),
        "label_text": sample.get("label_text"),
        "sample_weight": sample.get("sample_weight"),
        "metadata": _json_loads(sample.get("metadata_json")),
    }


def list_samples(
    conn: sqlite3.Connection,
    split: str | None = None,
    missing_masks_only: bool = False,
) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    clauses = []
    values: list[Any] = []
    if split:
        clauses.append("split = ?")
        values.append(split)
    if missing_masks_only:
        clauses.append("(output_blob IS NULL OR length(output_blob) = 0)")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT uuid, split, label_text, input_blob_encoding, input_blob_dimensions,
               output_blob IS NOT NULL AND length(output_blob) > 0 AS has_mask,
               metadata_json
        FROM samples
        {where}
        ORDER BY uuid
        """,
        values,
    ).fetchall()
    return [
        {
            "uuid": row["uuid"],
            "split": row["split"],
            "label_text": row["label_text"],
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
    with conn:
        existing = conn.execute(
            "SELECT input_blob_dimensions, metadata_json FROM samples WHERE uuid = ?",
            (sample_uuid,),
        ).fetchone()
        if existing is None:
            raise KeyError(f"Cannot save mask annotation; sample does not exist: {sample_uuid}")
        conn.execute(
            """
            INSERT INTO mask_annotations (
                annotation_id, sample_uuid, created_at, annotator, mask_blob, mask_blob_encoding,
                mask_blob_dimensions, method, parameters_json, validation_json, accepted, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                annotation_id,
                sample_uuid,
                _utc_now(),
                annotator,
                blob,
                encoding,
                dimensions,
                method,
                json.dumps(parameters, sort_keys=True, default=str),
                json.dumps(validation, sort_keys=True, default=str),
                1 if accepted else 0,
                notes,
            ),
        )
        metadata = _json_loads(existing["metadata_json"] if hasattr(existing, "keys") else existing[1])
        input_shape = _shape_from_dimensions(existing["input_blob_dimensions"] if hasattr(existing, "keys") else existing[0])
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
        if accepted:
            conn.execute(
                """
                UPDATE samples
                SET output_blob = ?, output_blob_encoding = ?, output_blob_dimensions = ?, metadata_json = ?
                WHERE uuid = ?
                """,
                (blob, encoding, dimensions, json.dumps(metadata, sort_keys=True, default=str), sample_uuid),
            )
        else:
            conn.execute(
                "UPDATE samples SET metadata_json = ? WHERE uuid = ?",
                (json.dumps(metadata, sort_keys=True, default=str), sample_uuid),
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
        existing = conn.execute("SELECT metadata_json FROM samples WHERE uuid = ?", (uuid,)).fetchone()
        metadata = _json_loads(existing[0]) if existing else {}
        metadata.update(image_metadata)
        metadata["mask_builder_import"] = {
            **metadata.get("mask_builder_import", {}),
            "imported_at": _utc_now(),
            "image_encoding": input_encoding,
            "input_shape": _shape_from_dimensions(dimensions),
            "candidate_mask_shape": _shape_from_dimensions(candidate_dimensions),
        }
        if existing:
            conn.execute(
                """
                UPDATE samples
                SET input_blob = ?, input_blob_encoding = ?, input_blob_dimensions = ?,
                    input_aux_blob = ?, input_aux_blob_encoding = ?, input_aux_blob_dimensions = ?,
                    metadata_json = ?
                WHERE uuid = ?
                """,
                (
                    image_blob,
                    input_encoding,
                    dimensions,
                    candidate_blob,
                    candidate_encoding,
                    candidate_dimensions,
                    json.dumps(metadata, sort_keys=True, default=str),
                    uuid,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO samples (
                    uuid, split, input_blob, input_blob_encoding, input_blob_dimensions,
                    input_aux_blob, input_aux_blob_encoding, input_aux_blob_dimensions,
                    output_blob, output_blob_encoding, output_blob_dimensions,
                    label_text, sample_weight, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid,
                    None,
                    image_blob,
                    input_encoding,
                    dimensions,
                    candidate_blob,
                    candidate_encoding,
                    candidate_dimensions,
                    None,
                    None,
                    None,
                    None,
                    None,
                    json.dumps(metadata, sort_keys=True, default=str),
                ),
            )


def open_database(path: str | Path, create: bool = True) -> sqlite3.Connection:
    db_path = Path(path)
    if not create and not db_path.exists():
        raise FileNotFoundError(f"SQLite dataset does not exist: {db_path}")
    if create and db_path.parent != Path("."):
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    ensure_schema(conn)
    ensure_mask_annotation_table(conn)
    ensure_mask_builder_columns(conn)
    conn.commit()
    return conn
