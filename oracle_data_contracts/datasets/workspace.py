"""Annotation-workspace writes for labels, reviews, and derived model evidence."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import uuid
from typing import Any

import numpy as np

from oracle_data_contracts.datasets.schema import (
    dataset_fingerprint,
    read_dataset_info,
    utc_now,
)


def _encode_npy(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    return buffer.getvalue()


def add_annotation_label(
    connection: sqlite3.Connection,
    name: str,
    *,
    label_id: str | None = None,
    display_name: str | None = None,
    parent_label_id: str | None = None,
    rank: str | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    info = read_dataset_info(connection)
    resolved = label_id or str(uuid.uuid4())
    connection.execute(
        """
        INSERT INTO annotation_labels (
            label_id, dataset_id, name, display_name, parent_label_id, rank,
            description, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            resolved,
            info["dataset_id"],
            name,
            display_name,
            parent_label_id,
            rank,
            description,
            json.dumps(metadata or {}, sort_keys=True, default=str),
            utc_now(),
        ),
    )
    return resolved


def add_item_label_annotation(
    connection: sqlite3.Connection,
    item_id: str,
    label_id: str,
    *,
    annotator: str | None = None,
    method: str = "manual",
    confidence: float | None = None,
    status: str = "accepted",
    parameters: dict[str, Any] | None = None,
    notes: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    prior = connection.execute(
        """
        SELECT annotation_id FROM item_label_annotations
        WHERE item_id = ? AND is_current = 1
        """,
        (item_id,),
    ).fetchone()
    if prior is not None:
        connection.execute(
            "UPDATE item_label_annotations SET is_current = 0 WHERE annotation_id = ?",
            (prior[0],),
        )
    annotation_id = str(uuid.uuid4())
    connection.execute(
        """
        INSERT INTO item_label_annotations (
            annotation_id, item_id, label_id, created_at, annotator, method,
            confidence, status, is_current, parent_annotation_id,
            parameters_json, notes, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
        """,
        (
            annotation_id,
            item_id,
            label_id,
            utc_now(),
            annotator,
            method,
            confidence,
            status,
            prior[0] if prior else None,
            json.dumps(parameters or {}, sort_keys=True, default=str),
            notes,
            json.dumps(metadata or {}, sort_keys=True, default=str),
        ),
    )
    return annotation_id


def add_annotation_review(
    connection: sqlite3.Connection,
    annotation_id: str,
    reviewer: str,
    decision: str,
    *,
    notes: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    review_id = str(uuid.uuid4())
    connection.execute(
        """
        INSERT INTO annotation_reviews (
            review_id, annotation_id, reviewer, decision, created_at, notes, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            review_id, annotation_id, reviewer, decision, utc_now(), notes,
            json.dumps(metadata or {}, sort_keys=True, default=str),
        ),
    )
    return review_id


def create_inference_run(
    connection: sqlite3.Connection,
    *,
    name: str | None = None,
    model_artifact_id: str | None = None,
    model_run_id: str | None = None,
    model_artifact_fingerprint_sha256: str | None = None,
    input_contract: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    software_environment: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    info = read_dataset_info(connection)
    run_id = str(uuid.uuid4())
    connection.execute(
        """
        INSERT INTO inference_runs (
            inference_run_id, dataset_id, dataset_fingerprint_sha256,
            model_artifact_id, model_run_id, model_artifact_fingerprint_sha256,
            name, status, created_at, input_contract_json, parameters_json,
            software_environment_json, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?)
        """,
        (
            run_id, info["dataset_id"], dataset_fingerprint(connection),
            model_artifact_id, model_run_id, model_artifact_fingerprint_sha256,
            name, utc_now(),
            json.dumps(input_contract or {}, sort_keys=True, default=str),
            json.dumps(parameters or {}, sort_keys=True, default=str),
            json.dumps(software_environment or {}, sort_keys=True, default=str),
            json.dumps(metadata or {}, sort_keys=True, default=str),
        ),
    )
    return run_id


def complete_inference_run(
    connection: sqlite3.Connection,
    inference_run_id: str,
    *,
    status: str = "complete",
) -> None:
    if status not in {"complete", "failed", "cancelled"}:
        raise ValueError("status must be complete, failed, or cancelled")
    cursor = connection.execute(
        """
        UPDATE inference_runs SET status = ?, completed_at = ?
        WHERE inference_run_id = ? AND status = 'running'
        """,
        (status, utc_now(), inference_run_id),
    )
    if cursor.rowcount != 1:
        raise ValueError(f"Inference run is not running: {inference_run_id}")


def store_evidence_array(
    connection: sqlite3.Connection, values: Any, *, metadata: dict[str, Any] | None = None
) -> str:
    array = np.asarray(values)
    payload = _encode_npy(array)
    digest = hashlib.sha256(payload).hexdigest()
    row = connection.execute(
        "SELECT array_id FROM evidence_arrays WHERE content_sha256 = ?", (digest,)
    ).fetchone()
    if row is not None:
        return str(row[0])
    array_id = str(uuid.uuid4())
    connection.execute(
        """
        INSERT INTO evidence_arrays (
            array_id, content_sha256, payload, shape_json, dtype, created_at, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            array_id, digest, payload, json.dumps(list(array.shape)), str(array.dtype),
            utc_now(), json.dumps(metadata or {}, sort_keys=True, default=str),
        ),
    )
    return array_id


def add_model_evidence(
    connection: sqlite3.Connection,
    inference_run_id: str,
    item_id: str,
    *,
    predicted_label_id: str | None = None,
    prediction_confidence: float | None = None,
    nearest_neighbor_similarity: float | None = None,
    top_k_label_agreement: float | None = None,
    weighted_label_support: float | None = None,
    label_margin: float | None = None,
    logits: Any | None = None,
    embedding: Any | None = None,
    output: Any | None = None,
    packet: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    evidence_id = str(uuid.uuid4())
    arrays = [
        store_evidence_array(connection, value)
        if value is not None else None
        for value in (logits, embedding, output)
    ]
    connection.execute(
        """
        INSERT INTO model_evidence (
            evidence_id, inference_run_id, item_id, predicted_label_id,
            prediction_confidence, nearest_neighbor_similarity, top_k_label_agreement,
            weighted_label_support, label_margin, logits_array_id, embedding_array_id,
            output_array_id, packet_json, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            evidence_id, inference_run_id, item_id, predicted_label_id,
            prediction_confidence, nearest_neighbor_similarity, top_k_label_agreement,
            weighted_label_support, label_margin, *arrays,
            json.dumps(packet or {}, sort_keys=True, default=str),
            json.dumps(metadata or {}, sort_keys=True, default=str), utc_now(),
        ),
    )
    return evidence_id
