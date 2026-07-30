from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from oracle_builder.classification.features import predict_classification_outputs
from oracle_builder.classification.features import build_feature_model
from oracle_builder.classification.evidence import IdentityEvidenceIndex
from oracle_builder.data.decoders import encode_npy
from oracle_builder.datasets.schema import dataset_fingerprint, read_dataset_info
from oracle_builder.evaluation.segmentation import binary_metrics
from oracle_builder.evaluation.segmentation import predict_reassembled_segmentation
from oracle_builder.evaluation.segmentation_targets import CANDIDATE_DELTA, reconstruct_validated_mask, reconstruct_validated_probability, segmentation_target_mode
from oracle_builder.inference.contracts import ArrayPayload
from oracle_builder.data.tiling import group_and_reassemble
from oracle_builder.progress import BatchProgress


def init_predictions_db(
    path: str | Path,
    source_sqlite: str | Path | None = None,
) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() and source_sqlite is not None:
        source = Path(source_sqlite)
        if source.resolve() == path.resolve():
            raise ValueError("Prediction output must differ from the source SQLite database")
        with sqlite3.connect(source) as source_connection, sqlite3.connect(path) as destination:
            source_connection.backup(destination)
    connection = sqlite3.connect(path)
    existing = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'predictions'"
    ).fetchone()
    if existing:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(predictions)")}
        if "prediction_set" not in columns:
            connection.close()
            raise ValueError(
                f"{path} uses the legacy single-set predictions schema; choose a new output path"
            )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_sets (
            prediction_set TEXT PRIMARY KEY,
            prediction_set_id TEXT,
            result_set_id TEXT,
            created_at TEXT NOT NULL,
            run_id TEXT,
            run_name TEXT,
            artifact_id TEXT,
            dataset_id TEXT,
            dataset_fingerprint_sha256 TEXT,
            config_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_set TEXT NOT NULL,
            uuid TEXT NOT NULL,
            result_id TEXT,
            split TEXT,
            y_true_blob BLOB,
            y_true_encoding TEXT,
            y_pred_blob BLOB,
            y_pred_encoding TEXT,
            y_prob_json TEXT,
            metrics_json TEXT,
            metadata_json TEXT,
            target_mode TEXT NOT NULL DEFAULT 'validated_mask',
            reconstructed_pred_blob BLOB,
            reconstructed_pred_encoding TEXT,
            features_blob BLOB,
            features_encoding TEXT,
            features_dim INTEGER,
            prediction_packet_json TEXT,
            logits_blob BLOB,
            logits_encoding TEXT,
            input_sha256 TEXT,
            output_sha256 TEXT,
            inference_result_json TEXT,
            PRIMARY KEY (prediction_set, uuid),
            FOREIGN KEY (prediction_set) REFERENCES prediction_sets(prediction_set),
            FOREIGN KEY (uuid) REFERENCES dataset_items(item_id)
        )
        """
    )
    prediction_columns = {row[1] for row in connection.execute("PRAGMA table_info(predictions)")}
    for column, definition in {
        "target_mode": "TEXT NOT NULL DEFAULT 'validated_mask'",
        "reconstructed_pred_blob": "BLOB",
        "reconstructed_pred_encoding": "TEXT",
        "features_blob": "BLOB",
        "features_encoding": "TEXT",
        "features_dim": "INTEGER",
        "prediction_packet_json": "TEXT",
        "result_id": "TEXT",
        "logits_blob": "BLOB",
        "logits_encoding": "TEXT",
        "input_sha256": "TEXT",
        "output_sha256": "TEXT",
        "inference_result_json": "TEXT",
    }.items():
        if column not in prediction_columns:
            connection.execute(f"ALTER TABLE predictions ADD COLUMN {column} {definition}")
    set_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(prediction_sets)")
    }
    for column in (
        "prediction_set_id",
        "result_set_id",
        "artifact_id",
        "dataset_id",
        "dataset_fingerprint_sha256",
    ):
        if column not in set_columns:
            connection.execute(
                f"ALTER TABLE prediction_sets ADD COLUMN {column} TEXT"
            )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_prediction_sets_id
        ON prediction_sets(prediction_set_id)
        WHERE prediction_set_id IS NOT NULL
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_prediction_result_id
        ON predictions(result_id)
        WHERE result_id IS NOT NULL
        """
    )
    return connection


def _upsert_prediction_set(
    connection: sqlite3.Connection,
    prediction_set: str,
    config: dict[str, Any],
) -> tuple[str, str]:
    run = config.get("run", {})
    artifact = config.get("artifact", {})
    dataset = config.get("dataset", {})
    actual_dataset = read_dataset_info(connection)
    actual_fingerprint = dataset_fingerprint(connection)
    if dataset.get("dataset_id") and dataset["dataset_id"] != actual_dataset["dataset_id"]:
        raise ValueError(
            "Prediction config dataset_id does not match the output dataset"
        )
    if (
        dataset.get("fingerprint_sha256")
        and dataset["fingerprint_sha256"] != actual_fingerprint
    ):
        raise ValueError(
            "Prediction config dataset fingerprint does not match the output dataset"
        )
    existing_ids = connection.execute(
        """
        SELECT prediction_set_id, result_set_id
        FROM prediction_sets WHERE prediction_set = ?
        """,
        (prediction_set,),
    ).fetchone()
    prediction_set_id = (
        existing_ids[0] if existing_ids and existing_ids[0] else str(uuid.uuid4())
    )
    result_set_id = (
        existing_ids[1] if existing_ids and existing_ids[1] else str(uuid.uuid4())
    )
    connection.execute(
        """
        INSERT INTO prediction_sets (
            prediction_set, prediction_set_id, result_set_id,
            created_at, run_id, run_name, artifact_id,
            dataset_id, dataset_fingerprint_sha256, config_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(prediction_set) DO UPDATE SET
            prediction_set_id = excluded.prediction_set_id,
            result_set_id = excluded.result_set_id,
            run_id = excluded.run_id,
            run_name = excluded.run_name,
            artifact_id = excluded.artifact_id,
            dataset_id = excluded.dataset_id,
            dataset_fingerprint_sha256 = excluded.dataset_fingerprint_sha256,
            config_json = excluded.config_json
        """,
        (
            prediction_set,
            prediction_set_id,
            result_set_id,
            datetime.now(timezone.utc).isoformat(),
            run.get("run_id"),
            run.get("name") or run.get("run_name"),
            artifact.get("artifact_id"),
            actual_dataset["dataset_id"],
            actual_fingerprint,
            json.dumps(config, sort_keys=True, default=str),
        ),
    )
    return prediction_set_id, result_set_id


def write_predictions_db(
    model,
    x: np.ndarray,
    y: Any,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    sqlite_path: str | Path,
    *,
    source_sqlite: str | Path | None = None,
    prediction_set: str = "default",
    evidence_index: IdentityEvidenceIndex | None = None,
    inference_batch_size: int | None = None,
) -> None:
    task = config["run"]["task"]
    if task == "segmentation":
        source_records = records
        predictions, y, records = predict_reassembled_segmentation(
            model,
            x,
            y,
            records,
            config,
            batch_size=inference_batch_size,
        )
        logits, logits_source = _segmentation_logits(
            model,
            x,
            source_records,
            predictions,
            config,
            batch_size=inference_batch_size,
        )
        features = None
    else:
        classification_outputs = predict_classification_outputs(
            model, x, batch_size=inference_batch_size
        )
        predictions = classification_outputs["probabilities"]
        features = classification_outputs["features"]
        logits = classification_outputs["logits"]
        logits_source = str(classification_outputs["logits_source"])
    connection = init_predictions_db(sqlite_path, source_sqlite=source_sqlite)
    _prediction_set_id, result_set_id = _upsert_prediction_set(
        connection, prediction_set, config
    )
    target_mode = segmentation_target_mode(config)
    segmentation_threshold = float(config.get("evaluation", {}).get("segmentation_threshold", 0.5))
    for record_index, (row, true_value, prediction, logit_values) in enumerate(
        zip(records, y, predictions, logits, strict=False)
    ):
        result_id = str(uuid.uuid4())
        input_payload = ArrayPayload(np.asarray(x[record_index]))
        output_payload = ArrayPayload(np.asarray(prediction))
        prediction_metadata = dict(row.get("metadata", {}))
        prediction_metadata["prediction"] = {
            "tile_count": int(row.get("tile_count", 1)),
            "source_shape": row.get("source_shape"),
            "tiling_enabled": bool(config.get("tiling", {}).get("enabled", False)),
            "overlap_fraction": float(config.get("tiling", {}).get("overlap_fraction", 0.0)),
            "blend_mode": config.get("tiling", {}).get("blend_mode", "uniform"),
        }
        if true_value is None:
            y_true_blob = None
            true_encoding = None
            if task == "classification":
                pred_class = int(np.argmax(prediction))
                y_pred_blob = str(pred_class).encode("utf-8")
                y_prob_json = json.dumps([float(v) for v in prediction])
                pred_encoding = "int"
            else:
                y_pred_blob = encode_npy(np.asarray(prediction))
                y_prob_json = None
                pred_encoding = "npy"
                reconstructed_prediction = _reconstructed_prediction(row, prediction, target_mode)
            metrics_json = None
        elif task == "classification":
            pred_class = int(np.argmax(prediction))
            y_true_blob = str(int(true_value)).encode("utf-8")
            y_pred_blob = str(pred_class).encode("utf-8")
            y_prob_json = json.dumps([float(v) for v in prediction])
            metrics_json = json.dumps({"correct": bool(pred_class == int(true_value))})
            true_encoding = pred_encoding = "int"
        else:
            if target_mode == CANDIDATE_DELTA:
                predicted_delta = np.asarray(prediction) >= segmentation_threshold
                reconstructed_mask = reconstruct_validated_mask(row["candidate_mask"], predicted_delta)
                reconstructed_prediction = reconstruct_validated_probability(row["candidate_mask"], prediction)
                delta_metrics = binary_metrics(true_value, prediction, threshold=segmentation_threshold)
                reconstructed_metrics = binary_metrics(row["validated_mask"], reconstructed_mask)
                candidate_metrics = binary_metrics(row["validated_mask"], row["candidate_mask"])
                candidate_binary = np.asarray(row["candidate_mask"]) > 0.5
                validated_binary = np.asarray(row["validated_mask"]) > 0.5
                metrics = {
                    **reconstructed_metrics,
                    **{f"delta_{key}": value for key, value in delta_metrics.items()},
                    "candidate_dice": candidate_metrics["dice"],
                    "dice_improvement": reconstructed_metrics["dice"] - candidate_metrics["dice"],
                    "correction_fraction": float(np.logical_xor(candidate_binary, validated_binary).mean()),
                    "addition_pixels": int(np.logical_and(~candidate_binary, validated_binary).sum()),
                    "removal_pixels": int(np.logical_and(candidate_binary, ~validated_binary).sum()),
                }
            else:
                metrics = binary_metrics(true_value, prediction, threshold=segmentation_threshold)
                reconstructed_prediction = np.asarray(prediction)
            metrics["probability_threshold"] = segmentation_threshold
            metrics["segmentation_target"] = target_mode
            y_true_blob = encode_npy(np.asarray(true_value))
            y_pred_blob = encode_npy(np.asarray(prediction))
            y_prob_json = None
            metrics_json = json.dumps(metrics)
            true_encoding = pred_encoding = "npy"
        feature = None
        if features is not None:
            feature = np.asarray(features[record_index], dtype="float32")
            prediction_metadata["prediction"]["features_dim"] = int(feature.shape[-1])
            prediction_metadata["prediction"]["features_normalized"] = bool(
                config.get("model", {}).get("normalize_embeddings", True)
            )
        prediction_packet_json = None
        if task == "classification" and evidence_index is not None and feature is not None:
            prediction_packet_json = json.dumps(
                evidence_index.packet(
                    feature,
                    np.asarray(prediction),
                    query_uuid=row["uuid"],
                    k=int(config.get("evidence", {}).get("knn_k", 5)),
                )
            )
        connection.execute(
            """
            INSERT OR REPLACE INTO predictions (
                prediction_set, uuid, split, y_true_blob, y_true_encoding,
                y_pred_blob, y_pred_encoding, y_prob_json, metrics_json, metadata_json,
                target_mode, reconstructed_pred_blob, reconstructed_pred_encoding,
                features_blob, features_encoding, features_dim, prediction_packet_json
                , result_id, logits_blob, logits_encoding, input_sha256,
                output_sha256, inference_result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_set,
                row["uuid"],
                row["split"],
                y_true_blob,
                true_encoding,
                y_pred_blob,
                pred_encoding,
                y_prob_json,
                metrics_json,
                json.dumps(prediction_metadata),
                target_mode,
                encode_npy(np.asarray(reconstructed_prediction)) if task == "segmentation" else None,
                "npy" if task == "segmentation" else None,
                encode_npy(feature) if feature is not None else None,
                "npy" if feature is not None else None,
                int(feature.shape[-1]) if feature is not None else None,
                prediction_packet_json,
                result_id,
                encode_npy(np.asarray(logit_values, dtype="float32")),
                "npy",
                input_payload.sha256,
                output_payload.sha256,
                json.dumps(
                    {
                        "schema_name": "oracle_builder.inference_result",
                        "schema_version": "1.0.0",
                        "result_id": result_id,
                        "result_set_id": result_set_id,
                        "item_id": row["uuid"],
                        "input_sha256": input_payload.sha256,
                        "status": "ok",
                        "model": {
                            "artifact_id": config.get("artifact", {}).get(
                                "artifact_id"
                            ),
                            "run_id": config.get("run", {}).get("run_id"),
                            "task": task,
                            "architecture": config.get("run", {}).get("model"),
                        },
                        "output": {
                            "type": (
                                "classification"
                                if task == "classification"
                                else "mask_refinement"
                            ),
                            "logits_source": logits_source,
                            "logits_sha256": ArrayPayload(
                                np.asarray(logit_values, dtype="float32")
                            ).sha256,
                            "output_sha256": output_payload.sha256,
                        },
                    },
                    sort_keys=True,
                ),
            ),
        )
    connection.commit()
    connection.close()


def _segmentation_logits(
    model,
    x: np.ndarray,
    records: list[dict[str, Any]],
    probabilities: list[np.ndarray],
    config: dict[str, Any],
    *,
    batch_size: int | None = None,
) -> tuple[list[np.ndarray], str]:
    try:
        from tensorflow import keras

        logits_model = keras.Model(model.input, model.get_layer("logits").output)
        predict_options = {"verbose": 0}
        if batch_size is not None:
            predict_options["batch_size"] = batch_size
        tile_logits = logits_model.predict(x, **predict_options)
        logits, _ = group_and_reassemble(
            tile_logits,
            records,
            blend_mode=config.get("tiling", {}).get("blend_mode", "hann"),
        )
        return logits, "model"
    except (AttributeError, ValueError):
        derived = [
            np.log(np.clip(value, 1e-7, 1.0 - 1e-7))
            - np.log1p(-np.clip(value, 1e-7, 1.0 - 1e-7))
            for value in probabilities
        ]
        return derived, "derived_inverse_sigmoid"


def _reconstructed_prediction(row: dict[str, Any], prediction: Any, target_mode: str) -> np.ndarray:
    if target_mode == CANDIDATE_DELTA:
        return reconstruct_validated_probability(row["candidate_mask"], prediction)
    return np.asarray(prediction)


def write_classification_predictions_streaming(
    model,
    dataset,
    sample_index,
    config: dict[str, Any],
    sqlite_path: str | Path,
    *,
    source_sqlite: str | Path,
    prediction_set: str,
    evidence_index: IdentityEvidenceIndex | None = None,
    progress: bool = True,
) -> int:
    connection = init_predictions_db(sqlite_path, source_sqlite=source_sqlite)
    _prediction_set_id, result_set_id = _upsert_prediction_set(
        connection, prediction_set, config
    )
    feature_model = build_feature_model(model)
    commit_batches = max(
        1, int(config.get("output", {}).get("prediction_commit_batches", 20))
    )
    written = 0
    display = BatchProgress(
        "Writing classification predictions",
        len(sample_index),
        enabled=progress,
    )
    for batch_number, (images, positions) in enumerate(dataset, start=1):
        outputs = feature_model(images, training=False)
        logits = np.asarray(outputs["logits"], dtype="float32")
        probabilities = np.asarray(outputs["probabilities"])
        features = np.asarray(outputs["features"], dtype="float32")
        for position, probability, logit_values, feature in zip(
            np.asarray(positions, dtype="int64"),
            probabilities,
            logits,
            features,
            strict=True,
        ):
            ref = sample_index.refs[int(position)]
            pred_class = int(np.argmax(probability))
            packet = (
                evidence_index.packet(
                    feature,
                    probability,
                    query_uuid=ref.uuid,
                    k=int(config.get("evidence", {}).get("knn_k", 5)),
                )
                if evidence_index is not None
                else None
            )
            metadata = json.loads(ref.metadata_json) if ref.metadata_json else {}
            metadata["prediction"] = {
                "features_dim": int(feature.shape[-1]),
                "features_normalized": bool(
                    config.get("model", {}).get("normalize_embeddings", True)
                ),
                "streaming": True,
            }
            connection.execute(
                """
                INSERT OR REPLACE INTO predictions (
                    prediction_set, uuid, split, y_true_blob, y_true_encoding,
                    y_pred_blob, y_pred_encoding, y_prob_json, metrics_json,
                    metadata_json, target_mode, reconstructed_pred_blob,
                    reconstructed_pred_encoding, features_blob, features_encoding,
                    features_dim, prediction_packet_json
                    , result_id, logits_blob, logits_encoding, input_sha256,
                    output_sha256, inference_result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_set,
                    ref.uuid,
                    ref.split,
                    str(ref.target).encode() if ref.target is not None else None,
                    "int" if ref.target is not None else None,
                    str(pred_class).encode(),
                    "int",
                    json.dumps([float(value) for value in probability]),
                    (
                        json.dumps({"correct": bool(pred_class == ref.target)})
                        if ref.target is not None
                        else None
                    ),
                    json.dumps(metadata),
                    "classification",
                    None,
                    None,
                    encode_npy(feature),
                    "npy",
                    int(feature.shape[-1]),
                    json.dumps(packet) if packet is not None else None,
                    (result_id := str(uuid.uuid4())),
                    encode_npy(logit_values),
                    "npy",
                    (input_payload := ArrayPayload(
                        np.asarray(images)[
                            list(np.asarray(positions, dtype="int64")).index(
                                int(position)
                            )
                        ]
                    )).sha256,
                    (output_payload := ArrayPayload(probability)).sha256,
                    json.dumps(
                        {
                            "schema_name": "oracle_builder.inference_result",
                            "schema_version": "1.0.0",
                            "result_id": result_id,
                            "result_set_id": result_set_id,
                            "item_id": ref.uuid,
                            "input_sha256": input_payload.sha256,
                            "status": "ok",
                            "model": {
                                "artifact_id": config.get("artifact", {}).get(
                                    "artifact_id"
                                ),
                                "run_id": config.get("run", {}).get("run_id"),
                                "task": "classification",
                                "architecture": config.get("run", {}).get(
                                    "model"
                                ),
                            },
                            "output": {
                                "type": "classification",
                                "logits_source": "model",
                                "logits": [float(value) for value in logit_values],
                                "output_sha256": output_payload.sha256,
                            },
                        },
                        sort_keys=True,
                    ),
                ),
            )
            written += 1
        display.update(len(np.asarray(positions)))
        if batch_number % commit_batches == 0:
            connection.commit()
    display.close()
    connection.commit()
    connection.close()
    return written
