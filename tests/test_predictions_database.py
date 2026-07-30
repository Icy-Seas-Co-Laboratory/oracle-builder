from __future__ import annotations

import json
import sqlite3
import uuid

import numpy as np

from oracle_builder.data.decoders import decode_blob
from oracle_builder.classification.evidence import build_evidence_index
from oracle_builder.data.sqlite_dataset import (
    create_synthetic_classification,
    create_synthetic_segmentation,
    load_arrays,
    load_prediction_arrays,
)
from oracle_builder.evaluation.predictions import write_predictions_db
from oracle_builder.datasets.schema import dataset_fingerprint, read_dataset_info
from oracle_builder.registry import get_model_builder


class FakeClassificationModel:
    def predict(self, x, verbose=0):
        probabilities = np.zeros((len(x), 2), dtype="float32")
        probabilities[:, 0] = 0.75
        probabilities[:, 1] = 0.25
        return probabilities


class FakeSegmentationModel:
    def predict(self, x, verbose=0):
        return np.zeros((*x.shape[:3], 1), dtype="float32")


def test_prediction_database_copies_source_and_supports_multiple_sets(tmp_path):
    source = tmp_path / "training.sqlite"
    output = tmp_path / "training_with_predictions.sqlite"
    create_synthetic_classification(source, n=10, shape=(8, 8, 1), classes=2)
    with sqlite3.connect(source) as source_connection:
        source_info = read_dataset_info(source_connection)
        source_fingerprint = dataset_fingerprint(source_connection)
    config = {
        "run": {
            "task": "classification",
            "model": "simple_cnn",
            "seed": 123,
            "run_id": "61893e71-dac4-4ddd-acd3-abaa84ae0f27",
        },
        "artifact": {
            "artifact_id": "89d7a5e7-483a-46ad-a2cf-1edfb8057e3d"
        },
        "dataset": {
            "dataset_id": source_info["dataset_id"],
            "fingerprint_sha256": source_fingerprint,
        },
        "data": {
            "input_shape": [8, 8, 1],
            "num_classes": 2,
            "validation_split": 0.2,
            "test_split": 0.1,
        },
    }
    model = FakeClassificationModel()

    total = 0
    for split in ("train", "validation", "test"):
        x, y, records = load_arrays(source, config, split=split)
        total += len(records)
        write_predictions_db(
            model,
            x,
            y,
            records,
            config,
            output,
            source_sqlite=source,
            prediction_set="run-a",
        )
    x, y, records = load_arrays(source, config, split="test")
    write_predictions_db(
        model,
        x,
        y,
        records,
        config,
        output,
        source_sqlite=source,
        prediction_set="run-b",
    )

    with sqlite3.connect(output) as connection:
        assert connection.execute("SELECT count(*) FROM dataset_items").fetchone()[0] == 10
        assert connection.execute("SELECT count(*) FROM predictions WHERE prediction_set = 'run-a'").fetchone()[0] == total
        assert connection.execute("SELECT count(*) FROM predictions WHERE prediction_set = 'run-b'").fetchone()[0] == 1
        assert connection.execute(
            "SELECT split, count(*) FROM predictions WHERE prediction_set = 'run-a' GROUP BY split ORDER BY split"
        ).fetchall() == [("test", 1), ("train", 7), ("validation", 2)]
        assert connection.execute("SELECT prediction_set FROM prediction_sets ORDER BY prediction_set").fetchall() == [
            ("run-a",),
            ("run-b",),
        ]
        assert connection.execute(
            """
            SELECT artifact_id, dataset_id, dataset_fingerprint_sha256
            FROM prediction_sets WHERE prediction_set = 'run-a'
            """
        ).fetchone() == (
            "89d7a5e7-483a-46ad-a2cf-1edfb8057e3d",
            source_info["dataset_id"],
            source_fingerprint,
        )
        identity = connection.execute(
            """
            SELECT ps.prediction_set_id, ps.result_set_id, p.result_id,
                   p.logits_blob, p.input_sha256, p.output_sha256,
                   p.inference_result_json
            FROM prediction_sets ps
            JOIN predictions p USING (prediction_set)
            WHERE ps.prediction_set = 'run-a'
            LIMIT 1
            """
        ).fetchone()
        assert all(uuid.UUID(value) for value in identity[:3])
        assert identity[3] is not None
        assert len(identity[4]) == len(identity[5]) == 64
        assert json.loads(identity[6])["schema_name"] == (
            "oracle_builder.inference_result"
        )


def test_all_roi_predictions_include_rows_without_validated_masks(tmp_path):
    source = tmp_path / "segmentation.sqlite"
    output = tmp_path / "segmentation_with_predictions.sqlite"
    create_synthetic_segmentation(source, n=5, shape=(8, 8, 1))
    with sqlite3.connect(source) as connection:
        missing_uuid = connection.execute(
            "SELECT item_id FROM dataset_items ORDER BY rowid LIMIT 1"
        ).fetchone()[0]
        connection.execute(
            "UPDATE mask_annotations SET is_current = 0 WHERE item_id = ?",
            (missing_uuid,),
        )
    config = {
        "run": {"task": "segmentation", "model": "unet", "seed": 123},
        "data": {
            "input_shape": [8, 8, 1],
            "output_shape": [8, 8, 1],
            "validation_split": 0.2,
            "test_split": 0.1,
        },
    }

    x, targets, records = load_prediction_arrays(source, config)
    write_predictions_db(
        FakeSegmentationModel(),
        x,
        targets,
        records,
        config,
        output,
        source_sqlite=source,
        prediction_set="all-rois",
    )

    with sqlite3.connect(output) as connection:
        assert connection.execute("SELECT count(*) FROM predictions").fetchone()[0] == 5
        assert connection.execute(
            "SELECT y_true_blob, metrics_json FROM predictions WHERE uuid = ?",
            (missing_uuid,),
        ).fetchone() == (None, None)
        assert connection.execute(
            "SELECT split FROM predictions WHERE uuid = ?",
            (missing_uuid,),
        ).fetchone()[0] in {"train", "validation", "test"}


def test_classification_predictions_store_fixed_size_features(tmp_path):
    source = tmp_path / "training.sqlite"
    output = tmp_path / "predictions.sqlite"
    create_synthetic_classification(source, n=3, shape=(8, 8, 1), classes=2)
    config = {
        "run": {"task": "classification", "model": "simple_cnn", "seed": 123},
        "data": {
            "input_shape": [8, 8, 1],
            "num_classes": 2,
            "validation_split": 0.0,
            "test_split": 0.0,
        },
        "model": {
            "base_filters": 2,
            "dropout": 0.0,
            "embedding_dim": 10,
            "normalize_embeddings": True,
        },
    }
    x, y, records = load_arrays(source, config, split="train")
    model = get_model_builder("simple_cnn")(config)
    evidence_index = build_evidence_index(
        model,
        x,
        y,
        records,
        tmp_path / "classification_evidence.npz",
    )

    write_predictions_db(
        model,
        x,
        y,
        records,
        config,
        output,
        source_sqlite=source,
        prediction_set="features",
        evidence_index=evidence_index,
    )

    with sqlite3.connect(output) as connection:
        rows = connection.execute(
            """
            SELECT features_blob, features_encoding, features_dim
            FROM predictions
            ORDER BY uuid
            """
        ).fetchall()
    assert len(rows) == len(records)
    for blob, encoding, dimension in rows:
        feature = decode_blob(blob, encoding)
        assert dimension == 10
        assert feature.shape == (10,)
        norm = np.linalg.norm(feature)
        assert np.isclose(norm, 0.0, atol=1e-5) or np.isclose(norm, 1.0, atol=1e-5)
    with sqlite3.connect(output) as connection:
        packets = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT prediction_packet_json FROM predictions ORDER BY uuid"
            )
        ]
    assert all("softmax" in packet for packet in packets)
    assert all("prototype" in packet for packet in packets)
    assert all("knn" in packet for packet in packets)
