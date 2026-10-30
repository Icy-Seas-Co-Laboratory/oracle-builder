from __future__ import annotations

import sqlite3
import threading

import numpy as np
import tensorflow as tf

from oracle_builder.classification.evidence import (
    IdentityEvidenceIndex,
    build_evidence_index_streaming,
)
from oracle_builder.data.sqlite_dataset import create_synthetic_classification, load_arrays
from oracle_builder.data.sqlite_stream import (
    SQLiteClassificationSource,
    build_all_classification_index,
    build_classification_index,
    make_streaming_classification_bundle,
)
from oracle_builder.evaluation.predictions import (
    write_classification_predictions_streaming,
)
from oracle_builder.datasets.schema import set_dataset_lifecycle
from oracle_builder.training.train import build_and_compile_model


def stream_config():
    return {
        "run": {"task": "classification", "model": "simple_cnn", "seed": 123},
        "data": {
            "input_shape": [12, 12, 1],
            "num_classes": 3,
            "batch_size": 2,
            "shuffle_buffer": 4,
            "validation_split": 0.2,
            "test_split": 0.1,
            "streaming": {
                "enabled": True,
                "reader_workers": 1,
                "prefetch_batches": 1,
                "deterministic": True,
                "sqlite_cache_kib": 1024,
            },
        },
        "preprocessing": {
            "resize_mode": "fit_pad",
            "normalization": "dtype",
            "rescale": True,
            "invert": False,
            "pad_value": 0.0,
            "interpolation": "bilinear",
            "channel_mode": "grayscale",
        },
        "model": {
            "base_filters": 2,
            "dropout": 0.0,
            "embedding_dim": 6,
            "normalize_embeddings": True,
        },
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"],
        },
        "augmentation": {"enabled": False, "repeats_per_epoch": 1},
        "output": {"prediction_commit_batches": 1},
        "evidence": {"enabled": True, "knn_k": 2},
    }


def test_stream_index_contains_no_image_blobs_and_matches_eager_values(tmp_path):
    database = tmp_path / "classification.sqlite"
    create_synthetic_classification(database, n=10, shape=(12, 12, 1), classes=3)
    config = stream_config()
    index = build_classification_index(
        database, config, "validation", labeled_only=True
    )

    assert index.refs
    assert not hasattr(index.refs[0], "input_blob")
    eager_x, eager_y, _ = load_arrays(database, config, split="validation")
    source = SQLiteClassificationSource(database, config)
    batches = list(source.training_dataset(index, shuffle=False, augment=False))
    streamed_x = np.concatenate([np.asarray(batch[0]) for batch in batches])
    streamed_y = np.concatenate([np.asarray(batch[1]) for batch in batches])

    np.testing.assert_allclose(streamed_x, eager_x)
    np.testing.assert_array_equal(streamed_y, eager_y)


def test_streaming_loader_does_not_require_dataset_split_column(tmp_path):
    database = tmp_path / "classification.sqlite"
    create_synthetic_classification(database, n=10, shape=(12, 12, 1), classes=3)
    config = stream_config()

    with sqlite3.connect(database) as connection:
        assert "split" not in {
            row[1]
            for row in connection.execute("PRAGMA table_info(dataset_items)")
        }


def test_streaming_source_bounds_connections_when_callback_threads_turn_over(tmp_path, monkeypatch):
    """``tf.py_function`` may use a fresh worker thread for each callback."""
    database = tmp_path / "classification.sqlite"
    create_synthetic_classification(database, n=4, shape=(12, 12, 1), classes=2)
    config = stream_config()
    config["data"]["streaming"]["reader_workers"] = 2
    item_id = sqlite3.connect(database).execute(
        "SELECT item_id FROM dataset_items LIMIT 1"
    ).fetchone()[0]
    original_connect = sqlite3.connect
    opened = 0

    def tracked_connect(*args, **kwargs):
        nonlocal opened
        opened += 1
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    source = SQLiteClassificationSource(database, config)
    # Separate short-lived threads model TensorFlow's callback worker turnover.
    errors = []

    def read() -> None:
        try:
            source.read_image(item_id)
        except Exception as error:  # pragma: no cover - assertion reports it
            errors.append(error)

    for _ in range(32):
        thread = threading.Thread(target=read)
        thread.start()
        thread.join()

    assert not errors
    assert opened <= 2
    source.close()


def test_tf_data_streaming_source_bounds_connections_across_callbacks(tmp_path, monkeypatch):
    """The production ``tf.py_function`` path cannot accumulate SQLite FDs."""
    database = tmp_path / "classification.sqlite"
    create_synthetic_classification(database, n=4, shape=(12, 12, 1), classes=2)
    config = stream_config()
    config["data"]["streaming"]["reader_workers"] = 2
    with sqlite3.connect(database) as connection:
        item_id = connection.execute("SELECT item_id FROM dataset_items LIMIT 1").fetchone()[0]
    original_connect = sqlite3.connect
    opened = 0

    def tracked_connect(*args, **kwargs):
        nonlocal opened
        opened += 1
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    source = SQLiteClassificationSource(database, config)
    dataset = tf.data.Dataset.from_tensor_slices(np.repeat(item_id, 128)).map(
        source._tf_read_image,
        num_parallel_calls=2,
        deterministic=True,
    ).prefetch(2)

    list(dataset)

    assert opened <= 2
    source.close()


def test_run_manifest_assignments_drive_stream_index(tmp_path):
    database = tmp_path / "classification.sqlite"
    create_synthetic_classification(database, n=6, shape=(12, 12, 1), classes=2)
    with sqlite3.connect(database) as connection:
        item_ids = [
            row[0]
            for row in connection.execute(
                "SELECT item_id FROM dataset_items ORDER BY item_id"
            )
        ]
    config = stream_config()
    config["_split_manifest"] = {
        "assignments": {
            item_id: ("validation" if index < 2 else "train")
            for index, item_id in enumerate(item_ids)
        }
    }

    validation = build_classification_index(
        database, config, "validation", labeled_only=True
    )

    assert [ref.item_id for ref in validation.refs] == item_ids[:2]


def test_frozen_splits_are_derived_without_mutating_database(tmp_path):
    database = tmp_path / "classification.sqlite"
    create_synthetic_classification(database, n=10, shape=(12, 12, 1), classes=3)
    with sqlite3.connect(database) as connection:
        set_dataset_lifecycle(connection, "frozen")
        connection.commit()

    bundle = make_streaming_classification_bundle(database, stream_config())

    assert bundle.counts == {"train": 7, "validation": 2, "test": 1}
    with sqlite3.connect(database) as connection:
        assert "split" not in {
            row[1]
            for row in connection.execute("PRAGMA table_info(dataset_items)")
        }


def test_streaming_bundle_uses_bounded_dataset_pipeline(tmp_path):
    database = tmp_path / "classification.sqlite"
    create_synthetic_classification(database, n=10, shape=(12, 12, 1), classes=3)

    bundle = make_streaming_classification_bundle(database, stream_config())

    assert bundle.counts == {"train": 7, "validation": 2, "test": 1}
    images, labels = next(iter(bundle.datasets["train"]))
    assert images.shape == (2, 12, 12, 1)
    assert labels.shape == (2,)


def test_pretraining_shuffle_changes_order_without_affecting_indexed_order(tmp_path):
    database = tmp_path / "classification.sqlite"
    create_synthetic_classification(database, n=10, shape=(12, 12, 1), classes=3)
    config = stream_config()
    index = build_all_classification_index(database, {**config, "_external_inference": True}, labeled_only=False)
    source = SQLiteClassificationSource(database, config)

    ordered = np.concatenate([np.asarray(batch[1]) for batch in source.indexed_image_dataset(index, batch_size=2)])
    shuffled = source.image_dataset(index, batch_size=2, shuffle=True)
    first = np.concatenate([np.asarray(batch)[:, 0, 0, 0] for batch in shuffled])
    second = np.concatenate([np.asarray(batch)[:, 0, 0, 0] for batch in shuffled])

    assert np.array_equal(ordered, np.arange(len(index)))
    assert set(first) == set(second)
    assert not np.array_equal(first, second)


def test_streaming_evidence_and_predictions_are_disk_backed(tmp_path):
    database = tmp_path / "classification.sqlite"
    output = tmp_path / "predictions.sqlite"
    create_synthetic_classification(database, n=8, shape=(12, 12, 1), classes=3)
    config = stream_config()
    bundle = make_streaming_classification_bundle(database, config)
    model = build_and_compile_model(config)
    train_index = bundle.indices["train"]
    evidence_path = tmp_path / "classification_evidence"

    evidence = build_evidence_index_streaming(
        model,
        bundle.source.indexed_image_dataset(train_index),
        train_index,
        evidence_path,
    )
    loaded = IdentityEvidenceIndex.load(evidence_path)

    assert isinstance(loaded.embeddings, np.memmap)
    assert loaded.embeddings.shape == (len(train_index), 6)
    all_index = build_all_classification_index(
        database, config, labeled_only=False
    )
    written = write_classification_predictions_streaming(
        model,
        bundle.source.indexed_image_dataset(all_index),
        all_index,
        config,
        output,
        source_sqlite=database,
        prediction_set="stream",
        evidence_index=evidence,
    )
    assert written == 8
    with sqlite3.connect(output) as connection:
        assert connection.execute("SELECT count(*) FROM predictions").fetchone()[0] == 8
        assert connection.execute(
            "SELECT count(*) FROM predictions WHERE prediction_packet_json IS NOT NULL"
        ).fetchone()[0] == 8


def test_streaming_predictions_include_posthoc_head_and_neural_prediction(tmp_path):
    database = tmp_path / "classification.sqlite"
    output = tmp_path / "predictions.sqlite"
    create_synthetic_classification(database, n=5, shape=(12, 12, 1), classes=3)
    config = stream_config()
    source = SQLiteClassificationSource(database, config)
    index = build_all_classification_index(database, config, labeled_only=False)
    model = build_and_compile_model(config)

    class Head:
        classes_ = np.array([0, 1, 2])

        def predict_proba(self, values):
            return np.tile([[0.1, 0.2, 0.7]], (len(values), 1))

    written = write_classification_predictions_streaming(
        model, source.indexed_image_dataset(index, batch_size=2), index, config,
        output, source_sqlite=database, prediction_set="stream", posthoc_head=Head(),
        progress=False,
    )
    assert written == 5
    with sqlite3.connect(output) as connection:
        neural, result = connection.execute(
            "SELECT y_prob_json, inference_result_json FROM predictions LIMIT 1"
        ).fetchone()
    import json

    assert len(json.loads(neural)) == 3
    assert json.loads(result)["output"]["posthoc"]["class_index"] == 2
