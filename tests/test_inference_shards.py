from __future__ import annotations

import json
import sqlite3

import pytest

from oracle_builder.evaluation.predictions import init_predictions_db
from oracle_builder.inference.workflow import _item_ids_digest, merge_inference_shards


def _shard(root, shard_id: str, item_ids: list[str], *, output: str = "same"):
    root.mkdir()
    database = root / "predictions.sqlite"
    with init_predictions_db(database) as connection:
        connection.execute("INSERT INTO prediction_sets(prediction_set, created_at, config_json) VALUES ('demo', 'now', '{}')")
        for item_id in item_ids:
            connection.execute(
                "INSERT INTO predictions(prediction_set, uuid, output_sha256, inference_result_json, target_mode) VALUES ('demo', ?, ?, ?, 'validated_mask')",
                (item_id, output, json.dumps({"item_id": item_id, "output": output}, sort_keys=True)),
            )
        connection.commit()
    (root / "artifact.json").write_text(json.dumps({
        "model": {"artifact_id": "model", "fingerprint_sha256": "model-hash", "reference": {"id": "model"}},
        "input": {"reference": {"id": "dataset"}},
        "parameters": {"split": "all", "prediction_set": "demo"},
    }))
    (root / "inference_shard.json").write_text(json.dumps({
        "schema": {"name": "oracle_builder_inference_shard", "version": 1},
        "shard_id": shard_id, "item_ids": sorted(item_ids), "item_ids_sha256": _item_ids_digest(item_ids),
    }))


def test_merge_inference_shards_requires_exact_coverage_and_stable_rows(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    _shard(first, "one", ["b", "a"]); _shard(second, "two", ["c"])

    manifest = merge_inference_shards([second, first], tmp_path / "merged", expected_item_ids=["c", "a", "b"])

    assert manifest["outputs"]["records"] == 3
    with sqlite3.connect(tmp_path / "merged" / "predictions.sqlite") as connection:
        assert [row[0] for row in connection.execute("SELECT uuid FROM predictions ORDER BY uuid")] == ["a", "b", "c"]


def test_merge_rejects_missing_and_conflicting_duplicate_outputs(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    _shard(first, "one", ["a"]); _shard(second, "two", ["a"], output="different")

    with pytest.raises(ValueError, match="conflicting duplicate"):
        merge_inference_shards([first, second], tmp_path / "conflict", expected_item_ids=["a"])
    with pytest.raises(ValueError, match="coverage mismatch"):
        merge_inference_shards([first], tmp_path / "missing", expected_item_ids=["a", "b"])
