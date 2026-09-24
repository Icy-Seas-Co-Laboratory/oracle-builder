import json
import sqlite3

import tensorflow as tf

from oracle_builder.classification.stratified_training import (
    StratifiedChildResult,
    StratifiedTrainingResult,
    _train_interleaved_epoch,
    write_stratified_metric_artifacts,
)
from oracle_builder.classification.stratification import (
    batch_plan,
    summarize_records,
    supra_epoch_schedule,
    stratum_for_shape,
    training_stratum,
)


def _config():
    return {
        "run": {"seed": 123},
        "data": {"input_shape": [32, 32, 1], "batch_size": 128},
        "classification": {"stratification": {
            "dimensions": [32, 64, 128],
            "training_routing": {"enabled": True, "adjacent_lower_probability": 0.10, "seed": 5},
        }},
    }


def test_batch_plan_keeps_input_tensor_budget_constant():
    assert batch_plan(_config()) == {32: 128, 64: 32, 128: 8}


def test_training_routing_is_reproducible_and_never_moves_up():
    config = _config()
    values = [training_stratum(64, item_id=f"item-{index}", epoch=3, config=config) for index in range(100)]
    assert values == [training_stratum(64, item_id=f"item-{index}", epoch=3, config=config) for index in range(100)]
    assert set(values) <= {32, 64}
    assert training_stratum(32, item_id="item", epoch=1, config=config) == 32


def test_summary_reports_canonical_routed_and_class_counts():
    summary = summarize_records(
        [
            {"uuid": "a", "original_shape": [20, 34], "class_index": 1},
            {"uuid": "b", "original_shape": [90, 20], "class_index": 0},
        ],
        _config(), split="train", epoch=0,
    )
    assert summary["canonical_counts"] == {32: 0, 64: 1, 128: 1}
    assert sum(summary["routed_counts"].values()) == 2
    assert summary["batch_plan"] == {32: 128, 64: 32, 128: 8}


def test_supra_epoch_schedule_interleaves_every_shared_parent_epoch():
    config = _config()
    config["classification"]["stratification"]["supra_epochs"] = 2
    assert list(supra_epoch_schedule(config, 5)) == [
        (0, 1, 32), (0, 1, 64), (0, 1, 128),
        (1, 2, 32), (1, 2, 64), (1, 2, 128),
        (2, 3, 32), (2, 3, 64), (2, 3, 128),
        (3, 4, 32), (3, 4, 64), (3, 4, 128),
        (4, 5, 32), (4, 5, 64), (4, 5, 128),
    ]


def test_supra_epoch_schedule_keeps_contiguous_mode_for_reproduction():
    config = _config()
    config["classification"]["stratification"].update({"supra_epochs": 2, "schedule": "contiguous"})
    assert list(supra_epoch_schedule(config, 3)) == [
        (0, 2, 32), (0, 2, 64), (0, 2, 128),
        (2, 3, 32), (2, 3, 64), (2, 3, 128),
    ]


def test_interleaved_epoch_round_robins_resolution_batches():
    class FakeModel:
        def __init__(self):
            self.order = []

        def reset_metrics(self):
            pass

        def train_on_batch(self, features, targets, *, return_dict):
            del targets, return_dict
            dimension = int(features["stratum_dimension"][0, 0])
            self.order.append(dimension)
            return {"loss": float(dimension)}

    def dataset(dimension: int, batches: int):
        return tf.data.Dataset.from_tensor_slices((
            {
                "image": tf.ones((batches, 1, 1, 1)),
                "stratum_dimension": tf.fill((batches, 1), dimension),
            },
            tf.zeros((batches,), dtype=tf.int32),
        )).batch(1)

    model = FakeModel()
    loading_batches = []
    started_batches = []
    reported_batches = []
    metrics = _train_interleaved_epoch(
        model,
        {32: dataset(32, 2), 64: dataset(64, 3)},
        on_batch_loading=loading_batches.append,
        on_batch_begin=started_batches.append,
        on_batch_end=lambda batch, logs: reported_batches.append((batch, logs["loss"])),
    )

    assert model.order == [32, 64, 32, 64, 64]
    assert loading_batches == [0, 1, 2, 3, 4, 4, 5]
    assert started_batches == [0, 1, 2, 3, 4]
    assert reported_batches == [(0, 32.0), (1, 64.0), (2, 32.0), (3, 64.0), (4, 64.0)]
    assert metrics[32]["loss"] == 32.0
    assert metrics[64]["loss"] == 64.0


def test_largest_not_exceeding_policy_prefers_the_next_smaller_stratum():
    dimensions = [32, 64, 128]
    assert stratum_for_shape((20, 34), dimensions) == 64
    assert stratum_for_shape(
        (20, 34), dimensions, policy="largest_not_exceeding"
    ) == 32
    assert stratum_for_shape(
        (20, 500), dimensions, policy="largest_not_exceeding"
    ) == 128


def test_stratified_metric_artifacts_publish_json_csv_jsonl_and_sqlite_rows(tmp_path):
    history_path = tmp_path / "model" / "strata" / "32" / "training_history.json"
    history_path.parent.mkdir(parents=True)
    history_path.write_text(json.dumps({"loss": [0.8, 0.4], "accuracy": [0.5, 0.8]}))
    log_path = tmp_path / "logs" / "training.sqlite"
    log_path.parent.mkdir()
    with sqlite3.connect(log_path) as connection:
        connection.execute(
            "CREATE TABLE epoch_metrics (run_id TEXT, epoch INTEGER, split TEXT, metric TEXT, value REAL)"
        )
    child = StratifiedChildResult(
        dimension=32,
        batch_size=8,
        completed_epochs=2,
        stopped_early=False,
        model_path="model/shared/final.keras",
        summary_path="model/strata/32/model_summary.txt",
        history_path=history_path.relative_to(tmp_path).as_posix(),
        canonical_counts={"train": 2},
    )
    result = StratifiedTrainingResult(
        children={32: child},
        manifest_path=tmp_path / "model" / "stratification_manifest.json",
        recovery_path=tmp_path / "model" / "recovery" / "stratified_state.json",
        split_summaries={},
    )
    report = write_stratified_metric_artifacts(
        tmp_path,
        "run-1",
        {"classification": {"stratification": {"supra_epochs": 2}}},
        result,
    )
    assert report["metric_records"] == 4
    assert (tmp_path / "metrics" / "history.json").exists()
    assert (tmp_path / "metrics" / "history.csv").exists()
    records = [json.loads(line) for line in (tmp_path / "metrics" / "metrics.jsonl").read_text().splitlines()]
    assert records[0]["stratum_dimension"] == 32
    assert records[0]["epoch"] == 0
    with sqlite3.connect(log_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM epoch_metrics").fetchone()[0] == 4
