from oracle_builder.classification.stratification import batch_plan, summarize_records, training_stratum


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
