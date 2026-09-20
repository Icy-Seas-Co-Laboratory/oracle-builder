from oracle_builder.classification.stratification import batch_plan, training_stratum


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
