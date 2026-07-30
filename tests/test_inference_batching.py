from __future__ import annotations

import pytest

tf = pytest.importorskip("tensorflow")

from oracle_builder.inference.batching import (
    estimate_inference_batch_size,
    resolve_inference_batch_size,
)


def config(batch_size="auto"):
    return {
        "run": {"task": "classification", "model": "simple_cnn"},
        "data": {"input_shape": [16, 16, 1]},
        "inference": {
            "batch_size": batch_size,
            "minimum_batch_size": 1,
            "maximum_batch_size": 16,
            "memory_budget_mb": 64,
        },
    }


class LimitedModel:
    def __init__(self, maximum: int):
        self.maximum = maximum

    def __call__(self, values, training=False):
        if int(values.shape[0]) > self.maximum:
            raise tf.errors.ResourceExhaustedError(
                None, None, "synthetic OOM"
            )
        return values


def test_auto_batch_is_bounded_and_halves_after_oom():
    plan = resolve_inference_batch_size(
        LimitedModel(4), config(), announce=None
    )

    assert estimate_inference_batch_size(config()) == 16
    assert plan.initial_candidate == 16
    assert plan.attempts == (16, 8, 4)
    assert plan.batch_size == 4
    assert plan.mode == "auto"


def test_fixed_batch_is_still_verified():
    plan = resolve_inference_batch_size(
        LimitedModel(8), config(6), announce=None
    )

    assert plan.batch_size == 6
    assert plan.mode == "fixed"
