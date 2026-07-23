from __future__ import annotations

import numpy as np

from oracle_builder.training.metrics import BinaryDice


def test_binary_dice_accumulates_confusion_counts_across_batches():
    metric = BinaryDice(threshold=0.5)
    metric.update_state(
        np.array([[[[1], [1], [0], [0]]]], dtype="float32"),
        np.array([[[[0.9], [0.2], [0.8], [0.1]]]], dtype="float32"),
    )

    assert np.isclose(float(metric.result()), 0.5)


def test_binary_dice_returns_one_for_two_empty_masks():
    metric = BinaryDice()
    metric.update_state(np.zeros((1, 2, 2, 1)), np.zeros((1, 2, 2, 1)))

    assert float(metric.result()) == 1.0
