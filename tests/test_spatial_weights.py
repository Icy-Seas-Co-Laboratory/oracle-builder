from __future__ import annotations

import numpy as np

from oracle_builder.training.spatial_weights import boundary_distance_weights


def test_boundary_pixels_receive_maximum_configured_weight():
    mask = np.zeros((9, 9), dtype="uint8")
    mask[2:7, 2:7] = 1

    weights = boundary_distance_weights(mask, weight_lambda=4.0, sigma=2.0)

    assert weights.shape == mask.shape
    assert weights.dtype == np.float32
    assert weights[2, 4] == 5.0
    assert weights[4, 4] < weights[2, 4]
    assert weights[0, 0] < weights[1, 2]
    assert np.all(weights >= 1.0)


def test_sigma_controls_how_far_edge_emphasis_extends():
    mask = np.zeros((11, 11), dtype="uint8")
    mask[3:8, 3:8] = 1

    narrow = boundary_distance_weights(mask, weight_lambda=2.0, sigma=0.5)
    broad = boundary_distance_weights(mask, weight_lambda=2.0, sigma=3.0)

    assert narrow[0, 0] < broad[0, 0]
    assert narrow[3, 5] == broad[3, 5] == 3.0


def test_empty_mask_has_neutral_weights():
    weights = boundary_distance_weights(np.zeros((5, 5)), weight_lambda=4.0, sigma=2.0)

    assert np.array_equal(weights, np.ones((5, 5), dtype="float32"))
