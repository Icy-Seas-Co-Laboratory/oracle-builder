from __future__ import annotations

import numpy as np

from oracle_builder.masking.validation import validate_mask


def test_empty_mask_is_invalid():
    report = validate_mask(np.zeros((8, 8), dtype="uint8"))
    assert not report["valid"]
    assert any("no foreground" in warning for warning in report["warnings"])


def test_full_mask_is_invalid_when_above_max_fraction():
    report = validate_mask(np.ones((8, 8), dtype="uint8"), max_foreground_fraction=0.95)
    assert not report["valid"]
    assert any("above" in warning for warning in report["warnings"])


def test_dimension_mismatch_is_detected():
    report = validate_mask(np.ones((8, 8), dtype="uint8"), image=np.zeros((9, 8), dtype="uint8"))
    assert not report["valid"]
    assert not report["dimension_matches_image"]


def test_valid_binary_mask_passes():
    mask = np.zeros((10, 10), dtype="uint8")
    mask[2:5, 2:5] = 1
    report = validate_mask(mask, image=np.zeros((10, 10, 3), dtype="uint8"))
    assert report["valid"]
    assert report["is_binary"]
    assert report["connected_component_count"] == 1


def test_nan_and_inf_are_detected():
    mask = np.zeros((4, 4), dtype="float32")
    mask[0, 0] = np.nan
    mask[1, 1] = np.inf
    report = validate_mask(mask)
    assert not report["valid"]
    assert report["has_nan"]
    assert report["has_inf"]
