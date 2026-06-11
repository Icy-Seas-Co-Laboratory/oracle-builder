from __future__ import annotations

import numpy as np

from oracle_builder.masking.threshold import invert_display_image, invert_normalized_image, normalize_for_threshold, threshold_mask


def test_threshold_creates_binary_mask():
    image = np.array([[0.0, 0.2], [0.8, 1.0]], dtype="float32")
    mask = threshold_mask(image, 0.5)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) == {0, 1}
    assert mask.shape == (2, 2)


def test_threshold_invert_inverts_image_not_mask():
    image = np.array([[0.0, 0.25, 0.75, 1.0]], dtype="float32")
    normal = threshold_mask(image, 0.5)
    inverted = threshold_mask(image, 0.5, invert=True)
    expected = (invert_normalized_image(image) >= 0.5).astype("uint8")
    assert np.array_equal(inverted, expected)


def test_threshold_invert_differs_from_mask_inversion_when_threshold_is_not_midpoint():
    image = np.array([[0.0, 0.4, 0.6, 1.0]], dtype="float32")
    normal = threshold_mask(image, 0.25)
    inverted = threshold_mask(image, 0.25, invert=True)
    assert not np.array_equal(inverted, 1 - normal)


def test_invert_display_image_preserves_rgba_alpha():
    image = np.zeros((2, 2, 4), dtype="uint8")
    image[..., :3] = 10
    image[..., 3] = 123
    inverted = invert_display_image(image)
    assert np.all(inverted[..., :3] == 245)
    assert np.all(inverted[..., 3] == 123)


def test_rgb_images_can_be_thresholded_to_2d():
    image = np.zeros((4, 5, 3), dtype="uint8")
    image[:, 3:, :] = 255
    normalized = normalize_for_threshold(image)
    mask = threshold_mask(image, 0.5)
    assert normalized.shape == (4, 5)
    assert mask.shape == (4, 5)
