import numpy as np
import pytest

from oracle_builder.data.channels import derive_channels_numpy, resolve_channel_names


def test_registry_has_stable_explicit_channel_order():
    settings = {"channels": ["intensity", "gx", "foreground_mask", "foreground_distance"]}
    assert resolve_channel_names(settings) == settings["channels"]
    image = np.zeros((9, 9), dtype="float32")
    image[2:7, 2:7] = 1
    output = derive_channels_numpy(image, settings)
    assert output.shape == (9, 9, 4)
    assert output[4, 4, 2] == 1
    assert output[4, 4, 3] > output[2, 2, 3]


def test_registry_supports_new_derivatives_and_provided_foreground_mask():
    image = np.zeros((11, 11), dtype="float32")
    image[:, 5:] = 1
    mask = np.zeros_like(image); mask[3:8, 3:8] = 1
    settings = {"derived_channels": {"gx": True, "gy": True, "gaussian_blur": True, "laplacian": True, "foreground_mask": True, "foreground_distance": True}}
    output = derive_channels_numpy(image, settings, foreground_mask=mask)
    assert output.shape[-1] == 7
    assert np.array_equal(output[..., 5], mask)
    assert np.all((output >= 0) & (output <= 1))


def test_tensor_derivation_operates_after_augmentation_and_matches_contract():
    tf = pytest.importorskip("tensorflow")
    from oracle_builder.data.channels import derive_channels_tensor
    settings = {"channels": ["intensity", "gradient_magnitude", "gx", "gy", "local_contrast"]}
    image = tf.constant(np.pad(np.ones((1, 4, 4, 1), np.float32), ((0, 0), (2, 2), (2, 2), (0, 0))))
    output = derive_channels_tensor(image, settings).numpy()
    assert output.shape == (1, 8, 8, 5)
    assert output[..., 1].max() > 0
