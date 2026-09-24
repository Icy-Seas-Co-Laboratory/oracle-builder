"""Deterministic, config-driven image channel generation.

The functions in this module deliberately have NumPy and TensorFlow entry
points.  The former is used by import/inference preprocessing, while the
latter permits derived features to be made *after* random augmentation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageFilter


# Stable ordering keeps old configs and exported models reproducible.
CHANNEL_ORDER = (
    "intensity", "gradient_magnitude", "gx", "gy", "gaussian_blur",
    "laplacian", "local_contrast", "foreground_mask", "foreground_distance",
)
_ALIASES = {"log": "laplacian", "laplacian_of_gaussian": "laplacian", "distance": "foreground_distance"}


def resolve_channel_names(settings: dict[str, Any]) -> list[str]:
    """Return requested channels, accepting old boolean and new list schemas."""
    channels = settings.get("derived_channels", {})
    explicit = settings.get("channels")
    if explicit is None and isinstance(channels, dict):
        explicit = channels.get("channels")
    if explicit is not None:
        if not isinstance(explicit, (list, tuple)):
            raise ValueError("preprocessing.channels must be a list")
        names = [str(_ALIASES.get(str(name), str(name))) for name in explicit]
    elif isinstance(channels, (list, tuple)):
        names = [str(_ALIASES.get(str(name), str(name))) for name in channels]
    elif isinstance(channels, dict):
        names = ["intensity"] + [name for name in CHANNEL_ORDER[1:] if bool(channels.get(name, False))]
    else:
        raise ValueError("preprocessing.derived_channels must be a table/object or list")
    if "intensity" not in names:
        names.insert(0, "intensity")
    invalid = [name for name in names if name not in CHANNEL_ORDER]
    if invalid:
        raise ValueError(f"Unknown preprocessing channel(s): {', '.join(invalid)}")
    if len(set(names)) != len(names):
        raise ValueError("preprocessing channels may not contain duplicates")
    return names


def has_derived_channels(settings: dict[str, Any]) -> bool:
    return len(resolve_channel_names(settings)) > 1


def _options(settings: dict[str, Any]) -> dict[str, Any]:
    value = settings.get("derived_channels", {})
    return value if isinstance(value, dict) else {}


def derive_channels_numpy(
    intensity: np.ndarray, settings: dict[str, Any], *, foreground_mask: np.ndarray | None = None
) -> np.ndarray:
    """Stack configured channels from normalized intensity (H, W or H, W, 1)."""
    image = np.asarray(intensity, dtype="float32")
    if image.ndim == 3:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError("Channel derivation requires a single intensity image")
    opts = _options(settings)
    sigma = float(opts.get("sigma", opts.get("gaussian_sigma", opts.get("local_contrast_sigma", 3.0))))
    if sigma <= 0:
        raise ValueError("derived channel sigma must be positive")
    names = resolve_channel_names(settings)
    gy, gx = np.gradient(image)
    blur = _blur_numpy(image, sigma)
    mask = _foreground_mask_numpy(image, opts, foreground_mask)
    values: dict[str, np.ndarray] = {
        "intensity": image, "gx": _signed_feature(gx), "gy": _signed_feature(gy),
        "gradient_magnitude": _positive_feature(np.hypot(gx, gy)), "gaussian_blur": blur,
        "laplacian": _signed_feature(_laplacian_numpy(image, sigma if bool(opts.get("laplacian_of_gaussian", False)) else 0.0)),
        "local_contrast": _positive_feature(np.abs(image - blur)), "foreground_mask": mask,
        "foreground_distance": _foreground_distance_numpy(mask, opts),
    }
    return np.stack([values[name] for name in names], axis=-1).astype("float32")


def _blur_numpy(image: np.ndarray, sigma: float) -> np.ndarray:
    return np.asarray(Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(radius=sigma)), dtype="float32") / 255.0


def _laplacian_numpy(image: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy import ndimage
        return ndimage.laplace(ndimage.gaussian_filter(image, sigma) if sigma else image).astype("float32")
    except ImportError:  # dependency-light fallback
        padded = np.pad(image, 1, mode="edge")
        return (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * image)


def _foreground_mask_numpy(image: np.ndarray, opts: dict[str, Any], provided: np.ndarray | None) -> np.ndarray:
    if provided is not None:
        mask = np.asarray(provided, dtype="float32").squeeze()
        if mask.shape != image.shape: raise ValueError("foreground mask must match intensity spatial shape")
        return (mask > 0.5).astype("float32")
    threshold = float(opts.get("foreground_threshold", opts.get("mask_threshold", 0.0)))
    return (image > threshold).astype("float32")


def _foreground_distance_numpy(mask: np.ndarray, opts: dict[str, Any]) -> np.ndarray:
    try:
        from scipy.ndimage import distance_transform_edt
        distance = distance_transform_edt(mask)
    except ImportError:
        distance = mask
    scale = float(opts.get("foreground_distance_scale", np.max(distance) or 1.0))
    return np.clip(distance / max(scale, 1e-12), 0, 1).astype("float32")


def _positive_feature(value: np.ndarray) -> np.ndarray:
    scale = float(np.percentile(np.abs(value), 99.0)) if value.size else 0.0
    return np.zeros_like(value, dtype="float32") if scale <= 1e-12 else np.clip(np.abs(value) / scale, 0, 1).astype("float32")


def _signed_feature(value: np.ndarray) -> np.ndarray:
    scale = float(np.percentile(np.abs(value), 99.0)) if value.size else 0.0
    return np.full_like(value, 0.5, dtype="float32") if scale <= 1e-12 else np.clip(0.5 + value / (2 * scale), 0, 1).astype("float32")


def derive_channels_tensor(intensity, settings: dict[str, Any], *, foreground_mask=None):
    """TensorFlow channel derivation for a [B,H,W,C] tensor after augmentation."""
    import tensorflow as tf
    names = resolve_channel_names(settings); opts = _options(settings)
    x = tf.cast(intensity[..., :1], tf.float32)
    sigma = float(opts.get("sigma", opts.get("gaussian_sigma", opts.get("local_contrast_sigma", 3.0))))
    if sigma <= 0: raise ValueError("derived channel sigma must be positive")
    blur = _gaussian_blur_tensor(x, sigma)
    # Sobel output ordering is dy, dx.
    sobel = tf.image.sobel_edges(x); gy, gx = sobel[..., 0], sobel[..., 1]
    magnitude = tf.sqrt(tf.square(gx) + tf.square(gy) + 1e-12)
    laplacian = tf.nn.depthwise_conv2d(x, tf.constant([[[[0.]], [[1.]], [[0.]]], [[[1.]], [[-4.]], [[1.]]], [[[0.]], [[1.]], [[0.]]]]), [1, 1, 1, 1], "SAME")
    mask = tf.cast((foreground_mask if foreground_mask is not None else x) > float(opts.get("foreground_threshold", opts.get("mask_threshold", 0.0))), tf.float32)
    values = {"intensity": x, "gx": _signed_tensor(gx), "gy": _signed_tensor(gy),
              "gradient_magnitude": _positive_tensor(magnitude), "gaussian_blur": blur,
              "laplacian": _signed_tensor(laplacian), "local_contrast": _positive_tensor(tf.abs(x - blur)),
              "foreground_mask": mask}
    if "foreground_distance" in names:
        distance = tf.numpy_function(lambda m: np.stack([_foreground_distance_numpy(v[..., 0], opts) for v in m], axis=0)[..., None], [mask], tf.float32)
        distance.set_shape(mask.shape); values["foreground_distance"] = distance
    return tf.concat([values[name] for name in names], axis=-1)


def _gaussian_blur_tensor(x, sigma: float):
    import tensorflow as tf
    radius = max(1, int(np.ceil(3 * sigma))); axis = tf.range(-radius, radius + 1, dtype=tf.float32)
    kernel = tf.exp(-(axis ** 2) / (2 * sigma ** 2)); kernel /= tf.reduce_sum(kernel)
    kernel2d = tf.einsum("i,j->ij", kernel, kernel)[:, :, None, None]
    return tf.nn.depthwise_conv2d(x, kernel2d, [1, 1, 1, 1], "SAME")


def _positive_tensor(x):
    import tensorflow as tf
    # Per-example max is graph-friendly and avoids an optional percentile op.
    scale = tf.reduce_max(tf.abs(x), axis=[1, 2, 3], keepdims=True)
    return tf.where(scale > 1e-12, tf.clip_by_value(tf.abs(x) / tf.maximum(scale, 1e-12), 0.0, 1.0), tf.zeros_like(x))


def _signed_tensor(x):
    import tensorflow as tf
    scale = tf.reduce_max(tf.abs(x), axis=[1, 2, 3], keepdims=True)
    return tf.where(scale > 1e-12, tf.clip_by_value(0.5 + x / (2.0 * tf.maximum(scale, 1e-12)), 0.0, 1.0), tf.fill(tf.shape(x), 0.5))
