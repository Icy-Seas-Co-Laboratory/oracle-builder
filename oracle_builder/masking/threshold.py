from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def normalize_for_threshold(image: np.ndarray) -> np.ndarray:
    """Return a 2D numeric image suitable for thresholding."""
    array = np.asarray(image)
    if array.ndim == 3:
        channels = array.shape[-1]
        rgb = array[..., :3].astype("float32")
        if channels >= 3:
            array = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        else:
            array = np.mean(array, axis=-1)
    elif array.ndim != 2:
        raise ValueError(f"Thresholding expects a 2D image or 2D color image, got shape {array.shape}")

    values = array.astype("float32")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype="float32")
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return np.zeros(values.shape, dtype="float32")
    normalized = (values - low) / (high - low)
    return np.clip(normalized, 0, 1).astype("float32")


def invert_normalized_image(image: np.ndarray) -> np.ndarray:
    """Return a normalized inverted 2D image on a 0-1 scale."""
    return (1.0 - normalize_for_threshold(image)).astype("float32")


def invert_display_image(image: np.ndarray) -> np.ndarray:
    """Invert an image for display without changing its shape."""
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] == 4:
        inverted = invert_display_image(array[..., :3])
        return np.concatenate([inverted, array[..., 3:4]], axis=-1)
    if array.dtype.kind == "f":
        finite = array[np.isfinite(array)]
        if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
            return (1.0 - array).astype(array.dtype)
        normalized = normalize_for_threshold(array)
        return (1.0 - normalized).astype("float32")
    if array.dtype.kind in {"u", "i"}:
        max_value = np.iinfo(array.dtype).max if array.dtype.kind == "u" else int(np.nanmax(array))
        return (max_value - array).astype(array.dtype)
    return 1.0 - normalize_for_threshold(array)


def apply_blur(image: np.ndarray, blur_method: str = "none", kernel_size: int = 0) -> np.ndarray:
    """Apply optional blur before thresholding."""
    method = (blur_method or "none").lower()
    if method == "none" or kernel_size <= 0:
        return np.asarray(image)
    image_2d = normalize_for_threshold(image)
    pil_image = Image.fromarray((image_2d * 255).astype("uint8"))
    if method == "gaussian":
        radius = max(float(kernel_size) / 2.0, 0.1)
        blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
    elif method == "median":
        size = int(kernel_size)
        if size % 2 == 0:
            size += 1
        size = max(size, 3)
        blurred = pil_image.filter(ImageFilter.MedianFilter(size=size))
    else:
        raise ValueError("blur_method must be one of: none, gaussian, median")
    return np.asarray(blurred).astype("float32") / 255.0


def threshold_mask(
    image: np.ndarray,
    threshold: float,
    invert: bool = False,
    blur_method: str = "none",
    kernel_size: int = 0,
) -> np.ndarray:
    """Return a binary uint8 mask with values 0 and 1.

    If invert is true, invert image intensities before thresholding.
    """
    normalized = invert_normalized_image(image) if invert else normalize_for_threshold(image)
    if blur_method and blur_method != "none":
        normalized = apply_blur(normalized, blur_method, kernel_size)
    mask = normalized >= float(threshold)
    return mask.astype("uint8")
