"""Thresholding utilities with only NumPy and Pillow dependencies."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def normalize_for_threshold(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        rgb = array[..., :3].astype("float32")
        array = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2] if array.shape[-1] >= 3 else np.mean(array, axis=-1)
    elif array.ndim != 2:
        raise ValueError(f"Thresholding expects a 2D image or 2D color image, got shape {array.shape}")
    values, finite = array.astype("float32"), array.astype("float32")[np.isfinite(array)]
    if not finite.size:
        return np.zeros(values.shape, dtype="float32")
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    return np.zeros(values.shape, dtype="float32") if high <= low else np.clip((values - low) / (high - low), 0, 1).astype("float32")


def invert_normalized_image(image: np.ndarray) -> np.ndarray:
    return (1.0 - normalize_for_threshold(image)).astype("float32")


def invert_display_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] == 4:
        return np.concatenate([invert_display_image(array[..., :3]), array[..., 3:4]], axis=-1)
    if array.dtype.kind == "f":
        finite = array[np.isfinite(array)]
        return (1.0 - array).astype(array.dtype) if finite.size and float(finite.min()) >= 0 and float(finite.max()) <= 1 else (1.0 - normalize_for_threshold(array)).astype("float32")
    if array.dtype.kind in {"u", "i"}:
        return ((np.iinfo(array.dtype).max if array.dtype.kind == "u" else int(np.nanmax(array))) - array).astype(array.dtype)
    return 1.0 - normalize_for_threshold(array)


def apply_blur(image: np.ndarray, blur_method: str = "none", kernel_size: int = 0) -> np.ndarray:
    method = (blur_method or "none").lower()
    if method == "none" or kernel_size <= 0:
        return np.asarray(image)
    pil_image = Image.fromarray((normalize_for_threshold(image) * 255).astype("uint8"))
    if method == "gaussian":
        blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=max(float(kernel_size) / 2, 0.1)))
    elif method == "median":
        size = int(kernel_size); size = max(size + (size % 2 == 0), 3)
        blurred = pil_image.filter(ImageFilter.MedianFilter(size=size))
    else:
        raise ValueError("blur_method must be one of: none, gaussian, median")
    return np.asarray(blurred).astype("float32") / 255.0


def threshold_mask(image: np.ndarray, threshold: float, invert: bool = False, blur_method: str = "none", kernel_size: int = 0) -> np.ndarray:
    normalized = invert_normalized_image(image) if invert else normalize_for_threshold(image)
    return (apply_blur(normalized, blur_method, kernel_size) >= float(threshold)).astype("uint8") if blur_method and blur_method != "none" else (normalized >= float(threshold)).astype("uint8")
