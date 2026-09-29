"""Portable local image loading and PNG encoding for ROI tooling."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_image_files(path: str | Path) -> list[Path]:
    image_dir = Path(path)
    if not image_dir.is_dir(): raise ValueError(f"Image folder does not exist: {image_dir}")
    return sorted(child for child in image_dir.iterdir() if child.is_file() and child.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)


def load_image(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    image_path = Path(path)
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES: raise ValueError(f"Unsupported image file type: {image_path.suffix}")
    with Image.open(image_path) as image:
        mode = image.mode; array = np.asarray(image) if mode in {"1", "L", "I;16", "I", "F", "RGB", "RGBA"} else np.asarray(image.convert("RGBA" if "A" in image.getbands() else "RGB"))
    return array, {"source_image_path": str(image_path), "source_image_name": image_path.name, "image_mode": mode, "image_shape": list(array.shape)}


def encode_image_png(image: np.ndarray) -> bytes:
    array = np.asarray(image)
    if array.dtype != np.uint8: array = (np.clip(array, 0, 1) * 255).astype("uint8") if array.dtype.kind == "f" else np.clip(array, 0, 255).astype("uint8")
    buffer = io.BytesIO(); Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()
