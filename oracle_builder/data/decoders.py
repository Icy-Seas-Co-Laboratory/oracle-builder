from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
from PIL import Image


def decode_blob(blob: bytes | str | int | float | None, encoding: str | None, dimensions: str | None = None) -> Any:
    if blob is None:
        return None
    encoding = (encoding or "utf-8").lower()
    shape = json.loads(dimensions) if dimensions else None

    if encoding in {"utf-8", "text", "str"}:
        value = blob.decode("utf-8") if isinstance(blob, bytes) else str(blob)
        return value
    if encoding == "json":
        text = blob.decode("utf-8") if isinstance(blob, bytes) else str(blob)
        return json.loads(text)
    if encoding == "int":
        text = blob.decode("utf-8") if isinstance(blob, bytes) else str(blob)
        return int(text)
    if encoding == "float":
        text = blob.decode("utf-8") if isinstance(blob, bytes) else str(blob)
        return float(text)
    if encoding == "png":
        image = Image.open(io.BytesIO(blob))
        array = np.asarray(image)
        if shape:
            array = array.reshape(shape)
        return array
    if encoding in {"npy", "nparray"}:
        array = np.load(io.BytesIO(blob), allow_pickle=False)
        if shape:
            array = array.reshape(shape)
        return array
    if encoding == "zstd":
        raise ValueError(
            "Encoding 'zstd' is not available. Install zstandard or convert the dataset to a supported encoding."
        )
    raise ValueError(f"Unsupported blob encoding: {encoding}")


def encode_npy(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def normalize_input(array: Any, input_shape: list[int] | tuple[int, ...]) -> np.ndarray:
    value = np.asarray(array)
    value = value.reshape(input_shape)
    if value.dtype.kind in {"u", "i"} and value.max(initial=0) > 1:
        value = value.astype("float32") / 255.0
    return value.astype("float32")

