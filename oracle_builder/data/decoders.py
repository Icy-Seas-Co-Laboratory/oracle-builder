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
    if encoding in {"png", "jpg", "jpeg", "tif", "tiff"}:
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


def prepare_classification_input(
    array: Any,
    input_shape: list[int] | tuple[int, ...],
    config: dict[str, Any],
) -> np.ndarray:
    target = tuple(int(value) for value in input_shape)
    if len(target) != 3:
        raise ValueError("Classification data.input_shape must be [height, width, channels]")
    settings = config.get("preprocessing", {})
    channel_mode = settings.get("channel_mode", "auto")
    if channel_mode == "auto":
        channel_mode = {1: "grayscale", 3: "rgb", 4: "rgba"}.get(target[-1])
        if channel_mode is None:
            raise ValueError(f"Cannot infer channel mode for {target[-1]} channels")
    image = _array_to_pil(np.asarray(array), channel_mode)
    mode = settings.get("resize_mode", "fit_pad")
    interpolation = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }[settings.get("interpolation", "bilinear")]
    target_h, target_w, _ = target
    if mode == "none":
        if image.size != (target_w, target_h):
            raise ValueError(
                f"Classification input shape {image.size[::-1]} does not match {target[:2]}"
            )
    elif mode == "stretch":
        image = image.resize((target_w, target_h), interpolation)
    else:
        source_w, source_h = image.size
        scale = (
            min(target_w / source_w, target_h / source_h)
            if mode in {"fit_pad", "fit"}
            else max(target_w / source_w, target_h / source_h)
        )
        resized = (
            max(1, int(round(source_w * scale))),
            max(1, int(round(source_h * scale))),
        )
        image = image.resize(resized, interpolation)
        if mode == "fill_crop":
            left = max(0, (image.width - target_w) // 2)
            top = max(0, (image.height - target_h) // 2)
            image = image.crop((left, top, left + target_w, top + target_h))
        elif mode == "fit_pad":
            pad_value = float(settings.get("pad_value", 0.0))
            fill = int(round(min(max(pad_value, 0.0), 1.0) * 255))
            canvas = Image.new(image.mode, (target_w, target_h), color=_pil_fill(image.mode, fill))
            canvas.paste(image, ((target_w - image.width) // 2, (target_h - image.height) // 2))
            image = canvas
    value = np.asarray(image)
    if value.ndim == 2:
        value = value[..., None]
    if value.shape != target:
        raise ValueError(
            f"Preprocessing mode {mode!r} produced {value.shape}; expected {target}. "
            "Use fit_pad, fill_crop, or stretch for batched training."
        )
    value = _normalize_classification_values(value, settings)
    if bool(settings.get("invert", False)):
        value = 1.0 - value
    return value.astype("float32")


def _array_to_pil(value: np.ndarray, channel_mode: str) -> Image.Image:
    if value.dtype.kind == "f":
        finite = value[np.isfinite(value)]
        if finite.size and finite.min() >= 0 and finite.max() <= 1:
            value = np.rint(value * 255).astype("uint8")
        else:
            low = float(finite.min()) if finite.size else 0.0
            high = float(finite.max()) if finite.size else 1.0
            value = np.clip((value - low) / max(high - low, 1e-12) * 255, 0, 255).astype(
                "uint8"
            )
    elif value.dtype != np.uint8:
        maximum = float(np.iinfo(value.dtype).max) if value.dtype.kind in {"u", "i"} else 255.0
        value = np.clip(value.astype("float32") / max(maximum, 1.0) * 255, 0, 255).astype(
            "uint8"
        )
    image = Image.fromarray(value.squeeze() if value.ndim == 3 and value.shape[-1] == 1 else value)
    return image.convert({"grayscale": "L", "rgb": "RGB", "rgba": "RGBA"}[channel_mode])


def _pil_fill(mode: str, value: int):
    if mode == "L":
        return value
    return tuple([value] * len(mode))


def _normalize_classification_values(value: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    value = value.astype("float32")
    if not bool(settings.get("rescale", True)):
        return value
    method = settings.get("normalization", "dtype")
    if method == "none":
        return value
    if method == "dtype":
        return value / 255.0
    if method == "minmax":
        low, high = float(value.min(initial=0)), float(value.max(initial=0))
    else:
        low = float(np.percentile(value, float(settings.get("percentile_low", 1.0))))
        high = float(np.percentile(value, float(settings.get("percentile_high", 99.0))))
    return np.clip((value - low) / max(high - low, 1e-12), 0.0, 1.0)
