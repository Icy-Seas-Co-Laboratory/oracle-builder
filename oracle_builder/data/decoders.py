from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


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
    derived_channels = _derived_channel_names(settings)
    channel_mode = settings.get("channel_mode", "auto")
    if derived_channels:
        expected_channels = 1 + len(derived_channels)
        if target[-1] != expected_channels:
            raise ValueError(
                "Derived classification channels require data.input_shape with "
                f"{expected_channels} channels; received {target[-1]}"
            )
        channel_mode = "grayscale"
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
        if mode in {"fit_pad", "fit", "fit_pad_max_2x", "fit_pad_max_3x"}:
            scale = min(target_w / source_w, target_h / source_h)
            # Preserve useful native detail in small ROIs.  Large ROIs still
            # downscale to fit, while small ones receive no more than 2x
            # enlargement and are centered on the padded canvas.
            maximum_upscale = {
                "fit_pad_max_2x": 2.0,
                "fit_pad_max_3x": 3.0,
            }.get(mode)
            if maximum_upscale is not None:
                scale = min(scale, maximum_upscale)
        else:
            scale = max(target_w / source_w, target_h / source_h)
        resized = (
            max(1, int(round(source_w * scale))),
            max(1, int(round(source_h * scale))),
        )
        image = image.resize(resized, interpolation)
        if mode == "fill_crop":
            left = max(0, (image.width - target_w) // 2)
            top = max(0, (image.height - target_h) // 2)
            image = image.crop((left, top, left + target_w, top + target_h))
        elif mode in {"fit_pad", "fit_pad_max_2x", "fit_pad_max_3x"}:
            pad_value = float(settings.get("pad_value", 0.0))
            fill = int(round(min(max(pad_value, 0.0), 1.0) * 255))
            canvas = Image.new(image.mode, (target_w, target_h), color=_pil_fill(image.mode, fill))
            canvas.paste(image, ((target_w - image.width) // 2, (target_h - image.height) // 2))
            image = canvas
    value = np.asarray(image)
    if value.ndim == 2:
        value = value[..., None]
    if value.shape[:2] != target[:2]:
        raise ValueError(
            f"Preprocessing mode {mode!r} produced {value.shape}; expected spatial shape {target[:2]}. "
            "Use fit_pad, fit_pad_max_2x, fit_pad_max_3x, fill_crop, or stretch for batched training."
        )
    value = _normalize_classification_values(value, settings)
    if bool(settings.get("invert", False)):
        value = 1.0 - value
    if derived_channels:
        base = value[..., 0]
        values = [base]
        for name in derived_channels:
            if name == "gradient_magnitude":
                values.append(_gradient_magnitude(base))
            elif name == "local_contrast":
                values.append(
                    _local_contrast(
                        base,
                        sigma=float(
                            settings.get("derived_channels", {}).get(
                                "local_contrast_sigma", 3.0
                            )
                        ),
                    )
                )
        value = np.stack(values, axis=-1)
    if value.shape != target:
        raise ValueError(
            f"Preprocessing produced {value.shape}; expected {target}."
        )
    return value.astype("float32")


def prepare_dataset_classification_input(
    array: Any,
    input_shape: list[int] | tuple[int, ...],
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> np.ndarray:
    """Prepare a database image without reprocessing a materialized tensor.

    Folder imports in ``materialized`` mode persist the complete, final model
    tensor. Reapplying the normal preprocessing pipeline would blend its
    channels back into grayscale and regenerate different derived features.
    """
    target = tuple(int(value) for value in input_shape)
    if (metadata or {}).get("storage_mode") == "materialized":
        value = np.asarray(array, dtype="float32")
        if value.shape != target:
            raise ValueError(
                "Materialized classification input shape "
                f"{value.shape} does not match the configured model input {target}"
            )
        return value
    return prepare_classification_input(array, input_shape, config)


def _derived_channel_names(settings: dict[str, Any]) -> list[str]:
    """Return the stable input-channel order requested by preprocessing."""
    channels = settings.get("derived_channels", {})
    if not isinstance(channels, dict):
        raise ValueError("preprocessing.derived_channels must be a table/object")
    result = []
    if bool(channels.get("gradient_magnitude", False)):
        result.append("gradient_magnitude")
    if bool(channels.get("local_contrast", False)):
        result.append("local_contrast")
    return result


def _gradient_magnitude(value: np.ndarray) -> np.ndarray:
    dy, dx = np.gradient(np.asarray(value, dtype="float32"))
    magnitude = np.hypot(dx, dy)
    return _normalize_positive_feature(magnitude)


def _local_contrast(value: np.ndarray, *, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("preprocessing.derived_channels.local_contrast_sigma must be positive")
    image = Image.fromarray(np.rint(np.clip(value, 0.0, 1.0) * 255).astype("uint8"))
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=sigma)), dtype="float32") / 255.0
    contrast = np.asarray(value, dtype="float32") - blurred
    # Use magnitude rather than a signed, midpoint-encoded high-pass signal:
    # zero now means no local texture, including a padded background.
    return _normalize_positive_feature(np.abs(contrast))


def _normalize_positive_feature(value: np.ndarray) -> np.ndarray:
    scale = float(np.percentile(value, 99.0)) if value.size else 0.0
    if scale <= 1e-12:
        return np.zeros_like(value, dtype="float32")
    return np.clip(value / scale, 0.0, 1.0).astype("float32")


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
