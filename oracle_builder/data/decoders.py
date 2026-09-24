from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from oracle_builder.data.channels import derive_channels_numpy, resolve_channel_names


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
    channel_names = resolve_channel_names(settings)
    derived_channels = channel_names[1:]
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
        fit_pad_modes = {
            "fit_pad", "fit", "fit_pad_max_2x", "fit_pad_max_3x",
            "center_pad", "center_roi_pad",
        }
        if mode in fit_pad_modes:
            scale = min(target_w / source_w, target_h / source_h)
            # Preserve useful native detail in small ROIs.  Large ROIs still
            # downscale to fit, while small ones receive no more than 2x
            # enlargement and are centered on the padded canvas.
            maximum_upscale = {
                "fit_pad_max_2x": 2.0,
                "fit_pad_max_3x": 3.0,
                # Center an ROI on a canvas without inventing image detail.
                "center_pad": 1.0,
                "center_roi_pad": 1.0,
            }.get(mode)
            configured_limit = settings.get("upscale_limit")
            if configured_limit is not None:
                maximum_upscale = min(
                    maximum_upscale if maximum_upscale is not None else float(configured_limit),
                    float(configured_limit),
                )
            if maximum_upscale is not None:
                scale = min(scale, maximum_upscale)
        else:
            scale = max(target_w / source_w, target_h / source_h)
        resized = (
            max(1, int(round(source_w * scale))),
            max(1, int(round(source_h * scale))),
        )
        image = image.resize(resized, interpolation)
        if mode in {"fill_crop", "center_crop"}:
            anchor = "center" if mode == "center_crop" else settings.get("crop_anchor", "center")
            left, top = _anchored_offset(
                image.width, image.height, target_w, target_h, anchor
            )
            image = image.crop((left, top, left + target_w, top + target_h))
        elif mode in fit_pad_modes:
            anchor = "center" if mode in {"center_pad", "center_roi_pad"} else settings.get("pad_anchor", "center")
            image = _pad_image(image, target_w, target_h, settings, anchor=anchor)
    value = np.asarray(image)
    if value.ndim == 2:
        value = value[..., None]
    if value.shape[:2] != target[:2]:
        raise ValueError(
            f"Preprocessing mode {mode!r} produced {value.shape}; expected spatial shape {target[:2]}. "
            "Use fit_pad, center_pad, fill_crop, center_crop, or stretch for batched training."
        )
    value = _normalize_classification_values(value, settings)
    if bool(settings.get("invert", False)):
        value = 1.0 - value
    if derived_channels:
        value = derive_channels_numpy(value[..., :1], settings)
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
    return resolve_channel_names(settings)[1:]


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


def _anchored_offset(
    container_w: int, container_h: int, item_w: int, item_h: int, anchor: str
) -> tuple[int, int]:
    """Position an item in a container, or select an anchored crop origin."""
    anchor = str(anchor).lower().replace("-", "_")
    if anchor not in {
        "center", "top", "bottom", "left", "right",
        "top_left", "top_right", "bottom_left", "bottom_right",
    }:
        raise ValueError(f"Unsupported geometry anchor {anchor!r}")
    slack_x, slack_y = container_w - item_w, container_h - item_h
    if anchor in {"top_left", "left", "bottom_left"}:
        x = 0
    elif anchor in {"top_right", "right", "bottom_right"}:
        x = slack_x
    else:
        x = slack_x // 2
    if anchor in {"top_left", "top", "top_right"}:
        y = 0
    elif anchor in {"bottom_left", "bottom", "bottom_right"}:
        y = slack_y
    else:
        y = slack_y // 2
    return max(0, x), max(0, y)


def _pad_image(
    image: Image.Image, target_w: int, target_h: int, settings: dict[str, Any], *, anchor: str
) -> Image.Image:
    """Pad a resized ROI with constant, edge, reflect, or symmetric borders."""
    left, top = _anchored_offset(target_w, target_h, image.width, image.height, anchor)
    right, bottom = target_w - image.width - left, target_h - image.height - top
    pad_mode = str(settings.get("pad_mode", "constant")).lower()
    if pad_mode == "constant":
        fill = _pad_fill(image.mode, settings.get("pad_value", 0.0))
        canvas = Image.new(image.mode, (target_w, target_h), color=fill)
        canvas.paste(image, (left, top))
        return canvas
    values = np.asarray(image)
    pad_width = ((top, bottom), (left, right))
    if values.ndim == 3:
        pad_width += ((0, 0),)
    try:
        padded = np.pad(values, pad_width, mode=pad_mode)
    except ValueError as exc:
        raise ValueError(
            f"preprocessing.pad_mode={pad_mode!r} cannot pad ROI shape {values.shape}; "
            "use edge or constant for one-pixel dimensions"
        ) from exc
    return Image.fromarray(padded.astype("uint8"), mode=image.mode)


def _pad_fill(mode: str, value: Any):
    values = value if isinstance(value, (list, tuple)) else [value]
    expected = 1 if mode == "L" else len(mode)
    if len(values) not in {1, expected}:
        raise ValueError(f"pad_value must be one value or {expected} values for {mode} images")
    if len(values) == 1:
        values = values * expected
    converted = [int(round(min(max(float(item), 0.0), 1.0) * 255)) for item in values]
    return converted[0] if mode == "L" else tuple(converted)


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
