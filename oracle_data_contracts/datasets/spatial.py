from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


Box = dict[str, int]


def normalize_box(value: Any) -> Box | None:
    """Normalize common ROI rectangle encodings to integer x/y/width/height values."""
    if isinstance(value, Mapping):
        x = value.get("x", value.get("left"))
        y = value.get("y", value.get("top"))
        width = value.get("w", value.get("width"))
        height = value.get("h", value.get("height"))
        if width is None and x is not None and value.get("right") is not None:
            width = float(value["right"]) - float(x)
        if height is None and y is not None and value.get("bottom") is not None:
            height = float(value["bottom"]) - float(y)
        values = (x, y, width, height)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 4:
        values = value[:4]
    else:
        return None
    try:
        x, y, width, height = (int(round(float(part))) for part in values)
    except (TypeError, ValueError, OverflowError):
        return None
    if width < 1 or height < 1:
        return None
    return {"x": x, "y": y, "w": width, "h": height}


def box_is_within(inner: Mapping[str, int], outer: Mapping[str, int]) -> bool:
    return (
        inner["x"] >= outer["x"]
        and inner["y"] >= outer["y"]
        and inner["x"] + inner["w"] <= outer["x"] + outer["w"]
        and inner["y"] + inner["h"] <= outer["y"] + outer["h"]
    )


def normalize_item_geometry(
    metadata: Mapping[str, Any] | None = None,
    *,
    image_shape: Sequence[Any] | None = None,
    bbox: Any = None,
    crop_bbox: Any = None,
    coordinate_space: str | None = None,
) -> dict[str, Any] | None:
    """Resolve canonical ROI geometry while retaining how the fallback was obtained.

    Pelagia rectangles use ``[x, y, width, height]`` in source-frame pixels. Generic
    datasets may provide equivalent dictionaries or only an image shape. If just one
    compatible rectangle exists, it defines both the object bbox and stored crop.
    """
    root = dict(metadata or {})
    pelagia = root.get("pelagia") if isinstance(root.get("pelagia"), Mapping) else {}
    spatial = pelagia.get("spatial") if isinstance(pelagia.get("spatial"), Mapping) else {}
    if not spatial and isinstance(root.get("spatial"), Mapping):
        spatial = root["spatial"]
    geometry = root.get("geometry") if isinstance(root.get("geometry"), Mapping) else {}

    bbox_candidates = (
        bbox, spatial.get("bbox"), pelagia.get("bbox"), geometry.get("bbox"),
        root.get("bbox"), root.get("roi_bbox"),
    )
    crop_candidates = (
        crop_bbox, spatial.get("crop_bbox"), spatial.get("crop"),
        pelagia.get("crop_bbox"), pelagia.get("crop"), geometry.get("crop_bbox"),
        geometry.get("crop"), root.get("crop_bbox"), root.get("crop"),
        root.get("crop_bounds"), root.get("image_crop"),
    )
    resolved_bbox = next((box for value in bbox_candidates if (box := normalize_box(value))), None)
    resolved_crop = next((box for value in crop_candidates if (box := normalize_box(value))), None)
    fallback = None
    if resolved_bbox is None and resolved_crop is not None:
        resolved_bbox = dict(resolved_crop)
        fallback = "bbox_from_crop"
    elif resolved_crop is None and resolved_bbox is not None:
        resolved_crop = dict(resolved_bbox)
        fallback = "crop_from_bbox"
    elif resolved_bbox is None and resolved_crop is None and image_shape and len(image_shape) >= 2:
        try:
            height, width = int(image_shape[0]), int(image_shape[1])
        except (TypeError, ValueError, OverflowError):
            height = width = 0
        if width > 0 and height > 0:
            resolved_crop = {"x": 0, "y": 0, "w": width, "h": height}
            resolved_bbox = dict(resolved_crop)
            fallback = "bbox_and_crop_from_image_extent"
    if resolved_bbox is None or resolved_crop is None:
        return None

    resolved_space = (
        coordinate_space
        or spatial.get("coordinate_space")
        or pelagia.get("coordinate_space")
        or geometry.get("coordinate_space")
        or root.get("coordinate_space")
        or ("source_frame_pixels" if pelagia else "image_pixels")
    )
    return {
        "coordinate_space": str(resolved_space),
        "bbox": resolved_bbox,
        "crop_bbox": resolved_crop,
        "metadata": {
            "normalization": "oracle_data_contracts.item_geometry.v1",
            "fallback": fallback,
            "bbox_within_crop": box_is_within(resolved_bbox, resolved_crop),
        },
    }


def geometry_row_values(geometry: Mapping[str, Any]) -> tuple[Any, ...]:
    bbox = geometry["bbox"]
    crop = geometry["crop_bbox"]
    return (
        geometry["coordinate_space"],
        bbox["x"], bbox["y"], bbox["w"], bbox["h"],
        crop["x"], crop["y"], crop["w"], crop["h"],
    )
