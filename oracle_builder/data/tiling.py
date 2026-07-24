from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def tile_starts(length: int, tile_size: int, overlap_fraction: float) -> list[int]:
    if tile_size <= 0:
        raise ValueError("Tile dimensions must be positive")
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError("tiling.overlap_fraction must be in [0, 1)")
    if length <= tile_size:
        return [0]
    stride = max(1, int(round(tile_size * (1.0 - overlap_fraction))))
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def plan_tiles(
    source_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    overlap_fraction: float,
) -> list[dict[str, Any]]:
    source_h, source_w = source_shape
    tile_h, tile_w = tile_shape
    plans = []
    for y in tile_starts(source_h, tile_h, overlap_fraction):
        for x in tile_starts(source_w, tile_w, overlap_fraction):
            crop_h = min(tile_h, source_h - y)
            crop_w = min(tile_w, source_w - x)
            offset_y = (tile_h - crop_h) // 2 if source_h <= tile_h else 0
            offset_x = (tile_w - crop_w) // 2 if source_w <= tile_w else 0
            plans.append(
                {
                    "origin": [y, x],
                    "source_shape": [source_h, source_w],
                    "tile_shape": [tile_h, tile_w],
                    "crop_shape": [crop_h, crop_w],
                    "tile_offset": [offset_y, offset_x],
                }
            )
    return plans


def extract_tile(array: Any, plan: dict[str, Any], fill_value: float = 0.0) -> np.ndarray:
    value = np.asarray(array)
    y, x = plan["origin"]
    crop_h, crop_w = plan["crop_shape"]
    tile_h, tile_w = plan["tile_shape"]
    offset_y, offset_x = plan["tile_offset"]
    output_shape = (tile_h, tile_w, *value.shape[2:])
    output = np.full(output_shape, fill_value, dtype=value.dtype)
    output[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w] = value[
        y : y + crop_h, x : x + crop_w
    ]
    return output


def coverage_map(plans: list[dict[str, Any]]) -> np.ndarray:
    if not plans:
        raise ValueError("At least one tile plan is required")
    source_h, source_w = plans[0]["source_shape"]
    coverage = np.zeros((source_h, source_w), dtype="float32")
    for plan in plans:
        y, x = plan["origin"]
        crop_h, crop_w = plan["crop_shape"]
        coverage[y : y + crop_h, x : x + crop_w] += 1.0
    return coverage


def blend_window(tile_shape: tuple[int, int], mode: str = "hann") -> np.ndarray:
    tile_h, tile_w = tile_shape
    if mode == "uniform":
        return np.ones((tile_h, tile_w), dtype="float32")
    if mode != "hann":
        raise ValueError("tiling.blend_mode must be 'uniform' or 'hann'")
    wy = np.hanning(tile_h) if tile_h > 1 else np.ones(1)
    wx = np.hanning(tile_w) if tile_w > 1 else np.ones(1)
    return np.maximum(np.outer(wy, wx), 1e-3).astype("float32")


def reassemble_tiles(
    tiles: list[np.ndarray],
    plans: list[dict[str, Any]],
    blend_mode: str = "hann",
) -> np.ndarray:
    if not tiles or len(tiles) != len(plans):
        raise ValueError("Tiles and plans must be non-empty and have equal lengths")
    source_h, source_w = plans[0]["source_shape"]
    trailing_shape = np.asarray(tiles[0]).shape[2:]
    accumulator = np.zeros((source_h, source_w, *trailing_shape), dtype="float64")
    denominator = np.zeros((source_h, source_w), dtype="float64")
    window = blend_window(tuple(plans[0]["tile_shape"]), mode=blend_mode)
    for tile, plan in zip(tiles, plans, strict=True):
        y, x = plan["origin"]
        crop_h, crop_w = plan["crop_shape"]
        offset_y, offset_x = plan["tile_offset"]
        tile_crop = np.asarray(tile)[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w]
        weights = window[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w]
        if trailing_shape:
            accumulator[y : y + crop_h, x : x + crop_w] += tile_crop * weights[..., None]
        else:
            accumulator[y : y + crop_h, x : x + crop_w] += tile_crop * weights
        denominator[y : y + crop_h, x : x + crop_w] += weights
    if np.any(denominator == 0):
        raise ValueError("Tile plans left uncovered source pixels")
    if trailing_shape:
        result = accumulator / denominator[..., None]
    else:
        result = accumulator / denominator
    return result.astype("float32")


def group_and_reassemble(
    values: Any,
    records: list[dict[str, Any]],
    blend_mode: str = "hann",
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    groups: dict[str, list[tuple[np.ndarray, dict[str, Any]]]] = defaultdict(list)
    order: list[str] = []
    for value, record in zip(values, records, strict=False):
        source_uuid = record.get("source_uuid", record["uuid"])
        if source_uuid not in groups:
            order.append(source_uuid)
        groups[source_uuid].append((np.asarray(value), record))
    outputs = []
    output_records = []
    for source_uuid in order:
        items = groups[source_uuid]
        first_record = items[0][1]
        if first_record.get("tile_plan") is None:
            outputs.append(items[0][0])
        else:
            outputs.append(
                reassemble_tiles(
                    [item[0] for item in items],
                    [item[1]["tile_plan"] for item in items],
                    blend_mode=blend_mode,
                )
            )
        output_records.append(first_record.get("source_record", first_record))
    return outputs, output_records
