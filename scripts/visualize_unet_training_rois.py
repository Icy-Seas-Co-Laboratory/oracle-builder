#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oracle_builder.config import load_toml
from oracle_builder.data.splits import assign_missing_splits
from oracle_builder.masking.sqlite_io import load_sample, open_database


BLUE = np.array([40, 130, 255], dtype="float32")
GREEN = np.array([40, 220, 80], dtype="float32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a contact sheet of U-Net training ROIs with candidate and refined mask overlays."
    )
    parser.add_argument("--database", required=True, type=Path, help="oracle-builder SQLite dataset.")
    parser.add_argument("--output", required=True, type=Path, help="Output PNG path.")
    parser.add_argument("--config", type=Path, help="Optional TOML config used for seed and split fractions.")
    parser.add_argument("--split", default="train", help="Dataset split to visualize. Defaults to train.")
    parser.add_argument("--thumbnail-size", type=int, default=180, help="Maximum tile image size in pixels.")
    parser.add_argument("--columns", type=int, default=0, help="Grid columns. Defaults to sqrt(sample count).")
    parser.add_argument("--limit", type=int, help="Optional maximum number of ROIs to render.")
    parser.add_argument("--candidate-alpha", type=float, default=0.35, help="Transparent blue candidate mask alpha.")
    parser.add_argument("--refined-alpha", type=float, default=0.45, help="Transparent green refined mask alpha.")
    parser.add_argument("--seed", type=int, help="Override split assignment seed.")
    parser.add_argument("--validation-split", type=float, help="Override validation split fraction.")
    parser.add_argument("--test-split", type=float, help="Override test split fraction.")
    parser.add_argument("--no-labels", action="store_true", help="Hide UUID labels under each tile.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_toml(args.config) if args.config else {}
    data_config = config.get("data", {})
    run_config = config.get("run", {})
    seed = args.seed if args.seed is not None else int(run_config.get("seed", 123))
    validation_split = (
        args.validation_split if args.validation_split is not None else float(data_config.get("validation_split", 0.2))
    )
    test_split = args.test_split if args.test_split is not None else float(data_config.get("test_split", 0.1))

    rows = read_training_rows(args.database, validation_split, test_split, seed, args.split)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No samples with validated masks found for split={args.split!r}.")

    with open_database(args.database, create=False) as conn:
        samples = [load_sample(conn, row["uuid"]) for row in rows]

    sheet = build_contact_sheet(
        samples,
        thumbnail_size=args.thumbnail_size,
        columns=args.columns,
        candidate_alpha=args.candidate_alpha,
        refined_alpha=args.refined_alpha,
        show_labels=not args.no_labels,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"Wrote {len(samples)} {args.split} ROI overlays to {args.output}")
    return 0


def read_training_rows(
    database: Path,
    validation_split: float,
    test_split: float,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT uuid, split, metadata_json
            FROM samples
            WHERE output_blob IS NOT NULL AND length(output_blob) > 0
            ORDER BY uuid
            """
        ).fetchall()
    finally:
        connection.close()
    assigned = assign_missing_splits([dict(row) for row in rows], validation_split, test_split, seed)
    return [row for row in assigned if row.get("split") == split]


def build_contact_sheet(
    samples: list[dict[str, Any]],
    thumbnail_size: int,
    columns: int,
    candidate_alpha: float,
    refined_alpha: float,
    show_labels: bool,
) -> Image.Image:
    if columns <= 0:
        columns = max(1, math.ceil(math.sqrt(len(samples))))
    tiles = [
        make_overlay_tile(
            sample,
            thumbnail_size=thumbnail_size,
            candidate_alpha=candidate_alpha,
            refined_alpha=refined_alpha,
            show_label=show_labels,
        )
        for sample in samples
    ]
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile_w
        y = (index // columns) * tile_h
        sheet.paste(tile, (x, y))
    return sheet


def make_overlay_tile(
    sample: dict[str, Any],
    thumbnail_size: int,
    candidate_alpha: float,
    refined_alpha: float,
    show_label: bool,
) -> Image.Image:
    roi = normalize_roi(sample["image"])
    candidate = sample.get("candidate_mask")
    refined = sample.get("mask")
    overlay = apply_mask_overlay(roi, candidate, BLUE, candidate_alpha)
    overlay = apply_mask_overlay(overlay, refined, GREEN, refined_alpha)
    image = Image.fromarray(overlay.astype("uint8"), mode="RGB")
    image.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.BILINEAR)
    if not show_label:
        return image
    label_h = 28
    tile = Image.new("RGB", (thumbnail_size, thumbnail_size + label_h), "white")
    tile.paste(image, ((thumbnail_size - image.width) // 2, 0))
    draw = ImageDraw.Draw(tile)
    label = str(sample["uuid"])[:18]
    shape = "x".join(str(part) for part in np.asarray(sample["image"]).shape[:2])
    draw.text((4, thumbnail_size + 2), label, fill=(20, 20, 20), font=ImageFont.load_default())
    draw.text((4, thumbnail_size + 14), shape, fill=(80, 80, 80), font=ImageFont.load_default())
    return tile


def normalize_roi(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] >= 3:
        array = array[..., :3].astype("float32")
        gray = 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
    elif array.ndim == 3 and array.shape[-1] == 1:
        gray = array[..., 0].astype("float32")
    elif array.ndim == 2:
        gray = array.astype("float32")
    else:
        raise ValueError(f"Unsupported ROI shape: {array.shape}")
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        scaled = np.zeros_like(gray, dtype="float32")
    elif finite.max() <= 1.0 and finite.min() >= 0.0:
        scaled = gray * 255.0
    else:
        lo, hi = np.percentile(finite, [1, 99])
        if hi <= lo:
            scaled = np.zeros_like(gray, dtype="float32")
        else:
            scaled = (gray - lo) * (255.0 / (hi - lo))
    scaled = np.clip(scaled, 0, 255).astype("uint8")
    return np.repeat(scaled[..., None], 3, axis=-1)


def apply_mask_overlay(image: np.ndarray, mask: np.ndarray | None, color: np.ndarray, alpha: float) -> np.ndarray:
    if mask is None:
        return image
    result = np.asarray(image, dtype="float32").copy()
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3 and mask_array.shape[-1] == 1:
        mask_array = mask_array[..., 0]
    if mask_array.shape != result.shape[:2]:
        raise ValueError(f"Mask shape {mask_array.shape} does not match ROI shape {result.shape[:2]}")
    foreground = mask_array > 0
    result[foreground] = (1.0 - alpha) * result[foreground] + alpha * color
    return np.clip(result, 0, 255)


if __name__ == "__main__":
    raise SystemExit(main())
