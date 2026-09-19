#!/usr/bin/env python3
"""Render native-size grayscale, gradient, and local-contrast ROI channels."""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oracle_builder.data.decoders import decode_blob, prepare_classification_input


MARGIN = 12
HEADER_HEIGHT = 36
ROW_LABEL_HEIGHT = 22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path, help="Classification SQLite dataset.")
    parser.add_argument("--output", required=True, type=Path, help="Output PNG path.")
    parser.add_argument("--count", type=int, default=5, help="Number of random ROIs to show (default: 5).")
    parser.add_argument("--seed", type=int, default=123, help="Seed used to select ROIs.")
    parser.add_argument("--local-contrast-sigma", type=float, default=3.0)
    parser.add_argument("--invert", action="store_true", help="Invert grayscale before deriving channels.")
    return parser.parse_args()


def read_random_rois(database: Path, count: int, seed: int) -> list[dict[str, Any]]:
    """Read a reproducible random sample of original classification assets."""
    if count < 1:
        raise ValueError("--count must be positive")
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT di.item_id, di.source_key, di.metadata_json,
                   a.payload, a.encoding, a.shape_json, l.name AS class_name
            FROM dataset_items di
            JOIN classification_items ci ON ci.item_id = di.item_id
            JOIN assets a ON a.asset_id = ci.image_asset_id
            LEFT JOIN classification_annotations ca
              ON ca.item_id = di.item_id AND ca.is_current = 1
            LEFT JOIN classification_labels l ON l.label_id = ca.label_id
            ORDER BY di.item_id
            """
        ).fetchall()
    finally:
        connection.close()
    originals = [
        row
        for row in rows
        if (json.loads(row["metadata_json"]) if row["metadata_json"] else {}).get(
            "storage_mode", "original"
        )
        == "original"
    ]
    if not originals:
        raise ValueError(
            "No original image assets were found. This visualizer needs a database "
            "imported with --storage-mode original."
        )
    sample = random.Random(seed).sample(originals, k=min(count, len(originals)))
    return [dict(row) for row in sample]


def derived_channels(array: Any, *, sigma: float, invert: bool) -> np.ndarray:
    """Use the shared training/inference preprocessing implementation at native size."""
    value = np.asarray(array)
    if value.ndim < 2:
        raise ValueError(f"ROI must have at least two dimensions; received {value.shape}")
    height, width = value.shape[:2]
    return prepare_classification_input(
        value,
        [height, width, 3],
        {
            "preprocessing": {
                "resize_mode": "none",
                "normalization": "dtype",
                "rescale": True,
                "invert": invert,
                "interpolation": "nearest",
                "channel_mode": "grayscale",
                "derived_channels": {
                    "gradient_magnitude": True,
                    "local_contrast": True,
                    "local_contrast_sigma": sigma,
                },
            }
        },
    )


def _channel_image(value: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(value, 0.0, 1.0) * 255).astype("uint8"), "L")


def render_sheet(samples: list[dict[str, Any]], *, sigma: float, invert: bool) -> Image.Image:
    """Render three native-size channels per sampled ROI without resizing pixels."""
    prepared: list[tuple[dict[str, Any], np.ndarray]] = []
    for sample in samples:
        raw = decode_blob(sample["payload"], sample["encoding"], sample["shape_json"])
        prepared.append((sample, derived_channels(raw, sigma=sigma, invert=invert)))
    if not prepared:
        raise ValueError("No ROI samples available to render")

    column_width = max(value.shape[1] for _, value in prepared)
    row_heights = [value.shape[0] + ROW_LABEL_HEIGHT for _, value in prepared]
    width = MARGIN * 4 + column_width * 3
    height = HEADER_HEIGHT + MARGIN * (len(prepared) + 1) + sum(row_heights)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    headers = ("Grayscale ROI", "Gradient magnitude", "Local contrast")
    for column, header in enumerate(headers):
        x = MARGIN + column * (column_width + MARGIN)
        draw.text((x, 10), header, fill="black", font=font)

    y = HEADER_HEIGHT
    for sample, value in prepared:
        height_px, width_px = value.shape[:2]
        source = sample.get("source_key") or sample["item_id"]
        label = f"{source}  ·  {width_px} × {height_px}px"
        for column in range(3):
            x = MARGIN + column * (column_width + MARGIN)
            canvas.paste(_channel_image(value[..., column]), (x, y))
            draw.rectangle((x, y, x + width_px - 1, y + height_px - 1), outline="gray")
        draw.text((MARGIN, y + height_px + 4), label, fill="black", font=font)
        y += height_px + ROW_LABEL_HEIGHT + MARGIN
    return canvas


def main() -> int:
    args = parse_args()
    if args.local_contrast_sigma <= 0:
        raise SystemExit("--local-contrast-sigma must be positive")
    try:
        samples = read_random_rois(args.database, args.count, args.seed)
        sheet = render_sheet(
            samples,
            sigma=args.local_contrast_sigma,
            invert=args.invert,
        )
    except (OSError, ValueError, sqlite3.Error) as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"Wrote {len(samples)} ROI rows × 3 channels to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
