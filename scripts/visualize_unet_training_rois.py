#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from oracle_builder.artifacts import read_run_config
from oracle_builder.data.decoders import decode_blob
from oracle_builder.data.splits import assign_run_splits
from oracle_builder.data.tiling import plan_tiles
from oracle_builder.masking.sqlite_io import load_sample, open_database
from oracle_builder.datasets.legacy_roi import ensure_mask_refinement_database


BLUE = np.array([40, 130, 255], dtype="float32")
GREEN = np.array([40, 220, 80], dtype="float32")
ORANGE = np.array([255, 145, 35], dtype="float32")
PURPLE = np.array([170, 80, 230], dtype="float32")
CYAN = np.array([30, 210, 230], dtype="float32")
RED = np.array([240, 55, 55], dtype="float32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a contact sheet of U-Net training ROIs with candidate and refined mask overlays."
    )
    parser.add_argument("--database", required=True, type=Path, help="oracle-builder SQLite dataset.")
    parser.add_argument("--output", required=True, type=Path, help="Output PNG path.")
    parser.add_argument("--predictions", type=Path, help="Predictions SQLite database used by --side-by-side.")
    parser.add_argument("--prediction-set", help="Named prediction set to visualize when the database contains multiple sets.")
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Render candidate, model output, and validated overlays in three columns per ROI.",
    )
    parser.add_argument("--config", type=Path, help="Optional TOML config used for seed and split fractions.")
    parser.add_argument(
        "--run",
        type=Path,
        help="Model-run artifact whose protocol/splits.json defines ROI membership.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to visualize. Defaults to train.")
    parser.add_argument("--thumbnail-size", type=int, default=180, help="Maximum tile image size in pixels.")
    parser.add_argument("--columns", type=int, default=0, help="Grid columns. Defaults to sqrt(sample count).")
    parser.add_argument("--limit", type=int, help="Optional maximum number of ROIs to render.")
    parser.add_argument("--candidate-alpha", type=float, default=0.35, help="Transparent blue candidate mask alpha.")
    parser.add_argument("--refined-alpha", type=float, default=0.45, help="Transparent green refined mask alpha.")
    parser.add_argument("--prediction-alpha", type=float, default=0.45, help="Transparent orange model mask alpha.")
    parser.add_argument("--prediction-threshold", type=float, help="Override the saved model probability threshold.")
    parser.add_argument("--seed", type=int, help="Override split assignment seed.")
    parser.add_argument("--validation-split", type=float, help="Override validation split fraction.")
    parser.add_argument("--test-split", type=float, help="Override test split fraction.")
    parser.add_argument("--no-labels", action="store_true", help="Hide UUID labels under each tile.")
    parser.add_argument("--show-tile-grid", action="store_true", help="Draw configured tile boundaries over each ROI.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_mask_refinement_database(args.database)
    if args.run and args.config:
        raise SystemExit("Use either --run or --config, not both.")
    config = read_run_config(args.run) if args.run else (
        load_toml(args.config) if args.config else {}
    )
    data_config = config.get("data", {})
    run_config = config.get("run", {})
    seed = args.seed if args.seed is not None else int(run_config.get("seed", 123))
    validation_split = (
        args.validation_split if args.validation_split is not None else float(data_config.get("validation_split", 0.2))
    )
    test_split = args.test_split if args.test_split is not None else float(data_config.get("test_split", 0.1))
    tile_shape = tuple(data_config.get("input_shape", [])[:2]) or None
    tile_overlap = float(config.get("tiling", {}).get("overlap_fraction", 0.5))

    rows = read_training_rows(
        args.database,
        validation_split,
        test_split,
        seed,
        args.split,
        config=config,
    )
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No samples with validated masks found for split={args.split!r}.")

    with open_database(args.database, create=False) as conn:
        samples = [load_sample(conn, row["uuid"]) for row in rows]

    if args.side_by_side:
        if args.predictions is None:
            raise SystemExit("--side-by-side requires --predictions.")
        predictions = read_prediction_details(
            args.predictions,
            {sample["uuid"] for sample in samples},
            prediction_set=args.prediction_set,
        )
        missing = [sample["uuid"] for sample in samples if sample["uuid"] not in predictions]
        if missing:
            raise SystemExit(
                f"No model predictions found for {len(missing)} selected ROI(s), including {missing[0]!r}. "
                "Confirm that --run and --prediction-set refer to the same model run "
                "and that predictions were generated for all dataset items."
            )
        sheet = build_side_by_side_sheet(
            samples,
            predictions,
            thumbnail_size=args.thumbnail_size,
            candidate_alpha=args.candidate_alpha,
            prediction_alpha=args.prediction_alpha,
            refined_alpha=args.refined_alpha,
            prediction_threshold=args.prediction_threshold,
            show_labels=not args.no_labels,
            show_tile_grid=args.show_tile_grid,
            tile_shape=tile_shape,
            tile_overlap=tile_overlap,
        )
    else:
        sheet = build_contact_sheet(
            samples,
            thumbnail_size=args.thumbnail_size,
            columns=args.columns,
            candidate_alpha=args.candidate_alpha,
            refined_alpha=args.refined_alpha,
            show_labels=not args.no_labels,
            show_tile_grid=args.show_tile_grid,
            tile_shape=tile_shape,
            tile_overlap=tile_overlap,
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
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT di.item_id AS uuid, di.metadata_json
            FROM dataset_items di
            JOIN mask_refinement_items mi ON mi.item_id = di.item_id
            JOIN mask_annotations ma
              ON ma.item_id = mi.item_id AND ma.is_current = 1
            """
        ).fetchall()
    finally:
        connection.close()
    split_config = config or {
        "run": {"seed": seed},
        "data": {
            "validation_split": validation_split,
            "test_split": test_split,
        },
    }
    assigned = assign_run_splits([dict(row) for row in rows], split_config)
    return [row for row in assigned if row.get("split") == split]


def read_predictions(
    database: Path,
    uuids: set[str] | None = None,
    prediction_set: str | None = None,
) -> dict[str, np.ndarray]:
    details = read_prediction_details(database, uuids=uuids, prediction_set=prediction_set)
    return {uuid: detail["raw"] for uuid, detail in details.items()}


def read_prediction_details(
    database: Path,
    uuids: set[str] | None = None,
    prediction_set: str | None = None,
) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    saved_threshold = 0.5
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(predictions)")}
        if "prediction_set" in columns:
            available = [row[0] for row in connection.execute(
                "SELECT DISTINCT prediction_set FROM predictions ORDER BY prediction_set"
            )]
            if prediction_set is None:
                if len(available) != 1:
                    choices = ", ".join(repr(value) for value in available) or "none"
                    raise ValueError(f"Select --prediction-set; available sets: {choices}")
                prediction_set = available[0]
            if prediction_set not in available:
                raise ValueError(f"Prediction set {prediction_set!r} not found; available sets: {available}")
            config_row = connection.execute(
                "SELECT config_json FROM prediction_sets WHERE prediction_set = ?",
                (prediction_set,),
            ).fetchone()
            if config_row:
                try:
                    saved_config = json.loads(config_row[0])
                    saved_threshold = float(
                        saved_config.get("evaluation", {}).get("segmentation_threshold", 0.5)
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    saved_threshold = 0.5
            detail_columns = (
                "uuid, y_pred_blob, y_pred_encoding, target_mode, "
                "reconstructed_pred_blob, reconstructed_pred_encoding"
                if "target_mode" in columns
                else "uuid, y_pred_blob, y_pred_encoding, 'validated_mask' AS target_mode, NULL AS reconstructed_pred_blob, NULL AS reconstructed_pred_encoding"
            )
            rows = connection.execute(
                """
                SELECT """ + detail_columns + """
                FROM predictions
                WHERE prediction_set = ?
                ORDER BY uuid
                """,
                (prediction_set,),
            ).fetchall()
        else:
            if prediction_set is not None:
                raise ValueError("--prediction-set cannot be used with a legacy predictions database")
            rows = connection.execute(
                "SELECT uuid, y_pred_blob, y_pred_encoding, 'validated_mask' AS target_mode, NULL AS reconstructed_pred_blob, NULL AS reconstructed_pred_encoding FROM predictions ORDER BY uuid"
            ).fetchall()
    finally:
        connection.close()
    return {
        row["uuid"]: {
            "raw": np.asarray(decode_blob(row["y_pred_blob"], row["y_pred_encoding"])),
            "reconstructed": np.asarray(decode_blob(row["reconstructed_pred_blob"], row["reconstructed_pred_encoding"]))
            if row["reconstructed_pred_blob"] is not None else None,
            "target_mode": row["target_mode"] or "validated_mask",
            "probability_threshold": saved_threshold,
        }
        for row in rows
        if uuids is None or row["uuid"] in uuids
    }


def build_side_by_side_sheet(
    samples: list[dict[str, Any]],
    predictions: dict[str, Any],
    thumbnail_size: int,
    candidate_alpha: float,
    prediction_alpha: float,
    refined_alpha: float,
    prediction_threshold: float | None,
    show_labels: bool,
    show_tile_grid: bool = False,
    tile_shape: tuple[int, int] | None = None,
    tile_overlap: float = 0.5,
) -> Image.Image:
    delta_mode = any(
        isinstance(value, dict) and value.get("target_mode") == "candidate_delta"
        for value in predictions.values()
    )
    headers = (
        ("Candidate mask", "Predicted changes", "Reconstructed mask", "Validated mask")
        if delta_mode
        else ("Original mask", "Model output", "Validated mask")
    )
    colors = (BLUE, ORANGE, PURPLE, GREEN) if delta_mode else (BLUE, ORANGE, GREEN)
    header_h = 24
    label_h = 28 if show_labels else 0
    row_h = thumbnail_size + label_h
    sheet = Image.new("RGB", (thumbnail_size * len(headers), header_h + row_h * len(samples)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column, header in enumerate(headers):
        draw.text((column * thumbnail_size + 4, 6), header, fill=(20, 20, 20), font=font)
    for row_index, sample in enumerate(samples):
        roi = normalize_roi(sample["image"])
        if show_tile_grid and tile_shape:
            roi = apply_tile_grid(roi, tile_shape, tile_overlap)
        detail = predictions[sample["uuid"]]
        raw_prediction = detail["raw"] if isinstance(detail, dict) else detail
        threshold = (
            prediction_threshold
            if prediction_threshold is not None
            else float(detail.get("probability_threshold", 0.5))
            if isinstance(detail, dict)
            else 0.5
        )
        model_mask = prediction_to_roi_mask(raw_prediction, roi.shape[:2], threshold=threshold)
        if delta_mode:
            candidate = np.asarray(sample.get("candidate_mask")) > 0
            reconstructed = np.logical_xor(candidate, model_mask > 0).astype("uint8")
            masks = (sample.get("candidate_mask"), model_mask, reconstructed, sample.get("mask"))
            alphas = (candidate_alpha, prediction_alpha, prediction_alpha, refined_alpha)
        else:
            masks = (sample.get("candidate_mask"), model_mask, sample.get("mask"))
            alphas = (candidate_alpha, prediction_alpha, refined_alpha)
        for column, (mask, color, alpha) in enumerate(zip(masks, colors, alphas, strict=True)):
            if delta_mode and column == 1:
                panel = make_change_overlay_tile(
                    roi, sample.get("candidate_mask"), mask, alpha, thumbnail_size
                )
            else:
                panel = make_colored_overlay_tile(roi, mask, color, alpha, thumbnail_size)
            x = column * thumbnail_size + (thumbnail_size - panel.width) // 2
            y = header_h + row_index * row_h
            sheet.paste(panel, (x, y))
        if show_labels:
            label_y = header_h + row_index * row_h + thumbnail_size + 2
            draw.text((4, label_y), str(sample["uuid"])[:24], fill=(20, 20, 20), font=font)
            shape = "x".join(str(part) for part in np.asarray(sample["image"]).shape[:2])
            draw.text((4, label_y + 12), shape, fill=(80, 80, 80), font=font)
    return sheet


def prediction_to_roi_mask(prediction: np.ndarray, roi_shape: tuple[int, int], threshold: float = 0.5) -> np.ndarray:
    value = np.asarray(prediction)
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError(f"Unsupported prediction shape: {value.shape}")
    target_h, target_w = value.shape
    roi_h, roi_w = roi_shape
    scale = min(target_h / roi_h, target_w / roi_w)
    content_h = min(target_h, max(1, int(round(roi_h * scale))))
    content_w = min(target_w, max(1, int(round(roi_w * scale))))
    offset_y = (target_h - content_h) // 2
    offset_x = (target_w - content_w) // 2
    content = value[offset_y : offset_y + content_h, offset_x : offset_x + content_w]
    resized = Image.fromarray(content.astype("float32")).resize((roi_w, roi_h), Image.Resampling.BILINEAR)
    return (np.asarray(resized) >= threshold).astype("uint8")


def make_colored_overlay_tile(
    roi: np.ndarray, mask: np.ndarray | None, color: np.ndarray, alpha: float, thumbnail_size: int
) -> Image.Image:
    overlay = apply_mask_overlay(roi, mask, color, alpha)
    image = Image.fromarray(overlay.astype("uint8"), mode="RGB")
    image.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.BILINEAR)
    return image


def make_change_overlay_tile(
    roi: np.ndarray,
    candidate: np.ndarray,
    delta: np.ndarray,
    alpha: float,
    thumbnail_size: int,
) -> Image.Image:
    candidate_mask = np.asarray(candidate) > 0
    delta_mask = np.asarray(delta) > 0
    additions = np.logical_and(~candidate_mask, delta_mask)
    removals = np.logical_and(candidate_mask, delta_mask)
    overlay = apply_mask_overlay(roi, additions, CYAN, alpha)
    overlay = apply_mask_overlay(overlay, removals, RED, alpha)
    image = Image.fromarray(overlay.astype("uint8"), mode="RGB")
    image.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.BILINEAR)
    return image


def build_contact_sheet(
    samples: list[dict[str, Any]],
    thumbnail_size: int,
    columns: int,
    candidate_alpha: float,
    refined_alpha: float,
    show_labels: bool,
    show_tile_grid: bool = False,
    tile_shape: tuple[int, int] | None = None,
    tile_overlap: float = 0.5,
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
            show_tile_grid=show_tile_grid,
            tile_shape=tile_shape,
            tile_overlap=tile_overlap,
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
    show_tile_grid: bool = False,
    tile_shape: tuple[int, int] | None = None,
    tile_overlap: float = 0.5,
) -> Image.Image:
    roi = normalize_roi(sample["image"])
    if show_tile_grid and tile_shape:
        roi = apply_tile_grid(roi, tile_shape, tile_overlap)
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


def apply_tile_grid(
    roi: np.ndarray,
    tile_shape: tuple[int, int],
    overlap_fraction: float,
) -> np.ndarray:
    image = Image.fromarray(np.asarray(roi, dtype="uint8"), mode="RGB")
    draw = ImageDraw.Draw(image)
    for plan in plan_tiles(image.size[::-1], tile_shape, overlap_fraction):
        y, x = plan["origin"]
        crop_h, crop_w = plan["crop_shape"]
        draw.rectangle(
            (x, y, x + crop_w - 1, y + crop_h - 1),
            outline=(255, 220, 30),
            width=1,
        )
    return np.asarray(image)


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
