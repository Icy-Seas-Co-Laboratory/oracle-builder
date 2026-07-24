#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from oracle_builder.masking.image_io import SUPPORTED_IMAGE_SUFFIXES, list_image_files, load_image
from oracle_builder.masking.api_io import (
    detection_id_from_summary,
    list_pelagia_detections,
    login_pelagia,
    load_api_roi,
    load_pelagia_detection,
)
from oracle_builder.masking.sqlite_io import (
    create_or_update_image_sample,
    list_samples,
    load_sample,
    open_database,
)
from oracle_builder.masking.unet_dataset import validate_unet_dataset, write_unet_config_from_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and save U-Net-ready masks for oracle-builder datasets.")
    parser.add_argument("--database", type=Path, help="SQLite dataset to read from and write to.")
    parser.add_argument("--input", type=Path, help="Legacy alias for --database. Image paths are accepted as a convenience.")
    parser.add_argument("--image", type=Path, help="Local image file to load.")
    parser.add_argument("--output", type=Path, help="Legacy alias for --database.")
    parser.add_argument("--api-base-url", default="http://localhost:8000", help="Base URL for ROI/mask API loading.")
    parser.add_argument("--api-token", help="Pelagia bearer token. Defaults to PELAGIA_API_TOKEN.")
    parser.add_argument("--api-username", help="Pelagia username for /auth/login. Defaults to PELAGIA_USERNAME.")
    parser.add_argument("--api-password", help="Pelagia password for /auth/login. Defaults to PELAGIA_PASSWORD.")
    parser.add_argument("--api-project-key", help="Pelagia project key for /auth/login. Defaults to PELAGIA_PROJECT_KEY or default.")
    parser.add_argument("--list-api-rois", action="store_true", help="List Pelagia detection/ROI ids and exit.")
    parser.add_argument("--api-browse-rois", action="store_true", help="Open the first Pelagia detection/ROI from the filtered list.")
    parser.add_argument("--random-api-roi", action="store_true", help="Choose a random Pelagia detection/ROI id to open.")
    parser.add_argument(
        "--api-roi-id",
        "--roi-id",
        "--detection-id",
        dest="api_roi_id",
        help="Pelagia detection id to load from the REST API.",
    )
    parser.add_argument(
        "--api-endpoint-template",
        default=None,
        help="Optional generic API path template. Use {roi_id}; if omitted, Pelagia /detections/{id}/roi and /mask are used.",
    )
    parser.add_argument("--api-run-id", "--run-id", dest="api_run_id", help="Optional Pelagia /detections run_id filter.")
    parser.add_argument("--api-asset-id", "--asset-id", dest="api_asset_id", help="Optional Pelagia /detections asset_id filter.")
    parser.add_argument("--api-collection", "--collection", dest="api_collection", help="Optional Pelagia /detections collection filter.")
    parser.add_argument("--api-frame-id", "--frame-id", dest="api_frame_id", help="Optional Pelagia /detections frame_id filter.")
    parser.add_argument("--api-start-frame", "--start-frame", dest="api_start_frame", type=int, help="Optional start_frame filter.")
    parser.add_argument("--api-end-frame", "--end-frame", dest="api_end_frame", type=int, help="Optional end_frame filter.")
    parser.add_argument("--api-roi-index", "--roi-index", dest="api_roi_index", type=int, help="Optional roi_index filter.")
    parser.add_argument("--api-min-bbox-x", "--min-bbox-x", dest="api_min_bbox_x", type=int, help="Optional min_bbox_x filter.")
    parser.add_argument("--api-max-bbox-x", "--max-bbox-x", dest="api_max_bbox_x", type=int, help="Optional max_bbox_x filter.")
    parser.add_argument("--api-min-bbox-y", "--min-bbox-y", dest="api_min_bbox_y", type=int, help="Optional min_bbox_y filter.")
    parser.add_argument("--api-max-bbox-y", "--max-bbox-y", dest="api_max_bbox_y", type=int, help="Optional max_bbox_y filter.")
    parser.add_argument("--api-min-bbox-w", "--min-bbox-w", "--min-width", dest="api_min_bbox_w", type=int, help="Optional min_bbox_w filter.")
    parser.add_argument("--api-max-bbox-w", "--max-bbox-w", dest="api_max_bbox_w", type=int, help="Optional max_bbox_w filter.")
    parser.add_argument("--api-min-bbox-h", "--min-bbox-h", "--min-height", dest="api_min_bbox_h", type=int, help="Optional min_bbox_h filter.")
    parser.add_argument("--api-max-bbox-h", "--max-bbox-h", dest="api_max_bbox_h", type=int, help="Optional max_bbox_h filter.")
    parser.add_argument("--api-min-area", "--min-area", dest="api_min_area", type=float, help="Optional min_area filter.")
    parser.add_argument("--api-max-area", "--max-area", dest="api_max_area", type=float, help="Optional max_area filter.")
    parser.add_argument("--api-min-perimeter", "--min-perimeter", dest="api_min_perimeter", type=float, help="Optional min_perimeter filter.")
    parser.add_argument("--api-max-perimeter", "--max-perimeter", dest="api_max_perimeter", type=float, help="Optional max_perimeter filter.")
    parser.add_argument("--api-roi-encoding", "--roi-encoding", dest="api_roi_encoding", help="Optional roi_encoding filter.")
    parser.add_argument("--api-roi-format", "--roi-format", dest="api_roi_format", help="Optional roi_format filter.")
    parser.add_argument("--api-mask-encoding", dest="api_mask_encoding", help="Optional mask_encoding filter.")
    parser.add_argument("--api-mask-format", "--mask-format", dest="api_mask_format", help="Optional mask_format filter.")
    parser.add_argument("--api-limit", "--limit", dest="api_limit", type=int, default=25, help="Maximum detections to list or sample from.")
    parser.add_argument("--api-offset", "--offset", dest="api_offset", type=int, default=0, help="Pelagia /detections offset.")
    parser.add_argument(
        "--api-sort-by",
        "--sort-by",
        dest="api_sort_by",
        help="Pelagia /detections sort_by value. Defaults to asset_frame, or random for --random-api-roi.",
    )
    parser.add_argument("--api-sort-dir", "--sort-dir", dest="api_sort_dir", choices=["asc", "desc"], default="desc", help="Pelagia /detections sort_dir value.")
    parser.add_argument("--uuid", help="Sample UUID. Defaults to the image filename stem for image imports.")
    parser.add_argument("--split", help="Optional split filter such as train, validation, test, holdout.")
    parser.add_argument("--missing-masks-only", action="store_true")
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--mask-encoding", choices=["png", "npy"], default="png")
    parser.add_argument("--validate-unet-dataset", action="store_true", help="Validate that --database is ready for U-Net training and exit.")
    parser.add_argument("--write-unet-config", type=Path, help="Write a U-Net config TOML inferred from --database and exit.")
    parser.add_argument("--unet-batch-size", type=int, default=8, help="Batch size for --write-unet-config.")
    parser.add_argument("--unet-epochs", type=int, default=20, help="Epoch count for --write-unet-config.")
    parser.add_argument(
        "--unet-model",
        choices=("unet", "residual_unet", "unet_plus_plus"),
        default="unet",
        help="Architecture for --write-unet-config.",
    )
    parser.add_argument(
        "--unet-segmentation-target",
        choices=("validated_mask", "candidate_delta"),
        default="validated_mask",
        help="Training target for --write-unet-config.",
    )
    parser.add_argument(
        "--unet-candidate-sdf",
        action="store_true",
        help="Append a signed-distance field derived from the candidate mask as model input channel 2.",
    )
    parser.add_argument(
        "--unet-candidate-sdf-clip-distance",
        type=float,
        default=32.0,
        help="Pixel distance mapped to +/-1 in the generated candidate SDF channel.",
    )
    parser.add_argument(
        "--unet-tiling",
        action="store_true",
        help="Enable large-ROI tiling in the generated U-Net config.",
    )
    parser.add_argument(
        "--unet-tiling-overlap",
        type=float,
        default=0.5,
        help="Fractional overlap between adjacent generated-config tiles.",
    )
    parser.add_argument(
        "--unet-tiling-blend",
        choices=("uniform", "hann"),
        default="hann",
        help="Probability blending window used to reassemble tiled predictions.",
    )
    parser.add_argument("--unet-input-shape", type=parse_shape, help="Target U-Net input shape, for example 256,256,2.")
    parser.add_argument("--unet-output-shape", type=parse_shape, help="Target U-Net output shape, for example 256,256,1.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def resolve_database_path(args: argparse.Namespace) -> Path | None:
    database = args.database
    if args.output:
        if database and database != args.output:
            raise SystemExit("--database and --output refer to different files. Use --database only.")
        database = args.output
    if args.input and not args.image and args.input.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        if database and database != args.input:
            raise SystemExit("--database and --input refer to different database files. Use --database only.")
        database = args.input
    return database


def pelagia_detection_filters(args: argparse.Namespace, default_sort_by: str = "asset_frame") -> dict:
    return {
        "run_id": args.api_run_id,
        "asset_id": args.api_asset_id,
        "collection": args.api_collection,
        "frame_id": args.api_frame_id,
        "start_frame": args.api_start_frame,
        "end_frame": args.api_end_frame,
        "roi_index": args.api_roi_index,
        "min_bbox_x": args.api_min_bbox_x,
        "max_bbox_x": args.api_max_bbox_x,
        "min_bbox_y": args.api_min_bbox_y,
        "max_bbox_y": args.api_max_bbox_y,
        "min_bbox_w": args.api_min_bbox_w,
        "max_bbox_w": args.api_max_bbox_w,
        "min_bbox_h": args.api_min_bbox_h,
        "max_bbox_h": args.api_max_bbox_h,
        "min_area": args.api_min_area,
        "max_area": args.api_max_area,
        "min_perimeter": args.api_min_perimeter,
        "max_perimeter": args.api_max_perimeter,
        "roi_encoding": args.api_roi_encoding,
        "roi_format": args.api_roi_format,
        "mask_encoding": args.api_mask_encoding,
        "mask_format": args.api_mask_format,
        "limit": args.api_limit,
        "offset": args.api_offset,
        "sort_by": args.api_sort_by or default_sort_by,
        "sort_dir": args.api_sort_dir,
    }


def resolve_pelagia_token(args: argparse.Namespace) -> str | None:
    token = args.api_token or os.environ.get("PELAGIA_API_TOKEN")
    if token:
        return token

    username = args.api_username or os.environ.get("PELAGIA_USERNAME")
    password = args.api_password or os.environ.get("PELAGIA_PASSWORD")
    project_key = args.api_project_key or os.environ.get("PELAGIA_PROJECT_KEY") or "default"
    if not username and not password:
        return None
    if not username or not password:
        raise SystemExit("Pelagia login requires both --api-username and --api-password, or PELAGIA_USERNAME and PELAGIA_PASSWORD.")
    return login_pelagia(args.api_base_url, username, password, project_key=project_key).token


def print_detection_list(detections: list[dict]) -> None:
    if not detections:
        print("No detections found.")
        return
    print("id\tasset_id\tframe_id\tframe_index\troi_index\tarea\tmask_payload_bytes")
    for detection in detections:
        print(
            "\t".join(
                str(value if value is not None else "")
                for value in (
                    detection_id_from_summary(detection),
                    detection.get("asset_id"),
                    detection.get("frame_id"),
                    detection.get("frame_index"),
                    detection.get("roi_index"),
                    detection.get("area"),
                    detection.get("mask_payload_bytes"),
                )
            )
        )


def path_is_image_source(path: Path) -> bool:
    return path.is_dir() or path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def image_queue_from_path(path: Path) -> list[dict]:
    if path.is_dir():
        files = list_image_files(path)
        if not files:
            raise SystemExit(f"No supported image files found in folder: {path}")
        return [{"uuid": file.stem, "path": file} for file in files]
    return [{"uuid": path.stem, "path": path}]


def parse_shape(value: str) -> list[int]:
    parts = value.replace("x", ",").split(",")
    try:
        shape = [int(part.strip()) for part in parts if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid shape {value!r}; use comma-separated integers.") from exc
    if len(shape) not in {2, 3} or any(part <= 0 for part in shape):
        raise argparse.ArgumentTypeError(f"Invalid shape {value!r}; expected two or three positive integers.")
    return shape


def main() -> int:
    args = parse_args()
    if args.input and not args.image and path_is_image_source(args.input):
        args.image = args.input
        args.input = None
        if not args.uuid:
            args.uuid = args.image.stem if args.image.is_file() else None

    database_path = resolve_database_path(args)
    if args.validate_unet_dataset or args.write_unet_config:
        if not database_path:
            raise SystemExit("--validate-unet-dataset and --write-unet-config require --database.")
        report = validate_unet_dataset(
            database_path,
            target_input_shape=args.unet_input_shape,
            target_output_shape=args.unet_output_shape,
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        if args.write_unet_config:
            result = write_unet_config_from_dataset(
                database_path,
                args.write_unet_config,
                model_name=args.unet_model,
                segmentation_target=args.unet_segmentation_target,
                candidate_sdf=args.unet_candidate_sdf,
                candidate_sdf_clip_distance=args.unet_candidate_sdf_clip_distance,
                tiling_enabled=args.unet_tiling,
                tiling_overlap_fraction=args.unet_tiling_overlap,
                tiling_blend_mode=args.unet_tiling_blend,
                batch_size=args.unet_batch_size,
                epochs=args.unet_epochs,
                target_input_shape=args.unet_input_shape,
                target_output_shape=args.unet_output_shape,
            )
            print(f"Wrote U-Net config to {args.write_unet_config}")
            if args.debug:
                print(json.dumps(result["config"], indent=2, sort_keys=True))
        return 0 if report["valid"] else 2
    api_token = None
    api_requested = args.list_api_rois or args.random_api_roi or args.api_browse_rois or bool(args.api_roi_id)
    if api_requested:
        api_token = resolve_pelagia_token(args)
    if args.list_api_rois:
        detections = list_pelagia_detections(args.api_base_url, token=api_token, **pelagia_detection_filters(args))
        print_detection_list(detections)
        return 0
    api_queue = None
    if args.random_api_roi or args.api_browse_rois:
        if args.api_roi_id:
            raise SystemExit("--api-browse-rois/--random-api-roi cannot be combined with --api-roi-id.")
        default_sort_by = "random" if args.random_api_roi else "asset_frame"
        api_queue = list_pelagia_detections(
            args.api_base_url,
            token=api_token,
            **pelagia_detection_filters(args, default_sort_by=default_sort_by),
        )
        if not api_queue:
            raise SystemExit("No Pelagia detections found for the requested filters.")
        args.api_roi_id = detection_id_from_summary(api_queue[0])
        if args.debug:
            selection_mode = "Random" if args.random_api_roi else "First"
            print(f"{selection_mode} Pelagia detection selected: {args.api_roi_id}")

    database_source = bool(database_path and not args.image and not args.api_roi_id)
    source_count = sum(bool(value) for value in (database_source, args.image, args.api_roi_id))
    if source_count != 1:
        raise SystemExit("Exactly one source must be supplied: --database, --image, or --api-roi-id.")
    if args.image and not database_path and not args.read_only:
        raise SystemExit("--image, or an image path passed to --input, requires --database unless --read-only is passed.")
    if args.image and args.image.is_file() and not args.uuid:
        args.uuid = args.image.stem
    if args.api_roi_id and not database_path and not args.read_only:
        raise SystemExit("--api-roi-id requires --database unless --read-only is passed.")

    sample_queue = None
    sample_loader = None
    initial_metadata = None

    if args.api_roi_id:
        def load_pelagia_queue_sample(sample_info: dict) -> dict:
            detection_id = detection_id_from_summary(sample_info)
            sample = load_pelagia_detection(args.api_base_url, detection_id, token=api_token)
            return {
                "uuid": sample.uuid,
                "image": sample.image,
                "mask": sample.mask,
                "candidate_mask": sample.mask,
                "metadata": sample.metadata,
            }

        if args.api_endpoint_template:
            api_sample = load_api_roi(
                args.api_base_url,
                args.api_roi_id,
                endpoint_template=args.api_endpoint_template,
                token=api_token,
            )
        else:
            api_sample = load_pelagia_detection(args.api_base_url, args.api_roi_id, token=api_token)
        image = api_sample.image
        sample_uuid = args.uuid or api_sample.uuid
        initial_candidate_mask = api_sample.mask if api_sample.mask is not None else np.zeros(image.shape[:2], dtype="uint8")
        initial_mask = initial_candidate_mask.copy()
        initial_metadata = api_sample.metadata
        if api_queue:
            sample_queue = api_queue
            sample_loader = load_pelagia_queue_sample
        if database_path and not args.read_only:
            with open_database(database_path) as conn:
                create_or_update_image_sample(conn, sample_uuid, image, "png", api_sample.metadata, candidate_mask=initial_candidate_mask)
        db_path = None
    elif args.image:
        image_queue = image_queue_from_path(args.image)
        first_info = image_queue[0]

        def load_image_queue_sample(sample_info: dict) -> dict:
            loaded_image, loaded_metadata = load_image(sample_info["path"])
            return {
                "uuid": sample_info["uuid"],
                "image": loaded_image,
                "mask": None,
                "candidate_mask": None,
                "metadata": loaded_metadata,
            }

        loaded_sample = load_image_queue_sample(first_info)
        image = loaded_sample["image"]
        initial_metadata = loaded_sample["metadata"]
        sample_uuid = args.uuid or loaded_sample["uuid"]
        initial_candidate_mask = None
        initial_mask = np.zeros(image.shape[:2], dtype="uint8")
        sample_queue = image_queue
        sample_loader = load_image_queue_sample
        if database_path and not args.read_only:
            with open_database(database_path) as conn:
                create_or_update_image_sample(conn, sample_uuid, image, "png", initial_metadata)
        db_path = None
    else:
        db_path = str(database_path)
        with open_database(database_path, create=True) as conn:
            if args.uuid:
                sample = load_sample(conn, args.uuid)
                sample_queue = [{"uuid": args.uuid}]
            else:
                sample_queue = list_samples(conn, split=args.split, missing_masks_only=args.missing_masks_only)
                if not sample_queue:
                    raise SystemExit("No matching samples found.")
                sample = load_sample(conn, sample_queue[0]["uuid"])
        sample_uuid = sample["uuid"]
        image = sample["image"]
        initial_candidate_mask = sample.get("candidate_mask")
        initial_mask = sample["mask"] if sample["mask"] is not None else np.zeros(image.shape[:2], dtype="uint8")
        initial_metadata = sample.get("metadata", {})

    if args.debug:
        print(f"Launching mask builder for uuid={sample_uuid}")
        print(f"image shape={np.asarray(image).shape}, mask shape={np.asarray(initial_mask).shape}")
        print(f"database={database_path}, read_only={args.read_only}")

    from oracle_builder.masking.napari_app import launch_mask_builder_app

    launch_mask_builder_app(
        image=image,
        sample_uuid=sample_uuid,
        db_path=db_path,
        output_db_path=str(database_path) if database_path else None,
        initial_mask=initial_mask,
        initial_candidate_mask=initial_candidate_mask,
        initial_metadata=initial_metadata,
        mask_encoding=args.mask_encoding,
        read_only=args.read_only,
        sample_queue=sample_queue,
        sample_loader=sample_loader,
        debug=args.debug,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
