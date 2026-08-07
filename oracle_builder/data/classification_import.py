from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from oracle_builder.data.decoders import encode_npy, prepare_classification_input
from oracle_builder.data.polarity import POLARITY_VALUES, resolve_polarity
from oracle_builder.data.sqlite_dataset import ensure_schema
from oracle_builder.datasets.metadata import discover_metadata_documents
from oracle_builder.datasets.repository import SQLiteDatasetRepository
from oracle_builder.datasets.schema import read_dataset_info, utc_now


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SPLIT_FOLDERS = {"train": "train", "validation": "validation", "val": "validation", "test": "test"}


@dataclass
class Candidate:
    path: Path
    relative_path: str
    class_name: str
    class_index: int
    source_partition: str | None
    sample_uuid: str
    sha256: str
    shape: list[int]
    mode: str
    status: str = "ready"
    error: str | None = None


def load_label_map(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Label map must be a JSON object mapping class names to integer indices")
    result = {str(name): int(index) for name, index in value.items()}
    if len(set(result.values())) != len(result):
        raise ValueError("Label-map indices must be unique")
    if sorted(result.values()) != list(range(len(result))):
        raise ValueError("Label-map indices must be contiguous and start at zero")
    return result


def discover_classes(root: Path) -> tuple[dict[str, list[tuple[Path, str | None]]], list[str]]:
    """Discover ordinary class folders or an explicit source-partition layout.

    A partition is source provenance, not a dataset split.  A model run may
    later elect to materialize these partitions in its own split manifest.
    """
    classes: dict[str, list[tuple[Path, str | None]]] = {}
    warnings = []
    top_level = [path for path in sorted(root.iterdir()) if path.is_dir()]
    partition_roots = [
        (path, SPLIT_FOLDERS[path.name.lower()])
        for path in top_level
        if path.name.lower() in SPLIT_FOLDERS
    ]
    # Require at least two recognized top-level partitions. This avoids
    # interpreting a legitimate class named "train" as a source layout.
    if len(partition_roots) >= 2:
        for split_root, partition in partition_roots:
            for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
                classes.setdefault(class_dir.name, []).extend(
                    (path, partition)
                    for path in sorted(class_dir.rglob("*"))
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                )
    else:
        for class_dir in top_level:
            classes[class_dir.name] = [
                (path, None)
                for path in sorted(class_dir.rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]
    if not classes:
        raise ValueError(f"No class folders containing supported images found under {root}")
    for name, files in classes.items():
        if not files:
            warnings.append(f"Class {name!r} contains no supported images")
    return classes, warnings


def build_candidates(
    root: Path,
    classes: dict[str, list[tuple[Path, str | None]]],
    label_map: dict[str, int],
    options: argparse.Namespace,
    dataset_id: str,
) -> list[Candidate]:
    candidates = []
    for class_name, files in classes.items():
        for path, source_partition in files:
            source_relative = path.relative_to(root)
            relative = source_relative.as_posix()
            try:
                raw = path.read_bytes()
                with Image.open(path) as image:
                    image.load()
                    shape = list(np.asarray(image).shape)
                    mode = image.mode
                if options.require_rgb and mode != "RGB":
                    raise ValueError(f"requires RGB but image mode is {mode}")
                if not options.allow_grayscale and mode in {"1", "L", "I", "F"}:
                    raise ValueError(f"grayscale image mode {mode} is not allowed")
                candidates.append(
                    Candidate(
                        path=path,
                        relative_path=relative,
                        class_name=class_name,
                        class_index=label_map[class_name],
                        source_partition=source_partition,
                        sample_uuid=str(uuid.uuid5(uuid.UUID(dataset_id), f"item:{relative}")),
                        sha256=hashlib.sha256(raw).hexdigest(),
                        shape=shape,
                        mode=mode,
                    )
                )
            except Exception as exc:
                candidate = Candidate(
                    path=path,
                    relative_path=relative,
                    class_name=class_name,
                    class_index=label_map[class_name],
                    source_partition=source_partition,
                    sample_uuid=str(uuid.uuid5(uuid.UUID(dataset_id), f"item:{relative}")),
                    sha256="",
                    shape=[],
                    mode="",
                    status="error",
                    error=str(exc),
                )
                candidates.append(candidate)
                if options.on_error == "error":
                    raise ValueError(f"Failed to read {path}: {exc}") from exc
    return candidates


def _existing_state(connection: sqlite3.Connection):
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT di.item_id, di.metadata_json, a.content_sha256, l.name AS class_name
        FROM dataset_items di
        JOIN classification_items ci ON ci.item_id = di.item_id
        JOIN assets a ON a.asset_id = ci.image_asset_id
        LEFT JOIN classification_annotations ca
          ON ca.item_id = di.item_id AND ca.is_current = 1
        LEFT JOIN classification_labels l ON l.label_id = ca.label_id
        """
    ).fetchall()
    by_uuid = {}
    by_hash = {}
    for row in rows:
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        metadata.setdefault("class_name", row["class_name"])
        by_uuid[row["item_id"]] = metadata
        content_hash = row["content_sha256"]
        if content_hash:
            by_hash.setdefault(content_hash, []).append((row["item_id"], metadata))
    return by_uuid, by_hash


def _encoded_input(candidate: Candidate, options: argparse.Namespace):
    raw = candidate.path.read_bytes()
    encoding = candidate.path.suffix.lower().lstrip(".")
    encoding = "tif" if encoding == "tiff" else encoding
    if options.storage_mode == "original":
        return raw, encoding, json.dumps(candidate.shape)
    with Image.open(candidate.path) as image:
        array = np.asarray(image)
    config = {
        "preprocessing": {
            "resize_mode": options.resize_mode,
            "normalization": options.normalization,
            "rescale": options.rescale,
            "invert": options.invert,
            "pad_value": options.pad_value,
            "interpolation": options.interpolation,
            "channel_mode": options.channel_mode,
        }
    }
    materialized = prepare_classification_input(array, options.input_shape, config)
    return encode_npy(materialized), "npy", json.dumps(list(materialized.shape))


def import_folders(options: argparse.Namespace) -> dict[str, Any]:
    root = Path(options.input).expanduser().resolve()
    output = Path(options.output).expanduser().resolve()
    classes, warnings = discover_classes(root)
    sidecars = discover_metadata_documents(root)
    primary_metadata = next(
        (
            sidecar["parsed"]
            for sidecar in sidecars
            if sidecar["source_filename"].lower() == "metadata.toml"
            and isinstance(sidecar["parsed"], dict)
        ),
        {},
    )
    dataset_metadata = (
        primary_metadata.get("dataset", {})
        if isinstance(primary_metadata.get("dataset"), dict)
        else {}
    )
    oracle_metadata = (
        primary_metadata.get("oracle_builder", {})
        if isinstance(primary_metadata.get("oracle_builder"), dict)
        else {}
    )
    sidecar_imaging = (
        primary_metadata.get("imaging", {})
        if isinstance(primary_metadata.get("imaging"), dict)
        else {}
    )
    polarity = resolve_polarity(
        options.source_polarity,
        metadata=sidecar_imaging,
        paths=(path for files in classes.values() for path, _ in files),
        sample_count=options.polarity_sample_count,
    )
    if polarity["value"] in {"mixed", "unknown"}:
        warnings.append(
            "Source polarity is %r; preprocessing.invert='auto' will not invert. "
            "Specify --source-polarity when the acquisition convention is known."
            % polarity["value"]
        )
    supplied_map = load_label_map(Path(options.label_map).expanduser() if options.label_map else None)
    if not options.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
    existing_dataset_id = None
    if options.dry_run and output.exists():
        with sqlite3.connect(output) as existing_connection:
            existing_dataset_id = read_dataset_info(existing_connection)["dataset_id"]
    requested_dataset_id = (
        getattr(options, "dataset_id", None)
        or oracle_metadata.get("dataset_id")
        or dataset_metadata.get("dataset_id")
        or existing_dataset_id
    )
    connection = sqlite3.connect(":memory:" if options.dry_run else output)
    info = ensure_schema(
        connection,
        "classification",
        dataset_id=requested_dataset_id,
        name=(
            dataset_metadata.get("training_set_name")
            or dataset_metadata.get("short_name")
            or dataset_metadata.get("name")
            or root.name
        ),
        title=dataset_metadata.get("dataset_title") or dataset_metadata.get("title"),
        description=dataset_metadata.get("description"),
        version=dataset_metadata.get("version"),
        metadata={
            **({"source_metadata": primary_metadata} if primary_metadata else {}),
            "imaging": {**sidecar_imaging, "polarity": polarity},
        },
    )
    connection.commit()
    if not options.dry_run:
        current_metadata = info.get("metadata", {})
        current_metadata["imaging"] = {**sidecar_imaging, "polarity": polarity}
        connection.execute(
            "UPDATE dataset SET metadata_json = ?, updated_at = ? WHERE singleton = 1",
            (json.dumps(current_metadata, sort_keys=True), utc_now()),
        )
        connection.commit()
        info = read_dataset_info(connection)
    label_connection = connection
    if options.dry_run and output.exists():
        label_connection = sqlite3.connect(output)
    try:
        existing_labels = {
            row[1]: int(row[0])
            for row in label_connection.execute(
                "SELECT class_index, name FROM classification_labels"
            )
        }
    except sqlite3.OperationalError:
        existing_labels = {}
    finally:
        if label_connection is not connection:
            label_connection.close()
    label_map = supplied_map or dict(existing_labels)
    if supplied_map is not None:
        missing = set(classes) - set(supplied_map)
        if missing:
            connection.close()
            raise ValueError(f"Label map does not contain classes: {sorted(missing)}")
    for name, index in existing_labels.items():
        if name in label_map and label_map[name] != index:
            connection.close()
            raise ValueError(
                f"Existing class {name!r} is index {index}, not requested index {label_map[name]}"
            )
    next_index = max(label_map.values(), default=-1) + 1
    for class_name in sorted(classes):
        if class_name not in label_map:
            if existing_labels and not options.allow_new_classes:
                connection.close()
                raise ValueError(
                    f"New class {class_name!r} requires --allow-new-classes or --label-map"
                )
            label_map[class_name] = next_index
            next_index += 1
    candidates = build_candidates(
        root, classes, label_map, options, info["dataset_id"]
    )
    state_connection = connection
    if options.dry_run and output.exists():
        state_connection = sqlite3.connect(output)
    try:
        by_uuid, by_hash = _existing_state(state_connection)
    except sqlite3.OperationalError:
        by_uuid, by_hash = {}, {}
    finally:
        if state_connection is not connection:
            state_connection.close()
    seen_hashes: dict[str, Candidate] = {}
    for candidate in candidates:
        if candidate.status != "ready":
            continue
        duplicate = seen_hashes.get(candidate.sha256)
        existing_duplicates = by_hash.get(candidate.sha256, [])
        if duplicate or existing_duplicates:
            conflicting = (
                duplicate is not None and duplicate.class_name != candidate.class_name
            ) or any(
                metadata.get("class_name") != candidate.class_name
                for _, metadata in existing_duplicates
            )
            if conflicting and options.duplicate_policy != "allow":
                candidate.status = "error"
                candidate.error = "Identical content has conflicting class assignments"
            elif options.duplicate_policy == "error":
                candidate.status = "error"
                candidate.error = "Duplicate image content"
            elif options.duplicate_policy == "skip":
                candidate.status = "skipped_duplicate"
        seen_hashes.setdefault(candidate.sha256, candidate)
        if candidate.sample_uuid in by_uuid:
            if options.existing_policy == "error":
                candidate.status = "error"
                candidate.error = "Sample UUID already exists"
            elif options.existing_policy == "skip":
                candidate.status = "skipped_existing"
            else:
                candidate.status = "update"
    errors = [candidate for candidate in candidates if candidate.status == "error"]
    if errors and options.on_error == "error":
        connection.close()
        raise ValueError(errors[0].error or "Import validation failed")
    counts_by_class = {
        name: sum(
            candidate.class_name == name and candidate.status in {"ready", "update"}
            for candidate in candidates
        )
        for name in sorted(classes)
    }
    counts_by_source_partition = {
        partition: sum(
            candidate.source_partition == partition
            and candidate.status in {"ready", "update"}
            for candidate in candidates
        )
        for partition in ("train", "validation", "test")
        if any(candidate.source_partition == partition for candidate in candidates)
    }
    too_small = [
        name
        for name in classes
        if sum(
            candidate.class_name == name and candidate.status != "error"
            for candidate in candidates
        )
        < options.minimum_images_per_class
    ]
    if too_small:
        connection.close()
        raise ValueError(f"Classes below --minimum-images-per-class: {too_small}")
    import_id = str(uuid.uuid4())
    summary = {
        "import_id": import_id,
        "source_root": str(root),
        "output": str(output),
        "dry_run": bool(options.dry_run),
        "class_labels": label_map,
        "counts_by_class": counts_by_class,
        "counts_by_source_partition": counts_by_source_partition,
        "imaging": {**sidecar_imaging, "polarity": polarity},
        "status_counts": {
            status: sum(candidate.status == status for candidate in candidates)
            for status in sorted({candidate.status for candidate in candidates})
        },
        "sidecars": [sidecar["source_filename"] for sidecar in sidecars],
        "warnings": warnings,
        "errors": [
            {"path": candidate.relative_path, "error": candidate.error}
            for candidate in errors
        ],
    }
    if not options.dry_run:
        try:
            connection.execute("BEGIN")
            repository = SQLiteDatasetRepository(connection)
            label_ids = {}
            for name, index in label_map.items():
                label_ids[name] = repository.add_classification_label(index, name)
            for candidate in candidates:
                if candidate.status not in {"ready", "update"}:
                    continue
                blob, encoding, dimensions = _encoded_input(candidate, options)
                metadata = {
                    "source_relative_path": candidate.relative_path,
                    "source_filename": candidate.path.name,
                    "source_encoding": candidate.path.suffix.lower().lstrip("."),
                    "source_shape": candidate.shape,
                    "source_mode": candidate.mode,
                    "class_name": candidate.class_name,
                    "class_index": candidate.class_index,
                    "content_sha256": candidate.sha256,
                    "import_id": import_id,
                    "storage_mode": options.storage_mode,
                }
                if candidate.source_partition is not None:
                    metadata["source_partition"] = candidate.source_partition
                shape = json.loads(dimensions) if dimensions else candidate.shape
                asset = repository.add_asset(
                    blob,
                    encoding=encoding,
                    media_type=(
                        "application/x-npy"
                        if encoding == "npy"
                        else f"image/{'jpeg' if encoding in {'jpg', 'jpeg'} else encoding}"
                    ),
                    shape=shape,
                    dtype="float32" if encoding == "npy" else None,
                    original_filename=candidate.path.name,
                    metadata={"source_mode": candidate.mode},
                    content_sha256=(
                        candidate.sha256 if options.storage_mode == "original" else None
                    ),
                )
                item_id = repository.add_item(
                    item_id=candidate.sample_uuid,
                    source_key=candidate.relative_path,
                    # Split membership is an experimental protocol owned by a
                    # model-run artifact. The import report may describe source
                    # layout without making it semantic dataset state.
                    metadata=metadata,
                )
                repository.add_classification_item(
                    item_id=item_id,
                    image_asset_id=asset.asset_id,
                    label_id=label_ids[candidate.class_name],
                    source="folder_import",
                    metadata={"import_id": import_id},
                )
            connection.execute(
                """
                INSERT INTO import_events
                (import_id, dataset_id, revision_id, created_at, importer,
                 source_uri, options_json, summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    import_id,
                    info["dataset_id"],
                    info["revision_id"],
                    utc_now(),
                    "classification_folders",
                    str(root),
                    json.dumps(vars(options), sort_keys=True, default=str),
                    json.dumps(summary, sort_keys=True),
                ),
            )
            for sidecar in sidecars:
                connection.execute(
                    """
                    INSERT INTO metadata_documents (
                        document_id, dataset_id, name, source_filename, source_format,
                        parsed_json, raw_text, sha256, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(dataset_id, name) DO UPDATE SET
                        source_filename = excluded.source_filename,
                        source_format = excluded.source_format,
                        parsed_json = excluded.parsed_json,
                        raw_text = excluded.raw_text,
                        sha256 = excluded.sha256,
                        created_at = excluded.created_at
                    """,
                    (
                        str(uuid.uuid5(uuid.UUID(info["dataset_id"]), f"metadata:{sidecar['metadata_name']}")),
                        info["dataset_id"],
                        sidecar["metadata_name"],
                        sidecar["source_filename"],
                        sidecar["source_format"],
                        sidecar["metadata_json"],
                        sidecar["raw_text"],
                        sidecar["sha256"],
                        utc_now(),
                    ),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    connection.close()
    report_path = (
        Path(options.report).expanduser()
        if options.report
        else output.with_suffix(".import_report.json")
    )
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    csv_path = report_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "relative_path",
                "class_name",
                "class_index",
                "source_partition",
                "shape",
                "mode",
                "sha256",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        for candidate in candidates:
            row = vars(candidate).copy()
            row["shape"] = json.dumps(row["shape"])
            row.pop("path")
            row.pop("sample_uuid")
            writer.writerow(row)
    output.with_suffix(".labels.json").write_text(
        json.dumps(label_map, indent=2, sort_keys=True) + "\n"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import class-folder image libraries into oracle-builder SQLite."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dataset-id",
        help="UUID to use for a new dataset (otherwise generated once and stored).",
    )
    parser.add_argument("--label-map")
    parser.add_argument("--allow-new-classes", action="store_true")
    parser.add_argument(
        "--source-polarity",
        choices=POLARITY_VALUES,
        default="auto",
        help="Source foreground/background polarity; auto records a conservative estimate.",
    )
    parser.add_argument("--polarity-sample-count", type=int, default=128)
    parser.add_argument("--minimum-images-per-class", type=int, default=1)
    parser.add_argument("--require-rgb", action="store_true")
    parser.add_argument("--allow-grayscale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--on-error", choices=("error", "skip"), default="error")
    parser.add_argument("--duplicate-policy", choices=("error", "skip", "allow"), default="skip")
    parser.add_argument("--existing-policy", choices=("error", "skip", "update"), default="skip")
    parser.add_argument("--storage-mode", choices=("original", "materialized"), default="original")
    parser.add_argument("--input-shape", type=int, nargs=3)
    parser.add_argument(
        "--resize-mode",
        choices=("fit_pad", "fill_crop", "stretch", "none", "fit"),
        default="fit_pad",
    )
    parser.add_argument(
        "--normalization",
        choices=("dtype", "minmax", "percentile", "none"),
        default="dtype",
    )
    parser.add_argument("--rescale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--pad-value", type=float, default=0.0)
    parser.add_argument(
        "--interpolation",
        choices=("nearest", "bilinear", "bicubic", "lanczos"),
        default="bilinear",
    )
    parser.add_argument(
        "--channel-mode",
        choices=("auto", "grayscale", "rgb", "rgba"),
        default="auto",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report")
    return parser


def main() -> int:
    options = build_parser().parse_args()
    if options.polarity_sample_count < 1:
        raise SystemExit("--polarity-sample-count must be positive")
    if options.storage_mode == "materialized" and not options.input_shape:
        raise SystemExit("--storage-mode materialized requires --input-shape H W C")
    summary = import_folders(options)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
