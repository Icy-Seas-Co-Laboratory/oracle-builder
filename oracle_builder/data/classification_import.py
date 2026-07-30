from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from oracle_builder.data.decoders import encode_npy, prepare_classification_input
from oracle_builder.data.sqlite_dataset import ensure_schema

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SIDECAR_SUFFIXES = {".json", ".toml", ".yaml", ".yml"}
SPLIT_FOLDERS = {"train": "train", "validation": "validation", "val": "validation", "test": "test"}


@dataclass
class Candidate:
    path: Path
    relative_path: str
    class_name: str
    class_index: int
    split: str
    sample_uuid: str
    sha256: str
    shape: list[int]
    mode: str
    status: str = "ready"
    error: str | None = None


def stable_split(relative_path: str, class_name: str, seed: int, validation: float, test: float):
    digest = hashlib.sha256(f"{seed}:{class_name}:{relative_path}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if fraction < test:
        return "test"
    if fraction < test + validation:
        return "validation"
    return "train"


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


def parse_sidecars(root: Path) -> list[dict[str, Any]]:
    sidecars = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SIDECAR_SUFFIXES:
            continue
        raw = path.read_text()
        suffix = path.suffix.lower()
        if suffix == ".json":
            parsed = json.loads(raw)
        elif suffix == ".toml":
            with path.open("rb") as handle:
                parsed = tomllib.load(handle)
        else:
            if yaml is None:
                raise RuntimeError("YAML metadata requires PyYAML; install requirements.txt")
            parsed = yaml.safe_load(raw)
        sidecars.append(
            {
                "metadata_name": path.name,
                "source_filename": path.name,
                "source_format": suffix.lstrip("."),
                "metadata_json": json.dumps(parsed, sort_keys=True, default=str),
                "raw_text": raw,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return sidecars


def discover_classes(root: Path, split_mode: str) -> tuple[dict[str, list[tuple[Path, str | None]]], list[str]]:
    classes: dict[str, list[tuple[Path, str | None]]] = {}
    warnings = []
    if split_mode == "existing-folders":
        split_roots = [
            (path, SPLIT_FOLDERS[path.name.lower()])
            for path in sorted(root.iterdir())
            if path.is_dir() and path.name.lower() in SPLIT_FOLDERS
        ]
        if not split_roots:
            raise ValueError("existing-folders mode requires train/validation/test directories")
        for split_root, split in split_roots:
            for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
                classes.setdefault(class_dir.name, []).extend(
                    (path, split)
                    for path in sorted(class_dir.rglob("*"))
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                )
    else:
        for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
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
) -> list[Candidate]:
    candidates = []
    for class_name, files in classes.items():
        for path, explicit_split in files:
            relative = path.relative_to(root).as_posix()
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
                split = explicit_split or (
                    "train"
                    if options.split_mode == "none"
                    else stable_split(
                        relative,
                        class_name,
                        options.seed,
                        options.validation_fraction,
                        options.test_fraction,
                    )
                )
                candidates.append(
                    Candidate(
                        path=path,
                        relative_path=relative,
                        class_name=class_name,
                        class_index=label_map[class_name],
                        split=split,
                        sample_uuid=str(uuid.uuid5(uuid.NAMESPACE_URL, relative)),
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
                    split="",
                    sample_uuid=str(uuid.uuid5(uuid.NAMESPACE_URL, relative)),
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
    rows = connection.execute("SELECT uuid, metadata_json FROM samples").fetchall()
    by_uuid = {}
    by_hash = {}
    for row in rows:
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        by_uuid[row["uuid"]] = metadata
        content_hash = metadata.get("content_sha256")
        if content_hash:
            by_hash.setdefault(content_hash, []).append((row["uuid"], metadata))
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
    classes, warnings = discover_classes(root, options.split_mode)
    supplied_map = load_label_map(Path(options.label_map).expanduser() if options.label_map else None)
    if not options.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(":memory:" if options.dry_run else output)
    ensure_schema(connection)
    label_connection = connection
    if options.dry_run and output.exists():
        label_connection = sqlite3.connect(output)
    try:
        existing_labels = {
            row[1]: int(row[0])
            for row in label_connection.execute(
                "SELECT class_index, class_name FROM class_labels"
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
    candidates = build_candidates(root, classes, label_map, options)
    sidecars = parse_sidecars(root)
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
        name: {
            split: sum(
                candidate.class_name == name
                and candidate.split == split
                and candidate.status in {"ready", "update"}
                for candidate in candidates
            )
            for split in ("train", "validation", "test")
        }
        for name in sorted(classes)
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
            for name, index in label_map.items():
                connection.execute(
                    "INSERT OR REPLACE INTO class_labels (class_index, class_name) VALUES (?, ?)",
                    (index, name),
                )
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
                connection.execute(
                    """
                    INSERT OR REPLACE INTO samples (
                        uuid, split, input_blob, input_blob_encoding, input_blob_dimensions,
                        output_blob, output_blob_encoding, output_blob_dimensions,
                        label_text, sample_weight, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.sample_uuid,
                        candidate.split,
                        blob,
                        encoding,
                        dimensions,
                        str(candidate.class_index).encode(),
                        "int",
                        None,
                        candidate.class_name,
                        None,
                        json.dumps(metadata, sort_keys=True),
                    ),
                )
            connection.execute(
                """
                INSERT INTO classification_imports
                (import_id, created_at, source_root, options_json, summary_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    import_id,
                    datetime.now(timezone.utc).isoformat(),
                    str(root),
                    json.dumps(vars(options), sort_keys=True, default=str),
                    json.dumps(summary, sort_keys=True),
                ),
            )
            for sidecar in sidecars:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO dataset_metadata (
                        metadata_name, source_filename, source_format, metadata_json,
                        raw_text, sha256, import_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sidecar["metadata_name"],
                        sidecar["source_filename"],
                        sidecar["source_format"],
                        sidecar["metadata_json"],
                        sidecar["raw_text"],
                        sidecar["sha256"],
                        import_id,
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
                "split",
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
    parser.add_argument("--label-map")
    parser.add_argument("--allow-new-classes", action="store_true")
    parser.add_argument(
        "--split-mode",
        choices=("stratified-hash", "existing-folders", "none"),
        default="stratified-hash",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
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
    if not 0 <= options.validation_fraction < 1 or not 0 <= options.test_fraction < 1:
        raise SystemExit("Split fractions must be in [0, 1)")
    if options.validation_fraction + options.test_fraction >= 1:
        raise SystemExit("Validation and test fractions must sum to less than 1")
    if options.storage_mode == "materialized" and not options.input_shape:
        raise SystemExit("--storage-mode materialized requires --input-shape H W C")
    summary = import_folders(options)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
