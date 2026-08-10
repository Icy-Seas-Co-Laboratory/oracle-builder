#!/usr/bin/env python3
"""Migrate legacy Plankline SavedModels into dataset-bound model products.

Each legacy model needs a JSON sidecar containing its ordered labels and the
``training.scnn_dir`` source path.  Models without that provenance are reported
instead of guessed at; an incorrect class order makes a product unsafe.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from oracle_builder.products.ingest import ingest_savedmodel


def _dataset_labels(database: Path) -> list[str]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT name FROM classification_labels ORDER BY class_index"
        ).fetchall()
    return [str(row[0]) for row in rows]


def _write_card(path: Path, name: str, metadata: dict[str, Any]) -> None:
    labels = metadata["labels"]
    model_type = str(metadata.get("model_type", "legacy classifier"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Generated from the legacy Plankline JSON model sidecar.\n"
        "# The SavedModel exposes class probabilities only; no penultimate\n"
        "# identity embedding was published by the legacy export.\n\n"
        f"labels = {json.dumps(labels)}\n\n"
        "[product]\n"
        f"name = {json.dumps(name)}\n"
        "task = \"classification\"\n"
        "version = \"legacy-import-1\"\n"
        f"description = {json.dumps(f'Legacy Plankline {model_type} classifier imported from TensorFlow SavedModel.')}\n"
        f"tags = [\"external\", \"plankline\", \"legacy\", {json.dumps(model_type.lower())}]\n\n"
        "[preprocessing]\n"
        "resize_mode = \"fit_pad\"\n"
        "channel_mode = \"grayscale\"\n"
        "# Preserve the legacy model's expected 0–255 float input range.\n"
        "rescale = false\n"
        "normalization = \"none\"\n"
        "invert = false\n"
        "embed_in_model = false\n\n"
        "[promotion]\n"
        "enabled = true\n"
        "activation = \"softmax\"\n"
        "output_name = \"output_0\"\n",
        encoding="utf-8",
    )


def migrate(args: argparse.Namespace) -> dict[str, Any]:
    model_root = Path(args.model_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    cards_root = output_root / "legacy_model_cards"
    report: dict[str, Any] = {"migrated": [], "skipped": []}
    for metadata_path in sorted(model_root.glob("*.json")):
        if args.only_model and metadata_path.stem not in args.only_model:
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            name = str(metadata.get("model_name") or metadata_path.stem)
            labels = metadata.get("labels")
            training_path = metadata.get("config", {}).get("training", {}).get("scnn_dir")
            if not isinstance(labels, list) or not labels or not training_path:
                raise ValueError("JSON sidecar lacks labels or config.training.scnn_dir")
            source = model_root / name
            if not (source / "saved_model.pb").is_file():
                report["skipped"].append({"model": name, "reason": "no_savedmodel_export"})
                continue
            dataset_name = Path(str(training_path)).name
            database = dataset_root / f"plankline_{dataset_name}.sqlite"
            if not database.is_file():
                report["skipped"].append({"model": name, "reason": f"dataset_not_found:{dataset_name}"})
                continue
            if [str(value) for value in labels] != _dataset_labels(database):
                report["skipped"].append({"model": name, "reason": "dataset_label_order_mismatch"})
                continue
            output = output_root / name
            if output.exists():
                report["skipped"].append({"model": name, "reason": "product_already_exists"})
                continue
            card = cards_root / f"{name}.toml"
            if not args.dry_run:
                _write_card(card, name, metadata)
                result = ingest_savedmodel(source, card, output, dataset=database)
                report["migrated"].append(
                    {"model": name, "dataset": database.name, "product": str(output),
                     "artifact_id": result["artifact_id"]}
                )
            else:
                report["migrated"].append(
                    {"model": name, "dataset": database.name, "product": str(output), "dry_run": True}
                )
        except Exception as exc:
            report["skipped"].append({"model": metadata_path.stem, "reason": str(exc)})
    savedmodel_without_sidecar = sorted(
        path.name for path in model_root.iterdir()
        if path.is_dir() and (path / "saved_model.pb").is_file() and not (model_root / f"{path.name}.json").is_file()
    )
    report["skipped"].extend(
        {"model": name, "reason": "no_json_provenance_sidecar"}
        for name in savedmodel_without_sidecar
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only-model", action="append", default=[],
        help="Migrate only this model name; repeat for multiple names.",
    )
    parser.add_argument("--report")
    args = parser.parse_args()
    report = migrate(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        Path(args.report).expanduser().write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
