#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from oracle_builder.artifacts import (
    read_run_config,
    read_run_manifest,
    split_manifest_matches_dataset,
)
from oracle_builder.data.sqlite_dataset import load_prediction_arrays
from oracle_builder.evaluation.predictions import write_predictions_db
from oracle_builder.saving.load_test import load_model_for_run
from oracle_builder.classification.evidence import IdentityEvidenceIndex
from oracle_builder.inference.batching import resolve_inference_batch_size


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference for a saved oracle-builder run.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="all", choices=("all", "train", "validation", "test"))
    parser.add_argument("--prediction-set", help="Name stored with this set of predictions. Defaults to the run directory name.")
    args = parser.parse_args()
    run_dir = Path(args.run).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if (run_dir / "artifact.json").exists():
        manifest = read_run_manifest(run_dir)
        if (
            manifest["lifecycle"] == "sealed"
            and (output_path == run_dir or run_dir in output_path.parents)
        ):
            raise ValueError(
                "Inference output cannot be written inside a sealed run artifact"
            )
    config = read_run_config(run_dir)
    model = load_model_for_run(run_dir, config)
    evidence_path = run_dir / "model" / "classification_evidence"
    if not evidence_path.exists():
        evidence_path = run_dir / "model" / "classification_evidence.npz"
    evidence_index = (
        IdentityEvidenceIndex.load(evidence_path)
        if config["run"]["task"] == "classification" and evidence_path.exists()
        else None
    )
    inference_batch_plan = resolve_inference_batch_size(model, config)
    inference_batch_size = inference_batch_plan.batch_size
    prediction_set = args.prediction_set or run_dir.name
    selected_split = None if args.split == "all" else args.split
    if not split_manifest_matches_dataset(config, args.input):
        if selected_split is not None:
            raise ValueError(
                "A named training split can only be selected when inference uses "
                "the exact dataset revision recorded by the run artifact"
            )
        config.pop("_split_manifest", None)
        config["_external_inference"] = True
    if (
        config["run"]["task"] == "classification"
        and config.get("data", {}).get("streaming", {}).get("enabled", True)
    ):
        from oracle_builder.data.sqlite_stream import (
            SQLiteClassificationSource,
            build_all_classification_index,
            build_classification_index,
        )
        from oracle_builder.evaluation.predictions import (
            write_classification_predictions_streaming,
        )

        index = (
            build_all_classification_index(args.input, config, labeled_only=False)
            if selected_split is None
            else build_classification_index(
                args.input, config, selected_split, labeled_only=False
            )
        )
        source = SQLiteClassificationSource(args.input, config)
        written = write_classification_predictions_streaming(
            model,
            source.indexed_image_dataset(
                index, batch_size=inference_batch_size
            ),
            index,
            config,
            output_path,
            source_sqlite=args.input,
            prediction_set=prediction_set,
            evidence_index=evidence_index,
            progress=bool(config.get("inference", {}).get("progress", True)),
        )
    else:
        x, targets, records = load_prediction_arrays(
            args.input, config, split=selected_split
        )
        write_predictions_db(
            model,
            x,
            targets,
            records,
            config,
            output_path,
            source_sqlite=args.input,
            prediction_set=prediction_set,
            evidence_index=evidence_index,
            inference_batch_size=inference_batch_size,
        )
        written = len(records)
    print(f"Wrote {written} predictions as set {prediction_set!r} to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
