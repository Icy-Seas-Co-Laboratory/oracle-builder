#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from oracle_builder.artifacts import read_run_config, split_manifest_matches_dataset
from oracle_builder.evaluation.reports import evaluate_run_model
from oracle_builder.saving.load_test import load_model_for_run
from oracle_builder.inference.batching import resolve_inference_batch_size


def _class_names(config: dict) -> dict[int, str]:
    return {
        int(row["class_index"]): str(row.get("name") or row["class_index"])
        for row in config.get("dataset", {}).get("labels", [])
    }


def evaluate_stratified_run(model, config: dict, input_path: str, output_dir: Path, split: str) -> dict:
    """Evaluate one shared model at every canonical resolution stratum."""
    from oracle_builder.classification.stratified_data import child_config
    from oracle_builder.classification.stratified_training import (
        build_indices,
        make_canonical_bundle,
    )
    from oracle_builder.classification.stratification import dimensions
    from oracle_builder.evaluation.classification import evaluate_classification_streaming

    indices = build_indices(input_path, config)
    reports: dict[str, dict] = {}
    for dimension in dimensions(config):
        bundle = make_canonical_bundle(
            input_path, config, dimension, indices=indices
        )
        evaluated_split = split
        index = bundle.indices.get(evaluated_split)
        if index is None and split == "test":
            evaluated_split = "validation"
            index = bundle.indices.get(evaluated_split)
        if index is None or not index.refs:
            continue
        child = child_config(config, dimension)
        result = evaluate_classification_streaming(
            model,
            bundle.source.indexed_image_dataset(
                index, batch_size=int(child["data"]["batch_size"])
            ),
            index,
            output_dir / "strata" / str(dimension),
            class_names=_class_names(config),
            progress=bool(config.get("inference", {}).get("progress", True)),
            evaluation_settings=config.get("evaluation", {}),
            evaluation_context={
                "artifact_id": config.get("artifact", {}).get("artifact_id"),
                "run_id": config.get("run", {}).get("run_id"),
                "split": evaluated_split,
                "stratum_dimension": dimension,
            },
        )
        reports[str(dimension)] = {
            "split": evaluated_split,
            "summary": result["summary"],
        }
    if not reports:
        raise ValueError(f"No canonical {split!r} examples were available in any stratum")
    totals = [int(value["summary"].get("sample_count", 0)) for value in reports.values()]
    total = sum(totals)
    metric_names = (
        "accuracy", "top_1_accuracy", "top_3_accuracy", "top_5_accuracy",
        "balanced_accuracy", "macro_f1", "log_loss",
    )
    aggregate = {
        name: sum(
            int(report["summary"].get("sample_count", 0))
            * float(report["summary"][name])
            for report in reports.values()
            if report["summary"].get(name) is not None
        )
        / total
        for name in metric_names
        if any(report["summary"].get(name) is not None for report in reports.values())
        and total
    }
    summary = {
        "task": "classification",
        "stratified": True,
        "sample_count": total,
        "aggregate": aggregate,
        "strata": reports,
    }
    root = output_dir / "evaluation"
    root.mkdir(parents=True, exist_ok=True)
    (root / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return {"summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved oracle-builder run.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="New directory for evaluation outputs; sealed run artifacts are not modified.",
    )
    args = parser.parse_args()
    run_dir = Path(args.run).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError(
            "Evaluation output must be outside the preserved run artifact"
        )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    config = read_run_config(run_dir)
    if not split_manifest_matches_dataset(config, args.input):
        raise ValueError(
            "Evaluation data does not match the dataset revision and fingerprint "
            "recorded by this run's split manifest"
        )
    model = load_model_for_run(run_dir, config)
    if config.get("classification", {}).get("stratification", {}).get("enabled", False):
        result = evaluate_stratified_run(
            model, config, args.input, output_dir, args.split
        )
    else:
        inference_batch_plan = resolve_inference_batch_size(model, config)
        result = evaluate_run_model(
            model,
            config,
            args.input,
            output_dir,
            split=args.split,
            inference_batch_size=inference_batch_plan.batch_size,
        )
    print(json.dumps(result.get("summary", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
