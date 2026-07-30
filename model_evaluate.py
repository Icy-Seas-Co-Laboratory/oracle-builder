#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from oracle_builder.artifacts import read_run_config, split_manifest_matches_dataset
from oracle_builder.evaluation.reports import evaluate_run_model
from oracle_builder.saving.load_test import load_model_for_run
from oracle_builder.inference.batching import resolve_inference_batch_size


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
