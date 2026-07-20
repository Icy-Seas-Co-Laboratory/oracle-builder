#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from oracle_builder.data.sqlite_dataset import load_prediction_arrays
from oracle_builder.evaluation.predictions import write_predictions_db
from oracle_builder.saving.load_test import load_model_for_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference for a saved oracle-builder run.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="all", choices=("all", "train", "validation", "test"))
    parser.add_argument("--prediction-set", help="Name stored with this set of predictions. Defaults to the run directory name.")
    args = parser.parse_args()
    run_dir = Path(args.run)
    config = json.loads((run_dir / "resolved_config.json").read_text())
    model = load_model_for_run(run_dir, config)
    prediction_set = args.prediction_set or run_dir.name
    selected_split = None if args.split == "all" else args.split
    x, targets, records = load_prediction_arrays(args.input, config, split=selected_split)
    write_predictions_db(
        model,
        x,
        targets,
        records,
        config,
        args.output,
        source_sqlite=args.input,
        prediction_set=prediction_set,
    )
    written = len(records)
    print(f"Wrote {written} predictions as set {prediction_set!r} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
