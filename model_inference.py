#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from oracle_builder.data.sqlite_dataset import load_arrays
from oracle_builder.evaluation.predictions import write_predictions_db
from oracle_builder.saving.load_test import load_model_for_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference for a saved oracle-builder run.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    run_dir = Path(args.run)
    config = json.loads((run_dir / "resolved_config.json").read_text())
    model = load_model_for_run(run_dir, config)
    x, y, records = load_arrays(args.input, config, split=args.split)
    write_predictions_db(model, x, y, records, config, args.output)
    print(f"Wrote predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

