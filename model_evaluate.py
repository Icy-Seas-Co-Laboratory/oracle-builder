#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from oracle_builder.evaluation.reports import evaluate_run_model
from oracle_builder.saving.load_test import load_model_for_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved oracle-builder run.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run)
    config = json.loads((run_dir / "resolved_config.json").read_text())
    model = load_model_for_run(run_dir, config)
    result = evaluate_run_model(model, config, args.input, run_dir, split=args.split)
    print(json.dumps(result.get("summary", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

