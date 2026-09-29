#!/usr/bin/env python3
"""Thin local/developer adapter for the reusable inference workflow.

Supported user-facing inference is submitted to the Orchestrator.  This entry
point remains useful for controlled local development and preserves the
historic ``--output FILE.sqlite`` interface.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile

from oracle_builder.inference.workflow import InferenceRequest, run_inference


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local inference for a saved oracle-builder run.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="all", choices=("all", "train", "validation", "test"))
    parser.add_argument("--prediction-set", help="Name stored with this set of predictions. Defaults to the run directory name.")
    args = parser.parse_args()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        raise ValueError("output prediction database already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="oracle-inference-", dir=output_path.parent))
    try:
        manifest = run_inference(InferenceRequest(
            model_run=args.run, input=args.input, output_dir=str(temporary),
            split=args.split, prediction_set=args.prediction_set,
        ))
        shutil.move(str(temporary / "predictions.sqlite"), str(output_path))
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    print(f"Wrote {manifest['outputs']['records']} predictions as set {manifest['parameters']['prediction_set']!r} to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
