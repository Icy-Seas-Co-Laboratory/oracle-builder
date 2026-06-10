#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare oracle-builder run summaries.")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    rows = []
    for run in args.runs:
        run_dir = Path(run)
        metadata = json.loads((run_dir / "run_metadata.json").read_text())
        summary_path = run_dir / "evaluation" / "evaluation_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        rows.append({"run": run_dir.name, "status": metadata.get("status"), **summary})
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

