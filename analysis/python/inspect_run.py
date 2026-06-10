#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect one oracle-builder run.")
    parser.add_argument("run")
    args = parser.parse_args()
    run_dir = Path(args.run)
    print(json.dumps(json.loads((run_dir / "run_metadata.json").read_text()), indent=2))
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        print("\nTraining metrics:")
        print(pd.read_csv(metrics_path).tail())
    log_path = run_dir / "training_log.sqlite"
    if log_path.exists():
        connection = sqlite3.connect(log_path)
        try:
            print("\nLogged epoch metrics:")
            print(pd.read_sql_query("SELECT * FROM epoch_metrics ORDER BY epoch, split, metric LIMIT 20", connection))
        finally:
            connection.close()
    sample_metrics = run_dir / "evaluation" / "sample_metrics.csv"
    if sample_metrics.exists():
        print("\nSample metrics:")
        print(pd.read_csv(sample_metrics).head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

