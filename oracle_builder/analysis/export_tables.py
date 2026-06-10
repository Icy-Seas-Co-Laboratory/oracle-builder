from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def export_training_metrics(training_log: str | Path, output_csv: str | Path) -> None:
    connection = sqlite3.connect(training_log)
    try:
        df = pd.read_sql_query("SELECT * FROM epoch_metrics", connection)
    finally:
        connection.close()
    df.to_csv(output_csv, index=False)

