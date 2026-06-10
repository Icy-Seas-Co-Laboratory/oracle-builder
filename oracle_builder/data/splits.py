from __future__ import annotations

import random
from collections.abc import Iterable


def assign_missing_splits(
    rows: Iterable[dict],
    validation_split: float,
    test_split: float,
    seed: int,
) -> list[dict]:
    result = [dict(row) for row in rows]
    missing = [row for row in result if not row.get("split")]
    random.Random(seed).shuffle(missing)
    n = len(missing)
    n_test = int(round(n * test_split))
    n_val = int(round(n * validation_split))
    for index, row in enumerate(missing):
        if index < n_test:
            row["split"] = "test"
        elif index < n_test + n_val:
            row["split"] = "validation"
        else:
            row["split"] = "train"
    for row in result:
        if not row.get("split"):
            row["split"] = "train"
    return result

