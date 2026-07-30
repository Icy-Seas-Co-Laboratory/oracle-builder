from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any


def assign_missing_splits(
    rows: Iterable[dict],
    validation_split: float,
    test_split: float,
    seed: int,
) -> list[dict]:
    result = [dict(row) for row in rows]
    shuffled = list(result)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = int(round(n * test_split))
    n_val = int(round(n * validation_split))
    for index, row in enumerate(shuffled):
        if index < n_test:
            row["split"] = "test"
        elif index < n_test + n_val:
            row["split"] = "validation"
        else:
            row["split"] = "train"
    return result


def assign_run_splits(rows: Iterable[dict], config: dict[str, Any]) -> list[dict]:
    """Apply a run-owned split manifest, with an in-memory debug fallback."""
    result = [dict(row) for row in rows]
    if config.get("_external_inference"):
        for row in result:
            row["split"] = "inference"
        return result
    runtime = config.get("_split_manifest")
    if runtime is not None:
        assignments = runtime.get("assignments", {})
        missing = [
            str(row.get("uuid") or row.get("item_id"))
            for row in result
            if str(row.get("uuid") or row.get("item_id")) not in assignments
        ]
        if missing:
            preview = ", ".join(missing[:3])
            raise ValueError(
                f"Split manifest does not cover {len(missing)} dataset items: {preview}"
            )
        for row in result:
            item_id = str(row.get("uuid") or row.get("item_id"))
            row["split"] = assignments[item_id]
        return result
    return assign_missing_splits(
        result,
        float(config["data"].get("validation_split", 0.2)),
        float(config["data"].get("test_split", 0.1)),
        int(config["run"].get("seed", 123)),
    )
