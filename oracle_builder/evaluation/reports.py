from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from oracle_builder.data.sqlite_dataset import load_arrays
from oracle_builder.evaluation.classification import evaluate_classification
from oracle_builder.evaluation.segmentation import evaluate_segmentation


def _class_names(config: dict[str, Any]) -> dict[int, str]:
    return {
        int(row["class_index"]): str(row.get("name") or row["class_index"])
        for row in config.get("dataset", {}).get("labels", [])
    }


def evaluate_run_model(
    model,
    config: dict[str, Any],
    input_path: str | Path,
    run_dir: str | Path,
    split: str = "test",
    *,
    inference_batch_size: int | None = None,
) -> dict[str, Any]:
    evaluation_started = time.perf_counter()
    if (
        config["run"]["task"] == "classification"
        and config.get("data", {}).get("streaming", {}).get("enabled", True)
    ):
        from oracle_builder.data.sqlite_stream import (
            SQLiteClassificationSource,
            build_classification_index,
        )
        from oracle_builder.evaluation.classification import (
            evaluate_classification_streaming,
        )

        index = build_classification_index(
            input_path, config, split, labeled_only=True
        )
        if not index.refs and split == "test":
            index = build_classification_index(
                input_path, config, "validation", labeled_only=True
            )
        source = SQLiteClassificationSource(input_path, config)
        if inference_batch_size is None:
            from oracle_builder.inference.batching import (
                resolve_inference_batch_size,
            )

            inference_batch_size = resolve_inference_batch_size(
                model, config
            ).batch_size
        result = evaluate_classification_streaming(
            model,
            source.indexed_image_dataset(
                index, batch_size=inference_batch_size
            ),
            index,
            run_dir,
            class_names=_class_names(config),
            progress=bool(config.get("inference", {}).get("progress", True)),
        )
    else:
        try:
            x, y, records = load_arrays(input_path, config, split=split)
        except ValueError:
            x, y, records = load_arrays(input_path, config, split="validation")
        if config["run"]["task"] == "classification":
            result = evaluate_classification(
                model,
                x,
                y,
                records,
                run_dir,
                class_names=_class_names(config),
                batch_size=inference_batch_size,
                progress=bool(
                    config.get("inference", {}).get("progress", True)
                ),
            )
        else:
            threshold = float(
                config.get("evaluation", {}).get(
                    "segmentation_threshold", 0.5
                )
            )
            result = evaluate_segmentation(
                model,
                x,
                y,
                records,
                run_dir,
                threshold=threshold,
                config=config,
                batch_size=inference_batch_size,
            )
    if inference_batch_size is None:
        from oracle_builder.inference.batching import resolve_inference_batch_size

        inference_batch_size = resolve_inference_batch_size(
            model, config
        ).batch_size
    from oracle_builder.evaluation.performance import benchmark_model

    print("Benchmarking model inference...", flush=True)
    performance = benchmark_model(
        model,
        config,
        batch_size=inference_batch_size,
        output_dir=run_dir,
        evaluation_pipeline_seconds=time.perf_counter() - evaluation_started,
    )
    result["performance"] = performance
    result.setdefault("summary", {})["performance"] = performance.get("benchmark")
    summary_path = Path(run_dir) / "evaluation" / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(result["summary"], indent=2) + "\n"
    )
    return result
