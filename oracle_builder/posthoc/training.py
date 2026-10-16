"""Fit and evaluate the configured downstream classifier on sealed split embeddings."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Mapping

import numpy as np

from oracle_builder.artifacts.representations import write_representation_artifacts
from oracle_builder.data.sqlite_stream import SQLiteClassificationSource, build_classification_index
from oracle_builder.evaluation.classification import (
    ClassificationMetricAccumulator,
    classification_epoch_metric_records,
    write_classification_evaluation,
)

from .cache import open_embedding_cache
from .pipelines import build_posthoc_pipeline


PosthocProgress = Callable[[str, str, Mapping[str, Any] | None], None]


def _emit(progress: PosthocProgress | None, event: str, message: str, data: Mapping[str, Any] | None = None) -> None:
    if progress is not None:
        progress(event, message, data)


@contextmanager
def _report_long_step(progress: PosthocProgress | None, event: str, description: str,
                      data: Mapping[str, Any], *, interval_seconds: float = 15.0):
    """Keep a long-running sklearn operation visible without claiming measured progress."""
    if progress is None:
        yield
        return
    finished = threading.Event()
    started = time.monotonic()

    def heartbeat() -> None:
        while not finished.wait(interval_seconds):
            _emit(progress, event, f"{description} · {int(time.monotonic() - started)}s elapsed", data)

    reporter = threading.Thread(target=heartbeat, name="posthoc-progress", daemon=True)
    reporter.start()
    try:
        yield
    finally:
        finished.set()
        reporter.join()


def _labeled_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cache = open_embedding_cache(path)
    labels = [row.get("label") for row in cache.iter_records()]
    if len(labels) != cache.shape[0] or any(label is None for label in labels):
        raise ValueError(f"Post-hoc cache lacks complete class labels: {path}")
    return cache.embeddings, np.asarray(labels, dtype="int64")


def fit_posthoc_from_caches(
    train_path: Path, validation_path: Path, test_path: Path, output: Path,
    *, representation: str = "fused_embedding", estimator: str = "logistic_regression",
    class_names: dict[int, str] | None = None,
    progress: PosthocProgress | None = None,
) -> dict[str, Any]:
    """Tune on validation only, then score once on the held-out test split."""
    import joblib
    from sklearn.metrics import f1_score

    if estimator != "logistic_regression":
        raise ValueError(f"Unsupported configured post-hoc estimator: {estimator}")
    train_x, train_y = _labeled_data(train_path)
    val_x, val_y = _labeled_data(validation_path)
    test_x, test_y = _labeled_data(test_path)
    if len(np.unique(train_y)) < 2:
        raise ValueError("Post-hoc logistic regression requires at least two training classes")
    if not (train_x.shape[1] == val_x.shape[1] == test_x.shape[1]):
        raise ValueError("Post-hoc representation dimensions differ between splits")

    candidates = []
    best = None
    strengths = (0.1, 1.0, 10.0)
    for candidate_index, strength in enumerate(strengths, start=1):
        _emit(progress, "posthoc_candidate_started", f"Fitting classifier candidate {candidate_index} of {len(strengths)} (C={strength:g})", {
            "candidate_index": candidate_index, "candidate_count": len(strengths), "C": strength,
            "stage_fraction": 0.75 + 0.2 * (candidate_index - 1) / len(strengths),
        })
        pipeline = build_posthoc_pipeline({
            "transforms": ("standard",),
            "classifier": estimator,
            "classifier_options": {"C": strength, "max_iter": 1000},
        })
        with _report_long_step(progress, "posthoc_candidate_running",
                               f"Fitting classifier candidate {candidate_index} of {len(strengths)}",
                               {"candidate_index": candidate_index, "candidate_count": len(strengths),
                                "C": strength, "stage_fraction": 0.75 + 0.2 * (candidate_index - 1) / len(strengths)}):
            pipeline.fit(train_x, train_y)
        prediction = pipeline.predict(val_x)
        score = float(f1_score(val_y, prediction, average="macro", zero_division=0))
        candidates.append({"C": strength, "validation_macro_f1": score})
        _emit(progress, "posthoc_candidate_completed", f"Candidate {candidate_index} of {len(strengths)} scored {score:.4f} validation macro F1", {
            "candidate_index": candidate_index, "candidate_count": len(strengths), "C": strength,
            "validation_macro_f1": score, "stage_fraction": 0.75 + 0.2 * candidate_index / len(strengths),
        })
        if best is None or score > best[0]:
            best = (score, strength, pipeline)
    assert best is not None
    _emit(progress, "posthoc_evaluation_started", "Calculating validation and held-out test metrics", {
        "selected_C": best[1], "stage_fraction": 0.95,
    })
    validation_probabilities = best[2].predict_proba(val_x)
    validation_metrics = {
        row["metric"]: row["value"]
        for row in classification_epoch_metric_records(val_y, validation_probabilities, class_names=class_names)
        if "label" not in row and "bin_index" not in row
        and row.get("value") is not None and np.isfinite(row["value"])
    }
    probabilities = best[2].predict_proba(test_x)
    metrics = {
        row["metric"]: row["value"]
        for row in classification_epoch_metric_records(test_y, probabilities, class_names=class_names)
        if "label" not in row and "bin_index" not in row
        and row.get("value") is not None and np.isfinite(row["value"])
    }
    output.mkdir(parents=True, exist_ok=True)
    accumulator = ClassificationMetricAccumulator(probabilities.shape[1])
    accumulator.update(test_y, probabilities)
    test_records = list(open_embedding_cache(test_path).iter_records())
    sample_rows = [
        {**record, "y_true": int(target), "y_pred": int(predicted)}
        for record, target, predicted in zip(
            test_records, test_y, np.argmax(probabilities, axis=1), strict=True
        )
    ]
    write_classification_evaluation(
        test_y,
        np.argmax(probabilities, axis=1),
        sample_rows,
        output,
        class_names=class_names,
        probability_metrics=accumulator.result(),
        probabilities=probabilities,
        calibration_rows=accumulator.calibration_rows(),
        evaluation_context={"split": "test", "head": "posthoc"},
    )
    report = {
        "status": "complete",
        "estimator": estimator,
        "representation": representation,
        "training_split": "train",
        "selection_split": "validation",
        "evaluation_split": "test",
        "selection_metric": "macro_f1",
        "selected_C": best[1],
        "candidates": candidates,
        "sample_counts": {"train": len(train_y), "validation": len(val_y), "test": len(test_y)},
        "validation_metrics": validation_metrics,
        "metrics": metrics,
        "classes": [int(value) for value in best[2].classes_],
        "class_names": {str(key): value for key, value in (class_names or {}).items()},
        "input_dimension": int(train_x.shape[1]),
        "model_path": "model/posthoc/estimator.joblib",
    }
    best[2].oracle_posthoc_labels_ = dict(class_names or {})
    joblib.dump(best[2], output / "estimator.joblib")
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _emit(progress, "posthoc_evaluation_completed", "Classifier metrics and held-out test evidence are ready", {
        "selected_C": best[1], "validation_macro_f1": best[0], "stage_fraction": 1.0,
    })
    return report


def train_posthoc_classifier(model: Any, sqlite_path: str | Path, config: dict[str, Any], run_dir: str | Path,
                             *, progress: PosthocProgress | None = None) -> dict[str, Any]:
    """Export only the configured representation for each labeled split."""
    settings = config.get("posthoc", {})
    representation = str(settings.get("representation", "fused_embedding"))
    estimator = str(settings.get("estimator", "logistic_regression"))
    root = Path(run_dir)
    source = SQLiteClassificationSource(sqlite_path, config)
    class_names = {
        int(row["class_index"]): str(row["name"])
        for row in config.get("dataset", {}).get("labels", [])
    }
    cache_paths = {}
    splits = ("train", "validation", "test")
    for split_index, split in enumerate(splits):
        index = build_classification_index(sqlite_path, config, split, labeled_only=True)
        if not index.refs:
            raise ValueError(f"Enabled post-hoc classifier requires a labeled {split} split")
        cache_path = root / "intermediates" / "representations" / split / representation
        _emit(progress, "posthoc_embedding_started", f"Extracting {split} embeddings from {len(index.refs):,} examples", {
            "split": split, "split_index": split_index + 1, "total_splits": len(splits),
            "completed": 0, "total": len(index.refs), "stage_fraction": 0.75 * split_index / len(splits),
        })
        if cache_path.exists():
            try:
                cached = open_embedding_cache(cache_path)
                labels = [row.get("label") for row in cached.iter_records()]
                valid_cache = len(labels) == len(index.refs) and all(label is not None for label in labels)
            except (OSError, ValueError, KeyError, RuntimeError):
                valid_cache = False
            if not valid_cache:
                shutil.rmtree(cache_path)
        if not (cache_path / "manifest.json").is_file():
            export_config = {
                **config,
                "output": {
                    **config.get("output", {}),
                    "intermediate_artifacts": {
                        "enabled": True,
                        "representations": [representation],
                        "batch_size": int(config.get("output", {}).get("intermediate_artifacts", {}).get("batch_size", 256)),
                    },
                },
            }
            last_reported_at = 0.0

            def report_embedding(completed: int, total: int) -> None:
                nonlocal last_reported_at
                now = time.monotonic()
                if completed != total and now - last_reported_at < 2.0:
                    return
                last_reported_at = now
                _emit(progress, "posthoc_embedding_progress", f"Extracted {completed:,} of {total:,} {split} embeddings", {
                    "split": split, "split_index": split_index + 1, "total_splits": len(splits),
                    "completed": completed, "total": total,
                    "stage_fraction": 0.75 * (split_index + completed / total) / len(splits),
                })

            write_representation_artifacts(
                model,
                source.indexed_image_dataset(index, batch_size=export_config["output"]["intermediate_artifacts"]["batch_size"]),
                index, export_config, root, progress=report_embedding,
            )
        else:
            _emit(progress, "posthoc_embedding_progress", f"Reusing {len(index.refs):,} cached {split} embeddings", {
                "split": split, "split_index": split_index + 1, "total_splits": len(splits),
                "completed": len(index.refs), "total": len(index.refs),
                "stage_fraction": 0.75 * (split_index + 1) / len(splits),
            })
        cache_paths[split] = cache_path
    report = fit_posthoc_from_caches(
        cache_paths["train"], cache_paths["validation"], cache_paths["test"],
        root / "model" / "posthoc", representation=representation, estimator=estimator,
        class_names=class_names, progress=progress,
    )
    # The neural outputs remain the primary contract; this adds an optional
    # head to the same portable bundle after fitting has completed.
    manifest_path = root / "model" / "model_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("outputs", {})["posthoc"] = {
            "type": "classification",
            "representation": representation,
            "estimator": estimator,
            "path": report["model_path"],
            "classes": report["classes"],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for contract_path in (root / "contract.json", root / "model" / "contract.json"):
            if contract_path.is_file():
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
                contract.setdefault("outputs", {})["posthoc"] = manifest["outputs"]["posthoc"]
                contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
