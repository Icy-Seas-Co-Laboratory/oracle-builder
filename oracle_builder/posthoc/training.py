"""Fit and evaluate the configured downstream classifier on sealed split embeddings."""

from __future__ import annotations

import json
import hashlib
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
from .pipelines import (
    build_posthoc_pipeline,
    normalize_posthoc_settings,
    pipeline_config_from_settings,
)


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


def _candidate_options(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Small, deterministic validation grids, with explicit user values pinned."""
    base = dict(settings["classifier_options"])
    if not settings["tune"]:
        return [base]
    grids = {
        "logistic_regression": {"C": (0.1, 1.0, 10.0)},
        "linear_svm": {"C": (0.1, 1.0, 10.0)},
        "gradient_boosting": {"n_estimators": (100, 250), "learning_rate": (0.05, 0.1)},
        "shallow_mlp": {"hidden_layer_sizes": ([32], [64]), "alpha": (0.0001, 0.001)},
    }
    varying = [(key, choices) for key, choices in grids[settings["estimator"]].items() if key not in base]
    if not varying:
        return [base]
    # Tune one dimension at a time to bound interactive training cost.  The
    # first grid supplies the baseline, preserving the legacy three-C search.
    first_key, first_choices = varying[0]
    candidates = [{**base, first_key: choice} for choice in first_choices]
    for key, choices in varying[1:]:
        candidates.extend({**base, key: choice} for choice in choices[1:])
    unique: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _probabilities(estimator: Any, values: np.ndarray, class_count: int) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(values), dtype="float64")
    result = np.zeros((len(values), class_count), dtype="float64")
    for column, label in enumerate(estimator.classes_):
        label = int(label)
        if label < 0 or label >= class_count:
            raise ValueError(f"Post-hoc class index {label} is outside the configured output")
        result[:, label] = raw[:, column]
    return result


def _preprocessing_report(estimator: Any, settings: Mapping[str, Any], input_dimension: int) -> dict[str, Any]:
    result: dict[str, Any] = {"scaling": settings["scaling"], "input_dimension": input_dimension,
                              "pca_enabled": settings["pca_enabled"]}
    if not settings["pca_enabled"]:
        result["output_dimension"] = input_dimension
        return result
    if settings["estimator"] == "linear_svm":
        folds = []
        for calibrated in estimator.calibrated_classifiers_:
            pipeline = getattr(calibrated, "estimator", getattr(calibrated, "base_estimator", None))
            if pipeline is None:
                raise RuntimeError("Could not inspect calibrated linear SVM preprocessing")
            pca = next((step for name, step in pipeline.named_steps.items() if name.startswith("pca_")), None)
            if pca is None:
                raise RuntimeError("Configured PCA was not fitted in the post-hoc pipeline")
            folds.append(np.asarray(pca.explained_variance_ratio_, dtype="float64"))
        ratios = np.stack(folds)
        result.update({"output_dimension": int(ratios.shape[1]),
                       "explained_variance_ratio": [float(item) for item in ratios.mean(axis=0)],
                       "explained_variance": float(ratios.sum(axis=1).mean()),
                       "pca_fit": "per_calibration_fold",
                       "per_fold_explained_variance": [float(row.sum()) for row in ratios]})
        return result
    pipeline = estimator
    pca = next((step for name, step in pipeline.named_steps.items() if name.startswith("pca_")), None)
    if pca is None:
        raise RuntimeError("Configured PCA was not fitted in the post-hoc pipeline")
    result.update({"output_dimension": int(pca.n_components_),
                   "explained_variance_ratio": [float(item) for item in pca.explained_variance_ratio_],
                   "explained_variance": float(np.sum(pca.explained_variance_ratio_))})
    return result


def fit_posthoc_from_caches(
    train_path: Path, validation_path: Path, test_path: Path, output: Path,
    *, representation: str = "fused_embedding", estimator: str = "logistic_regression",
    settings: Mapping[str, Any] | None = None,
    dataset_fingerprint_sha256: str | None = None,
    class_names: dict[int, str] | None = None,
    progress: PosthocProgress | None = None,
) -> dict[str, Any]:
    """Tune on validation only, then score once on the held-out test split."""
    import joblib
    from sklearn.metrics import f1_score

    supplied = dict(settings or {})
    if "estimator" in supplied and estimator != "logistic_regression" and supplied["estimator"] != estimator:
        raise ValueError("estimator conflicts with settings.estimator")
    supplied.setdefault("estimator", estimator)
    supplied.setdefault("representation", representation)
    settings = normalize_posthoc_settings(supplied)
    train_x, train_y = _labeled_data(train_path)
    val_x, val_y = _labeled_data(validation_path)
    test_x, test_y = _labeled_data(test_path)
    if len(np.unique(train_y)) < 2:
        raise ValueError(f"Post-hoc {settings['estimator']} requires at least two training classes")
    unseen = set(np.unique(np.concatenate((val_y, test_y)))) - set(np.unique(train_y))
    if unseen:
        raise ValueError(f"Post-hoc training split lacks classes present in validation/test: {sorted(int(value) for value in unseen)}")
    if not (train_x.shape[1] == val_x.shape[1] == test_x.shape[1]):
        raise ValueError("Post-hoc representation dimensions differ between splits")
    class_count = max([int(value) for value in np.concatenate((train_y, val_y, test_y))] + [int(key) for key in (class_names or {})]) + 1
    _, train_class_counts = np.unique(train_y, return_counts=True)
    min_train_class_count = int(train_class_counts.min())

    if settings["pca_enabled"]:
        max_components = min(train_x.shape[0], train_x.shape[1])
        if settings["estimator"] == "linear_svm":
            if min_train_class_count < settings["calibration_folds"]:
                raise ValueError("linear_svm calibration requires at least calibration_folds training examples in every class")
            max_components = min(max_components, train_x.shape[1], train_x.shape[0] - int(np.ceil(train_x.shape[0] / settings["calibration_folds"])))
        if settings["pca_components"] > max_components:
            raise ValueError(f"posthoc.pca_components={settings['pca_components']} exceeds the train-only PCA limit of {max_components}")
    elif settings["estimator"] == "linear_svm" and min_train_class_count < settings["calibration_folds"]:
        raise ValueError("linear_svm calibration requires at least calibration_folds training examples in every class")

    candidates = []
    best = None
    option_candidates = _candidate_options(settings)
    started = time.monotonic()
    for candidate_index, options in enumerate(option_candidates, start=1):
        _emit(progress, "posthoc_candidate_started", f"Fitting classifier candidate {candidate_index} of {len(option_candidates)}", {
            "candidate_index": candidate_index, "candidate_count": len(option_candidates), "parameters": options,
            "stage_fraction": 0.75 + 0.2 * (candidate_index - 1) / len(option_candidates),
        })
        pipeline_settings = {**settings, "classifier_options": options}
        pipeline = build_posthoc_pipeline(pipeline_config_from_settings(pipeline_settings))
        with _report_long_step(progress, "posthoc_candidate_running",
                               f"Fitting classifier candidate {candidate_index} of {len(option_candidates)}",
                               {"candidate_index": candidate_index, "candidate_count": len(option_candidates),
                                "parameters": options, "stage_fraction": 0.75 + 0.2 * (candidate_index - 1) / len(option_candidates)}):
            pipeline.fit(train_x, train_y)
        # Selection uses the same aligned probability argmax as reporting and
        # inference, including when a split lacks an otherwise known class.
        prediction = np.argmax(_probabilities(pipeline, val_x, class_count), axis=1)
        score = float(f1_score(val_y, prediction, labels=list(range(class_count)), average="macro", zero_division=0))
        candidates.append({"parameters": options, "validation_macro_f1": score})
        _emit(progress, "posthoc_candidate_completed", f"Candidate {candidate_index} of {len(option_candidates)} scored {score:.4f} validation macro F1", {
            "candidate_index": candidate_index, "candidate_count": len(option_candidates), "parameters": options,
            "validation_macro_f1": score, "stage_fraction": 0.75 + 0.2 * candidate_index / len(option_candidates),
        })
        if best is None or score > best[0]:
            best = (score, options, pipeline)
    assert best is not None
    _emit(progress, "posthoc_evaluation_started", "Calculating validation and held-out test metrics", {
        "selected_parameters": best[1], "stage_fraction": 0.95,
    })
    validation_probabilities = _probabilities(best[2], val_x, class_count)
    validation_metrics = {
        row["metric"]: row["value"]
        for row in classification_epoch_metric_records(val_y, validation_probabilities, class_names=class_names)
        if "label" not in row and "bin_index" not in row
        and row.get("value") is not None and np.isfinite(row["value"])
    }
    probabilities = _probabilities(best[2], test_x, class_count)
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
        {**record, "y_true": int(target), "y_pred": int(predicted),
         "confidence": float(row[int(predicted)])}
        for record, target, predicted, row in zip(
            test_records, test_y, np.argmax(probabilities, axis=1), probabilities, strict=True
        )
    ]
    signature_payload = {
        "dataset_fingerprint_sha256": dataset_fingerprint_sha256,
        "samples": [{"id": row.get("sample_id", row.get("item_id", row.get("source_key", str(index)))), "label": int(target)}
                     for index, (row, target) in enumerate(zip(test_records, test_y, strict=True))],
        "classes": {str(key): value for key, value in (class_names or {}).items()},
    }
    evaluation_signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
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
        "estimator": settings["estimator"],
        "representation": settings["representation"],
        "training_split": "train",
        "selection_split": "validation",
        "evaluation_split": "test",
        "selection_metric": "macro_f1",
        "selected_parameters": best[1],
        **({"selected_C": best[1]["C"]} if settings["estimator"] == "logistic_regression" and "C" in best[1] else {}),
        "candidates": candidates,
        "sample_counts": {"train": len(train_y), "validation": len(val_y), "test": len(test_y)},
        "validation_metrics": validation_metrics,
        "metrics": metrics,
        "classes": list(range(class_count)),
        "trained_classes": [int(value) for value in best[2].classes_],
        "class_count": class_count,
        "class_names": {str(key): value for key, value in (class_names or {}).items()},
        "input_dimension": int(train_x.shape[1]),
        "preprocessing": _preprocessing_report(best[2], settings, int(train_x.shape[1])),
        "calibration": {"method": "sigmoid", "folds": settings["calibration_folds"]} if settings["estimator"] == "linear_svm" else {"method": "native", "folds": None},
        "fit_seconds": float(time.monotonic() - started),
        "settings": settings,
        "evaluation_signature": evaluation_signature,
        "model_path": "model/posthoc/estimator.joblib",
    }
    best[2].oracle_posthoc_labels_ = dict(class_names or {})
    best[2].oracle_posthoc_class_count_ = class_count
    joblib.dump(best[2], output / "estimator.joblib")
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _emit(progress, "posthoc_evaluation_completed", "Classifier metrics and held-out test evidence are ready", {
        "selected_parameters": best[1], "validation_macro_f1": best[0], "stage_fraction": 1.0,
    })
    return report


def train_posthoc_classifier(model: Any, sqlite_path: str | Path, config: dict[str, Any], run_dir: str | Path,
                             *, progress: PosthocProgress | None = None) -> dict[str, Any]:
    """Export only the configured representation for each labeled split."""
    settings = normalize_posthoc_settings(config.get("posthoc", {}))
    representation = settings["representation"]
    estimator = settings["estimator"]
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
        root / "model" / "posthoc", representation=representation, estimator=estimator, settings=settings,
        dataset_fingerprint_sha256=config.get("dataset", {}).get("fingerprint_sha256"), class_names=class_names, progress=progress,
    )
    report["training_dataset"] = {
        "dataset_id": config.get("dataset", {}).get("dataset_id"),
        "fingerprint_sha256": config.get("dataset", {}).get("fingerprint_sha256"),
        "labels": config.get("dataset", {}).get("labels", []),
    }
    (root / "model" / "posthoc" / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
