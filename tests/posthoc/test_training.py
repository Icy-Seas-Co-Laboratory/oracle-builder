from __future__ import annotations

import json
import time

import numpy as np
import pytest

pytest.importorskip("sklearn")

from oracle_builder.posthoc.cache import export_embedding_cache
from oracle_builder.posthoc.training import _report_long_step, fit_posthoc_from_caches


def test_long_fit_reports_elapsed_time_without_advancing_fraction():
    events = []
    with _report_long_step(
        lambda event, message, data: events.append((event, message, data)),
        "posthoc_candidate_running", "Fitting classifier candidate 1 of 3",
        {"stage_fraction": 0.75}, interval_seconds=0.01,
    ):
        time.sleep(0.04)
    assert events
    assert all(event == "posthoc_candidate_running" and "elapsed" in message
               and data["stage_fraction"] == 0.75 for event, message, data in events)


def _cache(path, features, labels):
    export_embedding_cache(
        path, np.asarray(features, dtype="float32"),
        records=[{"label": int(label)} for label in labels], prefer_parquet=False,
    )
    return path


def test_posthoc_tunes_on_validation_and_records_held_out_metrics(tmp_path):
    train = _cache(tmp_path / "train", [[-2], [-1], [-1.5], [1], [2], [1.5]], [0, 0, 0, 1, 1, 1])
    validation = _cache(tmp_path / "validation", [[-1.2], [1.2]], [0, 1])
    test = _cache(tmp_path / "test", [[-1.8], [1.8]], [0, 1])

    events = []
    report = fit_posthoc_from_caches(
        train, validation, test, tmp_path / "posthoc",
        progress=lambda event, message, data: events.append((event, message, data)),
    )

    assert report["selected_C"] in {0.1, 1.0, 10.0}
    assert {key: report["metrics"][key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "micro_f1")} == {
        "accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0, "micro_f1": 1.0,
    }
    assert {"log_loss", "weighted_f1", "matthews_correlation_coefficient"} <= report["metrics"].keys()
    assert report["classes"] == [0, 1]
    assert report["sample_counts"] == {"train": 6, "validation": 2, "test": 2}
    assert json.loads((tmp_path / "posthoc" / "report.json").read_text()) == report
    assert (tmp_path / "posthoc" / "estimator.joblib").is_file()
    assert (tmp_path / "posthoc" / "evaluation" / "classification_report.json").is_file()
    assert (tmp_path / "posthoc" / "evaluation" / "confusion_matrix.csv").is_file()
    assert (tmp_path / "posthoc" / "evaluation" / "metrics_long.csv").is_file()
    assert [event for event, _, _ in events if event == "posthoc_candidate_completed"] == [
        "posthoc_candidate_completed"
    ] * 3
    assert events[-1][0] == "posthoc_evaluation_completed"
    assert events[-1][2]["stage_fraction"] == 1.0


def test_posthoc_rejects_missing_labels(tmp_path):
    train = _cache(tmp_path / "train", [[-1], [1]], [0, 1])
    invalid = tmp_path / "validation"
    export_embedding_cache(invalid, np.asarray([[-1], [1]], dtype="float32"), records=[{}, {}], prefer_parquet=False)
    with pytest.raises(ValueError, match="class labels"):
        fit_posthoc_from_caches(train, invalid, invalid, tmp_path / "posthoc")
