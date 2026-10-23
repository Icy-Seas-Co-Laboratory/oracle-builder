from __future__ import annotations

import json
import time

import numpy as np
import pytest

pytest.importorskip("sklearn")

from oracle_builder.posthoc.cache import export_embedding_cache
from oracle_builder.posthoc.training import _probabilities, _report_long_step, fit_posthoc_from_caches


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


@pytest.mark.parametrize("estimator", ["logistic_regression", "linear_svm", "gradient_boosting", "shallow_mlp"])
def test_posthoc_supports_probability_estimators_and_pca(tmp_path, estimator):
    train = _cache(tmp_path / "train", [[-2, -2], [-1, -1], [-1.5, -1.5], [1, 1], [2, 2], [1.5, 1.5]], [0, 0, 0, 1, 1, 1])
    validation = _cache(tmp_path / "validation", [[-1.2, -1.2], [1.2, 1.2]], [0, 1])
    test = _cache(tmp_path / "test", [[-1.8, -1.8], [1.8, 1.8]], [0, 1])
    report = fit_posthoc_from_caches(train, validation, test, tmp_path / "posthoc", settings={
        "estimator": estimator, "pca_enabled": True, "pca_components": 1, "tune": False,
        "classifier_options": (
            {"max_iter": 50} if estimator == "shallow_mlp" else
            {"n_estimators": 10} if estimator == "gradient_boosting" else {}
        ),
    })
    assert report["preprocessing"]["output_dimension"] == 1
    assert report["calibration"]["method"] == ("sigmoid" if estimator == "linear_svm" else "native")
    import joblib
    from oracle_builder.posthoc.head import posthoc_output
    restored = joblib.load(tmp_path / "posthoc" / "estimator.joblib")
    prediction = posthoc_output(restored, np.asarray([1.8, 1.8]))
    assert prediction["classes"] == [0, 1]
    assert len(prediction["probabilities"]) == 2
    assert sum(prediction["probabilities"]) == pytest.approx(1.0)
    assert prediction["confidence"] == prediction["probabilities"][prediction["class_index"]]
    assert json.loads((tmp_path / "posthoc" / "report.json").read_text()) == report


def test_linear_svm_is_calibrated_and_requires_enough_folds(tmp_path):
    train = _cache(tmp_path / "train", [[-2], [-1], [-1.5], [1], [2], [1.5]], [0, 0, 0, 1, 1, 1])
    validation = _cache(tmp_path / "validation", [[-1.2], [1.2]], [0, 1])
    test = _cache(tmp_path / "test", [[-1.8], [1.8]], [0, 1])
    report = fit_posthoc_from_caches(train, validation, test, tmp_path / "posthoc", settings={"estimator": "linear_svm", "tune": False})
    assert report["calibration"] == {"method": "sigmoid", "folds": 3}


def test_probability_alignment_preserves_missing_class_columns():
    class Estimator:
        classes_ = np.asarray([0, 2])

        def predict_proba(self, values):
            return np.tile([0.25, 0.75], (len(values), 1))

    assert _probabilities(Estimator(), np.zeros((2, 1)), 3).tolist() == [[0.25, 0.0, 0.75]] * 2


def test_posthoc_rejects_pca_larger_than_training_data(tmp_path):
    train = _cache(tmp_path / "train", [[-1], [1]], [0, 1])
    validation = _cache(tmp_path / "validation", [[-1], [1]], [0, 1])
    with pytest.raises(ValueError, match="PCA limit"):
        fit_posthoc_from_caches(train, validation, validation, tmp_path / "posthoc", settings={"pca_enabled": True, "pca_components": 2})


def test_selection_macro_f1_counts_classes_absent_from_validation(tmp_path):
    train = _cache(tmp_path / "train", [[-2], [-1], [-1.5], [1], [2], [1.5]], [0, 0, 0, 1, 1, 1])
    validation = _cache(tmp_path / "validation", [[-1.2]], [0])
    test = _cache(tmp_path / "test", [[-1.8], [1.8]], [0, 1])
    report = fit_posthoc_from_caches(train, validation, test, tmp_path / "posthoc", class_names={0: "zero", 1: "one"})
    assert report["validation_metrics"]["macro_f1"] == max(item["validation_macro_f1"] for item in report["candidates"])


def test_tuned_mlp_report_is_json_round_trippable(tmp_path):
    train = _cache(tmp_path / "train", [[-2], [-1], [-1.5], [1], [2], [1.5]], [0, 0, 0, 1, 1, 1])
    validation = _cache(tmp_path / "validation", [[-1.2], [1.2]], [0, 1])
    test = _cache(tmp_path / "test", [[-1.8], [1.8]], [0, 1])
    report = fit_posthoc_from_caches(train, validation, test, tmp_path / "posthoc", settings={
        "estimator": "shallow_mlp", "classifier_options": {"max_iter": 10},
    })
    assert json.loads((tmp_path / "posthoc" / "report.json").read_text()) == report
