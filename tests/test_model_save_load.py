from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from oracle_builder.saving.load_test import run_load_tests
from oracle_builder.saving.save_model import save_model_artifacts, write_load_test_report
from oracle_builder.training.train import build_and_compile_model


def test_tiny_model_save_and_reload(tmp_path: Path):
    config = {
        "run": {"task": "classification", "model": "simple_cnn", "seed": 123},
        "data": {"input_shape": [16, 16, 1], "num_classes": 2},
        "model": {"base_filters": 4, "dropout": 0.0},
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"],
        },
        "output": {"export_savedmodel": True},
    }
    (tmp_path / "model").mkdir()
    model = build_and_compile_model(config)
    x = np.zeros((2, 16, 16, 1), dtype="float32")
    y = np.array([0, 1], dtype="int64")
    model.fit(x, y, epochs=1, verbose=0)
    save_report = save_model_artifacts(model, tmp_path, config)
    report = run_load_tests(tmp_path, config, save_report)
    write_load_test_report(tmp_path, report)

    assert report["final_keras_reloaded"]
    assert report["weights_reloaded_into_rebuilt_model"]
    assert report["prediction_test_passed"]
    assert (tmp_path / "model" / "load_test_report.json").exists()

