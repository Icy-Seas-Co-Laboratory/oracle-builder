from __future__ import annotations

import json

import pytest

tf = pytest.importorskip("tensorflow")

from oracle_builder.evaluation.performance import benchmark_model
from oracle_builder.registry import get_model_builder


def test_performance_benchmark_records_latency_throughput_and_parameters(tmp_path):
    config = {
        "run": {"task": "classification", "model": "simple_cnn"},
        "data": {"input_shape": [8, 8, 1], "num_classes": 2},
        "model": {"base_filters": 2, "dropout": 0.0, "embedding_dim": 4},
        "evaluation": {
            "benchmark": {
                "enabled": True,
                "warmup_batches": 1,
                "measured_batches": 2,
            }
        },
    }
    model = get_model_builder("simple_cnn")(config)

    report = benchmark_model(
        model,
        config,
        batch_size=2,
        output_dir=tmp_path,
        evaluation_pipeline_seconds=1.25,
    )

    assert report["model_parameters"]["total"] == model.count_params()
    assert report["benchmark"]["measured_samples"] == 4
    assert report["benchmark"]["throughput_samples_per_second"] > 0
    assert report["benchmark"]["batch_latency_ms"]["p95"] > 0
    saved = json.loads(
        (tmp_path / "evaluation" / "performance.json").read_text()
    )
    assert saved["evaluation_pipeline_seconds"] == 1.25
