from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


def _synchronize(outputs: Any) -> None:
    for value in tf.nest.flatten(outputs):
        if hasattr(value, "numpy"):
            value.numpy()


def _parameter_counts(model) -> tuple[int | None, int | None, int | None]:
    try:
        total = int(model.count_params())
        trainable = int(
            sum(np.prod(variable.shape) for variable in model.trainable_weights)
        )
        non_trainable = total - trainable
        return total, trainable, non_trainable
    except (AttributeError, ValueError):
        return None, None, None


def benchmark_model(
    model,
    config: dict[str, Any],
    *,
    batch_size: int,
    output_dir: str | Path,
    evaluation_pipeline_seconds: float | None = None,
) -> dict[str, Any]:
    settings = config.get("evaluation", {}).get("benchmark", {})
    enabled = bool(settings.get("enabled", True))
    output = Path(output_dir) / "evaluation" / "performance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    total_params, trainable_params, non_trainable_params = _parameter_counts(model)
    report: dict[str, Any] = {
        "schema_name": "oracle_builder.performance",
        "schema_version": "1.0.0",
        "enabled": enabled,
        "task": config.get("run", {}).get("task"),
        "architecture": config.get("run", {}).get("model"),
        "input_shape": list(config["data"]["input_shape"]),
        "batch_size": int(batch_size),
        "model_parameters": {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": non_trainable_params,
        },
        "evaluation_pipeline_seconds": evaluation_pipeline_seconds,
        "tensorflow_version": tf.__version__,
        "physical_devices": [
            {"type": device.device_type, "name": device.name}
            for device in tf.config.list_physical_devices()
        ],
        "logical_devices": [
            {"type": device.device_type, "name": device.name}
            for device in tf.config.list_logical_devices()
        ],
    }
    if not enabled:
        output.write_text(json.dumps(report, indent=2) + "\n")
        return report

    warmup = max(0, int(settings.get("warmup_batches", 2)))
    measured = max(1, int(settings.get("measured_batches", 10)))
    sample = tf.zeros(
        [int(batch_size), *config["data"]["input_shape"]],
        dtype=tf.float32,
    )
    gpu_names = [
        device.name.replace("/physical_device:", "")
        for device in tf.config.list_physical_devices("GPU")
    ]
    for name in gpu_names:
        try:
            tf.config.experimental.reset_memory_stats(name)
        except (ValueError, RuntimeError):
            pass
    for _ in range(warmup):
        _synchronize(model(sample, training=False))
    latencies = []
    for _ in range(measured):
        started = time.perf_counter()
        _synchronize(model(sample, training=False))
        latencies.append(time.perf_counter() - started)
    values = np.asarray(latencies, dtype="float64")
    total_samples = measured * int(batch_size)
    total_seconds = float(values.sum())
    report["benchmark"] = {
        "warmup_batches": warmup,
        "measured_batches": measured,
        "measured_samples": total_samples,
        "total_seconds": total_seconds,
        "throughput_samples_per_second": (
            total_samples / total_seconds if total_seconds else None
        ),
        "batch_latency_ms": {
            "mean": float(values.mean() * 1000),
            "median": float(np.median(values) * 1000),
            "p95": float(np.percentile(values, 95) * 1000),
            "minimum": float(values.min() * 1000),
            "maximum": float(values.max() * 1000),
        },
        "approximate_sample_latency_ms": float(
            np.median(values) * 1000 / int(batch_size)
        ),
    }
    memory = {}
    for name in gpu_names:
        try:
            values_for_device = tf.config.experimental.get_memory_info(name)
            memory[name] = {
                "current_bytes": int(values_for_device["current"]),
                "peak_bytes": int(values_for_device["peak"]),
            }
        except (ValueError, RuntimeError):
            continue
    report["device_memory"] = memory
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
