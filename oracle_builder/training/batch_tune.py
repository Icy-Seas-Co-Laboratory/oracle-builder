"""Bounded, isolated training-batch calibration for a sealed run config.

This module is deliberately a short-lived process: TensorFlow can retain GPU
allocations after an OOM, so trying several batch sizes inside the compute API
process would make its later admission decisions less trustworthy.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from typing import Any


_RESULT_PREFIX = "ORACLE_BATCH_TUNE_RESULT="


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a safe Oracle Builder training batch size.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--maximum", required=True, type=int)
    parser.add_argument("--minimum", type=int, default=1)
    parser.add_argument("--safety-factor", type=float, default=0.8)
    return parser.parse_args()


def _is_memory_error(error: BaseException) -> bool:
    message = str(error).lower()
    return any(token in message for token in ("resourceexhausted", "out of memory", "oom", "cuda_error_out_of_memory"))


def _first_tensor(value: Any):
    import tensorflow as tf

    tensors = tf.nest.flatten(value)
    if not tensors:
        raise ValueError("The model did not produce a tensor output")
    return tf.cast(tensors[0], tf.float32)


def _probe(config: dict[str, Any], input_path: str, batch_size: int) -> dict[str, Any]:
    """Run one genuine forward/backward pass without updating model weights."""
    import tensorflow as tf

    from oracle_builder.data.sqlite_dataset import make_tf_datasets
    from oracle_builder.registry import get_model_builder
    from oracle_builder.training.distribution import select_distribution_strategy

    candidate = copy.deepcopy(config)
    candidate.setdefault("data", {})["batch_size"] = batch_size
    datasets, _records = make_tf_datasets(input_path, candidate)
    batch = next(iter(datasets["train"]))
    inputs = batch[0] if isinstance(batch, tuple) else batch
    first_input = _first_tensor(inputs)
    observed = int(tf.shape(first_input)[0].numpy())
    if observed < batch_size:
        return {"status": "dataset_limit", "observed_batch_size": observed}
    strategy, _distribution = select_distribution_strategy(candidate)
    with strategy.scope():
        model = get_model_builder(candidate["run"]["model"])(candidate)
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            # A scalar over model output intentionally exercises activations
            # and gradients without mutating optimizer state or model weights.
            loss = tf.reduce_mean(_first_tensor(outputs))
        gradients = tape.gradient(loss, model.trainable_variables)
    materialized = [tf.reduce_sum(gradient) for gradient in gradients if gradient is not None]
    if materialized:
        tf.add_n(materialized).numpy()
    del model, datasets, batch, inputs, gradients, materialized, strategy
    tf.keras.backend.clear_session()
    gc.collect()
    return {"status": "passed", "observed_batch_size": observed}


def tune(
    config_path: str | Path,
    input_path: str | Path,
    *,
    minimum: int = 1,
    maximum: int = 256,
    safety_factor: float = 0.8,
) -> dict[str, Any]:
    if minimum < 1 or maximum < minimum:
        raise ValueError("Batch calibration bounds must be positive and ordered")
    if not 0 < safety_factor <= 1:
        raise ValueError("Batch calibration safety factor must be in (0, 1]")
    from oracle_builder.config import resolve_config

    # Resolution gives the probe exactly the V2 configuration that training
    # would execute, including server-derived dataset facts and runtime aliases.
    config = resolve_config(config_path, input_path, Path(config_path).parent / ".batch-tune")
    attempts: list[dict[str, Any]] = []
    largest = 0
    candidate = minimum
    failure_ceiling: int | None = None
    while candidate <= maximum:
        try:
            result = _probe(config, str(input_path), candidate)
        except Exception as error:
            if not _is_memory_error(error):
                raise RuntimeError(f"Batch probe failed before memory admission: {type(error).__name__}: {error}") from error
            attempts.append({"batch_size": candidate, "status": "out_of_memory"})
            failure_ceiling = candidate
            break
        attempts.append({"batch_size": candidate, **result})
        largest = max(largest, int(result["observed_batch_size"]))
        if result["status"] == "dataset_limit":
            break
        candidate *= 2

    if largest < 1:
        raise RuntimeError("No batch size completed a forward/backward probe")
    if failure_ceiling is not None and largest + 1 < failure_ceiling:
        low, high = largest + 1, failure_ceiling - 1
        while low <= high:
            candidate = (low + high) // 2
            try:
                result = _probe(config, str(input_path), candidate)
            except Exception as error:
                if not _is_memory_error(error):
                    raise RuntimeError(f"Batch probe failed before memory admission: {type(error).__name__}: {error}") from error
                attempts.append({"batch_size": candidate, "status": "out_of_memory"})
                high = candidate - 1
                continue
            attempts.append({"batch_size": candidate, **result})
            largest = max(largest, int(result["observed_batch_size"]))
            low = candidate + 1

    recommended = max(minimum, min(largest, int(largest * safety_factor)))
    if recommended != largest:
        # Verify the conservative value too; the evidence must cover the batch
        # size we actually seal, not only a larger neighbouring candidate.
        try:
            result = _probe(config, str(input_path), recommended)
        except Exception as error:
            if not _is_memory_error(error):
                raise RuntimeError(f"Recommended batch probe failed: {type(error).__name__}: {error}") from error
            recommended = largest
            result = {"status": "fallback_to_largest"}
        attempts.append({"batch_size": recommended, **result})
    return {
        "ready": True,
        "probe_kind": "representative_forward_backward",
        "recommended_batch_size": recommended,
        "largest_verified_batch_size": largest,
        "safety_factor": safety_factor,
        "attempts": attempts,
    }


def main() -> int:
    args = _arguments()
    try:
        result = tune(
            args.config, args.input,
            minimum=args.minimum, maximum=args.maximum,
            safety_factor=args.safety_factor,
        )
    except Exception as error:
        result = {"ready": False, "reasons": [f"{type(error).__name__}: {error}"]}
    print(_RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
