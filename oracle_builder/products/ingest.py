"""Ingest externally supplied Keras and TensorFlow SavedModels into products."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from oracle_builder.classification.features import L2Normalization

from oracle_builder.artifacts import (
    RunLayout,
    create_run_artifact,
    create_unavailable_split_manifest,
    seal_run_artifact,
    update_run_artifact,
)
from oracle_builder.config import load_toml
from oracle_builder.datasets.schema import (
    dataset_fingerprint,
    read_dataset_info,
    validate_database,
)
from oracle_builder.environment import write_environment
from oracle_builder.training.logging_callbacks import (
    init_training_log,
    log_event,
    mark_run_complete,
)


SUPPORTED_SUFFIXES = {".keras", ".h5", ".hdf5"}


class _SavedModelClassificationAdapter(tf.Module):
    """Expose a legacy SavedModel through Oracle Builder's stable ABI.

    The source SavedModel remains preserved verbatim.  This small wrapper only
    assigns explicit semantics to its single class-score tensor, avoiding any
    dependency on legacy Keras deserialization at serving time.
    """

    def __init__(self, source_model, source_signature, source_input_name: str, output_name: str, activation: str):
        super().__init__()
        # Keep the loaded module trackable and alive.  ConcreteFunctions use
        # weak variable references and otherwise fail during adapter tracing.
        self.source_model = source_model
        self.source_signature = source_signature
        self.source_input_name = source_input_name
        self.output_name = output_name
        self.activation = activation

    @tf.function
    def serve(self, inputs):
        values = self.source_signature(**{self.source_input_name: inputs})[self.output_name]
        if self.activation == "linear":
            logits = values
            probabilities = tf.nn.softmax(logits, axis=-1)
        else:
            probabilities = values
            # A softmax logit vector is only identifiable up to a constant;
            # log(p) is the canonical, numerically stable representative.
            logits = tf.math.log(tf.clip_by_value(probabilities, 1e-7, 1.0))
        return {"logits": logits, "probabilities": probabilities}

    @tf.function
    def classify(self, inputs):
        return self.serve(inputs)


@keras.utils.register_keras_serializable(package="oracle_builder")
class ProbabilityToLogits(keras.layers.Layer):
    """Derive a stable canonical logit representation from probabilities."""

    def __init__(self, mode: str, **kwargs):
        super().__init__(**kwargs)
        if mode not in {"softmax", "sigmoid"}:
            raise ValueError("mode must be softmax or sigmoid")
        self.mode = mode

    def call(self, values):
        clipped = keras.ops.clip(values, 1e-7, 1.0 - 1e-7)
        if self.mode == "sigmoid":
            return keras.ops.log(clipped) - keras.ops.log(1.0 - clipped)
        # Softmax logits are identifiable only up to an additive constant.
        return keras.ops.log(clipped)

    def get_config(self):
        return {**super().get_config(), "mode": self.mode}


@keras.utils.register_keras_serializable(package="oracle_builder")
class InvertIntensity(keras.layers.Layer):
    """Invert normalized image intensity without a non-portable Lambda layer."""

    def call(self, values):
        return 1.0 - values


@keras.utils.register_keras_serializable(package="oracle_builder")
class ConvertChannels(keras.layers.Layer):
    """Convert between one-channel and RGB tensors for declared image inputs."""

    def __init__(self, target_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.target_channels = int(target_channels)
        if self.target_channels not in {1, 3}:
            raise ValueError("target_channels must be 1 or 3")

    def call(self, values):
        if self.target_channels == 1:
            return tf.image.rgb_to_grayscale(values)
        return tf.image.grayscale_to_rgb(values)

    def get_config(self):
        return {**super().get_config(), "target_channels": self.target_channels}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        paths = [path]
    else:
        paths = sorted(value for value in path.rglob("*") if value.is_file())
    for value in paths:
        if path.is_dir():
            digest.update(value.relative_to(path).as_posix().encode("utf-8"))
        with value.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _json_shape(shape) -> list[int | None]:
    return [int(value) if value is not None else None for value in shape]


def _tensor_description(tensor) -> dict[str, Any]:
    # Eager tensors deliberately do not expose ``.name``; symbolic Keras
    # tensors do. Both are useful during inspection and reload testing.
    try:
        name = str(tensor.name)
    except (AttributeError, ValueError):
        name = None
    return {
        "name": name,
        "shape": _json_shape(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def inspect_keras_model(model: keras.Model) -> dict[str, Any]:
    return {
        "framework": "keras",
        "keras_version": getattr(keras, "__version__", None),
        "name": model.name,
        "parameter_count": int(model.count_params()),
        "input_count": len(model.inputs),
        "output_count": len(model.outputs),
        "inputs": [_tensor_description(value) for value in model.inputs],
        "outputs": [_tensor_description(value) for value in model.outputs],
        "layers": [
            {
                "name": layer.name,
                "class_name": layer.__class__.__name__,
                "trainable": bool(layer.trainable),
            }
            for layer in model.layers
        ],
    }


def inspect_savedmodel(source: Path) -> tuple[Any, Any, str, dict[str, Any]]:
    """Load and inspect a single-input TensorFlow SavedModel serving signature."""
    loaded = tf.saved_model.load(str(source))
    signature = loaded.signatures.get("serving_default") or loaded.signatures.get("serve")
    if signature is None:
        raise ValueError("SavedModel must expose a serving_default or serve signature")
    positional, keyword = signature.structured_input_signature
    if positional or len(keyword) != 1:
        raise ValueError("SavedModel ingestion currently requires exactly one keyword input")
    input_name, input_spec = next(iter(keyword.items()))
    outputs = signature.structured_outputs
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("SavedModel serving signature must return named tensor outputs")
    return loaded, signature, input_name, {
        "framework": "tensorflow_savedmodel",
        "tensorflow_version": tf.__version__,
        "name": source.name,
        "input_count": 1,
        "output_count": len(outputs),
        "inputs": [{"name": input_name, "shape": _json_shape(input_spec.shape), "dtype": input_spec.dtype.name}],
        "outputs": [
            {"name": name, "shape": _json_shape(value.shape), "dtype": value.dtype.name}
            for name, value in outputs.items()
        ],
        "signatures": sorted(loaded.signatures),
    }


def _dataset_reference(path: str | Path | None, task: str) -> dict[str, Any]:
    if path is None:
        return {}
    database = Path(path).expanduser().resolve()
    with sqlite3.connect(database) as connection:
        info = read_dataset_info(connection)
        report = validate_database(connection)
        if not report["valid"]:
            raise ValueError("Dataset validation failed: " + "; ".join(report["errors"]))
        if task in {"classification", "segmentation"}:
            expected = "mask_refinement" if task == "segmentation" else task
            if info["dataset_type"] != expected:
                raise ValueError(
                    f"Model product task {task!r} does not match dataset type {info['dataset_type']!r}"
                )
        return {
            "dataset_id": info["dataset_id"],
            "revision_id": info["revision_id"],
            "dataset_type": info["dataset_type"],
            "schema_name": info["schema_name"],
            "schema_version": info["schema_version"],
            "version": info.get("version"),
            "lifecycle": info["lifecycle"],
            "fingerprint_sha256": dataset_fingerprint(connection),
        }


def _declared_labels(info: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize optional TOML labels into the inference label contract."""
    # Accept top-level labels (the original API) and product-scoped labels
    # (which reads more naturally in TOML).  The final fallback supports early
    # product cards that placed labels beside promotion settings.
    rows = (
        info.get("labels")
        or dict(info.get("product", {})).get("labels")
        or dict(info.get("promotion", {})).get("labels")
        or []
    )
    if not isinstance(rows, list):
        raise ValueError("labels must be an array of TOML tables or strings")
    labels: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if isinstance(row, str):
            labels.append({"class_index": index, "label_id": row, "name": row})
        elif isinstance(row, dict):
            name = str(row.get("name") or row.get("label_id") or index)
            labels.append(
                {
                    "class_index": int(row.get("class_index", index)),
                    "label_id": str(row.get("label_id") or name),
                    "name": name,
                }
            )
        else:
            raise ValueError("each labels entry must be a string or table")
    indices = [row["class_index"] for row in labels]
    if sorted(indices) != list(range(len(labels))):
        raise ValueError("labels.class_index values must be unique and contiguous from zero")
    return labels


def _copy_original(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        target = destination.with_suffix(source.suffix.lower())
        shutil.copy2(source, target)
        return target
    target = destination
    shutil.copytree(source, target)
    return target


def _save_keras_atomic(model: keras.Model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.keras")
    try:
        model.save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _forward_test(model: keras.Model, inspection: dict[str, Any]) -> dict[str, Any]:
    if inspection["input_count"] != 1:
        return {"attempted": False, "reason": "multiple model inputs"}
    input_info = inspection["inputs"][0]
    shape = input_info["shape"]
    if any(value is None for value in shape[1:]):
        return {"attempted": False, "reason": "dynamic non-batch input shape"}
    dtype = tf.as_dtype(input_info["dtype"])
    values = tf.zeros([1, *shape[1:]], dtype=dtype)
    output = model(values, training=False)
    flattened = tf.nest.flatten(output)
    return {
        "attempted": True,
        "output_count": len(flattened),
        "outputs": [_tensor_description(value) for value in flattened],
    }


def _tensor_rank(tensor) -> int | None:
    shape = tensor.shape
    rank = getattr(shape, "rank", None)
    if rank is not None:
        return int(rank)
    try:
        return len(shape)
    except TypeError:
        return None


def _layer_output(model: keras.Model, name: str | None):
    return model.get_layer(name).output if name else None


def _activation_name(tensor) -> str | None:
    history = getattr(tensor, "_keras_history", None)
    operation = getattr(history, "operation", None)
    activation = getattr(operation, "activation", None)
    name = getattr(activation, "__name__", None)
    return str(name).lower() if name else None


def _automatic_embedding(model: keras.Model, probability_tensor):
    """Return a conservative vector-valued penultimate representation."""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.InputLayer):
            continue
        output = getattr(layer, "output", None)
        if output is None or output is probability_tensor:
            continue
        if _tensor_rank(output) == 2:
            return output, layer.name
    return None, None


def _preprocessed_input(
    source_model: keras.Model,
    preprocessing: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    """Build an optional, explicit preprocessing wrapper around image models."""
    source_shape = _json_shape(source_model.inputs[0].shape)[1:]
    if not bool(preprocessing.get("embed_in_model", False)):
        return source_model.inputs[0], source_model.inputs[0], {"embedded": False}
    raw_shape = list(preprocessing.get("raw_input_shape") or source_shape)
    if len(raw_shape) != 3 or any(value is None for value in raw_shape):
        raise ValueError(
            "preprocessing.raw_input_shape must be [height, width, channels] "
            "when preprocessing.embed_in_model is true"
        )
    if len(source_shape) != 3 or any(value is None for value in source_shape):
        raise ValueError("embedded preprocessing requires a concrete rank-3 source input")
    raw = keras.Input(shape=raw_shape, dtype=source_model.inputs[0].dtype, name="raw_input")
    values = raw
    target_height, target_width, target_channels = source_shape
    if raw_shape[:2] != source_shape[:2]:
        values = keras.layers.Resizing(target_height, target_width, name="resize_to_model_input")(values)
    if raw_shape[2] != target_channels:
        if {raw_shape[2], target_channels} != {1, 3}:
            raise ValueError("only one-channel and RGB conversion is supported")
        values = ConvertChannels(target_channels, name="convert_channels")(values)
    if bool(preprocessing.get("invert", False)):
        values = InvertIntensity(name="invert_intensity")(values)
    if bool(preprocessing.get("rescale", False)):
        values = keras.layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(values)
    return raw, values, {
        "embedded": True,
        "raw_input_shape": raw_shape,
        "model_input_shape": source_shape,
        "operations": {
            "resize": raw_shape[:2] != source_shape[:2],
            "channel_conversion": raw_shape[2] != target_channels,
            "invert": bool(preprocessing.get("invert", False)),
            "rescale_0_1": bool(preprocessing.get("rescale", False)),
        },
    }


def promote_keras_model(
    model: keras.Model,
    *,
    task: str,
    info: dict[str, Any],
    enabled: bool | None = None,
) -> tuple[keras.Model, dict[str, Any]]:
    """Adapt a conventional Keras model to Oracle Builder's named layers.

    Explicit TOML layer names win. Automatic promotion only handles a single
    image input and a single probability/logit output with softmax or sigmoid
    semantics; otherwise the original model is retained with an explanation.
    """
    options = dict(info.get("promotion", {}))
    requested = bool(options.get("enabled", task in {"classification", "segmentation"}))
    if enabled is not None:
        requested = enabled
    report: dict[str, Any] = {"requested": requested, "promoted": False, "assumptions": []}
    if not requested:
        report["reason"] = "promotion disabled"
        return model, report
    if task not in {"classification", "segmentation"}:
        report["reason"] = "product.task must be classification or segmentation"
        return model, report
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        report["reason"] = "automatic promotion requires exactly one input and one output"
        return model, report

    probability_name = options.get("probabilities_layer")
    logits_name = options.get("logits_layer")
    embedding_name = options.get("embedding_layer")
    probability = _layer_output(model, probability_name)
    if probability is None:
        probability = model.outputs[0]
    activation = str(options.get("activation") or _activation_name(probability) or "linear").lower()
    expected_rank = 2 if task == "classification" else 4
    if _tensor_rank(probability) != expected_rank:
        report["reason"] = f"{task} promotion expects a rank-{expected_rank} output"
        return model, report
    if activation not in {"softmax", "sigmoid", "linear"}:
        report["reason"] = f"unsupported output activation {activation!r}"
        return model, report
    if task == "classification" and activation == "sigmoid":
        report["reason"] = "classification promotion requires softmax or linear class scores"
        return model, report
    if task == "segmentation" and activation == "softmax":
        report["reason"] = "segmentation promotion requires sigmoid or linear mask scores"
        return model, report

    selected = {"probabilities": probability}
    if logits_name:
        selected["declared_logits"] = _layer_output(model, logits_name)
    if embedding_name:
        selected["declared_embedding"] = _layer_output(model, embedding_name)
    raw_input, model_input, preprocessing_report = _preprocessed_input(
        model, dict(info.get("preprocessing", {}))
    )
    if preprocessing_report["embedded"]:
        # Calling the imported model as a single nested layer is robustly
        # serializable. Internal source-layer references are intentionally not
        # carried across this boundary; the output still gets logits/probability
        # promotion and the report calls out any omitted embedding.
        selected_values = {"probabilities": model(model_input)}
        if logits_name or embedding_name:
            report["assumptions"].append(
                "layer-specific outputs were omitted because preprocessing is embedded"
            )
    else:
        # Reuse the original Functional graph directly. This preserves all
        # imported variables when Keras serializes the promoted model.
        selected_values = selected
    probability = selected_values["probabilities"]
    if activation == "linear":
        mode = "softmax" if task == "classification" else "sigmoid"
        logits = keras.layers.Activation("linear", name="logits")(probability)
        probability = keras.layers.Activation(mode, name="predictions")(logits)
        report["assumptions"].append(f"linear output treated as {mode} logits")
    else:
        probability = keras.layers.Activation("linear", name="predictions")(probability)
        if logits_name and "declared_logits" in selected_values:
            logits = keras.layers.Activation("linear", name="logits")(selected_values["declared_logits"])
            report["assumptions"].append("logits supplied by promotion.logits_layer")
        else:
            logits = ProbabilityToLogits(activation, name="logits")(probability)
            report["assumptions"].append(f"logits derived from {activation} probabilities")

    outputs: dict[str, Any] = {"logits": logits, "probabilities": probability}
    embedding_tensor = selected_values.get("declared_embedding")
    embedding_layer = embedding_name
    if task == "classification" and embedding_tensor is None and not preprocessing_report["embedded"]:
        embedding_tensor, embedding_layer = _automatic_embedding(model, selected["probabilities"])
        if embedding_tensor is not None:
            report["assumptions"].append(f"embedding selected from penultimate layer {embedding_layer!r}")
    if task == "classification" and embedding_tensor is not None:
        if _tensor_rank(embedding_tensor) != 2:
            report["assumptions"].append("declared embedding was not rank-2 and was omitted")
        else:
            embedding = L2Normalization(name="features")(embedding_tensor)
            outputs["features"] = embedding
            report["embedding_layer"] = embedding_layer
            report["embedding_dimension"] = _json_shape(embedding.shape)[-1]
    promoted = keras.Model(raw_input, outputs, name=f"{model.name}_oracle_builder")
    report.update(
        {
            "promoted": True,
            "activation": activation,
            "preprocessing": preprocessing_report,
            "output_layers": {"logits": "logits", "probabilities": "predictions", "features": "features" if task == "classification" and embedding_tensor is not None and _tensor_rank(embedding_tensor) == 2 else None},
        }
    )
    return promoted, report


def _external_contract(
    task: str,
    inspection: dict[str, Any],
    info: dict[str, Any],
    promotion: dict[str, Any],
    *,
    architecture: str = "external_keras",
) -> dict[str, Any]:
    declared_outputs = dict(info.get("outputs", {}))
    return {
        "task": task,
        "architecture": architecture,
        "variant": info.get("model", {}).get("variant"),
        "input": {
            "tensors": inspection["inputs"],
            "preprocessing": dict(info.get("preprocessing", {})),
        },
        "outputs": (
            {
                "logits": True,
                "probabilities": True,
                "identity_embedding": promotion.get("output_layers", {}).get("features") is not None,
                "labels": _declared_labels(info),
            }
            if promotion.get("promoted") and task == "classification"
            else {
                "logits": True,
                "probability_map": True,
                "mask": True,
                "probability_threshold": float(info.get("outputs", {}).get("threshold", 0.5)),
            }
            if promotion.get("promoted") and task == "segmentation"
            else declared_outputs or {"tensors": inspection["outputs"]}
        ),
        "compatibility": {
            "oracle_builder_inference_bundle": bool(promotion.get("promoted")),
            "note": (
                "Promoted to Oracle Builder's named-output inference contract."
                if promotion.get("promoted")
                else "This model is preserved as a portable Keras product. Automatic "
                "Oracle Builder inference requires an explicit adapter or promotion "
                "to the standard named-output inference contract."
            ),
        },
    }


def ingest_keras_model(
    source: str | Path,
    info_path: str | Path,
    output: str | Path,
    *,
    dataset: str | Path | None = None,
    promote: bool | None = None,
) -> dict[str, Any]:
    """Create a sealed Oracle Builder model-product artifact from a Keras model."""
    source_path = Path(source).expanduser().resolve()
    metadata_path = Path(info_path).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if not source_path.is_file() or source_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"Expected a Keras .keras/.h5 model file ({allowed}): {source_path}")
    if destination.exists():
        raise FileExistsError(destination)
    info = load_toml(metadata_path)
    product = dict(info.get("product", {}))
    task = str(product.get("task", "generic")).lower()
    if task not in {"generic", "classification", "segmentation"}:
        raise ValueError("product.task must be generic, classification, or segmentation")
    name = str(product.get("name") or source_path.stem)
    source_model = keras.models.load_model(source_path, compile=False)
    source_inspection = inspect_keras_model(source_model)
    if not source_inspection["inputs"]:
        raise ValueError("The imported Keras model has no declared inputs")
    model, promotion = promote_keras_model(
        source_model, task=task, info=info, enabled=promote
    )
    inspection = inspect_keras_model(model)
    input_shape = inspection["inputs"][0]["shape"][1:]
    if any(value is None for value in input_shape):
        input_shape = []
    dataset_info = _dataset_reference(dataset, task)
    if task == "classification":
        dataset_info["labels"] = _declared_labels(info)
        class_count = source_inspection["outputs"][0]["shape"][-1]
        if dataset_info["labels"] and class_count is not None and len(dataset_info["labels"]) != class_count:
            raise ValueError(
                f"TOML declares {len(dataset_info['labels'])} labels but model output has {class_count} classes"
            )
    run_id = str(uuid.uuid4())
    config: dict[str, Any] = {
        "run": {"run_id": run_id, "run_name": name, "task": task, "model": "external_keras"},
        "data": {"input_shape": input_shape},
        "model": {"source_format": source_path.suffix.lower().lstrip(".")},
        "preprocessing": dict(info.get("preprocessing", {})),
        "product": product,
        # Preserve every declared TOML field in the resolved portable config;
        # source.toml remains the authoritative human-authored record.
        "product_metadata": info,
        "dataset": dataset_info,
        "promotion": promotion,
        "external_model_contract": _external_contract(task, inspection, info, promotion),
        "paths": {"source_model": str(source_path), "info_path": str(metadata_path), "dataset_path": str(Path(dataset).expanduser().resolve()) if dataset else None, "run_dir": str(destination)},
    }
    manifest = create_run_artifact(
        destination,
        run_id=run_id,
        name=name,
        config=config,
        source_config=metadata_path,
        artifact_type="model_product",
    )
    layout = RunLayout(destination)
    create_unavailable_split_manifest(
        destination,
        config,
        reason="Externally ingested model product has no Oracle Builder training split protocol.",
    )
    environment = write_environment(destination)
    init_training_log(layout.training_log, run_id, name, config, environment)
    try:
        original = _copy_original(source_path, layout.model / "source" / "original")
        _save_keras_atomic(source_model, layout.model / "imported.keras")
        _save_keras_atomic(model, layout.model / "final.keras")
        model.save_weights(layout.model / "weights.weights.h5")
        summary_lines: list[str] = []
        model.summary(print_fn=summary_lines.append)
        (layout.model / "model_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        forward = _forward_test(keras.models.load_model(layout.model / "final.keras", compile=False), inspection)
        inspection["source"] = {
            "filename": source_path.name,
            "sha256": _sha256(source_path),
            "preserved_path": original.relative_to(destination).as_posix(),
        }
        inspection["source_model"] = source_inspection
        inspection["promotion"] = promotion
        inspection["reload_test"] = {"keras_reloaded": True, **forward}
        (layout.model / "inspection.json").write_text(json.dumps(inspection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        model_manifest = {
            "schema_name": "oracle_builder_inference_bundle",
            "schema_version": "1.0.0",
            "model_asset_id": str(uuid.uuid4()),
            "artifact_id": manifest["artifact_id"],
            "run_id": run_id,
            "task": task,
            "architecture": "external_keras",
            "input": config["external_model_contract"]["input"],
            "outputs": config["external_model_contract"]["outputs"],
            "inference_runtime": {
                "batching": {},
                "note": "Batch sizing is selected by the calling inference host.",
            },
            "inference_contract": {
                "input_schema": "oracle_builder.inference_item",
                "result_schema": "oracle_builder.inference_result",
                "result_set_schema": "oracle_builder.inference_result_set",
                "version": "1.0.0",
                "persistence": "explicit",
                "adapter_required": not promotion.get("promoted", False),
            },
            "formats": [
                {"asset_id": str(uuid.uuid4()), "format": "keras_v3", "path": "final.keras", "role": "preferred_full_model", "sha256": _sha256(layout.model / "final.keras")},
                {"asset_id": str(uuid.uuid4()), "format": "keras_v3", "path": "imported.keras", "role": "normalized_import", "sha256": _sha256(layout.model / "imported.keras")},
                {"asset_id": str(uuid.uuid4()), "format": "keras_weights_hdf5", "path": "weights.weights.h5", "role": "source_architecture_required", "sha256": _sha256(layout.model / "weights.weights.h5")},
                {"asset_id": str(uuid.uuid4()), "format": "original_keras_source", "path": original.relative_to(layout.model).as_posix(), "role": "preserved_source", "sha256": _sha256(original)},
            ],
            "inspection_path": "inspection.json",
        }
        (layout.model / "model_manifest.json").write_text(json.dumps(model_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (layout.model / "load_test_report.json").write_text(json.dumps(inspection["reload_test"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        layout.metrics_json.write_text("{}\n", encoding="utf-8")
        layout.metrics_csv.write_text("epoch\n", encoding="utf-8")
        log_event(layout.training_log, run_id, "INFO", "Imported external Keras model", {"source_sha256": inspection["source"]["sha256"], "parameter_count": inspection["parameter_count"], "promotion": promotion})
        mark_run_complete(layout.training_log, run_id, "complete")
        update_run_artifact(destination, status="complete", summary={"product": product, "inspection": {"parameter_count": inspection["parameter_count"], "input_count": inspection["input_count"], "output_count": inspection["output_count"]}})
        sealed = seal_run_artifact(destination)
    except Exception:
        mark_run_complete(layout.training_log, run_id, "failed")
        update_run_artifact(destination, status="failed", error="Model product ingestion failed")
        seal_run_artifact(destination)
        raise
    return {"output": str(destination), "artifact_id": sealed["artifact_id"], "run_id": run_id, "fingerprint_sha256": sealed["fingerprint_sha256"], "inspection": inspection}


def ingest_savedmodel(
    source: str | Path,
    info_path: str | Path,
    output: str | Path,
    *,
    dataset: str | Path | None = None,
    promote: bool | None = None,
) -> dict[str, Any]:
    """Ingest a TensorFlow SavedModel without reconstructing it as Keras.

    A promoted classification product writes a compact SavedModel adapter with
    the canonical ``logits`` and ``probabilities`` outputs.  The original
    SavedModel is copied unchanged for provenance and future conversion.
    """
    source_path = Path(source).expanduser().resolve()
    metadata_path = Path(info_path).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if not source_path.is_dir() or not (source_path / "saved_model.pb").is_file():
        raise ValueError(f"Expected a TensorFlow SavedModel directory: {source_path}")
    if destination.exists():
        raise FileExistsError(destination)
    info = load_toml(metadata_path)
    product = dict(info.get("product", {}))
    task = str(product.get("task", "generic")).lower()
    if task not in {"generic", "classification", "segmentation"}:
        raise ValueError("product.task must be generic, classification, or segmentation")
    if task != "classification":
        raise ValueError("SavedModel ingestion currently supports classification products only")
    requested = bool(dict(info.get("promotion", {})).get("enabled", True))
    if promote is not None:
        requested = promote
    if not requested:
        raise ValueError("SavedModel classification ingestion requires promotion to the named-output contract")

    loaded, signature, input_name, inspection = inspect_savedmodel(source_path)
    input_shape = inspection["inputs"][0]["shape"][1:]
    if any(value is None for value in input_shape):
        raise ValueError("SavedModel must have a concrete non-batch input shape")
    options = dict(info.get("promotion", {}))
    activation = str(options.get("activation", "")).lower()
    if activation not in {"softmax", "linear"}:
        raise ValueError("SavedModel classification requires promotion.activation = 'softmax' or 'linear'")
    output_name = str(options.get("output_name") or "")
    output_specs = signature.structured_outputs
    if not output_name:
        if len(output_specs) != 1:
            raise ValueError("promotion.output_name is required for a multi-output SavedModel")
        output_name = next(iter(output_specs))
    if output_name not in output_specs:
        raise ValueError(f"promotion.output_name {output_name!r} is not a SavedModel output")
    output_shape = _json_shape(output_specs[output_name].shape)
    if len(output_shape) != 2:
        raise ValueError("SavedModel classification output must be rank-2 [batch, classes]")
    labels = _declared_labels(info)
    if labels and output_shape[-1] is not None and len(labels) != output_shape[-1]:
        raise ValueError(f"TOML declares {len(labels)} labels but model output has {output_shape[-1]} classes")

    name = str(product.get("name") or source_path.name)
    run_id = str(uuid.uuid4())
    promotion_report = {
        "requested": True,
        "promoted": True,
        "activation": activation,
        "source_signature": (
            signature.name.decode("utf-8")
            if isinstance(signature.name, bytes)
            else str(signature.name)
        ),
        "source_input": input_name,
        "source_output": output_name,
        "output_layers": {"logits": "logits", "probabilities": "probabilities", "features": None},
        "assumptions": [
            "SavedModel has no declared penultimate embedding; no identity_embedding was exported.",
        ],
    }
    dataset_info = _dataset_reference(dataset, task)
    dataset_info["labels"] = labels
    config: dict[str, Any] = {
        "run": {"run_id": run_id, "run_name": name, "task": task, "model": "external_savedmodel"},
        "data": {"input_shape": input_shape},
        "model": {"source_format": "tensorflow_savedmodel"},
        "preprocessing": dict(info.get("preprocessing", {})),
        "product": product,
        "product_metadata": info,
        "dataset": dataset_info,
        "promotion": promotion_report,
        "external_model_contract": _external_contract(
            task, inspection, info, promotion_report, architecture="external_savedmodel"
        ),
        "paths": {"source_model": str(source_path), "info_path": str(metadata_path), "dataset_path": str(Path(dataset).expanduser().resolve()) if dataset else None, "run_dir": str(destination)},
    }
    manifest = create_run_artifact(destination, run_id=run_id, name=name, config=config,
                                   source_config=metadata_path, artifact_type="model_product")
    layout = RunLayout(destination)
    create_unavailable_split_manifest(destination, config,
        reason="Externally ingested model product has no Oracle Builder training split protocol.")
    environment = write_environment(destination)
    init_training_log(layout.training_log, run_id, name, config, environment)
    try:
        original = _copy_original(source_path, layout.model / "source" / "original_savedmodel")
        adapter = _SavedModelClassificationAdapter(loaded, signature, input_name, output_name, activation)
        input_spec = tf.TensorSpec([None, *input_shape], tf.as_dtype(inspection["inputs"][0]["dtype"]), name="inputs")
        tf.saved_model.save(adapter, str(layout.model / "export_savedmodel"), signatures={
            "serving_default": adapter.serve.get_concrete_function(input_spec),
            "classify": adapter.classify.get_concrete_function(input_spec),
        })
        from oracle_builder.saving.load_test import run_load_tests
        reload_test = run_load_tests(destination, config)
        if not reload_test["prediction_test_passed"]:
            raise RuntimeError("The exported SavedModel adapter failed its reload prediction test")
        inspection["source"] = {"directory": source_path.name, "sha256": _sha256(source_path),
                                "preserved_path": original.relative_to(destination).as_posix()}
        inspection["promotion"] = promotion_report
        inspection["reload_test"] = reload_test
        (layout.model / "inspection.json").write_text(json.dumps(inspection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        exported = layout.model / "export_savedmodel"
        model_manifest = {
            "schema_name": "oracle_builder_inference_bundle", "schema_version": "1.0.0",
            "model_asset_id": str(uuid.uuid4()), "artifact_id": manifest["artifact_id"], "run_id": run_id,
            "task": task, "architecture": "external_savedmodel", "input": config["external_model_contract"]["input"],
            "outputs": config["external_model_contract"]["outputs"],
            "inference_runtime": {"batching": {}, "note": "Batch sizing is selected by the calling inference host."},
            "inference_contract": {"input_schema": "oracle_builder.inference_item", "result_schema": "oracle_builder.inference_result", "result_set_schema": "oracle_builder.inference_result_set", "version": "1.0.0", "persistence": "explicit", "adapter_required": False},
            "formats": [
                {"asset_id": str(uuid.uuid4()), "format": "tensorflow_savedmodel", "path": "export_savedmodel", "role": "preferred_inference_model", "sha256": _sha256(exported)},
                {"asset_id": str(uuid.uuid4()), "format": "original_tensorflow_savedmodel", "path": original.relative_to(layout.model).as_posix(), "role": "preserved_source", "sha256": _sha256(original)},
            ], "inspection_path": "inspection.json",
        }
        (layout.model / "model_manifest.json").write_text(json.dumps(model_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (layout.model / "load_test_report.json").write_text(json.dumps(reload_test, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        layout.metrics_json.write_text("{}\n", encoding="utf-8")
        layout.metrics_csv.write_text("epoch\n", encoding="utf-8")
        log_event(layout.training_log, run_id, "INFO", "Imported external TensorFlow SavedModel", {"source_sha256": inspection["source"]["sha256"], "promotion": promotion_report})
        mark_run_complete(layout.training_log, run_id, "complete")
        update_run_artifact(destination, status="complete", summary={"product": product, "inspection": {"input_count": 1, "output_count": len(inspection["outputs"])}})
        sealed = seal_run_artifact(destination)
    except Exception:
        mark_run_complete(layout.training_log, run_id, "failed")
        update_run_artifact(destination, status="failed", error="SavedModel product ingestion failed")
        seal_run_artifact(destination)
        raise
    return {"output": str(destination), "artifact_id": sealed["artifact_id"], "run_id": run_id,
            "fingerprint_sha256": sealed["fingerprint_sha256"], "inspection": inspection}
