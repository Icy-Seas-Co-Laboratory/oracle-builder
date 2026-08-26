from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import tensorflow as tf
from tensorflow import keras

from oracle_builder.artifacts import write_model_contract
from oracle_builder.classification.features import build_feature_model


class _ClassificationExport(tf.Module):
    def __init__(self, model: keras.Model):
        super().__init__()
        self.model = model
        self.feature_model = build_feature_model(model)

    @tf.function
    def classify(self, inputs):
        outputs = self.feature_model(inputs, training=False)
        return {
            "logits": outputs["logits"],
            "probabilities": outputs["probabilities"],
        }

    @tf.function
    def embed(self, inputs):
        outputs = self.feature_model(inputs, training=False)
        return {"features": outputs["features"]}

    @tf.function
    def serve(self, inputs):
        return self.feature_model(inputs, training=False)


class _SegmentationExport(tf.Module):
    def __init__(self, model: keras.Model):
        super().__init__()
        self.model = model
        self.inference_model = keras.Model(
            model.input,
            {
                "logits": model.get_layer("logits").output,
                "probabilities": model.output,
            },
        )

    @tf.function
    def serve(self, inputs):
        return self.inference_model(inputs, training=False)


def _export_classification_model(
    model: keras.Model,
    export_dir: Path,
    config: dict[str, Any],
) -> None:
    exported = _ClassificationExport(model)
    input_spec = tf.TensorSpec(
        [None, *config["data"]["input_shape"]],
        tf.float32,
        name="inputs",
    )
    tf.saved_model.save(
        exported,
        str(export_dir),
        signatures={
            "serving_default": exported.serve.get_concrete_function(input_spec),
            "classify": exported.classify.get_concrete_function(input_spec),
            "embed": exported.embed.get_concrete_function(input_spec),
        },
    )


def _export_segmentation_model(
    model: keras.Model,
    export_dir: Path,
    config: dict[str, Any],
) -> None:
    exported = _SegmentationExport(model)
    input_spec = tf.TensorSpec(
        [None, *config["data"]["input_shape"]],
        tf.float32,
        name="inputs",
    )
    signature = exported.serve.get_concrete_function(input_spec)
    tf.saved_model.save(
        exported,
        str(export_dir),
        signatures={
            "serving_default": signature,
            "segment": signature,
        },
    )


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(
        value for value in path.rglob("*") if value.is_file()
    )
    for value in paths:
        if path.is_dir():
            digest.update(value.relative_to(path).as_posix().encode("utf-8"))
        with value.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def save_model_artifacts(model: keras.Model, run_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    model_path = Path(run_dir) / "model"
    model_path.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "final_keras_saved": False,
        "weights_saved": False,
        "savedmodel_exported": False,
        "errors": [],
    }
    try:
        model.save(model_path / "final.keras")
        report["final_keras_saved"] = True
    except Exception as exc:
        report["errors"].append({"artifact": "final.keras", "error": str(exc)})
    try:
        model.save_weights(model_path / "weights.weights.h5")
        report["weights_saved"] = True
    except Exception as exc:
        report["errors"].append({"artifact": "weights.weights.h5", "error": str(exc)})
    if config.get("output", {}).get("export_savedmodel", True):
        try:
            export_dir = model_path / "export_savedmodel"
            if config.get("run", {}).get("task") in {"classification", "clustering"}:
                _export_classification_model(model, export_dir, config)
            else:
                _export_segmentation_model(model, export_dir, config)
            report["savedmodel_exported"] = True
        except Exception as exc:
            report["errors"].append({"artifact": "export_savedmodel", "error": str(exc)})
    formats = []
    if report["final_keras_saved"]:
        formats.append(
            {
                "asset_id": str(uuid.uuid4()),
                "format": "keras_v3",
                "path": "final.keras",
                "role": "preferred_full_model",
                "sha256": _path_sha256(model_path / "final.keras"),
            }
        )
    if report["weights_saved"]:
        formats.append(
            {
                "asset_id": str(uuid.uuid4()),
                "format": "keras_weights_hdf5",
                "path": "weights.weights.h5",
                "role": "rebuild_with_oracle_builder",
                "sha256": _path_sha256(model_path / "weights.weights.h5"),
            }
        )
    if report["savedmodel_exported"]:
        formats.append(
            {
                "asset_id": str(uuid.uuid4()),
                "format": "tensorflow_saved_model",
                "path": "export_savedmodel",
                "role": "portable_inference",
                "sha256": _path_sha256(model_path / "export_savedmodel"),
            }
        )
    model_manifest = {
        "schema_name": "oracle_builder_inference_bundle",
        "schema_version": "1.0.0",
        "model_asset_id": str(uuid.uuid4()),
        "artifact_id": config.get("artifact", {}).get("artifact_id"),
        "run_id": config.get("run", {}).get("run_id"),
        "task": config.get("run", {}).get("task"),
        "architecture": config.get("run", {}).get("model"),
        "input": {
            "shape": [None, *config["data"]["input_shape"]],
            "dtype": "float32",
            "preprocessing": config.get("preprocessing", {}),
            "segmentation_input": {
                "candidate_sdf": config.get("data", {}).get(
                    "candidate_sdf", False
                ),
                "candidate_sdf_clip_distance": config.get("data", {}).get(
                    "candidate_sdf_clip_distance"
                ),
            },
            "tiling": config.get("tiling", {}),
        },
        "outputs": (
            {
                "logits": True,
                "probabilities": True,
                "labels": config.get("dataset", {}).get("labels", []),
                "identity_embedding": True,
                "embedding_dimension": config.get("model", {}).get(
                    "embedding_dim", 256
                ),
                "cluster_evidence": bool(
                    config.get("clustering", {}).get("enabled", False)
                    or config.get("clustering", {}).get("structure")
                ),
                "cluster_count": config.get("clustering", {}).get(
                    "structure", {}
                ).get("cluster_count"),
            }
            if config.get("run", {}).get("task") == "classification"
            else {
                "embedding": True,
                "embedding_dimension": config.get("model", {}).get(
                    "embedding_dim", 256
                ),
                "embedding_normalized": config.get("model", {}).get(
                    "normalize_embeddings", True
                ),
                "cluster_evidence": True,
                "cluster_count": config.get("clustering", {}).get(
                    "structure", {}
                ).get("cluster_count"),
                "cluster_method": config.get("clustering", {}).get(
                    "method", "spherical_kmeans"
                ),
            }
            if config.get("run", {}).get("task") == "clustering"
            else {
                "logits": True,
                "probability_map": True,
                "mask": True,
                "segmentation_target": config.get("training", {}).get(
                    "segmentation_target", "validated_mask"
                ),
                "probability_threshold": config.get("evaluation", {}).get(
                    "segmentation_threshold", 0.5
                ),
            }
        ),
        "postprocessing": {
            "classification_evidence": config.get("evidence", {}),
            "clustering_evidence": config.get("clustering", {}),
            "segmentation_threshold": config.get("evaluation", {}).get(
                "segmentation_threshold", 0.5
            ),
            "segmentation_target": config.get("training", {}).get(
                "segmentation_target", "validated_mask"
            ),
        },
        "inference_runtime": {
            "batching": config.get("inference", {}),
            "note": (
                "Automatic batch sizing is resolved and verified on the "
                "inference host; the resolved runtime value is logged per run."
            ),
        },
        "inference_contract": {
            "input_schema": "oracle_builder.inference_item",
            "result_schema": "oracle_builder.inference_result",
            "result_set_schema": "oracle_builder.inference_result_set",
            "version": "1.0.0",
            "persistence": "explicit",
        },
        "formats": formats,
        "save_report": report,
        "contract_path": "contract.json",
    }
    (model_path / "model_manifest.json").write_text(
        json.dumps(model_manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    write_model_contract(
        run_dir,
        {
            "task": model_manifest["task"],
            "architecture": model_manifest["architecture"],
            "inputs": {"image": model_manifest["input"]},
            "outputs": model_manifest["outputs"],
            "preprocessing": model_manifest["input"].get("preprocessing", {}),
            "postprocessing": model_manifest.get("postprocessing", {}),
            "inference": model_manifest["inference_contract"],
        },
    )
    return report


def write_load_test_report(run_dir: str | Path, report: dict[str, Any]) -> None:
    Path(run_dir, "model", "load_test_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
