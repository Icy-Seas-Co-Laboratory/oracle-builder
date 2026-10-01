"""Versioned, shape-stable V2 image output for three model tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from oracle_builder.evaluation.segmentation_targets import CANDIDATE_DELTA, segmentation_target_mode
from oracle_builder.inference.contracts import ArrayPayload


IMAGE_OUTPUT_SCHEMA = "oracle_builder.inference.v2.image_output"
IMAGE_OUTPUT_VERSION = "1.0.0"
IMAGE_MODEL_TASKS = frozenset({"classification", "embedding", "clustering"})
IMAGE_FIELDS = frozenset({
    "class_probabilities", "primary_decision", "embedding", "embedding_normalized",
    "knn", "prototype_similarity", "secondary_heads", "cluster_evidence",
})
IMAGE_SELECTABLE_FIELDS = IMAGE_FIELDS | {"diagnostics"}


def validate_image_output_shape(output: Any) -> None:
    """Reject silently dropped stable fields at the transport boundary."""
    if not isinstance(output, dict):
        raise ValueError("Image result output must be an object")
    required = IMAGE_FIELDS | {
        "schema_name", "schema_version", "type", "model_task", "diagnostics",
    }
    missing = required - set(output)
    if missing:
        raise ValueError(f"Image result is missing required fields: {', '.join(sorted(missing))}")
    if output["schema_name"] != IMAGE_OUTPUT_SCHEMA or output["schema_version"] != IMAGE_OUTPUT_VERSION:
        raise ValueError("Unsupported image-result schema or version")
    if output["type"] != "image_analysis" or output["model_task"] not in IMAGE_MODEL_TASKS:
        raise ValueError("Image result has an invalid model task")
    if output["secondary_heads"] is not None and not isinstance(output["secondary_heads"], list):
        raise ValueError("secondary_heads must be an array or null")


def present_image_fields(raw: dict[str, Any], model_task: str) -> frozenset[str]:
    """Infer selectable fields for in-memory callers lacking catalog metadata."""
    present: set[str] = set()
    if raw.get("probabilities") is not None:
        present.add("class_probabilities")
    if raw.get("decision") is not None:
        present.add("primary_decision")
    if raw.get("embedding") is not None:
        present.update({"embedding", "embedding_normalized"})
    evidence = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}
    if "knn" in evidence or (str(model_task) == "clustering" and "nearest_neighbors" in evidence):
        present.add("knn")
    if "prototype" in evidence:
        present.add("prototype_similarity")
    if raw.get("clustering_evidence") is not None or (str(model_task) == "clustering" and evidence):
        present.add("cluster_evidence")
    if raw.get("secondary_heads"):
        present.add("secondary_heads")
    if raw.get("logits") is not None or raw.get("logits_source") is not None:
        present.add("diagnostics")
    return frozenset(present)


def request_task_for_model(model_task: str) -> str:
    task = str(model_task).strip().lower()
    if task in IMAGE_MODEL_TASKS:
        return "classification"
    if task == "segmentation":
        return "mask_refinement"
    raise ValueError(f"Unsupported V2 model task: {model_task}")


def _class_probabilities(value: Any, artifact_id: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, ArrayPayload):
        scores = np.asarray(value.values).reshape(-1)
        classes = [{"class_index": index, "label_id": None, "label_name": None, "probability": float(score)} for index, score in enumerate(scores)]
    elif isinstance(value, list):
        classes = []
        for row in value:
            if not isinstance(row, dict) or "class_index" not in row or "probability" not in row:
                raise ValueError("class probabilities require class_index and probability")
            classes.append({
                **row,
                "class_index": int(row["class_index"]),
                "label_id": row.get("label_id"),
                "label_name": row.get("label_name"),
                "concept_id": row.get("concept_id"),
                "concept_node_id": row.get("concept_node_id"),
                "probability": float(row["probability"]),
            })
    else:
        raise ValueError("class probabilities must be a class-indexed list")
    if not classes:
        raise ValueError("class probabilities must contain at least one class")
    if len({row["class_index"] for row in classes}) != len(classes):
        raise ValueError("class probabilities contain duplicate class indices")
    if any(not np.isfinite(row["probability"]) or not 0.0 <= row["probability"] <= 1.0 for row in classes):
        raise ValueError("class probabilities must be finite values between zero and one")
    return {"unit": "probability", "source_artifact_id": artifact_id, "classes": classes}


def _embedding(value: Any, normalized: Any, artifact_id: str) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = value if isinstance(value, ArrayPayload) else ArrayPayload(np.asarray(value))
    values = np.asarray(payload.values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("embedding must be a numeric vector")
    if not isinstance(normalized, bool):
        raise ValueError("embedding normalization must be declared")
    return {
        "values": payload,
        "dtype": str(values.dtype),
        "dimension": int(values.shape[0]),
        "normalized": normalized,
        "normalization": "l2" if normalized else "none",
        "source_artifact_id": artifact_id,
    }


def image_output(
    raw: dict[str, Any], *, model_task: str, artifact_id: str,
    selected: frozenset[str], supported: frozenset[str],
) -> dict[str, Any]:
    """Map existing scientific packets into one required-key response shape."""
    task = str(model_task).strip().lower()
    if task not in IMAGE_MODEL_TASKS:
        raise ValueError(f"Unsupported image model task: {model_task}")
    unsupported = selected - supported
    if unsupported:
        raise ValueError(f"Requested output is unsupported by this artifact: {', '.join(sorted(unsupported))}")
    class_evidence = raw.get("evidence") if isinstance(raw.get("evidence"), dict) and "knn" in raw["evidence"] else None
    cluster_packet = raw.get("evidence") if task == "clustering" else raw.get("clustering_evidence")
    if not isinstance(cluster_packet, dict):
        cluster_packet = None
    knn = class_evidence.get("knn") if class_evidence else None
    if knn is not None:
        knn = {**knn, "metric": "cosine_similarity", "source_artifact_id": artifact_id, "source_role": "classification_evidence"}
    elif cluster_packet is not None:
        neighbors = cluster_packet.get("nearest_neighbors")
        if isinstance(neighbors, list):
            knn = {
                "metric": cluster_packet.get("metric", "cosine_similarity"),
                "source_artifact_id": artifact_id, "source_role": "clustering_evidence",
                "k_used": len(neighbors), "neighbors": neighbors,
            }
    prototype = class_evidence.get("prototype") if class_evidence else None
    if prototype is not None:
        prototype = {
            **prototype, "metric": "cosine_similarity", "source_artifact_id": artifact_id,
            "source_role": "classification_evidence",
        }
    if cluster_packet is not None:
        cluster_packet = {**cluster_packet, "source_artifact_id": artifact_id, "source_role": "clustering_evidence"}
    normalized = raw.get("embedding_normalized")
    fields: dict[str, Any] = {
        "class_probabilities": _class_probabilities(raw.get("probabilities"), artifact_id),
        "primary_decision": raw.get("decision"),
        "embedding": _embedding(raw.get("embedding"), normalized, artifact_id),
        "embedding_normalized": normalized,
        "knn": knn,
        "prototype_similarity": prototype,
        "secondary_heads": raw.get("secondary_heads") or [],
        "cluster_evidence": cluster_packet,
    }
    if not isinstance(fields["secondary_heads"], list):
        raise ValueError("secondary_heads must be an array")
    for head in fields["secondary_heads"]:
        if not isinstance(head, dict) or not all(
            isinstance(head.get(key), str) and head[key]
            for key in ("head_id", "head_type", "schema_version", "source_artifact_id")
        ):
            raise ValueError("secondary head results require head_id, head_type, schema_version, and source_artifact_id")
    for key in IMAGE_FIELDS - selected:
        fields[key] = None
    for key in selected & IMAGE_FIELDS:
        if key != "secondary_heads" and fields[key] is None:
            raise ValueError(f"Requested output {key!r} was not produced by this artifact")
    diagnostics = None
    if "diagnostics" in selected:
        diagnostics = {
            "class_logits": raw.get("logits"),
            "class_logits_unit": "logit" if raw.get("logits") is not None else None,
            "logits_source": raw.get("logits_source"),
            "source_artifact_id": artifact_id,
            "class_evidence_schema_version": class_evidence.get("schema_version") if class_evidence else None,
            "cluster_evidence_schema_version": cluster_packet.get("schema_version") if cluster_packet else None,
        }
    return {
        "schema_name": IMAGE_OUTPUT_SCHEMA, "schema_version": IMAGE_OUTPUT_VERSION,
        "type": "image_analysis", "model_task": task,
        **fields, "diagnostics": diagnostics,
    }


def capabilities_for_artifact(run_dir: str | Path, model_task: str) -> dict[str, Any]:
    """Read sanitized metadata from a previously verified artifact snapshot."""
    from oracle_builder.artifacts import read_run_config

    root = Path(run_dir)
    task = str(model_task).strip().lower()
    request_task = request_task_for_model(task)
    try:
        config = read_run_config(root)
    except FileNotFoundError:
        # Only useful for synthetic/adaptor artifacts that carry no config;
        # real admitted run artifacts contain the sealed resolved config.
        config = {"data": {"input_shape": []}, "training": {}}
    data = config.get("data", {})
    shape = list(data.get("input_shape") or [])
    if task == "segmentation":
        target_mode = segmentation_target_mode(config)
        candidate_required = (
            target_mode == CANDIDATE_DELTA or bool(data.get("candidate_sdf"))
            or str(data.get("candidate_distance") or "none").strip().lower() != "none"
            or (len(shape) >= 3 and int(shape[-1]) == 2)
        )
        possible = ["mask", "probability_map", "logits"]
        if target_mode == CANDIDATE_DELTA:
            possible.extend(["delta_mask", "delta_probability_map"])
        return {
            "model_task": task, "request_task": request_task,
            "required_inputs": ["image", "candidate_mask"] if candidate_required else ["image"],
            "default_outputs": ["mask"], "possible_outputs": possible,
            "input_shape": shape, "target_mode": target_mode,
        }
    possible: set[str] = set()
    if task == "classification":
        possible.update({"class_probabilities", "primary_decision", "diagnostics"})
    model_manifest = root / "model" / "model_manifest.json"
    outputs: dict[str, Any] = {}
    if model_manifest.is_file():
        try:
            outputs = json.loads(model_manifest.read_text(encoding="utf-8")).get("outputs", {})
        except (OSError, ValueError):
            outputs = {}
    has_embedding = task in {"embedding", "clustering"} or bool(outputs.get("identity_embedding"))
    if has_embedding:
        possible.update({"embedding", "embedding_normalized"})
    has_class_evidence = (root / "model" / "classification_evidence").exists() or (root / "model" / "classification_evidence.npz").exists()
    has_cluster_evidence = (root / "model" / "clustering_evidence").exists()
    if task == "clustering" and not has_cluster_evidence:
        raise ValueError("Clustering model is missing its evidence index")
    if has_class_evidence or has_cluster_evidence:
        possible.add("knn")
    if has_class_evidence:
        possible.add("prototype_similarity")
    if has_cluster_evidence:
        possible.add("cluster_evidence")
    if task == "clustering":
        possible.discard("diagnostics")
    return {
        "model_task": task, "request_task": request_task,
        "required_inputs": ["image"],
        "default_outputs": sorted(possible), "possible_outputs": sorted(possible),
        "input_shape": shape,
        "output_schema_name": IMAGE_OUTPUT_SCHEMA,
        "output_schema_version": IMAGE_OUTPUT_VERSION,
    }
