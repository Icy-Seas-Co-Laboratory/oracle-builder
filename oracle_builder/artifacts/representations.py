"""Low-memory representation artifacts for completed classification runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from oracle_builder.classification.features import build_feature_model
from oracle_builder.posthoc import export_embedding_cache_stream


def _record(ref) -> dict[str, Any]:
    value = ref.record()
    shape = value.get("original_shape") or []
    return {
        "sample_id": value.get("uuid"),
        "label": value.get("class_index"),
        "split": value.get("split"),
        "resolution_stratum": max(shape[:2]) if len(shape) >= 2 else None,
        "original_roi_shape": shape,
        "metadata": value.get("metadata", {}),
    }


def write_representation_artifacts(
    model,
    dataset,
    index,
    config: dict[str, Any],
    run_dir: str | Path,
) -> dict[str, Any]:
    """Export named representations from an indexed streaming dataset.

    One representation is written at a time into an NPY memmap + JSONL
    provenance cache. The working set is one inference batch, making this safe
    for large datasets even when downstream analysis runs elsewhere.
    """
    settings = config.get("output", {}).get("intermediate_artifacts", {})
    if not settings.get("enabled", True):
        return {"enabled": False, "caches": []}
    if not hasattr(index, "refs"):
        raise TypeError("representation artifacts require an indexed classification dataset")
    feature_model = build_feature_model(model)
    available = set(feature_model.output.keys())
    requested = settings.get(
        "representations",
        ["image_embedding", "fused_embedding", "projection_embedding"],
    )
    if not isinstance(requested, list) or not all(isinstance(item, str) for item in requested):
        raise ValueError("output.intermediate_artifacts.representations must be a list of names")
    selected = [name for name in requested if name in available]
    if not selected and "features" in available:
        selected = ["features"]
    if bool(settings.get("include_feature_map", False)) and "feature_map" in available:
        selected.append("feature_map")
    selected = list(dict.fromkeys(selected))
    output = Path(run_dir) / "intermediates" / "representations" / str(index.split)
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "dataset_id": config.get("dataset", {}).get("dataset_id"),
        "dataset_fingerprint_sha256": config.get("dataset", {}).get("fingerprint_sha256"),
        "artifact_id": config.get("artifact", {}).get("artifact_id"),
        "run_id": config.get("run", {}).get("run_id"),
        "architecture_version": config.get("architecture", {}).get("version", 1),
        "model": config.get("run", {}).get("model"),
        "config": {
            "channels": config.get("preprocessing", {}).get("resolved_channels"),
            "pooling": config.get("pooling"),
            "image_embedding": config.get("image_embedding"),
            "metadata": config.get("metadata"),
            "fusion": config.get("fusion"),
            "classifier": config.get("classifier"),
        },
    }
    reports = []
    for representation in selected:
        # A fresh dataset iterator and model pass per representation trades a
        # modest amount of post-run compute for bounded memory and independent,
        # immediately-valid cache directories.
        def batches():
            for inputs, positions in dataset:
                outputs = feature_model(inputs, training=False)
                values = np.asarray(outputs[representation])
                if values.ndim != 2:
                    # Feature maps are intentionally excluded by default: a
                    # spatial map belongs in a separate analysis workflow.
                    raise ValueError(
                        f"Representation {representation!r} is rank {values.ndim}; only vector caches are supported"
                    )
                positions_array = np.asarray(positions, dtype="int64")
                yield values, positions_array, [_record(index.refs[int(position)]) for position in positions_array]

        cache = export_embedding_cache_stream(
            output / representation,
            batches(),
            count=len(index.refs),
            representation=representation,
            provenance=provenance,
        )
        reports.append({
            "representation": representation,
            "path": str(cache.path.relative_to(Path(run_dir))),
            "shape": list(cache.shape),
            "format": cache.manifest["format"],
            "version": cache.manifest["version"],
        })
    report = {"enabled": True, "split": index.split, "caches": reports}
    report_path = Path(run_dir) / "intermediates" / "representation_artifacts.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
