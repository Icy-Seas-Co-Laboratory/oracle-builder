"""Fit a new post-hoc head to an existing sealed classification model."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sqlite3
import uuid
from typing import Any, Callable, Mapping

import numpy as np

from oracle_builder.artifacts import (
    attach_split_manifest,
    create_run_artifact,
    create_split_manifest,
    read_run_config,
    read_run_manifest,
    seal_run_artifact,
    update_run_artifact,
    write_run_config,
)
from oracle_builder.datasets.schema import dataset_fingerprint, read_dataset_info
from oracle_builder.environment import write_environment
from oracle_builder.posthoc.training import train_posthoc_classifier
from oracle_builder.posthoc.shared_cache import reusable_representation_cache
from oracle_builder.posthoc.cache import open_embedding_cache
from oracle_builder.data.sqlite_stream import SQLiteClassificationSource, build_classification_index
from oracle_builder.classification.features import predict_classification_outputs
from oracle_builder.evaluation.classification import ClassificationMetricAccumulator, classification_epoch_metric_records, write_classification_evaluation
from oracle_builder.saving.load_test import load_model_for_run, run_load_tests
from oracle_builder.saving.save_model import write_load_test_report


def _label_signature(labels: list[Mapping[str, Any]]) -> list[tuple[int, str]]:
    return [(int(row["class_index"]), str(row["name"])) for row in labels]


def _neural_baseline(model: Any, dataset: Path, config: dict[str, Any], source_config: Mapping[str, Any],
                     posthoc_root: Path, labels: list[Mapping[str, Any]], fingerprint: str) -> dict[str, Any]:
    """Evaluate the original neural output on exactly the post-hoc test items."""
    neural_labels = source_config.get("neural_labels", source_config.get("dataset", {}).get("labels", []))
    if _label_signature(list(neural_labels)) != _label_signature(labels):
        return {"status": "incompatible", "reason": "source neural label vocabulary differs from target dataset"}
    index = build_classification_index(dataset, config, "test", labeled_only=True)
    cache = open_embedding_cache(posthoc_root.parent.parent / "intermediates" / "representations" / "test" / "fused_embedding")
    records = list(cache.iter_records())
    targets = np.asarray([row["label"] for row in records], dtype="int64")
    source = SQLiteClassificationSource(dataset, config)
    probabilities = []
    for images in source.image_dataset(index, batch_size=int(config.get("data", {}).get("batch_size", 16))).as_numpy_iterator():
        probabilities.append(np.asarray(predict_classification_outputs(model, images)["probabilities"], dtype="float64"))
    values = np.concatenate(probabilities)
    if values.shape != (len(targets), len(labels)):
        return {"status": "incompatible", "reason": "source neural output dimension differs from target vocabulary"}
    predicted = np.argmax(values, axis=1)
    names = {int(row["class_index"]): str(row["name"]) for row in labels}
    metrics = {row["metric"]: row["value"] for row in classification_epoch_metric_records(targets, values, class_names=names)
               if "label" not in row and "bin_index" not in row and row.get("value") is not None and np.isfinite(row["value"])}
    accumulator = ClassificationMetricAccumulator(values.shape[1]); accumulator.update(targets, values)
    sample_rows = [{**record, "y_true": int(target), "y_pred": int(pred), "confidence": float(row[int(pred)])}
                   for record, target, pred, row in zip(records, targets, predicted, values, strict=True)]
    evidence = posthoc_root / "neural_baseline"
    write_classification_evaluation(targets, predicted, sample_rows, evidence, class_names=names,
                                    probability_metrics=accumulator.result(), probabilities=values,
                                    calibration_rows=accumulator.calibration_rows(),
                                    evaluation_context={"split": "test", "head": "source_neural_baseline"})
    signature = {"dataset_fingerprint_sha256": fingerprint,
                 "samples": [{"id": row.get("sample_id", row.get("item_id", row.get("source_key", str(i)))), "label": int(label)}
                             for i, (row, label) in enumerate(zip(records, targets, strict=True))],
                 "classes": {str(key): value for key, value in names.items()}}
    import hashlib
    return {"status": "complete", "metrics": metrics,
            "evaluation_signature": hashlib.sha256(json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "path": "model/posthoc/neural_baseline/evaluation"}


def fit_existing_model_posthoc(
    model_run: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    name: str | None = None,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    split_strategy: str = "auto",
    posthoc_settings: Mapping[str, Any] | None = None,
    embedding_cache_root: str | Path | None = None,
    progress: Callable[[str, str, Mapping[str, Any] | None], None] | None = None,
) -> dict[str, Any]:
    """Produce a separate sealed model product; never alter the source model."""
    source = Path(model_run).expanduser().resolve()
    dataset = Path(dataset_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    report = progress or (lambda _event, _message, _data=None: None)
    if output.exists():
        raise FileExistsError(output)
    source_manifest = read_run_manifest(source)
    if source_manifest.get("lifecycle") != "sealed" or source_manifest.get("status") != "complete":
        raise ValueError("Post-hoc fitting requires a sealed, completed model")
    if source_manifest.get("artifact_type") not in {"model_run", "model_product"}:
        raise ValueError("Post-hoc fitting requires a model-run or model-product artifact")
    source_config = read_run_config(source)
    if source_config.get("run", {}).get("task") != "classification":
        raise ValueError("Post-hoc fitting currently requires a classification model")
    if not 0 < validation_split < 1 or not 0 < test_split < 1 or validation_split + test_split >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than one")
    if split_strategy not in {"auto", "random", "source_partitions", "class_stratified", "stratified_group", "group_only"}:
        raise ValueError("Unsupported split strategy")
    with sqlite3.connect(dataset) as connection:
        info = read_dataset_info(connection)
        if info.get("dataset_type") != "classification" or info.get("lifecycle") != "frozen":
            raise ValueError("Post-hoc fitting requires a frozen classification dataset")
        fingerprint = dataset_fingerprint(connection)
        labels = [
            {"label_id": row[0], "class_index": int(row[1]), "name": row[2]}
            for row in connection.execute(
                "SELECT label_id, class_index, name FROM classification_labels ORDER BY class_index"
            )
        ]
    if len(labels) < 2 or [row["class_index"] for row in labels] != list(range(len(labels))):
        raise ValueError("Post-hoc classes must have contiguous indices starting at zero")
    source_model = source / "model"
    for filename in ("final.keras", "model_manifest.json", "contract.json"):
        if not (source_model / filename).is_file():
            raise ValueError(f"Selected model has no portable {filename} output")
    source_config_path = source / "config" / "source.toml"
    if not source_config_path.is_file():
        raise ValueError("Selected model has no portable source configuration")

    run_id = str(uuid.uuid4())
    product_config = json.loads(json.dumps(source_config, default=str))
    product_config.pop("_split_manifest", None)
    product_config.setdefault("run", {})["run_id"] = run_id
    product_config["run"]["run_name"] = name or f"{source_manifest.get('name', source.name)} + post-hoc"
    from oracle_builder.posthoc.pipelines import normalize_posthoc_settings
    selected_posthoc = normalize_posthoc_settings(posthoc_settings or {})
    product_config["posthoc"] = {
        **selected_posthoc,
        "enabled": True, "representation": "fused_embedding",
        "training_dataset_id": info["dataset_id"],
        "training_dataset_fingerprint_sha256": fingerprint,
    }
    source_lineage = source_manifest.get("lineage") or {}
    source_evaluation_head = "posthoc" if (source_manifest.get("summary") or {}).get("evaluation_head") == "posthoc" or source_lineage.get("source_model_artifact_id") else "neural"
    original_neural_artifact_id = source_lineage.get("original_neural_artifact_id") or (source_manifest["artifact_id"] if source_evaluation_head == "neural" else None)
    original_neural_dataset_id = source_lineage.get("original_neural_dataset_id") or (source_manifest.get("dataset_id") if source_evaluation_head == "neural" else None)
    original_neural_dataset_fingerprint = source_lineage.get("original_neural_dataset_fingerprint_sha256") or (source_manifest.get("dataset_fingerprint_sha256") if source_evaluation_head == "neural" else None)
    lineage = {
        "source_model_artifact_id": source_manifest["artifact_id"],
        "source_evaluation_head": source_evaluation_head,
        "original_neural_artifact_id": original_neural_artifact_id,
        "original_neural_dataset_id": original_neural_dataset_id,
        "original_neural_dataset_fingerprint_sha256": original_neural_dataset_fingerprint,
        "source_model_fingerprint_sha256": source_manifest.get("fingerprint_sha256"),
        "source_training_dataset_id": source_manifest.get("dataset_id") or source_config.get("dataset", {}).get("dataset_id"),
        "source_training_dataset_fingerprint_sha256": source_manifest.get("dataset_fingerprint_sha256") or source_config.get("dataset", {}).get("fingerprint_sha256"),
        "posthoc_dataset_id": info["dataset_id"],
        "posthoc_dataset_fingerprint_sha256": fingerprint,
    }
    product_config["lineage"] = lineage
    fit_config = json.loads(json.dumps(product_config))
    fit_config["dataset"] = {
        "dataset_id": info["dataset_id"],
        "revision_id": info.get("revision_id"),
        "fingerprint_sha256": fingerprint,
        "dataset_type": "classification",
        "labels": labels,
    }
    fit_config.setdefault("data", {})["num_classes"] = len(labels)
    fit_config["data"]["validation_split"] = validation_split
    fit_config["data"]["test_split"] = test_split
    fit_config["data"]["split_strategy"] = split_strategy
    fit_config.setdefault("run", {})["seed"] = selected_posthoc["seed"]
    product_config["neural_labels"] = source_config.get("dataset", {}).get("labels", [])
    product_config["dataset"] = fit_config["dataset"]

    report("posthoc_preparing", "Creating a derived model product", {"source_model_artifact_id": source_manifest["artifact_id"]})
    product_manifest = create_run_artifact(
        output, run_id=run_id, name=product_config["run"]["run_name"],
        config=product_config, source_config=source_config_path,
        artifact_type="model_product", lineage=lineage,
    )
    try:
        write_environment(output)
        shutil.copytree(
            source_model, output / "model", dirs_exist_ok=True,
            ignore=lambda directory, names: {"posthoc"} if Path(directory) == source_model else set(),
        )
        # Epoch history belongs to the unchanged neural model. Keep its usual
        # path so the model inspector can plot it, but record its provenance.
        if (source / "metrics").is_dir():
            shutil.copytree(source / "metrics", output / "metrics", dirs_exist_ok=True)
        metrics_provenance = {
            "kind": "source_neural_training",
            "source_model_artifact_id": original_neural_artifact_id,
            "source_dataset_id": original_neural_dataset_id,
            "source_dataset_fingerprint_sha256": original_neural_dataset_fingerprint,
        }
        (output / "metrics").mkdir(exist_ok=True)
        (output / "metrics" / "provenance.json").write_text(
            json.dumps(metrics_provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for dirname in ("evaluation", "figures"):
            source_directory = source / dirname
            if source_directory.is_dir():
                shutil.copytree(source_directory, output / "provenance" / "source_model" / dirname)
        neural_evidence = source / "provenance" / "neural" / "evaluation"
        if not neural_evidence.is_dir() and source_evaluation_head == "neural":
            neural_evidence = source / "evaluation"
        if neural_evidence.is_dir():
            shutil.copytree(neural_evidence, output / "provenance" / "neural" / "evaluation")
        source_posthoc = source_model / "posthoc"
        if (source_posthoc / "report.json").is_file():
            target = output / "provenance" / "source_model" / "posthoc"
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_posthoc / "report.json", target / "report.json")
            for dirname in ("evaluation", "figures"):
                if (source_posthoc / dirname).is_dir():
                    shutil.copytree(source_posthoc / dirname, target / dirname)
        model_manifest_path = output / "model" / "model_manifest.json"
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        model_manifest["artifact_id"] = product_manifest["artifact_id"]
        model_manifest["run_id"] = run_id
        model_manifest["source_model_artifact_id"] = source_manifest["artifact_id"]
        model_manifest.get("outputs", {}).pop("posthoc", None)
        model_manifest_path.write_text(json.dumps(model_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        contract_path = output / "model" / "contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract.get("outputs", {}).pop("posthoc", None)
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        product_config["artifact"] = {"artifact_id": product_manifest["artifact_id"]}
        write_run_config(output, product_config)
        artifact_path = output / "artifact.json"
        artifact_manifest = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact_manifest["model"] = source_manifest["model"]
        artifact_path.write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report("posthoc_splitting", "Assigning train, validation, and test examples", {"dataset_id": info["dataset_id"]})
        split = create_split_manifest(output, dataset, fit_config, verified_dataset_fingerprint=fingerprint)
        attach_split_manifest(fit_config, split)
        report("posthoc_embedding", "Exporting model embeddings and tuning the classifier", {})
        model = load_model_for_run(source, source_config)
        cache_identity = {
            "source_model_artifact_id": source_manifest["artifact_id"],
            "source_model_fingerprint_sha256": source_manifest["fingerprint_sha256"],
            "dataset_fingerprint_sha256": fingerprint,
            # Assignments, rather than this derived product's random run id,
            # make reuse valid only for precisely the same three split sets.
            "split_assignments": split["assignments"],
            "representation": selected_posthoc["representation"],
        }
        with reusable_representation_cache(output, embedding_cache_root, cache_identity):
            result = train_posthoc_classifier(model, dataset, fit_config, output, progress=report)
        result["neural_baseline"] = _neural_baseline(
            model, dataset, fit_config, source_config, output / "model" / "posthoc", labels, fingerprint,
        )
        result["training_dataset"] = {"dataset_id": info["dataset_id"], "fingerprint_sha256": fingerprint, "labels": labels}
        result["source_model_artifact_id"] = source_manifest["artifact_id"]
        (output / "model" / "posthoc" / "report.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Standard artifact inspection and comparison read top-level evaluation.
        # Populate it with the new head's held-out scores, never old neural scores.
        for dirname in ("evaluation", "figures"):
            generated = output / "model" / "posthoc" / dirname
            if generated.is_dir():
                shutil.copytree(generated, output / dirname, dirs_exist_ok=True)
        report("posthoc_validating", "Checking the derived model bundle", {})
        write_load_test_report(output, run_load_tests(output, product_config, {}))
        posthoc_evaluation = json.loads((output / "evaluation" / "evaluation_summary.json").read_text(encoding="utf-8"))
        update_run_artifact(output, status="complete", summary={
            "evaluation": posthoc_evaluation,
            "evaluation_head": "posthoc",
            "posthoc": result,
            "neural_history": metrics_provenance,
        })
        sealed = seal_run_artifact(output)
        report("posthoc_complete", "Post-hoc model product is ready", {"artifact_id": sealed["artifact_id"]})
        return sealed
    except Exception as exc:
        update_run_artifact(output, status="failed", error=str(exc))
        seal_run_artifact(output)
        raise
