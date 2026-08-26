from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from oracle_builder.artifacts import (
    create_run_artifact,
    read_run_config,
    read_run_manifest,
    reopen_run_artifact,
    seal_run_artifact,
    update_run_artifact,
    write_run_config,
)
from oracle_builder.artifacts.splits import create_unavailable_split_manifest
from oracle_builder.classification.features import build_feature_model
from oracle_builder.config import (
    DEFAULT_CONFIG,
    deep_merge,
    load_toml,
    normalize_self_supervised_config,
    self_supervised_settings,
)
from oracle_builder.data.sqlite_stream import (
    SQLiteClassificationSource,
    build_all_classification_index,
)
from oracle_builder.datasets.schema import (
    dataset_fingerprint,
    read_dataset_info,
    validate_database,
)
from oracle_builder.environment import write_environment
from oracle_builder.inference.batching import resolve_inference_batch_size
from oracle_builder.registry import get_model_builder
from oracle_builder.saving.load_test import load_model_for_run, run_load_tests
from oracle_builder.saving.save_model import save_model_artifacts, write_load_test_report
from oracle_builder.training.logging_callbacks import (
    init_training_log,
    log_event,
    mark_run_complete,
    write_history_jsonl,
)
from oracle_builder.training.student_teacher import run_self_supervised_training
from oracle_builder.training.train import set_seed
from oracle_builder.training.distribution import select_distribution_strategy, write_distribution_info

from oracle_builder.clustering.evidence import ClusterEvidenceIndex


_SUPPORTED_CLUSTERING_METHODS = {"spherical_kmeans"}


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _validate_clustering_config(config: dict[str, Any], *, sample_count: int | None = None) -> None:
    """Validate options before expensive training or artifact mutation."""
    settings = config["clustering"]
    method = str(settings.get("method", "spherical_kmeans")).lower()
    if method not in _SUPPORTED_CLUSTERING_METHODS:
        supported = ", ".join(sorted(_SUPPORTED_CLUSTERING_METHODS))
        raise ValueError(f"clustering.method must be one of: {supported}")
    settings["method"] = method
    n_clusters = _positive_int(settings.get("n_clusters", 8), "clustering.n_clusters", minimum=2)
    settings["n_clusters"] = n_clusters
    if sample_count is not None and n_clusters > sample_count:
        raise ValueError(
            "clustering.n_clusters must not exceed the number of ROI items "
            f"({n_clusters} requested, {sample_count} available)"
        )
    settings["top_k"] = _positive_int(settings.get("top_k", 5), "clustering.top_k")
    settings["silhouette_max_samples"] = _positive_int(
        settings.get("silhouette_max_samples", 2_000),
        "clustering.silhouette_max_samples",
        minimum=2,
    )
    retained = settings.get("reference_neighbors_per_cluster", 64)
    if isinstance(retained, str) and retained.lower() == "all":
        retained = None
    if retained is not None:
        settings["reference_neighbors_per_cluster"] = _positive_int(
            retained,
            "clustering.reference_neighbors_per_cluster",
        )
    else:
        settings["reference_neighbors_per_cluster"] = None
    settings["fit_batch_size"] = _positive_int(
        settings.get("fit_batch_size", 4_096),
        "clustering.fit_batch_size",
        minimum=2,
    )
    try:
        percentile = float(settings.get("novelty_percentile", 5.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("clustering.novelty_percentile must be a number from 0 through 100") from exc
    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("clustering.novelty_percentile must be a number from 0 through 100")
    settings["novelty_percentile"] = percentile
    self_supervised = self_supervised_settings(config)
    minimum_batch = _positive_int(
        self_supervised.get("minimum_global_batch_size", 2),
        "self_supervised.minimum_global_batch_size",
        minimum=2,
    )
    batch_size = _positive_int(config.get("data", {}).get("batch_size", 16), "data.batch_size")
    if batch_size < minimum_batch:
        raise ValueError(
            "data.batch_size must be at least self_supervised.minimum_global_batch_size"
        )
    config["data"]["batch_size"] = batch_size
    threshold = float(self_supervised.get("collapse_std_threshold", 1e-3))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("self_supervised.collapse_std_threshold must be finite and positive")
    self_supervised["collapse_std_threshold"] = threshold
    spread_threshold = float(
        self_supervised.get("collapse_embedding_spread_threshold", 0.005)
    )
    if not np.isfinite(spread_threshold) or spread_threshold <= 0:
        raise ValueError(
            "self_supervised.collapse_embedding_spread_threshold must be finite and positive"
        )
    self_supervised["collapse_embedding_spread_threshold"] = spread_threshold


def _validate_self_supervised_history(history: Any, config: dict[str, Any]) -> None:
    values = getattr(history, "history", {})
    for name, series in values.items():
        if not np.all(np.isfinite(np.asarray(series, dtype="float64"))):
            raise ValueError(f"Clustering self-supervised training produced non-finite {name}")
    series = values.get("representation_std")
    if not series:
        raise ValueError(
            "Clustering self-supervised training did not report representation_std"
        )
    if float(series[-1]) <= float(self_supervised_settings(config)["collapse_std_threshold"]):
        raise ValueError("Clustering self-supervised training collapsed")


# Legacy import compatibility.
_validate_pretraining_history = _validate_self_supervised_history


def _validate_serving_embeddings(embeddings: np.ndarray, config: dict[str, Any]) -> None:
    values = np.asarray(embeddings, dtype="float32")
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Clustering produced non-finite serving embeddings")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Clustering produced zero-norm serving embeddings")
    centered_rms = float(np.sqrt(np.mean(np.square(values - values.mean(axis=0)))))
    threshold = float(self_supervised_settings(config)["collapse_embedding_spread_threshold"])
    if centered_rms <= threshold:
        raise ValueError(
            "Clustering produced collapsed serving embeddings: "
            f"centered_rms={centered_rms:.6g} <= threshold={threshold:.6g}"
        )


def _frozen_classification_index(input_path: str | Path, config: dict[str, Any]):
    """Return every frozen ROI without applying a supervised split manifest."""
    with sqlite3.connect(Path(input_path).expanduser().resolve()) as connection:
        lifecycle = str(connection.execute("SELECT lifecycle FROM dataset WHERE singleton = 1").fetchone()[0])
    if lifecycle != "frozen":
        raise ValueError("Clustering requires a frozen dataset checkpoint")
    # An encoder artifact may carry a train/validation/test manifest from the
    # dataset it was trained on. That manifest is intentionally irrelevant to
    # de novo fitting, including when the target ROI database is different.
    # The scoped external-inference flag makes build_all_classification_index
    # assign each item to its all-inclusive inference view without mutating the
    # resolved artifact configuration.
    index_config = dict(config)
    index_config["_external_inference"] = True
    index = build_all_classification_index(input_path, index_config, labeled_only=False)
    if not index.refs:
        raise ValueError("Clustering dataset contains no ROI items")
    _validate_clustering_config(config, sample_count=len(index.refs))
    return index


def _resolve_polarity(config: dict[str, Any], dataset_info: dict[str, Any]) -> None:
    imaging = dataset_info.get("metadata", {}).get("imaging", {})
    imaging = imaging if isinstance(imaging, dict) else {}
    polarity = imaging.get("polarity", {})
    polarity = polarity if isinstance(polarity, dict) else {}
    source_polarity = str(polarity.get("value", "unknown"))
    requested = config["preprocessing"].get("invert", "auto")
    if requested == "auto":
        resolved = source_polarity == "dark_on_light"
        method = "dataset_source_polarity"
    elif isinstance(requested, bool):
        resolved = requested
        method = "explicit_config"
    else:
        raise ValueError("preprocessing.invert must be true, false, or 'auto'")
    config["preprocessing"]["invert"] = resolved
    config["preprocessing"]["polarity_resolution"] = {
        "requested": requested,
        "source_polarity": source_polarity,
        "method": method,
        "model_polarity": "light_on_dark",
    }
    config["dataset"]["imaging"] = imaging


def resolve_clustering_config(
    config_path: str | Path,
    input_path: str | Path,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Resolve a clustering config without requiring classification labels."""
    user = load_toml(config_path)
    config = deep_merge(DEFAULT_CONFIG, user)
    normalize_self_supervised_config(config, user)
    config.setdefault("run", {})["task"] = "clustering"
    config["run"].setdefault("model", "resnet")
    if "input_shape" not in config.get("data", {}):
        raise ValueError("Clustering config requires data.input_shape")
    # The classifier head is disposable but must have at least two outputs so
    # Keras does not construct a degenerate one-class softmax. Clustering uses
    # only the named embedding layer.
    config["data"]["num_classes"] = 2
    config["training"]["loss"] = "sparse_categorical_crossentropy"
    config["training"]["metrics"] = []
    self_supervised = self_supervised_settings(config)
    self_supervised["enabled"] = True
    method = str(self_supervised.get("method", "byol")).lower()
    if method not in {"byol", "student_teacher", "simclr"}:
        raise ValueError("Clustering self_supervised.method must be byol, student_teacher, or simclr")
    config.setdefault("clustering", {})
    config["clustering"].setdefault("method", "spherical_kmeans")
    config["clustering"].setdefault("n_clusters", 8)
    config["clustering"].setdefault("novelty_percentile", 5.0)
    config["clustering"].setdefault("top_k", 5)
    config["clustering"].setdefault("silhouette_max_samples", 2_000)
    # Persist only representative references by default. This keeps serving
    # neighbor lookup bounded while the cluster statistics still reflect every
    # fitted ROI. Set this to null for a full reference index.
    config["clustering"].setdefault("reference_neighbors_per_cluster", 64)
    config["clustering"].setdefault("fit_batch_size", 4_096)
    _validate_clustering_config(config)

    database = Path(input_path).expanduser().resolve()
    with sqlite3.connect(database) as connection:
        report = validate_database(connection)
        if not report["valid"]:
            raise ValueError("Dataset validation failed: " + "; ".join(report["errors"]))
        info = read_dataset_info(connection)
        if info["dataset_type"] != "classification":
            raise ValueError("Clustering currently requires a classification Dataset V1 database")
        input_fingerprint = dataset_fingerprint(connection)
        config["dataset"] = {
            "dataset_id": info["dataset_id"],
            "revision_id": info["revision_id"],
            "parent_revision_id": info.get("parent_revision_id"),
            "dataset_type": info["dataset_type"],
            "schema_name": info["schema_name"],
            "schema_version": info["schema_version"],
            "version": info.get("version"),
            "lifecycle": info["lifecycle"],
            "fingerprint_sha256": dataset_fingerprint(connection),
            "labels": [],
        }
    _resolve_polarity(config, info)
    config["paths"] = {
        "config_path": str(Path(config_path).resolve()),
        "input_path": str(database),
        "run_dir": str(Path(run_dir).resolve()),
    }
    return config


def _extract_embeddings(model, dataset, count: int, embedding_dim: int) -> np.ndarray:
    feature_model = build_feature_model(model)
    embeddings = np.empty((count, embedding_dim), dtype="float32")
    for images, positions in dataset:
        outputs = feature_model(images, training=False)
        features = np.asarray(outputs["features"], dtype="float32")
        positions = np.asarray(positions, dtype="int64")
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = np.divide(features, norms, out=np.zeros_like(features), where=norms > 0)
        embeddings[positions] = features
    return embeddings


def train_clustering_run(
    config_path: str | Path,
    input_path: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = Path(output).expanduser().resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output artifact already exists: {output}")
        import shutil

        shutil.rmtree(output)
    config = resolve_clustering_config(config_path, input_path, output)
    output.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    config["run"]["run_id"] = run_id
    config["run"]["run_name"] = output.name
    manifest = create_run_artifact(
        output,
        run_id=run_id,
        name=output.name,
        config=config,
        source_config=config_path,
    )
    create_unavailable_split_manifest(output, config, reason="unsupervised clustering run")
    config["artifact"] = {
        "artifact_id": manifest["artifact_id"],
        "schema_name": manifest["artifact_schema"]["name"],
        "schema_version": manifest["artifact_schema"]["version"],
    }
    write_run_config(output, config)
    environment = write_environment(output)
    init_training_log(output / "logs" / "training.sqlite", run_id, output.name, config, environment)

    try:
        index = _frozen_classification_index(input_path, config)
        source = SQLiteClassificationSource(input_path, config)
        batch_size = int(config["data"].get("batch_size", 16))
        pretraining_dataset = source.image_dataset(index, batch_size=batch_size, shuffle=True)
        set_seed(int(config["run"].get("seed", 123)))
        strategy, distribution_info = select_distribution_strategy(config)
        if (
            str(self_supervised_settings(config).get("method", "byol")).lower() == "simclr"
            and batch_size // int(strategy.num_replicas_in_sync) < 2
        ):
            raise ValueError(
                "SimCLR requires at least two samples per synchronized replica; "
                f"global batch size is {batch_size} across "
                f"{strategy.num_replicas_in_sync} replicas"
            )
        with strategy.scope():
            model = get_model_builder(config["run"]["model"])(config)
        write_distribution_info(distribution_info, output)
        log_event(
            output / "logs" / "training.sqlite",
            run_id,
            "INFO",
            "Configured TensorFlow distribution strategy",
            {
                "requested_strategy": distribution_info.requested_strategy,
                "resolved_strategy": distribution_info.resolved_strategy,
                "replicas": distribution_info.replicas,
                "devices": distribution_info.devices,
                "global_batch_size": distribution_info.global_batch_size,
                "per_replica_batch_size": distribution_info.per_replica_batch_size,
                "cross_device_ops": distribution_info.cross_device_ops,
            },
        )
        log_event(
            output / "logs" / "training.sqlite",
            run_id,
            "INFO",
            "Started self-supervised clustering encoder training",
            {"method": self_supervised_settings(config)["method"], "samples": len(index.refs)},
        )
        history = run_self_supervised_training(
            model,
            pretraining_dataset,
            config,
            output,
            strategy=strategy,
            training_log=output / "logs" / "training.sqlite",
            run_id=run_id,
        )
        _validate_self_supervised_history(history, config)
        history_path = output / "metrics" / "history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history.history, indent=2, default=float) + "\n")
        write_history_jsonl(
            history.history,
            output / "metrics" / "metrics.jsonl",
            run_id=run_id,
            phase="self_supervised",
        )

        inference_plan = resolve_inference_batch_size(model, config, announce=None)
        extraction_dataset = source.indexed_image_dataset(index, batch_size=inference_plan.batch_size)
        embedding_dim = int(model.get_layer("features").output.shape[-1])
        embeddings = _extract_embeddings(model, extraction_dataset, len(index.refs), embedding_dim)
        _validate_serving_embeddings(embeddings, config)
        cluster_index = ClusterEvidenceIndex.fit(
            embeddings,
            [ref.uuid for ref in index.refs],
            n_clusters=int(config["clustering"]["n_clusters"]),
            seed=int(config["run"].get("seed", 123)),
            novelty_percentile=float(config["clustering"].get("novelty_percentile", 5.0)),
            silhouette_max_samples=int(config["clustering"]["silhouette_max_samples"]),
            reference_neighbors_per_cluster=config["clustering"]["reference_neighbors_per_cluster"],
            fit_batch_size=int(config["clustering"]["fit_batch_size"]),
        )
        if cluster_index.silhouette is not None and cluster_index.silhouette <= 0:
            raise ValueError(
                "Clustering produced non-positive cosine silhouette: "
                f"{cluster_index.silhouette:.6g}"
            )
        cluster_path = output / "model" / "clustering_evidence"
        cluster_index.save(cluster_path)
        config["clustering"]["structure"] = cluster_index.summary()
        write_run_config(output, config)
        log_event(
            output / "logs" / "training.sqlite",
            run_id,
            "INFO",
            "Built clustering evidence index",
            cluster_index.summary(),
        )

        save_report = save_model_artifacts(model, output, config)
        load_report = run_load_tests(output, config, save_report)
        write_load_test_report(output, load_report)
        summary = {
            "clustering": cluster_index.summary(),
            "training": {key: values[-1] for key, values in history.history.items() if values},
            "inference_batching": inference_plan.to_dict(),
            "distribution": {
                "requested_strategy": distribution_info.requested_strategy,
                "resolved_strategy": distribution_info.resolved_strategy,
                "replicas": distribution_info.replicas,
            },
            "load_test": load_report,
        }
        update_run_artifact(output, status="complete", summary=summary)
        mark_run_complete(output / "logs" / "training.sqlite", run_id, "complete")
        sealed = seal_run_artifact(output)
        return {"run_dir": str(output), "artifact_id": sealed["artifact_id"], **summary}
    except Exception as exc:
        update_run_artifact(output, status="failed", error=f"{type(exc).__name__}: {exc}")
        mark_run_complete(output / "logs" / "training.sqlite", run_id, "failed")
        raise


def fit_clustering_evidence_from_encoder(
    config_path: str | Path,
    input_path: str | Path,
    encoder_run: str | Path,
    *,
    reopen_and_reseal: bool = False,
) -> dict[str, Any]:
    """Fit and attach clustering evidence to a compatible sealed encoder run.

    Attachment is deliberately in-place: the existing model and its resolved
    preprocessing remain authoritative, allowing a classification bundle to
    expose clustering evidence without retraining or duplicating model assets.
    A sealed artifact is never reopened implicitly.
    """
    encoder_run = Path(encoder_run).expanduser().resolve()
    if not reopen_and_reseal:
        raise ValueError(
            "Attaching clustering evidence changes the encoder artifact; "
            "pass reopen_and_reseal=True to explicitly reopen and reseal it"
        )
    manifest = read_run_manifest(encoder_run)
    source_config = read_run_config(encoder_run)
    task = str(source_config.get("run", {}).get("task", ""))
    if task not in {"classification", "clustering"}:
        raise ValueError("Existing encoder artifact must be a classification or clustering run")

    requested = load_toml(config_path)
    requested_shape = requested.get("data", {}).get("input_shape")
    source_shape = source_config.get("data", {}).get("input_shape")
    if requested_shape is not None and list(requested_shape) != list(source_shape or []):
        raise ValueError(
            "data.input_shape must match the existing encoder artifact "
            f"({source_shape})"
        )
    # The artifact's model/preprocessing is the compatibility contract. Only
    # clustering controls are taken from the fitting config.
    config = deep_merge(source_config, {"clustering": requested.get("clustering", {})})
    config.setdefault("clustering", {})
    config["clustering"].setdefault("method", "spherical_kmeans")
    config["clustering"].setdefault("n_clusters", 8)
    config["clustering"].setdefault("novelty_percentile", 5.0)
    config["clustering"].setdefault("top_k", 5)
    config["clustering"].setdefault("silhouette_max_samples", 2_000)
    config["clustering"].setdefault("reference_neighbors_per_cluster", 64)
    config["clustering"].setdefault("fit_batch_size", 4_096)
    _validate_clustering_config(config)

    database = Path(input_path).expanduser().resolve()
    with sqlite3.connect(database) as connection:
        report = validate_database(connection)
        if not report["valid"]:
            raise ValueError("Dataset validation failed: " + "; ".join(report["errors"]))
        info = read_dataset_info(connection)
        if info["dataset_type"] != "classification":
            raise ValueError("Clustering currently requires a classification Dataset V1 database")
        input_fingerprint = dataset_fingerprint(connection)
    index = _frozen_classification_index(database, config)

    # Loading validates the sealed source artifact before it can be reopened.
    model = load_model_for_run(encoder_run, source_config, prefer_savedmodel=False)
    try:
        embedding_dim = int(model.get_layer("features").output.shape[-1])
    except (AttributeError, ValueError) as exc:
        raise ValueError("Existing artifact does not expose a named 'features' embedding layer") from exc
    expected_dim = int(source_config.get("model", {}).get("embedding_dim", embedding_dim))
    if embedding_dim != expected_dim:
        raise ValueError(
            "Existing encoder embedding dimension is incompatible with its config "
            f"({embedding_dim} model, {expected_dim} config)"
        )

    source = SQLiteClassificationSource(database, config)
    inference_plan = resolve_inference_batch_size(model, config, announce=None)
    extraction_dataset = source.indexed_image_dataset(index, batch_size=inference_plan.batch_size)
    embeddings = _extract_embeddings(model, extraction_dataset, len(index.refs), embedding_dim)
    cluster_index = ClusterEvidenceIndex.fit(
        embeddings,
        [ref.uuid for ref in index.refs],
        n_clusters=int(config["clustering"]["n_clusters"]),
        seed=int(config["run"].get("seed", 123)),
        novelty_percentile=float(config["clustering"]["novelty_percentile"]),
        silhouette_max_samples=int(config["clustering"]["silhouette_max_samples"]),
        reference_neighbors_per_cluster=config["clustering"]["reference_neighbors_per_cluster"],
        fit_batch_size=int(config["clustering"]["fit_batch_size"]),
    )

    cluster_path = encoder_run / "model" / "clustering_evidence"
    if cluster_path.exists():
        raise FileExistsError(
            f"Existing artifact already has clustering evidence: {cluster_path}; "
            "refusing to replace it"
        )
    # Prepare all new evidence outside the sealed artifact. The artifact is
    # reopened only for the final, explicit attach operation.
    with tempfile.TemporaryDirectory(
        prefix=f".{encoder_run.name}.clustering-", dir=encoder_run.parent
    ) as staging_dir:
        staged_cluster_path = Path(staging_dir) / "clustering_evidence"
        cluster_index.save(staged_cluster_path)
        reopen_run_artifact(
            encoder_run,
            reason="Attach clustering evidence fitted from a frozen ROI dataset",
        )
        try:
            shutil.move(str(staged_cluster_path), str(cluster_path))
            config["clustering"]["structure"] = cluster_index.summary()
            config["clustering"]["encoder_artifact"] = {
                "artifact_id": manifest["artifact_id"],
                "run_id": manifest["run_id"],
                "fingerprint_sha256": manifest.get("fingerprint_sha256"),
            }
            config["clustering"]["source_dataset"] = {
                "dataset_id": info["dataset_id"],
                "revision_id": info["revision_id"],
                "fingerprint_sha256": input_fingerprint,
            }
            write_run_config(encoder_run, config)
            model_manifest_path = encoder_run / "model" / "model_manifest.json"
            model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
            outputs = model_manifest.setdefault("outputs", {})
            outputs.update(
                {
                    "cluster_evidence": True,
                    "cluster_count": cluster_index.cluster_count,
                    "cluster_method": cluster_index.method,
                }
            )
            model_manifest.setdefault("postprocessing", {})["clustering_evidence"] = config["clustering"]
            model_manifest_path.write_text(
                json.dumps(model_manifest, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            sealed = seal_run_artifact(encoder_run)
        except Exception:
            # Do not leave an explicitly reopened artifact in an indeterminate
            # lifecycle state. Any written files are included in the new seal.
            seal_run_artifact(encoder_run)
            raise
    return {
        "run_dir": str(encoder_run),
        "artifact_id": sealed["artifact_id"],
        "clustering": cluster_index.summary(),
        "inference_batching": inference_plan.to_dict(),
        "attached": True,
    }
