"""Train a self-supervised encoder as a first-class embedding run."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from oracle_builder.artifacts import (
    RunLayout,
    create_run_artifact,
    create_unavailable_split_manifest,
    seal_run_artifact,
    update_run_artifact,
    write_run_config,
)
from oracle_builder.config import resolve_config, self_supervised_settings
from oracle_builder.data.sqlite_stream import (
    SQLiteClassificationSource,
    build_all_classification_index,
)
from oracle_builder.environment import write_environment
from oracle_builder.training.distribution import select_distribution_strategy, write_distribution_info
from oracle_builder.training.logging_callbacks import (
    append_jsonl_event,
    init_training_log,
    log_event,
    mark_run_complete,
    write_history_jsonl,
)
from oracle_builder.training.train import set_seed


def _validate_embedding_inputs(config: dict[str, Any], input_path: str | Path) -> None:
    with sqlite3.connect(Path(input_path).expanduser().resolve()) as connection:
        dataset_info = connection.execute(
            "SELECT dataset_type, lifecycle FROM dataset WHERE singleton = 1"
        ).fetchone()
    if not dataset_info or dataset_info[0] != "classification":
        raise ValueError("Embedding training requires a classification dataset")
    if dataset_info[1] != "frozen":
        raise ValueError("Embedding training requires a frozen dataset checkpoint")
    settings = self_supervised_settings(config)
    if not settings.get("enabled", False):
        raise ValueError("Embedding training requires self_supervised.enabled = true")
    if str(settings.get("method", "byol")).lower() == "grayscale_reconstruction":
        raise ValueError("grayscale_reconstruction is not an embedding training method")


def train_embedding_run(
    config_path: str | Path,
    input_path: str | Path,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a sealed embedding training record without fitting clusters.

    The classifier head is an internal training scaffold.  Only the shared
    encoder is saved in the model product and exposed by its contract.
    """
    output = Path(output).expanduser().resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output artifact already exists: {output}")
        import shutil

        shutil.rmtree(output)
    config = resolve_config(config_path, input_path, output)
    config.setdefault("run", {})["task"] = "embedding"
    _validate_embedding_inputs(config, input_path)
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
    create_unavailable_split_manifest(output, config, reason="self-supervised embedding run")
    config["artifact"] = {
        "artifact_id": manifest["artifact_id"],
        "schema_name": manifest["artifact_schema"]["name"],
        "schema_version": manifest["artifact_schema"]["version"],
    }
    write_run_config(output, config)
    environment = write_environment(output)
    layout = RunLayout(output)
    init_training_log(layout.training_log, run_id, output.name, config, environment)
    try:
        set_seed(int(config["run"].get("seed", 123)))
        index = build_all_classification_index(input_path, config, labeled_only=False)
        if not index.refs:
            raise ValueError("Embedding dataset contains no ROI images")
        source = SQLiteClassificationSource(input_path, config)
        dataset = source.image_dataset(
            index,
            batch_size=int(config["data"].get("batch_size", 16)),
            shuffle=True,
        )
        strategy, distribution_info = select_distribution_strategy(config)
        write_distribution_info(distribution_info, output)
        log_event(
            layout.training_log,
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
        from oracle_builder.registry import get_model_builder
        from oracle_builder.training.student_teacher import run_self_supervised_training

        with strategy.scope():
            classifier = get_model_builder(config["run"]["model"])(config)
        log_event(
            layout.training_log,
            run_id,
            "INFO",
            "Started self-supervised embedding training",
            {"method": self_supervised_settings(config).get("method", "byol"), "samples": len(index.refs)},
        )
        history = run_self_supervised_training(
            classifier,
            dataset,
            config,
            output,
            strategy=strategy,
            training_log=layout.training_log,
            run_id=run_id,
        )
        from oracle_builder.classification.features import build_embedding_model
        from oracle_builder.saving.load_test import run_load_tests
        from oracle_builder.saving.save_model import save_model_artifacts, write_load_test_report

        embedding_model = build_embedding_model(classifier)
        history_path = layout.metrics_json.parent / "history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_payload = json.dumps(history.history, indent=2, default=float) + "\n"
        history_path.write_text(history_payload)
        layout.metrics_json.write_text(history_payload)
        pd.DataFrame(history.history).to_csv(layout.metrics_csv, index_label="epoch")
        write_history_jsonl(history.history, layout.metrics_jsonl, run_id=run_id, phase="self_supervised")
        save_report = save_model_artifacts(embedding_model, output, config)
        load_report = run_load_tests(output, config, save_report)
        write_load_test_report(output, load_report)
        summary = {
            "embedding": {
                "dimension": int(config.get("model", {}).get("embedding_dim", 256)),
                "normalized": bool(config.get("model", {}).get("normalize_embeddings", True)),
                "source_reference_count": len(index.refs),
            },
            "training": {key: values[-1] for key, values in history.history.items() if values},
            "distribution": {
                "requested_strategy": distribution_info.requested_strategy,
                "resolved_strategy": distribution_info.resolved_strategy,
                "replicas": distribution_info.replicas,
            },
            "load_test": load_report,
        }
        update_run_artifact(output, status="complete", summary=summary)
        mark_run_complete(layout.training_log, run_id, "complete")
        sealed = seal_run_artifact(output)
        return {"run_dir": str(output), "artifact_id": sealed["artifact_id"], **summary}
    except Exception as exc:
        append_jsonl_event(layout.events_jsonl, run_id, "ERROR", "embedding_training_failed", {"error": str(exc)})
        update_run_artifact(output, status="failed", error=f"{type(exc).__name__}: {exc}")
        mark_run_complete(layout.training_log, run_id, "failed")
        seal_run_artifact(output)
        raise
