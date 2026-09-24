"""Training controller for resolution-stratified classification models.

The controller deliberately owns the epoch loop.  A training item may move to
the adjacent smaller stratum between epochs, so a single, static ``tf.data``
pipeline cannot describe the complete run.  Validation and all post-training
consumers use canonical (non-random) routing.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tensorflow as tf
from tensorflow import keras

from oracle_builder.classification.stratification import (
    batch_plan,
    dimensions,
    enabled,
    assignment_policy,
    settings,
    stratum_for_shape,
    summarize_records,
    supra_epoch_schedule,
    training_stratum,
    validate,
)
from oracle_builder.classification.stratified_data import add_stratum_dimension_input
from oracle_builder.data.sqlite_stream import (
    SQLiteClassificationSource,
    SQLiteDatasetBundle,
    SQLiteSplitIndex,
    build_classification_index,
)
from oracle_builder.training.distribution import (
    select_distribution_strategy,
    write_distribution_info,
)
from oracle_builder.saving.save_model import write_stratification_manifest


RECOVERY_SCHEMA = {
    "name": "oracle_builder_stratified_training_recovery",
    "version": "1.0.0",
}
MANIFEST_SCHEMA = {
    "name": "oracle_builder_stratified_classification_bundle",
    "version": "1.0.0",
}


def recovery_config_hash(config: dict[str, Any]) -> str:
    """Match the generic recovery contract without importing training UI code."""
    value = {
        "run": {
            key: item
            for key, item in config.get("run", {}).items()
            if key != "run_name"
        },
        "data": config.get("data", {}),
        "model": config.get("model", {}),
        "classification": config.get("classification", {}),
        "preprocessing": config.get("preprocessing", {}),
        "training": config.get("training", {}),
        "distribution": config.get("distribution", {}),
        "dataset": config.get("dataset", {}),
        "split_manifest": {
            key: config.get("_split_manifest", {}).get(key)
            for key in ("split_manifest_id", "fingerprint_sha256", "dataset")
        },
    }
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StratifiedChildResult:
    dimension: int
    batch_size: int
    completed_epochs: int
    stopped_early: bool
    model_path: str
    summary_path: str
    history_path: str
    canonical_counts: dict[str, int]


@dataclass(frozen=True)
class StratifiedTrainingResult:
    children: dict[int, StratifiedChildResult]
    manifest_path: Path
    recovery_path: Path
    split_summaries: dict[str, dict[str, Any]]

    def model_path(self, dimension: int) -> Path:
        return self.manifest_path.parent.parent / self.children[int(dimension)].model_path

    def load_model(self, dimension: int) -> keras.Model:
        # Finalization performs inference/evaluation only; avoiding optimizer
        # deserialization also keeps custom training losses out of this path.
        return keras.models.load_model(self.model_path(dimension), compile=False)


def write_stratified_metric_artifacts(
    run_dir: str | Path,
    run_id: str,
    config: dict[str, Any],
    result: StratifiedTrainingResult | None = None,
) -> dict[str, Any]:
    """Publish standard metric artifacts for the shared stratified schedule."""
    root = Path(run_dir)
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    block_size = int(settings(config).get("supra_epochs", 5))
    schedule = str(settings(config).get("schedule", "interleaved_steps"))
    histories: dict[str, dict[str, list[float]]] = {}
    rows: list[dict[str, Any]] = []
    history_sources = (
        {
            int(dimension): root / child.history_path
            for dimension, child in result.children.items()
        }
        if result is not None
        else {
            int(dimension): root / "model" / "strata" / str(dimension) / "training_history.json"
            for dimension in dimensions(config)
        }
    )
    for dimension, history_path in sorted(history_sources.items()):
        history = _read_history(history_path)
        histories[str(dimension)] = history
        for metric, values in sorted(history.items()):
            split = "validation" if metric.startswith("val_") else "train"
            metric_name = metric[4:] if split == "validation" else metric
            for stratum_epoch, value in enumerate(values, start=1):
                # Interleaved histories have one entry per common parent
                # epoch.  The old contiguous layout is retained for legacy
                # artifacts, where an entry still maps to its local epoch.
                parent_epoch = stratum_epoch
                rows.append(
                    {
                        "run_id": run_id,
                        "phase": "stratified_training",
                        "stratum_dimension": int(dimension),
                        "stratum_epoch": stratum_epoch,
                        "parent_epoch": parent_epoch,
                        "epoch": parent_epoch - 1,
                        "split": split,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )
    history_payload = {
        "schema": "oracle_builder_stratified_history",
        "schema_version": "1.0.0",
        "weight_sharing": "shared",
        "supra_epochs": block_size,
        "schedule": schedule,
        "strata": histories,
    }
    (metrics_dir / "history.json").write_text(
        json.dumps(history_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "run_id", "phase", "stratum_dimension", "stratum_epoch",
        "parent_epoch", "epoch", "split", "metric", "value",
    ]
    with (metrics_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    timestamp = datetime.now(timezone.utc).isoformat()
    jsonl_path = metrics_dir / "metrics.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "schema": "oracle_training_metric",
                        "schema_version": "1.0.0",
                        "metric_id": str(uuid.uuid4()),
                        "timestamp": timestamp,
                        **row,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    log_path = root / "logs" / "training.sqlite"
    if log_path.exists():
        with sqlite3.connect(log_path) as connection:
            connection.execute("DELETE FROM epoch_metrics WHERE run_id = ?", (run_id,))
            connection.executemany(
                "INSERT INTO epoch_metrics VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        row["run_id"],
                        int(row["epoch"]),
                        row["split"],
                        f"stratum_{row['stratum_dimension']}_{row['metric']}",
                        row["value"],
                    )
                    for row in rows
                ],
            )
    return {
        "history_path": "metrics/history.json",
        "csv_path": "metrics/history.csv",
        "jsonl_path": "metrics/metrics.jsonl",
        "metric_records": len(rows),
        "strata": sorted(histories),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def child_config(config: dict[str, Any], dimension: int) -> dict[str, Any]:
    """Return an isolated resolved config for one child model."""
    configured = dimensions(config)
    if int(dimension) not in configured:
        raise ValueError(f"Unknown classification stratum {dimension}; expected {configured}")
    result = copy.deepcopy(config)
    original_shape = list(result["data"]["input_shape"])
    tail = original_shape[2:] if len(original_shape) > 2 else []
    result["data"]["input_shape"] = [int(dimension), int(dimension), *tail]
    result["data"]["batch_size"] = int(batch_plan(config)[int(dimension)])
    result.setdefault("classification", {}).setdefault("stratification", {})[
        "_active_dimension"
    ] = int(dimension)
    return result


def shared_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build the one spatially-dynamic model configuration for all strata."""
    result = copy.deepcopy(config)
    input_shape = list(result["data"]["input_shape"])
    if len(input_shape) != 3:
        raise ValueError("Classification data.input_shape must be [height, width, channels]")
    result["data"]["input_shape"] = [None, None, input_shape[-1]]
    result.setdefault("classification", {}).setdefault("stratification", {})[
        "weight_sharing"
    ] = "shared"
    return result


def routed_index(
    index: SQLiteSplitIndex,
    config: dict[str, Any],
    dimension: int,
    *,
    epoch: int | None = None,
) -> SQLiteSplitIndex:
    """Filter an index for a child, using stochastic routing only for training."""
    selected = []
    configured = dimensions(config)
    stochastic = index.split == "train" and epoch is not None
    for ref in index.refs:
        record = ref.record()
        shape = record.get("original_shape")
        if not shape:
            raise ValueError(
                f"Stratification requires original dimensions for item {ref.item_id!r}"
            )
        canonical = stratum_for_shape(
            shape, configured, policy=assignment_policy(config)
        )
        assigned = (
            training_stratum(
                canonical,
                item_id=str(ref.uuid),
                epoch=int(epoch),
                config=config,
            )
            if stochastic
            else canonical
        )
        if assigned == int(dimension):
            selected.append(ref)
    return SQLiteSplitIndex(index.sqlite_path, index.split, selected)


def build_indices(
    sqlite_path: str | Path, config: dict[str, Any]
) -> dict[str, SQLiteSplitIndex]:
    result = {}
    for split in ("train", "validation", "test"):
        index = build_classification_index(
            sqlite_path, config, split, labeled_only=True
        )
        if index.refs:
            result[split] = index
    if "train" not in result:
        raise ValueError("Dataset must contain or create a train split")
    return result


def make_canonical_bundle(
    sqlite_path: str | Path,
    config: dict[str, Any],
    dimension: int,
    *,
    indices: dict[str, SQLiteSplitIndex] | None = None,
) -> SQLiteDatasetBundle:
    """Build unshuffled, unaugmented canonical datasets for evaluation/evidence."""
    child = child_config(config, dimension)
    source = SQLiteClassificationSource(sqlite_path, child)
    base = indices or build_indices(sqlite_path, config)
    child_indices: dict[str, SQLiteSplitIndex] = {}
    datasets = {}
    for split, index in base.items():
        selected = routed_index(index, config, dimension)
        if not selected.refs:
            continue
        child_indices[split] = selected
        datasets[split] = add_stratum_dimension_input(
            source.training_dataset(selected, shuffle=False, augment=False), dimension, config
        )
    return SQLiteDatasetBundle(datasets, child_indices, source)


def _split_summaries(
    indices: dict[str, SQLiteSplitIndex], config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        split: summarize_records(list(index.iter_records()), config, split=split)
        for split, index in indices.items()
    }


def _canonical_counts(
    indices: dict[str, SQLiteSplitIndex], config: dict[str, Any], dimension: int
) -> dict[str, int]:
    return {
        split: len(routed_index(index, config, dimension))
        for split, index in indices.items()
    }


def epoch_stratum_schedule(config: dict[str, Any], total_epochs: int):
    """Legacy epoch-major view retained for callers that need individual turns."""
    for epoch in range(int(total_epochs)):
        for dimension in dimensions(config):
            yield epoch, dimension


def _history_path(child_dir: Path) -> Path:
    return child_dir / "training_history.json"


def _read_history(child_dir: Path) -> dict[str, list[float]]:
    path = child_dir if Path(child_dir).name == "training_history.json" else _history_path(child_dir)
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): [float(item) for item in values] for key, values in value.items()}


def _append_history(
    history: dict[str, list[float]], epoch_history: dict[str, list[Any]]
) -> None:
    for name, values in epoch_history.items():
        if values:
            history.setdefault(str(name), []).append(float(values[-1]))


def _train_interleaved_epoch(
    model: keras.Model,
    datasets: dict[int, tf.data.Dataset],
    *,
    steps_per_turn: int = 1,
    on_batch_loading=None,
    on_batch_begin=None,
    on_batch_end=None,
) -> dict[int, dict[str, float]]:
    """Run one finite, round-robin update pass over resolution batches.

    ``tf.data`` batches retain their own spatial dimensions, while the shared
    model has dynamic spatial inputs.  Calling ``train_on_batch`` here is
    deliberate: concatenating these batches would force an artificial common
    resize and undo resolution stratification.  Values are sample-weighted so
    a final short batch cannot dominate a stratum history.
    """
    iterators = {dimension: iter(dataset) for dimension, dataset in datasets.items()}
    totals: dict[int, dict[str, float]] = {dimension: {} for dimension in datasets}
    counts: dict[int, int] = {dimension: 0 for dimension in datasets}
    active = list(sorted(iterators))
    batch_index = 0
    while active:
        for dimension in list(active):
            for _ in range(steps_per_turn):
                if on_batch_loading is not None:
                    on_batch_loading(batch_index)
                try:
                    batch = next(iterators[dimension])
                except StopIteration:
                    active.remove(dimension)
                    break
                # ``train_on_batch`` otherwise returns metric objects
                # accumulated from previous strata. Resetting here makes the
                # per-stratum history a true sample-weighted batch summary.
                model.reset_metrics()
                if on_batch_begin is not None:
                    on_batch_begin(batch_index)
                values = model.train_on_batch(*batch, return_dict=True)
                if on_batch_end is not None:
                    on_batch_end(batch_index, values)
                batch_index += 1
                inputs = batch[0]
                if isinstance(inputs, dict):
                    first = next(iter(inputs.values()))
                else:
                    first = inputs
                sample_count = int(tf.shape(first)[0])
                counts[dimension] += sample_count
                for name, value in values.items():
                    totals[dimension][str(name)] = (
                        totals[dimension].get(str(name), 0.0)
                        + float(value) * sample_count
                    )
    return {
        dimension: {
            name: value / counts[dimension]
            for name, value in values.items()
        }
        for dimension, values in totals.items()
        if counts[dimension]
    }


def _recovery_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "model" / "recovery" / "stratified_state.json"


def _recovery_model_path(run_dir: str | Path) -> Path:
    """Return the single rolling shared-model snapshot for a stratified run."""
    return Path(run_dir) / "model" / "recovery" / "latest.keras"


def read_recovery_state(run_dir: str | Path) -> dict[str, Any]:
    path = _recovery_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"No stratified recovery snapshot found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_recovery_state(
    run_dir: str | Path,
    config: dict[str, Any],
    *,
    artifact_id: str,
    run_id: str,
) -> dict[str, Any]:
    state = read_recovery_state(run_dir)
    if state.get("schema") != RECOVERY_SCHEMA:
        raise ValueError(f"Unsupported stratified recovery schema: {state.get('schema')}")
    if state.get("artifact_id") != artifact_id or state.get("run_id") != run_id:
        raise ValueError("Stratified recovery state does not belong to this run artifact")
    if state.get("config_sha256") != recovery_config_hash(config):
        raise ValueError("Stratified recovery state does not match the training contract")
    if state.get("dimensions") != dimensions(config):
        raise ValueError("Stratified recovery dimensions do not match the configuration")
    root = Path(run_dir)
    if state.get("weight_sharing") != "shared":
        raise ValueError(
            "This recovery snapshot predates shared-weight stratification; "
            "start a new stratified training run."
        )
    shared = state.get("shared", {})
    if not isinstance(shared, dict) or not shared.get("model_path"):
        raise ValueError("Shared-weight stratified recovery is missing its model snapshot")
    shared_target = root / str(shared["model_path"])
    if not shared_target.exists() or _sha256(shared_target) != shared.get("model_sha256"):
        raise ValueError("Shared stratified recovery model checksum mismatch")
    for key, child in state.get("children", {}).items():
        completed = int(child.get("completed_epochs", 0))
        if completed < 0 or completed > int(config["training"].get("epochs", 10)):
            raise ValueError(f"Invalid completed epoch for stratum {key}")
        model_path = child.get("model_path")
        if completed and not model_path:
            raise ValueError(f"Missing recovery model path for stratum {key}")
        if model_path:
            target = root / model_path
            if not target.exists() or _sha256(target) != child.get("model_sha256"):
                raise ValueError(f"Recovery model checksum mismatch for stratum {key}")
    return state


def clear_recovery_state(run_dir: str | Path) -> None:
    _recovery_path(run_dir).unlink(missing_ok=True)
    _recovery_model_path(run_dir).unlink(missing_ok=True)
    # Remove the former per-stratum layout as well, so completed artifacts do
    # not retain obsolete rolling snapshots after an upgrade.
    for dimension in (Path(run_dir) / "model" / "strata").glob("*/recovery"):
        for filename in ("latest.keras",):
            (dimension / filename).unlink(missing_ok=True)


def _initial_recovery(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "schema": RECOVERY_SCHEMA,
        "artifact_id": config.get("artifact", {}).get("artifact_id"),
        "run_id": run_id,
        "phase": "supervised_stratified",
        "config_sha256": recovery_config_hash(config),
        "dimensions": dimensions(config),
        "batch_plan": {str(k): v for k, v in batch_plan(config).items()},
        "supra_epochs": int(config.get("classification", {}).get("stratification", {}).get("supra_epochs", 5)),
        "schedule": str(settings(config).get("schedule", "interleaved_steps")),
        "weight_sharing": "shared",
        "children": {},
    }


def _recovered_children(
    state: dict[str, Any],
    config: dict[str, Any],
    run_path: Path,
    indices: dict[str, SQLiteSplitIndex],
) -> dict[int, StratifiedChildResult]:
    """Rebuild manifest entries for strata completed before a resume.

    Recovery stores progress rather than the public bundle manifest.  The
    controller can legitimately skip an already-completed supra-epoch block,
    so these entries must be available before the first in-progress manifest
    is written again.
    """
    recovered_children = state.get("children", {})
    if not isinstance(recovered_children, dict):
        return {}
    result: dict[int, StratifiedChildResult] = {}
    for dimension in dimensions(config):
        recovered = recovered_children.get(str(dimension), {})
        if not isinstance(recovered, dict):
            continue
        completed_epochs = int(recovered.get("completed_epochs", 0))
        if completed_epochs <= 0:
            continue
        child = child_config(config, dimension)
        control = recovered.get("control", {})
        if not isinstance(control, dict):
            control = {}
        child_dir = run_path / "model" / "strata" / str(dimension)
        result[dimension] = StratifiedChildResult(
            dimension=dimension,
            batch_size=int(control.get("effective_batch_size", child["data"]["batch_size"])),
            completed_epochs=completed_epochs,
            stopped_early=bool(control.get("stopped_early", False)),
            model_path="model/shared/final.keras",
            summary_path=(child_dir / "model_summary.txt").relative_to(run_path).as_posix(),
            history_path=_history_path(child_dir).relative_to(run_path).as_posix(),
            canonical_counts=_canonical_counts(indices, config, dimension),
        )
    return result


def _save_recovery_model(
    model: keras.Model,
    run_dir: Path,
    dimension: int,
    completed_epochs: int,
    state: dict[str, Any],
    history: dict[str, list[float]],
    control: dict[str, Any],
) -> None:
    # The weights are shared by every stratum.  Keeping the snapshot in the
    # conventional run-level recovery directory makes it discoverable and
    # avoids duplicating a potentially large Keras archive for every stratum.
    recovery_dir = run_dir / "model" / "recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    target = _recovery_model_path(run_dir)
    temporary = recovery_dir / f".latest.{uuid.uuid4().hex}.keras"
    model.save(temporary)
    os.replace(temporary, target)
    relative = target.relative_to(run_dir).as_posix()
    checksum = _sha256(target)
    # Child progress is historical metadata; every child intentionally points
    # to this one *current* shared model.  Refreshing existing entries avoids
    # retaining stale checksums after the next stratum updates the weights.
    for child in state["children"].values():
        child["model_path"] = relative
        child["model_sha256"] = checksum
    state["children"][str(dimension)] = {
        "completed_epochs": int(completed_epochs),
        "model_path": relative,
        "model_sha256": checksum,
        "history": history,
        "control": control,
    }
    _atomic_json(_recovery_path(run_dir), state)


def _is_improvement(value: float, best: float | None, mode: str) -> bool:
    return best is None or (value > best if mode == "max" else value < best)


def _monitor_mode(name: str) -> str:
    return "max" if any(token in name.lower() for token in ("acc", "f1", "auc")) else "min"


def _cycle_validation(
    model: keras.Model,
    sqlite_path: str | Path,
    indices: dict[str, SQLiteSplitIndex],
    config: dict[str, Any],
    effective_batches: dict[int, int],
) -> dict[str, Any]:
    """Evaluate stable shared weights over every canonical validation stratum."""
    validation = indices.get("validation")
    if validation is None or not validation.refs:
        return {"aggregate": {}, "per_stratum": {}, "sample_count": 0}
    scheduler = settings(config).get("cycle_scheduler", {})
    aggregation = (
        str(scheduler.get("aggregation", "sample_weighted"))
        if isinstance(scheduler, dict)
        else "sample_weighted"
    )
    weighted: dict[str, float] = {}
    weights: dict[str, float] = {}
    per_stratum: dict[str, dict[str, Any]] = {}
    for dimension in dimensions(config):
        selected = routed_index(validation, config, dimension)
        if not selected.refs:
            continue
        child = child_config(config, dimension)
        child["data"]["batch_size"] = int(
            effective_batches.get(dimension, child["data"]["batch_size"])
        )
        dataset = add_stratum_dimension_input(
            SQLiteClassificationSource(sqlite_path, child).training_dataset(
                selected, shuffle=False, augment=False
            ), dimension, config,
        )
        values = {
            str(name): float(value)
            for name, value in model.evaluate(dataset, verbose=0, return_dict=True).items()
        }
        count = len(selected)
        per_stratum[str(dimension)] = {"samples": count, "metrics": values}
        weight = float(count) if aggregation == "sample_weighted" else 1.0
        for name, value in values.items():
            weighted[name] = weighted.get(name, 0.0) + weight * value
            weights[name] = weights.get(name, 0.0) + weight
    return {
        "aggregate": {
            name: weighted[name] / weights[name] for name in sorted(weighted)
        },
        "per_stratum": per_stratum,
        "sample_count": sum(row["samples"] for row in per_stratum.values()),
        "aggregation": aggregation,
    }


def _cycle_monitor_name(config: dict[str, Any]) -> str:
    monitor = str(config.get("callbacks", {}).get("checkpoint_monitor", "val_loss"))
    return monitor[4:] if monitor.startswith("val_") else monitor


def _write_manifest(
    run_dir: Path,
    config: dict[str, Any],
    split_summaries: dict[str, dict[str, Any]],
    children: dict[int, StratifiedChildResult],
    *,
    status: str,
) -> Path:
    payload = {
        "schema_name": MANIFEST_SCHEMA["name"],
        "schema_version": MANIFEST_SCHEMA["version"],
        "status": status,
        "dimensions": dimensions(config),
        "basis": "max_original_dimension",
        "assignment_policy": assignment_policy(config),
        "training_routing": config.get("classification", {})
        .get("stratification", {})
        .get("training_routing", {}),
        "batch_plan": {str(k): v for k, v in batch_plan(config).items()},
        "supra_epochs": int(
            config.get("classification", {})
            .get("stratification", {})
            .get("supra_epochs", 5)
        ),
        "schedule": str(settings(config).get("schedule", "interleaved_steps")),
        "weight_sharing": "shared",
        "cycle_scheduler": config.get("classification", {})
        .get("stratification", {})
        .get("cycle_scheduler", {}),
        "split_summaries": split_summaries,
        "children": [
            {
                **asdict(value),
                "path": f"strata/{dimension}",
                "config": {"data": {"batch_size": value.batch_size}},
            }
            for dimension, value in sorted(children.items())
        ],
    }
    return write_stratification_manifest(run_dir, payload)


def train_stratified_models(
    config: dict[str, Any],
    sqlite_path: str | Path,
    run_dir: str | Path,
    training_log: str | Path,
    run_id: str,
    *,
    resume_state: dict[str, Any] | None = None,
) -> StratifiedTrainingResult:
    """Train and persist every configured resolution child model.

    Children are trained sequentially so several large networks are not kept in
    GPU memory simultaneously.  Routing is rebuilt for each epoch.  The return
    value contains paths rather than live models for the same reason.
    """
    if not enabled(config):
        raise ValueError("train_stratified_models requires stratification.enabled = true")
    validate(config)
    # Keep heavyweight evaluation/logging imports out of policy-only callers
    # (and out of dataset worker processes).
    from oracle_builder.training.logging_callbacks import log_event
    from oracle_builder.training.train import (
        build_and_compile_model,
        write_model_summary,
    )
    from oracle_builder.training.status import RichTrainingStatusCallback
    run_path = Path(run_dir)
    indices = build_indices(sqlite_path, config)
    split_summaries = _split_summaries(indices, config)
    total_epochs = int(config["training"].get("epochs", 10))
    save_every = max(1, int(config.get("recovery", {}).get("save_every_epochs", 1)))
    recovery_enabled = bool(config.get("recovery", {}).get("enabled", True))
    for dimension in dimensions(config):
        if not routed_index(indices["train"], config, dimension).refs:
            raise ValueError(
                f"Resolution stratum {dimension} has no canonical training samples"
            )

    strategy, distribution_info = select_distribution_strategy(config)
    write_distribution_info(distribution_info, run_path)
    log_event(
        training_log,
        run_id,
        "INFO",
        "Started stratified classification training",
        {
            "dimensions": dimensions(config),
            "batch_plan": batch_plan(config),
            "weight_sharing": "shared",
            "assignment_policy": assignment_policy(config),
            "supra_epochs": int(
                config.get("classification", {})
                .get("stratification", {})
                .get("supra_epochs", 5)
            ),
            "schedule": str(settings(config).get("schedule", "interleaved_steps")),
            "split_summaries": split_summaries,
            "distribution": asdict(distribution_info),
        },
    )
    print(
        "Stratified training enabled\n"
        f"  dimensions: {dimensions(config)}\n"
        f"  batch plan: {batch_plan(config)}\n"
        "  weights: shared dynamic-spatial model\n"
        f"  canonical routing: {assignment_policy(config)}\n"
        f"  supra-epochs: {config.get('classification', {}).get('stratification', {}).get('supra_epochs', 5)}",
        flush=True,
    )
    state = resume_state or _initial_recovery(config, run_id)
    children = _recovered_children(state, config, run_path, indices)
    if children:
        message = (
            "Restored completed stratified children from recovery: "
            + ", ".join(
                f"{dimension}x{dimension} (epoch {child.completed_epochs})"
                for dimension, child in sorted(children.items())
            )
        )
        print(message, flush=True)
        log_event(
            training_log,
            run_id,
            "INFO",
            "Restored stratified children from recovery",
            {
                "children": {
                    str(dimension): child.completed_epochs
                    for dimension, child in sorted(children.items())
                }
            },
        )
    cycle_control = dict(state.get("cycle_scheduler", {}))
    completed_cycles = {
        int(value) for value in cycle_control.get("completed_block_stops", [])
    }
    stop_training = bool(cycle_control.get("stopped_early", False))
    shared_config = shared_model_config(config)
    shared_recovery = state.get("shared", {})
    with strategy.scope():
        if shared_recovery.get("model_path"):
            model = keras.models.load_model(run_path / shared_recovery["model_path"])
        else:
            model = build_and_compile_model(shared_config)

    interleaved_step_mode = str(settings(config).get("schedule", "interleaved_steps")) == "interleaved_steps"
    round_logs: dict[int, dict[int, dict[str, float]]] = {}
    interleaved_status = None
    if interleaved_step_mode:
        interleaved_status = RichTrainingStatusCallback(
            phase="Stratified shared model · interleaved resolutions",
            epochs=total_epochs,
            display=str(config.get("training", {}).get("display", "rich")),
            training_log=training_log,
            run_id=run_id,
        )
        interleaved_status.set_model(model)
        interleaved_status.set_params({"epochs": total_epochs, "steps": None})
        interleaved_status.on_train_begin()
    # Each child remains resident for a supra-epoch block, avoiding a model
    # reload and graph rebuild for every individual parent epoch.
    for block_start, block_stop, dimension in supra_epoch_schedule(config, total_epochs):
        if stop_training:
            break
        child = child_config(config, dimension)
        child_dir = run_path / "model" / "strata" / str(dimension)
        child_dir.mkdir(parents=True, exist_ok=True)
        recovered = state.get("children", {}).get(str(dimension), {})
        initial_epoch = int(recovered.get("completed_epochs", 0))
        history = {
            str(key): [float(item) for item in values]
            for key, values in recovered.get("history", _read_history(child_dir)).items()
        }
        control = dict(recovered.get("control", {}))
        effective_batch_size = int(
            control.get("effective_batch_size", child["data"]["batch_size"])
        )
        child["data"]["batch_size"] = effective_batch_size
        summary_path = child_dir / "model_summary.txt"
        write_model_summary(model, summary_path)
        canonical_validation = routed_index(
            indices["validation"], config, dimension
        ) if "validation" in indices else None
        validation_source = SQLiteClassificationSource(sqlite_path, child)
        validation_data = (
            add_stratum_dimension_input(validation_source.training_dataset(
                canonical_validation, shuffle=False, augment=False
            ), dimension, config)
            if canonical_validation is not None and canonical_validation.refs
            else None
        )
        # Shared-weight early stopping and LR adaptation happen after a whole
        # resolution cycle, never from this stratum in isolation.
        stopped_early = False
        completed_epochs = initial_epoch

        for epoch in range(
            max(initial_epoch, block_start),
            min(total_epochs, block_stop),
        ):
            selected = routed_index(
                indices["train"], config, dimension, epoch=epoch
            )
            routing_summary = summarize_records(
                list(indices["train"].iter_records()),
                config,
                split="train",
                epoch=epoch,
            )
            log_event(
                training_log,
                run_id,
                "INFO",
                "Training resolution stratum epoch",
                {
                    "dimension": dimension,
                    "epoch": epoch + 1,
                    "samples": len(selected),
                    "routing": routing_summary,
                },
            )
            print(
                f"Training stratum {dimension}x{dimension}, epoch "
                f"{epoch + 1}/{total_epochs}: {len(selected)} samples",
                flush=True,
            )
            if selected.refs and not interleaved_step_mode:
                rich_status = RichTrainingStatusCallback(
                    phase=(
                        f"Stratified {dimension}×{dimension} "
                        f"· {len(selected):,} routed samples "
                        f"· parent epoch {epoch + 1}/{total_epochs}"
                    ),
                    epochs=total_epochs,
                    display=str(config.get("training", {}).get("display", "rich")),
                    training_log=training_log,
                    run_id=run_id,
                )
                while True:
                    source = SQLiteClassificationSource(sqlite_path, child)
                    train_data = add_stratum_dimension_input(source.training_dataset(
                        selected, shuffle=True, augment=True
                    ), dimension, config)
                    try:
                        epoch_history = model.fit(
                            train_data,
                            validation_data=validation_data,
                            initial_epoch=epoch,
                            epochs=epoch + 1,
                            callbacks=[rich_status],
                            verbose=0,
                        )
                        break
                    except tf.errors.ResourceExhaustedError:
                        next_batch_size = max(1, effective_batch_size // 2)
                        if next_batch_size == effective_batch_size:
                            raise
                        previous_batch_size = effective_batch_size
                        effective_batch_size = next_batch_size
                        child["data"]["batch_size"] = effective_batch_size
                        control["effective_batch_size"] = effective_batch_size
                        validation_source = SQLiteClassificationSource(sqlite_path, child)
                        validation_data = (
                            add_stratum_dimension_input(validation_source.training_dataset(
                                canonical_validation, shuffle=False, augment=False
                            ), dimension, config)
                            if canonical_validation is not None and canonical_validation.refs
                            else None
                        )
                        message = (
                            f"GPU memory exhausted in {dimension}x{dimension}; retrying "
                            f"epoch {epoch + 1} with batch size {effective_batch_size} "
                            f"(was {previous_batch_size})"
                        )
                        print(message, flush=True)
                        log_event(
                            training_log, run_id, "WARNING", "Reduced stratum batch after GPU OOM",
                            {
                                "dimension": dimension,
                                "epoch": epoch + 1,
                                "previous_batch_size": previous_batch_size,
                                "batch_size": effective_batch_size,
                            },
                        )
                _append_history(history, epoch_history.history)
                logs = {
                    name: float(values[-1])
                    for name, values in epoch_history.history.items()
                    if values
                }
            elif interleaved_step_mode:
                # Build every finite stratum dataset once, then alternate one
                # optimizer update per dimension until all are exhausted.
                # This is the shared-model schedule; individual model.fit
                # calls above remain the legacy contiguous implementation.
                if dimension == dimensions(config)[0]:
                    if interleaved_status is not None:
                        interleaved_status.on_epoch_begin(epoch)
                    round_datasets: dict[int, tf.data.Dataset] = {}
                    for round_dimension in dimensions(config):
                        round_child = child_config(config, round_dimension)
                        round_selected = routed_index(
                            indices["train"], config, round_dimension, epoch=epoch
                        )
                        if not round_selected.refs:
                            continue
                        round_source = SQLiteClassificationSource(sqlite_path, round_child)
                        round_datasets[round_dimension] = add_stratum_dimension_input(
                            round_source.training_dataset(
                                round_selected, shuffle=True, augment=True
                            ),
                            round_dimension,
                            config,
                        )
                    round_logs[epoch] = _train_interleaved_epoch(
                        model,
                        round_datasets,
                        steps_per_turn=int(
                            settings(config).get("steps_per_stratum", 1)
                        ),
                        on_batch_loading=(
                            interleaved_status.on_input_batch_loading
                            if interleaved_status is not None
                            else None
                        ),
                        on_batch_begin=(
                            interleaved_status.on_train_batch_begin
                            if interleaved_status is not None
                            else None
                        ),
                        on_batch_end=(
                            interleaved_status.on_train_batch_end
                            if interleaved_status is not None
                            else None
                        ),
                    )
                logs = round_logs.get(epoch, {}).get(dimension, {})
                _append_history(history, {name: [value] for name, value in logs.items()})
                if not selected.refs:
                    log_event(
                        training_log,
                        run_id,
                        "WARNING",
                        "No samples routed to resolution stratum for epoch",
                        {"dimension": dimension, "epoch": epoch + 1},
                    )
            else:
                logs = {}
                log_event(
                    training_log,
                    run_id,
                    "WARNING",
                    "No samples routed to resolution stratum for epoch",
                    {"dimension": dimension, "epoch": epoch + 1},
                )

            completed = epoch + 1
            completed_epochs = completed
            _atomic_json(_history_path(child_dir), history)
            # Make the normal run-level metric artifacts available during a
            # long stratified run, not only after finalization.
            write_stratified_metric_artifacts(run_path, run_id, config)
            if bool(config.get("output", {}).get("save_checkpoints", False)):
                checkpoint = child_dir / "checkpoints" / f"epoch_{completed:03d}.keras"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                model.save(checkpoint)
            log_event(
                training_log,
                run_id,
                "INFO",
                "Completed resolution stratum epoch",
                {
                    "dimension": dimension,
                    "epoch": completed,
                    "metrics": logs,
                    "stopped_early": stopped_early,
                },
            )
            # Persist the resident shared model periodically so recovery keeps
            # both its weights and optimizer state across stratum changes.
            if recovery_enabled and (completed % save_every == 0 or stopped_early):
                snapshot_data = {
                    "dimension": dimension,
                    "completed_epoch": completed,
                    "path": "model/recovery/latest.keras",
                }
                log_event(
                    training_log,
                    run_id,
                    "INFO",
                    "Saving rolling stratified recovery snapshot",
                    snapshot_data,
                )
                print(
                    f"Saving stratified recovery snapshot after {dimension}x{dimension} "
                    f"epoch {completed}",
                    flush=True,
                )
                _save_recovery_model(
                    model,
                    run_path,
                    dimension,
                    completed,
                    state,
                    history,
                    control,
                )
                state["shared"] = dict(state["children"][str(dimension)])
                _atomic_json(_recovery_path(run_path), state)
                log_event(
                    training_log,
                    run_id,
                    "INFO",
                    "Saved rolling stratified recovery snapshot",
                    snapshot_data,
                )
                print(
                    f"Stratified recovery snapshot saved: "
                    f"{snapshot_data['path']}",
                    flush=True,
                )
        result = StratifiedChildResult(
            dimension=dimension,
            batch_size=effective_batch_size,
            completed_epochs=completed_epochs,
            stopped_early=stopped_early,
            model_path="model/shared/final.keras",
            summary_path=summary_path.relative_to(run_path).as_posix(),
            history_path=_history_path(child_dir).relative_to(run_path).as_posix(),
            canonical_counts=_canonical_counts(indices, config, dimension),
        )
        children[dimension] = result
        _write_manifest(
            run_path, config, split_summaries, children, status="training"
        )
        log_event(
            training_log,
            run_id,
            "INFO",
            "Completed resolution stratum training",
            asdict(result),
        )
        # In the default interleaved schedule a parent epoch is a complete
        # pass through every dimension.  ``supra_epochs`` now controls how
        # often that stable, shared state is evaluated/checkpointed (rather
        # than how long one dimension is allowed to train in isolation).
        validation_due = (
            block_stop % int(settings(config).get("supra_epochs", 5)) == 0
            or block_stop == total_epochs
        )
        if (
            dimension == dimensions(config)[-1]
            and validation_due
            and block_stop not in completed_cycles
            and all(
                children.get(value) is not None
                and children[value].completed_epochs >= block_stop
                for value in dimensions(config)
            )
        ):
            cycle_metrics = _cycle_validation(
                model,
                sqlite_path,
                indices,
                config,
                {value: children[value].batch_size for value in dimensions(config)},
            )
            monitor = _cycle_monitor_name(config)
            aggregate = cycle_metrics["aggregate"]
            if interleaved_status is not None:
                # ``train_on_batch`` reports one highly variable mini-batch at
                # a time. Only publish metrics after every validation stratum
                # has evaluated the same shared weights, and make their
                # held-out provenance explicit in the display.
                interleaved_status.on_epoch_end(
                    block_stop - 1,
                    {f"val_{name}": value for name, value in aggregate.items()},
                )
            value = aggregate.get(monitor)
            if value is None:
                log_event(
                    training_log,
                    run_id,
                    "WARNING",
                    "Skipped stratified cycle scheduler: monitor unavailable",
                    {"monitor": monitor, "cycle": cycle_metrics},
                )
            else:
                mode = _monitor_mode(monitor)
                scheduler = settings(config).get("cycle_scheduler", {})
                best = cycle_control.get("best")
                improved = _is_improvement(
                    float(value), float(best) if best is not None else None, mode
                )
                # An aggregate can improve while a small but important
                # resolution silently collapses.  Guardrails compare every
                # stratum to its own best observed validation score.  They
                # intentionally affect checkpoint selection, not training:
                # a later cycle may recover the stratum.
                guardrail_metric = str(scheduler.get("guardrail_metric", monitor)) if isinstance(scheduler, dict) else monitor
                max_drop = scheduler.get("max_stratum_drop") if isinstance(scheduler, dict) else None
                best_per_stratum = dict(cycle_control.get("best_per_stratum", {}))
                guardrail_failures: dict[str, dict[str, float]] = {}
                if max_drop is not None:
                    for stratum, record in cycle_metrics["per_stratum"].items():
                        current = record["metrics"].get(guardrail_metric)
                        if current is None:
                            continue
                        previous = best_per_stratum.get(stratum)
                        if previous is not None:
                            drop = (float(previous) - float(current)) if _monitor_mode(guardrail_metric) == "max" else (float(current) - float(previous))
                            if drop > float(max_drop):
                                guardrail_failures[stratum] = {
                                    "best": float(previous), "current": float(current), "drop": drop,
                                }
                        if _is_improvement(float(current), float(previous) if previous is not None else None, _monitor_mode(guardrail_metric)):
                            best_per_stratum[stratum] = float(current)
                    cycle_control["best_per_stratum"] = best_per_stratum
                checkpoint_eligible = not guardrail_failures
                improved = improved and checkpoint_eligible
                if improved:
                    cycle_control["best"] = float(value)
                    cycle_control["wait"] = 0
                    cycle_control["reduce_lr_wait"] = 0
                    best_path = run_path / "model" / "shared" / "best.weights.h5"
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    model.save_weights(best_path)
                else:
                    cycle_control["wait"] = int(cycle_control.get("wait", 0)) + 1
                    cycle_control["reduce_lr_wait"] = int(
                        cycle_control.get("reduce_lr_wait", 0)
                    ) + 1
                reduce_patience = int(
                    scheduler.get("reduce_lr_patience", 3)
                    if isinstance(scheduler, dict)
                    else 3
                )
                reduced_learning_rate = None
                if (
                    not improved
                    and bool(config.get("callbacks", {}).get("reduce_lr_on_plateau", False))
                    and int(cycle_control.get("reduce_lr_wait", 0)) >= reduce_patience
                ):
                    factor = float(
                        scheduler.get("reduce_lr_factor", 0.5)
                        if isinstance(scheduler, dict)
                        else 0.5
                    )
                    learning_rate = model.optimizer.learning_rate
                    current_rate = float(keras.backend.get_value(learning_rate))
                    reduced_learning_rate = current_rate * factor
                    if hasattr(learning_rate, "assign"):
                        learning_rate.assign(reduced_learning_rate)
                    else:
                        model.optimizer.learning_rate = reduced_learning_rate
                    cycle_control["reduce_lr_wait"] = 0
                    cycle_control["learning_rate"] = reduced_learning_rate
                if (
                    not improved
                    and bool(config.get("callbacks", {}).get("early_stopping", False))
                    and int(cycle_control.get("wait", 0)) >= int(
                        config.get("callbacks", {}).get("early_stopping_patience", 5)
                    )
                ):
                    stop_training = True
                    cycle_control["stopped_early"] = True
                cycle_metrics["monitor"] = monitor
                cycle_metrics["monitor_value"] = float(value)
                cycle_metrics["improved"] = improved
                cycle_metrics["checkpoint_eligible"] = checkpoint_eligible
                cycle_metrics["guardrail_metric"] = guardrail_metric
                cycle_metrics["guardrail_failures"] = guardrail_failures
                cycle_metrics["reduced_learning_rate"] = reduced_learning_rate
                cycle_metrics["stopped_early"] = stop_training
                print(
                    f"Completed stratified cycle through epoch {block_stop}/{total_epochs}: "
                    f"aggregate validation {monitor}={float(value):.6g}",
                    flush=True,
                )
                log_event(
                    training_log,
                    run_id,
                    "INFO",
                    "Completed stratified cycle validation",
                    cycle_metrics,
                )
            completed_cycles.add(block_stop)
            cycle_control["completed_block_stops"] = sorted(completed_cycles)
            state["cycle_scheduler"] = cycle_control
            _atomic_json(_recovery_path(run_path), state)
        # The dynamic-spatial shared model intentionally remains resident while
        # the next stratum is trained.

    shared_dir = run_path / "model" / "shared"
    if interleaved_status is not None:
        interleaved_status.on_train_end()
    shared_dir.mkdir(parents=True, exist_ok=True)
    if cycle_control.get("stopped_early") and (shared_dir / "best.weights.h5").exists():
        model.load_weights(shared_dir / "best.weights.h5")
    write_model_summary(model, shared_dir / "model_summary.txt")
    for dimension in dimensions(config):
        write_model_summary(
            model,
            run_path / "model" / "strata" / str(dimension) / "model_summary.txt",
        )
    model.save(shared_dir / "final.keras")
    if children:
        latest_dimension = dimensions(config)[-1]
        latest = children[latest_dimension]
        _save_recovery_model(
            model,
            run_path,
            latest_dimension,
            latest.completed_epochs,
            state,
            _read_history(run_path / latest.history_path),
            state.get("children", {}).get(str(latest_dimension), {}).get("control", {}),
        )
        state["shared"] = dict(state["children"][str(latest_dimension)])
        _atomic_json(_recovery_path(run_path), state)
        log_event(
            training_log,
            run_id,
            "INFO",
            "Saved final rolling stratified recovery snapshot",
            {
                "dimension": latest_dimension,
                "completed_epoch": latest.completed_epochs,
                "path": "model/recovery/latest.keras",
            },
        )
    manifest = _write_manifest(
        run_path, config, split_summaries, children, status="trained"
    )
    return StratifiedTrainingResult(
        children=children,
        manifest_path=manifest,
        recovery_path=_recovery_path(run_path),
        split_summaries=split_summaries,
    )
