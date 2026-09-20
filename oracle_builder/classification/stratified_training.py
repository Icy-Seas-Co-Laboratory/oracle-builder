"""Training controller for resolution-stratified classification models.

The controller deliberately owns the epoch loop.  A training item may move to
the adjacent smaller stratum between epochs, so a single, static ``tf.data``
pipeline cannot describe the complete run.  Validation and all post-training
consumers use canonical (non-random) routing.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tensorflow import keras

from oracle_builder.classification.stratification import (
    batch_plan,
    dimensions,
    enabled,
    stratum_for_shape,
    summarize_records,
    training_stratum,
    validate,
)
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
        return keras.models.load_model(self.model_path(dimension))


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
        canonical = stratum_for_shape(shape, configured)
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
        datasets[split] = source.training_dataset(
            selected, shuffle=False, augment=False
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
    """Yield parent-epoch-major child turns in deterministic resolution order."""
    for epoch in range(int(total_epochs)):
        for dimension in dimensions(config):
            yield epoch, dimension


def _history_path(child_dir: Path) -> Path:
    return child_dir / "training_history.json"


def _read_history(child_dir: Path) -> dict[str, list[float]]:
    path = _history_path(child_dir)
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


def _recovery_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "model" / "recovery" / "stratified_state.json"


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
        "children": {},
    }


def _save_recovery_model(
    model: keras.Model,
    run_dir: Path,
    dimension: int,
    completed_epochs: int,
    state: dict[str, Any],
    history: dict[str, list[float]],
    control: dict[str, Any],
) -> None:
    recovery_dir = run_dir / "model" / "strata" / str(dimension) / "recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    target = recovery_dir / "latest.keras"
    temporary = recovery_dir / f".latest.{uuid.uuid4().hex}.keras"
    model.save(temporary)
    os.replace(temporary, target)
    relative = target.relative_to(run_dir).as_posix()
    state["children"][str(dimension)] = {
        "completed_epochs": int(completed_epochs),
        "model_path": relative,
        "model_sha256": _sha256(target),
        "history": history,
        "control": control,
    }
    _atomic_json(_recovery_path(run_dir), state)


def _is_improvement(value: float, best: float | None, mode: str) -> bool:
    return best is None or (value > best if mode == "max" else value < best)


def _monitor_mode(name: str) -> str:
    return "max" if any(token in name.lower() for token in ("acc", "f1", "auc")) else "min"


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
        "training_routing": config.get("classification", {})
        .get("stratification", {})
        .get("training_routing", {}),
        "batch_plan": {str(k): v for k, v in batch_plan(config).items()},
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
            "split_summaries": split_summaries,
            "distribution": asdict(distribution_info),
        },
    )
    print(
        "Stratified training enabled\n"
        f"  dimensions: {dimensions(config)}\n"
        f"  batch plan: {batch_plan(config)}",
        flush=True,
    )

    state = resume_state or _initial_recovery(config, run_id)
    children: dict[int, StratifiedChildResult] = {}

    # Every child completes parent epoch N before any child starts N + 1.
    for scheduled_epoch, dimension in epoch_stratum_schedule(config, total_epochs):
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
        with strategy.scope():
            if recovered.get("model_path"):
                model = keras.models.load_model(run_path / recovered["model_path"])
            else:
                model = build_and_compile_model(child)
        summary_path = child_dir / "model_summary.txt"
        write_model_summary(model, summary_path)
        canonical_validation = routed_index(
            indices["validation"], config, dimension
        ) if "validation" in indices else None
        validation_source = SQLiteClassificationSource(sqlite_path, child)
        validation_data = (
            validation_source.training_dataset(
                canonical_validation, shuffle=False, augment=False
            )
            if canonical_validation is not None and canonical_validation.refs
            else None
        )
        monitor = str(
            config.get("callbacks", {}).get(
                "checkpoint_monitor", "val_loss" if validation_data is not None else "loss"
            )
        )
        mode = _monitor_mode(monitor)
        stopped_early = bool(control.get("stopped_early", False))
        completed_epochs = initial_epoch

        for epoch in range(
            max(initial_epoch, scheduled_epoch),
            min(total_epochs, scheduled_epoch + 1),
        ):
            if stopped_early:
                break
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
            if selected.refs:
                source = SQLiteClassificationSource(sqlite_path, child)
                train_data = source.training_dataset(
                    selected, shuffle=True, augment=True
                )
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
                epoch_history = model.fit(
                    train_data,
                    validation_data=validation_data,
                    initial_epoch=epoch,
                    epochs=epoch + 1,
                    callbacks=[rich_status],
                    verbose=0,
                )
                _append_history(history, epoch_history.history)
                logs = {
                    name: float(values[-1])
                    for name, values in epoch_history.history.items()
                    if values
                }
            else:
                logs = {}
                log_event(
                    training_log,
                    run_id,
                    "WARNING",
                    "No samples routed to resolution stratum for epoch",
                    {"dimension": dimension, "epoch": epoch + 1},
                )

            value = logs.get(monitor)
            improved = False
            if value is not None:
                best = control.get("best")
                if _is_improvement(float(value), float(best) if best is not None else None, mode):
                    improved = True
                    control["best"] = float(value)
                    control["wait"] = 0
                    control["reduce_lr_wait"] = 0
                    model.save_weights(child_dir / "best.weights.h5")
                else:
                    control["wait"] = int(control.get("wait", 0)) + 1
                    control["reduce_lr_wait"] = int(
                        control.get("reduce_lr_wait", 0)
                    ) + 1
            if (
                value is not None
                and not improved
                and bool(
                    config.get("callbacks", {}).get("reduce_lr_on_plateau", False)
                )
                and int(control.get("reduce_lr_wait", 0)) >= 3
            ):
                learning_rate = model.optimizer.learning_rate
                current_rate = float(keras.backend.get_value(learning_rate))
                if hasattr(learning_rate, "assign"):
                    learning_rate.assign(current_rate * 0.5)
                else:
                    model.optimizer.learning_rate = current_rate * 0.5
                control["reduce_lr_wait"] = 0
                control["learning_rate"] = current_rate * 0.5
            if (
                value is not None
                and not improved
                and bool(config.get("callbacks", {}).get("early_stopping", False))
            ):
                patience = int(
                    config.get("callbacks", {}).get("early_stopping_patience", 5)
                )
                if int(control.get("wait", 0)) >= patience:
                    stopped_early = True
                    control["stopped_early"] = True

            completed = epoch + 1
            completed_epochs = completed
            _atomic_json(_history_path(child_dir), history)
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
            # Epoch-major scheduling needs a reloadable optimizer snapshot for
            # every child turn, even if optional long-term recovery is off.
            if recovery_enabled and (completed % save_every == 0 or stopped_early):
                _save_recovery_model(
                    model,
                    run_path,
                    dimension,
                    completed,
                    state,
                    history,
                    control,
                )
        if (completed_epochs >= total_epochs or stopped_early) and (child_dir / "best.weights.h5").exists() and bool(
            config.get("callbacks", {}).get("early_stopping", False)
        ):
            model.load_weights(child_dir / "best.weights.h5")
        final_path = child_dir / "final.keras"
        model.save(final_path)
        _save_recovery_model(
            model,
            run_path,
            dimension,
            completed_epochs,
            state,
            history,
            control,
        )
        result = StratifiedChildResult(
            dimension=dimension,
            batch_size=int(child["data"]["batch_size"]),
            completed_epochs=completed_epochs,
            stopped_early=stopped_early,
            model_path=final_path.relative_to(run_path).as_posix(),
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
        # Do not retain every model on the GPU.  Post-training code reloads one
        # child at a time from the paths returned above.
        del model
        keras.backend.clear_session()

    manifest = _write_manifest(
        run_path, config, split_summaries, children, status="trained"
    )
    return StratifiedTrainingResult(
        children=children,
        manifest_path=manifest,
        recovery_path=_recovery_path(run_path),
        split_summaries=split_summaries,
    )
