"""Generic rolling recovery snapshots for supervised Keras training."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from tensorflow import keras

from oracle_builder.artifacts.layout import RunLayout
from oracle_builder.training.logging_callbacks import log_event


# V2 keeps immutable checkpoint generations.  ``latest.keras`` remains a
# compatibility projection for existing local resume commands; authority is
# the atomically replaced recovery state and its generation checksum.
RECOVERY_SCHEMA = {"name": "oracle_builder_training_recovery", "version": "2.0.0"}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recovery_config_hash(config: dict[str, Any]) -> str:
    """Hash the continuation-relevant, portable training contract."""
    value = {
        "run": {key: value for key, value in config.get("run", {}).items() if key != "run_name"},
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
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_recovery_state(run_dir: str | Path) -> dict[str, Any]:
    layout = RunLayout(run_dir)
    if not layout.recovery_state.exists() or not layout.recovery_model.exists():
        raise FileNotFoundError(
            f"No recovery snapshot found in {layout.recovery}; rerun with recovery.enabled = true"
        )
    state = json.loads(layout.recovery_state.read_text(encoding="utf-8"))
    if state.get("schema") != RECOVERY_SCHEMA:
        raise ValueError(f"Unsupported recovery state schema: {state.get('schema')}")
    model_path = str(state.get("model_path", ""))
    if not model_path.startswith("model/recovery/") or Path(model_path).is_absolute() or ".." in Path(model_path).parts:
        raise ValueError("Recovery state points to an unexpected model path")
    model = Path(run_dir) / model_path
    observed = _sha256(model) if model.exists() else None
    if observed != state.get("model_sha256"):
        raise ValueError("Recovery model checksum mismatch")
    generation = state.get("generation")
    if not isinstance(generation, dict) or not generation.get("id"):
        raise ValueError("Recovery state is missing immutable generation metadata")
    return state


def validate_recovery_state(
    run_dir: str | Path,
    config: dict[str, Any],
    *,
    artifact_id: str,
    run_id: str,
) -> dict[str, Any]:
    state = read_recovery_state(run_dir)
    if state.get("artifact_id") != artifact_id or state.get("run_id") != run_id:
        raise ValueError("Recovery state does not belong to this run artifact")
    if state.get("phase") != "supervised":
        raise ValueError("Only supervised recovery snapshots are currently resumable")
    if state.get("config_sha256") != recovery_config_hash(config):
        raise ValueError("Recovery state does not match the resolved training contract")
    completed_epoch = int(state.get("completed_epoch", 0))
    if completed_epoch < 0:
        raise ValueError("Recovery state has an invalid completed epoch")
    return state


def read_recovery_callback_state(
    run_dir: str | Path, state: dict[str, Any]
) -> dict[str, Any]:
    """Load callback continuation data from the selected checkpoint generation.

    Older recovery snapshots did not include callback state.  They remain
    usable for ordinary local recovery, but segmented execution creates this
    payload before it advertises a resumable boundary.
    """
    descriptor = state.get("callback_state")
    if not isinstance(descriptor, dict):
        return {}
    relative = str(descriptor.get("path", ""))
    path = Path(run_dir) / relative
    if not relative.startswith("model/recovery/") or not path.exists():
        raise ValueError("Recovery callback state is missing")
    if _sha256(path) != descriptor.get("sha256"):
        raise ValueError("Recovery callback state checksum mismatch")
    values = descriptor.get("values", {})
    if not isinstance(values, dict):
        raise ValueError("Recovery callback state is malformed")
    result = json.loads(json.dumps(values))
    arrays = descriptor.get("arrays", {})
    if arrays:
        with np.load(path, allow_pickle=False) as archive:
            for name, keys in arrays.items():
                if not isinstance(keys, list):
                    raise ValueError("Recovery callback state has invalid array keys")
                result.setdefault(name, {})["best_weights"] = [
                    np.array(archive[str(key)]) for key in keys
                ]
    return result


def save_step_recovery_snapshot(
    model: keras.Model,
    run_dir: str | Path,
    config: dict[str, Any],
    training_log: str | Path,
    run_id: str,
    artifact_id: str,
    *,
    completed_epoch: int,
    cursor: dict[str, int],
    partial_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Commit an ordered-batch cursor using the same generation format.

    This deliberately supports only the deterministic batch cursor controller
    in ``train.py``.  It does not pretend that arbitrary tf.data shuffles or
    augmentation pipelines have a portable sampler position.
    """
    callback = RollingRecoveryCallback(
        run_dir, config, training_log, run_id, artifact_id, force_every_epoch=True
    )
    callback.set_model(model)
    # Keras epoch numbering is zero-based. -1 records a valid pre-epoch
    # cursor with completed_epoch=0.
    callback.on_epoch_end(int(completed_epoch) - 1)
    state = read_recovery_state(run_dir)
    state["cursor"] = {
        "kind": "ordered_batches_v1",
        "epoch": int(cursor["epoch"]),
        "batch_offset": int(cursor["batch_offset"]),
        "global_step": int(cursor["global_step"]),
    }
    state["continuation"] = {
        "boundary": "ordered_batch",
        "model_optimizer_state": "preserved",
        "data_order": "pinned_ordered_batches_v1",
        "rng_state": "not_used_by_supported_path",
        "reproducibility": "numerical_tolerance_only",
    }
    state["partial_metrics"] = partial_metrics
    _write_json(RunLayout(run_dir).recovery_state, state)
    generation = Path(run_dir) / str(state["generation"]["path"]) / "state.json"
    _write_json(generation, state)
    return state


def clear_recovery_snapshot(run_dir: str | Path) -> None:
    layout = RunLayout(run_dir)
    layout.recovery_model.unlink(missing_ok=True)
    layout.recovery_state.unlink(missing_ok=True)
    generations = layout.recovery / "generations"
    if generations.exists():
        import shutil
        shutil.rmtree(generations)


class RollingRecoveryCallback(keras.callbacks.Callback):
    """Persist one atomic full-model snapshot after selected supervised epochs."""

    def __init__(
        self,
        run_dir: str | Path,
        config: dict[str, Any],
        training_log: str | Path,
        run_id: str,
        artifact_id: str,
        *,
        force_every_epoch: bool = False,
        callback_state_provider=None,
    ):
        super().__init__()
        self.layout = RunLayout(run_dir)
        self.config = config
        self.training_log = training_log
        self.run_id = run_id
        self.artifact_id = artifact_id
        self.every = 1 if force_every_epoch else int(config.get("recovery", {}).get("save_every_epochs", 1))
        self.callback_state_provider = callback_state_provider
        self._last_completed_epoch = 0
        self._terminal_snapshot_written = False

    def on_epoch_end(self, epoch: int, logs=None):
        completed_epoch = int(epoch) + 1
        self._last_completed_epoch = completed_epoch
        if completed_epoch % self.every:
            return
        self.layout.recovery.mkdir(parents=True, exist_ok=True)
        generation_id = uuid.uuid4().hex
        generations = self.layout.recovery / "generations"
        staging = generations / f".{generation_id}.staging"
        generation_dir = generations / generation_id
        staging.mkdir(parents=True, exist_ok=False)
        model = staging / "model.keras"
        self.model.save(model)
        checksum = _sha256(model)
        callback_descriptor = None
        if self.callback_state_provider is not None:
            raw_callback_state = self.callback_state_provider() or {}
            values: dict[str, Any] = {}
            arrays: dict[str, list[str]] = {}
            archive_values: dict[str, np.ndarray] = {}
            for name, value in raw_callback_state.items():
                if not isinstance(value, dict):
                    continue
                scalar = dict(value)
                weights = scalar.pop("best_weights", None)
                values[str(name)] = scalar
                if weights is not None:
                    keys = []
                    for index, weight in enumerate(weights):
                        key = f"{name}_{index}"
                        archive_values[key] = np.asarray(weight)
                        keys.append(key)
                    arrays[str(name)] = keys
            if values or archive_values:
                callback_path = staging / "callback-state.npz"
                np.savez_compressed(callback_path, **archive_values)
                callback_descriptor = {
                    "path": f"model/recovery/generations/{generation_id}/callback-state.npz",
                    "sha256": _sha256(callback_path),
                    "values": values,
                    "arrays": arrays,
                }
        # Complete generation first.  A crash before the current-state rename
        # leaves an unreferenced, harmless generation rather than a torn one.
        os.replace(staging, generation_dir)
        generation_model = generation_dir / "model.keras"
        latest_temporary = self.layout.recovery / f".latest.{generation_id}.keras"
        try:
            os.link(generation_model, latest_temporary)
        except OSError:
            # Filesystems without hard links keep a compatibility copy only;
            # the generation remains the authoritative continuation input.
            import shutil
            shutil.copy2(generation_model, latest_temporary)
        os.replace(latest_temporary, self.layout.recovery_model)
        state = {
            "schema": RECOVERY_SCHEMA,
            "artifact_id": self.artifact_id,
            "run_id": self.run_id,
            "phase": "supervised",
            "completed_epoch": completed_epoch,
            "model_path": f"model/recovery/generations/{generation_id}/model.keras",
            "model_sha256": checksum,
            "config_sha256": recovery_config_hash(self.config),
            # Keras serialization preserves model/optimizer state. It does
            # not serialize arbitrary tf.data shuffle buffers or TensorFlow's
            # process RNG stream, so ordinary epoch recovery must never claim
            # bitwise continuation for randomized input pipelines.
            "continuation": {
                "boundary": "epoch",
                "model_optimizer_state": "preserved",
                "callback_state": "preserved_when_configured",
                "data_order": "reconstructed_from_seed_not_sampler_state",
                "rng_state": "not_portable",
                "reproducibility": "numerical_tolerance_only",
            },
            "generation": {
                "id": generation_id,
                "path": f"model/recovery/generations/{generation_id}",
                "schema": "oracle_builder_training_checkpoint_generation/v1",
            },
        }
        if callback_descriptor is not None:
            state["callback_state"] = callback_descriptor
        _write_json(generation_dir / "state.json", state)
        _write_json(self.layout.recovery_state, state)
        log_event(
            self.training_log,
            self.run_id,
            "INFO",
            "Saved rolling recovery snapshot",
            {"completed_epoch": completed_epoch, "path": state["model_path"]},
        )
        print(f"Recovery snapshot saved after epoch {completed_epoch}", flush=True)

    def on_train_end(self, logs=None):
        """Capture EarlyStopping's restored best weights when it terminates.

        Keras invokes callback ``on_train_end`` in list order.  This recovery
        callback is installed after EarlyStopping, so the model has already
        been restored to the selected best weights before this final snapshot.
        """
        if self._terminal_snapshot_written or not self.callback_state_provider:
            return
        state = self.callback_state_provider() or {}
        early = state.get("early_stopping", {}) if isinstance(state, dict) else {}
        if not isinstance(early, dict) or int(early.get("stopped_epoch", 0)) <= 0:
            return
        if self._last_completed_epoch <= 0:
            return
        self._terminal_snapshot_written = True
        every = self.every
        try:
            self.every = 1
            self.on_epoch_end(self._last_completed_epoch - 1, logs)
        finally:
            self.every = every
