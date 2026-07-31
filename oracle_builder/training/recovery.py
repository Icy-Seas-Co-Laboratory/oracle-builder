"""Generic rolling recovery snapshots for supervised Keras training."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from tensorflow import keras

from oracle_builder.artifacts.layout import RunLayout
from oracle_builder.training.logging_callbacks import log_event


RECOVERY_SCHEMA = {"name": "oracle_builder_training_recovery", "version": "1.0.0"}


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
    if state.get("model_path") != "model/recovery/latest.keras":
        raise ValueError("Recovery state points to an unexpected model path")
    observed = _sha256(layout.recovery_model)
    if observed != state.get("model_sha256"):
        raise ValueError("Recovery model checksum mismatch")
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


def clear_recovery_snapshot(run_dir: str | Path) -> None:
    layout = RunLayout(run_dir)
    layout.recovery_model.unlink(missing_ok=True)
    layout.recovery_state.unlink(missing_ok=True)


class RollingRecoveryCallback(keras.callbacks.Callback):
    """Persist one atomic full-model snapshot after selected supervised epochs."""

    def __init__(
        self,
        run_dir: str | Path,
        config: dict[str, Any],
        training_log: str | Path,
        run_id: str,
        artifact_id: str,
    ):
        super().__init__()
        self.layout = RunLayout(run_dir)
        self.config = config
        self.training_log = training_log
        self.run_id = run_id
        self.artifact_id = artifact_id
        self.every = int(config.get("recovery", {}).get("save_every_epochs", 1))

    def on_epoch_end(self, epoch: int, logs=None):
        completed_epoch = int(epoch) + 1
        if completed_epoch % self.every:
            return
        self.layout.recovery.mkdir(parents=True, exist_ok=True)
        temporary = self.layout.recovery / f".latest.{uuid.uuid4().hex}.keras"
        self.model.save(temporary)
        os.replace(temporary, self.layout.recovery_model)
        state = {
            "schema": RECOVERY_SCHEMA,
            "artifact_id": self.artifact_id,
            "run_id": self.run_id,
            "phase": "supervised",
            "completed_epoch": completed_epoch,
            "model_path": "model/recovery/latest.keras",
            "model_sha256": _sha256(self.layout.recovery_model),
            "config_sha256": recovery_config_hash(self.config),
        }
        _write_json(self.layout.recovery_state, state)
        log_event(
            self.training_log,
            self.run_id,
            "INFO",
            "Saved rolling recovery snapshot",
            {"completed_epoch": completed_epoch, "path": state["model_path"]},
        )
        print(f"Recovery snapshot saved after epoch {completed_epoch}", flush=True)
