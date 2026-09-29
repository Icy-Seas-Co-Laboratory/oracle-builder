"""Local, durable control boundary for a supervised training child process.

The worker supervisor owns network traffic.  Training sees only this bounded
file protocol, so callbacks never block on an orchestrator request.  Commands
are advisory until a supported safe boundary (currently an epoch/cycle) is
reached.  A parent may terminate the process for ``stop_now`` separately.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tensorflow import keras

Command = Literal["pause", "yield", "stop", "restart"]
SAFE_BOUNDARY_COMMANDS: frozenset[str] = frozenset({"pause", "yield", "stop", "restart"})


@dataclass(frozen=True, slots=True)
class ControlDecision:
    command: Command | None = None
    sequence: int = 0
    reason: str | None = None


class FileSegmentControl:
    """Read monotonic, idempotent directives from a supervisor-owned JSON file."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._last_sequence = 0
        self._decision = ControlDecision()

    @property
    def decision(self) -> ControlDecision:
        return self._decision

    def poll(self) -> ControlDecision:
        if self.path is None or not self.path.exists():
            return self._decision
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A concurrent atomic rename can only expose a complete file; a
            # malformed manual edit must not make training unsafe.
            return self._decision
        if not isinstance(value, dict):
            return self._decision
        try:
            sequence = int(value.get("sequence", 0))
        except (TypeError, ValueError):
            return self._decision
        command = str(value.get("command", "")).lower()
        if sequence <= self._last_sequence or command not in SAFE_BOUNDARY_COMMANDS:
            return self._decision
        self._last_sequence = sequence
        self._decision = ControlDecision(command=command, sequence=sequence, reason=value.get("reason"))
        return self._decision


class SegmentBoundaryCallback(keras.callbacks.Callback):
    """Stop Keras after an epoch once a durable snapshot callback has run."""

    def __init__(self, control: FileSegmentControl, *, stop_epoch: int | None = None):
        super().__init__()
        self.control = control
        self.stop_epoch = stop_epoch
        self.completed_epoch = 0
        self.boundary_command: ControlDecision | None = None

    def on_epoch_end(self, epoch: int, logs=None):
        self.completed_epoch = int(epoch) + 1
        decision = self.control.poll()
        reached_stop = self.stop_epoch is not None and self.completed_epoch >= self.stop_epoch
        if decision.command is not None or reached_stop:
            self.boundary_command = decision if decision.command else None
            self.model.stop_training = True


def write_segment_result(path: str | Path, value: dict) -> None:
    """Atomically publish the compact result consumed by the worker parent."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent,
        prefix=f".{target.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, target)
