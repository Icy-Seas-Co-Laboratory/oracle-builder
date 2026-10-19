"""Typed public boundary for the Oracle training workflow."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    """All explicit inputs for one training or recovery invocation.

    CLI callers and stateless workers construct this value directly.  The
    request deliberately contains no command, executable, or untrusted final
    output path selected by a leased work unit.
    """

    config: str | None = None
    input: str | None = None
    output: str | None = None
    display_name: str | None = None
    runs_dir: str = "./runs"
    overwrite: bool = False
    resume: str | None = None
    dry_run: bool = False
    preflight: bool = False
    debug: bool = False
    # A segment is a bounded, durable advancement of a supervised run.  These
    # fields are deliberately transport-neutral: the pull-worker parent can
    # supply them from a sealed work unit without importing orchestration code.
    segment_start_epoch: int | None = None
    segment_stop_epoch: int | None = None
    finalize_only: bool = False
    control_file: str | None = None
    segment_result_path: str | None = None
    split_manifest: str | None = None
    segment_max_steps: int | None = None
    segment_start_step: int | None = None
    # Run self-supervised pretraining as one sealed initialization phase.  The
    # resulting recovery snapshot is then the input to resumable supervised
    # segments; SSL optimizer/teacher state is deliberately not resumed.
    pretrain_only: bool = False

    def __post_init__(self) -> None:
        if not self.runs_dir:
            raise ValueError("runs_dir is required")
        if self.resume and (self.config or self.output or self.overwrite or self.split_manifest):
            raise ValueError("resume uses the existing artifact; do not supply config, output, or overwrite")
        if self.display_name is not None and (not isinstance(self.display_name, str) or not self.display_name.strip() or len(self.display_name.strip()) > 240
                                              or any(ord(char) < 32 or ord(char) == 127 for char in self.display_name.strip())):
            raise ValueError("display_name must be 1–240 characters")
        if self.finalize_only and (self.segment_start_epoch is not None or self.segment_stop_epoch is not None):
            raise ValueError("finalize_only cannot include a training epoch range")
        if self.segment_start_epoch is not None and self.segment_start_epoch < 0:
            raise ValueError("segment_start_epoch must be non-negative")
        if self.segment_stop_epoch is not None and self.segment_stop_epoch <= 0:
            raise ValueError("segment_stop_epoch must be positive")
        if self.segment_max_steps is not None and self.segment_max_steps <= 0:
            raise ValueError("segment_max_steps must be positive")
        if self.segment_start_step is not None and self.segment_start_step < 0:
            raise ValueError("segment_start_step must be non-negative")
        if self.segment_start_step is not None and self.segment_max_steps is None:
            raise ValueError("segment_start_step requires segment_max_steps")
        if self.pretrain_only and self.finalize_only:
            raise ValueError("pretrain_only cannot be combined with finalize_only")
        if (
            self.segment_start_epoch is not None
            and self.segment_stop_epoch is not None
            and self.segment_stop_epoch <= self.segment_start_epoch
        ):
            raise ValueError("segment_stop_epoch must be after segment_start_epoch")
        for value, name in (
            (self.control_file, "control_file"),
            (self.segment_result_path, "segment_result_path"),
            (self.split_manifest, "split_manifest"),
        ):
            if value is not None and not str(value).strip():
                raise ValueError(f"{name} cannot be blank")
