#!/usr/bin/env python3
"""Command-line adapter for the typed Oracle training workflow."""
from __future__ import annotations

import argparse
import atexit
import os
import sys
import threading

# This must be set before workflow imports can lazily load TensorFlow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get(
    "ORACLE_BUILDER_TF_CPP_MIN_LOG_LEVEL", "2"
)

_GPU_TIMER_WARNING = b"gpu_timer.cc"
_GPU_TIMER_WARNING_TEXT = b"Skipping the delay kernel, measurement accuracy will be reduced"


def _install_gpu_timer_warning_filter() -> None:
    """Suppress repeated, non-actionable XLA timer warnings for interactive CLI use."""
    if os.environ.get("ORACLE_BUILDER_FILTER_GPU_TIMER_WARNINGS", "1") in {"0", "false", "False"}:
        return
    if not sys.stderr.isatty():
        return
    try:
        original_stderr = os.dup(2)
        reader, writer = os.pipe()
        os.dup2(writer, 2)
        os.close(writer)
    except OSError:
        return
    suppressed = 0

    def forward_stderr() -> None:
        nonlocal suppressed
        with os.fdopen(reader, "rb", closefd=True) as stream:
            for line in iter(stream.readline, b""):
                if _GPU_TIMER_WARNING in line and _GPU_TIMER_WARNING_TEXT in line:
                    suppressed += 1
                    if suppressed == 1:
                        os.write(original_stderr, b"TensorFlow/XLA GPU timer warnings suppressed; set ORACLE_BUILDER_FILTER_GPU_TIMER_WARNINGS=0 to show them.\n")
                    continue
                os.write(original_stderr, line)

    thread = threading.Thread(target=forward_stderr, name="gpu-timer-stderr-filter", daemon=True)
    thread.start()

    def restore_stderr() -> None:
        try:
            sys.stderr.flush()
            os.dup2(original_stderr, 2)
            thread.join(timeout=1)
            os.close(original_stderr)
        except OSError:
            pass

    atexit.register(restore_stderr)


_install_gpu_timer_warning_filter()

from oracle_builder.training.api import TrainingRequest
from oracle_builder.training.workflow import run_training

# Kept as an importable alias for integration callers during the extraction.
train = run_training


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an oracle-builder model.")
    parser.add_argument("-c", "--config")
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    parser.add_argument("--runs-dir", default="./runs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", metavar="RUN_DIRECTORY", help="Resume a run from its rolling recovery snapshot.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Validate segmentation SQLite dataset compatibility and exit.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return train(TrainingRequest(
        config=args.config, input=args.input, output=args.output, runs_dir=args.runs_dir,
        overwrite=args.overwrite, resume=args.resume, dry_run=args.dry_run,
        preflight=args.preflight, debug=args.debug,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
