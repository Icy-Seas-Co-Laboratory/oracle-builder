from __future__ import annotations

from pathlib import Path

from oracle_builder.artifacts.layout import RunLayout


def create_run_dir(runs_dir: str | Path, run_name: str, overwrite: bool = False, dry_run: bool = False) -> Path:
    run_dir = Path(runs_dir) / run_name
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"Run directory already exists: {run_dir}. Use --overwrite to replace it.")
    if dry_run:
        return run_dir
    layout = RunLayout(run_dir)
    layout.create_directories()
    return layout.root


def model_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "model"
