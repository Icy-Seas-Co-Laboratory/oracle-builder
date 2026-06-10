#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

from oracle_builder.config import copy_run_config, resolve_config, write_json
from oracle_builder.environment import write_environment
from oracle_builder.paths import create_run_dir


def plot_history(history, run_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    history_dict = history.history
    for metric, filename in (("loss", "loss_curve.png"), ("accuracy", "accuracy_curve.png")):
        if metric not in history_dict and f"val_{metric}" not in history_dict:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        if metric in history_dict:
            ax.plot(history_dict[metric], label=metric)
        if f"val_{metric}" in history_dict:
            ax.plot(history_dict[f"val_{metric}"], label=f"val_{metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=150)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an oracle-builder model.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--runs-dir", default="./runs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="TODO: resume support is not implemented yet.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume:
        print("--resume is reserved for a future implementation.", file=sys.stderr)
        return 2
    run_dir = create_run_dir(args.runs_dir, args.output, overwrite=args.overwrite, dry_run=args.dry_run)
    config = resolve_config(args.config, args.input, run_dir)
    config["debug"] = bool(args.debug)
    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "resolved_config": config}, indent=2))
        return 0
    if args.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
        run_dir = create_run_dir(args.runs_dir, args.output, overwrite=True)

    copy_run_config(args.config, run_dir)
    write_json(run_dir / "resolved_config.json", config)
    run_id = str(uuid.uuid4())
    run_metadata = {"run_id": run_id, "run_name": args.output, "status": "running"}
    write_json(run_dir / "run_metadata.json", run_metadata)
    environment = write_environment(run_dir)
    training_log = run_dir / "training_log.sqlite"
    from oracle_builder.training.logging_callbacks import init_training_log, log_event, mark_run_complete

    init_training_log(training_log, run_id, args.output, config, environment)

    try:
        from oracle_builder.data.sqlite_dataset import load_arrays, make_tf_datasets
        from oracle_builder.evaluation.predictions import write_predictions_db
        from oracle_builder.evaluation.reports import evaluate_run_model
        from oracle_builder.saving.load_test import run_load_tests
        from oracle_builder.saving.save_model import save_model_artifacts, write_load_test_report
        from oracle_builder.training.train import train_model

        datasets, records_by_split = make_tf_datasets(args.input, config)
        log_event(training_log, run_id, "INFO", "Datasets loaded", {"splits": list(datasets)})
        model, history = train_model(config, datasets, run_dir, training_log, run_id)
        plot_history(history, run_dir)
        save_report = save_model_artifacts(model, run_dir, config)
        load_report = run_load_tests(run_dir, config, save_report)
        write_load_test_report(run_dir, load_report)
        evaluation = evaluate_run_model(model, config, args.input, run_dir, split="test")
        if config.get("output", {}).get("save_predictions", True):
            split = "test" if "test" in records_by_split else "validation"
            x, y, records = load_arrays(args.input, config, split=split)
            write_predictions_db(model, x, y, records, config, run_dir / "predictions" / "predictions.sqlite")
        run_metadata["status"] = "complete"
        run_metadata["evaluation_summary"] = evaluation.get("summary")
        write_json(run_dir / "run_metadata.json", run_metadata)
        mark_run_complete(training_log, run_id, "complete")
        return 0
    except Exception as exc:
        log_event(training_log, run_id, "ERROR", "Training run failed", {"error": str(exc)})
        mark_run_complete(training_log, run_id, "failed")
        run_metadata["status"] = "failed"
        run_metadata["error"] = str(exc)
        write_json(run_dir / "run_metadata.json", run_metadata)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
