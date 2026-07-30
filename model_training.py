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
    for metric, filename in (
        ("loss", "loss_curve.png"),
        ("accuracy", "accuracy_curve.png"),
        ("dice", "dice_curve.png"),
    ):
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
    parser.add_argument("--preflight", action="store_true", help="Validate segmentation SQLite dataset compatibility and exit.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume:
        print("--resume is reserved for a future implementation.", file=sys.stderr)
        return 2
    run_dir = Path(args.runs_dir) / args.output if args.preflight else create_run_dir(
        args.runs_dir,
        args.output,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    config = resolve_config(args.config, args.input, run_dir)
    config["debug"] = bool(args.debug)
    if args.preflight:
        if config["run"]["task"] != "segmentation":
            print("--preflight currently validates segmentation datasets only.")
            return 0
        from oracle_builder.masking.unet_dataset import validate_unet_dataset

        report = validate_unet_dataset(
            args.input,
            target_input_shape=config["data"].get("input_shape"),
            target_output_shape=config["data"].get("output_shape"),
            require_candidate_mask=config["training"].get("segmentation_target") == "candidate_delta",
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0 if report["valid"] else 2
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
        from oracle_builder.data.sqlite_dataset import load_arrays, load_prediction_arrays, make_tf_datasets
        from oracle_builder.evaluation.predictions import write_predictions_db
        from oracle_builder.evaluation.reports import evaluate_run_model
        from oracle_builder.evaluation.thresholds import analyze_validation_threshold
        from oracle_builder.saving.load_test import run_load_tests
        from oracle_builder.saving.save_model import save_model_artifacts, write_load_test_report
        from oracle_builder.training.train import train_model

        datasets, records_by_split = make_tf_datasets(args.input, config)
        log_event(training_log, run_id, "INFO", "Datasets loaded", {"splits": list(datasets)})
        pretraining_dataset = None
        if config.get("pretraining", {}).get("enabled", False):
            from oracle_builder.training.student_teacher import make_pretraining_dataset

            pretraining_x, _, pretraining_records = load_prediction_arrays(
                args.input,
                config,
                split="train",
            )
            pretraining_dataset = make_pretraining_dataset(pretraining_x, config)
            log_event(
                training_log,
                run_id,
                "INFO",
                "Loaded self-supervised pretraining inputs",
                {"samples": len(pretraining_records), "split": "train"},
            )
        model, history = train_model(
            config,
            datasets,
            run_dir,
            training_log,
            run_id,
            pretraining_dataset=pretraining_dataset,
        )
        plot_history(history, run_dir)
        evidence_index = None
        if (
            config["run"]["task"] == "classification"
            and config.get("evidence", {}).get("enabled", True)
        ):
            from oracle_builder.classification.evidence import build_evidence_index

            evidence_x, evidence_y, evidence_records = load_arrays(
                args.input,
                config,
                split="train",
            )
            evidence_index = build_evidence_index(
                model,
                evidence_x,
                evidence_y,
                evidence_records,
                run_dir / "model" / "classification_evidence.npz",
            )
            log_event(
                training_log,
                run_id,
                "INFO",
                "Built classification prototype and KNN evidence index",
                {
                    "reference_count": len(evidence_records),
                    "classes": [int(value) for value in evidence_index.prototype_labels],
                },
            )
        threshold_analysis = None
        if config["run"]["task"] == "segmentation" and "validation" in datasets:
            validation_x, validation_y, validation_records = load_arrays(
                args.input, config, split="validation"
            )
            threshold_analysis = analyze_validation_threshold(
                model, validation_x, validation_y, run_dir, config=config, records=validation_records
            )
            config.setdefault("evaluation", {})["segmentation_threshold"] = threshold_analysis[
                "best_threshold"
            ]
            write_json(run_dir / "resolved_config.json", config)
            log_event(
                training_log,
                run_id,
                "INFO",
                "Optimized segmentation probability threshold on validation data",
                threshold_analysis,
            )
        save_report = save_model_artifacts(model, run_dir, config)
        load_report = run_load_tests(run_dir, config, save_report)
        write_load_test_report(run_dir, load_report)
        evaluation = evaluate_run_model(model, config, args.input, run_dir, split="test")
        if config.get("output", {}).get("save_predictions", True):
            predictions_path = run_dir / "predictions" / "predictions.sqlite"
            x, targets, records = load_prediction_arrays(args.input, config)
            write_predictions_db(
                model,
                x,
                targets,
                records,
                config,
                predictions_path,
                source_sqlite=args.input,
                prediction_set=args.output,
                evidence_index=evidence_index,
            )
        run_metadata["status"] = "complete"
        run_metadata["evaluation_summary"] = evaluation.get("summary")
        if threshold_analysis is not None:
            run_metadata["validation_threshold_analysis"] = {
                key: value for key, value in threshold_analysis.items() if key != "curve"
            }
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
