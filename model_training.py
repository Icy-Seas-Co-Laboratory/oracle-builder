#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import sys
import threading
import uuid
from pathlib import Path

# TensorFlow's GPU timer can emit one warning per small kernel during autotuning.
# Do not inherit a noisy generic TensorFlow setting from a shell or conda profile.
# Set ORACLE_BUILDER_TF_CPP_MIN_LOG_LEVEL=0 before invocation to opt into native
# TensorFlow diagnostics.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get(
    "ORACLE_BUILDER_TF_CPP_MIN_LOG_LEVEL", "2"
)

_GPU_TIMER_WARNING = b"gpu_timer.cc"
_GPU_TIMER_WARNING_TEXT = b"Skipping the delay kernel, measurement accuracy will be reduced"


def _is_gpu_timer_warning(line: bytes) -> bool:
    return _GPU_TIMER_WARNING in line and _GPU_TIMER_WARNING_TEXT in line


def _install_gpu_timer_warning_filter() -> None:
    """Suppress only repeated XLA timer warnings written directly to stderr."""
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
                if _is_gpu_timer_warning(line):
                    suppressed += 1
                    if suppressed == 1:
                        os.write(
                            original_stderr,
                            b"TensorFlow/XLA GPU timer warnings suppressed; "
                            b"set ORACLE_BUILDER_FILTER_GPU_TIMER_WARNINGS=0 to show them.\n",
                        )
                    continue
                os.write(original_stderr, line)

    thread = threading.Thread(
        target=forward_stderr,
        name="gpu-timer-stderr-filter",
        daemon=True,
    )
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

from oracle_builder.artifacts import (
    RunLayout,
    attach_split_manifest,
    create_run_artifact,
    create_split_manifest,
    read_run_config,
    read_run_manifest,
    read_run_runtime,
    read_split_manifest,
    reopen_run_artifact,
    seal_run_artifact,
    update_run_artifact,
    validate_run_artifact,
    write_run_config,
)
from oracle_builder.config import resolve_config, self_supervised_settings
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
    parser.add_argument("-c", "--config")
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    parser.add_argument("--runs-dir", default="./runs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        metavar="RUN_DIRECTORY",
        help="Resume a run from its rolling recovery snapshot.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Validate segmentation SQLite dataset compatibility and exit.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resume_state = None
    is_resume = bool(args.resume)
    if is_resume:
        if args.dry_run or args.preflight:
            raise ValueError("--dry-run and --preflight cannot be combined with --resume")
        if args.config or args.output or args.overwrite:
            raise ValueError("--resume uses the existing artifact; do not pass --config, --output, or --overwrite")
        run_dir = Path(args.resume).expanduser().resolve()
        manifest = read_run_manifest(run_dir)
        if manifest["status"] == "complete":
            raise ValueError("Completed runs cannot be resumed; start a new run instead")
        report = validate_run_artifact(run_dir)
        if not report["valid"]:
            raise ValueError("Cannot resume an invalid run artifact: " + "; ".join(report["errors"]))
        config = read_run_config(run_dir)
        runtime = read_run_runtime(run_dir)
        input_path = args.input or runtime.get("paths", {}).get("input_path")
        if not input_path:
            raise ValueError("The run has no recorded input path; provide --input DATASET.sqlite")
        args.input = str(Path(input_path).expanduser().resolve())
        args.output = str(manifest["name"])
        config["paths"] = {
            **runtime.get("paths", {}),
            "input_path": args.input,
            "run_dir": str(run_dir),
        }
        config["run"]["run_id"] = manifest["run_id"]
        config["run"]["run_name"] = manifest["name"]
        config["artifact"] = {
            "artifact_id": manifest["artifact_id"],
            "schema_name": manifest["artifact_schema"]["name"],
            "schema_version": manifest["artifact_schema"]["version"],
        }
        if config.get("run", {}).get("task") == "embedding":
            raise ValueError(
                "Embedding runs currently restart from their sealed training record; "
                "use oracle-embed with a new output"
            )
        split_manifest = read_split_manifest(run_dir)
        attach_split_manifest(config, split_manifest)
        from oracle_builder.artifacts.splits import split_manifest_matches_dataset
        from oracle_builder.training.recovery import validate_recovery_state

        if not split_manifest_matches_dataset(config, args.input):
            raise ValueError("The supplied dataset does not exactly match this run's split manifest")
        resume_state = validate_recovery_state(
            run_dir,
            config,
            artifact_id=manifest["artifact_id"],
            run_id=manifest["run_id"],
        )
        if manifest["lifecycle"] == "sealed":
            reopen_run_artifact(run_dir, reason="resume from validated recovery snapshot")
        update_run_artifact(run_dir, status="running")
        print(
            f"Resuming {run_dir.name} at supervised epoch "
            f"{int(resume_state['completed_epoch']) + 1}",
            flush=True,
        )
    else:
        if not args.config or not args.input or not args.output:
            raise ValueError("New training requires --config, --input, and --output")
        run_dir = Path(args.runs_dir) / args.output
        config = resolve_config(args.config, args.input, run_dir)
    if not args.preflight and not args.dry_run and config["dataset"]["lifecycle"] != "frozen":
        raise ValueError(
            "Model training requires a frozen dataset checkpoint. Run "
            f"`oracle-dataset checkpoint {args.input}` and train from the resulting file."
        )
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
    if not is_resume and config["run"]["task"] == "embedding":
        from oracle_builder.embedding.training import train_embedding_run

        result = train_embedding_run(
            args.config,
            args.input,
            run_dir,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    if args.overwrite and run_dir.exists() and not is_resume:
        shutil.rmtree(run_dir)
    if not is_resume:
        run_dir = create_run_dir(args.runs_dir, args.output)
        run_id = str(uuid.uuid4())
        config["run"]["run_id"] = run_id
        config["run"]["run_name"] = args.output
        manifest = create_run_artifact(
            run_dir,
            run_id=run_id,
            name=args.output,
            config=config,
            source_config=args.config,
        )
        split_manifest = create_split_manifest(run_dir, args.input, config)
        attach_split_manifest(config, split_manifest)
        config["artifact"] = {
            "artifact_id": manifest["artifact_id"],
            "schema_name": manifest["artifact_schema"]["name"],
            "schema_version": manifest["artifact_schema"]["version"],
        }
        write_run_config(run_dir, config)
    else:
        run_id = config["run"]["run_id"]
    layout = RunLayout(run_dir)
    environment = write_environment(run_dir)
    training_log = layout.training_log
    from oracle_builder.training.logging_callbacks import init_training_log, log_event, mark_run_complete

    init_training_log(
        training_log,
        run_id,
        args.output,
        config,
        environment,
        resume=is_resume,
    )
    if is_resume:
        log_event(
            training_log,
            run_id,
            "INFO",
            "Resumed from validated rolling recovery snapshot",
            {"completed_epoch": int(resume_state["completed_epoch"])},
        )

    try:
        from oracle_builder.data.sqlite_dataset import load_arrays, load_prediction_arrays, make_tf_datasets
        from oracle_builder.evaluation.predictions import write_predictions_db
        from oracle_builder.evaluation.reports import evaluate_run_model
        from oracle_builder.evaluation.thresholds import analyze_validation_threshold
        from oracle_builder.saving.load_test import run_load_tests
        from oracle_builder.saving.save_model import save_model_artifacts, write_load_test_report
        from oracle_builder.training.train import train_model

        streaming_bundle = None
        if (
            config["run"]["task"] == "classification"
            and config.get("data", {}).get("streaming", {}).get("enabled", True)
        ):
            from oracle_builder.data.sqlite_stream import (
                build_classification_index,
                make_streaming_classification_bundle,
            )

            streaming_bundle = make_streaming_classification_bundle(args.input, config)
            datasets = streaming_bundle.datasets
            records_by_split = {
                split: list(index.iter_records())
                for split, index in streaming_bundle.indices.items()
            }
        else:
            datasets, records_by_split = make_tf_datasets(args.input, config)
        log_event(training_log, run_id, "INFO", "Datasets loaded", {"splits": list(datasets)})
        if config["run"]["task"] == "classification":
            from oracle_builder.training.class_weights import (
                resolve_class_weights,
                uses_weighted_cross_entropy,
            )

            if uses_weighted_cross_entropy(config):
                from oracle_builder.data.sqlite_stream import (
                    build_classification_index,
                )

                weight_index = build_classification_index(
                    args.input,
                    config,
                    "train",
                    labeled_only=True,
                )
                resolved_weights = resolve_class_weights(
                    [ref.target for ref in weight_index.refs],
                    int(config["data"]["num_classes"]),
                    config["training"].get("class_weights", {}),
                )
                config["training"]["class_weights"] = resolved_weights
                write_run_config(run_dir, config)
                log_event(
                    training_log,
                    run_id,
                    "INFO",
                    "Resolved weighted cross entropy class weights",
                    resolved_weights,
                )
        self_supervised_dataset = None
        self_supervised = self_supervised_settings(config)
        if self_supervised.get("enabled", False) and resume_state is None:
            if str(self_supervised.get("method", "byol")).lower() == "grayscale_reconstruction":
                from oracle_builder.training.student_teacher import load_grayscale_self_supervised_dataset
                source_database = self_supervised.get("database", args.input)
                self_supervised_dataset = load_grayscale_self_supervised_dataset(source_database, config)
                self_supervised_count = "database"
            elif streaming_bundle is not None:
                pretraining_index = build_classification_index(
                    args.input,
                    config,
                    "train",
                    labeled_only=False,
                )
                self_supervised_dataset = streaming_bundle.source.image_dataset(
                    pretraining_index,
                    shuffle=True,
                )
                self_supervised_count = len(pretraining_index)
            else:
                from oracle_builder.training.student_teacher import make_self_supervised_dataset

                self_supervised_x, _, self_supervised_records = load_prediction_arrays(
                    args.input,
                    config,
                    split="train",
                )
                self_supervised_dataset = make_self_supervised_dataset(self_supervised_x, config)
                self_supervised_count = len(self_supervised_records)
            log_event(
                training_log,
                run_id,
                "INFO",
        "Loaded self-supervised training inputs",
                {"samples": self_supervised_count, "split": "train"},
            )
        model, history = train_model(
            config,
            datasets,
            run_dir,
            training_log,
            run_id,
            pretraining_dataset=self_supervised_dataset,
            resume_state=resume_state,
        )
        from oracle_builder.inference.batching import (
            resolve_inference_batch_size,
        )
        from oracle_builder.progress import PostTrainingProgress

        post_stage_count = 6
        if config["run"]["task"] == "classification" and config.get(
            "evidence", {}
        ).get("enabled", True):
            post_stage_count += 1
        if config["run"]["task"] == "segmentation" and "validation" in datasets:
            post_stage_count += 1
        if config.get("output", {}).get("save_predictions", True):
            post_stage_count += 1
        post_progress = PostTrainingProgress(post_stage_count)

        with post_progress.stage("Rendering training-history figures"):
            plot_history(history, run_dir)
        with post_progress.stage("Selecting a safe inference batch size"):
            inference_batch_plan = resolve_inference_batch_size(model, config)
            inference_batch_size = inference_batch_plan.batch_size
            log_event(
                training_log,
                run_id,
                "INFO",
                "Resolved inference batch size",
                inference_batch_plan.to_dict(),
            )
        evidence_index = None
        if (
            config["run"]["task"] == "classification"
            and config.get("evidence", {}).get("enabled", True)
        ):
            with post_progress.stage(
                "Building prototype and nearest-neighbor evidence"
            ):
                if streaming_bundle is not None:
                    from oracle_builder.classification.evidence import (
                        build_evidence_index_streaming,
                    )

                    evidence_sample_index = streaming_bundle.indices["train"]
                    evidence_index = build_evidence_index_streaming(
                        model,
                        streaming_bundle.source.indexed_image_dataset(
                            evidence_sample_index,
                            batch_size=inference_batch_size,
                        ),
                        evidence_sample_index,
                        run_dir / "model" / "classification_evidence",
                        progress=bool(
                            config.get("inference", {}).get("progress", True)
                        ),
                    )
                    evidence_reference_count = len(evidence_sample_index)
                else:
                    from oracle_builder.classification.evidence import (
                        build_evidence_index,
                    )

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
                        run_dir / "model" / "classification_evidence",
                    )
                    evidence_reference_count = len(evidence_records)
            log_event(
                training_log,
                run_id,
                "INFO",
                "Built classification prototype and KNN evidence index",
                {
                    "reference_count": evidence_reference_count,
                    "classes": [int(value) for value in evidence_index.prototype_labels],
                },
            )
        threshold_analysis = None
        if config["run"]["task"] == "segmentation" and "validation" in datasets:
            with post_progress.stage(
                "Optimizing the segmentation threshold on validation data"
            ):
                validation_x, validation_y, validation_records = load_arrays(
                    args.input, config, split="validation"
                )
                threshold_analysis = analyze_validation_threshold(
                    model,
                    validation_x,
                    validation_y,
                    run_dir,
                    config=config,
                    records=validation_records,
                )
                config.setdefault("evaluation", {})[
                    "segmentation_threshold"
                ] = threshold_analysis["best_threshold"]
                write_run_config(run_dir, config)
            log_event(
                training_log,
                run_id,
                "INFO",
                "Optimized segmentation probability threshold on validation data",
                threshold_analysis,
            )
        with post_progress.stage("Saving portable model formats and manifests"):
            save_report = save_model_artifacts(model, run_dir, config)
        with post_progress.stage("Reloading and testing saved model formats"):
            load_report = run_load_tests(run_dir, config, save_report)
            write_load_test_report(run_dir, load_report)
        with post_progress.stage("Evaluating the held-out test split"):
            evaluation = evaluate_run_model(
                model,
                config,
                args.input,
                run_dir,
                split="test",
                inference_batch_size=inference_batch_size,
            )
        if config.get("output", {}).get("save_predictions", True):
            with post_progress.stage(
                "Generating and storing predictions for every dataset split"
            ):
                predictions_path = (
                    run_dir / "predictions" / "predictions.sqlite"
                )
                if streaming_bundle is not None:
                    from oracle_builder.data.sqlite_stream import (
                        build_all_classification_index,
                    )
                    from oracle_builder.evaluation.predictions import (
                        write_classification_predictions_streaming,
                    )

                    prediction_index = build_all_classification_index(
                        args.input,
                        config,
                        labeled_only=False,
                    )
                    write_classification_predictions_streaming(
                        model,
                        streaming_bundle.source.indexed_image_dataset(
                            prediction_index,
                            batch_size=inference_batch_size,
                        ),
                        prediction_index,
                        config,
                        predictions_path,
                        source_sqlite=args.input,
                        prediction_set=args.output,
                        evidence_index=evidence_index,
                        progress=bool(
                            config.get("inference", {}).get("progress", True)
                        ),
                    )
                else:
                    x, targets, records = load_prediction_arrays(
                        args.input, config
                    )
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
                        inference_batch_size=inference_batch_size,
                    )
        summary = {
            "evaluation": evaluation.get("summary"),
            "inference_batching": inference_batch_plan.to_dict(),
        }
        if threshold_analysis is not None:
            summary["validation_threshold_analysis"] = {
                key: value for key, value in threshold_analysis.items() if key != "curve"
            }
        with post_progress.stage("Finalizing and sealing the run artifact"):
            from oracle_builder.training.recovery import clear_recovery_snapshot

            clear_recovery_snapshot(run_dir)
            mark_run_complete(training_log, run_id, "complete")
            update_run_artifact(run_dir, status="complete", summary=summary)
            seal_run_artifact(run_dir)
        print(f"Training run complete: {run_dir}", flush=True)
        return 0
    except KeyboardInterrupt:
        log_event(training_log, run_id, "WARNING", "Training interrupted", {})
        mark_run_complete(training_log, run_id, "interrupted")
        update_run_artifact(run_dir, status="interrupted", error="Interrupted by user")
        seal_run_artifact(run_dir)
        print(f"Training interrupted; resume with --resume {run_dir}", file=sys.stderr)
        return 130
    except Exception as exc:
        log_event(training_log, run_id, "ERROR", "Training run failed", {"error": str(exc)})
        mark_run_complete(training_log, run_id, "failed")
        update_run_artifact(run_dir, status="failed", error=str(exc))
        seal_run_artifact(run_dir)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
