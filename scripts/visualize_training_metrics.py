#!/usr/bin/env python3
"""Render training and top-K evaluation metrics from an Oracle Builder run."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _histories(run_dir: Path) -> dict[str, dict[str, list[float]]]:
    payload = _read_json(run_dir / "metrics" / "history.json") or {}
    if isinstance(payload.get("strata"), dict):
        return {
            str(name): {
                str(metric): [float(value) for value in values]
                for metric, values in history.items()
                if isinstance(values, list)
            }
            for name, history in payload["strata"].items()
            if isinstance(history, dict)
        }
    return {
        "shared": {
            str(metric): [float(value) for value in values]
            for metric, values in payload.items()
            if isinstance(values, list)
        }
    }


def _cycle_metrics(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "logs" / "training.sqlite"
    if not path.exists():
        return []
    try:
        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                "SELECT details_json FROM events "
                "WHERE message = 'Completed stratified cycle validation' "
                "ORDER BY rowid"
            ).fetchall()
    except sqlite3.Error:
        return []
    result = []
    for (encoded,) in rows:
        try:
            value = json.loads(encoded)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _evaluations(run_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    root = _read_json(run_dir / "evaluation" / "evaluation_summary.json")
    if root:
        result["shared"] = root
    for path in sorted((run_dir / "model" / "strata").glob("*/evaluation/evaluation_summary.json")):
        summary = _read_json(path)
        if summary:
            result[path.parents[1].name] = summary
    return result


def _plot_histories(axis, histories: dict[str, dict[str, list[float]]], metric: str) -> None:
    plotted = False
    for name, history in sorted(histories.items(), key=lambda row: row[0]):
        values = history.get(metric, [])
        if values:
            axis.plot(np.arange(1, len(values) + 1), values, label=f"{name} train")
            plotted = True
    if not plotted:
        axis.text(0.5, 0.5, f"No {metric} history recorded", ha="center", va="center")
    axis.set_xlabel("Epoch within stratum" if len(histories) > 1 else "Epoch")
    axis.set_ylabel(metric.replace("_", " "))
    axis.grid(alpha=0.25)


def _plot_cycle_validation(axis, cycles: list[dict[str, Any]], metric: str) -> None:
    values = [
        float(cycle.get("aggregate", {}).get(metric))
        for cycle in cycles
        if cycle.get("aggregate", {}).get(metric) is not None
    ]
    if values:
        axis.plot(
            np.arange(1, len(values) + 1), values, "o--", color="black",
            label=f"cycle validation {metric}",
        )


def _plot_top_k(axis, evaluations: dict[str, dict[str, Any]]) -> None:
    if not evaluations:
        axis.text(0.5, 0.5, "No evaluation summaries found", ha="center", va="center")
        axis.set_axis_off()
        return
    labels = list(evaluations)
    requested = ("top_1_accuracy", "top_3_accuracy", "top_5_accuracy")
    positions = np.arange(len(labels))
    width = 0.24
    for index, metric in enumerate(requested):
        values = [summary.get(metric, summary.get("accuracy", np.nan)) for summary in evaluations.values()]
        axis.bar(positions + (index - 1) * width, values, width, label=metric.replace("_accuracy", "").replace("_", "-"))
    axis.set_xticks(positions, labels=labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Accuracy")
    axis.set_title("Held-out top-K accuracy")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Oracle Builder run directory")
    parser.add_argument("--output", type=Path, help="PNG destination (default: RUN/figures/training_metrics.png)")
    parser.add_argument("--show", action="store_true", help="Open the figure after writing it")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    histories = _histories(run_dir)
    evaluations = _evaluations(run_dir)
    cycles = _cycle_metrics(run_dir)
    output = args.output or run_dir / "figures" / "training_metrics.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    _plot_histories(axes[0], histories, "loss")
    _plot_cycle_validation(axes[0], cycles, "loss")
    axes[0].set_title("Training loss")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend()
    _plot_histories(axes[1], histories, "accuracy")
    _plot_cycle_validation(axes[1], cycles, "accuracy")
    axes[1].set_title("Training accuracy")
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend()
    _plot_top_k(axes[2], evaluations)
    figure.suptitle(f"Training metrics: {run_dir.name}")
    figure.tight_layout()
    figure.savefig(output, dpi=180)

    report = {
        name: {
            metric: summary.get(metric)
            for metric in ("accuracy", "top_3_accuracy", "top_5_accuracy")
        }
        for name, summary in evaluations.items()
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Wrote {summary_path}")
    for name, values in report.items():
        print(
            f"{name}: accuracy={values['accuracy']!s}, "
            f"top-3={values['top_3_accuracy']!s}, top-5={values['top_5_accuracy']!s}"
        )
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
