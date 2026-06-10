from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_classification(model, x: np.ndarray, y: np.ndarray, records: list[dict[str, Any]], run_dir: str | Path) -> dict[str, Any]:
    probabilities = model.predict(x, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    labels = sorted(set(y.astype(int).tolist()) | set(predicted.astype(int).tolist()))
    matrix = confusion_matrix(y, predicted, labels=labels)
    report = classification_report(y, predicted, labels=labels, output_dict=True, zero_division=0)
    evaluation_dir = Path(run_dir) / "evaluation"
    figures_dir = Path(run_dir) / "figures"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_df.to_csv(evaluation_dir / "confusion_matrix.csv")
    (evaluation_dir / "confusion_matrix.json").write_text(
        json.dumps({"labels": labels, "matrix": matrix.tolist()}, indent=2) + "\n"
    )
    (evaluation_dir / "classification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    sample_rows = []
    for row, true_value, pred_value, probs in zip(records, y, predicted, probabilities, strict=False):
        sample_rows.append(
            {
                "uuid": row["uuid"],
                "split": row["split"],
                "y_true": int(true_value),
                "y_pred": int(pred_value),
                "correct": bool(int(true_value) == int(pred_value)),
                "confidence": float(np.max(probs)),
            }
        )
    pd.DataFrame(sample_rows).to_csv(evaluation_dir / "sample_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    summary = {
        "task": "classification",
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
    }
    (evaluation_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"summary": summary, "probabilities": probabilities, "predicted": predicted}

