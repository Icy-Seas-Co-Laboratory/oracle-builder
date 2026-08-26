from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    """Canonical paths for a model-run artifact directory."""

    root: Path

    def __init__(self, root: str | Path):
        object.__setattr__(self, "root", Path(root).expanduser().resolve())

    @property
    def manifest(self) -> Path:
        return self.root / "artifact.json"

    @property
    def checksums(self) -> Path:
        return self.root / "checksums.sha256"

    @property
    def readme(self) -> Path:
        return self.root / "README.md"

    @property
    def model_card(self) -> Path:
        return self.root / "MODEL_CARD.md"

    @property
    def model_contract(self) -> Path:
        return self.root / "model" / "contract.json"

    @property
    def source_config(self) -> Path:
        return self.root / "config" / "source.toml"

    @property
    def resolved_config(self) -> Path:
        return self.root / "config" / "resolved.json"

    @property
    def split_manifest(self) -> Path:
        return self.root / "protocol" / "splits.json"

    @property
    def runtime(self) -> Path:
        return self.root / "provenance" / "runtime.json"

    @property
    def environment(self) -> Path:
        return self.root / "provenance" / "environment.json"

    @property
    def requirements(self) -> Path:
        return self.root / "provenance" / "requirements.txt"

    @property
    def distribution(self) -> Path:
        return self.root / "provenance" / "distribution.json"

    @property
    def training_log(self) -> Path:
        return self.root / "logs" / "training.sqlite"

    @property
    def events_jsonl(self) -> Path:
        return self.root / "logs" / "events.jsonl"

    @property
    def metrics_csv(self) -> Path:
        return self.root / "metrics" / "history.csv"

    @property
    def metrics_json(self) -> Path:
        return self.root / "metrics" / "history.json"

    @property
    def metrics_jsonl(self) -> Path:
        return self.root / "metrics" / "metrics.jsonl"

    @property
    def model(self) -> Path:
        return self.root / "model"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def self_supervised_metrics(self) -> Path:
        return self.root / "metrics" / "pretraining"

    @property
    def self_supervised_metrics_jsonl(self) -> Path:
        return self.self_supervised_metrics / "metrics.jsonl"

    @property
    def self_supervised_model(self) -> Path:
        return self.root / "model" / "pretraining"

    # Legacy path names remain aliases so V1 readers and existing tooling keep
    # working while new code can describe the phase accurately.
    @property
    def pretraining_metrics(self) -> Path:
        return self.self_supervised_metrics

    @property
    def pretraining_metrics_jsonl(self) -> Path:
        return self.self_supervised_metrics_jsonl

    @property
    def pretraining_model(self) -> Path:
        return self.self_supervised_model

    @property
    def recovery(self) -> Path:
        return self.model / "recovery"

    @property
    def recovery_model(self) -> Path:
        return self.recovery / "latest.keras"

    @property
    def recovery_state(self) -> Path:
        return self.recovery / "state.json"

    def create_directories(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for path in (
            self.source_config.parent,
            self.split_manifest.parent,
            self.environment.parent,
            self.training_log.parent,
            self.metrics_csv.parent,
            self.model / "checkpoints",
            self.recovery,
            self.evaluation,
            self.predictions,
            self.figures,
        ):
            path.mkdir(parents=True, exist_ok=True)
