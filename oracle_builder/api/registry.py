from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from oracle_builder.inference import InferenceBundle


class ModelNotFoundError(KeyError):
    pass


@dataclass
class RegisteredModel:
    alias: str
    run_dir: Path
    bundle: InferenceBundle | None = None
    load_error: str | None = None
    inference_lock: threading.Lock = field(default_factory=threading.Lock)


class InferenceModelRegistry:
    """Thread-safe alias and immutable-artifact lookup for resident bundles."""

    def __init__(self, *, loader: Callable[[Path], InferenceBundle] | None = None):
        self._loader = loader or InferenceBundle.load
        self._models: dict[str, RegisteredModel] = {}
        self._artifact_aliases: dict[str, str] = {}
        self._lock = threading.RLock()

    def register(self, alias: str, run_dir: str | Path) -> None:
        normalized = alias.strip()
        if not normalized or "/" in normalized:
            raise ValueError("Model alias must be a non-empty path-safe string")
        path = Path(run_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        with self._lock:
            if normalized in self._models:
                raise ValueError(f"Model alias is already registered: {normalized}")
            self._models[normalized] = RegisteredModel(normalized, path)

    def load(self, selector: str) -> InferenceBundle:
        with self._lock:
            alias = self._artifact_aliases.get(selector, selector)
            registration = self._models.get(alias)
            if registration is None:
                raise ModelNotFoundError(selector)
            if registration.bundle is not None:
                return registration.bundle
            try:
                registration.bundle = self._loader(registration.run_dir)
                registration.load_error = None
                reference = registration.bundle.model_reference
                self._artifact_aliases[reference.artifact_id] = alias
                if reference.artifact_fingerprint:
                    self._artifact_aliases[reference.artifact_fingerprint] = alias
                return registration.bundle
            except Exception as exc:
                registration.load_error = f"{type(exc).__name__}: {exc}"
                raise

    def preload(self, *, raise_errors: bool = False) -> None:
        for alias in list(self._models):
            try:
                self.load(alias)
            except Exception:
                if raise_errors:
                    raise

    def predict(self, selector: str, items: list[Any]):
        """Run one batch while bounding concurrency independently per model."""

        bundle = self.load(selector)
        with self._lock:
            alias = self._artifact_aliases.get(selector, selector)
            registration = self._models[alias]
        with registration.inference_lock:
            return bundle.predict_batch(items)

    def describe(self, *, task: str | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self._lock:
            registrations = list(self._models.values())
        for registration in registrations:
            row: dict[str, Any] = {
                "alias": registration.alias,
                "loaded": registration.bundle is not None,
                "available": registration.load_error is None,
                "load_error": registration.load_error,
            }
            if registration.bundle is not None:
                bundle = registration.bundle
                row["model"] = bundle.model_reference.to_dict()
                row["task"] = bundle.model_reference.task
                row["architecture"] = bundle.model_reference.architecture
                row["capabilities"] = self._capabilities(bundle)
            if task is None or row.get("task") == task:
                rows.append(row)
        return sorted(rows, key=lambda value: value["alias"])

    @staticmethod
    def _capabilities(bundle: InferenceBundle) -> dict[str, Any]:
        config = getattr(bundle, "config", {}) or {}
        labels = list((config.get("dataset") or {}).get("labels") or [])
        model = config.get("model") or {}
        evidence = config.get("evidence") or {}
        evidence_index = getattr(bundle, "evidence_index", None)
        return {
            "contract_version": "1.0.0",
            "labels": [
                {
                    "class_index": int(label.get("class_index", index)),
                    "label_id": label.get("label_id"),
                    "name": label.get("name"),
                }
                for index, label in enumerate(labels)
            ],
            "embedding": {
                "available": bool(model.get("embedding_dim")),
                "dimension": model.get("embedding_dim"),
                "normalized": bool(model.get("normalize_embeddings", True)),
            },
            "evidence": {
                "available": evidence_index is not None,
                "schema_version": 1 if evidence_index is not None else None,
                "knn_k": int(evidence.get("knn_k", 5)),
                "prototype": evidence_index is not None,
                "knn": evidence_index is not None,
                "visual_exemplars": False,
            },
        }

    def evidence_item(self, selector: str, item_id: str) -> dict[str, Any]:
        """Return metadata for a packaged evidence exemplar without exposing arrays."""

        bundle = self.load(selector)
        index = getattr(bundle, "evidence_index", None)
        if index is None:
            raise ModelNotFoundError(item_id)
        matches = [position for position, value in enumerate(index.uuids) if str(value) == item_id]
        if not matches:
            raise ModelNotFoundError(item_id)
        position = matches[0]
        class_index = int(index.labels[position])
        labels = self._capabilities(bundle)["labels"]
        label = next((value for value in labels if value["class_index"] == class_index), None)
        return {
            "item_id": item_id,
            "class_index": class_index,
            "label": label,
            "image_available": False,
        }

    @property
    def registered_count(self) -> int:
        return len(self._models)

    @property
    def healthy(self) -> bool:
        return bool(self._models) and all(
            row.bundle is not None and row.load_error is None
            for row in self._models.values()
        )
