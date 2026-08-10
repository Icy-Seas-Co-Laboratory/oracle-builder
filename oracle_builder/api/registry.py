from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from oracle_builder.inference import InferenceBundle
from oracle_builder.api.microbatch import MicroBatchExecutor


_ALIAS_SAFE = re.compile(r"[^a-z0-9._-]+")


def _root_alias(path: Path, root: Path, manifest: dict[str, Any]) -> str:
    """Produce a stable HTTP-safe selector from artifact metadata or location."""
    preferred = str(manifest.get("name") or path.relative_to(root).as_posix())
    alias = _ALIAS_SAFE.sub("-", preferred.lower()).strip("-.")
    return alias or "model"


def _manifest_catalog(run_dir: Path) -> dict[str, Any]:
    """Read display-safe model identity without loading the ML runtime."""

    manifest_path = run_dir / "artifact.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    model = manifest.get("model") or {}
    outputs = model.get("outputs") or {}
    labels = list(outputs.get("labels") or [])
    task = model.get("task")
    architecture = model.get("architecture")
    reference = {
        "artifact_id": manifest.get("artifact_id"),
        "run_id": manifest.get("run_id"),
        "task": task,
        "architecture": architecture,
    }
    return {
        "task": task,
        "architecture": architecture,
        "model": {key: value for key, value in reference.items() if value is not None},
        "capabilities": {
            "contract_version": str((manifest.get("contract") or {}).get("version") or "1.0.0"),
            "labels": labels,
            "embedding": {
                "available": bool(outputs.get("embedding")),
                "dimension": outputs.get("embedding_dimension"),
                "normalized": bool(outputs.get("embedding_normalized", True)),
            },
            "evidence": {
                "available": False,
                "schema_version": None,
                "prototype": False,
                "knn": False,
                "visual_exemplars": False,
            },
        },
    }


class ModelNotFoundError(KeyError):
    pass


@dataclass
class RegisteredModel:
    alias: str
    run_dir: Path
    bundle: InferenceBundle | None = None
    load_error: str | None = None
    catalog: dict[str, Any] = field(default_factory=dict)
    executor: MicroBatchExecutor | None = None


class InferenceModelRegistry:
    """Thread-safe alias and immutable-artifact lookup for resident bundles."""

    def __init__(
        self,
        *,
        loader: Callable[[Path], InferenceBundle] | None = None,
        serving_max_batch_size: int = 256,
        serving_max_wait_ms: int = 8,
        serving_queue_capacity: int = 1024,
    ):
        self._loader = loader or InferenceBundle.load
        self._models: dict[str, RegisteredModel] = {}
        self._artifact_aliases: dict[str, str] = {}
        self._lock = threading.RLock()
        self._serving_max_batch_size = max(1, int(serving_max_batch_size))
        self._serving_max_wait_ms = max(0, int(serving_max_wait_ms))
        self._serving_queue_capacity = max(1, int(serving_queue_capacity))

    def register(self, alias: str, run_dir: str | Path) -> None:
        normalized = alias.strip()
        if not normalized or "/" in normalized or _ALIAS_SAFE.search(normalized.lower()):
            raise ValueError("Model alias must be a non-empty path-safe string")
        path = Path(run_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        with self._lock:
            if normalized in self._models:
                raise ValueError(f"Model alias is already registered: {normalized}")
            self._models[normalized] = RegisteredModel(
                normalized,
                path,
                catalog=_manifest_catalog(path),
            )

    def register_root(self, root: str | Path) -> dict[str, list[dict[str, str]]]:
        """Discover sealed Oracle Builder artifacts beneath a model root.

        Discovery only registers artifacts whose manifest identifies a sealed
        model product or model run. Loading remains lazy unless the application
        elects to preload, so discovery does not allocate model memory.
        """
        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir():
            raise NotADirectoryError(root_path)
        registered: list[dict[str, str]] = []
        skipped: list[dict[str, str]] = []
        for manifest_path in sorted(root_path.rglob("artifact.json")):
            run_dir = manifest_path.parent
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                skipped.append({"path": str(run_dir), "reason": f"invalid manifest: {exc}"})
                continue
            schema = manifest.get("artifact_schema", {})
            if schema.get("name") != "oracle_builder_model_run":
                skipped.append({"path": str(run_dir), "reason": "unsupported artifact schema"})
                continue
            if manifest.get("lifecycle") != "sealed":
                skipped.append({"path": str(run_dir), "reason": "artifact is not sealed"})
                continue
            if manifest.get("artifact_type", "model_run") not in {"model_run", "model_product"}:
                skipped.append({"path": str(run_dir), "reason": "unsupported artifact type"})
                continue
            alias = _root_alias(run_dir, root_path, manifest)
            with self._lock:
                if alias in self._models:
                    artifact_id = str(manifest.get("artifact_id", ""))
                    alias = f"{alias}-{artifact_id[:8]}" if artifact_id else alias
                if alias in self._models:
                    skipped.append({"path": str(run_dir), "reason": f"duplicate alias: {alias}"})
                    continue
            self.register(alias, run_dir)
            registered.append({"alias": alias, "path": str(run_dir)})
        return {"registered": registered, "skipped": skipped}

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
                maximum = self._serving_max_batch_size
                warm = getattr(registration.bundle, "warm_for_serving", None)
                if callable(warm):
                    diagnostics = warm(batch_sizes=(1, maximum))
                    resolved = diagnostics.get("resolved_max_batch_size")
                    if isinstance(resolved, int):
                        maximum = min(maximum, max(1, resolved))
                registration.executor = MicroBatchExecutor(
                    registration.bundle,
                    max_batch_size=maximum,
                    max_wait_ms=self._serving_max_wait_ms,
                    queue_capacity=self._serving_queue_capacity,
                )
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
        """Submit one request to the resident model's bounded micro-batcher."""

        self.load(selector)
        with self._lock:
            alias = self._artifact_aliases.get(selector, selector)
            registration = self._models[alias]
        if registration.executor is None:
            raise RuntimeError(f"Inference executor for {selector!r} is unavailable")
        return registration.executor.submit(items)

    def max_items_for(self, selector: str) -> int:
        """Return the startup-configured safe request size for one loaded model."""
        self.load(selector)
        with self._lock:
            alias = self._artifact_aliases.get(selector, selector)
            registration = self._models[alias]
        if registration.executor is None:
            raise RuntimeError(f"Inference executor for {selector!r} is unavailable")
        return registration.executor.max_batch_size

    def close(self) -> None:
        with self._lock:
            executors = [value.executor for value in self._models.values() if value.executor is not None]
        for executor in executors:
            executor.close()

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
            row.update(registration.catalog)
            if registration.bundle is not None:
                bundle = registration.bundle
                row["model"] = bundle.model_reference.to_dict()
                row["task"] = bundle.model_reference.task
                row["architecture"] = bundle.model_reference.architecture
                row["capabilities"] = self._capabilities(bundle)
                row["runtime"] = {
                    **getattr(bundle, "runtime_diagnostics", {}),
                    "microbatch": registration.executor.diagnostics() if registration.executor is not None else None,
                }
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
