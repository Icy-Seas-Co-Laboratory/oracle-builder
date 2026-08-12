from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oracle_data_contracts.datasets import dataset_fingerprint, read_dataset_info
from oracle_builder.config import DEFAULT_CONFIG, deep_merge, load_toml
from oracle_builder.orchestration.database import connect


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    for key in ("metadata_json", "manifest_json", "plan_json", "parameters_json", "resources_json", "selection_json", "protocol_json", "summary_json", "validation_report_json", "readiness_json", "workers_json", "queue_json"):
        if key in result:
            raw = result.pop(key)
            result[key.removesuffix("_json")] = json.loads(raw) if raw is not None else None
    return result


class Orchestrator:
    """SQLite control plane for Oracle Builder artifacts and compute requests."""

    def __init__(
        self,
        database: str | Path,
        *,
        artifact_root: str | Path | None = None,
        browse_roots: list[str | Path] | None = None,
        upload_limit_bytes: int = 10 * 1024 * 1024 * 1024,
        workspace_root: str | Path | None = None,
        compute_endpoints: list[tuple[str, str]] | None = None,
    ):
        self.database = Path(database).expanduser().resolve()
        self.artifact_root = Path(artifact_root).expanduser().resolve() if artifact_root else self.database.parent / "oracle-artifacts"
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        if upload_limit_bytes < 1:
            raise ValueError("upload_limit_bytes must be positive")
        self.upload_limit_bytes = upload_limit_bytes
        self.workspace_root = Path(workspace_root).expanduser().resolve() if workspace_root else Path.cwd().resolve()
        if not self.workspace_root.is_dir():
            raise NotADirectoryError(self.workspace_root)
        self.browse_roots: dict[str, Path] = {"workspace": self.workspace_root, "artifacts": self.artifact_root}
        for index, root in enumerate(browse_roots or [], start=1):
            location = Path(root).expanduser().resolve()
            if not location.is_dir():
                raise NotADirectoryError(location)
            self.browse_roots[f"root-{index}"] = location
        with connect(self.database):
            pass
        for name, base_url in compute_endpoints or []:
            self.register_compute_endpoint(name=name, base_url=base_url)

    def _connection(self) -> sqlite3.Connection:
        return connect(self.database)

    def register_compute_endpoint(self, *, name: str, base_url: str) -> dict[str, Any]:
        normalized = base_url.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("Compute endpoint URL must start with http:// or https://")
        endpoint_id = str(uuid.uuid5(uuid.NAMESPACE_URL, normalized))
        now = _now()
        with self._connection() as db:
            db.execute("""INSERT INTO compute_endpoints VALUES (?, ?, ?, 1, 'unknown', NULL, NULL, NULL, NULL, NULL, ?, ?)
                ON CONFLICT(base_url) DO UPDATE SET name=excluded.name, enabled=1, updated_at=excluded.updated_at""",
                (endpoint_id, name.strip() or normalized, normalized, now, now))
        return self.compute_endpoint(endpoint_id)  # type: ignore[return-value]

    def refresh_compute_endpoint(self, endpoint_id: str) -> dict[str, Any]:
        endpoint = self.compute_endpoint(endpoint_id)
        if endpoint is None:
            raise KeyError(endpoint_id)
        now = _now()
        try:
            readiness = self._request(endpoint["base_url"], "GET", "/health/ready")
            compute = self._request(endpoint["base_url"], "GET", "/compute/status")
            status = "ready" if readiness.get("status") == "ready" and compute.get("status") == "ready" else "degraded"
            error = None
            workers, queue = compute.get("workers") or [], compute.get("queue") or {}
        except RuntimeError as exc:
            status, error, readiness, workers, queue = "unavailable", str(exc), None, [], {}
        with self._connection() as db:
            db.execute("""UPDATE compute_endpoints SET status=?, last_checked_at=?, error=?,
                readiness_json=?, workers_json=?, queue_json=?, updated_at=? WHERE endpoint_id=?""",
                (status, now, error, _json(readiness) if readiness is not None else None,
                 _json(workers), _json(queue), now, endpoint_id))
        return self.compute_endpoint(endpoint_id)  # type: ignore[return-value]

    def preflight(self, specification_id: str, endpoint_id: str, *, refresh: bool = True) -> dict[str, Any]:
        specification = self.specification(specification_id)
        if specification is None:
            raise KeyError(specification_id)
        endpoint = self.refresh_compute_endpoint(endpoint_id) if refresh else self.compute_endpoint(endpoint_id)
        if endpoint is None:
            raise KeyError(endpoint_id)
        reasons: list[str] = []
        if not endpoint["enabled"]:
            reasons.append("Compute endpoint is disabled")
        if endpoint["status"] != "ready":
            reasons.append(endpoint.get("error") or f"Compute endpoint is {endpoint['status']}")
        workers = endpoint.get("workers") or []
        capable = [worker for worker in workers if specification["action"] in (worker.get("capabilities") or {}).get("actions", [])]
        if endpoint["status"] == "ready" and not capable:
            reasons.append(f"No worker supports the {specification['action']} action")
        requested_gpus = int((specification.get("resources") or {}).get("gpu_count") or 0)
        if capable and requested_gpus > max((len((worker.get("capabilities") or {}).get("gpus") or []) for worker in capable), default=0):
            reasons.append(f"Run requests {requested_gpus} GPU(s), but no capable worker advertises that capacity")
        queue = endpoint.get("queue") or {}
        if queue.get("capacity") is not None and queue.get("depth", 0) >= queue["capacity"]:
            reasons.append("Compute queue is full")
        return {
            "ready": not reasons,
            "reasons": reasons,
            "specification_id": specification_id,
            "action": specification["action"],
            "requested_resources": specification["resources"],
            "endpoint": endpoint,
            "capable_workers": capable,
        }

    def file_roots(self) -> list[dict[str, str]]:
        return [{"id": root_id, "path": str(path)} for root_id, path in self.browse_roots.items()]

    def upload_destination(self, kind: str, filename: str) -> Path:
        allowed_extensions = {
            "datasets": {".sqlite"},
            "configs": {".toml"},
            "models": {".keras", ".h5", ".hdf5"},
        }
        if kind not in allowed_extensions:
            raise ValueError(f"Unsupported upload kind: {kind}")
        candidate = Path(filename)
        if candidate.name != filename or filename in {"", ".", ".."}:
            raise ValueError("Upload filename must be a plain filename")
        if candidate.suffix.lower() not in allowed_extensions[kind]:
            expected = ", ".join(sorted(allowed_extensions[kind]))
            raise ValueError(f"{kind} uploads must use one of: {expected}")
        destination = (self.artifact_root / "uploads" / kind / candidate.name).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def files(self, root_id: str, relative_path: str = ".") -> dict[str, Any]:
        """List an allow-listed directory without exposing arbitrary filesystem paths."""
        try:
            root = self.browse_roots[root_id]
        except KeyError as exc:
            raise KeyError(f"Unknown browse root: {root_id}") from exc
        requested = Path(relative_path)
        if requested.is_absolute():
            raise ValueError("File-browser paths must be relative to the selected root")
        directory = (root / requested).resolve()
        try:
            relative = directory.relative_to(root)
        except ValueError as exc:
            raise ValueError("File-browser path escapes the selected root") from exc
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        hidden = {".git", ".venv", "node_modules", ".svelte-kit", "__pycache__"}
        entries: list[dict[str, Any]] = []
        for entry in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if entry.name in hidden:
                continue
            entry_relative = entry.relative_to(root).as_posix()
            entries.append({
                "name": entry.name,
                "path": entry_relative,
                "kind": "directory" if entry.is_dir() else "file",
                "size_bytes": None if entry.is_dir() else entry.stat().st_size,
            })
            if len(entries) >= 500:
                break
        return {
            "root": {"id": root_id, "path": str(root)},
            "path": relative.as_posix() if relative.as_posix() != "." else "",
            "parent": None if relative == Path(".") else relative.parent.as_posix(),
            "entries": entries,
        }

    def ingest_dataset(self, path: str | Path) -> dict[str, Any]:
        source = Path(path).expanduser().resolve()
        self._require_browse_path(source)
        with sqlite3.connect(source) as dataset_connection:
            info = read_dataset_info(dataset_connection)
            fingerprint = dataset_fingerprint(dataset_connection)
        if info["lifecycle"] != "frozen":
            raise ValueError("Only frozen dataset revisions can be registered for training")
        now = _now()
        payload = (info["dataset_id"], info.get("revision_id"), info.get("name") or source.stem,
                   info.get("dataset_type"), info.get("lifecycle"), fingerprint, str(source), _json(info), now, now)
        with self._connection() as db:
            db.execute("""INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET revision_id=excluded.revision_id,name=excluded.name,dataset_type=excluded.dataset_type,lifecycle=excluded.lifecycle,fingerprint_sha256=excluded.fingerprint_sha256,path=excluded.path,metadata_json=excluded.metadata_json,updated_at=excluded.updated_at""", payload)
            return self.dataset(info["dataset_id"], connection=db)  # type: ignore[return-value]

    def create_recipe(self, *, name: str, config_path: str | Path, description: str = "") -> dict[str, Any]:
        source = Path(config_path).expanduser().resolve()
        self._require_browse_path(source)
        if source.suffix.lower() != ".toml":
            raise ValueError("Training recipes must reference a TOML configuration")
        raw = load_toml(source)
        resolved = deep_merge(DEFAULT_CONFIG, raw)
        run = resolved.get("run") or {}
        data = resolved.get("data") or {}
        task, model = str(run.get("task") or ""), str(run.get("model") or "")
        if task not in {"classification", "segmentation"}:
            raise ValueError("Recipe [run].task must be classification or segmentation")
        if not model or not data.get("input_shape"):
            raise ValueError("Recipe requires [run].model and data.input_shape")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        recipe_id, now = str(uuid.uuid4()), _now()
        summary = {"task": task, "model": model, "input_shape": data["input_shape"], "epochs": resolved.get("training", {}).get("epochs"), "loss": resolved.get("training", {}).get("loss"), "seed": run.get("seed", 123)}
        with self._connection() as db:
            db.execute("INSERT INTO recipes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (recipe_id, name.strip() or source.stem, description, str(source), digest, task, model, _json(summary), now, now))
        return self.recipe(recipe_id)  # type: ignore[return-value]

    def create_training_experiment(self, *, name: str, dataset_id: str, recipe_ids: list[str], seeds: list[int], description: str = "", resources: dict[str, Any] | None = None) -> dict[str, Any]:
        if not name.strip() or not recipe_ids or not seeds:
            raise ValueError("Training experiments require a name, at least one recipe, and at least one seed")
        dataset = self.dataset(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        if dataset["lifecycle"] != "frozen":
            raise ValueError("Training experiments require a frozen registered dataset")
        recipes = [self.recipe(recipe_id) for recipe_id in recipe_ids]
        if any(recipe is None for recipe in recipes):
            raise KeyError("One or more recipes were not found")
        selected = [recipe for recipe in recipes if recipe is not None]
        experiment_id, now = str(uuid.uuid4()), _now()
        plan = {"kind": "training", "dataset_id": dataset_id, "recipe_ids": recipe_ids, "seeds": seeds, "resources": resources or {}}
        config_dir = self.artifact_root / "experiments" / experiment_id / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute("INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (experiment_id, None, name.strip(), description, dataset_id, "expanded", _json(plan), now, now))
            ordinal = 0
            for recipe in selected:
                for seed in seeds:
                    ordinal += 1
                    specification_id = str(uuid.uuid4())
                    source = load_toml(recipe["config_path"])
                    source.setdefault("run", {})["seed"] = int(seed)
                    generated = config_dir / f"{ordinal:03d}-{recipe['recipe_id'][:8]}-seed-{seed}.toml"
                    import tomli_w
                    generated.write_text(tomli_w.dumps(source), encoding="utf-8")
                    parameters = self._assign_output_path("train", {"config": str(generated), "input": dataset["path"], "dataset_id": dataset_id, "recipe_id": recipe["recipe_id"], "seed": int(seed)}, specification_id)
                    digest = hashlib.sha256(_json({"parameters": parameters, "dataset_fingerprint": dataset["fingerprint_sha256"]}).encode()).hexdigest()
                    spec_name = f"{name}-{recipe['model']}-seed-{seed}".replace(" ", "-")
                    db.execute("INSERT INTO run_specifications VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (specification_id, experiment_id, ordinal, spec_name, "train", _json(parameters), _json(resources or {}), digest, "planned", None, now, now))
        return self.experiment(experiment_id)  # type: ignore[return-value]

    def create_model_import(self, *, name: str, model_path: str | Path, info_path: str | Path, dataset_id: str | None = None, description: str = "", resources: dict[str, Any] | None = None) -> dict[str, Any]:
        source, info = Path(model_path).expanduser().resolve(), Path(info_path).expanduser().resolve()
        self._require_browse_path(source)
        self._require_browse_path(info)
        if source.suffix.lower() not in {".keras", ".h5", ".hdf5"}:
            raise ValueError("Model import requires a .keras, .h5, or .hdf5 source")
        if info.suffix.lower() != ".toml":
            raise ValueError("Model import metadata must be a TOML file")
        metadata = load_toml(info)
        if not isinstance(metadata.get("product"), dict):
            raise ValueError("Model import metadata requires a [product] section")
        dataset = self.dataset(dataset_id) if dataset_id else None
        if dataset_id and dataset is None:
            raise KeyError(dataset_id)
        experiment_id, specification_id, now = str(uuid.uuid4()), str(uuid.uuid4()), _now()
        plan = {"kind": "model_import", "model_path": str(source), "info_path": str(info), "dataset_id": dataset_id, "resources": resources or {}}
        parameters = self._assign_output_path("model_ingest", {"model": str(source), "info": str(info), **({"dataset": dataset["path"]} if dataset else {})}, specification_id)
        digest = hashlib.sha256(_json(parameters).encode()).hexdigest()
        with self._connection() as db:
            db.execute("INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (experiment_id, None, name.strip() or source.stem, description, dataset_id, "expanded", _json(plan), now, now))
            db.execute("INSERT INTO run_specifications VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (specification_id, experiment_id, 1, f"{name or source.stem}-import", "model_ingest", _json(parameters), _json(resources or {}), digest, "planned", None, now, now))
        return self.experiment(experiment_id)  # type: ignore[return-value]

    def _require_browse_path(self, candidate: Path) -> None:
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        if not any(candidate.is_relative_to(root) for root in self.browse_roots.values()):
            raise ValueError("Path is outside the configured workspace roots")

    def scan(self, root: str | Path) -> dict[str, Any]:
        root_path = Path(root).expanduser().resolve()
        self._require_browse_path(root_path)
        if not root_path.is_dir():
            raise NotADirectoryError(root_path)
        discovered: list[str] = []
        skipped: list[dict[str, str]] = []
        for manifest_path in root_path.rglob("artifact.json"):
            try:
                from oracle_builder.artifacts import validate_run_artifact
                validation = validate_run_artifact(manifest_path.parent)
                if not validation.get("valid"):
                    raise ValueError("artifact validation failed: " + "; ".join(validation.get("errors") or []))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                schema = manifest.get("artifact_schema", {})
                if schema.get("name") != "oracle_builder_model_run":
                    raise ValueError("unsupported artifact schema")
                artifact_id = str(uuid.UUID(str(manifest["artifact_id"])))
                now = _now()
                model, dataset = manifest.get("model") or {}, manifest.get("dataset") or {}
                values = (artifact_id, manifest.get("run_id"), manifest.get("artifact_type", "model_run"), manifest.get("name") or manifest_path.parent.name,
                          model.get("task"), model.get("architecture"), model.get("variant"), manifest.get("status"), manifest.get("lifecycle"), dataset.get("dataset_id"), dataset.get("fingerprint_sha256"), manifest.get("fingerprint_sha256"), str(manifest_path.parent), _json(manifest), now, now)
                with self._connection() as db:
                    db.execute("""INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                      ON CONFLICT(artifact_id) DO UPDATE SET run_id=excluded.run_id,artifact_type=excluded.artifact_type,name=excluded.name,task=excluded.task,architecture=excluded.architecture,variant=excluded.variant,status=excluded.status,lifecycle=excluded.lifecycle,dataset_id=excluded.dataset_id,dataset_fingerprint_sha256=excluded.dataset_fingerprint_sha256,fingerprint_sha256=excluded.fingerprint_sha256,path=excluded.path,manifest_json=excluded.manifest_json,updated_at=excluded.updated_at""", values)
                discovered.append(artifact_id)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                skipped.append({"path": str(manifest_path.parent), "reason": str(exc)})
        return {"root": str(root_path), "artifacts": discovered, "skipped": skipped}

    @staticmethod
    def _duration_seconds(started_at: str | None, completed_at: str | None) -> float | None:
        if not started_at or not completed_at:
            return None
        try:
            return round((datetime.fromisoformat(completed_at.replace("Z", "+00:00")) - datetime.fromisoformat(started_at.replace("Z", "+00:00"))).total_seconds(), 3)
        except ValueError:
            return None

    def _artifact_result(self, artifact: dict[str, Any]) -> dict[str, Any]:
        manifest = artifact["manifest"]
        root = Path(artifact["path"])
        summary = manifest.get("summary") or {}
        evaluation = summary.get("evaluation") if isinstance(summary.get("evaluation"), dict) else None
        if evaluation is None:
            summary_path = root / "evaluation" / "evaluation_summary.json"
            try:
                loaded = json.loads(summary_path.read_text(encoding="utf-8"))
                evaluation = loaded if isinstance(loaded, dict) else None
            except (OSError, json.JSONDecodeError):
                evaluation = None
        metrics = {
            key: float(value)
            for key, value in (evaluation or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        task = artifact.get("task") or (evaluation or {}).get("task")
        split, metric_schema, split_source = None, None, None
        for relative in ("evaluation/metrics_long.csv", "evaluation/segmentation_metrics.csv", "evaluation/sample_metrics.csv"):
            path = root / relative
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    row = next(csv.DictReader(handle), None)
                if row:
                    split = row.get("split") or None
                    metric_schema = row.get("schema_name") or None
                    split_source = relative
                    break
            except OSError:
                continue
        if evaluation and not split:
            # Oracle Builder training evaluates the held-out test split before
            # sealing. Older summaries did not repeat that field explicitly.
            split, split_source = "test", "standard_training_summary"
        primary_names = {
            "classification": ["accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall", "macro_average_precision"],
            "segmentation": ["mean_dice", "mean_iou", "mean_precision", "mean_recall", "mean_pixel_accuracy"],
        }.get(str(task), [])
        primary = {name: metrics[name] for name in primary_names if name in metrics}
        return {
            "artifact_id": artifact["artifact_id"], "name": artifact["name"],
            "artifact_type": artifact["artifact_type"], "task": task,
            "architecture": artifact.get("architecture"), "variant": artifact.get("variant"),
            "status": artifact.get("status"), "lifecycle": artifact.get("lifecycle"),
            "path": artifact["path"], "fingerprint_sha256": artifact.get("fingerprint_sha256"),
            "dataset_id": artifact.get("dataset_id"),
            "dataset_fingerprint_sha256": artifact.get("dataset_fingerprint_sha256"),
            "metrics": metrics, "primary_metrics": primary,
            "protocol": {
                "task": task, "dataset_fingerprint_sha256": artifact.get("dataset_fingerprint_sha256"),
                "split": split, "split_source": split_source, "metric_schema": metric_schema,
                "decision_rule": (evaluation or {}).get("decision_rule"),
                "segmentation_target": (evaluation or {}).get("segmentation_target"),
            },
        }

    @staticmethod
    def _evidence_csv(path: Path, *, limit: int = 200) -> list[dict[str, Any]]:
        """Read a bounded, display-oriented view of a standard evidence table."""
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))[:limit]
        except OSError:
            return []
        converted: list[dict[str, Any]] = []
        for row in rows:
            clean: dict[str, Any] = {}
            for key, value in row.items():
                if value in {None, ""}:
                    clean[key] = None
                    continue
                try:
                    clean[key] = float(value)
                except ValueError:
                    clean[key] = value
            converted.append(clean)
        return converted

    def artifact_evidence(self, artifact_id: str) -> dict[str, Any]:
        """Return bounded detail from files already sealed into an artifact.

        This endpoint never evaluates a model or derives new scientific results.
        It only makes Oracle Builder's standard evidence inspectable by clients.
        """
        artifact = self.artifact(artifact_id)
        if artifact is None:
            raise KeyError(artifact_id)
        root = Path(artifact["path"])
        result = self._artifact_result(artifact)
        confusion: dict[str, Any] | None = None
        try:
            loaded = json.loads((root / "evaluation" / "confusion_matrix.json").read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("matrix"), list):
                confusion = loaded
        except (OSError, json.JSONDecodeError):
            pass

        per_class = self._evidence_csv(root / "evaluation" / "per_class_metrics.csv")
        top_confusions = self._evidence_csv(root / "evaluation" / "top_confusions.csv", limit=50)
        samples = self._evidence_csv(root / "evaluation" / "sample_metrics.csv", limit=500)
        if result.get("task") == "segmentation":
            samples.sort(key=lambda row: float(row.get("dice") or 0.0))

        media: list[dict[str, Any]] = []
        supported = {".png", ".jpg", ".jpeg", ".webp"}
        for directory in (root / "figures", root / "evaluation", root / "activations", root / "overlays"):
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in supported:
                    continue
                relative = path.relative_to(root).as_posix()
                label = path.stem.replace("_", " ").replace("-", " ").strip().title()
                lowered = relative.lower()
                kind = "activation" if "activat" in lowered or "saliency" in lowered or "gradcam" in lowered else (
                    "overlay" if "overlay" in lowered or "mask" in lowered or "prediction" in lowered else "figure"
                )
                media.append({
                    "name": label, "kind": kind, "path": relative,
                    "url": f"/api/v1/artifacts/{artifact_id}/evidence/files/{relative}",
                })

        return {
            "artifact": result,
            "classification": {
                "confusion_matrix": confusion,
                "per_class_metrics": per_class,
                "top_confusions": top_confusions,
            } if result.get("task") == "classification" else None,
            "segmentation": {
                "sample_metrics": samples,
                "worst_samples": samples[:25],
            } if result.get("task") == "segmentation" else None,
            "media": media,
            "availability": {
                "confusion_matrix": confusion is not None,
                "per_class_metrics": bool(per_class),
                "sample_metrics": bool(samples),
                "overlays": any(item["kind"] == "overlay" for item in media),
                "activations": any(item["kind"] == "activation" for item in media),
            },
        }

    def artifact_evidence_file(self, artifact_id: str, relative_path: str) -> Path:
        artifact = self.artifact(artifact_id)
        if artifact is None:
            raise KeyError(artifact_id)
        root = Path(artifact["path"]).resolve()
        requested = Path(relative_path)
        if requested.is_absolute():
            raise ValueError("Evidence paths must be relative")
        path = (root / requested).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Evidence path escapes the artifact") from exc
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _comparison_check(results: list[dict[str, Any]]) -> dict[str, Any]:
        reasons: list[str] = []
        if len(results) < 2:
            reasons.append("Select at least two indexed artifacts")
        for field, label in (("task", "task"), ("dataset_fingerprint_sha256", "dataset revision"), ("split", "evaluation split")):
            values = {result["protocol"].get(field) for result in results}
            if None in values:
                reasons.append(f"One or more artifacts do not record the {label}")
            elif len(values) > 1:
                reasons.append(f"Artifacts use different {label} values")
        for field, label in (("metric_schema", "metric schema"), ("decision_rule", "decision rule"), ("segmentation_target", "segmentation target")):
            values = {result["protocol"].get(field) for result in results if result["protocol"].get(field) is not None}
            if len(values) > 1:
                reasons.append(f"Artifacts use different {label} values")
        if any(not result.get("metrics") for result in results):
            reasons.append("One or more artifacts have no standard evaluation metrics")
        common_metrics = sorted(set.intersection(*(set(result["metrics"]) for result in results))) if results else []
        if results and not common_metrics:
            reasons.append("Artifacts have no evaluation metrics in common")
        return {
            "compatible": not reasons, "reasons": reasons, "common_metrics": common_metrics,
            "protocol": results[0]["protocol"] if results and not reasons else None,
        }

    def experiment_results(self, experiment_id: str) -> dict[str, Any]:
        experiment = self.experiment(experiment_id)
        if experiment is None:
            raise KeyError(experiment_id)
        specifications = self.specifications(experiment_id)
        candidates: list[dict[str, Any]] = []
        with self._connection() as db:
            for specification in specifications:
                job_row = db.execute("SELECT * FROM jobs WHERE specification_id=? ORDER BY submitted_at DESC LIMIT 1", (specification["specification_id"],)).fetchone()
                job = _row(job_row)
                artifact = self.artifact(specification["artifact_id"]) if specification.get("artifact_id") else None
                result = self._artifact_result(artifact) if artifact else None
                recipe_id = specification["parameters"].get("recipe_id")
                recipe = self.recipe(recipe_id) if recipe_id else None
                candidates.append({
                    "specification_id": specification["specification_id"], "name": specification["name"],
                    "ordinal": specification["ordinal"], "action": specification["action"],
                    "status": specification["status"], "seed": specification["parameters"].get("seed"),
                    "recipe_id": recipe_id, "recipe_name": recipe.get("name") if recipe else None,
                    "resources": specification["resources"], "job": job,
                    "runtime_seconds": self._duration_seconds(job.get("started_at") if job else None, job.get("completed_at") if job else None),
                    "artifact": result,
                })
        comparable = [candidate["artifact"] for candidate in candidates if candidate["artifact"]]
        return {
            "experiment": experiment, "dataset": self.dataset(experiment["dataset_id"]) if experiment.get("dataset_id") else None,
            "candidates": candidates, "comparison": self._comparison_check(comparable),
            "summary": {
                "total": len(candidates), "planned": sum(candidate["status"] == "planned" for candidate in candidates),
                "active": sum(candidate["status"] in {"dispatched", "queued", "running", "validating"} for candidate in candidates),
                "indexed": sum(candidate["artifact"] is not None for candidate in candidates),
                "failed": sum(candidate["status"] in {"failed", "dispatch_failed", "artifact_invalid", "cancelled"} for candidate in candidates),
            },
        }

    def create_comparison(self, *, name: str, artifact_ids: list[str], description: str = "") -> dict[str, Any]:
        unique_ids = list(dict.fromkeys(artifact_ids))
        artifacts = [self.artifact(artifact_id) for artifact_id in unique_ids]
        if any(artifact is None for artifact in artifacts):
            raise KeyError("One or more artifacts were not found")
        results = [self._artifact_result(artifact) for artifact in artifacts if artifact is not None]
        check = self._comparison_check(results)
        if not check["compatible"]:
            raise ValueError("Artifacts are not comparable: " + "; ".join(check["reasons"]))
        comparison_id, now = str(uuid.uuid4()), _now()
        selection = {"artifact_ids": unique_ids, "artifacts": results}
        protocol = {**check["protocol"], "common_metrics": check["common_metrics"]}
        with self._connection() as db:
            db.execute("INSERT INTO comparisons VALUES (?, ?, ?, ?, ?, ?, ?)", (comparison_id, name.strip() or "Model comparison", description, _json(selection), _json(protocol), now, now))
        return self.comparison(comparison_id)  # type: ignore[return-value]

    def _assign_output_path(self, action: str, parameters: dict[str, Any], specification_id: str) -> dict[str, Any]:
        """Assign paths owned by the orchestrator, never supplied by a UI client."""
        resolved = dict(parameters)
        if action == "train":
            runs_dir = self.artifact_root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            resolved["runs_dir"] = str(runs_dir)
            resolved["output"] = specification_id
        elif action == "evaluate":
            output = self.artifact_root / "evaluations" / specification_id
            output.parent.mkdir(parents=True, exist_ok=True)
            resolved["output"] = str(output)
        elif action == "model_ingest":
            output = self.artifact_root / "models" / specification_id
            output.parent.mkdir(parents=True, exist_ok=True)
            resolved["output"] = str(output)
        elif action == "run_pack":
            output = self.artifact_root / "packages" / f"{specification_id}.oracle-run.zip"
            output.parent.mkdir(parents=True, exist_ok=True)
            resolved["output"] = str(output)
        return resolved

    def dispatch(self, specification_id: str, endpoint_id: str) -> dict[str, Any]:
        check = self.preflight(specification_id, endpoint_id)
        if not check["ready"]:
            raise ValueError("Dispatch preflight failed: " + "; ".join(check["reasons"]))
        oracle_serve_url = check["endpoint"]["base_url"]
        with self._connection() as db:
            spec = self.specification(specification_id, connection=db)
            if spec is None:
                raise KeyError(specification_id)
            if spec["status"] != "planned":
                raise ValueError("Only planned run specifications can be dispatched")
            job_id, now = str(uuid.uuid4()), _now()
            request = {"job_id": job_id, "action": spec["action"], "parameters": spec["parameters"], "resources": spec["resources"]}
            db.execute("""INSERT INTO jobs (
                job_id, specification_id, oracle_serve_url, action, parameters_json,
                resources_json, status, remote_status, worker_id, error,
                submitted_at, updated_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (job_id, specification_id, oracle_serve_url.rstrip("/"), spec["action"], _json(spec["parameters"]), _json(spec["resources"]), "dispatching", None, None, None, now, now, None))
        try:
            response = self._request(oracle_serve_url, "POST", "/compute/jobs", request)
        except Exception as exc:
            with self._connection() as db:
                db.execute("UPDATE jobs SET status='dispatch_failed', error=?, updated_at=? WHERE job_id=?", (str(exc), _now(), job_id))
            raise
        with self._connection() as db:
            db.execute("UPDATE jobs SET status='submitted', remote_status=?, worker_id=?, updated_at=? WHERE job_id=?", (response.get("status"), response.get("worker_id"), _now(), job_id))
            db.execute("UPDATE run_specifications SET status='dispatched', updated_at=? WHERE specification_id=?", (_now(), specification_id))
        return self.job(job_id)  # type: ignore[return-value]

    def reconcile_job(self, job_id: str) -> dict[str, Any]:
        local = self.job(job_id)
        if local is None:
            raise KeyError(job_id)
        if local["status"] in {"indexed", "artifact_invalid", "failed", "cancelled"}:
            return local
        remote = self._request(local["oracle_serve_url"], "GET", f"/compute/jobs/{job_id}")
        try:
            self._capture_job_events(local)
        except RuntimeError:
            # Job state is authoritative for reconciliation. Event collection is
            # best-effort and can recover during a later poll.
            pass
        remote_status, now = remote["status"], _now()
        result = remote.get("result") or {}
        output_path = result.get("output_path") or self._expected_output_path(local)
        with self._connection() as db:
            local_status = "validating" if remote_status == "succeeded" and local["action"] in {"train", "model_ingest"} else remote_status
            db.execute("""UPDATE jobs SET status=?, remote_status=?, worker_id=?, error=?,
                updated_at=?, completed_at=?, started_at=?, output_path=? WHERE job_id=?""",
                (local_status, remote_status, remote.get("worker_id"), remote.get("error"), now,
                 remote.get("finished_at"), remote.get("started_at"), output_path, job_id))
            if remote_status in {"failed", "cancelled"}:
                db.execute("UPDATE run_specifications SET status=?, updated_at=? WHERE specification_id=?", (remote_status, now, local["specification_id"]))
            elif local_status == "validating":
                db.execute("UPDATE run_specifications SET status='validating', updated_at=? WHERE specification_id=?", (now, local["specification_id"]))
            elif remote_status == "succeeded":
                db.execute("UPDATE run_specifications SET status='succeeded', updated_at=? WHERE specification_id=?", (now, local["specification_id"]))
        if remote_status == "succeeded" and local["action"] in {"train", "model_ingest"}:
            self._record_event(job_id, "validating", "Compute completed; validating the produced artifact", {"output_path": output_path})
            try:
                report = self._index_completed_artifact({**local, "output_path": output_path})
            except Exception as exc:
                report = {"valid": False, "errors": [str(exc)], "warnings": [], "output_path": output_path}
            if report["valid"]:
                with self._connection() as db:
                    db.execute("UPDATE jobs SET status='indexed', validation_status='valid', validation_report_json=?, error=NULL, updated_at=? WHERE job_id=?", (_json(report), _now(), job_id))
                    db.execute("UPDATE run_specifications SET status='indexed', updated_at=? WHERE specification_id=?", (_now(), local["specification_id"]))
            else:
                error = "Completed output failed artifact validation: " + "; ".join(report.get("errors") or ["unknown validation error"])
                with self._connection() as db:
                    db.execute("UPDATE jobs SET status='artifact_invalid', validation_status='invalid', validation_report_json=?, error=?, updated_at=? WHERE job_id=?", (_json(report), error, _now(), job_id))
                    db.execute("UPDATE run_specifications SET status='artifact_invalid', updated_at=? WHERE specification_id=?", (_now(), local["specification_id"]))
                self._record_event(job_id, "artifact_invalid", error, {"validation": report})
        return self.job(job_id)  # type: ignore[return-value]

    def reconcile_active_jobs(self) -> list[dict[str, Any]]:
        active = [job for job in self.jobs() if job["status"] in {"dispatching", "submitted", "queued", "running", "validating"}]
        results = []
        for job in active:
            try:
                results.append(self.reconcile_job(job["job_id"]))
            except RuntimeError:
                results.append(job)
        return results

    def _capture_job_events(self, job: dict[str, Any]) -> None:
        with self._connection() as db:
            last = db.execute("SELECT COALESCE(MAX(sequence), 0) FROM job_events WHERE job_id=?", (job["job_id"],)).fetchone()[0]
        payload = self._request(job["oracle_serve_url"], "GET", f"/compute/jobs/{job['job_id']}/events?after={last}")
        with self._connection() as db:
            for event in payload.get("events", []):
                db.execute("INSERT OR IGNORE INTO job_events VALUES (?, ?, ?, ?, ?, ?)", (job["job_id"], event["sequence"], event["timestamp"], event["type"], event["message"], _json(event.get("data") or {})))

    @staticmethod
    def _expected_output_path(job: dict[str, Any]) -> str | None:
        params = job["parameters"]
        output = params.get("output")
        if not isinstance(output, str):
            return None
        return str(Path(params["runs_dir"]) / output) if job["action"] == "train" and isinstance(params.get("runs_dir"), str) else output

    def _index_completed_artifact(self, job: dict[str, Any]) -> dict[str, Any]:
        path_value = job.get("output_path") or self._expected_output_path(job)
        if not path_value:
            raise RuntimeError("Compute service did not report an artifact output path")
        path = Path(path_value)
        from oracle_builder.artifacts import validate_run_artifact
        report = {**validate_run_artifact(path), "output_path": str(path)}
        try:
            artifact_type = json.loads((path / "artifact.json").read_text(encoding="utf-8")).get("artifact_type", "model_run")
        except (OSError, json.JSONDecodeError) as exc:
            report["valid"] = False
            report.setdefault("errors", []).append(f"Artifact manifest could not be read: {exc}")
            artifact_type = None
        expected_type = "model_run" if job["action"] == "train" else "model_product"
        if report.get("valid") and artifact_type != expected_type:
            report["valid"] = False
            report.setdefault("errors", []).append(f"Expected {expected_type} output, received {artifact_type}")
        report["artifact_type"] = artifact_type
        if report.get("valid") and report.get("status") != "complete":
            report["valid"] = False
            report.setdefault("errors", []).append("Artifact status must be complete before indexing")
        if report.get("valid") and report.get("lifecycle") != "sealed":
            report["valid"] = False
            report.setdefault("errors", []).append("Artifact must be sealed before indexing")
        if not report.get("valid"):
            return report
        result = self.scan(path)
        artifact_id = report.get("artifact_id")
        if not artifact_id or artifact_id not in result["artifacts"]:
            report["valid"] = False
            report.setdefault("errors", []).append("Validated artifact could not be added to the catalog")
            return report
        with self._connection() as db:
            db.execute("UPDATE run_specifications SET artifact_id=?, updated_at=? WHERE specification_id=?", (artifact_id, _now(), job["specification_id"]))
        self._record_event(job["job_id"], "artifact_indexed", "Validated and indexed completed artifact", {"artifact_id": artifact_id, "path": str(path), "validation": report})
        return report

    def _record_event(self, job_id: str, event_type: str, message: str, data: dict[str, Any]) -> None:
        with self._connection() as db:
            sequence = db.execute("SELECT COALESCE(MAX(sequence), 0) + 1 FROM job_events WHERE job_id=?", (job_id,)).fetchone()[0]
            db.execute("INSERT INTO job_events VALUES (?, ?, ?, ?, ?, ?)", (job_id, sequence, _now(), event_type, message, _json(data)))

    @staticmethod
    def _request(base: str, method: str, endpoint: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        request = urllib.request.Request(base.rstrip("/") + endpoint, method=method)
        if body is not None:
            request.data = _json(body).encode()
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"oracle-serve returned {exc.code}: {exc.read().decode()}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not communicate with oracle-serve: {exc}") from exc

    def datasets(self) -> list[dict[str, Any]]: return self._many("SELECT * FROM datasets ORDER BY name")
    def artifacts(self) -> list[dict[str, Any]]: return self._many("SELECT * FROM artifacts ORDER BY updated_at DESC")
    def experiments(self) -> list[dict[str, Any]]: return self._many("SELECT * FROM experiments ORDER BY created_at DESC")
    def recipes(self) -> list[dict[str, Any]]: return self._many("SELECT * FROM recipes ORDER BY name")
    def specifications(self, experiment_id: str | None = None) -> list[dict[str, Any]]:
        if experiment_id:
            with self._connection() as db:
                return [_row(item) for item in db.execute("SELECT * FROM run_specifications WHERE experiment_id=? ORDER BY ordinal", (experiment_id,)).fetchall()]  # type: ignore[list-item]
        return self._many("SELECT * FROM run_specifications ORDER BY created_at DESC")
    def jobs(self) -> list[dict[str, Any]]:
        return self._many("""SELECT jobs.*, run_specifications.artifact_id,
            artifacts.name AS artifact_name, artifacts.path AS artifact_path,
            artifacts.status AS artifact_status, artifacts.lifecycle AS artifact_lifecycle
            FROM jobs
            LEFT JOIN run_specifications USING (specification_id)
            LEFT JOIN artifacts USING (artifact_id)
            ORDER BY jobs.submitted_at DESC""")
    def job_events(self, job_id: str) -> list[dict[str, Any]]:
        with self._connection() as db:
            rows = db.execute("SELECT * FROM job_events WHERE job_id=? ORDER BY sequence", (job_id,)).fetchall()
        return [{**dict(row), "data": json.loads(row["data_json"])} for row in rows]
    def comparisons(self) -> list[dict[str, Any]]: return self._many("SELECT * FROM comparisons ORDER BY created_at DESC")
    def compute_endpoints(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        endpoints = self._many("SELECT * FROM compute_endpoints WHERE enabled=1 ORDER BY name")
        return [self.refresh_compute_endpoint(item["endpoint_id"]) for item in endpoints] if refresh else endpoints
    def _many(self, query: str) -> list[dict[str, Any]]:
        with self._connection() as db: return [_row(item) for item in db.execute(query).fetchall()]  # type: ignore[list-item]
    def dataset(self, value: str, *, connection: sqlite3.Connection | None = None) -> dict[str, Any] | None:
        if connection is not None: return _row(connection.execute("SELECT * FROM datasets WHERE dataset_id=?", (value,)).fetchone())
        with self._connection() as db: return _row(db.execute("SELECT * FROM datasets WHERE dataset_id=?", (value,)).fetchone())
    def artifact(self, value: str) -> dict[str, Any] | None:
        with self._connection() as db: return _row(db.execute("SELECT * FROM artifacts WHERE artifact_id=?", (value,)).fetchone())
    def recipe(self, value: str) -> dict[str, Any] | None:
        with self._connection() as db: return _row(db.execute("SELECT * FROM recipes WHERE recipe_id=?", (value,)).fetchone())
    def experiment(self, value: str) -> dict[str, Any] | None:
        with self._connection() as db: return _row(db.execute("SELECT * FROM experiments WHERE experiment_id=?", (value,)).fetchone())
    def specification(self, value: str, *, connection: sqlite3.Connection | None = None) -> dict[str, Any] | None:
        if connection is not None: return _row(connection.execute("SELECT * FROM run_specifications WHERE specification_id=?", (value,)).fetchone())
        with self._connection() as db: return _row(db.execute("SELECT * FROM run_specifications WHERE specification_id=?", (value,)).fetchone())
    def job(self, value: str) -> dict[str, Any] | None:
        with self._connection() as db:
            return _row(db.execute("""SELECT jobs.*, run_specifications.artifact_id,
                artifacts.name AS artifact_name, artifacts.path AS artifact_path,
                artifacts.status AS artifact_status, artifacts.lifecycle AS artifact_lifecycle
                FROM jobs
                LEFT JOIN run_specifications USING (specification_id)
                LEFT JOIN artifacts USING (artifact_id)
                WHERE jobs.job_id=?""", (value,)).fetchone())
    def compute_endpoint(self, value: str) -> dict[str, Any] | None:
        with self._connection() as db:
            return _row(db.execute("SELECT * FROM compute_endpoints WHERE endpoint_id=?", (value,)).fetchone())
    def comparison(self, value: str) -> dict[str, Any] | None:
        with self._connection() as db:
            return _row(db.execute("SELECT * FROM comparisons WHERE comparison_id=?", (value,)).fetchone())
