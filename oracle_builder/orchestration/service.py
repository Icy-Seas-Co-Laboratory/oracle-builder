from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sqlite3
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from oracle_data_contracts.datasets import dataset_fingerprint, read_dataset_info, set_dataset_lifecycle
from oracle_builder.data.decoders import decode_blob
from oracle_builder.config import (
    DEFAULT_CONFIG,
    deep_merge,
    load_toml,
    normalize_component_config,
    resolve_v2_config,
    validate_config,
    validate_v2_config,
)
from oracle_builder.config_schema import configuration_schema as v2_configuration_schema
from oracle_builder.orchestration.database import connect


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    for key in ("metadata_json", "manifest_json", "plan_json", "parameters_json", "resources_json", "selection_json", "protocol_json", "summary_json", "validation_report_json", "readiness_json", "workers_json", "queue_json", "facts_json", "config_json", "layout_json", "initialization_json", "preflight_report_json"):
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
        runs_root: str | Path | None = None,
        datasets_root: str | Path | None = None,
        browse_roots: list[str | Path] | None = None,
        training_catalog_roots: list[str | Path] | None = None,
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
        # The control database and upload staging area are runtime concerns;
        # ordinary Oracle Builder science products stay in the familiar project
        # directories unless an operator chooses different roots explicitly.
        self.runs_root = Path(runs_root).expanduser().resolve() if runs_root else self.workspace_root / "runs"
        self.datasets_root = Path(datasets_root).expanduser().resolve() if datasets_root else self.workspace_root / "datasets"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.datasets_root.mkdir(parents=True, exist_ok=True)
        self.browse_roots: dict[str, Path] = {"workspace": self.workspace_root, "artifacts": self.artifact_root}
        for index, root in enumerate(browse_roots or [], start=1):
            location = Path(root).expanduser().resolve()
            if not location.is_dir():
                raise NotADirectoryError(location)
            self.browse_roots[f"root-{index}"] = location
        for root_id, location in (("runs", self.runs_root), ("datasets", self.datasets_root)):
            if not any(location.is_relative_to(allowed) for allowed in self.browse_roots.values()):
                self.browse_roots[root_id] = location
        configured_catalog_roots = [self.datasets_root, *(Path(root).expanduser().resolve() for root in training_catalog_roots or [])]
        self.training_catalog_roots: dict[str, Path] = {}
        for root in configured_catalog_roots:
            location = Path(root).expanduser().resolve()
            if not location.is_dir():
                raise NotADirectoryError(location)
            if location not in self.training_catalog_roots.values():
                self.training_catalog_roots[f"catalog-{len(self.training_catalog_roots) + 1}"] = location
            if not any(location.is_relative_to(allowed) for allowed in self.browse_roots.values()):
                self.browse_roots[f"catalog-{len(self.browse_roots) + 1}"] = location
        with connect(self.database):
            pass
        for name, base_url in compute_endpoints or []:
            self.register_compute_endpoint(name=name, base_url=base_url)

    def _connection(self) -> sqlite3.Connection:
        return connect(self.database)

    @staticmethod
    def _architecture_config_path(architecture: str) -> Path:
        """Return the explicit V2 starter recipe for a supported architecture.

        Family names remain useful shorthand in the API, but resolve to a
        concrete supported variant.  This deliberately avoids treating the
        old EfficientNet V1 recipe as an EfficientNetV2 baseline.
        """
        root = Path(__file__).resolve().parents[2] / "configs"
        classification = root / "classification_defaults"
        paths = {
            "simple_cnn": classification / "simple_cnn.toml",
            "resnet_like": classification / "resnet_like.toml",
            "densenet_like": classification / "densenet_like.toml",
            "resnet": classification / "resnet18.toml",
            "resnet18": classification / "resnet18.toml",
            "resnet34": classification / "resnet34.toml",
            "resnet50": classification / "resnet50.toml",
            "resnet101": classification / "resnet101.toml",
            "resnet152": classification / "resnet152.toml",
            "densenet": classification / "densenet121.toml",
            "densenet121": classification / "densenet121.toml",
            "densenet169": classification / "densenet169.toml",
            "densenet201": classification / "densenet201.toml",
            "efficientnet": classification / "efficientnet_b0.toml",
            "efficientnet_b0": classification / "efficientnet_b0.toml",
            "efficientnet_b1": classification / "efficientnet_b1.toml",
            "efficientnet_b2": classification / "efficientnet_b2.toml",
            "efficientnet_b3": classification / "efficientnet_b3.toml",
            "efficientnet_b4": classification / "efficientnet_b4.toml",
            "efficientnet_b5": classification / "efficientnet_b5.toml",
            "efficientnet_b6": classification / "efficientnet_b6.toml",
            "efficientnet_b7": classification / "efficientnet_b7.toml",
            "efficientnet_v2": classification / "efficientnet_v2_b0.toml",
            "efficientnet_v2_b0": classification / "efficientnet_v2_b0.toml",
            "efficientnet_v2_b1": classification / "efficientnet_v2_b1.toml",
            "efficientnet_v2_b2": classification / "efficientnet_v2_b2.toml",
            "efficientnet_v2_b3": classification / "efficientnet_v2_b3.toml",
            "efficientnet_v2_s": classification / "efficientnet_v2_s.toml",
            "efficientnet_v2_m": classification / "efficientnet_v2_m.toml",
            "efficientnet_v2_l": classification / "efficientnet_v2_l.toml",
            "efficientnetv2": classification / "efficientnet_v2_b0.toml",
            "efficientnetv2_b0": classification / "efficientnet_v2_b0.toml",
            "efficientnetv2_b1": classification / "efficientnet_v2_b1.toml",
            "efficientnetv2_b2": classification / "efficientnet_v2_b2.toml",
            "efficientnetv2_b3": classification / "efficientnet_v2_b3.toml",
            "efficientnetv2_s": classification / "efficientnet_v2_s.toml",
            "efficientnetv2_m": classification / "efficientnet_v2_m.toml",
            "efficientnetv2_l": classification / "efficientnet_v2_l.toml",
            "convnext": classification / "convnext_tiny.toml",
            "convnext_tiny": classification / "convnext_tiny.toml",
            "convnext_small": classification / "convnext_small.toml",
            "mobilenet": classification / "mobilenet_v3_small.toml",
            "mobilenet_v3_small": classification / "mobilenet_v3_small.toml",
            "mobilenet_v3_large": classification / "mobilenet_v3_large.toml",
            "unet": root / "example_segmentation_unet.toml",
            "residual_unet": root / "example_segmentation_residual_unet.toml",
            "resunet": root / "example_segmentation_residual_unet.toml",
            "unet_plus_plus": root / "example_segmentation_unet_plus_plus.toml",
            "unetpp": root / "example_segmentation_unet_plus_plus.toml",
        }
        try:
            return paths[architecture]
        except KeyError as exc:
            raise ValueError(f"Unsupported architecture: {architecture}") from exc

    def model_setup(self, architecture: str) -> dict[str, Any]:
        """Expose the maintained TOML defaults as a safe UI setup schema."""
        source = load_toml(self._architecture_config_path(architecture))
        source.setdefault("run", {})["model"] = architecture
        config = deep_merge(DEFAULT_CONFIG, source)
        # model_setup is a runtime-preview helper, not the V2 authoring API.
        # Translate its temporary alias so existing builders and previews see
        # the same internal structure they will receive at execution.
        normalize_component_config(config, source)
        fields: list[dict[str, Any]] = []

        def collect(value: dict[str, Any], prefix: str) -> None:
            for key, item in value.items():
                path = f"{prefix}.{key}"
                if isinstance(item, (str, int, float, bool)):
                    fields.append({"path": path, "value": item, "type": "boolean" if isinstance(item, bool) else "number" if isinstance(item, (int, float)) else "text"})
                elif isinstance(item, list):
                    fields.append({"path": path, "value": item, "type": "list"})

        # These are the knobs users can safely understand before later advanced
        # configuration support. They still come directly from each TOML.
        for section in ("data", "model", "training"):
            collect(config.get(section, {}), section)
        return {"architecture": architecture, "task": config["run"].get("task"), "config": config, "fields": fields}

    @staticmethod
    def configuration_schema() -> dict[str, Any]:
        """Return the complete V2 contract for TOML, GUI, and planning clients."""
        return v2_configuration_schema()

    # Model definitions -------------------------------------------------
    # Maintained TOMLs are deliberately read-only sources.  A user must make
    # a definition (or duplicate an existing definition) before changing any
    # scientific setting, which gives queueing a stable revision to pin.
    @property
    def _model_template_root(self) -> Path:
        # Templates ship with Oracle Builder rather than with an arbitrary
        # project workspace.  Definitions persist the template digest, so a
        # later package/template change cannot rewrite prior science.
        return Path(__file__).resolve().parents[2] / "configs" / "classification_defaults"

    @property
    def _first_class_template_root(self) -> Path:
        """Maintained V2 templates whose task is part of the product surface.

        Classification family defaults predate model definitions and retain
        their stable short identifiers (for example ``resnet18``).  New
        first-class workflows live separately so their template names describe
        the scientific task rather than being confused with a family preset.
        """
        return Path(__file__).resolve().parents[2] / "configs" / "model_definition_templates"

    @staticmethod
    def _config_digest(config: dict[str, Any]) -> str:
        return hashlib.sha256(_json(config).encode("utf-8")).hexdigest()

    def _definition_catalog_fingerprint(self) -> str | None:
        # Schema work evolves independently of persistence.  A definition is
        # still valid without a catalog fingerprint; once the catalog is
        # available its immutable fingerprint is recorded on new revisions.
        try:
            schema = self.configuration_schema()
        except (TypeError, ValueError):
            return None
        return schema.get("fingerprint") or schema.get("schema_fingerprint")

    def _definition_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Resolve and enforce the only user-authorable configuration dialect."""
        # Definition records are public V2 documents.  Runtime-only aliases
        # (run.model/model/pretraining) are created only by resolve_config at
        # execution time and never leak back through this API.
        validate_v2_config(config)
        return resolve_v2_config(config)

    def _template_path(self, template_id: str) -> Path:
        if not isinstance(template_id, str) or not template_id or "/" in template_id or "\\" in template_id:
            raise ValueError("Invalid model-definition template ID")
        for item_id, candidate in self._template_sources():
            if item_id == template_id:
                return candidate
        raise KeyError(template_id)

    def _template_sources(self) -> list[tuple[str, Path]]:
        classification = [
            (path.stem, path) for path in sorted(self._model_template_root.glob("*.toml"))
        ]
        first_class = [
            (path.stem, path)
            for path in sorted(self._first_class_template_root.glob("*.toml"))
        ]
        # Only templates which already satisfy the V2 definition contract are
        # offered here.  Historical examples remain runnable through their CLI
        # paths, but cannot silently become authoring templates.
        sources = classification + first_class
        ids = [template_id for template_id, _ in sources]
        if len(ids) != len(set(ids)):
            raise ValueError("Model-definition template IDs must be unique")
        return [
            (template_id, path.resolve())
            for template_id, path in sources
            if path.is_file()
        ]

    def model_definition_templates(self) -> list[dict[str, Any]]:
        templates: list[dict[str, Any]] = []
        for template_id, path in self._template_sources():
            raw = path.read_bytes()
            source = load_toml(path)
            try:
                config = self._definition_config(source)
            except (TypeError, ValueError):
                # A maintained file that is not V2 must never leak into the
                # authoring catalog.  It remains visible on disk for migration.
                continue
            templates.append({
                "template_id": template_id,
                "name": template_id.replace("_", " "),
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "task": (config.get("run") or {}).get("task"),
                "architecture": {"family": (config.get("encoder") or {}).get("family"), "variant": (config.get("encoder") or {}).get("variant")},
                "config": config,
            })
        return templates

    def model_definition_template(self, template_id: str) -> dict[str, Any]:
        for template in self.model_definition_templates():
            if template["template_id"] == template_id:
                return template
        raise KeyError(template_id)

    def _definition_result(self, definition: dict[str, Any]) -> dict[str, Any]:
        definition["architecture"] = self.architecture_view_from_config(definition["config"])
        return definition

    def model_definitions(self) -> list[dict[str, Any]]:
        return [self._definition_result(item) for item in self._many("SELECT * FROM model_definitions ORDER BY updated_at DESC")]

    def model_definition(self, definition_id: str, *, revision: int | None = None) -> dict[str, Any] | None:
        with self._connection() as db:
            if revision is None:
                item = _row(db.execute("SELECT * FROM model_definitions WHERE definition_id=?", (definition_id,)).fetchone())
            else:
                item = _row(db.execute("""SELECT d.definition_id,d.name,d.description,r.revision,d.template_id,d.template_sha256,
                    d.parent_definition_id,d.parent_revision,d.lineage_kind,r.config_json,r.config_sha256,r.catalog_fingerprint,
                    d.created_at,r.created_at AS updated_at FROM model_definitions d JOIN model_definition_revisions r
                    ON d.definition_id=r.definition_id WHERE d.definition_id=? AND r.revision=?""", (definition_id, int(revision))).fetchone())
        return self._definition_result(item) if item is not None else None

    def model_definition_revisions(self, definition_id: str) -> list[dict[str, Any]]:
        if self.model_definition(definition_id) is None:
            raise KeyError(definition_id)
        with self._connection() as db:
            rows = db.execute("SELECT definition_id,revision,config_json,config_sha256,catalog_fingerprint,created_at FROM model_definition_revisions WHERE definition_id=? ORDER BY revision DESC", (definition_id,)).fetchall()
        return [_row(row) for row in rows]  # type: ignore[list-item]

    def create_model_definition(self, *, name: str, template_id: str, description: str = "", config: dict[str, Any] | None = None) -> dict[str, Any]:
        template = self.model_definition_template(template_id)
        if config is not None and not isinstance(config, dict):
            raise ValueError("Model definition config must be an object")
        resolved = self._definition_config(deep_merge(template["config"], config or {}))
        definition_id, now = str(uuid.uuid4()), _now()
        digest, fingerprint = self._config_digest(resolved), self._definition_catalog_fingerprint()
        with self._connection() as db:
            definition_name = name.strip() or template["name"]
            try:
                db.execute("""INSERT INTO model_definitions VALUES (?, ?, ?, 1, ?, ?, NULL, NULL, 'template', ?, ?, ?, ?, ?)""",
                           (definition_id, definition_name, description, template_id, template["sha256"], _json(resolved), digest, fingerprint, now, now))
            except sqlite3.IntegrityError as exc:
                if "model_definitions_name_idx" in str(exc) or "model_definitions.name" in str(exc):
                    raise ValueError(f"Model definition name already exists: {definition_name}") from exc
                raise
            db.execute("INSERT INTO model_definition_revisions VALUES (?, 1, ?, ?, ?, ?)", (definition_id, _json(resolved), digest, fingerprint, now))
        return self.model_definition(definition_id)  # type: ignore[return-value]

    def update_model_definition(self, definition_id: str, *, expected_revision: int, name: str | None = None, description: str | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
        current = self.model_definition(definition_id)
        if current is None: raise KeyError(definition_id)
        if int(current["revision"]) != int(expected_revision):
            raise ValueError("Model definition revision conflict; refresh before saving")
        if config is not None and not isinstance(config, dict): raise ValueError("Model definition config must be an object")
        resolved = self._definition_config(config if config is not None else current["config"])
        revision, now = int(current["revision"]) + 1, _now()
        digest, fingerprint = self._config_digest(resolved), self._definition_catalog_fingerprint()
        with self._connection() as db:
            definition_name = name.strip() if isinstance(name, str) and name.strip() else current["name"]
            try:
                updated = db.execute("""UPDATE model_definitions SET name=?,description=?,revision=?,config_json=?,config_sha256=?,catalog_fingerprint=?,updated_at=?
                    WHERE definition_id=? AND revision=?""", (definition_name, description if description is not None else current["description"], revision, _json(resolved), digest, fingerprint, now, definition_id, expected_revision)).rowcount
            except sqlite3.IntegrityError as exc:
                if "model_definitions_name_idx" in str(exc) or "model_definitions.name" in str(exc):
                    raise ValueError(f"Model definition name already exists: {definition_name}") from exc
                raise
            if updated != 1: raise ValueError("Model definition revision conflict; refresh before saving")
            db.execute("INSERT INTO model_definition_revisions VALUES (?, ?, ?, ?, ?, ?)", (definition_id, revision, _json(resolved), digest, fingerprint, now))
        return self.model_definition(definition_id)  # type: ignore[return-value]

    def duplicate_model_definition(self, definition_id: str, *, revision: int | None = None, name: str | None = None, lineage_kind: str = "duplicate") -> dict[str, Any]:
        source = self.model_definition(definition_id, revision=revision)
        if source is None: raise KeyError(definition_id)
        if lineage_kind not in {"duplicate", "sensitivity", "fork"}: raise ValueError("Unsupported definition lineage kind")
        new_id, now = str(uuid.uuid4()), _now()
        digest, fingerprint = self._config_digest(source["config"]), self._definition_catalog_fingerprint()
        requested_name = name.strip() if isinstance(name, str) and name.strip() else None
        with self._connection() as db:
            # Naming has to be decided inside a write transaction.  Otherwise
            # two users duplicating the same definition could both observe
            # "V.2" as available and create indistinguishable library rows.
            db.execute("BEGIN IMMEDIATE")
            duplicate_name = requested_name or self._next_definition_duplicate_name(db, source["name"])
            try:
                db.execute("""INSERT INTO model_definitions VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                           (new_id, duplicate_name, source["description"], source.get("template_id"), source.get("template_sha256"), definition_id, source["revision"], lineage_kind, _json(source["config"]), digest, fingerprint, now, now))
                db.execute("INSERT INTO model_definition_revisions VALUES (?, 1, ?, ?, ?, ?)", (new_id, _json(source["config"]), digest, fingerprint, now))
            except sqlite3.IntegrityError as exc:
                if "model_definitions_name_idx" in str(exc) or "model_definitions.name" in str(exc):
                    raise ValueError(f"Model definition name already exists: {duplicate_name}") from exc
                raise
        return self.model_definition(new_id)  # type: ignore[return-value]

    @staticmethod
    def _next_definition_duplicate_name(db: sqlite3.Connection, source_name: str) -> str:
        """Choose the next free ``base V.N`` name for an automatic duplicate.

        A copy of either ``baseline`` or ``baseline V.2`` belongs to the same
        sequence.  We start at V.2 for a base definition and immediately after
        a versioned source, then fill the next available number.  The caller
        holds a write transaction, and the unique name index is the final
        guard against collisions from another process.
        """
        match = re.fullmatch(r"(?P<base>.+?)\s+[Vv]\.(?P<version>[1-9]\d*)", source_name.strip())
        base = match.group("base").strip() if match else source_name.strip()
        version = int(match.group("version")) + 1 if match else 2
        while True:
            candidate = f"{base} V.{version}"
            exists = db.execute(
                "SELECT 1 FROM model_definitions WHERE name = ? COLLATE NOCASE LIMIT 1",
                (candidate,),
            ).fetchone()
            if exists is None:
                return candidate
            version += 1

    # Validated queue ---------------------------------------------------
    # Definitions stay editable and versioned.  A queued run instead owns a
    # sealed TOML and dataset pairing, so later edits cannot change queued
    # science or execution behaviour.
    @staticmethod
    def _toml_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: Orchestrator._toml_safe(item) for key, item in value.items() if item is not None}
        if isinstance(value, list):
            return [Orchestrator._toml_safe(item) for item in value if item is not None]
        return value

    def _definition_queue_config(self, definition: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
        """Bind trusted dataset facts before sealing a definition revision."""
        config = deep_merge(definition["config"], {})
        dataset_facts: dict[str, Any] = {}
        task = str((config.get("run") or {}).get("task", "classification"))
        if task in {"classification", "embedding"}:
            try:
                with sqlite3.connect(dataset["path"]) as dataset_db:
                    count = int(dataset_db.execute("SELECT count(*) FROM classification_labels").fetchone()[0])
            except sqlite3.DatabaseError as exc:
                raise ValueError("Could not read class labels from the frozen dataset") from exc
            if count < 1:
                raise ValueError("Frozen classification dataset has no class labels")
            dataset_facts["num_classes"] = count
        # Keep the sealed TOML V2-only.  The training runtime translates this
        # isolated input to internal builder aliases while resolving it.
        return resolve_v2_config(config, dataset_facts=dataset_facts)

    def queued_runs(self) -> list[dict[str, Any]]:
        rows = self._many("SELECT * FROM queued_runs ORDER BY priority DESC, created_at")
        for row in rows:
            row["start_authorized"] = bool(row.get("start_authorized"))
            definition = self.model_definition(str(row["definition_id"]), revision=int(row["definition_revision"]))
            row["definition_name"] = definition.get("name") if definition else None
            try:
                row["batch_size"] = int(load_toml(row["resolved_toml_path"]).get("data", {}).get("batch_size"))
            except (OSError, TypeError, ValueError):
                row["batch_size"] = None
        return rows

    def queued_run(self, queued_run_id: str) -> dict[str, Any] | None:
        with self._connection() as db:
            row = _row(db.execute("SELECT * FROM queued_runs WHERE queued_run_id=?", (queued_run_id,)).fetchone())
        if row is not None:
            row["start_authorized"] = bool(row.get("start_authorized"))
            try:
                row["batch_size"] = int(load_toml(row["resolved_toml_path"]).get("data", {}).get("batch_size"))
            except (OSError, TypeError, ValueError):
                row["batch_size"] = None
        return row

    def validate_and_queue_model_definition(
        self,
        definition_id: str,
        *,
        name: str,
        dataset_id: str,
        endpoint_id: str,
        revision: int | None = None,
        description: str = "",
        resources: dict[str, Any] | None = None,
        initialization: dict[str, Any] | None = None,
        batch_size_mode: str = "manual",
        batch_size: int | None = None,
        maximum_batch_size: int = 256,
    ) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("A queued run requires a name")
        definition = self.model_definition(definition_id, revision=revision)
        dataset = self.dataset(dataset_id)
        if definition is None:
            raise KeyError("Model definition was not found")
        if dataset is None:
            raise KeyError("Dataset was not found")
        if dataset.get("lifecycle") != "frozen":
            raise ValueError("Queued training requires a frozen registered dataset")
        endpoint = self.compute_endpoint(endpoint_id)
        if endpoint is None:
            raise KeyError("Compute endpoint was not found")
        requested_resources = dict(resources or {})
        raw_gpu_ids = requested_resources.get("gpu_ids")
        if raw_gpu_ids is not None:
            if not isinstance(raw_gpu_ids, list) or any(not isinstance(item, (str, int)) or not str(item).strip() for item in raw_gpu_ids):
                raise ValueError("GPU allocation must be a list of non-empty GPU IDs")
            gpu_ids = [str(item).strip() for item in raw_gpu_ids]
            if len(gpu_ids) != len(set(gpu_ids)):
                raise ValueError("GPU allocation cannot contain the same GPU more than once")
            if len(gpu_ids) > 1:
                raise ValueError("Validated Queue currently allocates one GPU per training run")
            gpu_count = requested_resources.get("gpu_count", len(gpu_ids))
            if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count != len(gpu_ids):
                raise ValueError("gpu_count must equal the number of selected gpu_ids")
            # Preserve an explicit empty allocation: it is an intentional CPU
            # run, not an omitted legacy GPU request.
            requested_resources["gpu_ids"] = gpu_ids
            requested_resources["gpu_count"] = gpu_count
        else:
            gpu_count = requested_resources.get("gpu_count", 0)
        if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 0:
            raise ValueError("GPU request must be a non-negative integer")
        if batch_size_mode not in {"manual", "auto"}:
            raise ValueError("batch_size_mode must be 'manual' or 'auto'")
        if batch_size is not None and (isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")
        if isinstance(maximum_batch_size, bool) or not isinstance(maximum_batch_size, int) or maximum_batch_size < 1:
            raise ValueError("maximum_batch_size must be a positive integer")
        if batch_size_mode == "auto" and batch_size is not None:
            raise ValueError("batch_size is only valid with manual batch_size_mode")
        queue_id, experiment_id, specification_id, now = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4()), _now()
        config = self._definition_queue_config(definition, dataset)
        # Queue allocation owns physical-device choice.  The runtime sees just
        # that selected device via CUDA_VISIBLE_DEVICES, so use one logical
        # device rather than allowing the legacy automatic selector to choose
        # another host GPU.
        if "gpu_ids" in requested_resources:
            distribution = config.setdefault("distribution", {})
            distribution["strategy"] = "single" if requested_resources["gpu_ids"] else "cpu"
            distribution["devices"] = []
        if batch_size_mode == "manual":
            resolved_batch_size = batch_size if batch_size is not None else int(config.get("data", {}).get("batch_size", 16))
            config.setdefault("data", {})["batch_size"] = resolved_batch_size
            batch_execution: dict[str, Any] = {"mode": "manual", "batch_size": resolved_batch_size}
        else:
            batch_execution = {"mode": "auto", "maximum_batch_size": maximum_batch_size, "safety_factor": 0.8}
        queue_dir = self.artifact_root / "queued-runs" / queue_id
        queue_dir.mkdir(parents=True, exist_ok=False)
        config_path = queue_dir / "resolved.toml"
        import tomli_w
        config_path.write_text(tomli_w.dumps(self._toml_safe(config)), encoding="utf-8")
        config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        parameters = self._assign_output_path(
            "train",
            {
                "config": str(config_path), "input": dataset["path"], "dataset_id": dataset_id,
                "definition_id": definition_id, "definition_revision": int(definition["revision"]),
                "initialization": dict(initialization or {}),
                "queue_execution": batch_execution,
            },
            specification_id,
        )
        plan = {
            "kind": "validated_queue", "queued_run_id": queue_id,
            "definition_id": definition_id, "definition_revision": int(definition["revision"]),
            "dataset_id": dataset_id, "dataset_fingerprint_sha256": dataset.get("fingerprint_sha256"),
            "batch_execution": batch_execution,
        }
        with self._connection() as db:
            db.execute("INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                experiment_id, None, name.strip(), description, dataset_id, "queued", _json(plan), now, now,
            ))
            db.execute("INSERT INTO run_specifications VALUES (?, ?, 1, ?, 'train', ?, ?, ?, 'planned', NULL, ?, ?)", (
                specification_id, experiment_id, name.strip(), _json(parameters), _json(requested_resources),
                hashlib.sha256(_json({"parameters": parameters, "config_sha256": config_digest}).encode()).hexdigest(), now, now,
            ))
        # Both the orchestrator and compute host inspect the immutable inputs.
        # A host-side preflight is advisory when telemetry is unavailable; its
        # provenance stays attached to the queue row and is rechecked at launch.
        report = self.preflight(specification_id, endpoint_id)
        if report.get("ready") and batch_size_mode == "auto":
            capable_workers = report.get("capable_workers") or []
            idle_workers = [worker for worker in capable_workers if worker.get("status") == "idle"]
            if not idle_workers:
                states = sorted({str(worker.get("status") or "unknown") for worker in capable_workers})
                report["ready"] = False
                report.setdefault("reasons", []).append(
                    "Batch-size auto-tuning requires an idle compute worker"
                    + (f" (currently: {', '.join(states)})" if states else "")
                    + ". Wait for the current training or calibration to finish, then retry."
                )
        if report.get("ready") and batch_size_mode == "auto":
            try:
                tune_report = self._request(endpoint["base_url"], "POST", "/compute/batch-size-tune", {
                    "parameters": {
                        "config": str(config_path), "input": dataset["path"],
                        "minimum_batch_size": 1, "maximum_batch_size": maximum_batch_size,
                        "safety_factor": 0.8,
                    },
                    "resources": requested_resources,
                }, timeout_seconds=930)
                report["batch_tune"] = tune_report
                if tune_report.get("ready") and isinstance(tune_report.get("recommended_batch_size"), int):
                    resolved_batch_size = int(tune_report["recommended_batch_size"])
                    config.setdefault("data", {})["batch_size"] = resolved_batch_size
                    batch_execution = {
                        "mode": "auto", "batch_size": resolved_batch_size,
                        "maximum_batch_size": maximum_batch_size,
                        "probe_kind": tune_report.get("probe_kind"),
                        "largest_verified_batch_size": tune_report.get("largest_verified_batch_size"),
                        "safety_factor": tune_report.get("safety_factor"),
                    }
                    parameters["queue_execution"] = batch_execution
                    plan["batch_execution"] = batch_execution
                    import tomli_w
                    config_path.write_text(tomli_w.dumps(self._toml_safe(config)), encoding="utf-8")
                    config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
                    with self._connection() as db:
                        db.execute(
                            "UPDATE run_specifications SET parameters_json=?, config_hash=?, updated_at=? WHERE specification_id=?",
                            (
                                _json(parameters),
                                hashlib.sha256(_json({"parameters": parameters, "config_sha256": config_digest}).encode()).hexdigest(),
                                _now(), specification_id,
                            ),
                        )
                        db.execute(
                            "UPDATE experiments SET plan_json=?, updated_at=? WHERE experiment_id=?",
                            (_json(plan), _now(), experiment_id),
                        )
                else:
                    report["ready"] = False
                    report.setdefault("reasons", []).extend(tune_report.get("reasons") or ["Batch-size auto-tuning did not produce a safe batch size"])
            except RuntimeError as exc:
                report["ready"] = False
                report.setdefault("reasons", []).append(f"Batch-size auto-tuning unavailable: {exc}")
        if report.get("ready"):
            try:
                compute_report = self._request(endpoint["base_url"], "POST", "/compute/preflight", {
                    "action": "train", "parameters": parameters, "resources": requested_resources,
                })
                report["compute_preflight"] = compute_report
                if not compute_report.get("ready"):
                    report["ready"] = False
                    report.setdefault("reasons", []).extend(compute_report.get("reasons") or ["Compute preflight failed"])
            except RuntimeError as exc:
                report["ready"] = False
                report.setdefault("reasons", []).append(f"Compute preflight unavailable: {exc}")
        status = "ready" if report.get("ready") else "needs_attention"
        schema_fingerprint = self._definition_catalog_fingerprint()
        with self._connection() as db:
            db.execute("""INSERT INTO queued_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                queue_id, definition_id, int(definition["revision"]), dataset_id, dataset.get("fingerprint_sha256"),
                name.strip(), description, specification_id, str(config_path), config_digest, schema_fingerprint,
                _json(requested_resources), _json(initialization or {}), endpoint_id, "valid" if report.get("ready") else "invalid",
                _json(report), status, 0, 0, "; ".join(report.get("reasons") or []) or None, now, now,
            ))
        return self.queued_run(queue_id)  # type: ignore[return-value]

    def authorize_queued_runs(
        self, *, queued_run_ids: list[str] | None, endpoint_id: str, all_ready: bool = False
    ) -> dict[str, Any]:
        endpoint = self.compute_endpoint(endpoint_id)
        if endpoint is None:
            raise KeyError("Compute endpoint was not found")
        if all_ready:
            selected = [
                row["queued_run_id"] for row in self.queued_runs()
                if row["status"] == "ready" and row.get("preflight_endpoint_id") == endpoint_id
            ]
        else:
            selected = list(dict.fromkeys(str(item) for item in (queued_run_ids or []) if item))
        if not selected:
            raise ValueError("Select at least one ready queued run")
        placeholders = ",".join("?" for _ in selected)
        with self._connection() as db:
            rows = db.execute(f"SELECT queued_run_id,status,preflight_endpoint_id FROM queued_runs WHERE queued_run_id IN ({placeholders})", selected).fetchall()
            if len(rows) != len(selected):
                raise KeyError("One or more queued runs were not found")
            blocked = [
                row["queued_run_id"] for row in rows
                if row["status"] != "ready" or row["preflight_endpoint_id"] != endpoint_id
            ]
            if blocked:
                raise ValueError("Only ready runs validated for this compute endpoint can be started")
            db.execute(f"UPDATE queued_runs SET start_authorized=1, updated_at=? WHERE queued_run_id IN ({placeholders})", [_now(), *selected])
        dispatched = self.schedule_queued_runs(endpoint_id)
        return {"authorized_ids": selected, "dispatched": dispatched, "queued_runs": [self.queued_run(item) for item in selected]}

    def schedule_queued_runs(self, endpoint_id: str) -> list[dict[str, Any]]:
        """Dispatch at most one training run per endpoint in the initial policy."""
        endpoint = self.compute_endpoint(endpoint_id)
        if endpoint is None:
            raise KeyError("Compute endpoint was not found")
        active = [
            job for job in self.jobs()
            if job.get("oracle_serve_url", "").rstrip("/") == endpoint["base_url"].rstrip("/")
            and job.get("status") in {"dispatching", "submitted", "queued", "running", "paused", "validating"}
        ]
        if active:
            return []
        with self._connection() as db:
            row = _row(db.execute("""SELECT * FROM queued_runs
                WHERE status='ready' AND start_authorized=1 AND preflight_endpoint_id=?
                ORDER BY priority DESC, created_at LIMIT 1""", (endpoint_id,)).fetchone())
        if row is None:
            return []
        # Capacity is volatile.  Re-run the compute-host admission check after
        # the user authorizes this row and immediately before dispatching it.
        # This intentionally remains an inventory-based VRAM check; the
        # report records that provenance rather than promising a train-step
        # memory guarantee we have not executed.
        specification = self.specification(str(row["specification_id"]))
        if specification is None:
            raise KeyError("Queued run specification was not found")
        try:
            compute_report = self._request(endpoint["base_url"], "POST", "/compute/preflight", {
                "action": specification["action"],
                "parameters": specification["parameters"],
                "resources": specification["resources"],
            })
        except RuntimeError as exc:
            compute_report = {"ready": False, "reasons": [f"Compute preflight unavailable: {exc}"]}
        if not compute_report.get("ready"):
            reason = "; ".join(compute_report.get("reasons") or ["Compute capacity is no longer ready"])
            with self._connection() as db:
                db.execute(
                    "UPDATE queued_runs SET status='waiting_for_resources', preflight_report_json=?, failure_reason=?, updated_at=? WHERE queued_run_id=?",
                    (_json({"launch_preflight": compute_report}), reason, _now(), row["queued_run_id"]),
                )
            return []
        try:
            job = self.dispatch(str(row["specification_id"]), endpoint_id)
        except (RuntimeError, ValueError) as exc:
            with self._connection() as db:
                db.execute("UPDATE queued_runs SET status='waiting_for_resources', failure_reason=?, updated_at=? WHERE queued_run_id=?", (str(exc), _now(), row["queued_run_id"]))
            return []
        with self._connection() as db:
            db.execute("UPDATE jobs SET queued_run_id=? WHERE job_id=?", (row["queued_run_id"], job["job_id"]))
            db.execute("UPDATE queued_runs SET status='submitted', updated_at=? WHERE queued_run_id=?", (_now(), row["queued_run_id"]))
        return [job]

    def cancel_queued_run(self, queued_run_id: str) -> dict[str, Any]:
        queued = self.queued_run(queued_run_id)
        if queued is None:
            raise KeyError(queued_run_id)
        if queued["status"] in {"submitted", "running"}:
            raise ValueError("Cancel the active execution job before cancelling its queued run")
        with self._connection() as db:
            db.execute("UPDATE queued_runs SET status='cancelled', start_authorized=0, updated_at=? WHERE queued_run_id=?", (_now(), queued_run_id))
        return self.queued_run(queued_run_id)  # type: ignore[return-value]

    def control_job(self, job_id: str, action: str) -> dict[str, Any]:
        """Proxy an explicit lifecycle control to the owning compute service.

        The worker remains authoritative for the process, while the
        orchestrator immediately records the observed lifecycle and a durable
        audit event.  A cancellation may remain ``running`` briefly while the
        worker drains the process; normal reconciliation records its terminal
        state.
        """
        if action not in {"pause", "resume", "cancel"}:
            raise ValueError("Unsupported job control")
        local = self.job(job_id)
        if local is None:
            raise KeyError(job_id)
        if local["status"] in {"indexed", "artifact_invalid", "failed", "cancelled"}:
            raise ValueError("This job has already reached a terminal state")
        remote = self._request(local["oracle_serve_url"], "POST", f"/compute/jobs/{job_id}/{action}")
        remote_status = str(remote.get("status") or local.get("remote_status") or local["status"])
        now = _now()
        with self._connection() as db:
            db.execute(
                "UPDATE jobs SET status=?, remote_status=?, worker_id=?, error=?, updated_at=? WHERE job_id=?",
                (remote_status, remote_status, remote.get("worker_id"), remote.get("error"), now, job_id),
            )
            if remote_status in {"paused", "running"}:
                db.execute("UPDATE run_specifications SET status=?, updated_at=? WHERE specification_id=?", (remote_status, now, local["specification_id"]))
            queued_run_id = local.get("queued_run_id")
            if queued_run_id:
                queue_status = {"paused": "paused", "running": "running", "queued": "submitted"}.get(remote_status, remote_status)
                db.execute("UPDATE queued_runs SET status=?, updated_at=? WHERE queued_run_id=?", (queue_status, now, queued_run_id))
        verb = {"pause": "paused", "resume": "resumed", "cancel": "cancellation requested"}[action]
        self._record_event(job_id, action, f"Job {verb} by user", {"remote_status": remote_status})
        return self.job(job_id)  # type: ignore[return-value]

    def clear_queued_runs(self, *, endpoint_id: str | None = None) -> dict[str, Any]:
        """Cancel only local, non-running queue entries.

        This is intentionally not a bulk compute-job cancellation endpoint.
        Jobs that have been submitted or are running remain untouched and must
        be handled through their own execution controls.
        """
        query, params = "SELECT queued_run_id,status FROM queued_runs", []
        if endpoint_id:
            query += " WHERE preflight_endpoint_id=?"; params.append(endpoint_id)
        with self._connection() as db:
            rows = [dict(row) for row in db.execute(query, params).fetchall()]
            cancellable = [row["queued_run_id"] for row in rows if row["status"] in {"ready", "needs_attention", "waiting_for_resources"}]
            skipped = [{"queued_run_id": row["queued_run_id"], "status": row["status"]} for row in rows if row["queued_run_id"] not in cancellable and row["status"] != "cancelled"]
            if cancellable:
                db.execute(
                    f"UPDATE queued_runs SET status='cancelled', start_authorized=0, updated_at=? WHERE queued_run_id IN ({','.join('?' for _ in cancellable)})",
                    [_now(), *cancellable],
                )
        return {"cleared": cancellable, "skipped": skipped, "endpoint_id": endpoint_id}

    def model_preview(self, *, architecture: str, dataset_id: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build a disposable model for an honest pre-run structural summary."""
        dataset = self.dataset(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        setup = self.model_setup(architecture)
        config = deep_merge(setup["config"], overrides or {})
        config.setdefault("run", {})["model"] = architecture
        with sqlite3.connect(dataset["path"]) as db:
            if config["run"].get("task") == "classification":
                config.setdefault("data", {})["num_classes"] = int(db.execute("SELECT count(*) FROM classification_labels").fetchone()[0])
        try:
            from oracle_builder.registry import get_model_builder
            model = get_model_builder(architecture)(config)
            lines: list[str] = []
            model.summary(print_fn=lines.append)
            return {"architecture": architecture, "input_shape": list(model.input_shape), "output_shape": list(model.output_shape), "parameters": int(model.count_params()), "layers": len(model.layers), "summary": "\n".join(lines[:35]), "config": config}
        except Exception as exc:  # A preview must never prevent configuring a run.
            return {"architecture": architecture, "error": str(exc), "config": config}

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
        requested_resources = specification.get("resources") or {}
        requested_gpus = int(requested_resources.get("gpu_count") or 0)
        if capable and requested_gpus > max((len((worker.get("capabilities") or {}).get("gpus") or []) for worker in capable), default=0):
            reasons.append(f"Run requests {requested_gpus} GPU(s), but no capable worker advertises that capacity")
        requested_gpu_ids = requested_resources.get("gpu_ids")
        if isinstance(requested_gpu_ids, list) and requested_gpu_ids:
            advertised_gpu_ids = {
                str(gpu.get("id"))
                for worker in capable
                for gpu in ((worker.get("capabilities") or {}).get("gpus") or [])
                if isinstance(gpu, dict) and gpu.get("id") is not None
            }
            missing_gpu_ids = [str(item) for item in requested_gpu_ids if str(item) not in advertised_gpu_ids]
            if missing_gpu_ids:
                reasons.append(
                    "Selected GPU(s) are not advertised by a capable worker: "
                    + ", ".join(missing_gpu_ids)
                )
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

    def create_training_experiment(self, *, name: str, dataset_id: str, recipe_ids: list[str], seeds: list[int], description: str = "", resources: dict[str, Any] | None = None, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
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
        if config_overrides is not None and not isinstance(config_overrides, dict):
            raise ValueError("Configuration overrides must be an object")
        overrides = config_overrides or {}
        allowed_override_sections = {"run", "data", "model", "training", "augmentation", "callbacks", "output", "architecture", "input", "encoder", "stem", "normalization", "pooling", "image_embedding", "metadata", "fusion", "classifier", "preprocessing"}
        unknown_sections = set(overrides) - allowed_override_sections
        if unknown_sections or any(not isinstance(value, dict) for value in overrides.values()):
            raise ValueError("Configuration overrides may contain only supported object-valued configuration sections")
        plan = {"kind": "training", "dataset_id": dataset_id, "recipe_ids": recipe_ids, "seeds": seeds, "resources": resources or {}, "config_overrides": overrides}
        config_dir = self.artifact_root / "experiments" / experiment_id / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute("INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (experiment_id, None, name.strip(), description, dataset_id, "expanded", _json(plan), now, now))
            ordinal = 0
            for recipe in selected:
                for seed in seeds:
                    ordinal += 1
                    specification_id = str(uuid.uuid4())
                    source = deep_merge(load_toml(recipe["config_path"]), overrides)
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

    def scan(self, root: str | Path, *, only_unindexed: bool = False) -> dict[str, Any]:
        root_path = Path(root).expanduser().resolve()
        self._require_browse_path(root_path)
        if not root_path.is_dir():
            raise NotADirectoryError(root_path)
        discovered: list[str] = []
        already_indexed: list[str] = []
        skipped: list[dict[str, str]] = []
        for manifest_path in root_path.rglob("artifact.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                schema = manifest.get("artifact_schema", {})
                if schema.get("name") != "oracle_builder_model_run":
                    raise ValueError("unsupported artifact schema")
                artifact_id = str(uuid.UUID(str(manifest["artifact_id"])))
                if only_unindexed and self.artifact(artifact_id) is not None:
                    already_indexed.append(artifact_id)
                    continue
                from oracle_builder.artifacts import validate_run_artifact
                validation = validate_run_artifact(manifest_path.parent)
                if not validation.get("valid"):
                    raise ValueError("artifact validation failed: " + "; ".join(validation.get("errors") or []))
                now = _now()
                model, dataset = manifest.get("model") or {}, manifest.get("dataset") or {}
                facts = self._build_artifact_facts(manifest, manifest_path.parent)
                values = (artifact_id, manifest.get("run_id"), manifest.get("artifact_type", "model_run"), manifest.get("name") or manifest_path.parent.name,
                          facts.get("task") or model.get("task"), facts.get("architecture") or model.get("architecture"), facts.get("variant") or model.get("variant"), manifest.get("status"), manifest.get("lifecycle"), dataset.get("dataset_id"), dataset.get("fingerprint_sha256"), manifest.get("fingerprint_sha256"), str(manifest_path.parent), _json(manifest), now, now)
                with self._connection() as db:
                    db.execute("""INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                      ON CONFLICT(artifact_id) DO UPDATE SET run_id=excluded.run_id,artifact_type=excluded.artifact_type,name=excluded.name,task=excluded.task,architecture=excluded.architecture,variant=excluded.variant,status=excluded.status,lifecycle=excluded.lifecycle,dataset_id=excluded.dataset_id,dataset_fingerprint_sha256=excluded.dataset_fingerprint_sha256,fingerprint_sha256=excluded.fingerprint_sha256,path=excluded.path,manifest_json=excluded.manifest_json,updated_at=excluded.updated_at""", values)
                    db.execute("""INSERT INTO artifact_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(artifact_id) DO UPDATE SET training_set=excluded.training_set,classifier_type=excluded.classifier_type,stem_size=excluded.stem_size,macro_f1=excluded.macro_f1,loss=excluded.loss,training_seconds=excluded.training_seconds,facts_json=excluded.facts_json,updated_at=excluded.updated_at""",
                        (artifact_id, facts.get("training_set"), facts.get("classifier_type"), facts.get("stem_size"), facts.get("macro_f1"), facts.get("loss"), facts.get("training_seconds"), _json(facts), now))
                discovered.append(artifact_id)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                skipped.append({"path": str(manifest_path.parent), "reason": str(exc)})
        return {"root": str(root_path), "artifacts": discovered, "already_indexed": already_indexed, "skipped": skipped}

    def reindex_artifact_catalog(self, artifact_ids: list[str] | None = None) -> dict[str, Any]:
        """Refresh DB-owned catalog facts from registered artifacts only.

        This intentionally reads the sealed run directory and writes only the
        control-plane SQLite database. It never calls artifact update/seal APIs
        and never changes a file within the artifact.
        """
        requested = list(dict.fromkeys(artifact_ids or []))
        with self._connection() as db:
            if requested:
                rows = db.execute(f"SELECT * FROM artifacts WHERE artifact_id IN ({','.join('?' for _ in requested)})", requested).fetchall()
            else:
                rows = db.execute("SELECT * FROM artifacts ORDER BY artifact_id").fetchall()
        found = {row["artifact_id"] for row in rows}
        missing = [artifact_id for artifact_id in requested if artifact_id not in found]
        refreshed: list[str] = []
        skipped: list[dict[str, str]] = []
        for row in rows:
            artifact = _row(row)
            assert artifact is not None
            root = Path(artifact["path"]).resolve()
            try:
                # Registration only occurs from allow-listed browse roots; keep
                # that invariant on later maintenance calls too.
                self._require_browse_path(root)
                manifest = artifact["manifest"]
                if not isinstance(manifest, dict):
                    raise ValueError("Registered artifact has no manifest")
                facts = self._build_artifact_facts(manifest, root)
                now = _now()
                with self._connection() as db:
                    db.execute("""UPDATE artifacts SET task=?,architecture=?,variant=?,updated_at=? WHERE artifact_id=?""",
                        (facts.get("task") or artifact.get("task"), facts.get("architecture") or artifact.get("architecture"), facts.get("variant") or artifact.get("variant"), now, artifact["artifact_id"]))
                    db.execute("""INSERT INTO artifact_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(artifact_id) DO UPDATE SET training_set=excluded.training_set,classifier_type=excluded.classifier_type,stem_size=excluded.stem_size,macro_f1=excluded.macro_f1,loss=excluded.loss,training_seconds=excluded.training_seconds,facts_json=excluded.facts_json,updated_at=excluded.updated_at""",
                        (artifact["artifact_id"], facts.get("training_set"), facts.get("classifier_type"), facts.get("stem_size"), facts.get("macro_f1"), facts.get("loss"), facts.get("training_seconds"), _json(facts), now))
                refreshed.append(artifact["artifact_id"])
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                skipped.append({"artifact_id": artifact["artifact_id"], "reason": str(exc)})
        return {"requested": requested or None, "refreshed": refreshed, "missing": missing, "skipped": skipped, "artifact_files_changed": False}

    def reconcile_startup(self) -> dict[str, Any]:
        """Re-index durable local work without changing its source files.

        This is deliberately safe to repeat.  Sealed run artifacts and frozen
        dataset revisions have stable IDs, so existing database rows are left
        intact and newly discovered records are inserted by the normal catalog
        and ingestion paths.
        """
        result: dict[str, Any] = {"runs_root": str(self.runs_root), "datasets_root": str(self.datasets_root)}
        try:
            known_artifacts = {artifact["artifact_id"] for artifact in self.artifacts()}
            artifact_report = self.scan(self.runs_root)
            result["artifacts"] = {
                "indexed": [artifact_id for artifact_id in artifact_report["artifacts"] if artifact_id not in known_artifacts],
                "refreshed": [artifact_id for artifact_id in artifact_report["artifacts"] if artifact_id in known_artifacts],
                "skipped": artifact_report["skipped"],
            }
        except (OSError, ValueError) as exc:
            result["artifacts"] = {"indexed": [], "refreshed": [], "skipped": [{"path": str(self.runs_root), "reason": str(exc)}]}

        dataset_report = self.scan_training_catalog()
        imported, already_registered, skipped = [], [], []
        for entry in dataset_report["entries"]:
            if entry.get("source_type") != "oracle_sqlite":
                continue
            info = entry.get("dataset_info") if isinstance(entry.get("dataset_info"), dict) else {}
            dataset_id = info.get("dataset_id")
            if entry.get("status") != "frozen" or not isinstance(dataset_id, str):
                skipped.append({"path": entry["path"], "reason": "Only frozen Oracle SQLite dataset revisions are registered"})
                continue
            if self.dataset(dataset_id) is not None:
                already_registered.append(dataset_id)
                continue
            try:
                self.ingest_dataset(entry["path"])
                imported.append(dataset_id)
            except (OSError, ValueError, sqlite3.DatabaseError) as exc:
                skipped.append({"path": entry["path"], "reason": str(exc)})
        result["datasets"] = {
            "catalog_entries": len(dataset_report["entries"]),
            "registered": imported,
            "already_registered": already_registered,
            "skipped": skipped,
        }
        return result

    @staticmethod
    def _nested(value: dict[str, Any], *keys: str) -> Any:
        current: Any = value
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    def _build_artifact_facts(self, manifest: dict[str, Any], root: Path) -> dict[str, Any]:
        """Extract catalog facts without ever changing an artifact.

        The catalog is deliberately allowed to be more helpful than a legacy
        manifest.  Every fallback is retained alongside its provenance so a
        client never has to mistake an inferred display value for sealed
        evaluation evidence.
        """
        config = self._json_file(root / "config" / "resolved.json") or {}
        summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
        manifest_evaluation = summary.get("evaluation") if isinstance(summary.get("evaluation"), dict) else {}
        summary_evaluation = self._json_file(root / "evaluation" / "evaluation_summary.json") or {}
        evaluation = manifest_evaluation or summary_evaluation
        runtime = self._json_file(root / "provenance" / "runtime.json") or {}
        contract = self._json_file(root / "model" / "contract.json") or {}
        history = self._evidence_csv(root / "metrics" / "history.csv", limit=2_000)
        final_history = history[-1] if history else {}
        model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
        dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
        model_config = config.get("model") if isinstance(config.get("model"), dict) else {}
        outputs = model.get("outputs") if isinstance(model.get("outputs"), dict) else {}
        classifier = self._nested(config, "classifier") or self._nested(config, "model", "classifier") or {}
        stem = self._nested(config, "stem") or self._nested(config, "model", "stem") or {}
        pooling = self._nested(config, "pooling") or self._nested(config, "model", "pooling") or {}
        metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else {}
        posthoc = config.get("posthoc") if isinstance(config.get("posthoc"), dict) else config.get("post_hoc") if isinstance(config.get("post_hoc"), dict) else {}
        training = config.get("training") if isinstance(config.get("training"), dict) else {}
        sources: dict[str, dict[str, str]] = {}
        availability: dict[str, str] = {}

        def choose(field: str, *candidates: tuple[Any, str, str]) -> Any:
            """Use the first meaningful source in explicit provenance order."""
            for value, source, status in candidates:
                if value is not None and value != "":
                    sources[field] = {"source": source, "status": status}
                    availability[field] = status
                    return value
            sources[field] = {"source": "unavailable", "status": "not_recorded"}
            availability[field] = "not_recorded"
            return None

        seconds = choose(
            "training_seconds",
            (runtime.get("training_seconds") if isinstance(runtime, dict) else None, "runtime_provenance", "recorded"),
            (runtime.get("duration_seconds") if isinstance(runtime, dict) else None, "runtime_provenance", "recorded"),
        )
        try:
            artifact_size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
            model_size = sum(path.stat().st_size for path in (root / "model").rglob("*") if path.is_file())
        except OSError:
            artifact_size, model_size = None, None
        summary_text = ""
        try:
            summary_text = (root / "model" / "model_summary.txt").read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
        def summary_number(label: str) -> int | None:
            match = re.search(rf"{label}\s*:\s*([\d,]+)", summary_text, flags=re.IGNORECASE)
            return int(match.group(1).replace(",", "")) if match else None
        metric_values = {key: value for key, value in evaluation.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
        # A history value is operational evidence, not a substitute for a
        # held-out evaluation; mark it inferred when it fills an old catalog.
        for key, value in final_history.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metric_values.setdefault(key.removeprefix("val_"), value)
        evaluation_source = "sealed_manifest" if manifest_evaluation else "evaluation_summary"
        classifier_type = choose(
            "classifier_type",
            (classifier.get("type") if isinstance(classifier, dict) else None, "resolved_config", "recorded"),
            (model_config.get("classifier_type") or model_config.get("classifier"), "resolved_config", "recorded"),
            (self._nested(contract, "classifier", "type") or contract.get("classifier_type") if isinstance(contract, dict) else None, "model_contract", "inferred"),
            ("linear" if model.get("task") == "classification" else None, "legacy_classification_default", "inferred"),
        )
        task = choose("task", (model.get("task"), "sealed_manifest", "recorded"), (config.get("task") if isinstance(config, dict) else None, "resolved_config", "recorded"), (contract.get("task") if isinstance(contract, dict) else None, "model_contract", "inferred"))
        architecture = choose("architecture", (model.get("architecture"), "sealed_manifest", "recorded"), (model_config.get("architecture"), "resolved_config", "recorded"), (contract.get("architecture") if isinstance(contract, dict) else None, "model_contract", "inferred"))
        variant = choose("variant", (model.get("variant"), "sealed_manifest", "recorded"), (model_config.get("variant"), "resolved_config", "recorded"), (contract.get("variant") if isinstance(contract, dict) else None, "model_contract", "inferred"))
        epochs_from_history = max((int(row["epoch"]) + 1 for row in history if isinstance(row.get("epoch"), (int, float)) and not isinstance(row.get("epoch"), bool)), default=None)
        facts = {
            "training_set": choose("training_set", (dataset.get("dataset_id") or dataset.get("name"), "sealed_manifest", "recorded"), (self._nested(config, "data", "dataset_id"), "resolved_config", "recorded")),
            "classifier_type": classifier_type,
            "stem_size": choose("stem_size", (stem.get("filters", stem.get("kernel_size", stem.get("size"))) if isinstance(stem, dict) else None, "resolved_config", "recorded"), (self._nested(contract, "stem", "filters") if isinstance(contract, dict) else None, "model_contract", "inferred")),
            "macro_f1": choose("macro_f1", (manifest_evaluation.get("macro_f1", manifest_evaluation.get("f1_macro")), "sealed_manifest", "recorded"), (summary_evaluation.get("macro_f1", summary_evaluation.get("f1_macro")) if isinstance(summary_evaluation, dict) else None, "evaluation_summary", "recorded"), (final_history.get("val_macro_f1", final_history.get("macro_f1")), "training_history", "inferred")),
            "loss": choose("loss", (manifest_evaluation.get("loss", summary.get("loss")), "sealed_manifest", "recorded"), (summary_evaluation.get("loss") if isinstance(summary_evaluation, dict) else None, "evaluation_summary", "recorded"), (final_history.get("val_loss", final_history.get("loss")), "training_history", "inferred")),
            "training_seconds": seconds,
            "created_at": choose("created_at", (manifest.get("created_at"), "sealed_manifest", "recorded")), "completed_at": choose("completed_at", (manifest.get("completed_at"), "sealed_manifest", "recorded")),
            "modified_at": choose("modified_at", (datetime.fromtimestamp(root.stat().st_mtime, timezone.utc).isoformat() if root.exists() else None, "filesystem_metadata", "inferred")),
            "artifact_size_bytes": choose("artifact_size_bytes", (artifact_size, "filesystem_metadata", "inferred")), "model_size_bytes": choose("model_size_bytes", (model_size, "filesystem_metadata", "inferred")),
            "epochs": choose("epochs", (training.get("epochs"), "resolved_config", "recorded"), (epochs_from_history, "training_history", "inferred")), "seed": choose("seed", ((config.get("run") or {}).get("seed") if isinstance(config.get("run"), dict) else None, "resolved_config", "recorded")),
            "parameter_count": choose("parameter_count", (summary_number("Total params"), "model_summary", "inferred")), "trainable_parameters": choose("trainable_parameters", (summary_number("Trainable params"), "model_summary", "inferred")),
            "input_shape": choose("input_shape", ((config.get("data") or {}).get("input_shape") if isinstance(config.get("data"), dict) else None, "resolved_config", "recorded"), (model.get("input", {}).get("shape") if isinstance(model.get("input"), dict) else None, "sealed_manifest", "recorded"), (contract.get("input_shape") if isinstance(contract, dict) else None, "model_contract", "inferred")),
            "num_classes": choose("num_classes", ((config.get("data") or {}).get("num_classes") if isinstance(config.get("data"), dict) else None, "resolved_config", "recorded"), (outputs.get("class_count"), "sealed_manifest", "recorded"), (contract.get("num_classes") if isinstance(contract, dict) else None, "model_contract", "inferred")),
            "embedding_dim": choose("embedding_dim", ((config.get("image_embedding") or {}).get("dimension") if isinstance(config.get("image_embedding"), dict) else model_config.get("embedding_dim"), "resolved_config", "recorded"), (outputs.get("embedding_dimension"), "sealed_manifest", "recorded"), (contract.get("embedding_dim") if isinstance(contract, dict) else None, "model_contract", "inferred")),
            "pooling_type": choose("pooling_type", (pooling.get("type") if isinstance(pooling, dict) else model_config.get("pooling"), "resolved_config", "recorded"), (contract.get("pooling") if isinstance(contract, dict) else None, "model_contract", "inferred")),
            "metadata_field_count": choose("metadata_field_count", (len(metadata.get("fields", metadata.get("features", []))) if metadata and isinstance(metadata.get("fields", metadata.get("features", [])), list) else None, "resolved_config", "recorded")),
            "posthoc_type": choose("posthoc_type", (posthoc.get("type") if isinstance(posthoc, dict) else None, "resolved_config", "recorded")),
            "architecture_version": choose("architecture_version", ((config.get("architecture") or {}).get("version") if isinstance(config.get("architecture"), dict) else None, "resolved_config", "recorded")),
            "architecture": architecture, "variant": variant, "task": task, "dataset_fingerprint": choose("dataset_fingerprint", (dataset.get("fingerprint_sha256"), "sealed_manifest", "recorded")),
            "metrics": metric_values, "config": config, "evaluation": evaluation, "runtime": runtime, "field_sources": sources, "field_availability": availability,
        }
        for name, value in metric_values.items():
            # Evaluation values retain their source; history-only values are explicitly inferred.
            choose(name, (manifest_evaluation.get(name), "sealed_manifest", "recorded"), (summary_evaluation.get(name) if isinstance(summary_evaluation, dict) else None, "evaluation_summary", "recorded"), (value, "training_history", "inferred"))
        facts.update(metric_values)
        for key in ("stem_size", "epochs", "seed", "parameter_count", "trainable_parameters", "num_classes", "embedding_dim", "metadata_field_count", "architecture_version"):
            if isinstance(facts[key], bool): facts[key] = None
        for key in ("macro_f1", "loss", "training_seconds", "artifact_size_bytes", "model_size_bytes"):
            if not isinstance(facts[key], (int, float)) or isinstance(facts[key], bool): facts[key] = None
        return facts

    def training_catalog_roots_info(self) -> list[dict[str, str]]:
        return [{"root_id": root_id, "path": str(path)} for root_id, path in self.training_catalog_roots.items()]

    def scan_training_catalog(self, root_id: str | None = None) -> dict[str, Any]:
        """Safely inspect an allow-listed source directory without importing it."""
        from oracle_builder.orchestration.training_catalog import scan_training_catalog
        if root_id is not None and root_id not in self.training_catalog_roots:
            raise KeyError(f"Unknown training catalog root: {root_id}")
        root_ids = [root_id] if root_id else list(self.training_catalog_roots)
        now = _now()
        reports: list[tuple[str, dict[str, Any]]] = []
        for selected_root_id in root_ids:
            reports.append((selected_root_id, scan_training_catalog(self.training_catalog_roots[selected_root_id])))
        with self._connection() as db:
            for selected_root_id, report in reports:
                catalog_ids = [entry["catalog_id"] for entry in report["entries"]]
                if catalog_ids:
                    db.execute(f"DELETE FROM training_catalog_entries WHERE root_id=? AND catalog_id NOT IN ({','.join('?' for _ in catalog_ids)})", [selected_root_id, *catalog_ids])
                else:
                    db.execute("DELETE FROM training_catalog_entries WHERE root_id=?", (selected_root_id,))
                for entry in report["entries"]:
                    db.execute("""INSERT INTO training_catalog_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(catalog_id) DO UPDATE SET root_id=excluded.root_id,name=excluded.name,path=excluded.path,source_type=excluded.source_type,fingerprint_sha256=excluded.fingerprint_sha256,metadata_json=excluded.metadata_json,scanned_at=excluded.scanned_at,updated_at=excluded.updated_at""",
                        (entry["catalog_id"], selected_root_id, entry["name"], entry["path"], entry["source_type"], entry.get("fingerprint_sha256"), _json(entry), report["scanned_at"], now))
        entries = [entry for _, report in reports for entry in report["entries"]]
        return {
            "root_id": root_id,
            "roots": [{"root_id": selected_root_id, "path": report["root"]} for selected_root_id, report in reports],
            "entries": entries,
            "scanned_at": now,
        }

    def training_catalog(self, *, root_id: str | None = None) -> list[dict[str, Any]]:
        query, params = "SELECT * FROM training_catalog_entries", []
        if root_id:
            query += " WHERE root_id=?"; params.append(root_id)
        query += " ORDER BY name COLLATE NOCASE"
        with self._connection() as db:
            return [self._training_catalog_view(_row(row)) for row in db.execute(query, params).fetchall()]  # type: ignore[list-item]

    def training_catalog_bundles(self, *, root_id: str | None = None) -> list[dict[str, Any]]:
        """Group compatible revisions without hiding their immutable identity."""
        bundles: dict[str, dict[str, Any]] = {}
        for entry in self.training_catalog(root_id=root_id):
            family_id = str(entry.get("family_id") or entry.get("training_set_family") or entry["catalog_id"])
            bundle = bundles.setdefault(family_id, {
                "family_id": family_id,
                "name": entry.get("training_set_family") or entry.get("name"),
                "task": entry.get("task"),
                "versions": [],
            })
            bundle["versions"].append(entry)
        for bundle in bundles.values():
            bundle["versions"].sort(key=lambda entry: (str(entry.get("modified_at") or ""), str(entry.get("training_set_version") or "")), reverse=True)
            versions = bundle["versions"]
            bundle["frozen_count"] = sum(entry.get("status") == "frozen" for entry in versions)
            bundle["unfrozen_count"] = len(versions) - bundle["frozen_count"]
            bundle["latest"] = versions[0]
        return sorted(bundles.values(), key=lambda bundle: str(bundle["name"]).casefold())

    def training_catalog_entry(self, catalog_id: str) -> dict[str, Any] | None:
        with self._connection() as db:
            return _row(db.execute("SELECT * FROM training_catalog_entries WHERE catalog_id=?", (catalog_id,)).fetchone())

    @staticmethod
    def _training_catalog_view(entry: dict[str, Any] | None) -> dict[str, Any]:
        """Expose catalog metadata as convenient read-only fields without losing its source map."""
        if entry is None:
            return {}
        result = dict(entry)
        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                result.setdefault(key, value)
        return result

    def _training_catalog_sqlite_path(self, entry: dict[str, Any]) -> Path:
        if entry.get("source_type") != "oracle_sqlite":
            raise KeyError(entry.get("catalog_id"))
        path = Path(entry["path"]).resolve()
        if not path.is_file() or not any(path.is_relative_to(root) for root in self.training_catalog_roots.values()):
            raise FileNotFoundError(path)
        return path

    def training_catalog_previews(self, catalog_id: str, *, label: str | None = None, offset: int = 0, limit: int = 24) -> dict[str, Any]:
        entry = self.training_catalog_entry(catalog_id)
        if entry is None: raise KeyError(catalog_id)
        path = self._training_catalog_sqlite_path(entry)
        offset, limit = max(0, offset), min(max(1, limit), 100)
        with sqlite3.connect(path) as db:
            db.row_factory = sqlite3.Row
            info = read_dataset_info(db)
            where, params = "", []
            if info["dataset_type"] == "classification":
                if label:
                    where, params = " WHERE l.name=?", [label]
                total = int(db.execute("""SELECT count(*) FROM dataset_items di
                    LEFT JOIN classification_annotations ca ON ca.item_id=di.item_id AND ca.is_current=1 AND ca.status='accepted'
                    LEFT JOIN classification_labels l ON l.label_id=ca.label_id""" + where, params).fetchone()[0])
                query = """SELECT di.item_id,di.source_key,l.name AS label FROM dataset_items di
                    JOIN classification_items ci ON ci.item_id=di.item_id
                    LEFT JOIN classification_annotations ca ON ca.item_id=di.item_id AND ca.is_current=1 AND ca.status='accepted'
                    LEFT JOIN classification_labels l ON l.label_id=ca.label_id""" + where + " ORDER BY di.item_id LIMIT ? OFFSET ?"
            else:
                total = int(db.execute("SELECT count(*) FROM dataset_items").fetchone()[0])
                query, params = """SELECT di.item_id,di.source_key,NULL AS label FROM dataset_items di
                    JOIN mask_refinement_items mi ON mi.item_id=di.item_id ORDER BY di.item_id LIMIT ? OFFSET ?""", []
            rows = [dict(row) for row in db.execute(query, [*params, limit, offset])]
        items = [{**row, "preview_url": f"/api/v1/training-catalog/{catalog_id}/previews/{row['item_id']}"} for row in rows]
        return {"catalog_id": catalog_id, "items": items, "total": total, "offset": offset, "limit": limit}

    def training_catalog_preview_image(self, catalog_id: str, item_id: str, *, max_size: int = 320) -> bytes:
        """Render a bounded JPEG preview for an allow-listed catalog image.

        The UI receives opaque item IDs rather than source paths, so this never
        turns the catalog endpoint into a general file reader.
        """
        entry = self.training_catalog_entry(catalog_id)
        if entry is None: raise KeyError(catalog_id)
        path = self._training_catalog_sqlite_path(entry)
        return self._sqlite_preview_image(path, item_id, max_size=max_size)

    def freeze_training_catalog_entry(self, catalog_id: str) -> dict[str, Any]:
        """Freeze one explicit SQLite revision, then register its immutable state."""
        entry = self.training_catalog_entry(catalog_id)
        if entry is None: raise KeyError(catalog_id)
        path = self._training_catalog_sqlite_path(entry)
        with sqlite3.connect(path) as db:
            info = read_dataset_info(db)
            if info["lifecycle"] not in {"working", "frozen"}:
                raise ValueError(f"Only working datasets can be frozen; this revision is {info['lifecycle']!r}")
            if info["lifecycle"] == "working":
                set_dataset_lifecycle(db, "frozen", actor="oracle-orchestrator", details={"catalog_id": catalog_id})
            db.commit()
        report = self.scan_training_catalog(entry["root_id"])
        refreshed = next((item for item in report["entries"] if item["catalog_id"] == catalog_id), None)
        if refreshed is None: raise RuntimeError("Frozen dataset was not returned by catalog reconciliation")
        dataset = self.ingest_dataset(path)
        return {"entry": refreshed, "dataset": dataset}

    def compare_training_catalog(self, catalog_ids: list[str]) -> dict[str, Any]:
        entries = [self.training_catalog_entry(value) for value in dict.fromkeys(catalog_ids)]
        if len(entries) < 2 or any(entry is None for entry in entries): raise KeyError("Select at least two catalog entries")
        values = [entry for entry in entries if entry]
        return {"entries": [self._training_catalog_view(entry) for entry in values], "comparison": {"item_counts": {entry["catalog_id"]: entry["metadata"].get("item_count") for entry in values}, "class_counts": {entry["catalog_id"]: entry["metadata"].get("class_count") for entry in values}, "fingerprints": {entry["catalog_id"]: entry.get("fingerprint_sha256") for entry in values}}}

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
    def _json_file(path: Path) -> dict[str, Any] | list[Any] | None:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            return loaded if isinstance(loaded, (dict, list)) else None
        except (OSError, json.JSONDecodeError):
            return None

    def dataset_detail(self, dataset_id: str) -> dict[str, Any]:
        """Summarize a registered dataset without returning item payloads."""
        dataset = self.dataset(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        path = Path(dataset["path"])
        with sqlite3.connect(path) as db:
            db.row_factory = sqlite3.Row
            info = read_dataset_info(db)
            total = int(db.execute("SELECT count(*) FROM dataset_items").fetchone()[0])
            asset_count = int(db.execute("SELECT count(*) FROM assets").fetchone()[0])
            if info["dataset_type"] == "classification":
                labels = [dict(row) for row in db.execute("""SELECT l.label_id,l.class_index,l.name,count(ca.annotation_id) AS item_count
                    FROM classification_labels l LEFT JOIN classification_annotations ca
                    ON ca.label_id=l.label_id AND ca.is_current=1 AND ca.status='accepted'
                    GROUP BY l.label_id ORDER BY l.class_index""")]
            else:
                labels = []
            annotations_table = "classification_annotations" if info["dataset_type"] == "classification" else "mask_annotations"
            annotated = int(db.execute(f"SELECT count(*) FROM {annotations_table} WHERE is_current=1 AND status='accepted'").fetchone()[0])
        return {
            "dataset": dataset,
            "info": info,
            "counts": {"items": total, "assets": asset_count, "current_annotations": annotated},
            "labels": labels,
            "preview": {"available": total > 0, "max_limit": 100, "max_size": 1024},
        }

    def dataset_previews(self, dataset_id: str, *, offset: int = 0, limit: int = 24) -> dict[str, Any]:
        """List page-sized preview metadata; image bytes remain behind item URLs."""
        dataset = self.dataset(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        offset, limit = max(0, offset), min(max(1, limit), 100)
        with sqlite3.connect(dataset["path"]) as db:
            db.row_factory = sqlite3.Row
            info = read_dataset_info(db)
            total = int(db.execute("SELECT count(*) FROM dataset_items").fetchone()[0])
            if info["dataset_type"] == "classification":
                query = """SELECT di.item_id,di.source_key,di.metadata_json,a.shape_json,l.name AS label
                    FROM dataset_items di JOIN classification_items ci ON ci.item_id=di.item_id
                    JOIN assets a ON a.asset_id=ci.image_asset_id
                    LEFT JOIN classification_annotations ca ON ca.item_id=di.item_id AND ca.is_current=1 AND ca.status='accepted'
                    LEFT JOIN classification_labels l ON l.label_id=ca.label_id
                    ORDER BY di.item_id LIMIT ? OFFSET ?"""
            else:
                query = """SELECT di.item_id,di.source_key,di.metadata_json,a.shape_json,NULL AS label
                    FROM dataset_items di JOIN mask_refinement_items mi ON mi.item_id=di.item_id
                    JOIN assets a ON a.asset_id=mi.image_asset_id ORDER BY di.item_id LIMIT ? OFFSET ?"""
            items = []
            for row in db.execute(query, (limit, offset)):
                item = dict(row)
                shape = json.loads(item.pop("shape_json") or "null")
                item["shape"] = shape
                item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
                item["image_url"] = f"/api/v1/datasets/{dataset_id}/previews/{item['item_id']}"
                if info["dataset_type"] == "mask_refinement":
                    item["mask_url"] = f"/api/v1/datasets/{dataset_id}/previews/{item['item_id']}?kind=mask"
                    item["candidate_mask_url"] = f"/api/v1/datasets/{dataset_id}/previews/{item['item_id']}?kind=candidate_mask"
                items.append(item)
        return {"dataset_id": dataset_id, "offset": offset, "limit": limit, "total": total, "items": items}

    @staticmethod
    def _sqlite_preview_image(path: Path, item_id: str, *, kind: str = "image", max_size: int = 320) -> bytes:
        """Decode an item from an allow-listed Oracle SQLite revision as JPEG."""
        if kind not in {"image", "mask", "candidate_mask"}:
            raise ValueError("Preview kind must be image, mask, or candidate_mask")
        max_size = min(max(32, max_size), 1024)
        with sqlite3.connect(path) as db:
            db.row_factory = sqlite3.Row
            info = read_dataset_info(db)
            if info["dataset_type"] == "classification" and kind != "image":
                raise FileNotFoundError(item_id)
            if info["dataset_type"] == "classification":
                query = """SELECT a.payload,a.encoding,a.shape_json,a.external_uri FROM classification_items ci
                    JOIN assets a ON a.asset_id=ci.image_asset_id WHERE ci.item_id=?"""
            elif kind == "image":
                query = """SELECT a.payload,a.encoding,a.shape_json,a.external_uri FROM mask_refinement_items mi
                    JOIN assets a ON a.asset_id=mi.image_asset_id WHERE mi.item_id=?"""
            elif kind == "candidate_mask":
                query = """SELECT a.payload,a.encoding,a.shape_json,a.external_uri FROM mask_refinement_items mi
                    JOIN assets a ON a.asset_id=mi.candidate_mask_asset_id WHERE mi.item_id=?"""
            else:
                query = """SELECT a.payload,a.encoding,a.shape_json,a.external_uri FROM mask_annotations ma
                    JOIN assets a ON a.asset_id=ma.mask_asset_id WHERE ma.item_id=? AND ma.is_current=1 AND ma.status='accepted'"""
            row = db.execute(query, (item_id,)).fetchone()
        if row is None or row["payload"] is None:
            # External asset URIs are intentionally not fetched by the API.
            raise FileNotFoundError(item_id)
        array = np.asarray(decode_blob(row["payload"], row["encoding"], row["shape_json"]))
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim not in {2, 3}:
            raise ValueError("Dataset asset is not an image array")
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        if array.dtype.kind == "f":
            low, high = float(array.min(initial=0)), float(array.max(initial=1))
            array = ((array - low) / (high - low) * 255 if high > low else array * 255).clip(0, 255).astype("uint8")
        else:
            array = np.clip(array, 0, 255).astype("uint8")
        image = Image.fromarray(array)
        if image.mode not in {"L", "RGB"}:
            image = image.convert("RGB")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=85, optimize=True)
        return output.getvalue()

    def dataset_preview_image(self, dataset_id: str, item_id: str, *, kind: str = "image", max_size: int = 320) -> bytes:
        """Decode one registered local dataset asset and emit a bounded JPEG preview."""
        dataset = self.dataset(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        return self._sqlite_preview_image(Path(dataset["path"]), item_id, kind=kind, max_size=max_size)

    def artifact_history(self, artifact_id: str, *, limit: int = 500) -> dict[str, Any]:
        artifact = self.artifact(artifact_id)
        if artifact is None:
            raise KeyError(artifact_id)
        limit = min(max(1, limit), 2000)
        root = Path(artifact["path"])
        rows = self._evidence_csv(root / "metrics" / "history.csv", limit=limit)
        columns = list(rows[0]) if rows else []
        return {"artifact_id": artifact_id, "rows": rows, "columns": columns, "limit": limit,
                "truncated": len(rows) == limit and (root / "metrics" / "history.csv").is_file()}

    def artifact_detail(self, artifact_id: str) -> dict[str, Any]:
        """Read-only inspector payload for a sealed artifact; all large media stays separate."""
        artifact = self.artifact(artifact_id)
        if artifact is None:
            raise KeyError(artifact_id)
        root = Path(artifact["path"])
        model_summary = None
        try:
            model_summary = (root / "model" / "model_summary.txt").read_text(encoding="utf-8")[:100_000]
        except OSError:
            pass
        runtime = self._json_file(root / "provenance" / "runtime.json")
        config = self._json_file(root / "config" / "resolved.json")
        contract = self._json_file(root / "model" / "contract.json")
        return {"artifact": self._artifact_result(artifact), "facts": self._artifact_facts(artifact_id), "manifest": artifact["manifest"],
                "runtime": runtime, "config": config, "model_contract": contract,
                "architecture": {"summary": model_summary, "available": model_summary is not None},
                "history_url": f"/api/v1/artifacts/{artifact_id}/history",
                "evidence_url": f"/api/v1/artifacts/{artifact_id}/evidence"}

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
                "active": sum(candidate["status"] in {"dispatched", "queued", "running", "paused", "validating"} for candidate in candidates),
                "indexed": sum(candidate["artifact"] is not None for candidate in candidates),
                "failed": sum(candidate["status"] in {"failed", "dispatch_failed", "artifact_invalid", "cancelled"} for candidate in candidates),
            },
        }

    def create_comparison(self, *, name: str, artifact_ids: list[str], description: str = "") -> dict[str, Any]:
        """Create a legacy comparison.

        The historical endpoint remains strict so older API consumers retain
        their reproducibility guardrail. New UI work should use comparison
        groups, which intentionally allow a user to relate unlike runs.
        """
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

    def _artifact_runtime(self, artifact_id: str) -> dict[str, float | None]:
        """Return recorded queue and training duration when this artifact has a job."""
        with self._connection() as db:
            row = db.execute("""SELECT jobs.submitted_at, jobs.started_at, jobs.completed_at
                FROM jobs JOIN run_specifications USING (specification_id)
                WHERE run_specifications.artifact_id=?
                ORDER BY jobs.submitted_at DESC LIMIT 1""", (artifact_id,)).fetchone()
        if row is None:
            return {"queue_seconds": None, "runtime_seconds": None}
        return {
            "queue_seconds": self._duration_seconds(row["submitted_at"], row["started_at"]),
            "runtime_seconds": self._duration_seconds(row["started_at"], row["completed_at"]),
        }

    def artifact_catalog(self) -> list[dict[str, Any]]:
        """A compact, display-ready global run/artifact catalog for selection UIs."""
        return self.artifact_catalog_query()["artifacts"]

    def _artifact_facts(self, artifact_id: str) -> dict[str, Any]:
        with self._connection() as db:
            row = db.execute("SELECT training_set,classifier_type,stem_size,macro_f1,loss,training_seconds,facts_json FROM artifact_facts WHERE artifact_id=?", (artifact_id,)).fetchone()
        if row is None:
            return {}
        raw = dict(row)
        raw["facts"] = json.loads(raw.pop("facts_json"))
        return raw

    _CATALOG_COLUMNS = {
        "artifact_id": "artifacts.artifact_id", "name": "artifacts.name", "task": "artifacts.task",
        "architecture": "artifacts.architecture", "variant": "artifacts.variant", "status": "artifacts.status",
        "lifecycle": "artifacts.lifecycle", "dataset_id": "artifacts.dataset_id", "updated_at": "artifacts.updated_at",
        "discovered_at": "artifacts.discovered_at", "training_set": "artifact_facts.training_set",
        "classifier_type": "artifact_facts.classifier_type", "stem_size": "artifact_facts.stem_size",
        "macro_f1": "artifact_facts.macro_f1", "loss": "artifact_facts.loss", "training_seconds": "artifact_facts.training_seconds",
        "created_at": "json_extract(artifact_facts.facts_json, '$.created_at')",
        "completed_at": "json_extract(artifact_facts.facts_json, '$.completed_at')",
        "modified_at": "json_extract(artifact_facts.facts_json, '$.modified_at')",
        "artifact_size_bytes": "json_extract(artifact_facts.facts_json, '$.artifact_size_bytes')",
        "model_size_bytes": "json_extract(artifact_facts.facts_json, '$.model_size_bytes')",
        "epochs": "json_extract(artifact_facts.facts_json, '$.epochs')",
        "parameter_count": "json_extract(artifact_facts.facts_json, '$.parameter_count')",
        "trainable_parameters": "json_extract(artifact_facts.facts_json, '$.trainable_parameters')",
        "accuracy": "json_extract(artifact_facts.facts_json, '$.accuracy')",
        "balanced_accuracy": "json_extract(artifact_facts.facts_json, '$.balanced_accuracy')",
        "macro_precision": "json_extract(artifact_facts.facts_json, '$.macro_precision')",
        "macro_recall": "json_extract(artifact_facts.facts_json, '$.macro_recall')",
        "macro_average_precision": "json_extract(artifact_facts.facts_json, '$.macro_average_precision')",
        "num_classes": "json_extract(artifact_facts.facts_json, '$.num_classes')",
        "embedding_dim": "json_extract(artifact_facts.facts_json, '$.embedding_dim')",
        "pooling_type": "json_extract(artifact_facts.facts_json, '$.pooling_type')",
        "metadata_field_count": "json_extract(artifact_facts.facts_json, '$.metadata_field_count')",
        "posthoc_type": "json_extract(artifact_facts.facts_json, '$.posthoc_type')",
        "architecture_version": "json_extract(artifact_facts.facts_json, '$.architecture_version')",
    }

    def artifact_filter_schema(self) -> dict[str, Any]:
        fields = [
            {"key": key, "label": key.replace("_", " ").title(),
             "type": "number" if key in {"stem_size", "macro_f1", "loss", "training_seconds", "artifact_size_bytes", "model_size_bytes", "epochs", "parameter_count", "trainable_parameters", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_average_precision", "num_classes", "embedding_dim", "metadata_field_count", "architecture_version"} else "datetime" if key in {"updated_at", "discovered_at", "created_at", "completed_at", "modified_at"} else "text",
             "operators": ["eq", "in", "contains"] if key not in {"stem_size", "macro_f1", "loss", "training_seconds", "artifact_size_bytes", "model_size_bytes", "epochs", "parameter_count", "trainable_parameters", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_average_precision", "num_classes", "embedding_dim", "metadata_field_count", "architecture_version", "updated_at", "discovered_at", "created_at", "completed_at", "modified_at"} else ["eq", "gte", "lte"]}
            for key in self._CATALOG_COLUMNS
        ]
        fields.append({"key": "tag", "label": "Tag", "type": "text", "operators": ["eq", "in"]})
        return {"fields": fields, "default_columns": ["name", "status", "architecture", "variant", "classifier_type", "training_set", "macro_f1", "accuracy", "epochs", "artifact_size_bytes", "created_at", "tags"]}

    def artifact_catalog_query(self, *, filters: dict[str, Any] | None = None, sort: str = "updated_at", order: str = "desc", offset: int = 0, limit: int = 100) -> dict[str, Any]:
        """Server-side catalog paging with a deliberately small, typed filter language.

        A filter can be a scalar, a list (membership), or ``{"gte": value}``,
        ``{"lte": value}``, ``{"contains": text}``, and ``{"eq": value}``.
        """
        filters, params, clauses = filters or {}, [], []
        for key, expression in filters.items():
            if key == "search":
                clauses.append("(LOWER(artifacts.name) LIKE ? OR LOWER(COALESCE(artifacts.architecture, '')) LIKE ? OR LOWER(COALESCE(artifacts.variant, '')) LIKE ? OR LOWER(COALESCE(artifact_facts.training_set, '')) LIKE ?)")
                params.extend([f"%{str(expression).lower()}%"] * 4)
                continue
            if key == "tag":
                values = expression if isinstance(expression, list) else [expression]
                if not values: continue
                clauses.append("EXISTS (SELECT 1 FROM artifact_tag_assignments ata JOIN artifact_tags at ON at.tag_id=ata.tag_id WHERE ata.artifact_id=artifacts.artifact_id AND at.name IN (%s))" % ",".join("?" for _ in values))
                params.extend(str(item) for item in values)
                continue
            column = self._CATALOG_COLUMNS.get(key)
            if column is None:
                raise ValueError(f"Unsupported artifact catalog filter: {key}")
            if isinstance(expression, dict):
                for operator, value in expression.items():
                    if operator == "eq": clauses.append(f"{column}=?"); params.append(value)
                    elif operator == "gte": clauses.append(f"{column}>=?"); params.append(value)
                    elif operator == "lte": clauses.append(f"{column}<=?"); params.append(value)
                    elif operator == "contains": clauses.append(f"LOWER(CAST({column} AS TEXT)) LIKE ?"); params.append(f"%{str(value).lower()}%")
                    else: raise ValueError(f"Unsupported artifact catalog operator: {operator}")
            elif isinstance(expression, list):
                if expression: clauses.append(f"{column} IN ({','.join('?' for _ in expression)})"); params.extend(expression)
            else:
                clauses.append(f"{column}=?"); params.append(expression)
        sort_column = self._CATALOG_COLUMNS.get(sort)
        if sort_column is None: raise ValueError(f"Unsupported artifact catalog sort: {sort}")
        direction = "ASC" if order.lower() == "asc" else "DESC"
        offset, limit = max(0, int(offset)), min(max(1, int(limit)), 500)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        source = " FROM artifacts LEFT JOIN artifact_facts USING (artifact_id)"
        with self._connection() as db:
            total = int(db.execute("SELECT COUNT(*)" + source + where, params).fetchone()[0])
            rows = db.execute("SELECT artifacts.*,artifact_facts.training_set,artifact_facts.classifier_type,artifact_facts.stem_size,artifact_facts.macro_f1,artifact_facts.loss,artifact_facts.training_seconds,artifact_facts.facts_json" + source + where + f" ORDER BY {sort_column} {direction}, artifacts.artifact_id ASC LIMIT ? OFFSET ?", [*params, limit, offset]).fetchall()
        values = []
        for row in rows:
            artifact = _row(row)
            if artifact:
                facts = {**(artifact.get("facts") if isinstance(artifact.get("facts"), dict) else {}), **{key: artifact.get(key) for key in ("training_set", "classifier_type", "stem_size", "macro_f1", "loss", "training_seconds")}}
                values.append({**self._artifact_result(artifact), **facts, "timing": self._artifact_runtime(artifact["artifact_id"]), "tags": self.tags_for_artifact(artifact["artifact_id"])})
        return {"artifacts": values, "total": total, "offset": offset, "limit": limit, "sort": sort, "order": direction.lower()}

    def tags(self) -> list[dict[str, Any]]:
        return self._many("SELECT * FROM artifact_tags ORDER BY name COLLATE NOCASE")

    def create_tag(self, *, name: str, color: str | None = None) -> dict[str, Any]:
        normalized = name.strip()
        if not normalized or len(normalized) > 80: raise ValueError("Tag name must be between 1 and 80 characters")
        now, tag_id = _now(), str(uuid.uuid4())
        with self._connection() as db:
            db.execute("INSERT INTO artifact_tags VALUES (?, ?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET color=COALESCE(excluded.color, artifact_tags.color),updated_at=excluded.updated_at", (tag_id, normalized, color, now, now))
            row = db.execute("SELECT * FROM artifact_tags WHERE name=? COLLATE NOCASE", (normalized,)).fetchone()
        return _row(row)  # type: ignore[return-value]

    def tags_for_artifact(self, artifact_id: str) -> list[dict[str, Any]]:
        with self._connection() as db:
            return [_row(row) for row in db.execute("SELECT artifact_tags.* FROM artifact_tags JOIN artifact_tag_assignments USING (tag_id) WHERE artifact_id=? ORDER BY name COLLATE NOCASE", (artifact_id,)).fetchall()]  # type: ignore[list-item]

    def set_artifact_tags(self, artifact_id: str, tag_names: list[str]) -> list[dict[str, Any]]:
        if self.artifact(artifact_id) is None: raise KeyError(artifact_id)
        names = list(dict.fromkeys(str(name).strip() for name in tag_names if str(name).strip()))
        if len(names) > 30: raise ValueError("An artifact can have at most 30 tags")
        now = _now()
        with self._connection() as db:
            db.execute("DELETE FROM artifact_tag_assignments WHERE artifact_id=?", (artifact_id,))
            for name in names:
                if len(name) > 80: raise ValueError("Tag name must be between 1 and 80 characters")
                tag_id = str(uuid.uuid4())
                db.execute("INSERT INTO artifact_tags VALUES (?, ?, NULL, ?, ?) ON CONFLICT(name) DO UPDATE SET updated_at=excluded.updated_at", (tag_id, name, now, now))
                saved = db.execute("SELECT tag_id FROM artifact_tags WHERE name=? COLLATE NOCASE", (name,)).fetchone()[0]
                db.execute("INSERT INTO artifact_tag_assignments VALUES (?, ?, ?)", (artifact_id, saved, now))
        return self.tags_for_artifact(artifact_id)

    @staticmethod
    def architecture_view_from_config(config: dict[str, Any]) -> dict[str, Any]:
        """Stable UI graph, intentionally derived from config rather than Keras internals."""
        version = int((config.get("architecture") or {}).get("version", 1))
        model = config.get("model") or {}
        input_module = config.get("input") or config.get("data") or {}
        modules = [
            {"id": "input", "kind": "input", "label": "Input", "config": input_module},
            {"id": "preprocessing", "kind": "preprocessing", "label": "Geometry & channels", "config": config.get("preprocessing") or {}},
            {"id": "encoder", "kind": "encoder", "label": "CNN encoder", "config": config.get("encoder") or {"architecture": model.get("architecture") or (config.get("run") or {}).get("model"), "variant": model.get("variant")}},
            {"id": "stem", "kind": "stem", "label": "Stem", "config": config.get("stem") or {}},
            {"id": "pooling", "kind": "pooling", "label": "Pooling", "config": config.get("pooling") or {}},
            {"id": "image_embedding", "kind": "embedding", "label": "Image embedding", "config": config.get("image_embedding") or {}},
        ]
        metadata = config.get("metadata") or {}
        fusion = config.get("fusion") or {}
        if metadata.get("fields") or metadata.get("enabled") or metadata.get("features"):
            modules.extend([
                {"id": "metadata", "kind": "metadata", "label": "Metadata encoder", "config": metadata},
                {"id": "fusion", "kind": "fusion", "label": "Fusion", "config": fusion},
            ])
        modules.append({"id": "classifier", "kind": "classifier", "label": "Primary classifier", "config": config.get("classifier") or {}})
        posthoc = (config.get("posthoc") or config.get("post_hoc") or {})
        if posthoc.get("enabled") or posthoc.get("type"):
            modules.append({"id": "posthoc", "kind": "posthoc", "label": "Post-hoc predictor", "config": posthoc})
        return {"version": version, "modules": modules, "edges": [{"from": modules[index]["id"], "to": modules[index + 1]["id"]} for index in range(len(modules) - 1)],
                "representations": ["feature_map", "image_embedding", "metadata_embedding", "fused_embedding", "projection_embedding"] if version >= 2 else ["features"]}

    def artifact_architecture_view(self, artifact_id: str) -> dict[str, Any]:
        artifact = self.artifact(artifact_id)
        if artifact is None: raise KeyError(artifact_id)
        config = self._json_file(Path(artifact["path"]) / "config" / "resolved.json") or {}
        return {"artifact_id": artifact_id, "architecture": self.architecture_view_from_config(config), "config_available": bool(config)}

    @staticmethod
    def _draft_config(config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(config, dict): raise ValueError("Draft config must be an object")
        resolved = deep_merge(DEFAULT_CONFIG, config)
        # Drafts are authored against the composable contract unless explicitly
        # cloned from an archived V1 run.
        resolved.setdefault("architecture", {}).setdefault("version", 2)
        # A new visual draft is intentionally constructible before a dataset is
        # selected. These placeholders are replaced during training planning.
        resolved.setdefault("run", {}).setdefault("task", "classification")
        resolved["run"].setdefault("model", "resnet18")
        resolved.setdefault("data", {}).setdefault("input_shape", [128, 128])
        resolved["data"].setdefault("num_classes", 2)
        if not resolved.setdefault("training", {}).get("loss"):
            resolved["training"]["loss"] = "sparse_categorical_crossentropy"
        resolved.setdefault("preprocessing", {})["invert"] = bool(resolved["preprocessing"].get("invert") or False)
        validate_config(resolved)
        return resolved

    def create_model_draft(self, *, name: str, config: dict[str, Any] | None = None, description: str = "", layout: dict[str, Any] | None = None, source_artifact_id: str | None = None) -> dict[str, Any]:
        if source_artifact_id and self.artifact(source_artifact_id) is None: raise KeyError(source_artifact_id)
        source_config: dict[str, Any] = {}
        if source_artifact_id:
            source = self.artifact(source_artifact_id)
            source_config = self._json_file(Path(source["path"]) / "config" / "resolved.json") if source else {}
            source_config = source_config or {}
        resolved = self._draft_config(deep_merge(source_config, config or {}))
        now, draft_id = _now(), str(uuid.uuid4())
        safe_layout = layout if isinstance(layout, dict) else {}
        with self._connection() as db:
            db.execute("INSERT INTO model_drafts VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)", (draft_id, name.strip() or "Untitled model", description, source_artifact_id, _json(resolved), _json(safe_layout), now, now))
            db.execute("INSERT INTO model_draft_revisions VALUES (?, 1, ?, ?, ?)", (draft_id, _json(resolved), _json(safe_layout), now))
        return self.model_draft(draft_id)  # type: ignore[return-value]

    def model_drafts(self) -> list[dict[str, Any]]:
        return self._many("SELECT * FROM model_drafts ORDER BY updated_at DESC")

    def model_draft(self, draft_id: str) -> dict[str, Any] | None:
        with self._connection() as db:
            draft = _row(db.execute("SELECT * FROM model_drafts WHERE draft_id=?", (draft_id,)).fetchone())
            if draft is not None:
                draft["architecture"] = self.architecture_view_from_config(draft["config"])
            return draft

    def update_model_draft(self, draft_id: str, *, name: str | None = None, description: str | None = None, config: dict[str, Any] | None = None, layout: dict[str, Any] | None = None) -> dict[str, Any]:
        current = self.model_draft(draft_id)
        if current is None: raise KeyError(draft_id)
        if config is not None and not isinstance(config, dict): raise ValueError("Draft config must be an object")
        if layout is not None and not isinstance(layout, dict): raise ValueError("Draft layout must be an object")
        resolved = self._draft_config(config if config is not None else current["config"])
        next_layout = layout if layout is not None else current["layout"]
        revision, now = int(current["revision"]) + 1, _now()
        with self._connection() as db:
            db.execute("UPDATE model_drafts SET name=?,description=?,revision=?,config_json=?,layout_json=?,updated_at=? WHERE draft_id=?", (name.strip() if isinstance(name, str) and name.strip() else current["name"], description if description is not None else current["description"], revision, _json(resolved), _json(next_layout), now, draft_id))
            db.execute("INSERT INTO model_draft_revisions VALUES (?, ?, ?, ?, ?)", (draft_id, revision, _json(resolved), _json(next_layout), now))
        return self.model_draft(draft_id)  # type: ignore[return-value]

    def clone_model_draft(self, draft_id: str, *, name: str | None = None) -> dict[str, Any]:
        draft = self.model_draft(draft_id)
        if draft is None: raise KeyError(draft_id)
        return self.create_model_draft(name=name or f"{draft['name']} copy", description=draft["description"], config=draft["config"], layout=draft["layout"], source_artifact_id=draft.get("source_artifact_id"))

    def validate_model_draft(self, draft_id: str) -> dict[str, Any]:
        draft = self.model_draft(draft_id)
        if draft is None: raise KeyError(draft_id)
        try:
            config = self._draft_config(draft["config"])
            return {"valid": True, "errors": [], "warnings": [], "architecture": self.architecture_view_from_config(config)}
        except (TypeError, ValueError) as exc:
            return {"valid": False, "errors": [str(exc)], "warnings": []}

    def preview_model_draft(self, draft_id: str, *, dataset_id: str | None = None) -> dict[str, Any]:
        draft = self.model_draft(draft_id)
        if draft is None: raise KeyError(draft_id)
        config = self._draft_config(draft["config"])
        architecture = str((config.get("run") or {}).get("model") or (config.get("encoder") or {}).get("architecture") or "")
        result = {"draft_id": draft_id, "architecture": self.architecture_view_from_config(config), "config": config}
        if dataset_id:
            result["model_preview"] = self.model_preview(architecture=architecture, dataset_id=dataset_id, overrides=config)
        return result

    def plan_draft_training(self, draft_id: str, *, name: str, dataset_id: str, description: str = "", resources: dict[str, Any] | None = None, training_overrides: dict[str, Any] | None = None, initialization: dict[str, Any] | None = None) -> dict[str, Any]:
        """Seal a draft revision into one planned training specification.

        The draft itself remains editable; the generated TOML and initialization
        record are owned by the experiment, making later UI changes harmless.
        """
        if not name.strip(): raise ValueError("A training plan requires a name")
        draft, dataset = self.model_draft(draft_id), self.dataset(dataset_id)
        if draft is None: raise KeyError("Model draft was not found")
        if dataset is None: raise KeyError("Dataset was not found")
        if dataset["lifecycle"] != "frozen": raise ValueError("Training requires a frozen registered dataset")
        overrides, initialization = training_overrides or {}, initialization or {}
        if not isinstance(overrides, dict) or not isinstance(initialization, dict): raise ValueError("Training overrides and initialization must be objects")
        config = deep_merge(draft["config"], overrides)
        # The selected frozen dataset is authoritative for the classifier
        # output dimension. A draft can estimate capacity before data is
        # selected, but it cannot accidentally seal an incompatible class head.
        if str((config.get("run") or {}).get("task")) == "classification":
            try:
                with sqlite3.connect(dataset["path"]) as dataset_db:
                    class_count = int(dataset_db.execute("SELECT count(*) FROM classification_labels").fetchone()[0])
                if class_count > 0:
                    config.setdefault("data", {})["num_classes"] = class_count
            except sqlite3.DatabaseError:
                pass
        config = self._draft_config(config)
        source_id = initialization.get("source_artifact_id")
        if source_id:
            source = self.artifact(str(source_id))
            if source is None: raise KeyError("Initialization source artifact was not found")
            if source.get("task") and source.get("task") != (config.get("run") or {}).get("task"):
                raise ValueError("Initialization source task is incompatible with the draft task")
            source_config = self._json_file(Path(source["path"]) / "config" / "resolved.json") or {}
            source_shape = self._nested(source_config, "data", "input_shape")
            target_shape = self._nested(config, "data", "input_shape")
            if source_shape and target_shape and list(source_shape) != list(target_shape):
                raise ValueError("Initialization source input shape is incompatible with the draft")
        mode = str(initialization.get("mode", "scratch"))
        if mode not in {"scratch", "fine_tune", "transfer_encoder", "resume"}: raise ValueError("Unsupported initialization mode")
        if mode != "scratch" and not source_id: raise ValueError("A non-scratch initialization requires source_artifact_id")
        experiment_id, specification_id, now = str(uuid.uuid4()), str(uuid.uuid4()), _now()
        config_dir = self.artifact_root / "experiments" / experiment_id / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"draft-{draft_id[:8]}-revision-{draft['revision']}.toml"
        import tomli_w
        def toml_safe(value: Any) -> Any:
            if isinstance(value, dict): return {key: toml_safe(item) for key, item in value.items() if item is not None}
            if isinstance(value, list): return [toml_safe(item) for item in value if item is not None]
            return value
        config_path.write_text(tomli_w.dumps(toml_safe(config)), encoding="utf-8")
        parameters = self._assign_output_path("train", {"config": str(config_path), "input": dataset["path"], "dataset_id": dataset_id, "draft_id": draft_id, "draft_revision": draft["revision"], "initialization": initialization}, specification_id)
        digest = hashlib.sha256(_json({"parameters": parameters, "dataset_fingerprint": dataset["fingerprint_sha256"]}).encode()).hexdigest()
        plan = {"kind": "training", "dataset_id": dataset_id, "draft_id": draft_id, "draft_revision": draft["revision"], "resources": resources or {}, "initialization": initialization}
        with self._connection() as db:
            db.execute("INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (experiment_id, None, name.strip(), description, dataset_id, "expanded", _json(plan), now, now))
            db.execute("INSERT INTO run_specifications VALUES (?, ?, 1, ?, 'train', ?, ?, ?, 'planned', NULL, ?, ?)", (specification_id, experiment_id, name.strip(), _json(parameters), _json(resources or {}), digest, now, now))
        return self.experiment(experiment_id)  # type: ignore[return-value]

    def _comparison_group_detail(self, group: dict[str, Any], members: list[dict[str, Any]]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for member in members:
            artifact = self.artifact(member["artifact_id"])
            # Members should always resolve due to the foreign key. Keeping a
            # missing row visible makes imported/corrupt old databases diagnosable.
            if artifact is None:
                results.append({**member, "artifact": None, "timing": {"queue_seconds": None, "runtime_seconds": None}})
                continue
            results.append({**member, "artifact": self._artifact_result(artifact), "timing": self._artifact_runtime(member["artifact_id"])})

        artifact_results = [row["artifact"] for row in results if row["artifact"]]
        compatibility = self._comparison_check(artifact_results)
        baseline_id = group.get("baseline_artifact_id") or (results[0]["artifact_id"] if results else None)
        baseline = next((row for row in results if row["artifact_id"] == baseline_id), None)
        baseline_metrics = (baseline or {}).get("artifact", {}).get("metrics", {})
        all_metrics = sorted(set().union(*(set(row["artifact"]["metrics"]) for row in results if row["artifact"]))) if results else []
        matrix: list[dict[str, Any]] = []
        for row in results:
            artifact = row.get("artifact")
            metrics = artifact.get("metrics", {}) if artifact else {}
            values = {}
            for metric in all_metrics:
                value = metrics.get(metric)
                reference = baseline_metrics.get(metric)
                values[metric] = {
                    "value": value,
                    "delta": (value - reference) if isinstance(value, (int, float)) and isinstance(reference, (int, float)) else None,
                    "relative_delta": ((value - reference) / abs(reference)) if isinstance(value, (int, float)) and isinstance(reference, (int, float)) and reference != 0 else None,
                }
            matrix.append({"artifact_id": row["artifact_id"], "values": values,
                           "runtime_seconds": row["timing"]["runtime_seconds"],
                           "queue_seconds": row["timing"]["queue_seconds"]})
        return {
            **group, "members": results, "baseline_artifact_id": baseline_id,
            "compatibility": compatibility,
            "metrics": {"names": all_metrics, "baseline_artifact_id": baseline_id, "matrix": matrix},
        }

    def create_comparison_group(
        self, *, name: str, members: list[dict[str, Any]], description: str = "",
        relationship_label: str = "related", baseline_artifact_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist a manual relationship group without requiring protocol equality."""
        if not members:
            raise ValueError("Select at least one artifact")
        artifact_ids = [str(member.get("artifact_id", "")) for member in members]
        if not all(artifact_ids) or len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("Each comparison group member must reference a unique artifact")
        if any(self.artifact(artifact_id) is None for artifact_id in artifact_ids):
            raise KeyError("One or more artifacts were not found")
        if baseline_artifact_id is not None and baseline_artifact_id not in artifact_ids:
            raise ValueError("The baseline artifact must be a member of the comparison group")
        group_id, now = str(uuid.uuid4()), _now()
        with self._connection() as db:
            db.execute("INSERT INTO comparison_groups VALUES (?, ?, ?, ?, ?, ?, ?)", (
                group_id, name.strip() or "Model comparison", description, relationship_label.strip() or "related",
                baseline_artifact_id, now, now,
            ))
            for ordinal, member in enumerate(members):
                db.execute("INSERT INTO comparison_group_members VALUES (?, ?, ?, ?, ?)", (
                    group_id, artifact_ids[ordinal], ordinal, member.get("relationship_label"), str(member.get("note") or ""),
                ))
        return self.comparison_group(group_id)  # type: ignore[return-value]

    def _assign_output_path(self, action: str, parameters: dict[str, Any], specification_id: str) -> dict[str, Any]:
        """Assign paths owned by the orchestrator, never supplied by a UI client."""
        resolved = dict(parameters)
        if action == "train":
            resolved["runs_dir"] = str(self.runs_root)
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

    def specification_detail(self, specification_id: str) -> dict[str, Any]:
        specification = self.specification(specification_id)
        if specification is None:
            raise KeyError(specification_id)
        config_path = specification.get("parameters", {}).get("config")
        config = load_toml(config_path) if config_path else {}
        return {"specification": specification, "config": config}

    def update_planned_specification(self, specification_id: str, *, name: str | None = None, resources: dict[str, Any] | None = None, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Edit a saved run before it has been handed to compute."""
        with self._connection() as db:
            specification = self.specification(specification_id, connection=db)
            if specification is None:
                raise KeyError(specification_id)
            if specification["status"] != "planned":
                raise ValueError("Only runs that have not started can be edited. Create a new run to change a queued or completed execution.")
            next_name = specification["name"] if name is None else name.strip()
            if not next_name:
                raise ValueError("Run name cannot be empty")
            next_resources = dict(specification.get("resources") or {})
            if resources:
                if set(resources) - {"gpu_count"}:
                    raise ValueError("Only the GPU request can be changed after a run is saved")
                gpu_count = resources.get("gpu_count")
                if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 0:
                    raise ValueError("GPU request must be a non-negative integer")
                next_resources["gpu_count"] = gpu_count
            overrides = config_overrides or {}
            allowed_sections = {"run", "data", "model", "training", "augmentation", "callbacks", "output", "architecture", "input", "encoder", "stem", "normalization", "pooling", "image_embedding", "metadata", "fusion", "classifier", "preprocessing"}
            if set(overrides) - allowed_sections or any(not isinstance(value, dict) for value in overrides.values()):
                raise ValueError("Configuration changes must use known configuration sections")
            if overrides:
                config_path = specification.get("parameters", {}).get("config")
                if not config_path:
                    raise ValueError("This run has no editable training configuration")
                import tomli_w
                Path(config_path).write_text(tomli_w.dumps(deep_merge(load_toml(config_path), overrides)), encoding="utf-8")
            db.execute("UPDATE run_specifications SET name=?, resources_json=?, updated_at=? WHERE specification_id=?", (next_name, _json(next_resources), _now(), specification_id))
        return self.specification(specification_id)  # type: ignore[return-value]

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
        result_job = self.job(job_id)  # type: ignore[assignment]
        queued_run_id = result_job.get("queued_run_id") if result_job else None
        if queued_run_id:
            queue_statuses = {
                "dispatching": "starting", "submitted": "submitted", "queued": "submitted",
                "running": "running", "paused": "paused", "validating": "validating_artifact", "indexed": "complete",
                "succeeded": "complete", "failed": "failed", "cancelled": "cancelled",
                "dispatch_failed": "waiting_for_resources", "artifact_invalid": "failed",
            }
            with self._connection() as db:
                db.execute(
                    "UPDATE queued_runs SET status=?, failure_reason=?, updated_at=? WHERE queued_run_id=?",
                    (queue_statuses.get(str(result_job.get("status")), str(result_job.get("status"))), result_job.get("error"), _now(), queued_run_id),
                )
        return result_job  # type: ignore[return-value]

    def reconcile_active_jobs(self) -> list[dict[str, Any]]:
        active = [job for job in self.jobs() if job["status"] in {"dispatching", "submitted", "queued", "running", "paused", "validating"}]
        results = []
        for job in active:
            try:
                results.append(self.reconcile_job(job["job_id"]))
            except RuntimeError:
                results.append(job)
        return results

    def reset_stuck_jobs(self, *, endpoint_id: str | None = None) -> dict[str, Any]:
        """Reset only jobs proven absent from their compute endpoint.

        A transient network failure is never treated as proof a job is gone.
        The original dispatch record is retained as ``stale`` for audit, and a
        linked validated queue row is returned to ``ready`` so the user must
        explicitly authorize its next dispatch attempt.
        """
        active_states = {"dispatching", "submitted", "queued", "running", "paused", "validating"}
        candidates = [job for job in self.jobs() if job.get("status") in active_states]
        if endpoint_id:
            endpoint = self.compute_endpoint(endpoint_id)
            if endpoint is None:
                raise KeyError("Compute endpoint was not found")
            candidates = [job for job in candidates if job.get("oracle_serve_url", "").rstrip("/") == endpoint["base_url"].rstrip("/")]
        reset: list[str] = []
        still_active: list[str] = []
        unavailable: list[dict[str, str]] = []
        for job in candidates:
            try:
                remote = self._request(job["oracle_serve_url"], "GET", f"/compute/jobs/{job['job_id']}")
            except RuntimeError as exc:
                message = str(exc)
                if "returned 404" not in message:
                    unavailable.append({"job_id": job["job_id"], "reason": message})
                    continue
                now = _now()
                reason = "Compute endpoint confirmed this dispatch no longer exists; reset for an explicit retry."
                with self._connection() as db:
                    db.execute("UPDATE jobs SET status='stale', remote_status='missing', error=?, completed_at=?, updated_at=? WHERE job_id=?", (reason, now, now, job["job_id"]))
                    db.execute("UPDATE run_specifications SET status='planned', updated_at=? WHERE specification_id=?", (now, job["specification_id"]))
                    queue = db.execute("SELECT queued_run_id,preflight_status FROM queued_runs WHERE specification_id=?", (job["specification_id"],)).fetchone()
                    if queue is not None:
                        next_status = "ready" if queue["preflight_status"] == "valid" else "needs_attention"
                        db.execute("UPDATE queued_runs SET status=?, start_authorized=0, failure_reason=?, updated_at=? WHERE queued_run_id=?", (next_status, reason, now, queue["queued_run_id"]))
                self._record_event(job["job_id"], "stale_reset", reason, {"remote_status": "missing"})
                reset.append(job["job_id"])
                continue
            if remote.get("status") in active_states:
                still_active.append(job["job_id"])
            else:
                # A completed remote record follows normal reconciliation;
                # this endpoint is only a recovery tool for missing records.
                self.reconcile_job(job["job_id"])
        return {"reset": reset, "still_active": still_active, "unavailable": unavailable, "endpoint_id": endpoint_id}

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
    def _request(
        base: str,
        method: str,
        endpoint: str,
        body: dict[str, Any] | None = None,
        *,
        timeout_seconds: float = 15,
    ) -> dict[str, Any]:
        request = urllib.request.Request(base.rstrip("/") + endpoint, method=method)
        if body is not None:
            request.data = _json(body).encode()
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
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

    def job_training_status(self, job_id: str) -> dict[str, Any]:
        """Return the worker's live training projection without exposing paths.

        A live snapshot is intentionally best-effort: a job may be queued, run
        on an older worker, or briefly lose contact with its compute service.
        The web client can therefore render a useful, explicit unavailable
        state instead of treating a transient worker failure as a failed job.
        """
        job = self.job(job_id)
        if job is None:
            raise KeyError(job_id)

        fallback = {
            "job_id": job_id,
            "job_status": job["status"],
            "controls": (
                ["pause", "cancel"] if job["status"] == "running"
                else ["resume", "cancel"] if job["status"] == "paused"
                else ["cancel"] if job["status"] in {"dispatching", "submitted", "queued"}
                else []
            ),
            "available": False,
            "stale": True,
            "phase": job.get("remote_status") or job["status"],
            "message": "Live training status is not available yet.",
        }
        try:
            remote = self._request(job["oracle_serve_url"], "GET", f"/compute/jobs/{job_id}/training-status")
        except RuntimeError as exc:
            return {**fallback, "message": f"Live training status is temporarily unavailable: {exc}"}

        # A compute service controls the metric shape, but not the identity or
        # lifecycle information returned to the browser.  This prevents a
        # malformed/old worker response from making the local job ambiguous.
        if not isinstance(remote, dict):
            return {**fallback, "message": "Compute returned an invalid live training status response."}
        available = bool(remote.get("available", True))
        return {
            **remote,
            "job_id": job_id,
            "job_status": job["status"],
            "controls": fallback["controls"],
            "available": available,
            "stale": bool(remote.get("stale", not available)),
            "message": remote.get("message") or (None if available else fallback["message"]),
        }
    def comparisons(self) -> list[dict[str, Any]]: return self._many("SELECT * FROM comparisons ORDER BY created_at DESC")
    def comparison_groups(self) -> list[dict[str, Any]]:
        with self._connection() as db:
            groups = [_row(item) for item in db.execute("SELECT * FROM comparison_groups ORDER BY created_at DESC").fetchall()]
            counts = {row["comparison_group_id"]: row["member_count"] for row in db.execute(
                "SELECT comparison_group_id, COUNT(*) AS member_count FROM comparison_group_members GROUP BY comparison_group_id"
            ).fetchall()}
        return [{**group, "member_count": counts.get(group["comparison_group_id"], 0)} for group in groups if group]
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
    def comparison_group(self, value: str) -> dict[str, Any] | None:
        with self._connection() as db:
            group = _row(db.execute("SELECT * FROM comparison_groups WHERE comparison_group_id=?", (value,)).fetchone())
            if group is None:
                return None
            members = [_row(row) for row in db.execute(
                "SELECT * FROM comparison_group_members WHERE comparison_group_id=? ORDER BY ordinal", (value,)
            ).fetchall()]
        return self._comparison_group_detail(group, [member for member in members if member])
