from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oracle_data_contracts.artifacts.layout import RunLayout


RUN_ARTIFACT_SCHEMA_NAME = "oracle_builder_model_run"
RUN_ARTIFACT_SCHEMA_VERSION = "1.0.0"
RUN_LIFECYCLES = {"working", "sealed"}
RUN_STATUSES = {"running", "interrupted", "complete", "failed"}
ARTIFACT_TYPES = {"model_run", "model_product"}
_INVENTORY_EXCLUDES = {"artifact.json", "checksums.sha256"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _portable_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    portable = json.loads(json.dumps(config, default=str))
    runtime = {"paths": portable.pop("paths", {})}
    portable.pop("_split_manifest", None)
    return portable, runtime


def _model_contract(config: dict[str, Any]) -> dict[str, Any]:
    external = config.get("external_model_contract")
    if external is not None:
        return json.loads(json.dumps(external, default=str))
    task = config["run"]["task"]
    outputs = (
        {
            "primary": "class_probabilities",
            "logits": "class_logits",
            "class_count": config["data"].get("num_classes"),
            "labels": config.get("dataset", {}).get("labels", []),
            "embedding": "identity_embedding",
            "embedding_dimension": config.get("model", {}).get("embedding_dim", 256),
            "embedding_normalized": config.get("model", {}).get(
                "normalize_embeddings", True
            ),
        }
        if task == "classification"
        else {
            "primary": config.get("training", {}).get(
                "segmentation_target", "validated_mask"
            ),
            "logits": "segmentation_logits",
            "probability_map": "segmentation_probabilities",
            "shape": [None, *config["data"].get("output_shape", [])],
            "activation": config.get("model", {}).get("final_activation", "sigmoid"),
            "threshold": config.get("evaluation", {}).get(
                "segmentation_threshold", 0.5
            ),
        }
    )
    return {
        "task": task,
        "architecture": config["run"]["model"],
        "variant": config.get("model", {}).get("variant"),
        "input": {
            "shape": [None, *config["data"]["input_shape"]],
            "dtype": "float32",
            "preprocessing": config.get("preprocessing", {}),
        },
        "outputs": outputs,
    }


def _producer() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("oracle-builder")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"name": "oracle-builder", "version": version}


def _readme(manifest: dict[str, Any]) -> str:
    dataset = manifest["dataset"]
    model = manifest["model"]
    artifact_label = (
        "model product"
        if manifest.get("artifact_type", "model_run") == "model_product"
        else "model-run artifact"
    )
    return f"""# {manifest['name']}

Oracle Builder {artifact_label} `{manifest['artifact_id']}`.

- Schema: `{manifest['artifact_schema']['name']}` `{manifest['artifact_schema']['version']}`
- Type: `{manifest.get('artifact_type', 'model_run')}`
- Run ID: `{manifest['run_id']}`
- Status: `{manifest['status']}`
- Lifecycle: `{manifest['lifecycle']}`
- Task: `{model['task']}`
- Architecture: `{model['architecture']}`
- Dataset ID: `{dataset.get('dataset_id', '')}`
- Dataset fingerprint: `{dataset.get('fingerprint_sha256', '')}`

`artifact.json` is the authoritative manifest. `checksums.sha256` and its
matching inventory are written when the artifact is sealed. Paths inside the
manifest are relative to this directory.
"""


def _model_card(manifest: dict[str, Any]) -> str:
    model = manifest["model"]
    dataset = manifest["dataset"]
    return f"""# Model card: {manifest['name']}

## Identity

- Artifact ID: `{manifest['artifact_id']}`
- Run ID: `{manifest['run_id']}`
- Architecture: `{model['architecture']}`
- Task: `{model['task']}`

## Training data

- Dataset ID: `{dataset.get('dataset_id', '')}`
- Dataset type: `{dataset.get('dataset_type', '')}`
- Dataset version: `{dataset.get('version') or ''}`
- Semantic fingerprint: `{dataset.get('fingerprint_sha256', '')}`
- Usage constraints: `{json.dumps(dataset.get('usage', {}), sort_keys=True)}`

The dataset itself is referenced by stable identity and fingerprint and is not
duplicated in this model artifact.

## Input and output contract

```json
{json.dumps(model, indent=2, sort_keys=True)}
```

## Validation and limitations

Machine-readable evaluation results are stored under `evaluation/`. Review the
dataset metadata, evaluation split, class balance or mask quality, and known
out-of-distribution conditions before deployment. This generated card does not
assert fitness for an unstated scientific or operational use.
"""


def create_run_artifact(
    run_dir: str | Path,
    *,
    run_id: str,
    name: str,
    config: dict[str, Any],
    source_config: str | Path,
    artifact_type: str = "model_run",
) -> dict[str, Any]:
    if artifact_type not in ARTIFACT_TYPES:
        raise ValueError(f"Unsupported artifact type: {artifact_type}")
    layout = RunLayout(run_dir)
    layout.create_directories()
    if layout.manifest.exists():
        raise FileExistsError(f"Run artifact already initialized: {layout.manifest}")
    portable, runtime = _portable_config(config)
    shutil.copy2(source_config, layout.source_config)
    _write_json(layout.resolved_config, portable)
    _write_json(layout.runtime, runtime)
    now = _utc_now()
    manifest = {
        "artifact_schema": {
            "name": RUN_ARTIFACT_SCHEMA_NAME,
            "version": RUN_ARTIFACT_SCHEMA_VERSION,
        },
        "artifact_type": artifact_type,
        "producer": _producer(),
        "artifact_id": str(uuid.uuid4()),
        "run_id": str(uuid.UUID(run_id)),
        "name": name,
        "status": "running",
        "lifecycle": "working",
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
        "dataset": dict(config.get("dataset", {})),
        "product": dict(config.get("product", {})),
        "model": _model_contract(config),
        "paths": {
            "readme": "README.md",
            "model_card": "MODEL_CARD.md",
            "source_config": "config/source.toml",
            "resolved_config": "config/resolved.json",
            "split_manifest": "protocol/splits.json",
            "runtime_provenance": "provenance/runtime.json",
            "environment": "provenance/environment.json",
            "requirements": "provenance/requirements.txt",
            "training_log": "logs/training.sqlite",
            "training_metrics": "metrics",
            "model": "model",
            "evaluation": "evaluation",
            "predictions": "predictions",
            "figures": "figures",
        },
        "summary": {},
        "inventory": [],
        "fingerprint_sha256": None,
    }
    _write_json(layout.manifest, manifest)
    layout.readme.write_text(_readme(manifest), encoding="utf-8")
    layout.model_card.write_text(_model_card(manifest), encoding="utf-8")
    return manifest


def read_run_manifest(run_dir: str | Path) -> dict[str, Any]:
    layout = RunLayout(run_dir)
    if not layout.manifest.exists():
        raise ValueError(f"Not an Oracle Builder V1 run artifact: {layout.root}")
    return json.loads(layout.manifest.read_text(encoding="utf-8"))


def read_run_config(run_dir: str | Path) -> dict[str, Any]:
    layout = RunLayout(run_dir)
    if layout.manifest.exists():
        config = json.loads(layout.resolved_config.read_text(encoding="utf-8"))
        if layout.split_manifest.exists():
            from oracle_data_contracts.artifacts.splits import (
                attach_split_manifest,
                read_split_manifest,
            )

            attach_split_manifest(config, read_split_manifest(layout.root))
        return config
    legacy = layout.root / "resolved_config.json"
    if legacy.exists():
        return json.loads(legacy.read_text(encoding="utf-8"))
    raise FileNotFoundError(layout.resolved_config)


def read_run_runtime(run_dir: str | Path) -> dict[str, Any]:
    """Return machine-local provenance such as the original dataset path."""
    layout = RunLayout(run_dir)
    if not layout.runtime.exists():
        raise FileNotFoundError(layout.runtime)
    return json.loads(layout.runtime.read_text(encoding="utf-8"))


def write_run_config(run_dir: str | Path, config: dict[str, Any]) -> None:
    layout = RunLayout(run_dir)
    manifest = read_run_manifest(layout.root)
    if manifest["lifecycle"] != "working":
        raise ValueError("Run artifact is sealed; reopen it before changing config")
    portable, runtime = _portable_config(config)
    _write_json(layout.resolved_config, portable)
    _write_json(layout.runtime, runtime)
    manifest["model"] = _model_contract(config)
    manifest["dataset"] = dict(config.get("dataset", {}))
    manifest["updated_at"] = _utc_now()
    _write_json(layout.manifest, manifest)
    layout.model_card.write_text(_model_card(manifest), encoding="utf-8")


def update_run_artifact(
    run_dir: str | Path,
    *,
    status: str | None = None,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    layout = RunLayout(run_dir)
    manifest = read_run_manifest(layout.root)
    if manifest["lifecycle"] != "working":
        raise ValueError("Run artifact is sealed; reopen it before updating")
    if status is not None:
        if status not in RUN_STATUSES:
            raise ValueError(f"Unsupported run status: {status}")
        manifest["status"] = status
        if status in {"complete", "failed"}:
            manifest["completed_at"] = _utc_now()
        elif status in {"running", "interrupted"}:
            manifest["completed_at"] = None
        if status == "running":
            manifest.pop("error", None)
    if summary is not None:
        manifest["summary"] = summary
    if error is not None:
        manifest["error"] = error
    manifest["updated_at"] = _utc_now()
    _write_json(layout.manifest, manifest)
    return manifest


def _role(relative: str) -> str:
    top = relative.split("/", 1)[0]
    return {
        "config": "configuration",
        "protocol": "training_protocol",
        "provenance": "provenance",
        "logs": "execution_log",
        "metrics": "training_metrics",
        "model": "model",
        "evaluation": "evaluation",
        "predictions": "predictions",
        "figures": "visualization",
    }.get(top, "documentation")


def _inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in _INVENTORY_EXCLUDES:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append(
            {
                "path": relative,
                "sha256": digest.hexdigest(),
                "size_bytes": path.stat().st_size,
                "role": _role(relative),
            }
        )
    return rows


def _completed_required_paths(layout: RunLayout) -> tuple[Path, ...]:
    return (
        layout.environment,
        layout.requirements,
        layout.training_log,
        layout.metrics_json,
        layout.split_manifest,
        layout.model / "model_manifest.json",
        layout.model / "load_test_report.json",
    )


def _fingerprint(
    manifest: dict[str, Any],
    inventory: list[dict[str, Any]],
    *,
    include_product_identity: bool = True,
) -> str:
    semantic = {
        "artifact_schema": manifest["artifact_schema"],
        "producer": manifest.get("producer"),
        "artifact_id": manifest["artifact_id"],
        "run_id": manifest["run_id"],
        "name": manifest["name"],
        "status": manifest["status"],
        "dataset": manifest["dataset"],
        "model": manifest["model"],
        "inventory": inventory,
    }
    if include_product_identity:
        # New artifacts include product identity fields in their fingerprint.
        if "artifact_type" in manifest:
            semantic["artifact_type"] = manifest["artifact_type"]
        if "product" in manifest:
            semantic["product"] = manifest["product"]
    encoded = json.dumps(
        semantic, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seal_run_artifact(run_dir: str | Path) -> dict[str, Any]:
    layout = RunLayout(run_dir)
    manifest = read_run_manifest(layout.root)
    if manifest["status"] == "running":
        raise ValueError("Cannot seal a run while its status is still running")
    if manifest["status"] == "complete":
        missing = [
            path.relative_to(layout.root).as_posix()
            for path in _completed_required_paths(layout)
            if not path.exists()
        ]
        if missing:
            raise ValueError(
                "Cannot seal an incomplete completed-run artifact; missing: "
                + ", ".join(missing)
            )
    inventory = _inventory(layout.root)
    layout.checksums.write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in inventory),
        encoding="utf-8",
    )
    manifest["inventory"] = inventory
    manifest["fingerprint_sha256"] = _fingerprint(manifest, inventory)
    manifest["lifecycle"] = "sealed"
    manifest["updated_at"] = _utc_now()
    _write_json(layout.manifest, manifest)
    layout.readme.write_text(_readme(manifest), encoding="utf-8")
    # README changed after initial inventory; recalculate once to seal exact bytes.
    inventory = _inventory(layout.root)
    layout.checksums.write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in inventory),
        encoding="utf-8",
    )
    manifest["inventory"] = inventory
    manifest["fingerprint_sha256"] = _fingerprint(manifest, inventory)
    _write_json(layout.manifest, manifest)
    return manifest


def reopen_run_artifact(
    run_dir: str | Path, *, reason: str | None = None
) -> dict[str, Any]:
    layout = RunLayout(run_dir)
    manifest = read_run_manifest(layout.root)
    manifest["lifecycle"] = "working"
    manifest["inventory"] = []
    manifest["fingerprint_sha256"] = None
    manifest["updated_at"] = _utc_now()
    manifest.setdefault("events", []).append(
        {"type": "reopened", "timestamp": _utc_now(), "reason": reason}
    )
    layout.checksums.unlink(missing_ok=True)
    _write_json(layout.manifest, manifest)
    return manifest


def validate_run_artifact(run_dir: str | Path) -> dict[str, Any]:
    layout = RunLayout(run_dir)
    errors: list[str] = []
    warnings: list[str] = []
    try:
        manifest = read_run_manifest(layout.root)
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": []}
    schema = manifest.get("artifact_schema", {})
    if schema != {
        "name": RUN_ARTIFACT_SCHEMA_NAME,
        "version": RUN_ARTIFACT_SCHEMA_VERSION,
    }:
        errors.append(f"Unsupported run artifact schema: {schema}")
    # Artifacts produced before the explicit type field was introduced are
    # ordinary model runs.
    if manifest.get("artifact_type", "model_run") not in ARTIFACT_TYPES:
        errors.append("Unsupported artifact type")
    for key in ("artifact_id", "run_id"):
        try:
            uuid.UUID(manifest[key])
        except Exception:
            errors.append(f"{key} is not a valid UUID")
    if manifest.get("lifecycle") not in RUN_LIFECYCLES:
        errors.append("Invalid lifecycle")
    if manifest.get("status") not in RUN_STATUSES:
        errors.append("Invalid status")
    for required in (
        layout.source_config,
        layout.resolved_config,
        layout.runtime,
        layout.model_card,
    ):
        if not required.exists():
            errors.append(f"Required file is missing: {required.relative_to(layout.root)}")
    if layout.split_manifest.exists():
        try:
            from oracle_data_contracts.artifacts.splits import read_split_manifest

            split_manifest = read_split_manifest(layout.root)
            split_dataset = split_manifest.get("dataset", {})
            run_dataset = manifest.get("dataset", {})
            for key in ("dataset_id", "revision_id", "fingerprint_sha256"):
                if split_dataset.get(key) != run_dataset.get(key):
                    errors.append(
                        f"Split manifest dataset {key} does not match run artifact"
                    )
        except Exception as exc:
            errors.append(str(exc))
    if manifest.get("status") == "complete":
        historical_split_manifest = any(
            row.get("path") == "protocol/splits.json"
            for row in manifest.get("inventory", [])
        )
        for required in _completed_required_paths(layout):
            if not required.exists():
                relative = required.relative_to(layout.root).as_posix()
                if relative == "protocol/splits.json" and not historical_split_manifest:
                    warnings.append(
                        "Legacy run has no item-level split protocol"
                    )
                else:
                    errors.append(f"Completed run is missing: {relative}")
    if manifest.get("lifecycle") == "sealed":
        if not layout.checksums.exists():
            errors.append("Sealed artifact lacks checksums.sha256")
        observed = _inventory(layout.root)
        expected = manifest.get("inventory", [])
        if observed != expected:
            expected_by_path = {row["path"]: row for row in expected}
            observed_by_path = {row["path"]: row for row in observed}
            for path in sorted(set(expected_by_path) | set(observed_by_path)):
                if expected_by_path.get(path) != observed_by_path.get(path):
                    errors.append(f"Inventory mismatch: {path}")
        if _fingerprint(manifest, expected) != manifest.get("fingerprint_sha256"):
            # Early V1 runs received artifact_type during the product-schema
            # transition, but their already-sealed fingerprints correctly used
            # the former semantic payload. Accept that form only when its
            # recorded inventory has already passed verification above.
            legacy_fingerprint = _fingerprint(
                manifest, expected, include_product_identity=False
            )
            if legacy_fingerprint == manifest.get("fingerprint_sha256"):
                warnings.append(
                    "Artifact uses the verified pre-product fingerprint form"
                )
            else:
                errors.append("Artifact fingerprint mismatch")
    else:
        warnings.append("Run artifact is working and has not been sealed")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "artifact_id": manifest.get("artifact_id"),
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "lifecycle": manifest.get("lifecycle"),
        "fingerprint_sha256": manifest.get("fingerprint_sha256"),
    }


def pack_run_artifact(run_dir: str | Path, output: str | Path) -> dict[str, Any]:
    layout = RunLayout(run_dir)
    report = validate_run_artifact(layout.root)
    if not report["valid"] or report["lifecycle"] != "sealed":
        raise ValueError("Only a valid sealed run artifact can be packed")
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(value for value in layout.root.rglob("*") if value.is_file()):
            relative = path.relative_to(layout.root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            with path.open("rb") as source, archive.open(info, "w") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    package_digest = hashlib.sha256()
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            package_digest.update(chunk)
    return {
        "input": str(layout.root),
        "output": str(destination),
        "artifact_id": report["artifact_id"],
        "artifact_fingerprint_sha256": report["fingerprint_sha256"],
        "package_sha256": package_digest.hexdigest(),
        "size_bytes": destination.stat().st_size,
    }


def unpack_run_artifact(package: str | Path, output: str | Path) -> dict[str, Any]:
    source = Path(package).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    try:
        with zipfile.ZipFile(source) as archive:
            root = destination.resolve()
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise ValueError(f"Unsafe package path: {member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source_handle, target.open(
                        "wb"
                    ) as target_handle:
                        shutil.copyfileobj(
                            source_handle, target_handle, length=1024 * 1024
                        )
        report = validate_run_artifact(destination)
        if not report["valid"]:
            raise ValueError(
                "Unpacked artifact validation failed: " + "; ".join(report["errors"])
            )
        return {"input": str(source), "output": str(destination), **report}
    except Exception:
        shutil.rmtree(destination)
        raise


def migrate_legacy_run(
    source_run: str | Path, output: str | Path
) -> dict[str, Any]:
    """Copy a pre-V1 run into the V1 layout without modifying its source."""
    source = Path(source_run).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    config_path = source / "resolved_config.json"
    source_config = source / "run_config.toml"
    if not config_path.exists() or not source_config.exists():
        raise ValueError(
            "Legacy migration requires resolved_config.json and run_config.toml"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata_path = source / "run_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    run_id = (
        metadata.get("run_id")
        or config.get("run", {}).get("run_id")
        or str(uuid.uuid4())
    )
    try:
        run_id = str(uuid.UUID(run_id))
    except (TypeError, ValueError):
        run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"legacy-run:{run_id}"))
    name = (
        metadata.get("run_name")
        or config.get("run", {}).get("run_name")
        or source.name
    )
    config.setdefault("run", {})["run_id"] = run_id
    config["run"]["run_name"] = name
    create_run_artifact(
        destination,
        run_id=run_id,
        name=name,
        config=config,
        source_config=source_config,
    )
    layout = RunLayout(destination)
    manifest = read_run_manifest(destination)
    config["artifact"] = {
        "artifact_id": manifest["artifact_id"],
        "schema_name": RUN_ARTIFACT_SCHEMA_NAME,
        "schema_version": RUN_ARTIFACT_SCHEMA_VERSION,
    }
    write_run_config(destination, config)
    from oracle_data_contracts.artifacts.splits import (
        create_unavailable_split_manifest,
    )

    create_unavailable_split_manifest(
        destination,
        config,
        reason=(
            "The source predates the V1 split-manifest contract; exact "
            "per-item train/validation/test assignments were not preserved."
        ),
    )

    file_mappings = {
        "environment.json": layout.environment,
        "requirements_freeze.txt": layout.requirements,
        "distribution.json": layout.distribution,
        "training_log.sqlite": layout.training_log,
        "metrics.csv": layout.metrics_csv,
        "metrics.json": layout.metrics_json,
    }
    for legacy_name, target in file_mappings.items():
        legacy = source / legacy_name
        if legacy.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy, target)
    for directory, target in (
        ("model", layout.model),
        ("evaluation", layout.evaluation),
        ("predictions", layout.predictions),
        ("figures", layout.figures),
    ):
        legacy = source / directory
        if legacy.exists():
            shutil.copytree(legacy, target, dirs_exist_ok=True)
    legacy_pretraining = source / "pretraining"
    if legacy_pretraining.exists():
        for path in legacy_pretraining.iterdir():
            target_root = (
                layout.pretraining_model
                if "weights" in path.name or path.suffix in {".keras", ".h5"}
                else layout.pretraining_metrics
            )
            target_root.mkdir(parents=True, exist_ok=True)
            target = target_root / path.name
            if path.is_dir():
                shutil.copytree(path, target)
            else:
                shutil.copy2(path, target)

    known = {
        "resolved_config.json",
        "run_config.toml",
        "run_metadata.json",
        *file_mappings,
        "model",
        "evaluation",
        "predictions",
        "figures",
        "pretraining",
    }
    attachments = destination / "attachments"
    for path in sorted(source.iterdir()):
        if path.name in known or path.name == ".DS_Store":
            continue
        attachments.mkdir(parents=True, exist_ok=True)
        target = attachments / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)

    model_manifest = layout.model / "model_manifest.json"
    if not model_manifest.exists():
        formats = []
        for filename, format_name, role in (
            ("final.keras", "keras_v3", "preferred_full_model"),
            (
                "weights.weights.h5",
                "keras_weights_hdf5",
                "rebuild_with_oracle_builder",
            ),
            (
                "export_savedmodel",
                "tensorflow_saved_model",
                "portable_inference",
            ),
        ):
            if (layout.model / filename).exists():
                formats.append(
                    {"format": format_name, "path": filename, "role": role}
                )
        _write_json(
            model_manifest,
            {
                "schema_name": "oracle_builder_model",
                "schema_version": "1.0.0",
                "artifact_id": manifest["artifact_id"],
                "run_id": run_id,
                "task": config["run"].get("task"),
                "architecture": config["run"].get("model"),
                "input": _model_contract(config)["input"],
                "outputs": _model_contract(config)["outputs"],
                "formats": formats,
                "migration": {"source": source.name, "legacy_layout": True},
            },
        )

    status = metadata.get("status", "complete")
    if status not in RUN_STATUSES:
        status = "failed"
    summary = {
        "evaluation": metadata.get("evaluation_summary"),
        "validation_threshold_analysis": metadata.get(
            "validation_threshold_analysis"
        ),
        "migration": {
            "source_directory_name": source.name,
            "migrated_at": _utc_now(),
        },
    }
    update_run_artifact(
        destination,
        status=status,
        summary=summary,
        error=metadata.get("error"),
    )
    if status != "running":
        seal_run_artifact(destination)
    report = validate_run_artifact(destination)
    return {
        "source": str(source),
        "output": str(destination),
        "artifact_id": manifest["artifact_id"],
        **report,
    }
