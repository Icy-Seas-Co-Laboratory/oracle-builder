from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oracle_data_contracts.artifacts.layout import RunLayout
from oracle_data_contracts.datasets.schema import dataset_fingerprint, read_dataset_info


SPLIT_MANIFEST_SCHEMA_NAME = "oracle_builder_split_manifest"
SPLIT_MANIFEST_SCHEMA_VERSION = "1.1.0"
_SUPPORTED_SCHEMA_VERSIONS = {"1.0.0", SPLIT_MANIFEST_SCHEMA_VERSION}
SPLIT_NAMES = ("train", "validation", "test")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manifest_fingerprint(manifest: dict[str, Any]) -> str:
    semantic = {key: value for key, value in manifest.items() if key != "fingerprint_sha256"}
    encoded = json.dumps(
        semantic, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assignment_digest(assignments: list[dict[str, str]]) -> str:
    """Return the stable identity of a split assignment.

    A run-manifest fingerprint contains run-local bookkeeping such as creation
    time and its random UUID.  This digest deliberately does not, so callers
    can compare split identity across independently created manifests.
    """
    canonical = [
        {"item_id": str(row["item_id"]), "split": str(row["split"])}
        for row in sorted(assignments, key=lambda row: str(row["item_id"]))
    ]
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _assignment_rows(
    item_ids: list[str], *, seed: int, validation: float, test: float
) -> list[dict[str, str]]:
    """Return deterministic assignments with exact rounded split counts."""
    ranked = sorted(
        item_ids,
        key=lambda item_id: (
            hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).digest(),
            item_id,
        ),
    )
    n_test = int(round(len(ranked) * test))
    n_validation = int(round(len(ranked) * validation))
    if n_test + n_validation >= len(ranked):
        overflow = n_test + n_validation - len(ranked) + 1
        reduce_validation = min(n_validation, overflow)
        n_validation -= reduce_validation
        n_test -= overflow - reduce_validation
    assigned: dict[str, str] = {}
    for index, item_id in enumerate(ranked):
        if index < n_test:
            assigned[item_id] = "test"
        elif index < n_test + n_validation:
            assigned[item_id] = "validation"
        else:
            assigned[item_id] = "train"
    return [
        {"item_id": item_id, "split": assigned[item_id]}
        for item_id in sorted(item_ids)
    ]


def _enabled_splits(*, validation: float, test: float) -> tuple[str, ...]:
    enabled = ["train"]
    if validation > 0:
        enabled.append("validation")
    if test > 0:
        enabled.append("test")
    return tuple(enabled)


def _stable_rank(seed: int, item_id: str) -> tuple[bytes, str]:
    return hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).digest(), item_id


def _split_weights(*, validation: float, test: float, enabled: tuple[str, ...]) -> dict[str, float]:
    weights = {"train": 1.0 - validation - test, "validation": validation, "test": test}
    return {split: weights[split] for split in enabled}


def _allocate_counts(
    total: int,
    *,
    enabled: tuple[str, ...],
    weights: dict[str, float],
    minimum: int,
) -> dict[str, int]:
    required = minimum * len(enabled)
    if total < required:
        raise ValueError(
            f"requires at least {required} samples for minimum class coverage "
            f"across {len(enabled)} enabled splits, but has {total}"
        )
    counts = {split: minimum for split in enabled}
    remaining = total - required
    ideal = {split: remaining * weights[split] / sum(weights.values()) for split in enabled}
    for split in enabled:
        counts[split] += int(ideal[split])
    remainder = remaining - sum(int(ideal[split]) for split in enabled)
    # The ordering is deterministic even on ties.  It has no scientific
    # meaning; the item hash controls which individual samples are selected.
    for split in sorted(enabled, key=lambda name: (-(ideal[name] % 1), name))[:remainder]:
        counts[split] += 1
    return counts


def _class_stratified_assignments(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    validation: float,
    test: float,
    minimum: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    enabled = _enabled_splits(validation=validation, test=test)
    weights = _split_weights(validation=validation, test=test, enabled=enabled)
    classes: dict[str, list[str]] = {}
    unlabeled: list[str] = []
    for row in rows:
        label = row.get("label")
        if label is None:
            unlabeled.append(str(row["item_id"]))
        else:
            classes.setdefault(str(label), []).append(str(row["item_id"]))
    if unlabeled:
        raise ValueError(
            "class_stratified splitting requires an effective class label for every "
            f"training item; {len(unlabeled)} unlabeled (for example: {', '.join(sorted(unlabeled)[:3])})"
        )
    if not classes:
        raise ValueError("class_stratified splitting requires at least one labeled item")
    assignments: list[dict[str, str]] = []
    class_counts: dict[str, dict[str, int]] = {}
    infeasible: dict[str, str] = {}
    for label, item_ids in sorted(classes.items()):
        try:
            allocated = _allocate_counts(
                len(item_ids), enabled=enabled, weights=weights, minimum=minimum
            )
        except ValueError as exc:
            infeasible[label] = str(exc)
            continue
        ranked = sorted(item_ids, key=lambda item_id: _stable_rank(seed, item_id))
        offset = 0
        for split in enabled:
            selected = ranked[offset : offset + allocated[split]]
            assignments.extend({"item_id": item_id, "split": split} for item_id in selected)
            offset += allocated[split]
        class_counts[label] = {split: allocated.get(split, 0) for split in SPLIT_NAMES}
    if infeasible:
        details = "; ".join(f"class {label}: {reason}" for label, reason in infeasible.items())
        raise ValueError(
            "Class-coverage split is infeasible: " + details + ". Reduce enabled splits or "
            "data.split_minimum_per_class, or add labeled examples."
        )
    return sorted(assignments, key=lambda row: row["item_id"]), {
        "enabled_splits": list(enabled),
        "minimum_per_class": minimum,
        "class_counts": class_counts,
        "unlabeled_count": 0,
        "missing_coverage": {},
        "valid": True,
    }


def _group_value(metadata: dict[str, Any], key: str, item_id: str) -> str:
    value: Any = metadata
    for component in key.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(
                f"data.split_group_metadata_key={key!r} is missing for item {item_id}"
            )
        value = value[component]
    if value is None or isinstance(value, (dict, list)):
        raise ValueError(
            f"data.split_group_metadata_key={key!r} must be a scalar for item {item_id}"
        )
    return str(value)


def _group_assignments(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    validation: float,
    test: float,
    minimum: int,
    group_key: str,
    require_class_coverage: bool,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    enabled = _enabled_splits(validation=validation, test=test)
    weights = _split_weights(validation=validation, test=test, enabled=enabled)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = _group_value(dict(row["metadata"]), group_key, str(row["item_id"]))
        groups.setdefault(group, []).append(row)
    if not groups:
        raise ValueError("Cannot create a group split for an empty dataset")
    class_groups: dict[str, set[str]] = {}
    if require_class_coverage:
        for group, group_rows in groups.items():
            for row in group_rows:
                label = row.get("label")
                if label is None:
                    raise ValueError(
                        "stratified_group splitting requires an effective class label for every training item"
                    )
                class_groups.setdefault(str(label), set()).add(group)
        required_groups = minimum * len(enabled)
        infeasible = {
            label: len(group_ids)
            for label, group_ids in class_groups.items()
            if len(group_ids) < required_groups
        }
        if infeasible:
            details = ", ".join(
                f"class {label} has {count} independent groups (requires {required_groups})"
                for label, count in sorted(infeasible.items())
            )
            raise ValueError(
                "Grouped class-coverage split is infeasible: " + details + ". Add independent groups, reduce enabled splits, or use class_stratified."
            )

    # First reserve one group per class/split.  The deterministic rarest-class
    # order makes the common case reproducible.  The post-check below never
    # pretends a greedy allocation proved feasibility for a hard mixed-class
    # group layout.
    assigned: dict[str, str] = {}
    coverage: dict[str, dict[str, int]] = {
        label: {split: 0 for split in enabled} for label in class_groups
    }
    if require_class_coverage:
        for label in sorted(class_groups, key=lambda value: (len(class_groups[value]), value)):
            candidates = sorted(class_groups[label], key=lambda group: _stable_rank(seed, group))
            for split in enabled:
                if coverage[label][split] >= minimum:
                    continue
                selected = next(
                    (
                        group
                        for group in candidates
                        if group not in assigned
                        and all(coverage[other_label][split] < minimum for other_label in {
                            str(row["label"]) for row in groups[group] if row.get("label") is not None
                        })
                    ),
                    None,
                )
                if selected is None:
                    selected = next((group for group in candidates if group not in assigned), None)
                if selected is None:
                    break
                assigned[selected] = split
                for row in groups[selected]:
                    if row.get("label") is not None:
                        coverage[str(row["label"])][split] += 1

    totals = {split: 0 for split in enabled}
    for group, split in assigned.items():
        totals[split] += len(groups[group])
    total_items = sum(len(group_rows) for group_rows in groups.values())
    desired = {split: total_items * weights[split] / sum(weights.values()) for split in enabled}
    for group in sorted(groups, key=lambda value: _stable_rank(seed, value)):
        if group in assigned:
            continue
        split = min(enabled, key=lambda name: ((totals[name] - desired[name]) / max(desired[name], 1), name))
        assigned[group] = split
        totals[split] += len(groups[group])
        for row in groups[group]:
            if require_class_coverage and row.get("label") is not None:
                coverage[str(row["label"])][split] += 1
    if require_class_coverage:
        missing = {
            label: [split for split in enabled if coverage[label][split] < minimum]
            for label in coverage
            if any(coverage[label][split] < minimum for split in enabled)
        }
        if missing:
            details = "; ".join(f"class {label} missing {', '.join(splits)}" for label, splits in sorted(missing.items()))
            raise ValueError(
                "Grouped class-coverage allocation could not satisfy the declared constraints: "
                + details
                + ". The group layout may be infeasible; use class_stratified or revise groups."
            )
    assignments = [
        {"item_id": str(row["item_id"]), "split": assigned[group]}
        for group, group_rows in groups.items()
        for row in group_rows
    ]
    class_counts = {
        label: {split: coverage[label].get(split, 0) for split in SPLIT_NAMES}
        for label in coverage
    }
    return sorted(assignments, key=lambda row: row["item_id"]), {
        "enabled_splits": list(enabled),
        "minimum_per_class": minimum if require_class_coverage else None,
        "class_counts": class_counts,
        "group_count": len(groups),
        "group_metadata_key": group_key,
        "missing_coverage": {},
        "valid": True,
    }


def _source_partition_assignments(
    rows: list[tuple[str, str | None]], *, required: bool
) -> list[dict[str, str]] | None:
    """Convert imported source partitions into one run-owned protocol.

    Source partitions are immutable import provenance in ``metadata_json``.
    They are not model splits until this function records them in a run's
    manifest. ``None`` means the dataset has no complete partition layout.
    """
    assignments: list[dict[str, str]] = []
    missing: list[str] = []
    invalid: set[str] = set()
    for item_id, raw_metadata in rows:
        try:
            metadata = json.loads(raw_metadata) if raw_metadata else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid metadata_json for dataset item {item_id}") from exc
        partition = metadata.get("source_partition")
        if partition is None:
            missing.append(item_id)
            continue
        partition = str(partition)
        if partition not in SPLIT_NAMES:
            invalid.add(partition)
            continue
        assignments.append({"item_id": str(item_id), "split": partition})
    if invalid:
        raise ValueError(f"Invalid source partitions: {sorted(invalid)}")
    if missing:
        if required:
            preview = ", ".join(missing[:3])
            raise ValueError(
                "split_strategy='source_partitions' requires every item to have "
                f"source_partition metadata; {len(missing)} missing (for example: {preview})"
            )
        return None
    return sorted(assignments, key=lambda row: row["item_id"])


def _effective_split_rows(
    connection: sqlite3.Connection, dataset_type: str
) -> list[dict[str, Any]]:
    """Resolve the actual single-label supervised population before splitting."""
    if dataset_type != "classification":
        return [
            {
                "item_id": str(item_id),
                "metadata": json.loads(metadata or "{}"),
                "label": None,
            }
            for item_id, metadata in connection.execute(
                "SELECT item_id, metadata_json FROM dataset_items ORDER BY item_id"
            )
        ]
    return [
        {
            "item_id": str(row[0]),
            "metadata": json.loads(row[1] or "{}"),
            "label": None if row[2] is None else str(int(row[2])),
        }
        for row in connection.execute(
            """
            SELECT di.item_id, di.metadata_json, label.class_index
            FROM dataset_items di
            JOIN classification_items ci ON ci.item_id = di.item_id
            LEFT JOIN classification_annotations annotation
              ON annotation.item_id = di.item_id AND annotation.is_current = 1
            LEFT JOIN classification_labels label ON label.label_id = annotation.label_id
            ORDER BY di.item_id
            """
        )
    ]


def _coverage_report(
    assignments: list[dict[str, str]],
    rows: list[dict[str, Any]],
    *,
    enabled: tuple[str, ...],
    minimum: int,
) -> dict[str, Any]:
    assignment_by_item = {row["item_id"]: row["split"] for row in assignments}
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        if row.get("label") is None:
            continue
        label = str(row["label"])
        counts.setdefault(label, {split: 0 for split in SPLIT_NAMES})
        counts[label][assignment_by_item[str(row["item_id"])]] += 1
    missing = {
        label: [split for split in enabled if counts[label][split] < minimum]
        for label in sorted(counts)
        if any(counts[label][split] < minimum for split in enabled)
    }
    return {
        "enabled_splits": list(enabled),
        "minimum_per_class": minimum,
        "class_counts": counts,
        "missing_coverage": missing,
        "valid": not missing,
    }


def create_split_manifest(
    run_dir: str | Path,
    sqlite_path: str | Path,
    config: dict[str, Any],
    *,
    verified_dataset_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Create the immutable dataset-item assignment protocol for one model run."""
    data = config["data"]
    validation = float(data.get("validation_split", 0.2))
    test = float(data.get("test_split", 0.1))
    if validation < 0 or test < 0 or validation + test >= 1:
        raise ValueError("Split fractions must be non-negative and sum to less than 1")
    seed = int(config["run"].get("seed", 123))
    layout = RunLayout(run_dir)
    if layout.split_manifest.exists():
        raise FileExistsError(layout.split_manifest)

    # A fully resolved new-run config always contains split_strategy and uses
    # the new auto policy.  Keep this low-level helper compatible with old
    # callers that supplied only fractions/seed (for artifact inspection and
    # migrations), whose historical contract was deterministic random.
    strategy = str(data.get("split_strategy", "random")).lower()
    supported = {
        "auto", "random", "source_partitions", "class_stratified",
        "stratified_group", "group_only",
    }
    if strategy not in supported:
        raise ValueError(
            "data.split_strategy must be auto, random, source_partitions, "
            "class_stratified, stratified_group, or group_only"
        )
    minimum = int(data.get("split_minimum_per_class", 1))
    if minimum < 0:
        raise ValueError("data.split_minimum_per_class must be non-negative")
    group_key = data.get("split_group_metadata_key")
    if strategy in {"stratified_group", "group_only"} and not isinstance(group_key, str):
        raise ValueError(
            "data.split_group_metadata_key is required for stratified_group and group_only splitting"
        )
    if isinstance(group_key, str) and not group_key.strip():
        raise ValueError("data.split_group_metadata_key cannot be empty")
    with sqlite3.connect(Path(sqlite_path).expanduser().resolve()) as connection:
        info = read_dataset_info(connection)
        # The training launcher has already validated its immutable checkpoint
        # and calculated this canonical identity. Reuse that exact value rather
        # than serializing every dataset row a second time before training can
        # begin. Other callers retain the standalone verification behavior.
        fingerprint = verified_dataset_fingerprint or dataset_fingerprint(connection)
        item_rows = [
            (str(row[0]), row[1])
            for row in connection.execute(
                "SELECT item_id, metadata_json FROM dataset_items ORDER BY item_id"
            )
        ]
        effective_rows = _effective_split_rows(connection, str(info["dataset_type"]))
    item_ids = [row[0] for row in item_rows]
    expected_dataset = config.get("dataset", {})
    if (
        expected_dataset.get("dataset_id")
        and expected_dataset["dataset_id"] != info["dataset_id"]
    ):
        raise ValueError("Resolved config dataset_id does not match the input dataset")
    if (
        expected_dataset.get("fingerprint_sha256")
        and expected_dataset["fingerprint_sha256"] != fingerprint
    ):
        raise ValueError(
            "Resolved config dataset fingerprint does not match the input dataset"
        )
    if not item_ids:
        raise ValueError("Cannot create a split manifest for an empty dataset")
    source_assignments = None if strategy in {
        "random", "class_stratified", "stratified_group", "group_only"
    } else _source_partition_assignments(item_rows, required=strategy == "source_partitions")
    if source_assignments is not None:
        assignments = source_assignments
        policy: dict[str, Any] = {
            "method": "source_partitions",
            "source_metadata_key": "source_partition",
        }
        coverage = _coverage_report(
            assignments, effective_rows,
            enabled=_enabled_splits(validation=validation, test=test), minimum=minimum,
        )
        if info["dataset_type"] == "classification" and minimum and not coverage["valid"]:
            details = "; ".join(
                f"class {label} missing {', '.join(splits)}"
                for label, splits in coverage["missing_coverage"].items()
            )
            raise ValueError(
                "Trusted source partitions violate the configured class-coverage requirement: "
                + details
                + ". Keep the benchmark partition and set data.split_minimum_per_class = 0, "
                "or explicitly choose class_stratified."
            )
    elif strategy in {"auto", "class_stratified"}:
        if info["dataset_type"] != "classification":
            if strategy == "class_stratified":
                raise ValueError(
                    "class_stratified splitting is only supported for single-label classification datasets"
                )
            assignments = _assignment_rows(item_ids, seed=seed, validation=validation, test=test)
            policy = {
                "method": "stable_sha256_rank", "seed": seed,
                "validation_fraction": validation, "test_fraction": test,
                "fallback_from": "auto",
            }
            coverage = {"valid": True, "unassessed": "non_classification_dataset"}
        else:
            assignments, coverage = _class_stratified_assignments(
                effective_rows, seed=seed, validation=validation, test=test, minimum=minimum
            )
            policy = {
                "method": "class_stratified", "algorithm_version": "class_stratified_v1",
                "seed": seed, "validation_fraction": validation,
                "test_fraction": test, "minimum_per_class": minimum,
                **({"fallback_from": "auto"} if strategy == "auto" else {}),
            }
    elif strategy in {"stratified_group", "group_only"}:
        assignments, coverage = _group_assignments(
            effective_rows, seed=seed, validation=validation, test=test, minimum=minimum,
            group_key=str(group_key), require_class_coverage=strategy == "stratified_group",
        )
        policy = {
            "method": strategy, "algorithm_version": "group_assignment_v1", "seed": seed,
            "validation_fraction": validation, "test_fraction": test,
            "minimum_per_class": minimum if strategy == "stratified_group" else None,
            "group_metadata_key": group_key,
        }
    else:
        assignments = _assignment_rows(
            item_ids, seed=seed, validation=validation, test=test
        )
        policy = {
            "method": "stable_sha256_rank",
            "seed": seed,
            "validation_fraction": validation,
            "test_fraction": test,
        }
        coverage = {"valid": True, "unassessed": "random_policy"}
    counts = {
        name: sum(row["split"] == name for row in assignments)
        for name in SPLIT_NAMES
    }
    manifest: dict[str, Any] = {
        "schema": {
            "name": SPLIT_MANIFEST_SCHEMA_NAME,
            "version": SPLIT_MANIFEST_SCHEMA_VERSION,
        },
        "split_manifest_id": str(uuid.uuid4()),
        "created_at": _utc_now(),
        "dataset": {
            "dataset_id": info["dataset_id"],
            "revision_id": info["revision_id"],
            "fingerprint_sha256": fingerprint,
        },
        "policy": policy,
        "counts": counts,
        "assignments": assignments,
        "assignment_digest_sha256": _assignment_digest(assignments),
        "coverage": coverage,
        "fingerprint_sha256": None,
    }
    manifest["fingerprint_sha256"] = _manifest_fingerprint(manifest)
    _write_json(layout.split_manifest, manifest)
    return manifest


def create_unavailable_split_manifest(
    run_dir: str | Path,
    config: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Record that historical per-item split assignments are unavailable."""
    layout = RunLayout(run_dir)
    if layout.split_manifest.exists():
        raise FileExistsError(layout.split_manifest)
    dataset = config.get("dataset", {})
    manifest: dict[str, Any] = {
        "schema": {
            "name": SPLIT_MANIFEST_SCHEMA_NAME,
            "version": SPLIT_MANIFEST_SCHEMA_VERSION,
        },
        "split_manifest_id": str(uuid.uuid4()),
        "created_at": _utc_now(),
        "dataset": {
            "dataset_id": dataset.get("dataset_id"),
            "revision_id": dataset.get("revision_id"),
            "fingerprint_sha256": dataset.get("fingerprint_sha256"),
        },
        "policy": {
            "method": "unavailable",
            "reason": str(reason),
        },
        "counts": {name: None for name in SPLIT_NAMES},
        "assignments": [],
        "fingerprint_sha256": None,
    }
    manifest["fingerprint_sha256"] = _manifest_fingerprint(manifest)
    _write_json(layout.split_manifest, manifest)
    return manifest


def read_split_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = RunLayout(run_dir).split_manifest
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    schema = manifest.get("schema", {})
    if (
        schema.get("name") != SPLIT_MANIFEST_SCHEMA_NAME
        or schema.get("version") not in _SUPPORTED_SCHEMA_VERSIONS
    ):
        raise ValueError(f"Unsupported split manifest schema in {path}")
    expected = manifest.get("fingerprint_sha256")
    if not expected or expected != _manifest_fingerprint(manifest):
        raise ValueError(f"Split manifest fingerprint mismatch: {path}")
    assignment_digest = manifest.get("assignment_digest_sha256")
    if assignment_digest and assignment_digest != _assignment_digest(manifest.get("assignments", [])):
        raise ValueError(f"Split manifest assignment digest mismatch: {path}")
    return manifest


def split_assignments(manifest: dict[str, Any]) -> dict[str, str]:
    result = {
        str(row["item_id"]): str(row["split"])
        for row in manifest.get("assignments", [])
    }
    invalid = set(result.values()).difference(SPLIT_NAMES)
    if invalid:
        raise ValueError(f"Invalid split names in manifest: {sorted(invalid)}")
    return result


def attach_split_manifest(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Attach runtime-only assignments without embedding them in resolved config."""
    config["_split_manifest"] = {
        "split_manifest_id": manifest["split_manifest_id"],
        "fingerprint_sha256": manifest["fingerprint_sha256"],
        "dataset": dict(manifest["dataset"]),
        "assignments": split_assignments(manifest),
        "assignment_digest_sha256": manifest.get("assignment_digest_sha256"),
        "policy": dict(manifest.get("policy", {})),
        "coverage": dict(manifest.get("coverage", {})),
    }


def split_manifest_matches_dataset(
    config: dict[str, Any], sqlite_path: str | Path
) -> bool:
    runtime = config.get("_split_manifest")
    if runtime is None:
        return False
    # This identity check is also reached before the regular segmentation
    # loader. Preserve its legacy ROI migration guarantee at this boundary.
    if config.get("run", {}).get("task") == "segmentation":
        from oracle_builder.datasets.legacy_roi import migrate_legacy_roi_if_needed

        migrate_legacy_roi_if_needed(sqlite_path)
    with sqlite3.connect(Path(sqlite_path).expanduser().resolve()) as connection:
        info = read_dataset_info(connection)
        fingerprint = dataset_fingerprint(connection)
    dataset = runtime.get("dataset", {})
    return (
        dataset.get("dataset_id") == info["dataset_id"]
        and dataset.get("revision_id") == info["revision_id"]
        and dataset.get("fingerprint_sha256") == fingerprint
    )
