from __future__ import annotations

import json
import sqlite3

import pytest

from oracle_builder.artifacts import create_split_manifest, read_split_manifest
from oracle_builder.data.sqlite_dataset import create_synthetic_classification
from oracle_data_contracts.artifacts.splits import _class_stratified_assignments


def _config(**data):
    return {
        "run": {"seed": 91},
        "data": {
            "validation_split": 0.2,
            "test_split": 0.2,
            "split_strategy": "class_stratified",
            **data,
        },
    }


def test_class_stratified_manifest_covers_each_label_and_has_stable_assignment_digest(tmp_path):
    database = tmp_path / "classification.sqlite"
    create_synthetic_classification(database, n=18, shape=(8, 8, 1), classes=3)

    first = create_split_manifest(tmp_path / "first", database, _config())
    second = create_split_manifest(tmp_path / "second", database, _config())

    assert first["policy"]["method"] == "class_stratified"
    assert first["assignment_digest_sha256"] == second["assignment_digest_sha256"]
    assert first["coverage"]["valid"]
    assert all(
        counts[split] >= 1
        for counts in first["coverage"]["class_counts"].values()
        for split in ("train", "validation", "test")
    )
    assert read_split_manifest(tmp_path / "first")["assignment_digest_sha256"] == first["assignment_digest_sha256"]


def test_class_stratified_assignment_is_independent_of_input_row_order():
    rows = [
        {"item_id": f"{label}-{number}", "label": str(label), "metadata": {}}
        for label in range(2)
        for number in range(6)
    ]
    forward, _ = _class_stratified_assignments(
        rows, seed=4, validation=0.2, test=0.2, minimum=1
    )
    reverse, _ = _class_stratified_assignments(
        list(reversed(rows)), seed=4, validation=0.2, test=0.2, minimum=1
    )
    assert forward == reverse


def test_class_stratified_reports_rare_class_feasibility_failure(tmp_path):
    database = tmp_path / "classification.sqlite"
    create_synthetic_classification(database, n=5, shape=(8, 8, 1), classes=2)

    with pytest.raises(ValueError, match="Class-coverage split is infeasible.*class"):
        create_split_manifest(tmp_path / "run", database, _config())


def test_auto_preserves_complete_source_partitions_but_reports_coverage_conflicts(tmp_path):
    database = tmp_path / "classification.sqlite"
    create_synthetic_classification(database, n=9, shape=(8, 8, 1), classes=3)
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """
            SELECT di.item_id, label.class_index
            FROM dataset_items di
            JOIN classification_annotations annotation ON annotation.item_id = di.item_id AND annotation.is_current = 1
            JOIN classification_labels label ON label.label_id = annotation.label_id
            ORDER BY di.item_id
            """
        ).fetchall()
        for item_id, label in rows:
            # Every class lands in one source partition, an intentionally bad
            # imported benchmark layout which must never be silently resplit.
            connection.execute(
                "UPDATE dataset_items SET metadata_json = ? WHERE item_id = ?",
                (json.dumps({"source_partition": ("train", "validation", "test")[label]}), item_id),
            )
        connection.commit()

    with pytest.raises(ValueError, match="Trusted source partitions violate"):
        create_split_manifest(
            tmp_path / "run", database,
            {"run": {"seed": 91}, "data": {"split_strategy": "auto"}},
        )


def test_auto_derives_validation_from_official_train_without_moving_test(tmp_path):
    database = tmp_path / "cifar-like.sqlite"
    create_synthetic_classification(database, n=60, shape=(8, 8, 3), classes=10)
    official_test: set[str] = set()
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """
            SELECT di.item_id, label.class_index
            FROM dataset_items di
            JOIN classification_annotations annotation ON annotation.item_id = di.item_id AND annotation.is_current = 1
            JOIN classification_labels label ON label.label_id = annotation.label_id
            ORDER BY di.item_id
            """
        ).fetchall()
        per_class: dict[int, int] = {}
        for item_id, label in rows:
            index = per_class.get(label, 0)
            per_class[label] = index + 1
            partition = "test" if index == 0 else "train"
            if partition == "test":
                official_test.add(item_id)
            connection.execute(
                "UPDATE dataset_items SET metadata_json = ? WHERE item_id = ?",
                (json.dumps({"source_partition": partition}), item_id),
            )
        connection.commit()

    config = _config(split_strategy="auto")
    first = create_split_manifest(tmp_path / "first", database, config)
    second = create_split_manifest(tmp_path / "second", database, config)
    assigned = {row["item_id"]: row["split"] for row in first["assignments"]}

    assert first["policy"]["method"] == "source_partitions_with_derived_validation"
    assert first["assignment_digest_sha256"] == second["assignment_digest_sha256"]
    assert {item_id for item_id, split in assigned.items() if split == "test"} == official_test
    assert first["counts"]["validation"] > 0
    assert first["coverage"]["valid"] is True
    assert all(counts["validation"] >= 1 for counts in first["coverage"]["class_counts"].values())

    with pytest.raises(ValueError, match="Trusted source partitions violate"):
        create_split_manifest(tmp_path / "explicit", database, _config(split_strategy="source_partitions"))


def test_explicit_group_policy_requires_metadata_and_never_splits_a_group(tmp_path):
    database = tmp_path / "classification.sqlite"
    create_synthetic_classification(database, n=18, shape=(8, 8, 1), classes=3)
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """
            SELECT di.item_id, label.class_index
            FROM dataset_items di
            JOIN classification_annotations annotation ON annotation.item_id = di.item_id AND annotation.is_current = 1
            JOIN classification_labels label ON label.label_id = annotation.label_id
            ORDER BY di.item_id
            """
        ).fetchall()
        per_class: dict[int, int] = {}
        for item_id, label in rows:
            number = per_class.get(label, 0)
            per_class[label] = number + 1
            connection.execute(
                "UPDATE dataset_items SET metadata_json = ? WHERE item_id = ?",
                (json.dumps({"specimen": f"{label}-{number}"}), item_id),
            )
        connection.commit()

    manifest = create_split_manifest(
        tmp_path / "run",
        database,
        _config(split_strategy="stratified_group", split_group_metadata_key="specimen"),
    )
    assert manifest["coverage"]["valid"] is not False
    with sqlite3.connect(database) as connection:
        metadata = dict(connection.execute("SELECT item_id, metadata_json FROM dataset_items"))
    assigned = {row["item_id"]: row["split"] for row in manifest["assignments"]}
    groups: dict[str, set[str]] = {}
    for item_id, raw in metadata.items():
        groups.setdefault(json.loads(raw)["specimen"], set()).add(assigned[item_id])
    assert all(len(splits) == 1 for splits in groups.values())

    with pytest.raises(ValueError, match="split_group_metadata_key"):
        create_split_manifest(
            tmp_path / "missing-key",
            database,
            _config(split_strategy="stratified_group"),
        )
