from __future__ import annotations

import sqlite3

from oracle_builder.data.sqlite_dataset import create_synthetic_classification
from oracle_builder.datasets.schema import read_dataset_info, validate_database
from oracle_data_contracts.datasets.subset import subset_classification_dataset


def test_subset_filters_rare_classes_and_creates_editable_provenanced_dataset(tmp_path):
    source = tmp_path / "source.sqlite"
    output = tmp_path / "subset.sqlite"
    create_synthetic_classification(source, n=10, shape=(8, 8, 1), classes=3)

    result = subset_classification_dataset(
        source, output, minimum_class_count=4, seed=7
    )

    assert result["selection"]["selected_item_count"] == 4
    assert {row["name"] for row in result["selection"]["excluded_labels"]} == {"1", "2"}
    with sqlite3.connect(source) as connection:
        source_id = read_dataset_info(connection)["dataset_id"]
    with sqlite3.connect(output) as connection:
        info = read_dataset_info(connection)
        assert info["dataset_id"] != source_id
        assert info["lifecycle"] == "working"
        assert info["metadata"]["derived_from"]["operation"] == "subset"
        assert validate_database(connection)["valid"]
        labels = connection.execute(
            "SELECT class_index, name FROM classification_labels ORDER BY class_index"
        ).fetchall()
        assert labels == [(0, "0")]
        assert connection.execute("SELECT COUNT(*) FROM dataset_items").fetchone()[0] == 4
        # A derived working database is intentionally editable before checkpointing.
        connection.execute("UPDATE dataset_items SET sample_weight = 1.5 WHERE rowid = 1")


def test_subset_limit_is_deterministic_and_dry_run_does_not_create_output(tmp_path):
    source = tmp_path / "source.sqlite"
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    dry_run_output = tmp_path / "dry-run.sqlite"
    create_synthetic_classification(source, n=12, shape=(8, 8, 1), classes=3)

    dry_run = subset_classification_dataset(
        source, dry_run_output, max_items=5, seed=42, dry_run=True
    )
    assert dry_run["dry_run"] is True
    assert not dry_run_output.exists()
    subset_classification_dataset(source, first, max_items=5, seed=42)
    subset_classification_dataset(source, second, max_items=5, seed=42)

    with sqlite3.connect(first) as connection:
        first_ids = [row[0] for row in connection.execute("SELECT item_id FROM dataset_items ORDER BY item_id")]
    with sqlite3.connect(second) as connection:
        second_ids = [row[0] for row in connection.execute("SELECT item_id FROM dataset_items ORDER BY item_id")]
    assert first_ids == second_ids
