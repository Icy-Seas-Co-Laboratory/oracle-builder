from __future__ import annotations

import sqlite3

from oracle_builder.data.sqlite_dataset import create_synthetic_classification
from oracle_data_contracts.datasets.schema import dataset_fingerprint, validate_database
from oracle_data_contracts.datasets.taxonomy import (
    import_taxonomy_concepts,
    map_classification_label_to_concept,
    taxonomy_concept_id,
)


def test_imports_only_taxonomy_nodes_and_maps_classification_label(tmp_path):
    database = tmp_path / "dataset.sqlite"
    vocabulary = tmp_path / "taxonomy.toml"
    vocabulary.write_text(
        """
[vocabulary]
id = "pelagia-core"
version = "0.2.0"

[taxonomy]
[[taxonomy.nodes]]
id = "identity"
name = "Identity"
concept_type = "root"
selectable = false

[[taxonomy.nodes]]
id = "taxon_copepoda"
name = "Copepoda"
parent_id = "identity"
concept_type = "taxon"
rank = "class"

[[taxonomy.nodes.mappings]]
authority = "worms"
scheme = "aphia_id"
identifier = "1080"
relationship = "exact"

[target_tags]
[[target_tags.nodes]]
id = "stage_adult"
name = "Adult"
"""
    )
    create_synthetic_classification(database, n=4, shape=(8, 8, 1), classes=2)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        imported = import_taxonomy_concepts(connection, vocabulary)
        mapped = map_classification_label_to_concept(
            connection, "0", "taxon_copepoda"
        )
        connection.commit()
        assert imported == {
            "vocabulary_id": "pelagia-core",
            "vocabulary_version": "0.2.0",
            "taxonomy_nodes": 2,
            "imported": 2,
            "updated": 0,
            "mappings": 1,
        }
        assert mapped["concept_id"] == taxonomy_concept_id(
            "pelagia-core", "taxon_copepoda"
        )
        assert connection.execute("SELECT count(*) FROM taxonomy_concepts").fetchone()[0] == 2
        assert validate_database(connection)["valid"]
        assert dataset_fingerprint(connection)
