"""Controlled taxonomy import and explicit classifier-label mapping."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from oracle_data_contracts.datasets.schema import read_dataset_info, utc_now

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


# Kept constant so a Pelagia vocabulary node resolves to the same UUID in every
# dataset database before concept IDs are embedded directly in a future release.
PELAGIA_TAXONOMY_NAMESPACE = uuid.UUID("73d96396-4a85-5a62-a0ef-8b119f6c98c0")


def taxonomy_concept_id(vocabulary_id: str, node_id: str) -> str:
    return str(uuid.uuid5(PELAGIA_TAXONOMY_NAMESPACE, f"{vocabulary_id}:{node_id}"))


def _load_taxonomy(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(path).expanduser().resolve()
    with source.open("rb") as handle:
        document = tomllib.load(handle)
    vocabulary = document.get("vocabulary")
    taxonomy = document.get("taxonomy")
    if not isinstance(vocabulary, dict) or not isinstance(taxonomy, dict):
        raise ValueError("Taxonomy TOML must contain [vocabulary] and [taxonomy]")
    if not vocabulary.get("id") or not vocabulary.get("version"):
        raise ValueError("Taxonomy vocabulary requires id and version")
    nodes = taxonomy.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Taxonomy TOML contains no taxonomy.nodes")
    for node in nodes:
        if not isinstance(node, dict) or not node.get("id") or not node.get("name"):
            raise ValueError("Every taxonomy node requires id and name")
    return vocabulary, nodes


def import_taxonomy_concepts(
    connection: sqlite3.Connection,
    taxonomy_path: str | Path,
    *,
    actor: str | None = None,
) -> dict[str, Any]:
    """Import only the taxonomy vocabulary; target and image tags are excluded."""
    info = read_dataset_info(connection)
    if info["lifecycle"] != "working":
        raise ValueError("Thaw the dataset before importing taxonomy concepts")
    vocabulary, nodes = _load_taxonomy(taxonomy_path)
    vocabulary_id = str(vocabulary["id"])
    vocabulary_version = str(vocabulary["version"])
    node_ids = {str(node["id"]) for node in nodes}
    imported = updated = mappings = 0
    for node in nodes:
        node_id = str(node["id"])
        concept_id = str(node.get("concept_id") or taxonomy_concept_id(vocabulary_id, node_id))
        parent_node_id = node.get("parent_id")
        parent_concept_id = (
            taxonomy_concept_id(vocabulary_id, str(parent_node_id))
            if parent_node_id in node_ids
            else None
        )
        metadata = {
            key: value
            for key, value in node.items()
            if key not in {
                "id", "name", "display_name", "scientific_name", "concept_type",
                "rank", "parent_id", "selectable", "mappings", "concept_id",
            }
        }
        exists = connection.execute(
            "SELECT 1 FROM taxonomy_concepts WHERE concept_id = ?", (concept_id,)
        ).fetchone()
        connection.execute(
            """
            INSERT INTO taxonomy_concepts (
                concept_id, vocabulary_id, vocabulary_version, vocabulary_node_id,
                name, display_name, scientific_name, concept_type, rank,
                parent_concept_id, selectable, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(concept_id) DO UPDATE SET
                vocabulary_version = excluded.vocabulary_version,
                name = excluded.name, display_name = excluded.display_name,
                scientific_name = excluded.scientific_name, concept_type = excluded.concept_type,
                rank = excluded.rank, parent_concept_id = excluded.parent_concept_id,
                selectable = excluded.selectable, metadata_json = excluded.metadata_json
            """,
            (
                concept_id, vocabulary_id, vocabulary_version, node_id, node["name"],
                node.get("display_name"), node.get("scientific_name"),
                node.get("concept_type", "taxon"), node.get("rank"), parent_concept_id,
                int(bool(node.get("selectable", True))),
                json.dumps(metadata, sort_keys=True, default=str), utc_now(),
            ),
        )
        updated += int(exists is not None)
        imported += int(exists is None)
        for mapping in node.get("mappings", []):
            authority = str(mapping["authority"])
            scheme = str(mapping["scheme"])
            identifier = str(mapping["identifier"])
            connection.execute(
                """
                INSERT INTO taxonomy_concept_mappings (
                    mapping_id, concept_id, authority, scheme, identifier, uri,
                    relationship, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(authority, scheme, identifier) DO UPDATE SET
                    concept_id = excluded.concept_id, uri = excluded.uri,
                    relationship = excluded.relationship, metadata_json = excluded.metadata_json
                """,
                (
                    str(uuid.uuid5(uuid.UUID(concept_id), f"{authority}:{scheme}:{identifier}")),
                    concept_id, authority, scheme, identifier, mapping.get("uri"),
                    mapping.get("relationship", "exact"),
                    json.dumps(
                        {key: value for key, value in mapping.items() if key not in {"authority", "scheme", "identifier", "uri", "relationship"}},
                        sort_keys=True,
                        default=str,
                    ),
                    utc_now(),
                ),
            )
            mappings += 1
    connection.execute(
        """
        INSERT INTO dataset_events (
            event_id, dataset_id, revision_id, event_type, created_at, actor, details_json
        ) VALUES (?, ?, ?, 'taxonomy.imported', ?, ?, ?)
        """,
        (
            str(uuid.uuid4()), info["dataset_id"], info["revision_id"], utc_now(), actor,
            json.dumps(
                {
                    "vocabulary_id": vocabulary_id,
                    "vocabulary_version": vocabulary_version,
                    "taxonomy_path": str(Path(taxonomy_path).expanduser().resolve()),
                    "taxonomy_nodes": len(nodes),
                },
                sort_keys=True,
            ),
        ),
    )
    return {
        "vocabulary_id": vocabulary_id,
        "vocabulary_version": vocabulary_version,
        "taxonomy_nodes": len(nodes),
        "imported": imported,
        "updated": updated,
        "mappings": mappings,
    }


def map_classification_label_to_concept(
    connection: sqlite3.Connection,
    label: str,
    concept: str,
    *,
    relationship: str = "exact",
    mapped_by: str | None = None,
) -> dict[str, Any]:
    """Map one dataset classifier label to one imported taxonomy node or UUID."""
    info = read_dataset_info(connection)
    if info["lifecycle"] != "working":
        raise ValueError("Thaw the dataset before changing classifier concept mappings")
    label_row = connection.execute(
        "SELECT label_id, name FROM classification_labels WHERE name = ?", (label,)
    ).fetchone()
    if label_row is None:
        raise ValueError(f"Classification label not found: {label!r}")
    concept_row = connection.execute(
        """
        SELECT concept_id, vocabulary_node_id, name FROM taxonomy_concepts
        WHERE concept_id = ? OR vocabulary_node_id = ?
        """,
        (concept, concept),
    ).fetchone()
    if concept_row is None:
        raise ValueError(f"Imported taxonomy concept not found: {concept!r}")
    if relationship not in {"exact", "broader", "narrower", "related"}:
        raise ValueError("relationship must be exact, broader, narrower, or related")
    connection.execute(
        """
        INSERT INTO classification_label_concepts (
            label_id, concept_id, relationship, mapped_by, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, '{}', ?)
        ON CONFLICT(label_id) DO UPDATE SET
            concept_id = excluded.concept_id, relationship = excluded.relationship,
            mapped_by = excluded.mapped_by, created_at = excluded.created_at
        """,
        (label_row[0], concept_row[0], relationship, mapped_by, utc_now()),
    )
    return {
        "label": label_row[1], "label_id": label_row[0],
        "concept": concept_row[2], "concept_id": concept_row[0],
        "vocabulary_node_id": concept_row[1], "relationship": relationship,
    }
