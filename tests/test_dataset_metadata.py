from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from oracle_builder.data.sqlite_dataset import create_synthetic_classification
from oracle_builder.datasets.cli import main as dataset_cli
from oracle_builder.datasets.metadata import add_metadata_document
from oracle_builder.datasets.schema import (
    DatasetSchemaError,
    dataset_fingerprint,
    set_dataset_lifecycle,
)


def test_add_metadata_document_preserves_source_and_records_provenance(tmp_path):
    database = tmp_path / "library.sqlite"
    document = tmp_path / "metadata.toml"
    document.write_text('[dataset]\ntitle = "Example library"\n', encoding="utf-8")
    create_synthetic_classification(database, n=3, shape=(8, 8, 1), classes=2)
    with sqlite3.connect(database) as connection:
        before = dataset_fingerprint(connection)

    result = add_metadata_document(database, document, actor="tester")

    assert result["action"] == "added"
    assert result["name"] == "metadata.toml"
    assert result["sha256"] == hashlib.sha256(document.read_bytes()).hexdigest()
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            """
            SELECT document_id, source_filename, source_format, parsed_json,
                   raw_text, sha256
            FROM metadata_documents
            """
        ).fetchone()
        event = connection.execute(
            """
            SELECT event_type, actor, details_json
            FROM dataset_events
            WHERE event_type = 'metadata_document.added'
            """
        ).fetchone()
        after = dataset_fingerprint(connection)

    assert row[0] == result["document_id"]
    assert row[1:3] == ("metadata.toml", "toml")
    assert json.loads(row[3])["dataset"]["title"] == "Example library"
    assert row[4] == document.read_text(encoding="utf-8")
    assert row[5] == result["sha256"]
    assert event[0:2] == ("metadata_document.added", "tester")
    assert json.loads(event[2])["document_id"] == result["document_id"]
    assert after != before


def test_add_metadata_document_replaces_same_logical_name(tmp_path):
    database = tmp_path / "library.sqlite"
    first = tmp_path / "first.json"
    second = tmp_path / "second.yml"
    first.write_text('{"version": 1}\n', encoding="utf-8")
    second.write_text("version: 2\n", encoding="utf-8")
    create_synthetic_classification(database, n=2, shape=(8, 8, 1), classes=2)

    added = add_metadata_document(database, first, name="about", actor="tester")
    updated = add_metadata_document(database, second, name="about", actor="tester")

    assert updated["action"] == "updated"
    assert updated["document_id"] == added["document_id"]
    assert updated["previous_sha256"] == added["sha256"]
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """
            SELECT name, source_filename, source_format, parsed_json
            FROM metadata_documents
            """
        ).fetchall()
        event = connection.execute(
            """
            SELECT details_json FROM dataset_events
            WHERE event_type = 'metadata_document.updated'
            """
        ).fetchone()

    assert rows == [("about", "second.yml", "yml", '{"version": 2}')]
    assert json.loads(event[0])["previous_sha256"] == added["sha256"]


def test_add_metadata_document_requires_an_editable_dataset(tmp_path):
    database = tmp_path / "library.sqlite"
    document = tmp_path / "metadata.json"
    document.write_text('{"title": "Frozen"}\n', encoding="utf-8")
    create_synthetic_classification(database, n=2, shape=(8, 8, 1), classes=2)
    with sqlite3.connect(database) as connection:
        set_dataset_lifecycle(connection, "frozen", actor="tester")
        connection.commit()

    with pytest.raises(DatasetSchemaError, match="thaw it"):
        add_metadata_document(database, document)

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT count(*) FROM metadata_documents"
        ).fetchone()[0] == 0


def test_metadata_add_cli_outputs_attachment_receipt(tmp_path, capsys):
    database = tmp_path / "library.sqlite"
    document = tmp_path / "about.json"
    document.write_text('{"source": "test"}\n', encoding="utf-8")
    create_synthetic_classification(database, n=2, shape=(8, 8, 1), classes=2)

    assert dataset_cli(
        [
            "metadata-add",
            str(database),
            str(document),
            "--actor",
            "cli-test",
        ]
    ) == 0

    receipt = json.loads(capsys.readouterr().out)
    assert receipt["action"] == "added"
    assert receipt["name"] == "about.json"
    assert receipt["actor"] == "cli-test"
