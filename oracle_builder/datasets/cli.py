from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from oracle_builder.datasets.lifecycle import save_checkpoint, thaw_database
from oracle_builder.datasets.metadata import add_metadata_document
from oracle_builder.datasets.schema import (
    dataset_fingerprint,
    read_dataset_info,
    set_dataset_lifecycle,
    validate_database,
)
from oracle_builder.datasets.transfer import export_dataset, import_dataset_export
from oracle_builder.datasets.legacy_roi import (
    migrate_legacy_roi_database,
    migrate_legacy_roi_if_needed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oracle-dataset",
        description="Inspect and manage Oracle Builder V1 datasets.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("info", "validate"):
        command = commands.add_parser(name)
        command.add_argument("database")
    freeze = commands.add_parser("freeze")
    freeze.add_argument("database")
    freeze.add_argument("--actor")
    checkpoint = commands.add_parser("checkpoint")
    checkpoint.add_argument("database")
    checkpoint.add_argument("--output")
    checkpoint.add_argument("--actor")
    thaw = commands.add_parser("thaw")
    thaw.add_argument("database")
    thaw.add_argument("--actor")
    thaw.add_argument("--reason")
    metadata_add = commands.add_parser(
        "metadata-add",
        help="Attach or replace a TOML, JSON, or YAML metadata document.",
    )
    metadata_add.add_argument("database")
    metadata_add.add_argument("document")
    metadata_add.add_argument(
        "--name",
        help="Logical document name. Defaults to the source filename.",
    )
    metadata_add.add_argument("--actor")
    export = commands.add_parser("export")
    export.add_argument("database")
    export.add_argument("output")
    restore = commands.add_parser("import")
    restore.add_argument("input")
    restore.add_argument("output")
    migrate_roi = commands.add_parser(
        "migrate-roi",
        help="Migrate a legacy samples/mask_annotations ROI database to V1.",
    )
    migrate_roi.add_argument("database")
    migrate_roi.add_argument("--backup")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {
        "info",
        "validate",
        "freeze",
        "checkpoint",
        "thaw",
        "metadata-add",
        "export",
    }:
        migrate_legacy_roi_if_needed(args.database)
    if args.command == "migrate-roi":
        result = migrate_legacy_roi_database(
            args.database, backup_path=args.backup
        )
    elif args.command in {"info", "validate", "freeze"}:
        database = Path(args.database).expanduser().resolve()
        with sqlite3.connect(database) as connection:
            if args.command == "info":
                result = read_dataset_info(connection)
                result["fingerprint"] = dataset_fingerprint(connection)
            elif args.command == "validate":
                result = validate_database(connection)
            else:
                report = validate_database(connection)
                if not report["valid"]:
                    raise ValueError(
                        "Cannot freeze an invalid dataset: "
                        + "; ".join(report["errors"])
                    )
                result = set_dataset_lifecycle(
                    connection, "frozen", actor=args.actor
                )
                connection.commit()
                result["fingerprint"] = dataset_fingerprint(connection)
    elif args.command == "checkpoint":
        result = save_checkpoint(
            args.database, args.output, actor=args.actor
        )
    elif args.command == "thaw":
        result = thaw_database(
            args.database, actor=args.actor, reason=args.reason
        )
    elif args.command == "metadata-add":
        result = add_metadata_document(
            args.database,
            args.document,
            name=args.name,
            actor=args.actor,
        )
    elif args.command == "export":
        result = export_dataset(args.database, args.output)
    else:
        result = import_dataset_export(args.input, args.output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
