from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from oracle_data_contracts.datasets.lifecycle import (
    restore_workspace_snapshot,
    save_checkpoint,
    save_workspace_snapshot,
    thaw_database,
)
from oracle_data_contracts.datasets.metadata import add_metadata_document
from oracle_data_contracts.datasets.schema import (
    dataset_fingerprint,
    read_dataset_info,
    set_dataset_lifecycle,
    validate_database,
    workspace_fingerprint,
)
from oracle_data_contracts.datasets.transfer import export_dataset, import_dataset_export
from oracle_data_contracts.datasets.subset import subset_classification_dataset
from oracle_data_contracts.datasets.taxonomy import (
    import_taxonomy_concepts,
    map_classification_label_to_concept,
)
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
    subset = commands.add_parser(
        "subset",
        help="Create an editable, deterministic classification subset.",
    )
    subset.add_argument("database")
    subset.add_argument("output")
    subset.add_argument(
        "--minimum-class-count",
        type=int,
        default=1,
        help="Retain only classes with at least this many current annotations (default: 1).",
    )
    subset.add_argument(
        "--max-items",
        type=int,
        help="Deterministically retain at most this many eligible items.",
    )
    subset.add_argument("--seed", type=int, default=123)
    subset.add_argument("--include-unlabeled", action="store_true")
    subset.add_argument("--name")
    subset.add_argument("--dry-run", action="store_true")
    taxonomy_import = commands.add_parser(
        "taxonomy-import", help="Import Pelagia taxonomy concepts into an editable dataset."
    )
    taxonomy_import.add_argument("database")
    taxonomy_import.add_argument("taxonomy")
    taxonomy_import.add_argument("--actor")
    taxonomy_map = commands.add_parser(
        "taxonomy-map", help="Explicitly link a classifier label to an imported taxonomy concept."
    )
    taxonomy_map.add_argument("database")
    taxonomy_map.add_argument("--label", required=True)
    taxonomy_map.add_argument("--concept", required=True)
    taxonomy_map.add_argument("--relationship", default="exact")
    taxonomy_map.add_argument("--actor")
    snapshot = commands.add_parser(
        "snapshot", help="Create a frozen full annotation-workspace snapshot."
    )
    snapshot.add_argument("database")
    snapshot.add_argument("--output")
    snapshot.add_argument("--actor")
    snapshot.add_argument("--note")
    restore_workspace = commands.add_parser(
        "restore-workspace", help="Fork a frozen workspace snapshot into an editable database."
    )
    restore_workspace.add_argument("snapshot")
    restore_workspace.add_argument("output")
    restore_workspace.add_argument("--actor")
    restore_workspace.add_argument("--reason")
    release = commands.add_parser(
        "release-training",
        help="Create a frozen training dataset from an annotation workspace.",
    )
    release.add_argument("workspace")
    release.add_argument("output")
    release.add_argument("--name")
    release.add_argument("--actor")
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
        "subset",
        "taxonomy-import",
        "taxonomy-map",
        "snapshot",
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
                result["workspace_fingerprint"] = workspace_fingerprint(connection)
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
    elif args.command == "subset":
        result = subset_classification_dataset(
            args.database,
            args.output,
            minimum_class_count=args.minimum_class_count,
            max_items=args.max_items,
            seed=args.seed,
            include_unlabeled=args.include_unlabeled,
            name=args.name,
            dry_run=args.dry_run,
        )
    elif args.command == "taxonomy-import":
        with sqlite3.connect(Path(args.database).expanduser().resolve()) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            result = import_taxonomy_concepts(connection, args.taxonomy, actor=args.actor)
            connection.commit()
    elif args.command == "taxonomy-map":
        with sqlite3.connect(Path(args.database).expanduser().resolve()) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            result = map_classification_label_to_concept(
                connection, args.label, args.concept,
                relationship=args.relationship, mapped_by=args.actor,
            )
            connection.commit()
    elif args.command == "snapshot":
        result = save_workspace_snapshot(
            args.database, args.output, actor=args.actor, note=args.note
        )
    elif args.command == "restore-workspace":
        result = restore_workspace_snapshot(
            args.snapshot,
            args.output,
            actor=args.actor,
            reason=args.reason,
        )
    elif args.command == "release-training":
        from oracle_data_contracts.datasets.lifecycle import release_training_dataset

        result = release_training_dataset(
            args.workspace, args.output, name=args.name, actor=args.actor
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
