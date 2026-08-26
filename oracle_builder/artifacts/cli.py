from __future__ import annotations

import argparse
import json

from oracle_builder.artifacts.run import (
    pack_run_artifact,
    migrate_legacy_run,
    read_run_manifest,
    reopen_run_artifact,
    seal_run_artifact,
    unpack_run_artifact,
    validate_run_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oracle-run", description="Inspect and preserve Oracle Builder run artifacts."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("info", "validate", "seal"):
        command = commands.add_parser(name)
        command.add_argument("run")
    reopen = commands.add_parser("reopen")
    reopen.add_argument("run")
    reopen.add_argument("--reason")
    pack = commands.add_parser("pack")
    pack.add_argument("run")
    pack.add_argument("output")
    unpack = commands.add_parser("unpack")
    unpack.add_argument("package")
    unpack.add_argument("output")
    migrate = commands.add_parser("migrate-legacy")
    migrate.add_argument("source")
    migrate.add_argument("output")
    publish = commands.add_parser(
        "publish-deployment",
        help="Create a lean sealed deployment asset from a sealed training record.",
    )
    publish.add_argument("training_run")
    publish.add_argument("output")
    publish.add_argument("--include-weights", action="store_true")
    publish.add_argument("--no-evidence", action="store_true")
    materialize = commands.add_parser(
        "materialize-training",
        help="Copy a sealed training record and embed retraining resources.",
    )
    materialize.add_argument("training_record")
    materialize.add_argument("output")
    materialize.add_argument("--dataset")
    materialize.add_argument("--source-root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "info":
        result = read_run_manifest(args.run)
    elif args.command == "validate":
        result = validate_run_artifact(args.run)
    elif args.command == "seal":
        result = seal_run_artifact(args.run)
    elif args.command == "reopen":
        result = reopen_run_artifact(args.run, reason=args.reason)
    elif args.command == "pack":
        result = pack_run_artifact(args.run, args.output)
    elif args.command == "migrate-legacy":
        result = migrate_legacy_run(args.source, args.output)
    elif args.command == "publish-deployment":
        from oracle_builder.artifacts.deployment import publish_deployment_asset

        result = publish_deployment_asset(
            args.training_run,
            args.output,
            include_weights=args.include_weights,
            include_evidence=not args.no_evidence,
        )
    elif args.command == "materialize-training":
        from oracle_builder.artifacts.training import materialize_training_record

        result = materialize_training_record(
            args.training_record,
            args.output,
            dataset=args.dataset,
            source_root=args.source_root,
        )
    else:
        result = unpack_run_artifact(args.package, args.output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("valid", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
