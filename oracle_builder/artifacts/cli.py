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
    else:
        result = unpack_run_artifact(args.package, args.output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("valid", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
