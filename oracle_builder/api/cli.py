from __future__ import annotations

import argparse
import os
import sys

import uvicorn

from oracle_builder.api.app import create_app
from oracle_builder.api.registry import InferenceModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Oracle Builder inference bundles over HTTP.")
    parser.add_argument("--model", action="append", default=[], metavar="ALIAS=RUN_DIR")
    parser.add_argument(
        "--models-root",
        action="append",
        default=[],
        metavar="DIRECTORY",
        help="Recursively discover sealed model artifacts beneath DIRECTORY.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--no-preload", action="store_true")
    args = parser.parse_args()

    registry = InferenceModelRegistry()
    for root in args.models_root:
        try:
            report = registry.register_root(root)
        except OSError as exc:
            parser.error(f"Cannot discover --models-root {root!r}: {exc}")
        for skipped in report["skipped"]:
            print(
                f"Skipping {skipped['path']}: {skipped['reason']}",
                file=sys.stderr,
            )
    for value in args.model:
        if "=" not in value:
            parser.error("--model must use ALIAS=RUN_DIR syntax")
        alias, run_dir = value.split("=", 1)
        try:
            registry.register(alias, run_dir)
        except ValueError as exc:
            parser.error(str(exc))
    if not registry.registered_count:
        parser.error("Register at least one model with --model or --models-root")
    uvicorn.run(
        create_app(
            registry,
            auth_token=os.environ.get("ORACLE_BUILDER_API_TOKEN"),
            preload=not args.no_preload,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
