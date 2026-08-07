from __future__ import annotations

import argparse
import os

import uvicorn

from oracle_builder.api.app import create_app
from oracle_builder.api.registry import InferenceModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Oracle Builder inference bundles over HTTP.")
    parser.add_argument("--model", action="append", default=[], metavar="ALIAS=RUN_DIR")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--no-preload", action="store_true")
    args = parser.parse_args()

    registry = InferenceModelRegistry()
    for value in args.model:
        if "=" not in value:
            parser.error("--model must use ALIAS=RUN_DIR syntax")
        alias, run_dir = value.split("=", 1)
        registry.register(alias, run_dir)
    if not registry.registered_count:
        parser.error("Register at least one model with --model")
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
