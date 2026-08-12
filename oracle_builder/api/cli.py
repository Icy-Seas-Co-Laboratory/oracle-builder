from __future__ import annotations

import argparse
import os
import sys

import uvicorn

from oracle_builder.api.app import create_app
from oracle_builder.api.compute import ComputeService
from oracle_builder.api.registry import InferenceModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Oracle Builder inference bundles and compute workers over HTTP.")
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
    parser.add_argument(
        "--root-path",
        default=os.environ.get("ORACLE_BUILDER_ROOT_PATH", ""),
        help="Externally visible path prefix when served behind a reverse proxy.",
    )
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--no-compute", action="store_true", help="Disable the local compute worker API.")
    parser.add_argument("--compute-queue-size", type=int, default=128)
    parser.add_argument("--worker-id", default=os.environ.get("ORACLE_BUILDER_WORKER_ID", "local"))
    parser.add_argument("--max-batch-size", type=int, default=256, help="Maximum combined inference items per model execution.")
    parser.add_argument("--max-wait-ms", type=int, default=8, help="Maximum queueing delay while forming a micro-batch.")
    parser.add_argument("--queue-capacity", type=int, default=1024, help="Maximum pending inference requests per model.")
    args = parser.parse_args()

    registry = InferenceModelRegistry(
        serving_max_batch_size=args.max_batch_size,
        serving_max_wait_ms=args.max_wait_ms,
        serving_queue_capacity=args.queue_capacity,
    )
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
    if not registry.registered_count and args.no_compute:
        parser.error("Register at least one model with --model or --models-root when --no-compute is set")
    uvicorn.run(
        create_app(
            registry,
            compute=None if args.no_compute else ComputeService(
                max_queue_size=args.compute_queue_size,
                worker_id=args.worker_id,
            ),
            auth_token=os.environ.get("ORACLE_BUILDER_API_TOKEN"),
            preload=not args.no_preload,
            root_path=args.root_path,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
