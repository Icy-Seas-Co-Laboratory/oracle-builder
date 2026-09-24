from __future__ import annotations

import argparse

import uvicorn

from oracle_builder.orchestration.api import create_app
from oracle_builder.orchestration.service import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Oracle Builder orchestration API.")
    parser.add_argument("--database", required=True, help="Central SQLite orchestration database.")
    parser.add_argument("--workspace-root", required=True, help="Root containing approved working inputs and recipes.")
    parser.add_argument("--artifact-root", help="Owned runtime staging root for uploads, evaluations, imported products, and packages.")
    parser.add_argument("--runs-root", help="Canonical model-run directory; defaults to WORKSPACE_ROOT/runs.")
    parser.add_argument("--datasets-root", help="Canonical training-dataset directory; defaults to WORKSPACE_ROOT/datasets.")
    parser.add_argument("--browse-root", action="append", default=[], help="Additional allow-listed root exposed to the file explorer.")
    parser.add_argument("--training-catalog-root", action="append", default=[], help="Read-only allow-listed root to scan for available training sets; may be repeated.")
    parser.add_argument("--upload-limit-mib", type=int, default=10_240, help="Maximum uploaded file size in MiB.")
    parser.add_argument("--oracle-serve", action="append", default=[], metavar="NAME=URL", help="Configured Oracle Serve compute endpoint; may be repeated.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8110)
    args = parser.parse_args()
    endpoints = []
    for value in args.oracle_serve:
        if "=" not in value:
            parser.error("--oracle-serve must use NAME=URL syntax")
        endpoints.append(tuple(value.split("=", 1)))
    uvicorn.run(create_app(Orchestrator(args.database, workspace_root=args.workspace_root, artifact_root=args.artifact_root, runs_root=args.runs_root, datasets_root=args.datasets_root, browse_roots=args.browse_root, training_catalog_roots=args.training_catalog_root or None, upload_limit_bytes=args.upload_limit_mib * 1024 * 1024, compute_endpoints=endpoints)), host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
