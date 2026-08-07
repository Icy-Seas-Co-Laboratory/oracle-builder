from __future__ import annotations

import argparse
import json

from oracle_builder.products.ingest import ingest_keras_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oracle-model", description="Ingest portable external model products.")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Ingest a .keras or legacy .h5 Keras model.")
    ingest.add_argument("--model", required=True)
    ingest.add_argument("--info", required=True, help="TOML product description.")
    ingest.add_argument("--output", required=True, help="New model-product artifact directory.")
    ingest.add_argument("--dataset", help="Optional SQLite dataset provenance reference.")
    ingest.add_argument(
        "--no-promote",
        action="store_true",
        help="Preserve the model without attempting standard named-output promotion.",
    )
    args = parser.parse_args(argv)
    result = ingest_keras_model(
        args.model,
        args.info,
        args.output,
        dataset=args.dataset,
        promote=False if args.no_promote else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
