from __future__ import annotations

import argparse
import json

from oracle_builder.clustering.training import (
    fit_clustering_evidence_from_encoder,
    train_clustering_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Legacy clustering compatibility CLI; new runs should use oracle-embed."
    )
    parser.add_argument("-c", "--config", required=True, help="TOML clustering configuration")
    parser.add_argument("-i", "--input", required=True, help="Frozen classification Dataset V1 SQLite file")
    parser.add_argument("-o", "--output", help="Sealed clustering run artifact directory (train mode)")
    parser.add_argument(
        "--mode",
        choices=("train", "fit"),
        help="Operation mode; inferred as fit when --encoder-run is supplied",
    )
    parser.add_argument("--encoder-run", help="Existing sealed encoder artifact (fit mode)")
    parser.add_argument(
        "--reopen-reseal",
        action="store_true",
        help="Explicitly reopen and reseal --encoder-run to attach fitted evidence",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mode = args.mode or ("fit" if args.encoder_run else "train")
    if mode == "fit":
        if not args.encoder_run:
            parser.error("--encoder-run is required in fit mode")
        if args.output:
            parser.error("--output is not used in fit mode; evidence is attached to --encoder-run")
        if args.overwrite:
            parser.error("--overwrite is not supported in fit mode")
        result = fit_clustering_evidence_from_encoder(
            args.config,
            args.input,
            args.encoder_run,
            reopen_and_reseal=args.reopen_reseal,
        )
    else:
        if not args.output:
            parser.error("--output is required in train mode")
        if args.encoder_run or args.reopen_reseal:
            parser.error("--encoder-run and --reopen-reseal are only valid in fit mode")
        result = train_clustering_run(
            args.config,
            args.input,
            args.output,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
