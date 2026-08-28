from __future__ import annotations

import argparse
import json

from oracle_builder.embedding.training import train_embedding_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a self-supervised embedding model.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = train_embedding_run(args.config, args.input, args.output, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
