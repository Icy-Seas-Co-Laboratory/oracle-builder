#!/usr/bin/env python3
"""Download CIFAR benchmarks into Oracle Builder's SQLite dataset format."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from oracle_builder.data.cifar_import import main


if __name__ == "__main__":
    raise SystemExit(main())
