#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_builder.runner import load_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate evaluation outputs for a benchmark manifest.")
    parser.add_argument("--manifest", required=True, help="Benchmark manifest path or registry name.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = load_runner(args.manifest).summarize()
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
