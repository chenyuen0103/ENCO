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
    parser = argparse.ArgumentParser(description="Build prompts and prompt-bundle metadata for a benchmark config.")
    parser.add_argument("--config", required=True, help="Benchmark config path or registry name.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = load_runner(args.config)
    runner.build(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
