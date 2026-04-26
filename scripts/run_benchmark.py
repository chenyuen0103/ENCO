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
    parser = argparse.ArgumentParser(description="Run a config-driven LLM causal-discovery benchmark.")
    parser.add_argument("--config", required=True, help="Benchmark config path or registry name.")
    parser.add_argument(
        "--steps",
        default="build,models,baselines,summarize",
        help="Comma-separated steps: build,models,baselines,summarize",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-query model outputs if they already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps = {part.strip() for part in args.steps.split(",") if part.strip()}
    runner = load_runner(args.config)
    if "build" in steps:
        runner.build(dry_run=args.dry_run)
    if "models" in steps:
        runner.run_models(dry_run=args.dry_run, overwrite=args.overwrite)
    if "baselines" in steps:
        runner.run_baselines(dry_run=args.dry_run)
    if "summarize" in steps and not args.dry_run:
        outputs = runner.summarize()
        for name, path in outputs.items():
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
