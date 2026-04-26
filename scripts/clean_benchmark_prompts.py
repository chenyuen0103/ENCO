#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_builder.cleanup import collect_prompt_cleanup_targets, delete_cleanup_targets
from benchmark_builder.runner import load_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete prompt files produced for a benchmark config.")
    parser.add_argument("--config", required=True, help="Benchmark config path or registry name.")
    parser.add_argument(
        "--example-prompts",
        action="store_true",
        help="Also delete saved example prompts from in-memory runs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print directories that would be deleted.")
    parser.add_argument("--yes", action="store_true", help="Delete without an interactive confirmation prompt.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = load_runner(args.config)
    targets = collect_prompt_cleanup_targets(
        spec=runner.spec,
        repo_root=runner.repo_root,
        include_examples=args.example_prompts,
    )

    print("Prompt cleanup targets:")
    for target in targets:
        exists = "exists" if target.path.exists() else "missing"
        print(f"- {target.label}: {target.path} [{exists}]")

    if args.dry_run:
        return 0

    if not args.yes:
        reply = input("Delete these prompt directories? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("Aborted.")
            return 1

    deleted = delete_cleanup_targets(targets)
    if deleted:
        for path in deleted:
            print(f"deleted: {path}")
    else:
        print("No prompt directories needed deletion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
