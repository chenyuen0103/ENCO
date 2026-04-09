#!/usr/bin/env python3
"""Compatibility wrapper for the older paper_slices workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_builder.runner import load_runner


REFINE_LOGS = REPO_ROOT / "refine-logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one frozen paper slice via the benchmark runner.")
    parser.add_argument("--manifest", required=True, help="Path to a paper_slices/*.json file.")
    parser.add_argument("--overwrite", action="store_true", help="Re-query model outputs if they already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def _write_handoff(manifest: str, outputs: dict[str, Path]) -> None:
    REFINE_LOGS.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        f"# Latest Slice Summary: {manifest}",
        "",
        "This slice was executed through the manifest-driven benchmark runner.",
        "",
        "## Produced Artifacts",
    ]
    for name, path in outputs.items():
        summary_lines.append(f"- {name}: `{path}`")
    summary_lines.extend(
        [
            "",
            "## Next Review Prompt",
            "",
            "Use the auto-review-loop and result-to-claim skills.",
            "Read CLAUDE.md, docs/research_contract.md, refine-logs/EXPERIMENT_PLAN.md, and this file.",
            "Treat this as a NeurIPS evaluation-and-dataset paper.",
            "Decide what this slice proves, identify the smallest remaining evidence gap, and choose exactly one next slice.",
        ]
    )
    (REFINE_LOGS / "latest_slice_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (REFINE_LOGS / "NEXT_SLICE.md").write_text(
        "# NEXT SLICE\n\nAfter reviewing `latest_slice_summary.md`, choose exactly one next slice.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    runner = load_runner(args.manifest)
    runner.build(dry_run=args.dry_run)
    runner.run_models(dry_run=args.dry_run, overwrite=args.overwrite)
    runner.run_baselines(dry_run=args.dry_run)
    if not args.dry_run:
        outputs = runner.summarize()
        _write_handoff(args.manifest, outputs)
        print(f"Wrote handoff files to {REFINE_LOGS / 'latest_slice_summary.md'} and {REFINE_LOGS / 'NEXT_SLICE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
