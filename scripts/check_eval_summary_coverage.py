#!/usr/bin/env python3
"""Check whether per-graph eval summaries preserve paper-relevant information.

The response directories can contain two summary styles:

  * eval_summary.csv: direct output from the evaluation pipeline.
  * <graph>_summary.csv: richer consolidated table used by older Sachs plots.

This script makes the relationship explicit.  It reports schema differences
when both files exist and verifies that every eval_summary.csv contains the
columns needed by the MICAD paper collection/plotting pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_GRAPHS = ["cancer", "earthquake", "asia", "sachs"]

PAPER_REQUIRED_COLUMNS = [
    "dataset",
    "model",
    "response_csv",
    "obs_n",
    "int_n",
    "prompt_style",
    "anonymize",
    "valid_rows",
    "num_rows",
    "avg_f1",
    "avg_shd",
    "acyclic_rate",
    "avg_skeleton_f1",
    "avg_ancestor_f1",
    "consensus_f1",
    "consensus_shd",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses-root",
        type=Path,
        default=Path("scripts/responses"),
        help="Root containing per-graph response directories.",
    )
    parser.add_argument(
        "--graphs",
        nargs="*",
        default=DEFAULT_GRAPHS,
        help="Graphs to check.",
    )
    parser.add_argument(
        "--fail-on-paper-missing",
        action="store_true",
        help="Exit nonzero if an eval_summary.csv lacks paper-required columns.",
    )
    parser.add_argument(
        "--fail-on-rich-missing",
        action="store_true",
        help="Exit nonzero if eval_summary.csv lacks any columns present in <graph>_summary.csv.",
    )
    return parser.parse_args()


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def main() -> None:
    args = parse_args()
    failed = False

    for graph in args.graphs:
        graph_dir = args.responses_root / graph
        eval_path = graph_dir / "eval_summary.csv"
        rich_path = graph_dir / f"{graph}_summary.csv"

        print(f"\n[{graph}]")
        if not eval_path.exists():
            print(f"  missing eval summary: {eval_path}")
            failed = True
            continue

        eval_cols = read_header(eval_path)
        missing_paper = [col for col in PAPER_REQUIRED_COLUMNS if col not in eval_cols]
        print(f"  eval_summary.csv columns: {len(eval_cols)}")
        if missing_paper:
            print(f"  missing paper-required columns: {', '.join(missing_paper)}")
            failed = failed or args.fail_on_paper_missing
        else:
            print("  paper-required columns: complete")

        if not rich_path.exists():
            print(f"  no richer {graph}_summary.csv present; eval_summary.csv is the only summary source")
            continue

        rich_cols = read_header(rich_path)
        missing_from_eval = sorted(set(rich_cols) - set(eval_cols))
        extra_in_eval = sorted(set(eval_cols) - set(rich_cols))
        print(f"  {graph}_summary.csv columns: {len(rich_cols)}")
        if missing_from_eval:
            print(
                "  eval_summary.csv omits richer derived columns: "
                f"{len(missing_from_eval)}"
            )
            print("  examples:", ", ".join(missing_from_eval[:20]))
            failed = failed or args.fail_on_rich_missing
        else:
            print(f"  eval_summary.csv contains all columns in {graph}_summary.csv")
        if extra_in_eval:
            print("  eval-only columns:", ", ".join(extra_in_eval[:20]))

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
