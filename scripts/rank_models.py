#!/usr/bin/env python3
"""Rank fine-tuned models from sachs_summary.csv.

Compares three conditions:
  name_only   — no observational/interventional data, real variable names
  withdata    — matrix/summary prompts, real variable names (anonymize=0)
  anonymized  — matrix/summary prompts, anonymized variable names (anonymize=1)

Ranking is done on shared configs (conditions where every model has ≥1 valid row),
using a valid_rows-weighted mean F1 as the primary metric and mean SHD as tiebreaker.

Usage:
  python scripts/rank_models.py
  python scripts/rank_models.py --csv path/to/sachs_summary.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CSV = REPO_ROOT / "scripts" / "responses" / "sachs" / "sachs_summary.csv"

PRETTY_MODEL: dict[str, str] = {
    "Qwen3-4B-Thinking-2507":                                "Qwen3-4B (base)",
    "qwen3_4b_sft_5way_v4_2gpu_merged":                      "SFT-5way-v4",
    "qwen3_4b_cd_format_v5_rerun_2gpu":                      "CD-v5",
    "qwen3_4b_cd_format_v5_rerun_2gpu_checkpoint-100":       "CD-v5 Ckpt-100",
    "qwen3_4b_cd_format_v5_rerun_2gpu_checkpoint-300_merged": "CD-v5 Ckpt-300",
    "grpo_from_qwen3_4b_sft_5way_v4_2gpu_checkpoint-100_merged": "GRPO Ckpt-100",
    "grpo_from_qwen3_4b_sft_5way_v4_2gpu_checkpoint-400":        "GRPO Ckpt-400",
}

CONFIG_COLS = ["obs_n", "int_n", "prompt_style", "anonymize"]


def _wmean(g: pd.DataFrame, val: str, wt: str) -> float:
    w = g[wt].clip(lower=1)
    return float((g[val] * w).sum() / w.sum())


def _condition(row: pd.Series) -> str:
    if row["prompt_style"] == "names_only":
        return "name_only"
    return "anonymized" if row["anonymize"] == 1 else "withdata"


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["valid_rows"].fillna(0) > 0].copy()
    df["display_model"] = df["model"].map(PRETTY_MODEL).fillna(df["model"])
    df["condition"] = df.apply(_condition, axis=1)
    return df


def shared_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Overall ranking on configs shared by every model (wtd-mean F1, SHD tiebreak)."""
    n_models = df["model"].nunique()
    shared_cfgs = (
        df.groupby(CONFIG_COLS)["model"]
        .nunique()
        .reset_index(name="n")
        .query("n == @n_models")[CONFIG_COLS]
    )
    shared = df.merge(shared_cfgs, on=CONFIG_COLS)

    rank = (
        shared.groupby("display_model")
        .apply(
            lambda g: pd.Series(
                {
                    "F1":    round(_wmean(g, "avg_f1", "valid_rows"), 3),
                    "SHD":   round(_wmean(g, "avg_shd", "valid_rows"), 1),
                    "configs":      len(g),
                    "valid_rows":   int(g["valid_rows"].sum()),
                }
            ),
            include_groups=False,
        )
        .sort_values(["F1", "SHD"], ascending=[False, True])
        .reset_index()
        .rename(columns={"display_model": "Model"})
    )
    rank.insert(0, "Rank", range(1, len(rank) + 1))
    return rank, len(shared_cfgs)


def condition_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition wtd-mean F1 and SHD for every model."""
    rows = []
    for (display_model, cond), g in df.groupby(["display_model", "condition"]):
        rows.append(
            {
                "Model":    display_model,
                "condition": cond,
                "F1":       round(_wmean(g, "avg_f1", "valid_rows"), 3),
                "SHD":      round(_wmean(g, "avg_shd", "valid_rows"), 1),
                "configs":  len(g),
                "valid_rows": int(g["valid_rows"].sum()),
            }
        )
    long = pd.DataFrame(rows)

    # Pivot to wide: one row per model, columns = (metric, condition)
    wide = long.pivot_table(
        index="Model",
        columns="condition",
        values=["F1", "SHD"],
        aggfunc="first",
    )
    wide.columns = [f"{m} ({c})" for m, c in wide.columns]

    # Order columns: F1 first, SHD second; conditions: name_only → withdata → anonymized
    cond_order = ["name_only", "withdata", "anonymized"]
    f1_cols  = [f"F1 ({c})"  for c in cond_order if f"F1 ({c})"  in wide.columns]
    shd_cols = [f"SHD ({c})" for c in cond_order if f"SHD ({c})" in wide.columns]
    wide = wide[f1_cols + shd_cols]

    # Sort by withdata F1 descending
    sort_col = "F1 (withdata)" if "F1 (withdata)" in wide.columns else wide.columns[0]
    wide = wide.sort_values(sort_col, ascending=False).reset_index()
    wide.insert(0, "Rank", range(1, len(wide) + 1))
    return wide


def _print_df(title: str, df: pd.DataFrame) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")
    print(df.to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to sachs_summary.csv")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        return 1

    df = load(args.csv)
    n_models = df["model"].nunique()
    print(f"Loaded {len(df)} valid rows, {n_models} models from {args.csv}")

    rank, n_shared = shared_rank(df)
    _print_df(
        f"Overall ranking — shared configs ({n_shared} conditions, wtd-mean F1 ↑, SHD ↓)",
        rank,
    )

    breakdown = condition_breakdown(df)
    _print_df(
        "Per-condition breakdown — wtd-mean F1 (↑) and SHD (↓) by naming regime",
        breakdown,
    )

    # Delta: withdata − anonymized (semantic-prior lift)
    if "F1 (withdata)" in breakdown.columns and "F1 (anonymized)" in breakdown.columns:
        delta = breakdown[["Model", "F1 (withdata)", "F1 (anonymized)"]].copy()
        delta["F1 lift (real − anon)"] = (
            delta["F1 (withdata)"] - delta["F1 (anonymized)"]
        ).round(3)
        delta = delta.sort_values("F1 lift (real − anon)", ascending=False).reset_index(drop=True)
        delta.insert(0, "Rank", range(1, len(delta) + 1))
        _print_df("Semantic-prior lift — F1(real names) − F1(anonymized)", delta)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
