#!/usr/bin/env python3
"""
Build a compact comparison table (wide + macro) from an evaluation summary CSV.

This matches the shape of the `experiments/collect_result.ipynb` snippet:
  - filter evaluated rows
  - map prompt_style -> method label
  - aggregate over shuffles/runs
  - pivot to a side-by-side "wide" table
  - compute a macro-average across grid cells

Works with:
  - `experiments/responses/<dataset>/<dataset>_summary.csv` (recommended), or
  - a summary CSV produced by `experiments/evaluate.py --summary-csv ...`
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


_RESP_META_RE = re.compile(
    r"^responses_obs(?P<obs>\d+)_int(?P<int>\d+)_shuf(?P<shuf>\d+)(?P<tail>.*)$",
    flags=re.IGNORECASE,
)


def _infer_prompt_style_from_stem(stem: str) -> str:
    styles = [
        "payload_topk",
        "summary_hist_rows",
        "summary",
        "matrix",
        "payload",
        "cases",
        "names_only",
        "enco",
    ]
    for style in styles:
        if re.search(rf"(?:^|_){re.escape(style)}(?:_|$)", stem, flags=re.IGNORECASE):
            return style
    return ""


def _infer_from_response_csv(response_csv: str) -> dict[str, Any]:
    p = Path(str(response_csv))
    stem = p.stem
    m = _RESP_META_RE.match(stem)

    model = ""
    try:
        m_names = re.match(r"^responses_names_only_p\d+_(?P<model>.+)$", stem, flags=re.IGNORECASE)
        if m_names:
            model = m_names.group("model")
        else:
            m_resp = re.match(
                r"^responses_obs\d+_int\d+_shuf\d+_p\d+_(?:anon_)?thinktags_"
                r"(?:matrix|summary|cases|payload|payload_topk)_(?P<model>.+)$",
                stem,
                flags=re.IGNORECASE,
            )
            if m_resp:
                model = m_resp.group("model")
    except Exception:
        model = ""

    out: dict[str, Any] = {
        "dataset": p.parent.name if p.parent.name else "",
        "model": model,
        "obs_n": None,
        "int_n": None,
        "prompt_style": _infer_prompt_style_from_stem(stem),
        "anonymize": int(bool(re.search(r"(?:^|_)anon(?:_|$)", stem, flags=re.IGNORECASE))),
    }
    if m:
        out["obs_n"] = int(m.group("obs"))
        out["int_n"] = int(m.group("int"))
    return out


def _coalesce_col(df: pd.DataFrame, dst: str, srcs: list[str]) -> None:
    if dst in df.columns:
        return
    for s in srcs:
        if s in df.columns:
            df[dst] = df[s]
            return
    df[dst] = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", required=True, type=Path)
    ap.add_argument("--metric", choices=["avg", "consensus"], default="avg")
    ap.add_argument("--dataset", default=None, help="Optional dataset filter (e.g., sachs).")
    ap.add_argument("--model", default=None, help="Optional model filter (e.g., gpt-5-mini).")
    ap.add_argument(
        "--prompt-styles",
        nargs="*",
        default=[],
        help="Optional prompt_style filter (e.g., summary matrix).",
    )
    ap.add_argument("--out-wide", type=Path, default=None)
    ap.add_argument("--out-macro", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)

    # Accept either schema:
    # - experiment1 summary: response_csv
    # - evaluate.py summary: response_csv (preferred) or older: csv
    if "response_csv" not in df.columns and "csv" in df.columns:
        df = df.rename(columns={"csv": "response_csv"})

    if "response_csv" not in df.columns:
        raise SystemExit(f"Missing `response_csv` column in: {args.summary_csv}")

    # If evaluated is missing, treat all rows as evaluated summaries.
    if "evaluated" not in df.columns:
        df["evaluated"] = 1

    df = df[df["evaluated"] == 1].copy()

    # Fill metadata columns if missing, using filename parsing.
    _coalesce_col(df, "dataset", ["dataset"])
    _coalesce_col(df, "model", ["model", "model_tag"])
    _coalesce_col(df, "obs_n", ["obs_n", "obs"])
    _coalesce_col(df, "int_n", ["int_n", "int"])
    _coalesce_col(df, "prompt_style", ["prompt_style"])
    _coalesce_col(df, "anonymize", ["anonymize"])

    need_infer = df[["dataset", "model", "obs_n", "int_n", "prompt_style", "anonymize"]].isna().any(axis=1)
    if need_infer.any():
        inferred = df.loc[need_infer, "response_csv"].apply(_infer_from_response_csv).apply(pd.Series)
        for col in inferred.columns:
            if col not in df.columns:
                df[col] = None
            df.loc[need_infer, col] = df.loc[need_infer, col].where(df.loc[need_infer, col].notna(), inferred[col])

    # Normalize types.
    df["anonymize"] = df["anonymize"].fillna(0).astype(int)
    df["obs_n"] = pd.to_numeric(df["obs_n"], errors="coerce")
    df["int_n"] = pd.to_numeric(df["int_n"], errors="coerce")

    if args.dataset:
        df = df[df["dataset"] == args.dataset]
    if args.model:
        df = df[df["model"] == args.model]
    if args.prompt_styles:
        allow = {s.strip() for s in args.prompt_styles if s.strip()}
        df = df[df["prompt_style"].isin(allow)]

    method_map = {
        "summary": "LLM-summary",
        "matrix": "LLM-matrix",
        "enco": "ENCO",
    }
    df["method"] = df["prompt_style"].map(method_map)
    df = df[df["method"].notna()].copy()

    score_col = f"{args.metric}_f1"
    err_col = f"{args.metric}_shd"
    if score_col not in df.columns or err_col not in df.columns:
        raise SystemExit(
            f"Missing required columns for metric={args.metric}: {score_col}, {err_col} in {args.summary_csv}"
        )

    agg = (
        df.groupby(["method", "anonymize", "obs_n", "int_n"], as_index=False)
        .agg(f1=(score_col, "mean"), shd=(err_col, "mean"), n=("response_csv", "count"))
    )

    wide = agg.pivot_table(
        index=["anonymize", "obs_n", "int_n"],
        columns="method",
        values=["f1", "shd", "n"],
        aggfunc="first",
    )
    wide = wide.sort_index(axis=0).sort_index(axis=1)

    # Flatten MultiIndex columns for CSV-friendliness.
    wide_flat = wide.copy()
    wide_flat.columns = [f"{a}__{b}" for a, b in wide_flat.columns.to_flat_index()]
    wide_flat = wide_flat.reset_index()

    macro = (
        agg.groupby(["method", "anonymize"], as_index=False)
        .agg(mean_f1=("f1", "mean"), mean_shd=("shd", "mean"), cells=("f1", "count"))
        .sort_values(["anonymize", "mean_shd", "mean_f1"], ascending=[True, True, False])
    )

    if args.out_wide:
        args.out_wide.parent.mkdir(parents=True, exist_ok=True)
        wide_flat.to_csv(args.out_wide, index=False)
        print(f"[done] wrote {args.out_wide}")

    if args.out_macro:
        args.out_macro.parent.mkdir(parents=True, exist_ok=True)
        macro.to_csv(args.out_macro, index=False)
        print(f"[done] wrote {args.out_macro}")

    # Always print macro for quick inspection.
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 140):
        print(macro.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
