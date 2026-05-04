#!/usr/bin/env python3
"""Generate MICAD valid-run coverage tables.

Inputs are produced by scripts/collect_micad_paper_results.py.  The script
writes:

  experiments/out/micad_paper/paper_valid_rate_primary_budget.csv
  experiments/out/micad_paper/paper_valid_rate_by_budget.csv
  experiments/out/micad_paper/tables/valid_rate_primary_budget.tex
  experiments/out/micad_paper/tables/valid_rate_by_budget.tex
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


GRAPH_ORDER = ["cancer", "earthquake", "asia", "sachs"]
CONDITION_ORDER = ["names_only", "real+summary", "real+matrix", "anon+summary", "anon+matrix"]
CONDITION_LABELS = {
    "names_only": "Names",
    "real+summary": "Sum",
    "real+matrix": "Mat",
    "anon+summary": "Anon. sum",
    "anon+matrix": "Anon. mat",
}
CONDITION_SHORT = {
    "names_only": "names",
    "real+summary": "sum",
    "real+matrix": "mat",
    "anon+summary": "anon_sum",
    "anon+matrix": "anon_mat",
}
DATA_ONLY_METHODS = {"PC", "GES", "ENCO"}
PAPER_MODEL_ORDER = [
    "GPT-5 mini",
    "GPT-5.2 pro",
    "Qwen3-4B",
    "Qwen3-30B-A3B",
    "Qwen2.5-7B",
    "Qwen2.5-14B",
    "Qwen2.5-72B",
    "Llama-3.1-8B",
    "Llama-3.1-8B-Inst.",
    "Llama-3.1-70B",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("experiments/out/micad_paper"),
        help="Directory containing canonical_condition_results.csv.",
    )
    parser.add_argument("--obs", type=int, default=5000, help="Primary observational budget.")
    parser.add_argument("--inter", type=int, default=200, help="Primary interventional budget.")
    parser.add_argument(
        "--include-anon",
        action="store_true",
        help="Include anonymized summary/matrix columns in the primary-budget wide table.",
    )
    parser.add_argument(
        "--max-appendix-rows",
        type=int,
        default=120,
        help="Maximum rows to include in the LaTeX by-budget appendix table.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Include every model/checkpoint in the primary table. Default is the curated paper model list.",
    )
    return parser.parse_args()


def read_canonical(paper_dir: Path) -> pd.DataFrame:
    path = paper_dir / "canonical_condition_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    df = pd.read_csv(path)
    for col in ["obs", "inter", "valid_rows", "n_rows"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[~df["method"].isin(DATA_ONLY_METHODS)].copy()


def format_count(valid: object, total: object) -> str:
    if pd.isna(valid) or pd.isna(total):
        return "--"
    return f"{int(valid)}/{int(total)}"


def valid_rate(valid: object, total: object) -> float:
    if pd.isna(valid) or pd.isna(total) or float(total) <= 0:
        return np.nan
    return float(valid) / float(total)


def resolve_response_path(path: object, graph: object) -> Path | None:
    text = "" if pd.isna(path) else str(path)
    if not text:
        return None
    candidate = Path(text)
    if candidate.exists():
        return candidate
    basename = candidate.name
    graph_name = "" if pd.isna(graph) else str(graph)
    fallbacks = [
        Path("scripts/responses") / graph_name / basename,
        Path("scripts/responses_old") / graph_name / basename,
        Path("experiments/responses") / graph_name / basename,
        Path("experiments/responses_old") / graph_name / basename,
    ]
    for fallback in fallbacks:
        if fallback.exists():
            return fallback
    return None


@lru_cache(maxsize=None)
def response_has_context_window_error(path: str) -> bool:
    try:
        df = pd.read_csv(path, usecols=lambda c: c == "raw_response")
    except Exception:
        return False
    raw = df.get("raw_response", pd.Series("", index=df.index)).fillna("").astype(str)
    error = raw.str.contains(r"\[ERROR\]", na=False)
    context = raw.str.contains(r"max_model_len|context window|context length|maximum context", case=False, na=False)
    return bool((error & context).any())


def is_context_window_cell(row: pd.Series) -> bool:
    path = resolve_response_path(row.get("path"), row.get("graph"))
    return bool(path and response_has_context_window_error(str(path)))


def format_coverage_cell(row: pd.Series | None) -> str:
    if row is None:
        return "--"
    if is_context_window_cell(row):
        return "--"
    return format_count(row["valid_rows"], row["n_rows"])


def valid_rate_cell(row: pd.Series | None) -> float:
    if row is None or is_context_window_cell(row):
        return np.nan
    return valid_rate(row["valid_rows"], row["n_rows"])


def get_cell(df: pd.DataFrame, graph: str, model: str, condition: str, obs: int, inter: int) -> pd.Series | None:
    sub = df[(df["graph"].eq(graph)) & (df["method_display"].eq(model)) & (df["condition"].eq(condition))]
    if condition != "names_only":
        sub = sub[(sub["obs"].eq(obs)) & (sub["inter"].eq(inter))]
    if sub.empty:
        return None
    return sub.sort_values(["valid_rows", "n_rows"], ascending=[False, False]).iloc[0]


def ordered_models(df: pd.DataFrame, all_models: bool) -> list[str]:
    models = [str(model) for model in df["method_display"].dropna().unique()]
    if all_models:
        return sorted(models)
    present = set(models)
    return [model for model in PAPER_MODEL_ORDER if model in present]


def build_primary_budget_table(df: pd.DataFrame, obs: int, inter: int, include_anon: bool, all_models: bool) -> pd.DataFrame:
    conditions = ["names_only", "real+summary", "real+matrix"]
    if include_anon:
        conditions.extend(["anon+summary", "anon+matrix"])

    models = ordered_models(df, all_models)
    rows = []
    for model in models:
        row = {"model": model}
        has_any = False
        for graph in GRAPH_ORDER:
            parts = []
            rates = []
            for condition in conditions:
                cell = get_cell(df, graph, model, condition, obs, inter)
                col = f"{graph}_{CONDITION_SHORT[condition]}"
                if cell is None:
                    parts.append("--")
                    row[col] = "--"
                    rates.append(np.nan)
                    continue
                coverage = format_coverage_cell(cell)
                parts.append(coverage)
                row[col] = coverage
                rates.append(valid_rate_cell(cell))
                has_any = True
            row[f"{graph}_coverage"] = " ".join(parts)
            finite_rates = [rate for rate in rates if np.isfinite(rate)]
            row[f"{graph}_min_valid_rate"] = min(finite_rates) if finite_rates else np.nan
        if has_any:
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_order"] = out["model"].map({model: idx for idx, model in enumerate(models)})
    return out.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def build_by_budget_table(df: pd.DataFrame) -> pd.DataFrame:
    keep = df[df["condition"].isin(CONDITION_ORDER)].copy()
    keep["condition_label"] = keep["condition"].map(CONDITION_LABELS)
    keep["context_window_error"] = keep.apply(is_context_window_cell, axis=1)
    keep["coverage"] = keep.apply(
        lambda r: "--" if r["context_window_error"] else format_count(r["valid_rows"], r["n_rows"]),
        axis=1,
    )
    keep["valid_rate"] = keep.apply(
        lambda r: np.nan if r["context_window_error"] else valid_rate(r["valid_rows"], r["n_rows"]),
        axis=1,
    )
    keep = keep[
        [
            "graph",
            "method_display",
            "obs",
            "inter",
            "condition",
            "condition_label",
            "valid_rows",
            "n_rows",
            "coverage",
            "valid_rate",
            "context_window_error",
        ]
    ].copy()
    keep["graph"] = pd.Categorical(keep["graph"], categories=GRAPH_ORDER, ordered=True)
    keep["condition"] = pd.Categorical(keep["condition"], categories=CONDITION_ORDER, ordered=True)
    return keep.sort_values(["graph", "obs", "inter", "method_display", "condition"]).reset_index(drop=True)


def latex_escape(text: object) -> str:
    s = "" if pd.isna(text) else str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


def write_primary_latex(df: pd.DataFrame, path: Path, obs: int, inter: int, include_anon: bool) -> None:
    conditions = ["names_only", "real+summary", "real+matrix"]
    if include_anon:
        conditions.extend(["anon+summary", "anon+matrix"])
    subheaders = "/".join(CONDITION_LABELS[condition] for condition in conditions)
    n_condition_cols = len(conditions)
    total_condition_cols = len(GRAPH_ORDER) * n_condition_cols
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Valid parsed runs at the primary budget ($N={obs}$, $M={inter}$). Each graph cell reports {subheaders}, where entries are valid/total runs. ``--'' denotes a missing matched cell or a context-window error, not a parser failure.}}",
        r"\label{tab:valid_rate_primary_budget}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l" + "c" * total_condition_cols + "}",
        r"\toprule",
        "Model "
        + "".join(rf"& \multicolumn{{{n_condition_cols}}}{{c}}{{{graph.capitalize()}}} " for graph in GRAPH_ORDER)
        + r"\\",
        "".join(
            rf"\cmidrule(lr){{{2 + idx * n_condition_cols}-{1 + (idx + 1) * n_condition_cols}}}"
            for idx in range(len(GRAPH_ORDER))
        ),
        " & "
        + " & ".join(
            CONDITION_LABELS[condition]
            for _graph in GRAPH_ORDER
            for condition in conditions
        )
        + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        vals = [latex_escape(row["model"])]
        vals.extend(
            latex_escape(row.get(f"{graph}_{CONDITION_SHORT[condition]}", "--"))
            for graph in GRAPH_ORDER
            for condition in conditions
        )
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {path}")


def write_by_budget_latex(df: pd.DataFrame, path: Path, max_rows: int) -> None:
    show = df.head(max_rows).copy()
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Valid parsed runs by graph, budget, model, and condition. Showing the first {len(show)} rows; the CSV contains the full audit table.}}",
        r"\label{tab:valid_rate_by_budget}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrlc}",
        r"\toprule",
        r"Graph & Model & $N$ & $M$ & Condition & Valid \\",
        r"\midrule",
    ]
    for _, row in show.iterrows():
        vals = [
            latex_escape(str(row["graph"]).capitalize()),
            latex_escape(row["method_display"]),
            str(int(row["obs"])),
            str(int(row["inter"])),
            latex_escape(row["condition_label"]),
            latex_escape(row["coverage"]),
        ]
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {path}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path}")


def main() -> None:
    args = parse_args()
    df = read_canonical(args.paper_dir)

    primary = build_primary_budget_table(df, args.obs, args.inter, args.include_anon, args.all_models)
    by_budget = build_by_budget_table(df)

    table_dir = args.paper_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    write_csv(primary, args.paper_dir / "paper_valid_rate_primary_budget.csv")
    write_csv(by_budget, args.paper_dir / "paper_valid_rate_by_budget.csv")
    write_primary_latex(
        primary,
        table_dir / "valid_rate_primary_budget.tex",
        args.obs,
        args.inter,
        args.include_anon,
    )
    write_by_budget_latex(by_budget, table_dir / "valid_rate_by_budget.tex", args.max_appendix_rows)


if __name__ == "__main__":
    main()
