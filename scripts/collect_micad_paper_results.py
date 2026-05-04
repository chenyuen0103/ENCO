#!/usr/bin/env python3
"""Collect MICAD paper-ready result tables from aggregated condition metrics.

This script takes the broad condition-level metric file produced by the
evaluation collector and creates a smaller, auditable set of CSV/LaTeX tables
and plot-data files for the paper.  It deliberately keeps provenance columns so
every table entry can be traced back to the selected source artifact.

Default input:
  experiments/out/micad_eval_results/all_condition_metrics.csv

Default output:
  experiments/out/micad_paper/
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


GRAPH_ORDER = ["cancer", "earthquake", "asia", "sachs"]
CONDITION_ORDER = ["names_only", "real+summary", "anon+summary", "real+matrix", "anon+matrix"]
DATA_ONLY_METHODS = ["PC", "GES", "ENCO"]


@dataclass(frozen=True)
class ModelInfo:
    method: str
    display: str
    family: str
    size_label: str
    size_value: float
    openness: str


MODEL_INFOS = [
    ModelInfo("gpt-5-mini", "GPT-5 mini", "OpenAI", "mini", 1.0, "closed"),
    ModelInfo("gpt-5.2-pro", "GPT-5.2 pro", "OpenAI", "pro", 2.0, "closed"),
    ModelInfo("Qwen/Qwen3-4B-Thinking-2507", "Qwen3-4B", "Qwen3", "4B", 4.0, "open"),
    ModelInfo("Qwen/Qwen3-30B-A3B-Thinking-2507", "Qwen3-30B-A3B", "Qwen3", "30B-A3B", 30.0, "open"),
    ModelInfo("Qwen/Qwen2.5-7B-Instruct-1M", "Qwen2.5-7B", "Qwen2.5", "7B", 7.0, "open"),
    ModelInfo("Qwen/Qwen2.5-14B-Instruct-1M", "Qwen2.5-14B", "Qwen2.5", "14B", 14.0, "open"),
    ModelInfo("Qwen/Qwen2.5-72B-Instruct-AWQ", "Qwen2.5-72B", "Qwen2.5", "72B", 72.0, "open"),
    ModelInfo("meta-llama/Meta-Llama-3.1-8B", "Llama-3.1-8B", "Llama 3.1", "8B", 8.0, "open"),
    ModelInfo("meta-llama/Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B-Inst.", "Llama 3.1", "8B-Inst.", 8.1, "open"),
    ModelInfo("meta-llama/Llama-3.1-70B-Instruct", "Llama-3.1-70B", "Llama 3.1", "70B", 70.0, "open"),
]

MODEL_INFO_BY_METHOD = {m.method: m for m in MODEL_INFOS}
DISPLAY_BY_METHOD = {m.method: m.display for m in MODEL_INFOS} | {
    "PC": "PC",
    "GES": "GES",
    "ENCO": "ENCO",
    "CausalLLMPrompt_names_only": "Causal-LLM prompt",
    "JiralerspongBFS": "BFS-style LLM",
    "TakayamaSCP": "Takayama SCP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("experiments/out/micad_eval_results/all_condition_metrics.csv"),
        help="Input condition-level metrics CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/out/micad_paper"),
        help="Output directory for paper-ready tables and plot data.",
    )
    parser.add_argument("--graph", default="sachs", choices=GRAPH_ORDER, help="Primary graph for headline tables.")
    parser.add_argument("--obs", type=int, default=5000, help="Primary observational budget N.")
    parser.add_argument("--inter", type=int, default=200, help="Primary interventional budget M.")
    parser.add_argument(
        "--dedup-policy",
        choices=["max-valid", "best-f1", "first"],
        default="max-valid",
        help="How to choose among duplicate source artifacts for one cell.",
    )
    parser.add_argument(
        "--variant-contains",
        default=None,
        help="Optional substring filter for LLM prompt variants; baselines and names_only are always kept.",
    )
    return parser.parse_args()


def load_metrics(path: Path, variant_contains: str | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {path}")
    df = pd.read_csv(path)
    required = {"graph", "method", "obs", "inter", "semantic", "format", "condition", "mean_f1", "mean_shd"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df = df[df["graph"].isin(GRAPH_ORDER)].copy()
    for col in ["obs", "inter", "mean_f1", "mean_shd", "std_f1", "std_shd", "valid_rows", "n_rows"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["mean_f1"].notna()].copy()
    df["condition"] = df["condition"].fillna(df.apply(infer_condition, axis=1))
    df["method_display"] = df["method"].map(display_method)
    df["is_data_only"] = df["method"].isin(DATA_ONLY_METHODS)
    df["is_llm_model"] = df["method"].isin(MODEL_INFO_BY_METHOD)

    if variant_contains:
        variant = df.get("variant", pd.Series("", index=df.index)).astype(str)
        keep = (
            df["is_data_only"]
            | df["condition"].eq("names_only")
            | variant.str.contains(variant_contains, case=False, na=False)
        )
        df = df[keep].copy()
    return df


def infer_condition(row: pd.Series) -> str:
    semantic = str(row.get("semantic", ""))
    fmt = str(row.get("format", ""))
    if semantic == "names_only" or fmt == "names_only":
        return "names_only"
    if fmt == "data_only":
        return f"{row.get('method', 'data')} data-only"
    if semantic in {"real", "anon"} and fmt:
        return f"{semantic}+{fmt}"
    return fmt or semantic or "unknown"


def display_method(method: str) -> str:
    if method in DISPLAY_BY_METHOD:
        return DISPLAY_BY_METHOD[method]
    return method.replace("Qwen/", "").replace("meta-llama/", "")


def source_priority(source_kind: object) -> int:
    source = str(source_kind)
    if source == "per_row":
        return 2
    if source == "summary_json":
        return 1
    return 0


def canonicalize(df: pd.DataFrame, dedup_policy: str) -> pd.DataFrame:
    df = df.copy()
    df["_source_priority"] = df.get("source_kind", "").map(source_priority)
    df["_valid_rows"] = df.get("valid_rows", 0).fillna(0)
    df["_n_rows"] = df.get("n_rows", 0).fillna(0)
    df["_path"] = df.get("path", "").fillna("").astype(str)

    group_cols = ["graph", "method", "obs", "inter", "condition"]
    rows = []
    for _, sub in df.groupby(group_cols, dropna=False, sort=False):
        if dedup_policy == "best-f1":
            chosen = sub.sort_values(
                ["mean_f1", "_valid_rows", "_source_priority", "_n_rows", "_path"],
                ascending=[False, False, False, False, True],
            ).iloc[0]
        elif dedup_policy == "first":
            chosen = sub.sort_values("_path").iloc[0]
        else:
            chosen = sub.sort_values(
                ["_valid_rows", "_source_priority", "_n_rows", "mean_f1", "_path"],
                ascending=[False, False, False, False, True],
            ).iloc[0]
        rows.append(chosen)
    out = pd.DataFrame(rows).drop(columns=[c for c in ["_source_priority", "_valid_rows", "_n_rows", "_path"] if c in df])
    out = out.sort_values(["graph", "method", "obs", "inter", "condition"]).reset_index(drop=True)
    return out


def get_cell(df: pd.DataFrame, graph: str, method: str, condition: str, obs: int, inter: int) -> pd.Series | None:
    sub = df[(df.graph == graph) & (df.method == method) & (df.condition == condition)]
    if condition != "names_only":
        sub = sub[(sub.obs == obs) & (sub.inter == inter)]
    if sub.empty:
        return None
    return sub.iloc[0]


def score(row: pd.Series | None, col: str = "mean_f1") -> float:
    if row is None:
        return np.nan
    val = row.get(col, np.nan)
    return float(val) if pd.notna(val) else np.nan


def fmt_f1_shd(f1: float, shd: float) -> str:
    if pd.isna(f1):
        return "--"
    if pd.isna(shd):
        return f"{f1:.2f}"
    return f"{f1:.2f} ({shd:.1f})"


def best_data_only(df: pd.DataFrame, graph: str, obs: int, inter: int) -> pd.Series | None:
    sub = df[(df.graph == graph) & (df.method.isin(DATA_ONLY_METHODS)) & (df.obs == obs) & (df.inter == inter)]
    if sub.empty:
        return None
    return sub.sort_values(["mean_f1", "valid_rows"], ascending=[False, False]).iloc[0]


def collect_evidence_ladder(df: pd.DataFrame, graph: str, obs: int, inter: int) -> pd.DataFrame:
    rows = []
    for info in MODEL_INFOS:
        cells = {cond: get_cell(df, graph, info.method, cond, obs, inter) for cond in CONDITION_ORDER}
        base = best_data_only(df, graph, obs, inter)
        values = {cond: score(row) for cond, row in cells.items()}
        shds = {cond: score(row, "mean_shd") for cond, row in cells.items()}
        best_data_f1 = score(base)
        llm_candidates = [values.get("real+summary", np.nan), values.get("real+matrix", np.nan)]
        finite_candidates = [v for v in llm_candidates if pd.notna(v)]
        best_llm_f1 = max(finite_candidates) if finite_candidates else np.nan
        rows.append(
            {
                "graph": graph,
                "method": info.method,
                "model": info.display,
                "family": info.family,
                "size_label": info.size_label,
                "size_value": info.size_value,
                "obs": obs,
                "inter": inter,
                "names_only_f1": values["names_only"],
                "real_summary_f1": values["real+summary"],
                "real_matrix_f1": values["real+matrix"],
                "anon_summary_f1": values["anon+summary"],
                "anon_matrix_f1": values["anon+matrix"],
                "names_only_shd": shds["names_only"],
                "real_summary_shd": shds["real+summary"],
                "real_matrix_shd": shds["real+matrix"],
                "anon_summary_shd": shds["anon+summary"],
                "anon_matrix_shd": shds["anon+matrix"],
                "best_llm_f1": best_llm_f1,
                "best_data_only_method": None if base is None else base.method,
                "best_data_only_f1": best_data_f1,
                "gap_to_data_only": best_data_f1 - best_llm_f1 if pd.notna(best_data_f1) and pd.notna(best_llm_f1) else np.nan,
                "best_data_format": best_format(values["real+summary"], values["real+matrix"]),
            }
        )
    return pd.DataFrame(rows)


def best_format(summary: float, matrix: float) -> str:
    if pd.isna(summary) and pd.isna(matrix):
        return "--"
    if pd.isna(summary):
        return "matrix"
    if pd.isna(matrix):
        return "summary"
    if abs(summary - matrix) < 1e-9:
        return "tie"
    return "matrix" if matrix > summary else "summary"


def collect_cross_graph(df: pd.DataFrame, obs: int, inter: int) -> pd.DataFrame:
    rows = []
    for graph in GRAPH_ORDER:
        ladder = collect_evidence_ladder(df, graph, obs, inter)
        rows.append(ladder)
    return pd.concat(rows, ignore_index=True)


def collect_contrast_metrics(cross: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in cross.iterrows():
        contrasts = {
            "summary_gain": row.real_summary_f1 - row.names_only_f1,
            "matrix_gain": row.real_matrix_f1 - row.names_only_f1,
            "format_gap_matrix_minus_summary": row.real_matrix_f1 - row.real_summary_f1,
            "anonymization_drop_summary": row.real_summary_f1 - row.anon_summary_f1,
            "anonymization_drop_matrix": row.real_matrix_f1 - row.anon_matrix_f1,
            "gap_to_data_only": row.gap_to_data_only,
        }
        for metric, value in contrasts.items():
            rows.append(
                {
                    "graph": row.graph,
                    "method": row.method,
                    "model": row.model,
                    "family": row.family,
                    "size_label": row.size_label,
                    "obs": row.obs,
                    "inter": row.inter,
                    "metric": metric,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def collect_model_size_ladder(cross: pd.DataFrame, graph: str) -> pd.DataFrame:
    rows = cross[cross.graph.eq(graph) & cross.family.isin(["Qwen2.5", "Qwen3", "Llama 3.1"])].copy()
    return rows.sort_values(["family", "size_value", "model"])


def collect_classical_baselines(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[df.method.isin(DATA_ONLY_METHODS)].copy()
    rows = rows[["graph", "method", "method_display", "obs", "inter", "mean_f1", "mean_shd", "path"]]
    return rows.sort_values(["graph", "obs", "inter", "method"])


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path}")


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


def write_latex_table(
    df: pd.DataFrame,
    path: Path,
    columns: list[str],
    headers: list[str],
    caption: str,
    label: str,
    align: str,
) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append("--" if pd.isna(val) else f"{val:.2f}")
            else:
                vals.append(latex_escape(val))
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {path}")


def make_tables(out_dir: Path, ladder: pd.DataFrame, size_ladder: pd.DataFrame, classical: pd.DataFrame, graph: str) -> None:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    headline = ladder.copy()
    headline = headline[
        [
            "model",
            "family",
            "names_only_f1",
            "real_summary_f1",
            "real_matrix_f1",
            "anon_summary_f1",
            "anon_matrix_f1",
            "best_data_only_method",
            "best_data_only_f1",
            "gap_to_data_only",
        ]
    ]
    write_latex_table(
        headline,
        table_dir / "headline_evidence_ladder.tex",
        ["model", "names_only_f1", "real_summary_f1", "real_matrix_f1", "best_data_only_method", "best_data_only_f1", "gap_to_data_only"],
        ["Model", "Names", "Summary", "Matrix", "Best classical", "Classical F1", "Gap"],
        f"Headline evidence-use ladder on {graph.capitalize()}. Values are directed-edge F1 at the selected data budget.",
        "tab:headline_evidence_ladder",
        "lrrrrrr",
    )

    size_table = size_ladder[
        ["family", "model", "names_only_f1", "real_summary_f1", "real_matrix_f1", "best_data_format"]
    ].copy()
    write_latex_table(
        size_table,
        table_dir / "model_size_ladder.tex",
        ["family", "model", "names_only_f1", "real_summary_f1", "real_matrix_f1", "best_data_format"],
        ["Family", "Model", "Names", "Summary", "Matrix", "Best format"],
        f"Within-family model-size comparison on {graph.capitalize()}.",
        "tab:model_size_ladder",
        "llrrrl",
    )

    classical_short = classical[classical.graph.eq(graph)].copy()
    classical_short["budget"] = classical_short.apply(lambda r: f"N={int(r.obs)}, M={int(r.inter)}", axis=1)
    write_latex_table(
        classical_short.head(30),
        table_dir / "classical_baselines.tex",
        ["method", "budget", "mean_f1", "mean_shd"],
        ["Method", "Budget", "F1", "SHD"],
        f"Classical causal discovery baselines on {graph.capitalize()}.",
        "tab:classical_baselines",
        "llrr",
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_metrics(args.metrics_csv, args.variant_contains)
    canonical = canonicalize(raw, args.dedup_policy)
    cross = collect_cross_graph(canonical, args.obs, args.inter)
    ladder = cross[cross.graph.eq(args.graph)].copy()
    contrasts = collect_contrast_metrics(cross)
    size_ladder = collect_model_size_ladder(cross, args.graph)
    classical = collect_classical_baselines(canonical)

    write_csv(canonical, args.out_dir / "canonical_condition_results.csv")
    write_csv(cross, args.out_dir / "paper_cross_graph_evidence_ladder.csv")
    write_csv(ladder, args.out_dir / "paper_headline_evidence_ladder.csv")
    write_csv(contrasts, args.out_dir / "paper_contrast_metrics.csv")
    write_csv(size_ladder, args.out_dir / "paper_model_size_ladder.csv")
    write_csv(classical, args.out_dir / "paper_classical_baselines.csv")

    make_tables(args.out_dir, ladder, size_ladder, classical, args.graph)

    manifest = pd.DataFrame(
        [
            {"key": "input", "value": str(args.metrics_csv)},
            {"key": "graph", "value": args.graph},
            {"key": "obs", "value": args.obs},
            {"key": "inter", "value": args.inter},
            {"key": "dedup_policy", "value": args.dedup_policy},
            {"key": "variant_contains", "value": args.variant_contains or ""},
            {"key": "canonical_rows", "value": len(canonical)},
        ]
    )
    write_csv(manifest, args.out_dir / "paper_results_manifest.csv")


if __name__ == "__main__":
    main()
