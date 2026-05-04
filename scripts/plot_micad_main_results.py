#!/usr/bin/env python3
"""Create main-paper MICAD result visualizations.

Inputs are produced by notebooks/collect_micad_eval_results.ipynb:
  experiments/out/micad_eval_results/all_condition_metrics.csv

Figures written by default:
  experiments/out/micad_eval_results/figures/lead_matched_control.{pdf,png}
  experiments/out/micad_eval_results/figures/cross_graph_heatmap_<model>.{pdf,png}
  experiments/out/micad_eval_results/figures/contrastive_metrics.{pdf,png}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GRAPH_ORDER = ["cancer", "earthquake", "asia", "sachs"]
CONDITION_ORDER = ["names_only", "real+summary", "anon+summary", "real+matrix", "anon+matrix"]
CONDITION_LABELS = {
    "names_only": "Names\nonly",
    "real+summary": "Real\nsummary",
    "anon+summary": "Anon.\nsummary",
    "real+matrix": "Real\nmatrix",
    "anon+matrix": "Anon.\nmatrix",
    "best_data_only": "Best\ndata-only",
}
DATA_ONLY_METHODS = ["PC", "GES", "ENCO"]

DISPLAY_MODEL = {
    "gpt-5-mini": "GPT-5 mini",
    "gpt-5.2-pro": "GPT-5.2 pro",
    "Qwen/Qwen3-4B-Thinking-2507": "Qwen3-4B",
    "Qwen/Qwen2.5-7B-Instruct-1M": "Qwen2.5-7B",
    "Qwen/Qwen2.5-14B-Instruct-1M": "Qwen2.5-14B",
    "Qwen/Qwen3-30B-A3B-Thinking-2507": "Qwen3-30B-A3B",
    "Qwen/Qwen2.5-72B-Instruct-AWQ": "Qwen2.5-72B",
    "meta-llama/Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-8B": "Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B-Inst.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("experiments/out/micad_eval_results/all_condition_metrics.csv"),
        help="Aggregated condition metrics CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/out/micad_eval_results/figures"),
        help="Directory for output figures.",
    )
    parser.add_argument("--budget", nargs=2, type=int, default=[5000, 200], metavar=("N", "M"))
    parser.add_argument("--lead-graph", default="sachs", choices=GRAPH_ORDER)
    parser.add_argument("--lead-model", default="gpt-5.2-pro")
    parser.add_argument(
        "--heatmap-models",
        nargs="*",
        default=["gpt-5-mini", "gpt-5.2-pro", "Qwen/Qwen2.5-14B-Instruct-1M", "Qwen/Qwen2.5-72B-Instruct-AWQ"],
        help="Models to render cross-graph heatmaps for.",
    )
    parser.add_argument(
        "--contrast-models",
        nargs="*",
        default=["gpt-5-mini", "gpt-5.2-pro", "Qwen/Qwen2.5-14B-Instruct-1M", "Qwen/Qwen2.5-72B-Instruct-AWQ"],
        help="Models to include in contrastive metric plot.",
    )
    parser.add_argument(
        "--variant-contains",
        default=None,
        help="Optional substring filter on the provenance/variant field, e.g. reasonconcise.",
    )
    parser.add_argument(
        "--selection",
        choices=["best", "first"],
        default="best",
        help="How to resolve duplicate files for the same graph/model/condition/budget.",
    )
    parser.add_argument("--formats", nargs="*", default=["pdf", "png"], choices=["pdf", "png", "svg"])
    return parser.parse_args()


def display_model(model: str) -> str:
    return DISPLAY_MODEL.get(model, model.replace("Qwen/", "").replace("meta-llama/", ""))


def load_metrics(path: Path, variant_contains: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"graph", "method", "condition", "obs", "inter", "mean_f1", "mean_shd"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df = df[df["graph"].isin(GRAPH_ORDER)].copy()
    df = df[pd.to_numeric(df["mean_f1"], errors="coerce").notna()].copy()
    df["mean_f1"] = pd.to_numeric(df["mean_f1"], errors="coerce")
    df["mean_shd"] = pd.to_numeric(df["mean_shd"], errors="coerce")
    if variant_contains:
        keep_baseline = df["method"].isin(DATA_ONLY_METHODS)
        keep_names = df["condition"].eq("names_only")
        keep_variant = df.get("variant", pd.Series("", index=df.index)).astype(str).str.contains(variant_contains, case=False, na=False)
        df = df[keep_baseline | keep_names | keep_variant].copy()
    return df


def choose_row(df: pd.DataFrame, selection: str = "best") -> pd.Series | None:
    if df.empty:
        return None
    if selection == "best":
        return df.sort_values(["mean_f1", "valid_rows"], ascending=[False, False]).iloc[0]
    return df.sort_values("path").iloc[0]


def row_for_condition(
    df: pd.DataFrame,
    graph: str,
    method: str,
    condition: str,
    budget: tuple[int, int],
    selection: str,
) -> pd.Series | None:
    sub = df[(df.graph == graph) & (df.method == method) & (df.condition == condition)]
    if condition != "names_only":
        n, m = budget
        sub = sub[(sub.obs == n) & (sub.inter == m)]
    return choose_row(sub, selection)


def best_data_only_row(df: pd.DataFrame, graph: str, budget: tuple[int, int], selection: str) -> pd.Series | None:
    n, m = budget
    sub = df[(df.graph == graph) & (df.method.isin(DATA_ONLY_METHODS)) & (df.obs == n) & (df.inter == m)]
    if sub.empty and m > 0:
        # Fallback: use any ENCO result at the same graph and largest available intervention budget.
        sub = df[(df.graph == graph) & (df.method == "ENCO")]
    elif sub.empty:
        sub = df[(df.graph == graph) & (df.method.isin(["PC", "GES"]))]
    return choose_row(sub, selection)


def collect_plot_rows(df: pd.DataFrame, graph: str, method: str, budget: tuple[int, int], selection: str) -> list[dict]:
    rows: list[dict] = []
    for condition in CONDITION_ORDER:
        row = row_for_condition(df, graph, method, condition, budget, selection)
        rows.append({
            "condition": condition,
            "label": CONDITION_LABELS[condition],
            "mean_f1": np.nan if row is None else row.mean_f1,
            "mean_shd": np.nan if row is None else row.mean_shd,
            "method": method,
            "source_method": method,
            "path": None if row is None else row.path,
        })
    base = best_data_only_row(df, graph, budget, selection)
    rows.append({
        "condition": "best_data_only",
        "label": CONDITION_LABELS["best_data_only"] if base is None else f"{base.method}\ndata-only",
        "mean_f1": np.nan if base is None else base.mean_f1,
        "mean_shd": np.nan if base is None else base.mean_shd,
        "method": method,
        "source_method": None if base is None else base.method,
        "path": None if base is None else base.path,
    })
    return rows


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: Iterable[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"[write] {path}")


def plot_lead_matched_control(df: pd.DataFrame, args: argparse.Namespace) -> None:
    budget = tuple(args.budget)
    rows = collect_plot_rows(df, args.lead_graph, args.lead_model, budget, args.selection)
    plot_df = pd.DataFrame(rows)
    colors = ["#8e77b8", "#8bc28c", "#bcd7f0", "#6aaed6", "#d9e8f5", "#f2a65a"]

    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    x = np.arange(len(plot_df))
    heights = plot_df["mean_f1"].to_numpy(dtype=float)
    bars = ax.bar(x, np.nan_to_num(heights, nan=0.0), color=colors, edgecolor="black", linewidth=0.5)
    for bar, f1, shd in zip(bars, plot_df["mean_f1"], plot_df["mean_shd"]):
        if pd.notna(f1):
            ax.text(bar.get_x() + bar.get_width() / 2, f1 + 0.025, f"{f1:.2f}\nSHD {shd:.0f}" if pd.notna(shd) else f"{f1:.2f}",
                    ha="center", va="bottom", fontsize=8)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.05, "missing", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x, plot_df["label"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Directed-edge F1")
    ax.set_title(f"Matched controls on {args.lead_graph.capitalize()} ({display_model(args.lead_model)}, N={budget[0]}, M={budget[1]})")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, args.out_dir, "lead_matched_control", args.formats)
    plt.close(fig)

    provenance = plot_df[["condition", "source_method", "mean_f1", "mean_shd", "path"]]
    provenance.to_csv(args.out_dir / "lead_matched_control_sources.csv", index=False)
    print(f"[write] {args.out_dir / 'lead_matched_control_sources.csv'}")


def plot_cross_graph_heatmaps(df: pd.DataFrame, args: argparse.Namespace) -> None:
    budget = tuple(args.budget)
    for method in args.heatmap_models:
        values = []
        annot = []
        for graph in GRAPH_ORDER:
            rows = collect_plot_rows(df, graph, method, budget, args.selection)
            values.append([r["mean_f1"] for r in rows])
            annot.append(["" if pd.isna(r["mean_f1"]) else f"{r['mean_f1']:.2f}" for r in rows])
        mat = np.array(values, dtype=float)
        fig, ax = plt.subplots(figsize=(8.8, 3.1))
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(np.arange(len(CONDITION_ORDER) + 1), [CONDITION_LABELS[c].replace("\n", " ") for c in CONDITION_ORDER] + ["Best data-only"], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(GRAPH_ORDER)), [g.capitalize() for g in GRAPH_ORDER])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                txt = "--" if np.isnan(mat[i, j]) else f"{mat[i, j]:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")
        ax.set_title(f"Cross-graph matched cells: {display_model(method)} (N={budget[0]}, M={budget[1]})")
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Directed-edge F1")
        save_figure(fig, args.out_dir, f"cross_graph_heatmap_{safe_name(method)}", args.formats)
        plt.close(fig)


def safe_name(s: str) -> str:
    return re_sub_nonword(s)


def re_sub_nonword(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def compute_contrast_summary(df: pd.DataFrame, models: list[str], budget: tuple[int, int], selection: str) -> pd.DataFrame:
    rows = []
    for method in models:
        for graph in GRAPH_ORDER:
            name = row_for_condition(df, graph, method, "names_only", budget, selection)
            real_sum = row_for_condition(df, graph, method, "real+summary", budget, selection)
            anon_sum = row_for_condition(df, graph, method, "anon+summary", budget, selection)
            real_mat = row_for_condition(df, graph, method, "real+matrix", budget, selection)
            anon_mat = row_for_condition(df, graph, method, "anon+matrix", budget, selection)
            score = lambda r: np.nan if r is None else r.mean_f1
            name_s, rs, ans, rm, anm = map(score, [name, real_sum, anon_sum, real_mat, anon_mat])
            rows.extend([
                {"method": method, "graph": graph, "metric": "Mixed-info gain", "value": rs - name_s if pd.notna(rs) and pd.notna(name_s) else np.nan},
                {"method": method, "graph": graph, "metric": "Anonymization drop", "value": rs - ans if pd.notna(rs) and pd.notna(ans) else np.nan},
                {"method": method, "graph": graph, "metric": "Format gap", "value": rs - rm if pd.notna(rs) and pd.notna(rm) else np.nan},
            ])
    cdf = pd.DataFrame(rows)
    return cdf.groupby(["method", "metric"], as_index=False).agg(mean_value=("value", "mean"), n=("value", lambda x: int(x.notna().sum())))


def plot_contrastive_metrics(df: pd.DataFrame, args: argparse.Namespace) -> None:
    budget = tuple(args.budget)
    cdf = compute_contrast_summary(df, args.contrast_models, budget, args.selection)
    cdf.to_csv(args.out_dir / "contrastive_metrics_plot_data.csv", index=False)
    metrics = ["Mixed-info gain", "Anonymization drop", "Format gap"]
    models = args.contrast_models
    x = np.arange(len(models))
    width = 0.24
    colors = {"Mixed-info gain": "#4c78a8", "Anonymization drop": "#f58518", "Format gap": "#54a24b"}

    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    for offset, metric in zip([-width, 0, width], metrics):
        vals = []
        ns = []
        for model in models:
            sub = cdf[(cdf.method == model) & (cdf.metric == metric)]
            vals.append(np.nan if sub.empty else sub.iloc[0].mean_value)
            ns.append(0 if sub.empty else int(sub.iloc[0].n))
        bars = ax.bar(x + offset, vals, width=width, label=metric, color=colors[metric], edgecolor="black", linewidth=0.4)
        for bar, val, n in zip(bars, vals, ns):
            if pd.notna(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + (0.015 if val >= 0 else -0.035), f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [display_model(m) for m in models], rotation=20, ha="right")
    ax.set_ylabel("Mean F1 difference across available graphs")
    ax.set_title(f"Contrastive metrics across matched cells (N={budget[0]}, M={budget[1]})")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, args.out_dir, "contrastive_metrics", args.formats)
    plt.close(fig)
    print(f"[write] {args.out_dir / 'contrastive_metrics_plot_data.csv'}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_metrics(args.input_csv, args.variant_contains)
    print(f"Loaded {len(df)} scored rows from {args.input_csv}")
    plot_lead_matched_control(df, args)
    plot_cross_graph_heatmaps(df, args)
    plot_contrastive_metrics(df, args)


if __name__ == "__main__":
    main()
