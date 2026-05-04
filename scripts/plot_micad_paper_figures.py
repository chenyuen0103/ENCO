#!/usr/bin/env python3
"""Generate MICAD paper-quality result figures from collected paper tables.

Run scripts/collect_micad_paper_results.py first.  This script reads the
paper-ready CSVs from experiments/out/micad_paper and creates a compact figure
set aimed at a top-tier benchmark paper:

  1. Decisive evidence ladder: names_only vs summary vs matrix vs classical.
  2. Model-size ladder: same-family scale comparison.
  3. Contrast heatmap: what information each model uses.
  4. Cross-graph heatmap: matched controls across the real reference graphs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


GRAPH_ORDER = ["cancer", "earthquake", "asia", "sachs"]
CONDITION_COLUMNS = ["names_only_f1", "real_summary_f1", "real_matrix_f1"]
CONDITION_LABELS = {
    "names_only_f1": "Names only",
    "real_summary_f1": "Summary",
    "real_matrix_f1": "Matrix",
}
CONDITION_COLORS = {
    "names_only_f1": "#7b5ea7",
    "real_summary_f1": "#59a14f",
    "real_matrix_f1": "#4e79a7",
}
CONTRAST_LABELS = {
    "summary_gain": "Summary gain",
    "matrix_gain": "Matrix gain",
    "format_gap_matrix_minus_summary": "Matrix - summary",
    "anonymization_drop_summary": "Name drop\n(summary)",
    "anonymization_drop_matrix": "Name drop\n(matrix)",
    "gap_to_data_only": "Gap to\nclassical",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("experiments/out/micad_paper"),
        help="Directory produced by collect_micad_paper_results.py.",
    )
    parser.add_argument("--graph", default="sachs", choices=GRAPH_ORDER)
    parser.add_argument(
        "--lead-models",
        nargs="*",
        default=["GPT-5 mini", "GPT-5.2 pro", "Qwen2.5-7B", "Qwen2.5-14B", "Qwen2.5-72B"],
        help="Display names to include in the decisive evidence ladder.",
    )
    parser.add_argument(
        "--heatmap-models",
        nargs="*",
        default=["GPT-5 mini", "GPT-5.2 pro", "Qwen2.5-7B", "Qwen2.5-14B", "Qwen2.5-72B"],
        help="Display names to include in the contrast heatmap.",
    )
    parser.add_argument("--formats", nargs="*", default=["pdf", "png"], choices=["pdf", "png", "svg"])
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: Iterable[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path)
        print(f"[write] {path}")


def finite_or_nan(value: object) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return np.nan
    return val if np.isfinite(val) else np.nan


def plot_decisive_evidence_ladder(headline: pd.DataFrame, out_dir: Path, formats: list[str], lead_models: list[str]) -> None:
    plot_df = headline[headline["model"].isin(lead_models)].copy()
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=lead_models, ordered=True)
    plot_df = plot_df.sort_values("model")
    if plot_df.empty:
        raise ValueError("No rows available for requested lead models.")

    x = np.arange(len(plot_df))
    width = 0.22
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for idx, cond in enumerate(CONDITION_COLUMNS):
        offset = (idx - 1) * width
        vals = plot_df[cond].map(finite_or_nan).to_numpy()
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            label=CONDITION_LABELS[cond],
            color=CONDITION_COLORS[cond],
            edgecolor="black",
            linewidth=0.35,
        )
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.018, f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    classical_vals = plot_df["best_data_only_f1"].map(finite_or_nan)
    if classical_vals.notna().any():
        classical = float(classical_vals.dropna().iloc[0])
        method = str(plot_df["best_data_only_method"].dropna().iloc[0])
        ax.axhline(classical, color="#e15759", linewidth=1.2, linestyle="--", label=f"{method} classical")
        ax.text(len(plot_df) - 0.45, classical + 0.018, f"{method} {classical:.2f}", color="#b23b3b", ha="right", va="bottom", fontsize=8)

    ax.set_xticks(x, plot_df["model"])
    ax.set_ylabel("Directed-edge F1")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    save_figure(fig, out_dir, "fig1_decisive_evidence_ladder", formats)
    plt.close(fig)


def plot_model_size_ladder(size_df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    rows = size_df[size_df["family"].isin(["Qwen2.5", "Qwen3", "Llama 3.1"])].copy()
    rows = rows.dropna(subset=["size_value"], how="any")
    families = []
    for family in ["Qwen2.5", "Qwen3", "Llama 3.1"]:
        sub = rows[rows.family.eq(family)]
        finite_points = sub[CONDITION_COLUMNS].notna().any(axis=1).sum()
        if finite_points >= 2:
            families.append(family)
    if not families:
        raise ValueError("No model-size ladder rows available.")

    fig, axes = plt.subplots(1, len(families), figsize=(3.0 * len(families), 2.7), sharey=True)
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        sub = rows[rows.family.eq(family)].sort_values("size_value")
        x = np.arange(len(sub))
        for cond in CONDITION_COLUMNS:
            vals = sub[cond].map(finite_or_nan).to_numpy()
            ax.plot(
                x,
                vals,
                marker="o",
                linewidth=1.4,
                markersize=4.0,
                label=CONDITION_LABELS[cond],
                color=CONDITION_COLORS[cond],
            )
        ax.set_xticks(x, sub["size_label"])
        ax.set_xlabel(family)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    axes[0].set_ylabel("Directed-edge F1")
    axes[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.02))
    save_figure(fig, out_dir, "fig2_model_size_ladders", formats)
    plt.close(fig)


def plot_contrast_heatmap(contrasts: pd.DataFrame, out_dir: Path, formats: list[str], graph: str, heatmap_models: list[str]) -> None:
    metrics = [
        "summary_gain",
        "matrix_gain",
        "format_gap_matrix_minus_summary",
        "anonymization_drop_summary",
        "anonymization_drop_matrix",
        "gap_to_data_only",
    ]
    sub = contrasts[contrasts.graph.eq(graph) & contrasts.model.isin(heatmap_models) & contrasts.metric.isin(metrics)].copy()
    pivot = sub.pivot_table(index="model", columns="metric", values="value", aggfunc="mean")
    pivot = pivot.reindex(index=heatmap_models, columns=metrics)
    mat = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.8, 3.0))
    lim = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    lim = max(0.25, float(lim))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)), [CONTRAST_LABELS[m] for m in metrics], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(heatmap_models)), heatmap_models)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(j, i, "--" if not np.isfinite(val) else f"{val:.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("F1 difference")
    save_figure(fig, out_dir, "fig3_information_use_heatmap", formats)
    plt.close(fig)


def best_cross_graph_model(cross: pd.DataFrame, preferred: list[str]) -> str:
    score = {}
    cols = ["names_only_f1", "real_summary_f1", "real_matrix_f1", "best_data_only_f1"]
    for model, sub in cross.groupby("model"):
        graph_count = sub[sub[cols].notna().any(axis=1)]["graph"].nunique()
        cell_count = int(sub[cols].notna().sum().sum())
        preference = len(preferred) - preferred.index(model) if model in preferred else 0
        score[model] = (graph_count, cell_count, preference)
    if not score:
        raise ValueError("No models found in cross-graph data.")
    return max(score, key=score.get)


def plot_cross_graph_heatmap(cross: pd.DataFrame, out_dir: Path, formats: list[str], model: str) -> None:
    sub = cross[cross.model.eq(model)].copy()
    if sub.empty:
        raise ValueError(f"No cross-graph rows found for model {model!r}")
    cols = ["names_only_f1", "real_summary_f1", "real_matrix_f1", "best_data_only_f1"]
    labels = ["Names", "Summary", "Matrix", "Classical"]
    pivot = sub.set_index("graph").reindex(GRAPH_ORDER)[cols]
    mat = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(cols)), labels)
    ax.set_yticks(np.arange(len(GRAPH_ORDER)), [g.capitalize() for g in GRAPH_ORDER])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(j, i, "--" if not np.isfinite(val) else f"{val:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Directed-edge F1")
    save_figure(fig, out_dir, "fig4_cross_graph_matched_controls", formats)
    plt.close(fig)


def write_latex_includes(out_dir: Path) -> None:
    text = r"""
% Generated by scripts/plot_micad_paper_figures.py

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.92\textwidth]{experiments/out/micad_paper/figures/fig1_decisive_evidence_ladder.pdf}
  \caption{Matched evidence-use ladder on the primary graph and budget. Each model is evaluated on the same graph and data; only the exposed information changes from names-only to summary statistics to raw matrix rows. The dashed line shows the strongest classical causal discovery reference at the same data budget.}
  \label{fig:decisive_evidence_ladder}
\end{figure*}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{experiments/out/micad_paper/figures/fig2_model_size_ladders.pdf}
  \caption{Within-family model-size ladders. Larger models sometimes use supplied data more effectively, but the trend is not monotonic across matched cells.}
  \label{fig:model_size_ladders}
\end{figure}

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.92\textwidth]{experiments/out/micad_paper/figures/fig3_information_use_heatmap.pdf}
  \caption{Contrastive information-use metrics. Each cell is a matched F1 difference, showing whether performance changes with supplied data, data representation, semantic cues, or distance from the classical baseline.}
  \label{fig:information_use_heatmap}
\end{figure*}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{experiments/out/micad_paper/figures/fig4_cross_graph_matched_controls.pdf}
  \caption{Optional appendix coverage view for one representative model. Rows hold the graph fixed; columns expose semantic-only, data-bearing, and classical reference conditions. Sparse entries indicate cells that should be completed before using this as a main-paper cross-graph claim.}
  \label{fig:cross_graph_matched_controls}
\end{figure}
"""
    path = out_dir / "latex_includes.tex"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"[write] {path}")


def main() -> None:
    args = parse_args()
    configure_style()
    out_dir = args.paper_dir / "figures"
    headline = read_csv(args.paper_dir / "paper_headline_evidence_ladder.csv")
    size_df = read_csv(args.paper_dir / "paper_model_size_ladder.csv")
    contrasts = read_csv(args.paper_dir / "paper_contrast_metrics.csv")
    cross = read_csv(args.paper_dir / "paper_cross_graph_evidence_ladder.csv")

    plot_decisive_evidence_ladder(headline, out_dir, args.formats, args.lead_models)
    plot_model_size_ladder(size_df, out_dir, args.formats)
    plot_contrast_heatmap(contrasts, out_dir, args.formats, args.graph, args.heatmap_models)
    cross_model = best_cross_graph_model(cross, args.lead_models)
    plot_cross_graph_heatmap(cross, out_dir, args.formats, model=cross_model)
    write_latex_includes(out_dir)


if __name__ == "__main__":
    main()
