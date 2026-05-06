#!/usr/bin/env python3
"""Generate MICAD paper-quality result figures from collected paper tables.

Run scripts/collect_micad_paper_results.py first.  This script reads the
paper-ready CSVs from experiments/out/micad_paper and creates a compact figure
set aimed at a top-tier benchmark paper:

  1. Decisive evidence ladder: names_only vs summary vs matrix vs classical.
  2. Contrast bars: what information each model uses.
  3. Cross-graph profiles: matched controls across the real and anonymized reference graphs.

Each comparison figure is emitted twice: pretrained models and Qwen3-4B vs
Qwen3-4B-FT.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_ORDER = ["cancer", "earthquake", "asia", "sachs"]
ANONYMIZATION_STRESS_MODELS = [
    "GPT-5.2 Pro",
    "GPT-5 Mini",
    "Qwen2.5-72B",
    "Qwen2.5-14B",
    "Qwen2.5-7B",
]
PRETRAINED_MODELS = [
    "GPT-5 Mini",
    "GPT-5.2 Pro",
    "Qwen3-4B",
    "Qwen2.5-72B",
    "Qwen2.5-14B",
    "Qwen2.5-7B",
]
QWEN3_POSTTRAINING_MODELS = [
    "Qwen3-4B",
    "Qwen3-4B-FT",
]
DEFAULT_LEAD_MODELS = [
    *PRETRAINED_MODELS,
    "Qwen3-4B-FT",
]
COMPARISON_SETS = [
    ("pretrained", PRETRAINED_MODELS),
    ("qwen3_ft", QWEN3_POSTTRAINING_MODELS),
]
MODEL_DISPLAY_ALIASES = {
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-5.2-pro": "GPT-5.2 Pro",
    "Qwen/Qwen3-4B-Thinking-2507": "Qwen3-4B",
    "Qwen3-4B-Thinking-2507": "Qwen3-4B",
    "grpo_from_qwen3_4b_cd_format_v5_rerun_no_cancer_full_checkpoint-1200": (
        "Qwen3-4B-FT"
    ),
    "grpo_from_qwen3_4b_cd_format_v5_rerun_no_cancer_full_checkpoint-1200_merged": (
        "Qwen3-4B-FT"
    ),
    "GRPO CD no-cancer ckpt-1200 merged": "Qwen3-4B-FT",
    "qwen3_4b_cd_format_v5_rerun_2gpu_checkpoint-100": "CD v5 ckpt-100 merged",
    "qwen3_4b_cd_format_v5_rerun_2gpu_checkpoint-100_merged": "CD v5 ckpt-100 merged",
}
ALWAYS_SHOW_MODELS = {"Qwen3-4B"}
PLOT_EXCLUDED_MODELS = {"CD v5 ckpt-100 merged"}
FIG1_SIZE = (7.2, 3.0)
F1_YLIM = (0.0, 1.08)
SEMANTIC_SCALING_GRAPHS = ["cancer", "asia", "sachs", "child", "alarm"]
SEMANTIC_SCALING_LABELS = {
    "cancer": "Cancer",
    "asia": "Asia",
    "sachs": "Sachs",
    "child": "Child",
    "alarm": "Alarm",
}
CONDITION_COLUMNS = ["names_only_f1", "real_summary_f1", "real_matrix_f1"]
CONDITION_ERROR_COLUMNS = {
    "names_only_f1": "names_only_f1_se",
    "real_summary_f1": "real_summary_f1_se",
    "real_matrix_f1": "real_matrix_f1_se",
    "anon_summary_f1": "anon_summary_f1_se",
    "anon_matrix_f1": "anon_matrix_f1_se",
}
CONDITION_VALID_COLUMNS = {
    "names_only_f1": "names_only_valid_rows",
    "real_summary_f1": "real_summary_valid_rows",
    "real_matrix_f1": "real_matrix_valid_rows",
    "anon_summary_f1": "anon_summary_valid_rows",
    "anon_matrix_f1": "anon_matrix_valid_rows",
}
CONDITION_TOTAL_COLUMNS = {
    "names_only_f1": "names_only_n_rows",
    "real_summary_f1": "real_summary_n_rows",
    "real_matrix_f1": "real_matrix_n_rows",
    "anon_summary_f1": "anon_summary_n_rows",
    "anon_matrix_f1": "anon_matrix_n_rows",
}
CONDITION_LABELS = {
    "names_only_f1": "Names only",
    "real_summary_f1": "Summary",
    "real_matrix_f1": "Matrix",
    "anon_summary_f1": "Anon. summary",
    "anon_matrix_f1": "Anon. matrix",
}
CONDITION_COLORS = {
    "names_only_f1": "#7b5ea7",
    "real_summary_f1": "#59a14f",
    "real_matrix_f1": "#4e79a7",
    "anon_summary_f1": "#59a14f",
    "anon_matrix_f1": "#4e79a7",
}
CONTRAST_LABELS = {
    "summary_gain": "Summary gain",
    "matrix_gain": "Matrix gain",
    "interventional_summary_gain": "Interventional summary gain",
    "interventional_matrix_gain": "Interventional matrix gain",
    "observational_summary_gain": "Observational summary gain",
    "observational_matrix_gain": "Observational matrix gain",
    "format_gap_matrix_minus_summary": "Matrix - summary",
    "anonymization_drop_summary": "Name drop\n(summary)",
    "anonymization_drop_matrix": "Name drop\n(matrix)",
    "gap_to_data_only": "Gap to\nclassical",
}

MIN_CONTRAST_VALID_RUNS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("experiments/out/micad_paper"),
        help="Directory produced by collect_micad_paper_results.py.",
    )
    parser.add_argument(
        "--graph",
        default=None,
        choices=GRAPH_ORDER,
        help="Single graph to plot. Defaults to all available reference graphs.",
    )
    parser.add_argument(
        "--graphs",
        nargs="*",
        default=None,
        help=(
            "One or more graphs to plot, e.g. --graphs asia sachs. "
            "Use --graphs all to plot all available reference graphs."
        ),
    )
    parser.add_argument(
        "--lead-models",
        nargs="*",
        default=DEFAULT_LEAD_MODELS,
        help=(
            "Display names to include in the decisive evidence ladder. "
            "Defaults to GPT-5 Mini, GPT-5.2 Pro, Qwen3-4B, and the selected fine-tuned checkpoints; "
            "use --lead-models all for all models."
        ),
    )
    parser.add_argument(
        "--contrast-models",
        "--heatmap-models",
        dest="contrast_models",
        nargs="*",
        default=QWEN3_POSTTRAINING_MODELS,
        help=(
            "Display names to include in the contrast bar plot. "
            "Defaults to Qwen3-4B plus the selected fine-tuned checkpoints; use --contrast-models all for all models."
        ),
    )
    parser.add_argument("--formats", nargs="*", default=["pdf", "png"], choices=["pdf", "png", "svg"])
    return parser.parse_args()


def resolve_graphs(args: argparse.Namespace, available_graphs: Iterable[str]) -> list[str]:
    available = set(available_graphs)
    if args.graphs is not None and len(args.graphs) > 0:
        requested = args.graphs
    elif args.graph is not None:
        requested = [args.graph]
    else:
        requested = ["all"]
    if len(requested) == 1 and requested[0].lower() == "all":
        return [graph for graph in GRAPH_ORDER if graph in available]

    graphs: list[str] = []
    for graph in requested:
        graph = graph.lower()
        if graph not in GRAPH_ORDER:
            raise ValueError(f"Unknown graph {graph!r}. Choose from {GRAPH_ORDER} or use 'all'.")
        if graph not in available:
            print(f"[warn] Requested graph {graph!r} is absent from collected inputs; skipping.")
            continue
        graphs.append(graph)
    if not graphs:
        raise ValueError("No requested graphs are available in the collected inputs.")
    return graphs


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
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.framealpha": 0.0,
            "legend.edgecolor": "none",
            "legend.borderpad": 0.3,
            "legend.handlelength": 1.5,
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


def fmt_value(value: float) -> str:
    return "1" if np.isclose(value, 1.0, atol=1e-9) else f"{value:.2f}"


def fmt_signed_value(value: float) -> str:
    return "+1" if np.isclose(value, 1.0, atol=1e-9) else f"{value:+.2f}"


def _stem(base: str, graph: str, *, legacy: bool) -> str:
    return base if legacy else f"{base}_{graph}"


def _stem_with_suffix(stem: str, stem_suffix: str | None) -> str:
    return f"{stem}_{stem_suffix}" if stem_suffix else stem


def expand_model_patterns(models: list[str], requested: list[str]) -> list[str]:
    selected: list[str] = []
    unmatched: list[str] = []
    for token in requested:
        patterns = [part.strip() for part in str(token).split(",") if part.strip()]
        for pattern in patterns:
            pattern = MODEL_DISPLAY_ALIASES.get(pattern, pattern)
            matches = [model for model in models if fnmatch.fnmatchcase(model, pattern)]
            if not matches:
                unmatched.append(pattern)
                continue
            for match in matches:
                if match not in selected:
                    selected.append(match)
    if unmatched:
        print(f"[warn] No models matched requested pattern(s): {', '.join(unmatched)}")
    return selected


def available_models(df: pd.DataFrame, requested: list[str] | None = None) -> list[str]:
    models = [str(model) for model in df["model"].dropna().unique()]
    if not requested:
        return models
    if len(requested) == 1 and str(requested[0]).lower() == "all":
        return models
    return expand_model_patterns(models, requested)


def apply_model_display_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["model", "method_display"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(MODEL_DISPLAY_ALIASES)
    return df


def drop_models_without_valid_responses(
    plot_df: pd.DataFrame,
    selected_models: list[str],
    condition_columns: list[str],
    *,
    graph: str,
    figure_label: str,
    preserve_selected_models: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop models with fewer than half valid displayed bars."""
    validity_by_condition = []
    for cond in condition_columns:
        valid_col = CONDITION_VALID_COLUMNS.get(cond)
        if valid_col in plot_df.columns:
            valid = pd.to_numeric(plot_df[valid_col], errors="coerce").gt(0)
        else:
            valid = np.isfinite(pd.to_numeric(plot_df[cond], errors="coerce").to_numpy(dtype=float))
            valid = pd.Series(valid, index=plot_df.index)
        validity_by_condition.append(valid)

    if validity_by_condition:
        valid = pd.concat(validity_by_condition, axis=1)
        min_valid_bars = (len(condition_columns) + 1) // 2
        valid_bar_count = valid.sum(axis=1)
        keep = valid_bar_count.ge(min_valid_bars)
    else:
        min_valid_bars = 1
        keep = pd.Series(False, index=plot_df.index)

    protected_models = set(selected_models) if preserve_selected_models else ALWAYS_SHOW_MODELS
    protected = plot_df["model"].astype(str).isin(protected_models)
    keep = keep | protected

    dropped = plot_df.loc[~keep, "model"].dropna().astype(str).tolist()
    if dropped:
        print(
            f"[info] Dropping {len(dropped)} model(s) from {figure_label} "
            f"for graph={graph} with fewer than {min_valid_bars}/"
            f"{len(condition_columns)} valid displayed bars: "
            + ", ".join(dropped)
        )

    filtered = plot_df.loc[keep].copy()
    kept = set(filtered["model"].dropna().astype(str))
    return filtered, [model for model in selected_models if model in kept]


def plot_decisive_evidence_ladder(
    headline: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    lead_models: list[str] | None,
    graph: str,
    *,
    legacy_stem: bool,
    anonymized: bool = False,
    stem_suffix: str | None = None,
) -> None:
    graph_df = headline[headline["graph"].eq(graph)].copy() if "graph" in headline.columns else headline.copy()
    lead_models = available_models(graph_df, lead_models)
    plot_df = graph_df[graph_df["model"].isin(lead_models)].copy()
    semantic = "anon" if anonymized else "real"
    condition_columns = (
        [f"{semantic}_summary_f1", f"{semantic}_matrix_f1"]
        if anonymized
        else ["names_only_f1", f"{semantic}_summary_f1", f"{semantic}_matrix_f1"]
    )
    missing = [col for col in condition_columns if col not in plot_df.columns]
    if missing:
        raise ValueError(f"Evidence-ladder table missing required columns for Figure 1: {missing}")
    plot_df, lead_models = drop_models_without_valid_responses(
        plot_df,
        lead_models,
        condition_columns,
        graph=graph,
        figure_label="Figure 1" + (" anonymized" if anonymized else ""),
        preserve_selected_models=True,
    )
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=lead_models, ordered=True)
    plot_df = plot_df.sort_values("model")
    if plot_df.empty:
        suffix = " anonymized" if anonymized else ""
        print(f"[warn] No evidence-ladder{suffix} rows with valid responses available for graph={graph}; skipping.")
        return

    x = np.arange(len(plot_df))
    width = 0.26 if anonymized else 0.22
    fig, ax = plt.subplots(figsize=FIG1_SIZE)
    any_low_valid = False
    for idx, cond in enumerate(condition_columns):
        offset = (idx - (len(condition_columns) - 1) / 2) * width
        vals = plot_df[cond].map(finite_or_nan).to_numpy()
        err_col = CONDITION_ERROR_COLUMNS.get(cond)
        yerr = (
            plot_df[err_col].map(finite_or_nan).fillna(0.0).to_numpy()
            if err_col in plot_df.columns
            else np.zeros_like(vals)
        )
        valid_col = CONDITION_VALID_COLUMNS.get(cond)
        total_col = CONDITION_TOTAL_COLUMNS.get(cond)
        valid = (
            plot_df[valid_col].map(finite_or_nan).to_numpy()
            if valid_col in plot_df.columns
            else np.full_like(vals, np.nan)
        )
        total = (
            plot_df[total_col].map(finite_or_nan).to_numpy()
            if total_col in plot_df.columns
            else np.full_like(vals, np.nan)
        )
        low_valid = np.isfinite(vals) & np.isfinite(valid) & (valid < 3)
        any_low_valid = any_low_valid or bool(low_valid.any())
        bar_vals = np.nan_to_num(vals, nan=0.0)
        bars = ax.bar(
            x + offset,
            bar_vals,
            width=width,
            label=CONDITION_LABELS[cond],
            color=CONDITION_COLORS[cond],
            edgecolor="black",
            linewidth=0.35,
        )
        for bar, is_low_valid in zip(bars, low_valid):
            if is_low_valid:
                bar.set_hatch("///")
                bar.set_edgecolor("#333333")
                bar.set_linewidth(0.55)
        show_err = np.isfinite(vals) & np.isfinite(yerr) & (yerr > 1e-4)
        if show_err.any():
            ax.errorbar(
                (x + offset)[show_err],
                vals[show_err],
                yerr=yerr[show_err],
                fmt="none",
                ecolor="#2f2f2f",
                elinewidth=0.75,
                capsize=2.4,
                capthick=0.75,
                alpha=0.9,
                zorder=4,
            )
        for bar, val, err, is_low_valid, valid_count, total_count in zip(bars, vals, yerr, low_valid, valid, total):
            if np.isfinite(val):
                label_y = val + max(float(err), 0.0) + 0.018
                ax.text(bar.get_x() + bar.get_width() / 2, label_y, fmt_value(val), ha="center", va="bottom", fontsize=7)
                if is_low_valid:
                    total_text = f"/{int(total_count)}" if np.isfinite(total_count) else ""
                    # ax.text(
                    #     bar.get_x() + bar.get_width() / 2,
                    #     max(0.045, min(val - 0.055, val * 0.20)),
                    #     f"n={int(valid_count)}{total_text}",
                    #     ha="center",
                    #     va="center",
                    #     fontsize=6.5,
                    #     color="white",
                    #     fontweight="semibold",
                    # )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.032,
                    "NA",
                    ha="center",
                    va="bottom",
                    fontsize=6.7,
                    color="#555555",
                    rotation=45,
                )

    classical_vals = plot_df["best_data_only_f1"].map(finite_or_nan)
    if classical_vals.notna().any():
        classical = float(classical_vals.dropna().iloc[0])
        method = str(plot_df["best_data_only_method"].dropna().iloc[0])
        ax.axhline(classical, color="#e15759", linewidth=1.2, linestyle="--", label=f"{method} classical")
        ax.text(len(plot_df) - 0.45, classical + 0.018, f"{method} {fmt_value(classical)}", color="#b23b3b", ha="right", va="bottom", fontsize=8)

    ax.set_xticks(x, plot_df["model"])
    ax.set_ylabel(r"Directed-edge $F_1$")
    ax.set_title(
        f"{graph.capitalize()} — {'anonymized' if anonymized else 'real node names'}",
        fontsize=9, pad=4,
    )
    ax.set_ylim(*F1_YLIM)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    if any_low_valid:
        handles.append(Patch(facecolor="white", edgecolor="#333333", hatch="///", label="<3 valid runs (60%)"))
        labels.append("<60% valid runs")
    ax.legend(handles, labels, frameon=False, ncol=len(labels), loc="upper center", bbox_to_anchor=(0.5, 1.13))
    stem = _stem("fig1_ladder", graph, legacy=legacy_stem)
    stem = _stem_with_suffix(stem, stem_suffix)
    if anonymized:
        stem = f"{stem}_anonymized"
    save_figure(fig, out_dir, stem, formats)
    plt.close(fig)


def plot_model_size_ladder(
    size_df: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    graph: str,
    *,
    legacy_stem: bool,
) -> None:
    graph_df = size_df[size_df["graph"].eq(graph)].copy() if "graph" in size_df.columns else size_df.copy()
    rows = graph_df[
        graph_df["family"].isin(["Qwen2.5", "Qwen3", "Llama 3.1"])
        & ~graph_df["model"].astype(str).isin(PLOT_EXCLUDED_MODELS)
    ].copy()
    rows = rows.dropna(subset=["size_value"], how="any")
    families = []
    # for family in ["Qwen2.5", "Qwen3", "Llama 3.1"]:
    for family in ["Qwen2.5", "Qwen3"]:
        sub = rows[rows.family.eq(family)]
        finite_points = sub[CONDITION_COLUMNS].notna().any(axis=1).sum()
        if finite_points >= 2:
            families.append(family)
    if not families:
        print(f"[warn] No model-size ladder rows available for graph={graph}; skipping.")
        return

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
        ax.set_title(graph.capitalize(), fontsize=9, pad=4)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    axes[0].set_ylabel(r"Directed-edge $F_1$")
    axes[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.02))
    save_figure(fig, out_dir, _stem("fig2_model_size_ladders", graph, legacy=legacy_stem), formats)
    plt.close(fig)


def _contrast_value(pivot: pd.DataFrame, model: str, metric: str) -> float:
    if model not in pivot.index or metric not in pivot.columns:
        return np.nan
    return finite_or_nan(pivot.loc[model, metric])


def _annotate_barh(ax: plt.Axes, bars, values: np.ndarray, *, pad: float = 0.012) -> None:
    xmin, xmax = ax.get_xlim()
    span = xmax - xmin
    for bar, val in zip(bars, values):
        if not np.isfinite(val):
            continue
        x = bar.get_width()
        xpos = x + pad * span if x >= 0 else x - pad * span
        ha = "left" if x >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2, fmt_signed_value(val), ha=ha, va="center", fontsize=6.4)


def _annotate_na_barh(ax: plt.Axes, positions: np.ndarray, values: np.ndarray) -> None:
    xmin, xmax = ax.get_xlim()
    xpos = 0.018 * (xmax - xmin)
    for ypos, val in zip(positions, values):
        if np.isfinite(val):
            continue
        ax.text(
            xpos,
            ypos,
            "NA",
            ha="left",
            va="center",
            fontsize=6.3,
            color="#555555",
            fontstyle="italic",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35},
            zorder=5,
        )


def _style_contrast_axis(ax: plt.Axes, y: np.ndarray, title: str, subtitle: str) -> None:
    for ypos in y[::2]:
        ax.axhspan(ypos - 0.5, ypos + 0.5, color="#f6f6f4", zorder=-3)
    ax.axvline(0, color="#2c2c2c", linewidth=0.7, alpha=0.85, zorder=0)
    ax.grid(axis="x", alpha=0.20, linewidth=0.45, color="#777777")
    ax.set_title(f"{title}\n{subtitle}", fontsize=7.8, pad=6, linespacing=1.15)
    ax.tick_params(axis="x", labelsize=6.8, length=2.2, width=0.5, color="#5f5f5f")
    ax.tick_params(axis="y", labelsize=7.1, length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)


def _condition_key(semantic: str, fmt: str, obs: int, inter: int) -> tuple[str, str, int, int]:
    return semantic, fmt, int(obs), int(inter)


def _valid_condition_map(canonical: pd.DataFrame | None, graph: str) -> dict[tuple[str, tuple[str, str, int, int]], bool]:
    if canonical is None or canonical.empty:
        return {}
    needed = {"graph", "method_display", "semantic", "format", "obs", "inter", "mean_f1", "valid_rows"}
    if not needed.issubset(canonical.columns):
        return {}
    graph_rows = canonical[canonical["graph"].astype(str).eq(graph)].copy()
    valid_map: dict[tuple[str, tuple[str, str, int, int]], bool] = {}
    for _, row in graph_rows.iterrows():
        model = str(row["method_display"])
        key = _condition_key(str(row["semantic"]), str(row["format"]), int(row["obs"]), int(row["inter"]))
        f1 = finite_or_nan(row.get("mean_f1"))
        valid_rows = finite_or_nan(row.get("valid_rows"))
        valid_map[(model, key)] = bool(np.isfinite(f1) and np.isfinite(valid_rows) and valid_rows >= MIN_CONTRAST_VALID_RUNS)
    return valid_map


def _contrast_is_supported(
    valid_map: dict[tuple[str, tuple[str, str, int, int]], bool],
    model: str,
    metric: str,
    obs: int,
    inter: int,
) -> bool:
    if not valid_map:
        return True
    names = _condition_key("names_only", "names_only", 0, 0)
    required = {
        "observational_summary_gain": [names, _condition_key("real", "summary", obs, 0)],
        "observational_matrix_gain": [names, _condition_key("real", "matrix", obs, 0)],
        "interventional_summary_gain": [names, _condition_key("real", "summary", 0, inter)],
        "interventional_matrix_gain": [names, _condition_key("real", "matrix", 0, inter)],
        "format_gap_matrix_minus_summary": [
            _condition_key("real", "summary", obs, inter),
            _condition_key("real", "matrix", obs, inter),
        ],
        "anonymization_drop_summary": [
            _condition_key("real", "summary", obs, inter),
            _condition_key("anon", "summary", obs, inter),
        ],
        "anonymization_drop_matrix": [
            _condition_key("real", "matrix", obs, inter),
            _condition_key("anon", "matrix", obs, inter),
        ],
    }.get(metric, [])
    return all(valid_map.get((model, key), False) for key in required)


def plot_contrast_bars(
    contrasts: pd.DataFrame,
    canonical: pd.DataFrame | None,
    out_dir: Path,
    formats: list[str],
    graph: str,
    selected_models: list[str] | None,
    *,
    legacy_stem: bool,
    stem_suffix: str | None = None,
) -> None:
    metrics = [
        "observational_summary_gain",
        "observational_matrix_gain",
        "interventional_summary_gain",
        "interventional_matrix_gain",
        "format_gap_matrix_minus_summary",
        "anonymization_drop_summary",
        "anonymization_drop_matrix",
    ]
    graph_contrasts = contrasts[contrasts.graph.eq(graph)].copy()
    selected_models = available_models(graph_contrasts, selected_models)
    sub = graph_contrasts[graph_contrasts.model.isin(selected_models) & graph_contrasts.metric.isin(metrics)].copy()
    pivot = sub.pivot_table(index="model", columns="metric", values="value", aggfunc="mean")
    pivot = pivot.reindex(index=selected_models, columns=metrics)
    obs_budget = int(pd.to_numeric(graph_contrasts["obs"], errors="coerce").dropna().max()) if "obs" in graph_contrasts else 5000
    int_budget = int(pd.to_numeric(graph_contrasts["inter"], errors="coerce").dropna().max()) if "inter" in graph_contrasts else 200
    valid_map = _valid_condition_map(canonical, graph)
    support = pd.DataFrame(False, index=pivot.index, columns=pivot.columns)
    for model in pivot.index:
        for metric in pivot.columns:
            supported = _contrast_is_supported(valid_map, str(model), str(metric), obs_budget, int_budget)
            support.loc[model, metric] = supported and np.isfinite(finite_or_nan(pivot.loc[model, metric]))
    pivot = pivot.where(support)
    if not np.isfinite(pivot.to_numpy(dtype=float)).any():
        print(f"[warn] No contrast values available for graph={graph}; skipping.")
        return

    panel_metrics = [
        ("observational_summary_gain", "observational_matrix_gain"),
        ("interventional_summary_gain", "interventional_matrix_gain"),
        ("format_gap_matrix_minus_summary",),
        ("anonymization_drop_summary", "anonymization_drop_matrix"),
    ]
    models = []
    no_value_models = []
    for model in selected_models:
        if model not in pivot.index:
            continue
        has_any_value = np.isfinite(pivot.loc[model, metrics].to_numpy(dtype=float)).any()
        if has_any_value or model in ALWAYS_SHOW_MODELS:
            models.append(model)
        else:
            no_value_models.append(model)
    if no_value_models:
        print(
            f"[info] Dropping {len(no_value_models)} model(s) from Figure 3 for graph={graph} "
            "because no supported contrasts are available: "
            + ", ".join(no_value_models)
        )
    if not models:
        print(f"[warn] No Figure 3 rows have supported contrasts in every panel for graph={graph}; skipping.")
        return
    y = np.arange(len(models))
    shared_values = pivot.loc[models, metrics].to_numpy(dtype=float)
    finite_shared_values = shared_values[np.isfinite(shared_values)]
    shared_lim = max(0.12, float(np.nanmax(np.abs(finite_shared_values))) if finite_shared_values.size else 0.12)
    fig_height = max(2.7, 0.37 * len(models) + 1.35)
    fig, axes = plt.subplots(1, 4, figsize=(8.9, fig_height), sharey=True, gridspec_kw={"wspace": 0.18})

    data = {
        ("Observational data", rf"$F_1(N={obs_budget},M=0)-F_1$ names-only"): [
            ("observational_summary_gain", "Summary", CONDITION_COLORS["real_summary_f1"]),
            ("observational_matrix_gain", "Matrix", CONDITION_COLORS["real_matrix_f1"]),
        ],
        ("Interventional data", rf"$F_1(N=0,M={int_budget})-F_1$ names-only"): [
            ("interventional_summary_gain", "Summary", CONDITION_COLORS["real_summary_f1"]),
            ("interventional_matrix_gain", "Matrix", CONDITION_COLORS["real_matrix_f1"]),
        ],
        ("Representation", rf"Matrix $-$ summary at $N={obs_budget},M={int_budget}$"): [
            ("format_gap_matrix_minus_summary", "Matrix - summary", "#686868")
        ],
        ("Name reliance", rf"Real $-$ anonymized at $N={obs_budget},M={int_budget}$"): [
            ("anonymization_drop_summary", "Summary", "#8cc084"),
            ("anonymization_drop_matrix", "Matrix", "#86a9cc"),
        ],
    }

    for ax, ((title, subtitle), series) in zip(axes, data.items()):
        if len(series) == 1:
            metric, label, color = series[0]
            vals = np.array([_contrast_value(pivot, model, metric) for model in models], dtype=float)
            bars = ax.barh(
                y,
                np.nan_to_num(vals, nan=0.0),
                height=0.58,
                color=color,
                edgecolor="#303030",
                linewidth=0.30,
                label=label,
                zorder=2,
            )
            ax.set_xlim(-shared_lim * 1.32, shared_lim * 1.32)
            _annotate_barh(ax, bars, vals)
            _annotate_na_barh(ax, y, vals)
        else:
            height = 0.28
            pending_annotations = []
            for idx, (metric, label, color) in enumerate(series):
                vals = np.array([_contrast_value(pivot, model, metric) for model in models], dtype=float)
                ypos = y + (idx - 0.5) * height
                bars = ax.barh(
                    ypos,
                    np.nan_to_num(vals, nan=0.0),
                    height=height,
                    color=color,
                    edgecolor="#303030",
                    linewidth=0.30,
                    label=label,
                    zorder=2,
                )
                pending_annotations.append((bars, ypos, vals))
            ax.set_xlim(-shared_lim * 1.32, shared_lim * 1.32)
            for bars, ypos, vals in pending_annotations:
                _annotate_barh(ax, bars, vals)
                _annotate_na_barh(ax, ypos, vals)
        _style_contrast_axis(ax, y, title, subtitle)

    axes[0].set_yticks(y, models)
    axes[0].set_ylim(len(models) - 0.5, -0.5)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0, labelleft=False)
    fig.legend(
        handles=[
            Patch(facecolor=CONDITION_COLORS["real_summary_f1"], edgecolor="#303030", label="Summary"),
            Patch(facecolor=CONDITION_COLORS["real_matrix_f1"], edgecolor="#303030", label="Matrix"),
        ],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.965),
        fontsize=7.6,
        columnspacing=1.4,
        handlelength=1.4,
    )
    fig.suptitle(f"{graph.capitalize()}: information-use contrasts", fontsize=9.2, y=0.995)
    fig.supxlabel(
        rf"$F_1$ difference; cells require $\geq${MIN_CONTRAST_VALID_RUNS}/5 valid runs per matched condition",
        fontsize=7.4,
        y=0.040,
    )
    fig.subplots_adjust(left=0.19, right=0.995, bottom=0.20, top=0.79)
    stem = _stem_with_suffix(_stem("fig3_information_use_contrasts", graph, legacy=legacy_stem), stem_suffix)
    save_figure(fig, out_dir, stem, formats)
    plt.close(fig)


def plot_graph_model_profiles(
    cross: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    graph: str,
    selected_models: list[str] | None,
    *,
    legacy_stem: bool,
    anonymized: bool = False,
    stem_suffix: str | None = None,
) -> None:
    sub = cross[cross.graph.eq(graph)].copy()
    if sub.empty:
        print(f"[warn] No Figure 4 rows available for graph={graph}; skipping.")
        return
    selected_models = available_models(sub, selected_models)
    sub = sub[sub["model"].isin(selected_models)].copy()
    semantic = "anon" if anonymized else "real"
    cols = ["names_only_f1", f"{semantic}_summary_f1", f"{semantic}_matrix_f1"]
    labels = ["Names", "Anon. summary" if anonymized else "Summary", "Anon. matrix" if anonymized else "Matrix"]
    missing = [col for col in cols if col not in sub.columns]
    if missing:
        raise ValueError(f"Cross-graph table missing required columns for Figure 4: {missing}")
    sub, selected_models = drop_models_without_valid_responses(
        sub,
        selected_models,
        cols,
        graph=graph,
        figure_label="Figure 4" + (" anonymized" if anonymized else ""),
    )
    if sub.empty:
        suffix = " anonymized" if anonymized else ""
        print(f"[warn] No Figure 4{suffix} rows with valid responses available for graph={graph}; skipping.")
        return
    pivot = sub.set_index("model").reindex(selected_models)[cols]
    mat = pivot.to_numpy(dtype=float)
    if not np.isfinite(mat).any() and not any(model in ALWAYS_SHOW_MODELS for model in selected_models):
        suffix = " anonymized" if anonymized else ""
        print(f"[warn] No Figure 4{suffix} values available for graph={graph}; skipping.")
        return

    models = [
        model
        for model in selected_models
        if model in pivot.index and (np.isfinite(pivot.loc[model].to_numpy(dtype=float)).any() or model in ALWAYS_SHOW_MODELS)
    ]
    fig_height = 3.5 if len(models) <= 10 else 3.9
    fig, ax = plt.subplots(figsize=(6.7, fig_height))
    x = np.arange(len(cols))
    _profile_colors = [
        "#4878d0", "#ee854a", "#6acc65", "#d65f5f",
        "#956cb4", "#8c613c", "#dc7ec0", "#797979",
        "#d5bb67", "#82c6e2",
    ]
    palette = lambda i: _profile_colors[i % len(_profile_colors)]  # noqa: E731
    for idx, model in enumerate(models):
        vals = pivot.loc[model].to_numpy(dtype=float)
        if not np.isfinite(vals).any():
            ax.plot([], [], color=palette(idx % 10), label=f"{model} (NA)")
            ax.text(
                x[len(x) // 2],
                0.035 + 0.035 * (idx % 3),
                f"{model}: NA",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=palette(idx % 10),
            )
            continue
        ax.plot(
            x,
            vals,
            marker="o",
            linewidth=1.35,
            markersize=4.0,
            color=palette(idx % 10),
            alpha=0.94,
            label=model,
        )

    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(r"Directed-edge $F_1$")
    ax.set_title(
        f"{graph.capitalize()}: evidence-use profiles" + (" — anonymized" if anonymized else ""),
        fontsize=9, pad=5,
    )
    ax.grid(axis="y", alpha=0.20, linewidth=0.55)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=7.5, length=0)
    ax.legend(
        frameon=False,
        ncol=2,
        fontsize=6.5,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.02),
        borderaxespad=0,
        handlelength=1.6,
    )
    fig.subplots_adjust(left=0.10, right=0.66, bottom=0.16, top=0.88)
    stem = _stem("fig4_cross_graph_matched_controls", graph, legacy=legacy_stem)
    stem = _stem_with_suffix(stem, stem_suffix)
    if anonymized:
        stem = f"{stem}_anonymized"
    save_figure(fig, out_dir, stem, formats)
    plt.close(fig)


def _supported_pair(row: pd.Series, real_col: str, anon_col: str) -> tuple[float, str]:
    real = finite_or_nan(row.get(real_col))
    anon = finite_or_nan(row.get(anon_col))
    real_valid = finite_or_nan(row.get(CONDITION_VALID_COLUMNS[real_col]))
    anon_valid = finite_or_nan(row.get(CONDITION_VALID_COLUMNS[anon_col]))
    supported = (
        np.isfinite(real)
        and np.isfinite(anon)
        and np.isfinite(real_valid)
        and np.isfinite(anon_valid)
        and real_valid >= MIN_CONTRAST_VALID_RUNS
        and anon_valid >= MIN_CONTRAST_VALID_RUNS
    )
    label = "" if supported else "low n"
    return (anon - real if supported else np.nan), label


def plot_anonymization_stress_test(
    cross: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    models: list[str] | None = None,
    graphs: list[str] | None = None,
    stem_suffix: str | None = None,
) -> None:
    """Plot anonymized minus real-name F1 at the primary budget."""
    models = models or ANONYMIZATION_STRESS_MODELS
    graphs = graphs or GRAPH_ORDER
    sub = cross[cross["graph"].isin(graphs) & cross["model"].isin(models)].copy()
    if sub.empty:
        print("[warn] No rows available for anonymization stress-test plot; skipping.")
        return

    sub["model"] = pd.Categorical(sub["model"], categories=models, ordered=True)
    sub["graph"] = pd.Categorical(sub["graph"], categories=graphs, ordered=True)
    sub = sub.sort_values(["model", "graph"])

    panels = [
        ("Summary prompts", "real_summary_f1", "anon_summary_f1"),
        ("Matrix prompts", "real_matrix_f1", "anon_matrix_f1"),
    ]
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for _, real_col, anon_col in panels:
        mat = np.full((len(models), len(graphs)), np.nan, dtype=float)
        lab = np.full((len(models), len(graphs)), "", dtype=object)
        for i, model in enumerate(models):
            for j, graph in enumerate(graphs):
                rows = sub[sub["model"].eq(model) & sub["graph"].eq(graph)]
                if rows.empty:
                    continue
                val, label = _supported_pair(rows.iloc[0], real_col, anon_col)
                mat[i, j] = val
                lab[i, j] = label
        matrices.append(mat)
        labels.append(lab)

    finite_values = np.concatenate([mat[np.isfinite(mat)] for mat in matrices if np.isfinite(mat).any()])
    if finite_values.size == 0:
        print("[warn] No supported anonymization contrasts available; skipping.")
        return
    lim = max(0.12, float(np.nanmax(np.abs(finite_values))))

    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#eeeeee")
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.65), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (title, _, _), mat, lab in zip(axes, panels, matrices, labels):
        image = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_title(title, fontsize=8.8, pad=5)
        ax.set_xticks(np.arange(len(graphs)))
        ax.set_xticklabels([graph.capitalize() for graph in graphs], rotation=25, ha="right", rotation_mode="anchor")
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels(models)
        ax.set_xticks(np.arange(-0.5, len(graphs), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="both", length=0)
        for i in range(len(models)):
            for j in range(len(graphs)):
                val = mat[i, j]
                if np.isfinite(val):
                    color = "white" if abs(val) > 0.55 * lim else "#1f1f1f"
                    ax.text(j, i, fmt_signed_value(val), ha="center", va="center", fontsize=7.0, color=color)
                elif lab[i, j]:
                    ax.text(j, i, lab[i, j], ha="center", va="center", fontsize=6.4, color="#555555")
                else:
                    ax.text(j, i, "--", ha="center", va="center", fontsize=6.8, color="#777777")

    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    cbar.set_label(r"Anon.$\,-\,$real $F_1$", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.2, length=2.5)
    fig.suptitle("Name-anonymization drop at the primary budget", fontsize=9.4, y=0.985)
    fig.supxlabel(
        rf"Anon.$\,-\,$real $F_1$; gray cells lack $\geq${MIN_CONTRAST_VALID_RUNS}/5 valid paired runs",
        fontsize=7.4,
        y=0.010,
    )
    fig.subplots_adjust(left=0.18, right=0.90, bottom=0.24, top=0.80)

    stem = _stem_with_suffix("fig_anonymization_stress_test", stem_suffix)
    save_figure(fig, out_dir, stem, formats)
    plt.close(fig)


def _node_count_from_bif(graph: str) -> float:
    bif_path = REPO_ROOT / "causal_graphs" / "real_data" / "small_graphs" / f"{graph}.bif"
    if not bif_path.exists():
        return np.nan
    text = bif_path.read_text(encoding="utf-8", errors="ignore")
    return float(len(re.findall(r"^\s*variable\s+([^\s{]+)", text, flags=re.M)))


def _gpt52_names_only_from_eval_summary(graph: str) -> dict[str, object] | None:
    candidates = [
        REPO_ROOT / "experiments" / "responses" / graph / "eval_summary.csv",
        REPO_ROOT / "scripts" / "responses" / graph / "eval_summary.csv",
        REPO_ROOT / "responses" / graph / "eval_summary.csv",
    ]
    rows: list[dict[str, object]] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[warn] Could not read {path}: {exc}")
            continue
        if "model" not in df.columns:
            continue
        model_mask = df["model"].astype(str).str.lower().eq("gpt-5.2-pro")
        if "prompt_style" in df.columns:
            names_mask = df["prompt_style"].astype(str).eq("names_only")
        elif "is_names_only" in df.columns:
            names_mask = pd.to_numeric(df["is_names_only"], errors="coerce").fillna(0).astype(int).eq(1)
        else:
            names_mask = pd.Series(False, index=df.index)
        sub = df.loc[model_mask & names_mask].copy()
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            f1 = finite_or_nan(row.get("avg_f1", row.get("avg_F1", np.nan)))
            if not np.isfinite(f1):
                continue
            rows.append(
                {
                    "graph": graph,
                    "nodes": _node_count_from_bif(graph),
                    "f1": f1,
                    "se": finite_or_nan(row.get("avg_f1_se", 0.0)),
                    "valid": finite_or_nan(row.get("valid_rows", np.nan)),
                    "n": finite_or_nan(row.get("num_rows", np.nan)),
                    "source": str(path),
                }
            )
    if not rows:
        return None
    # Prefer summaries with explicit SE/valid-rate metadata, then higher valid count.
    rows.sort(
        key=lambda r: (
            np.isfinite(float(r["se"])),
            np.isfinite(float(r["valid"])),
            float(r["valid"]) if np.isfinite(float(r["valid"])) else -1.0,
        ),
        reverse=True,
    )
    return rows[0]


def collect_gpt52_names_only_scaling(cross: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for graph in SEMANTIC_SCALING_GRAPHS:
        row: dict[str, object] | None = None
        if {"graph", "method", "names_only_f1"}.issubset(cross.columns):
            sub = cross[
                cross["graph"].astype(str).eq(graph)
                & cross["method"].astype(str).str.lower().eq("gpt-5.2-pro")
            ]
            if not sub.empty:
                src = sub.iloc[0]
                f1 = finite_or_nan(src.get("names_only_f1", np.nan))
                if np.isfinite(f1):
                    row = {
                        "graph": graph,
                        "nodes": _node_count_from_bif(graph),
                        "f1": f1,
                        "se": finite_or_nan(src.get("names_only_f1_se", 0.0)),
                        "valid": finite_or_nan(src.get("names_only_valid_rows", np.nan)),
                        "n": finite_or_nan(src.get("names_only_n_rows", np.nan)),
                        "source": "paper_cross_graph_evidence_ladder.csv",
                    }
        if row is None:
            row = _gpt52_names_only_from_eval_summary(graph)
        if row is None:
            print(f"[warn] Missing GPT-5.2 Pro names-only result for graph={graph}; omitting from scaling figure.")
            continue
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("nodes").reset_index(drop=True)
    for col in ("nodes", "f1", "se", "valid", "n"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def plot_gpt52_names_only_scaling(
    cross: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
) -> None:
    plot_df = collect_gpt52_names_only_scaling(cross)
    if plot_df.empty:
        print("[warn] No GPT-5.2 Pro names-only rows available for semantic-scaling figure; skipping.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "fig_rq1_gpt52_name_only_scaling.csv"
    plot_df.to_csv(data_path, index=False)
    print(f"[write] {data_path}")

    fig, ax = plt.subplots(figsize=(3.35, 2.18))
    ax.axhline(1.0, color="#9a9a9a", lw=0.7, ls=(0, (3, 2)), zorder=1)

    full_valid = plot_df["valid"].fillna(0).ge(plot_df["n"].fillna(np.inf))
    low_valid = ~full_valid
    err = plot_df["se"].fillna(0.0)

    ax.errorbar(
        plot_df.loc[full_valid, "nodes"],
        plot_df.loc[full_valid, "f1"],
        yerr=err.loc[full_valid],
        fmt="o",
        color="#2f6f9f",
        markerfacecolor="#2f6f9f",
        markeredgecolor="white",
        markeredgewidth=0.5,
        markersize=4.6,
        capsize=2.3,
        capthick=0.75,
        elinewidth=0.75,
        zorder=3,
    )
    if low_valid.any():
        ax.errorbar(
            plot_df.loc[low_valid, "nodes"],
            plot_df.loc[low_valid, "f1"],
            yerr=err.loc[low_valid],
            fmt="o",
            color="#2f6f9f",
            markerfacecolor="white",
            markeredgecolor="#2f6f9f",
            markeredgewidth=1.05,
            markersize=4.7,
            capsize=2.3,
            capthick=0.75,
            elinewidth=0.75,
            zorder=4,
        )

    label_offsets = {
        "cancer": (-0.95, -0.062, "top"),
        "asia": (0.95, -0.062, "top"),
        "sachs": (0.0, 0.043, "bottom"),
        "child": (0.0, 0.043, "bottom"),
        "alarm": (0.0, 0.043, "bottom"),
    }
    for _, row in plot_df.iterrows():
        graph = str(row["graph"])
        dx, dy, va = label_offsets.get(graph, (0.0, 0.043, "bottom"))
        label = f"{SEMANTIC_SCALING_LABELS.get(graph, graph.capitalize())}\n{fmt_value(row['f1'])}"
        if np.isfinite(row["valid"]) and np.isfinite(row["n"]) and row["valid"] < row["n"]:
            label += f"\n({int(row['valid'])}/{int(row['n'])})"
        ax.text(
            row["nodes"] + dx,
            row["f1"] + dy,
            label,
            ha="center",
            va=va,
            fontsize=6.4,
            color="#222222",
        )

    x_max = max(40.0, float(plot_df["nodes"].max()) + 2.5)
    ax.set_xlim(2.2, x_max)
    ax.set_ylim(0.43, 1.05)
    ax.set_xticks([int(v) for v in plot_df["nodes"].dropna().tolist()])
    ax.set_yticks([0.5, 0.7, 0.9, 1.0])
    ax.set_xlabel(r"Graph size ($|V|$)")
    ax.set_ylabel(r"Names-only $F_1$")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.85)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.5, alpha=0.5)
    ax.tick_params(width=0.8, length=2.5)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)
    fig.tight_layout(pad=0.25)
    save_figure(fig, out_dir, "fig_rq1_gpt52_name_only_scaling", formats)
    plt.close(fig)


def write_latex_includes(out_dir: Path) -> None:
    text = r"""
% Generated by scripts/plot_micad_paper_figures.py

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.49\textwidth]{experiments/out/micad_paper/figures/fig1_ladder_sachs_pretrained.pdf}
  \includegraphics[width=0.49\textwidth]{experiments/out/micad_paper/figures/fig1_ladder_sachs_qwen3_ft.pdf}
  \caption{Matched evidence-use ladders on Sachs. Left: pretrained models. Right: Qwen3-4B versus Qwen3-4B-FT. Each model is evaluated on the same graph and data; only the exposed information changes from names-only to summary statistics to raw matrix rows.}
  \label{fig:decisive_evidence_ladder_sachs}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.49\textwidth]{experiments/out/micad_paper/figures/fig1_ladder_sachs_pretrained_anonymized.pdf}
  \includegraphics[width=0.49\textwidth]{experiments/out/micad_paper/figures/fig1_ladder_sachs_qwen3_ft_anonymized.pdf}
  \caption{Anonymized matched evidence-use ladders on Sachs. Left: pretrained models. Right: Qwen3-4B versus Qwen3-4B-FT.}
  \label{fig:decisive_evidence_ladder_sachs_anonymized}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.49\textwidth]{experiments/out/micad_paper/figures/fig3_information_use_contrasts_sachs_pretrained.pdf}
  \includegraphics[width=0.49\textwidth]{experiments/out/micad_paper/figures/fig3_information_use_contrasts_sachs_qwen3_ft.pdf}
  \caption{Matched information-use contrasts on Sachs. Left: pretrained models. Right: Qwen3-4B versus Qwen3-4B-FT. Bars report directed-edge F1 differences; NA marks unsupported matched cells.}
  \label{fig:information_use_contrasts_sachs}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.49\textwidth]{experiments/out/micad_paper/figures/fig_anonymization_stress_test_pretrained.pdf}
  \includegraphics[width=0.49\textwidth]{experiments/out/micad_paper/figures/fig_anonymization_stress_test_qwen3_ft.pdf}
  \caption{Anonymization stress tests at the primary budget. Left: pretrained models. Right: Qwen3-4B versus Qwen3-4B-FT. Each cell reports anonymized minus real-name directed-edge F1 for the same graph, model, prompt format, and data budget.}
  \label{fig:anonymization_stress_test}
\end{figure*}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.5\textwidth]{experiments/out/micad_paper/figures/fig_rq1_gpt52_name_only_scaling.pdf}
  \caption{GPT-5.2 Pro name-only recovery across graph sizes. Filled points mark fully valid parsed runs; the open point marks a condition with fewer than five valid parsed runs. Semantic priors solve the smallest graphs but are insufficient for reliable recovery on larger networks.}
  \label{fig:gpt52_name_only_scaling}
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
    canonical = read_csv(args.paper_dir / "canonical_condition_results.csv")
    headline = apply_model_display_aliases(headline)
    size_df = apply_model_display_aliases(size_df)
    contrasts = apply_model_display_aliases(contrasts)
    cross = apply_model_display_aliases(cross)
    canonical = apply_model_display_aliases(canonical)
    graphs = resolve_graphs(args, cross["graph"].dropna().unique())

    # Prefer the all-graph ladder table for graph-specific plotting.  The
    # headline CSV is kept for backward compatibility with older collected
    # outputs that only contained one graph.
    ladder_source = cross if "graph" in cross.columns else headline
    for graph in graphs:
        legacy_stem = len(graphs) == 1 and graph == "sachs"
        for stem_suffix, models in COMPARISON_SETS:
            plot_decisive_evidence_ladder(
                ladder_source,
                out_dir,
                args.formats,
                models,
                graph,
                legacy_stem=legacy_stem,
                stem_suffix=stem_suffix,
            )
            plot_decisive_evidence_ladder(
                ladder_source,
                out_dir,
                args.formats,
                models,
                graph,
                legacy_stem=legacy_stem,
                anonymized=True,
                stem_suffix=stem_suffix,
            )
            plot_contrast_bars(
                contrasts,
                canonical,
                out_dir,
                args.formats,
                graph,
                models,
                legacy_stem=legacy_stem,
                stem_suffix=stem_suffix,
            )

    for stem_suffix, models in COMPARISON_SETS:
        plot_anonymization_stress_test(cross, out_dir, args.formats, models=models, stem_suffix=stem_suffix)
    plot_gpt52_names_only_scaling(cross, out_dir, args.formats)
    write_latex_includes(out_dir)


if __name__ == "__main__":
    main()
