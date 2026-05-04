#!/usr/bin/env python3
"""Generate MICAD paper-quality result figures from collected paper tables.

Run scripts/collect_micad_paper_results.py first.  This script reads the
paper-ready CSVs from experiments/out/micad_paper and creates a compact figure
set aimed at a top-tier benchmark paper:

  1. Decisive evidence ladder: names_only vs summary vs matrix vs classical.
  2. Model-size ladder: same-family scale comparison.
  3. Contrast bars: what information each model uses.
  4. Cross-graph profiles: matched controls across the real and anonymized reference graphs.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


GRAPH_ORDER = ["cancer", "earthquake", "asia", "sachs"]
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
        default=None,
        help="Display names to include in the decisive evidence ladder. Defaults to all available models.",
    )
    parser.add_argument(
        "--contrast-models",
        "--heatmap-models",
        dest="contrast_models",
        nargs="*",
        default=None,
        help="Display names to include in the contrast bar plot. Defaults to all available models.",
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


def _stem(base: str, graph: str, *, legacy: bool) -> str:
    return base if legacy else f"{base}_{graph}"


def expand_model_patterns(models: list[str], requested: list[str]) -> list[str]:
    selected: list[str] = []
    unmatched: list[str] = []
    for token in requested:
        patterns = [part.strip() for part in str(token).split(",") if part.strip()]
        for pattern in patterns:
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
    return expand_model_patterns(models, requested)


def drop_models_without_valid_responses(
    plot_df: pd.DataFrame,
    selected_models: list[str],
    condition_columns: list[str],
    *,
    graph: str,
    figure_label: str,
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
    )
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=lead_models, ordered=True)
    plot_df = plot_df.sort_values("model")
    if plot_df.empty:
        suffix = " anonymized" if anonymized else ""
        print(f"[warn] No evidence-ladder{suffix} rows with valid responses available for graph={graph}; skipping.")
        return

    x = np.arange(len(plot_df))
    width = 0.26 if anonymized else 0.22
    fig_width = max(7.2, 0.62 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 3.0))
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
        bars = ax.bar(
            x + offset,
            vals,
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
                ax.text(bar.get_x() + bar.get_width() / 2, label_y, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
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

    classical_vals = plot_df["best_data_only_f1"].map(finite_or_nan)
    if classical_vals.notna().any():
        classical = float(classical_vals.dropna().iloc[0])
        method = str(plot_df["best_data_only_method"].dropna().iloc[0])
        ax.axhline(classical, color="#e15759", linewidth=1.2, linestyle="--", label=f"{method} classical")
        ax.text(len(plot_df) - 0.45, classical + 0.018, f"{method} {classical:.2f}", color="#b23b3b", ha="right", va="bottom", fontsize=8)

    ax.set_xticks(x, plot_df["model"])
    ax.set_ylabel("Directed-edge F1")
    ax.set_title(f"{graph.capitalize()} ({'anonymized' if anonymized else 'real names'})", fontsize=9, pad=4)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    if any_low_valid:
        handles.append(Patch(facecolor="white", edgecolor="#333333", hatch="///", label="<3 valid runs (60%)"))
        labels.append("<60% valid runs")
    ax.legend(handles, labels, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    stem = _stem("fig1_ladder", graph, legacy=legacy_stem)
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
    rows = graph_df[graph_df["family"].isin(["Qwen2.5", "Qwen3", "Llama 3.1"])].copy()
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
    axes[0].set_ylabel("Directed-edge F1")
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
        ax.text(xpos, bar.get_y() + bar.get_height() / 2, f"{val:+.2f}", ha=ha, va="center", fontsize=6.4)


def _style_contrast_axis(ax: plt.Axes, y: np.ndarray, title: str, subtitle: str) -> None:
    for ypos in y[::2]:
        ax.axhspan(ypos - 0.5, ypos + 0.5, color="#f6f6f4", zorder=-3)
    ax.axvline(0, color="#2c2c2c", linewidth=0.7, alpha=0.85, zorder=0)
    ax.grid(axis="x", alpha=0.20, linewidth=0.45, color="#777777")
    ax.set_title(f"{title}\n{subtitle}", fontsize=8.0, pad=6, linespacing=1.15)
    ax.tick_params(axis="x", labelsize=6.8, length=2.2, width=0.5, color="#5f5f5f")
    ax.tick_params(axis="y", labelsize=7.1, length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)


def plot_contrast_bars(
    contrasts: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    graph: str,
    selected_models: list[str] | None,
    *,
    legacy_stem: bool,
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
    if not np.isfinite(pivot.to_numpy(dtype=float)).any():
        print(f"[warn] No contrast values available for graph={graph}; skipping.")
        return

    models = [
        model
        for model in selected_models
        if model in pivot.index and np.isfinite(pivot.loc[model].to_numpy(dtype=float)).any()
    ]
    y = np.arange(len(models))
    fig_height = max(3.5, 0.34 * len(models) + 1.15)
    fig, axes = plt.subplots(1, 4, figsize=(8.9, fig_height), sharey=True, gridspec_kw={"wspace": 0.16})

    data = {
        ("Obs. gain", r"$N:0\to5000,\ M=0$"): [
            ("observational_summary_gain", "Summary", CONDITION_COLORS["real_summary_f1"]),
            ("observational_matrix_gain", "Matrix", CONDITION_COLORS["real_matrix_f1"]),
        ],
        ("Int. gain", r"$M:0\to200,\ N=0$"): [
            ("interventional_summary_gain", "Summary", CONDITION_COLORS["real_summary_f1"]),
            ("interventional_matrix_gain", "Matrix", CONDITION_COLORS["real_matrix_f1"]),
        ],
        ("Format", "Matrix - summary"): [("format_gap_matrix_minus_summary", "Matrix - summary", "#686868")],
        ("Name reliance", "Real - anonymized"): [
            ("anonymization_drop_summary", "Summary", "#8cc084"),
            ("anonymization_drop_matrix", "Matrix", "#86a9cc"),
        ],
    }

    for ax, ((title, subtitle), series) in zip(axes, data.items()):
        if len(series) == 1:
            metric, label, color = series[0]
            vals = np.array([_contrast_value(pivot, model, metric) for model in models], dtype=float)
            bars = ax.barh(y, vals, height=0.58, color=color, edgecolor="#303030", linewidth=0.30, label=label, zorder=2)
            finite = vals[np.isfinite(vals)]
            lim = max(0.12, float(np.nanmax(np.abs(finite))) if finite.size else 0.12)
            ax.set_xlim(-lim * 1.32, lim * 1.32)
            _annotate_barh(ax, bars, vals)
        else:
            height = 0.28
            all_vals = []
            pending_annotations = []
            for idx, (metric, label, color) in enumerate(series):
                vals = np.array([_contrast_value(pivot, model, metric) for model in models], dtype=float)
                all_vals.extend(vals[np.isfinite(vals)])
                ypos = y + (idx - 0.5) * height
                bars = ax.barh(
                    ypos,
                    vals,
                    height=height,
                    color=color,
                    edgecolor="#303030",
                    linewidth=0.30,
                    label=label,
                    zorder=2,
                )
                pending_annotations.append((bars, vals))
            lim = max(0.12, float(np.nanmax(np.abs(all_vals))) if all_vals else 0.12)
            ax.set_xlim(-lim * 1.32, lim * 1.32)
            for bars, vals in pending_annotations:
                _annotate_barh(ax, bars, vals)
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
    fig.supxlabel("Matched directed-edge F1 difference", fontsize=8.0, y=0.035)
    fig.subplots_adjust(left=0.19, right=0.995, bottom=0.17, top=0.80)
    save_figure(fig, out_dir, _stem("fig3_information_use_contrasts", graph, legacy=legacy_stem), formats)
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
    if not np.isfinite(mat).any():
        suffix = " anonymized" if anonymized else ""
        print(f"[warn] No Figure 4{suffix} values available for graph={graph}; skipping.")
        return

    models = [model for model in selected_models if model in pivot.index and np.isfinite(pivot.loc[model].to_numpy(dtype=float)).any()]
    fig_height = 3.5 if len(models) <= 10 else 3.9
    fig, ax = plt.subplots(figsize=(6.7, fig_height))
    x = np.arange(len(cols))
    palette = plt.get_cmap("tab10")
    for idx, model in enumerate(models):
        vals = pivot.loc[model].to_numpy(dtype=float)
        if not np.isfinite(vals).any():
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
    ax.set_ylabel("Directed-edge F1")
    ax.set_title(f"{graph.capitalize()}: model evidence-use profiles" + (" (anonymized)" if anonymized else ""), fontsize=9, pad=5)
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
    if anonymized:
        stem = f"{stem}_anonymized"
    save_figure(fig, out_dir, stem, formats)
    plt.close(fig)


def write_latex_includes(out_dir: Path) -> None:
    text = r"""
% Generated by scripts/plot_micad_paper_figures.py

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.92\textwidth]{experiments/out/micad_paper/figures/fig1_ladder.pdf}
  \caption{Matched evidence-use ladder on the primary graph and budget. Each model is evaluated on the same graph and data; only the exposed information changes from names-only to summary statistics to raw matrix rows. Error bars show standard errors over valid parsed runs; hatched bars mark conditions with fewer than three valid parsed runs (60\%). The dashed line shows the strongest classical causal discovery reference at the same data budget.}
  \label{fig:decisive_evidence_ladder}
\end{figure*}

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.92\textwidth]{experiments/out/micad_paper/figures/fig1_ladder_anonymized.pdf}
  \caption{Anonymized companion to the matched evidence-use ladder. The bars use anonymized summaries and matrices; the dashed line preserves the strongest classical reference at the same data budget.}
  \label{fig:decisive_evidence_ladder_anonymized}
\end{figure*}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{experiments/out/micad_paper/figures/fig2_model_size_ladders.pdf}
  \caption{Within-family model-size ladders. Larger models sometimes use supplied data more effectively, but the trend is not monotonic across matched cells.}
  \label{fig:model_size_ladders}
\end{figure}

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.92\textwidth]{experiments/out/micad_paper/figures/fig3_information_use_contrasts.pdf}
  \caption{Contrastive information-use metrics. Bars are matched F1 differences grouped by the question they answer: whether observational data alone helps from $N=0$ to $N=5000$ at $M=0$, whether interventional data alone helps from $M=0$ to $M=200$ at $N=0$, whether matrix rows beat summaries, and whether real names matter.}
  \label{fig:information_use_contrasts}
\end{figure*}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{experiments/out/micad_paper/figures/fig4_cross_graph_matched_controls.pdf}
  \caption{Optional appendix evidence-use profile on one graph. Each line is a model, and the x-axis traces the exposed information from names-only to summary statistics to raw matrix rows. Sparse line segments indicate missing matched cells.}
  \label{fig:cross_graph_matched_controls}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{experiments/out/micad_paper/figures/fig4_cross_graph_matched_controls_anonymized.pdf}
  \caption{Anonymized companion to the evidence-use profile. Each line is a model; the data-bearing points use anonymized summaries and matrices while preserving the names-only reference point.}
  \label{fig:cross_graph_matched_controls_anonymized}
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
    graphs = resolve_graphs(args, cross["graph"].dropna().unique())

    # Prefer the all-graph ladder table for graph-specific plotting.  The
    # headline CSV is kept for backward compatibility with older collected
    # outputs that only contained one graph.
    ladder_source = cross if "graph" in cross.columns else headline
    size_source = cross if "graph" in cross.columns else size_df
    contrast_models = args.contrast_models
    for graph in graphs:
        legacy_stem = len(graphs) == 1 and graph == "sachs"
        plot_decisive_evidence_ladder(
            ladder_source,
            out_dir,
            args.formats,
            args.lead_models,
            graph,
            legacy_stem=legacy_stem,
        )
        plot_decisive_evidence_ladder(
            ladder_source,
            out_dir,
            args.formats,
            args.lead_models,
            graph,
            legacy_stem=legacy_stem,
            anonymized=True,
        )
        plot_model_size_ladder(
            size_source,
            out_dir,
            args.formats,
            graph,
            legacy_stem=legacy_stem,
        )
        plot_contrast_bars(
            contrasts,
            out_dir,
            args.formats,
            graph,
            contrast_models,
            legacy_stem=legacy_stem,
        )
        plot_graph_model_profiles(
            cross,
            out_dir,
            args.formats,
            graph,
            args.lead_models,
            legacy_stem=legacy_stem,
        )
        plot_graph_model_profiles(
            cross,
            out_dir,
            args.formats,
            graph,
            args.lead_models,
            legacy_stem=legacy_stem,
            anonymized=True,
        )

    write_latex_includes(out_dir)


if __name__ == "__main__":
    main()
