#!/usr/bin/env python3
"""Plot obs/int budget effectiveness from best prompt configs.

The plot treats obs and intervention budgets as two axes. Each point is one
(obs, int) cell; color is mean F1 or mean data-use gain after selecting the
better of summary/matrix for each graph/model at that budget.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

if not os.environ.get("MPLCONFIGDIR"):
    mplconfig = Path("/tmp") / f"matplotlib_{os.getuid()}"
    mplconfig.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mplconfig)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEST_CONFIGS = REPO_ROOT / "experiments" / "out" / "best_prompt_configs_all_models_by_valid_rate.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "out" / "qwen3_posttraining_maintext"
DEFAULT_GRAPHS = ["cancer", "earthquake", "asia", "sachs"]
DEFAULT_MODELS = [
    "Qwen3-4B-Thinking-2507",
    "qwen3_4b_cd_format_v5_rerun_2gpu_checkpoint-100_merged",
    "grpo_from_qwen3_4b_cd_format_v5_rerun_no_cancer_full_checkpoint-1200_merged",
    "grpo_from_qwen3_4b_sft_mix_guide_v2_lenfix_2gpu_checkpoint-200",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-configs-csv", type=Path, default=DEFAULT_BEST_CONFIGS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--graphs", nargs="*", default=DEFAULT_GRAPHS)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--anonymization", choices=["real", "anon"], default="real")
    parser.add_argument("--quantity", choices=["f1", "gain", "valid_rate"], default="f1")
    parser.add_argument("--min-coverage", type=int, default=0, help="Hide cells with fewer graph/model rows.")
    parser.add_argument("--fixed-marker-size", action="store_true", help="Use one marker size instead of scaling by coverage.")
    parser.add_argument("--marker-size", type=float, default=10.0, help="Marker size used with --fixed-marker-size.")
    parser.add_argument("--no-marker-text", action="store_true", help="Do not write values/coverage labels inside marks.")
    parser.add_argument("--linear-axes", action="store_true", help="Use raw obs/int values instead of log10(value+1) axes.")
    parser.add_argument(
        "--cmap",
        default=None,
        help="Matplotlib colormap. Defaults to RdBu_r for gain and YlOrRd otherwise.",
    )
    parser.add_argument(
        "--color-range",
        choices=["auto", "unit", "tight"],
        default="auto",
        help=(
            "Color normalization: auto uses matplotlib defaults except centered gain; "
            "unit fixes non-gain metrics to [0, 1]; tight zooms to the observed finite range."
        ),
    )
    parser.add_argument("--vmin", type=float, default=None, help="Manual lower color limit.")
    parser.add_argument("--vmax", type=float, default=None, help="Manual upper color limit.")
    parser.add_argument(
        "--one-per-model-graph",
        action="store_true",
        help="Write one separate plot for each requested model x graph combination.",
    )
    parser.add_argument("--formats", nargs="*", default=["pdf", "png"], choices=["pdf", "png", "svg"])
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 9.5,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def log_budget(values: pd.Series) -> np.ndarray:
    return np.log10(values.astype(float).to_numpy() + 1.0)


def axis_budget(values: pd.Series, linear_axes: bool) -> np.ndarray:
    if linear_axes:
        return values.astype(float).to_numpy()
    return log_budget(values)


def axis_value(value: float, linear_axes: bool) -> float:
    if linear_axes:
        return float(value)
    return math.log10(float(value) + 1.0)


def model_label(raw: str) -> str:
    labels = {
        "Qwen3-4B-Thinking-2507": "Base",
        "qwen3_4b_cd_format_v5_rerun_2gpu_checkpoint-100_merged": "CD v5 ckpt-100",
        "grpo_from_qwen3_4b_cd_format_v5_rerun_no_cancer_full_checkpoint-1200_merged": "GRPO no-cancer ckpt-1200",
        "grpo_from_qwen3_4b_sft_mix_guide_v2_lenfix_2gpu_checkpoint-200": "GRPO mix v2 ckpt-200",
    }
    return labels.get(raw, raw)


def safe_slug(text: str) -> str:
    chars = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            chars.append(char)
        else:
            chars.append("-")
    return "-".join(part for part in "".join(chars).strip("-").split("-") if part)


def collect_budget_cells(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    graphs = {graph.lower() for graph in args.graphs}
    models = set(args.models)
    anonymize = 1 if args.anonymization == "anon" else 0

    work = df[
        df["graph"].astype(str).str.lower().isin(graphs)
        & df["model_raw"].astype(str).isin(models)
        & df["prompt_style"].isin(["summary", "matrix", "names_only"])
    ].copy()
    work["score"] = pd.to_numeric(work["avg_f1"], errors="coerce").fillna(0.0)
    work["valid_rate_num"] = pd.to_numeric(work["valid_rate"], errors="coerce")

    names = (
        work[work["prompt_style"].eq("names_only")]
        .sort_values(["graph", "model_raw", "score", "valid_rate_num"], ascending=[True, True, False, False])
        .drop_duplicates(["graph", "model_raw"])[["graph", "model_raw", "score"]]
        .rename(columns={"score": "names_f1"})
    )

    data = work[work["prompt_style"].isin(["summary", "matrix"]) & work["anonymize"].eq(anonymize)].copy()
    best = (
        data.sort_values(
            ["graph", "model_raw", "obs_n", "int_n", "score", "valid_rate_num"],
            ascending=[True, True, True, True, False, False],
        )
        .drop_duplicates(["graph", "model_raw", "obs_n", "int_n"])
        .merge(names, on=["graph", "model_raw"], how="left")
    )
    best["gain"] = best["score"] - best["names_f1"].fillna(0.0)
    value_col = {"f1": "score", "gain": "gain", "valid_rate": "valid_rate_num"}[args.quantity]

    agg = (
        best.groupby(["obs_n", "int_n"], dropna=False)
        .agg(
            value=(value_col, "mean"),
            avg_f1=("score", "mean"),
            avg_gain=("gain", "mean"),
            avg_valid=("valid_rate_num", "mean"),
            coverage=("score", "size"),
        )
        .reset_index()
    )
    if args.min_coverage > 0:
        agg = agg[agg["coverage"] >= args.min_coverage].copy()
    return agg


def output_stem(args: argparse.Namespace) -> str:
    graphs = "graphs-" + "-".join(args.graphs)
    if args.models == DEFAULT_MODELS:
        models = "candidates"
    elif len(args.models) == 1:
        models = f"model-{safe_slug(args.models[0])}"
    else:
        models = f"models{len(args.models)}"
    cov = "" if args.min_coverage <= 0 else f"_mincov{args.min_coverage}"
    fixed = "_fixedmarkers" if args.fixed_marker_size else ""
    text = "_notext" if args.no_marker_text else ""
    scale = "_linear" if args.linear_axes else ""
    color = ""
    if args.cmap is not None or args.color_range != "auto" or args.vmin is not None or args.vmax is not None:
        color = "_colors"
    return f"obs_int_budget_map_{args.quantity}_{args.anonymization}_{models}_{graphs}{cov}{fixed}{text}{scale}{color}"


def color_norm(values: np.ndarray, args: argparse.Namespace) -> mpl.colors.Normalize | None:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None

    if args.vmin is not None or args.vmax is not None:
        vmin = float(np.nanmin(finite)) if args.vmin is None else args.vmin
        vmax = float(np.nanmax(finite)) if args.vmax is None else args.vmax
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        if args.quantity == "gain" and vmin < 0.0 < vmax:
            return mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        return mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if args.quantity == "gain":
        bound = max(0.05, float(np.nanmax(np.abs(finite))))
        return mpl.colors.TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)

    if args.color_range == "unit":
        return mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    if args.color_range == "tight":
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if math.isclose(vmin, vmax):
            pad = max(0.02, abs(vmin) * 0.05)
        else:
            pad = max(0.02, 0.08 * (vmax - vmin))
        return mpl.colors.Normalize(vmin=max(0.0, vmin - pad), vmax=min(1.0, vmax + pad))

    return None


def plot(cells: pd.DataFrame, args: argparse.Namespace) -> None:
    if cells.empty:
        raise SystemExit("No matching obs/int budget cells found.")
    configure_style()
    x = axis_budget(cells["obs_n"], args.linear_axes)
    y = axis_budget(cells["int_n"], args.linear_axes)
    values = cells["value"].astype(float).to_numpy()
    coverage = cells["coverage"].astype(float).to_numpy()
    sizes = np.full_like(coverage, args.marker_size, dtype=float) if args.fixed_marker_size else 95 + 34 * coverage

    cmap = args.cmap or ("RdBu_r" if args.quantity == "gain" else "YlOrRd")
    norm = color_norm(values, args)

    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    ax.grid(color="#d8d8d8", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    scatter = ax.scatter(x, y, c=values, s=sizes, cmap=cmap, norm=norm, edgecolor="#333333", linewidth=0.7, zorder=3)

    if not args.no_marker_text:
        for _, row in cells.iterrows():
            label = f"{row['value']:.2f}\nn={int(row['coverage'])}"
            ax.text(
                axis_value(float(row["obs_n"]), args.linear_axes),
                axis_value(float(row["int_n"]), args.linear_axes),
                label,
                ha="center",
                va="center",
                fontsize=6.6,
                color="white" if row["value"] > cells["value"].median() else "#111111",
                linespacing=0.84,
                zorder=4,
            )

    obs_ticks = sorted(cells["obs_n"].astype(int).unique())
    int_ticks = sorted(cells["int_n"].astype(int).unique())
    ax.set_xticks([axis_value(v, args.linear_axes) for v in obs_ticks])
    ax.set_xticklabels([str(v) for v in obs_ticks], rotation=35, ha="right")
    ax.set_yticks([axis_value(v, args.linear_axes) for v in int_ticks])
    ax.set_yticklabels([str(v) for v in int_ticks])
    scale = "" if args.linear_axes else ", log10(N+1)"
    inter_scale = "" if args.linear_axes else ", log10(M+1)"
    ax.set_xlabel(f"Observational samples N{scale}")
    ax.set_ylabel(f"Interventions per target M{inter_scale}")

    quantity_label = {
        "f1": "Mean best data F1",
        "gain": "Mean data gain over names-only",
        "valid_rate": "Mean valid rate",
    }[args.quantity]
    ax.set_title(f"{quantity_label} by data budget ({args.anonymization})", fontsize=11, pad=8)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label(quantity_label)
    fig.text(
        0.01,
        0.01,
        "Each cell chooses the better of summary/matrix per graph and model; n is graph-model coverage.",
        ha="left",
        va="bottom",
        fontsize=7.2,
    )
    fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.20)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args)
    cells.to_csv(args.out_dir / f"{stem}.csv", index=False)
    print(f"[write] {args.out_dir / f'{stem}.csv'}")
    for fmt in args.formats:
        out_path = args.out_dir / f"{stem}.{fmt}"
        fig.savefig(out_path)
        print(f"[write] {out_path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(resolve(args.best_configs_csv))
    if args.one_per_model_graph:
        models = list(args.models)
        graphs = list(args.graphs)
        for model in models:
            for graph in graphs:
                child = argparse.Namespace(**vars(args))
                child.models = [model]
                child.graphs = [graph]
                cells = collect_budget_cells(df, child)
                if cells.empty:
                    print(f"[skip] no cells for model={model} graph={graph}")
                    continue
                plot(cells, child)
        return

    cells = collect_budget_cells(df, args)
    plot(cells, args)


if __name__ == "__main__":
    main()
