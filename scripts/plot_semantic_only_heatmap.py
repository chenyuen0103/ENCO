#!/usr/bin/env python3
"""Plot semantic-only MICAD performance across graphs and models.

This script creates the first Results-section heatmap for the names_only
condition. By default it reads the selected best-valid-rate prompt configs from
``experiments/out/best_prompt_configs_by_valid_rate.csv``.

  experiments/out/micad_paper/semantic_only_heatmap_data.csv
  experiments/out/micad_paper/figures/semantic_only_heatmap.{pdf,png}
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

GRAPH_ORDER = ["cancer", "earthquake", "asia", "sachs"]
GRAPH_LABELS = {
    "cancer": "Cancer",
    "earthquake": "Earthquake",
    "asia": "Asia",
    "sachs": "Sachs",
}

MODEL_ORDER = [
    "GPT-5 Mini",
    "GPT-5.2 Pro",
    "Qwen3-4B",
    "Qwen3-30B-A3B",
    "Qwen2.5-7B",
    "Qwen2.5-14B",
    "Qwen2.5-72B",
    "Llama-3.1-8B",
    "Llama-3.1-8B-Inst.",
    "Llama-3.1-70B",
]

FAMILY_BOUNDARIES = [1.5, 3.5, 6.5]

MODEL_DISPLAY_BY_RAW = {
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-5.2-pro": "GPT-5.2 Pro",
    "Qwen3-4B-Thinking-2507": "Qwen3-4B",
    "Qwen3-30B-A3B-Thinking-2507": "Qwen3-30B-A3B",
    "Qwen2.5-7B-Instruct-1M": "Qwen2.5-7B",
    "Qwen2.5-14B-Instruct-1M": "Qwen2.5-14B",
    "Qwen2.5-72B-Instruct-AWQ": "Qwen2.5-72B",
    "Meta-Llama-3.1-8B": "Llama-3.1-8B",
    "Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B-Inst.",
    "Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-8B": "Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B-Inst.",
    "meta-llama/Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "Qwen/Qwen3-4B-Thinking-2507": "Qwen3-4B",
    "Qwen/Qwen3-30B-A3B-Thinking-2507": "Qwen3-30B-A3B",
    "Qwen/Qwen2.5-7B-Instruct-1M": "Qwen2.5-7B",
    "Qwen/Qwen2.5-14B-Instruct-1M": "Qwen2.5-14B",
    "Qwen/Qwen2.5-72B-Instruct-AWQ": "Qwen2.5-72B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("experiments/out/micad_paper"),
        help="Paper output directory.",
    )
    parser.add_argument(
        "--best-configs-csv",
        type=Path,
        default=Path("experiments/out/best_prompt_configs_by_valid_rate.csv"),
        help="CSV produced by scripts/collect_best_prompt_configs.py.",
    )
    parser.add_argument(
        "--responses-root",
        type=Path,
        default=Path("scripts/responses"),
        help="Legacy root containing per-graph eval_summary.csv files; used only with --use-eval-summaries.",
    )
    parser.add_argument(
        "--use-eval-summaries",
        action="store_true",
        help="Use legacy per-graph eval_summary.csv files instead of --best-configs-csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Figure output directory. Defaults to <paper-dir>/figures.",
    )
    parser.add_argument(
        "--data-out",
        type=Path,
        default=None,
        help="Long-form plot-data CSV. Defaults to <paper-dir>/semantic_only_heatmap_data.csv.",
    )
    parser.add_argument(
        "--formats",
        nargs="*",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
        help="Figure formats to write.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=MODEL_ORDER,
        help="Model display names to include as heatmap columns.",
    )
    parser.add_argument(
        "--graphs",
        nargs="*",
        default=GRAPH_ORDER,
        help="Graphs to include as heatmap rows.",
    )
    parser.add_argument(
        "--stem",
        default="semantic_only_heatmap",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--show-title",
        action="store_true",
        help="Add an in-figure title. Omit for paper-ready caption-driven figures.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    repo_relative = REPO_ROOT / path
    return repo_relative if repo_relative.exists() or not path.parent.exists() else path


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 8,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def normalize_model_display(model: object) -> str:
    raw = "" if pd.isna(model) else str(model).strip()
    if raw in MODEL_DISPLAY_BY_RAW:
        return MODEL_DISPLAY_BY_RAW[raw]
    basename = raw.split("/")[-1]
    if basename in MODEL_DISPLAY_BY_RAW:
        return MODEL_DISPLAY_BY_RAW[basename]
    return raw


def read_response_summaries(responses_root: Path, graphs: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for graph in graphs:
        path = responses_root / graph / "eval_summary.csv"
        if not path.exists():
            print(f"[warn] Missing eval summary for graph {graph!r}: {path}")
            continue
        frame = pd.read_csv(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No per-graph eval_summary.csv files found under {responses_root}")
    df = pd.concat(frames, ignore_index=True)
    required = {
        "response_csv",
        "dataset",
        "model",
        "prompt_style",
        "avg_f1",
        "avg_shd",
        "var_f1_sd",
        "var_shd_sd",
        "valid_rows",
        "num_rows",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Response summaries under {responses_root} missing required columns: {missing}")
    return df


def semantic_only_candidates(df: pd.DataFrame, graphs: list[str], models: list[str]) -> pd.DataFrame:
    out = df[(df["dataset"].isin(graphs)) & (df["prompt_style"].eq("names_only"))].copy()
    out["method_display"] = out["model"].map(normalize_model_display)
    out = out[out["method_display"].isin(models)].copy()

    rename = {
        "dataset": "graph",
        "avg_f1": "mean_f1",
        "avg_shd": "mean_shd",
        "var_f1_sd": "std_f1",
        "var_shd_sd": "std_shd",
        "num_rows": "n_rows",
        "response_csv": "path",
    }
    out = out.rename(columns=rename)
    for col in ["mean_f1", "mean_shd", "std_f1", "std_shd", "valid_rows", "n_rows"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["valid_rate"] = out["valid_rows"] / out["n_rows"].replace(0, np.nan)
    out["uses_wrapchat_fmthint"] = out["path"].astype(str).str.contains("wrapchat_fmthint", regex=False)
    out["selected_config"] = np.where(out["uses_wrapchat_fmthint"], "wrapchat_fmthint", "no_wrapchat_fmthint")
    out["semantic"] = "names_only"
    out["format"] = "names_only"
    out["condition"] = "names_only"
    out["source_kind"] = "eval_summary_valid_rate_selection"
    out["variant"] = out["path"].map(lambda p: Path(str(p)).name)
    out["is_data_only"] = False
    out["is_llm_model"] = True
    out["config_tie_order"] = out["uses_wrapchat_fmthint"].astype(int)
    return out


def semantic_only_rows(df: pd.DataFrame, graphs: list[str], models: list[str]) -> pd.DataFrame:
    out = semantic_only_candidates(df, graphs, models)

    # Select the prompt wrapper variant by parsed-output valid rate. If validity
    # ties, prefer the non-wrapchat_fmthint prompt to avoid choosing by score.
    out = (
        out.sort_values(
            ["graph", "method_display", "valid_rate", "valid_rows", "n_rows", "config_tie_order", "variant"],
            ascending=[True, True, False, False, False, True, True],
        )
        .drop_duplicates(["graph", "method_display"], keep="first")
        .copy()
    )
    out["graph_label"] = out["graph"].map(GRAPH_LABELS).fillna(out["graph"])
    out["model_order"] = out["method_display"].map({model: idx for idx, model in enumerate(models)})
    out["graph_order"] = out["graph"].map({graph: idx for idx, graph in enumerate(graphs)})
    return out.sort_values(["graph_order", "model_order"]).reset_index(drop=True)


def read_best_configs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Best-config CSV not found: {path}")
    df = pd.read_csv(path)
    required = {
        "graph",
        "model",
        "obs_n",
        "int_n",
        "prompt_style",
        "anonymize",
        "valid_rate",
        "avg_f1",
        "avg_shd",
        "valid_rows",
        "num_rows",
        "prompt_config_key",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Best-config CSV {path} missing required columns: {missing}")
    return df


def semantic_only_rows_from_best_configs(df: pd.DataFrame, graphs: list[str], models: list[str]) -> pd.DataFrame:
    out = df[
        df["graph"].isin(graphs)
        & df["prompt_style"].eq("names_only")
        & pd.to_numeric(df["obs_n"], errors="coerce").eq(0)
        & pd.to_numeric(df["int_n"], errors="coerce").eq(0)
        & pd.to_numeric(df["anonymize"], errors="coerce").fillna(0).eq(0)
    ].copy()
    out["method_display"] = out["model"].map(normalize_model_display)
    out = out[out["method_display"].isin(models)].copy()

    rename = {
        "avg_f1": "mean_f1",
        "avg_shd": "mean_shd",
        "num_rows": "n_rows",
        "response_csv": "path",
        "prompt_config_key": "selected_config",
    }
    out = out.rename(columns=rename)
    for col in ["mean_f1", "mean_shd", "valid_rate", "valid_rows", "n_rows"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "path" not in out.columns:
        out["path"] = ""
    if "response_basename" in out.columns:
        out["variant"] = out["response_basename"]
    else:
        out["variant"] = out["selected_config"]
    out["semantic"] = "names_only"
    out["format"] = "names_only"
    out["condition"] = "names_only"
    out["source_kind"] = "best_prompt_configs_by_valid_rate"
    out["is_data_only"] = False
    out["is_llm_model"] = True

    # The input CSV should already have one selected row per
    # graph/model/obs/int/style/anonymize group. This final sort/drop makes the
    # plotting script robust to accidental duplicates.
    out = (
        out.sort_values(
            ["graph", "method_display", "valid_rate", "mean_f1", "valid_rows", "variant"],
            ascending=[True, True, False, False, False, True],
            na_position="last",
        )
        .drop_duplicates(["graph", "method_display"], keep="first")
        .copy()
    )
    out["graph_label"] = out["graph"].map(GRAPH_LABELS).fillna(out["graph"])
    out["model_order"] = out["method_display"].map({model: idx for idx, model in enumerate(models)})
    out["graph_order"] = out["graph"].map({graph: idx for idx, graph in enumerate(graphs)})
    return out.sort_values(["graph_order", "model_order"]).reset_index(drop=True)


def complete_grid(rows: pd.DataFrame, graphs: list[str], models: list[str]) -> pd.DataFrame:
    index = pd.MultiIndex.from_product([graphs, models], names=["graph", "method_display"])
    grid = rows.set_index(["graph", "method_display"]).reindex(index).reset_index()
    grid["graph_label"] = grid["graph"].map(GRAPH_LABELS).fillna(grid["graph"])
    grid["model_order"] = grid["method_display"].map({model: idx for idx, model in enumerate(models)})
    grid["graph_order"] = grid["graph"].map({graph: idx for idx, graph in enumerate(graphs)})
    return grid.sort_values(["graph_order", "model_order"]).reset_index(drop=True)


def build_matrices(grid: pd.DataFrame, graphs: list[str], models: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f1 = np.full((len(graphs), len(models)), np.nan)
    valid = np.full((len(graphs), len(models)), np.nan)
    total = np.full((len(graphs), len(models)), np.nan)
    for row in grid.itertuples(index=False):
        i = graphs.index(row.graph)
        j = models.index(row.method_display)
        f1[i, j] = row.mean_f1 if pd.notna(row.mean_f1) else np.nan
        valid[i, j] = row.valid_rows if pd.notna(row.valid_rows) else np.nan
        total[i, j] = row.n_rows if pd.notna(row.n_rows) else np.nan
    return f1, valid, total


def annotate_cells(ax: plt.Axes, f1: np.ndarray, valid: np.ndarray, total: np.ndarray) -> None:
    for i in range(f1.shape[0]):
        for j in range(f1.shape[1]):
            value = f1[i, j]
            v = valid[i, j]
            n = total[i, j]
            if np.isfinite(value):
                label = f"{value:.2f}"
                if np.isfinite(v) and np.isfinite(n) and v < n:
                    label += f"\n{int(v)}/{int(n)}"
                text_color = "white" if value >= 0.62 else "black"
                ax.text(j, i, label, ha="center", va="center", color=text_color, fontsize=6.6)
            else:
                label = "--"
                if np.isfinite(v) and np.isfinite(n):
                    label = f"{int(v)}/{int(n)}"
                ax.text(j, i, label, ha="center", va="center", color="#555555", fontsize=6.6)


def plot_heatmap(grid: pd.DataFrame, graphs: list[str], models: list[str], show_title: bool) -> plt.Figure:
    f1, valid, total = build_matrices(grid, graphs, models)
    masked = np.ma.masked_invalid(f1)

    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#eeeeee")

    fig, ax = plt.subplots(figsize=(7.15, 2.95))
    image = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=42, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(graphs)))
    ax.set_yticklabels([GRAPH_LABELS.get(graph, graph) for graph in graphs])
    ax.set_ylabel("Graph")

    ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(graphs), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for boundary in FAMILY_BOUNDARIES:
        if boundary < len(models) - 0.5:
            ax.axvline(boundary, color="#333333", linewidth=0.8)

    if show_title:
        ax.set_title("Semantic-only graph recovery (names_only)")

    annotate_cells(ax, f1, valid, total)

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Directed-edge F1")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: Iterable[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path)
        print(f"[write] {path}")


def main() -> None:
    args = parse_args()
    configure_style()

    paper_dir = resolve_path(args.paper_dir)
    best_configs_csv = resolve_path(args.best_configs_csv)
    responses_root = resolve_path(args.responses_root)
    out_dir = resolve_path(args.out_dir) if args.out_dir else paper_dir / "figures"
    data_out = resolve_path(args.data_out) if args.data_out else paper_dir / "semantic_only_heatmap_data.csv"

    graphs = [graph.lower() for graph in args.graphs]
    unknown_graphs = sorted(set(graphs) - set(GRAPH_ORDER))
    if unknown_graphs:
        raise ValueError(f"Unknown graph(s): {unknown_graphs}. Supported: {GRAPH_ORDER}")

    if args.use_eval_summaries:
        df = read_response_summaries(responses_root, graphs)
        rows = semantic_only_rows(df, graphs, args.models)
    else:
        df = read_best_configs(best_configs_csv)
        rows = semantic_only_rows_from_best_configs(df, graphs, args.models)
    grid = complete_grid(rows, graphs, args.models)

    data_out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(data_out, index=False)
    print(f"[write] {data_out}")

    missing = grid["mean_f1"].isna().sum()
    if missing:
        print(f"[warn] {missing} graph/model cells have no valid F1 and are shown as gray cells.")

    fig = plot_heatmap(grid, graphs, args.models, args.show_title)
    save_figure(fig, out_dir, args.stem, args.formats)
    plt.close(fig)


if __name__ == "__main__":
    main()
