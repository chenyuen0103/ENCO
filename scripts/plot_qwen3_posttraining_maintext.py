#!/usr/bin/env python3
"""Plot the main-text base-vs-post-training probe for Qwen3-4B.

The figure uses an end-to-end score: invalid or unparseable graph outputs count
as zero directed-edge F1. This makes the plot reflect usable causal-graph
outputs rather than graph quality conditional on successful parsing.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

if not os.environ.get("MPLCONFIGDIR"):
    mplconfig = Path("/tmp") / f"matplotlib_{os.getuid()}"
    mplconfig.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mplconfig)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSES_DIR = REPO_ROOT / "scripts" / "responses"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "out" / "qwen3_posttraining_maintext"

BASE_MODEL = "Qwen3-4B-Thinking-2507"
FT_MODEL = "qwen3_4b_cd_format_v5_rerun_2gpu_checkpoint-100_merged"

GRAPH_ORDER = ["cancer", "earthquake", "asia", "sachs"]
GRAPH_LABELS = {
    "cancer": "Cancer",
    "earthquake": "Earthquake",
    "asia": "Asia",
    "sachs": "Sachs",
}
METHOD_ORDER = ["base_data", "ft_data"]
METHOD_LABELS = {
    "base_data": "Base Qwen3-4B\n+ data",
    "ft_data": "Fine-tuned Qwen3-4B\n+ data",
}
METHOD_COLORS = {
    "base_data": "#7B5EA7",
    "ft_data": "#59A14F",
}
LOW_VALID_THRESHOLD = 0.60
BUDGET_RE = re.compile(r"^responses_obs(?P<obs>\d+)_int(?P<inter>\d+)_")
BUDGET_ANY = {"*", "all", "any"}


@dataclass(frozen=True)
class Score:
    graph: str
    series: str
    label: str
    value: float
    ci95: float
    valid_rows: int
    total_rows: int
    source: str
    source_path: Path | None

    @property
    def valid_rate(self) -> float:
        return self.valid_rows / self.total_rows if self.total_rows else math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-dir", type=Path, default=DEFAULT_RESPONSES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--ft-model", default=FT_MODEL)
    parser.add_argument(
        "--obs",
        type=budget_arg,
        default="all",
        help="Observational sample budget to plot, or 'all' to search every obs* budget.",
    )
    parser.add_argument(
        "--inter",
        type=budget_arg,
        default="all",
        help="Interventional sample budget to plot, or 'all' to search every int* budget.",
    )
    parser.add_argument(
        "--ft-selection",
        choices=["best-real", "real-summary", "real-matrix", "best-any"],
        default="best-real",
        help=(
            "Which data-bearing cell to plot for both base and fine-tuned models. "
            "best-real chooses the better real-name summary/matrix cell per graph "
            "within the selected budget filter."
        ),
    )
    parser.add_argument(
        "--prompt-style",
        choices=["all", "plain", "wrapchat", "fmthint", "wrapchat-fmthint"],
        default="all",
        help=(
            "Prompt wrapper/filter to include. 'plain' means no wrapchat and no fmthint; "
            "'wrapchat' means chat wrapper without fmthint."
        ),
    )
    parser.add_argument("--formats", nargs="*", default=["pdf", "png"], choices=["pdf", "png", "svg"])
    parser.add_argument(
        "--no-appendix",
        action="store_true",
        help="Only write the main best-operating-point figure, not the config-by-config appendix figure.",
    )
    parser.add_argument(
        "--names-only",
        action="store_true",
        help="Plot the best names-only config for each model/graph instead of data-bearing obs/int configs.",
    )
    return parser.parse_args()


def budget_arg(raw: str) -> int | None:
    text = str(raw).strip().lower()
    if text in BUDGET_ANY:
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("budget must be an integer or one of: all, any, *") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("budget must be non-negative")
    return value


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 9.5,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
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


def finite_float(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def end_to_end_values(path: Path, metric: str = "f1") -> tuple[list[float], int, int]:
    values: list[float] = []
    valid_rows = 0
    total_rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1
            value = finite_float(row.get(metric))
            if value is None:
                values.append(0.0)
            else:
                valid_rows += 1
                values.append(value)
    return values, valid_rows, total_rows


def score_path(graph: str, series: str, label: str, path: Path, source: str) -> Score | None:
    values, valid_rows, total_rows = end_to_end_values(path)
    if not values:
        return None
    value = mean(values)
    ci95 = 1.96 * stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return Score(
        graph=graph,
        series=series,
        label=label,
        value=value,
        ci95=ci95,
        valid_rows=valid_rows,
        total_rows=total_rows,
        source=source,
        source_path=path,
    )


def choose_best(scores: list[Score]) -> Score | None:
    if not scores:
        return None
    return sorted(
        scores,
        key=lambda row: (
            row.value if math.isfinite(row.value) else -math.inf,
            row.valid_rate if math.isfinite(row.valid_rate) else -math.inf,
            row.valid_rows,
            row.total_rows,
        ),
        reverse=True,
    )[0]


def config_from_path(path: Path, model: str) -> str | None:
    suffix = f"_{model}.csv.per_row.csv"
    if not path.name.endswith(suffix):
        return None
    return path.name[: -len(suffix)]


def budget_from_config(config: str) -> tuple[int, int] | None:
    match = BUDGET_RE.match(config)
    if not match:
        return None
    return int(match.group("obs")), int(match.group("inter"))


def budget_matches(config: str, obs: int | None, inter: int | None) -> bool:
    budget = budget_from_config(config)
    if budget is None:
        return False
    config_obs, config_inter = budget
    if obs is not None and config_obs != obs:
        return False
    if inter is not None and config_inter != inter:
        return False
    return True


def budget_glob(obs: int | None, inter: int | None, model: str) -> str:
    obs_part = "*" if obs is None else str(obs)
    inter_part = "*" if inter is None else str(inter)
    return f"responses_obs{obs_part}_int{inter_part}_*_{model}.csv.per_row.csv"


def budget_stem(args: argparse.Namespace) -> str:
    obs_part = "all" if args.obs is None else str(args.obs)
    inter_part = "all" if args.inter is None else str(args.inter)
    return f"obs{obs_part}_int{inter_part}"


def prompt_style_stem(args: argparse.Namespace) -> str:
    return "" if args.prompt_style == "all" else f"_{args.prompt_style}"


def names_only_stem(args: argparse.Namespace) -> str:
    return f"qwen3_base_vs_finetuned_names_only_best{prompt_style_stem(args)}"


def budget_title(args: argparse.Namespace) -> str:
    obs_part = "*" if args.obs is None else str(args.obs)
    inter_part = "*" if args.inter is None else str(args.inter)
    if args.obs is None or args.inter is None:
        return f"best over N={obs_part}, M={inter_part}"
    return f"N={obs_part}, M={inter_part}"


def prompt_style_title(args: argparse.Namespace) -> str:
    if args.prompt_style == "all":
        return ""
    return f", {prompt_style_label(args.prompt_style)} prompts"


def budget_filter_title(args: argparse.Namespace) -> str:
    obs_part = "*" if args.obs is None else str(args.obs)
    inter_part = "*" if args.inter is None else str(args.inter)
    if args.obs is None and args.inter is None:
        return "all budgets"
    return f"N={obs_part}, M={inter_part}"


def config_prompt_style(config: str) -> str:
    has_wrapchat = "_wrapchat" in config
    has_fmthint = "_fmthint" in config
    if has_wrapchat and has_fmthint:
        return "wrapchat-fmthint"
    if has_wrapchat:
        return "wrapchat"
    if has_fmthint:
        return "fmthint"
    return "plain"


def prompt_style_label(style: str) -> str:
    labels = {
        "plain": "plain",
        "wrapchat": "chat",
        "fmthint": "hint",
        "wrapchat-fmthint": "chat+hint",
    }
    return labels.get(style, style)


def config_allowed(
    config: str,
    obs: int | None,
    inter: int | None,
    selection: str,
    prompt_style: str,
) -> bool:
    if not budget_matches(config, obs, inter):
        return False
    if prompt_style != "all" and config_prompt_style(config) != prompt_style:
        return False
    is_summary = "_summary" in config
    is_matrix = "_matrix" in config
    is_anon = "_anon_" in config
    if not (is_summary or is_matrix):
        return False
    if selection == "real-summary":
        return is_summary and not is_anon
    if selection == "real-matrix":
        return is_matrix and not is_anon
    if selection == "best-real":
        return not is_anon
    if selection == "best-any":
        return True
    raise ValueError(f"Unknown selection: {selection}")


def names_only_config_allowed(config: str, prompt_style: str) -> bool:
    if not config.startswith("responses_names_only"):
        return False
    if prompt_style != "all" and config_prompt_style(config) != prompt_style:
        return False
    return True


def names_only_source_label(config: str) -> str:
    style = prompt_style_label(config_prompt_style(config))
    extra = []
    if "reasonconcise" in config:
        extra.append("concise reasoning")
    if "colRandom" in config:
        extra.append("column-randomized")
    suffix = f"; {', '.join(extra)}" if extra else ""
    return f"names only ({style}{suffix})"


def config_source_label(config: str) -> str:
    budget = budget_from_config(config)
    budget_label = f"N={budget[0]}, M={budget[1]}; " if budget is not None else ""
    naming = "anon" if "_anon_" in config else "real"
    form = "summary" if "_summary" in config else "matrix" if "_matrix" in config else "other"
    wrapper = prompt_style_label(config_prompt_style(config))
    return f"{budget_label}{naming} {form} ({wrapper})"


def config_axis_label(config: str) -> str:
    budget = budget_from_config(config)
    budget_label = f"N={budget[0]}, M={budget[1]}\n" if budget is not None else ""
    naming = "anon" if "_anon_" in config else "real"
    form = "summary" if "_summary" in config else "matrix" if "_matrix" in config else "other"
    wrapper = prompt_style_label(config_prompt_style(config))
    return f"{budget_label}{naming} {form}\n{wrapper}"


def config_sort_key(config: str) -> tuple[int, int, int, int, int, str]:
    budget = budget_from_config(config)
    obs_idx, inter_idx = budget if budget is not None else (10**9, 10**9)
    naming_idx = 1 if "_anon_" in config else 0
    form_idx = 0 if "_summary" in config else 1 if "_matrix" in config else 2
    style_order = {"plain": 0, "wrapchat": 1, "fmthint": 2, "wrapchat-fmthint": 3}
    wrapper_idx = style_order.get(config_prompt_style(config), 99)
    return obs_idx, inter_idx, naming_idx, form_idx, wrapper_idx, config


def model_data_paths(
    responses_dir: Path,
    graph: str,
    model: str,
    obs: int | None,
    inter: int | None,
    selection: str,
    prompt_style: str,
) -> list[tuple[Path, str]]:
    graph_dir = responses_dir / graph
    paths: list[tuple[Path, str]] = []
    for path in sorted(graph_dir.glob(budget_glob(obs, inter, model))):
        config = config_from_path(path, model)
        if config is not None and config_allowed(config, obs, inter, selection, prompt_style):
            paths.append((path, config_source_label(config)))
    return paths


def model_names_only_paths(
    responses_dir: Path,
    graph: str,
    model: str,
    prompt_style: str,
) -> list[tuple[Path, str]]:
    graph_dir = responses_dir / graph
    paths: list[tuple[Path, str]] = []
    for path in sorted(graph_dir.glob(f"responses_names_only*_{model}.csv.per_row.csv")):
        config = config_from_path(path, model)
        if config is not None and names_only_config_allowed(config, prompt_style):
            paths.append((path, names_only_source_label(config)))
    return paths


def collect_scores(args: argparse.Namespace) -> list[Score]:
    scores: list[Score] = []
    for graph in GRAPH_ORDER:
        base_scores = [
            score
            for path, source in model_data_paths(
                args.responses_dir,
                graph,
                args.base_model,
                args.obs,
                args.inter,
                args.ft_selection,
                args.prompt_style,
            )
            if (score := score_path(graph, "base_data", "base + data", path, f"base {source}"))
            is not None
        ]
        scores.append(
            choose_best(base_scores)
            or Score(graph, "base_data", "base + data", math.nan, 0.0, 0, 0, "missing", None)
        )

        ft_scores = [
            score
            for path, source in model_data_paths(
                args.responses_dir,
                graph,
                args.ft_model,
                args.obs,
                args.inter,
                args.ft_selection,
                args.prompt_style,
            )
            if (score := score_path(graph, "ft_data", "post-trained + data", path, source)) is not None
        ]
        scores.append(
            choose_best(ft_scores)
            or Score(graph, "ft_data", "post-trained + data", math.nan, 0.0, 0, 0, "missing", None)
        )
    return scores


def collect_names_only_scores(args: argparse.Namespace) -> list[Score]:
    scores: list[Score] = []
    for graph in GRAPH_ORDER:
        base_scores = [
            score
            for path, source in model_names_only_paths(args.responses_dir, graph, args.base_model, args.prompt_style)
            if (score := score_path(graph, "base_data", "base names only", path, f"base {source}"))
            is not None
        ]
        scores.append(
            choose_best(base_scores)
            or Score(graph, "base_data", "base names only", math.nan, 0.0, 0, 0, "missing", None)
        )

        ft_scores = [
            score
            for path, source in model_names_only_paths(args.responses_dir, graph, args.ft_model, args.prompt_style)
            if (score := score_path(graph, "ft_data", "fine-tuned names only", path, source)) is not None
        ]
        scores.append(
            choose_best(ft_scores)
            or Score(graph, "ft_data", "fine-tuned names only", math.nan, 0.0, 0, 0, "missing", None)
        )
    return scores


def paired_config_scores(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for graph in GRAPH_ORDER:
        base_by_config: dict[str, Path] = {}
        ft_by_config: dict[str, Path] = {}

        for path in sorted((args.responses_dir / graph).glob(budget_glob(args.obs, args.inter, args.base_model))):
            config = config_from_path(path, args.base_model)
            if config is not None and config_allowed(
                config, args.obs, args.inter, args.ft_selection, args.prompt_style
            ):
                base_by_config[config] = path
        for path in sorted((args.responses_dir / graph).glob(budget_glob(args.obs, args.inter, args.ft_model))):
            config = config_from_path(path, args.ft_model)
            if config is not None and config_allowed(
                config, args.obs, args.inter, args.ft_selection, args.prompt_style
            ):
                ft_by_config[config] = path

        for config in sorted(set(base_by_config) & set(ft_by_config), key=config_sort_key):
            base_score = score_path(graph, "base_data", "base + data", base_by_config[config], config_source_label(config))
            ft_score = score_path(graph, "ft_data", "fine-tuned + data", ft_by_config[config], config_source_label(config))
            if base_score is None or ft_score is None:
                continue
            budget = budget_from_config(config)
            rows.append(
                {
                    "graph": graph,
                    "config": config,
                    "config_label": config_axis_label(config),
                    "obs": budget[0] if budget else "",
                    "inter": budget[1] if budget else "",
                    "prompt_style": config_prompt_style(config),
                    "base_value": base_score.value,
                    "base_ci95": base_score.ci95,
                    "base_valid_rows": base_score.valid_rows,
                    "base_total_rows": base_score.total_rows,
                    "ft_value": ft_score.value,
                    "ft_ci95": ft_score.ci95,
                    "ft_valid_rows": ft_score.valid_rows,
                    "ft_total_rows": ft_score.total_rows,
                    "delta": ft_score.value - base_score.value,
                    "base_path": str(base_score.source_path),
                    "ft_path": str(ft_score.source_path),
                }
            )
    return rows


def plot(
    scores: list[Score],
    args: argparse.Namespace,
    title: str | None = None,
    stem: str | None = None,
    legend_labels: dict[str, str] | None = None,
) -> None:
    configure_style()
    legend_labels = legend_labels or {
        "base_data": "Base Qwen3-4B + data",
        "ft_data": "Fine-tuned Qwen3-4B + data",
    }
    by_key = {(score.graph, score.series): score for score in scores}
    x = np.arange(len(GRAPH_ORDER))
    width = 0.28
    offsets = {
        "base_data": -width / 2,
        "ft_data": width / 2,
    }

    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)

    for series in METHOD_ORDER:
        values = []
        errors = []
        hatches = []
        for graph in GRAPH_ORDER:
            score = by_key[(graph, series)]
            values.append(0.0 if not math.isfinite(score.value) else score.value)
            errors.append(score.ci95 if math.isfinite(score.value) else 0.0)
            hatches.append("///" if score.valid_rate < LOW_VALID_THRESHOLD else None)
        rects = ax.bar(
            x + offsets[series],
            values,
            width=width,
            color=METHOD_COLORS[series],
            edgecolor="#333333",
            linewidth=0.75,
            hatch=hatches[0],
            label=METHOD_LABELS[series].replace("\n", " "),
            zorder=3,
        )
        # Matplotlib applies one hatch per BarContainer by default, so set per patch.
        for rect, hatch in zip(rects, hatches):
            rect.set_hatch(hatch)
        ax.errorbar(
            x + offsets[series],
            values,
            yerr=errors,
            fmt="none",
            ecolor="#333333",
            elinewidth=0.75,
            capsize=2.5,
            zorder=4,
        )
        for idx, (rect, graph) in enumerate(zip(rects, GRAPH_ORDER)):
            score = by_key[(graph, series)]
            if not math.isfinite(score.value):
                label = "--"
            else:
                label = f"{score.value:.2f}"
            if score.total_rows:
                label = f"{label}\n{score.valid_rows}/{score.total_rows}"
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                values[idx] + 0.026,
                label,
                ha="center",
                va="bottom",
                fontsize=7.7,
                linespacing=0.85,
                color="#222222",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([GRAPH_LABELS[graph] for graph in GRAPH_ORDER])
    ax.set_ylabel("End-to-end directed-edge F1")
    ax.set_ylim(0.0, 1.16)
    ax.set_title(title or f"Base vs fine-tuned Qwen3-4B ({budget_title(args)}{prompt_style_title(args)})", pad=7, fontsize=11)
    ax.text(
        0.01,
        -0.16,
        "Second line gives parsed graph outputs / total runs; invalid outputs score 0.",
        transform=ax.transAxes,
        fontsize=7.4,
        ha="left",
        va="top",
        color="#333333",
    )

    handles = [
        Patch(facecolor=METHOD_COLORS["base_data"], edgecolor="#333333", label=legend_labels["base_data"]),
        Patch(facecolor=METHOD_COLORS["ft_data"], edgecolor="#333333", label=legend_labels["ft_data"]),
        Patch(facecolor="white", edgecolor="#333333", hatch="///", label="<60% parsed LLM runs"),
    ]
    ax.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.31), frameon=False)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.74, bottom=0.22)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or f"qwen3_base_vs_finetuned_maintext_{budget_stem(args)}_{args.ft_selection}{prompt_style_stem(args)}"
    for fmt in args.formats:
        out_path = args.out_dir / f"{stem}.{fmt}"
        fig.savefig(out_path)
        print(f"[write] {out_path}")
    plt.close(fig)


def plot_config_by_config(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    if not rows:
        print("[warn] No paired config-by-config rows found; skipping appendix figure.")
        return
    configure_style()

    configs = sorted({str(row["config"]) for row in rows}, key=config_sort_key)
    config_labels = {str(row["config"]): str(row["config_label"]) for row in rows}
    data = np.full((len(GRAPH_ORDER), len(configs)), np.nan, dtype=float)
    cell_text: list[list[str]] = [["" for _ in configs] for _ in GRAPH_ORDER]
    for row in rows:
        graph_idx = GRAPH_ORDER.index(str(row["graph"]))
        config_idx = configs.index(str(row["config"]))
        base_value = float(row["base_value"])
        ft_value = float(row["ft_value"])
        delta = float(row["delta"])
        data[graph_idx, config_idx] = delta
        cell_text[graph_idx][config_idx] = (
            f"{base_value:.2f}->{ft_value:.2f}\n"
            f"{int(row['base_valid_rows'])}/{int(row['base_total_rows'])} -> "
            f"{int(row['ft_valid_rows'])}/{int(row['ft_total_rows'])}"
        )

    finite = data[np.isfinite(data)]
    limit = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite)))) if finite.size else 1.0
    limit = max(limit, 1e-6)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#eeeeee")
    norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig_width = max(6.8, 1.0 * len(configs) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, 3.6))
    im = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(np.arange(len(GRAPH_ORDER)))
    ax.set_yticklabels([GRAPH_LABELS[graph] for graph in GRAPH_ORDER])
    ax.set_xticks(np.arange(len(configs)))
    ax.set_xticklabels([config_labels[config] for config in configs], rotation=35, ha="right")
    ax.tick_params(axis="both", length=0)
    ax.set_title(
        f"Config-by-config fine-tuning deltas ({budget_filter_title(args)}{prompt_style_title(args)})",
        pad=8,
    )

    for i in range(len(GRAPH_ORDER)):
        for j in range(len(configs)):
            if np.isfinite(data[i, j]):
                ax.text(j, i, cell_text[i][j], ha="center", va="center", fontsize=7.2, color="#222222")

    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.018)
    cbar.ax.set_ylabel("F1 delta\n(FT - base)", rotation=90, va="center", labelpad=11)
    fig.text(
        0.01,
        -0.02,
        "Cell text: base F1 -> fine-tuned F1; second line is parsed graph outputs / total runs. Blank cells have no paired files.",
        ha="left",
        va="top",
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.12, right=0.88, top=0.84, bottom=0.30)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"qwen3_base_vs_finetuned_config_by_config_"
        f"{budget_stem(args)}_{args.ft_selection}{prompt_style_stem(args)}"
    )
    for fmt in args.formats:
        out_path = args.out_dir / f"{stem}.{fmt}"
        fig.savefig(out_path)
        print(f"[write] {out_path}")
    plt.close(fig)


def write_sources(scores: list[Score], args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        args.out_dir
        / f"qwen3_base_vs_finetuned_maintext_{budget_stem(args)}_{args.ft_selection}{prompt_style_stem(args)}.csv"
    )
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "graph",
                "series",
                "obs",
                "inter",
                "prompt_style",
                "config",
                "value",
                "ci95",
                "valid_rows",
                "total_rows",
                "valid_rate",
                "source",
                "source_path",
            ],
        )
        writer.writeheader()
        for score in scores:
            model = args.base_model if score.series == "base_data" else args.ft_model
            config = config_from_path(score.source_path, model) if score.source_path is not None else None
            budget = budget_from_config(config) if config is not None else None
            writer.writerow(
                {
                    "graph": score.graph,
                    "series": score.series,
                    "obs": "" if budget is None else budget[0],
                    "inter": "" if budget is None else budget[1],
                    "prompt_style": "" if config is None else config_prompt_style(config),
                    "config": "" if config is None else config,
                    "value": "" if not math.isfinite(score.value) else f"{score.value:.8g}",
                    "ci95": f"{score.ci95:.8g}",
                    "valid_rows": score.valid_rows,
                    "total_rows": score.total_rows,
                    "valid_rate": "" if not math.isfinite(score.valid_rate) else f"{score.valid_rate:.8g}",
                    "source": score.source,
                    "source_path": "" if score.source_path is None else str(score.source_path),
                }
            )
    print(f"[write] {out_path}")


def write_config_by_config(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        args.out_dir
        / f"qwen3_base_vs_finetuned_config_by_config_"
        f"{budget_stem(args)}_{args.ft_selection}{prompt_style_stem(args)}.csv"
    )
    fieldnames = [
        "graph",
        "obs",
        "inter",
        "prompt_style",
        "config",
        "config_label",
        "base_value",
        "base_ci95",
        "base_valid_rows",
        "base_total_rows",
        "ft_value",
        "ft_ci95",
        "ft_valid_rows",
        "ft_total_rows",
        "delta",
        "base_path",
        "ft_path",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        f"{float(row[key]):.8g}"
                        if key in {"base_value", "base_ci95", "ft_value", "ft_ci95", "delta"}
                        else row[key]
                    )
                    for key in fieldnames
                }
            )
    print(f"[write] {out_path}")


def write_names_only_sources(scores: list[Score], args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{names_only_stem(args)}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "graph",
                "series",
                "prompt_style",
                "config",
                "value",
                "ci95",
                "valid_rows",
                "total_rows",
                "valid_rate",
                "source",
                "source_path",
            ],
        )
        writer.writeheader()
        for score in scores:
            model = args.base_model if score.series == "base_data" else args.ft_model
            config = config_from_path(score.source_path, model) if score.source_path is not None else None
            writer.writerow(
                {
                    "graph": score.graph,
                    "series": score.series,
                    "prompt_style": "" if config is None else config_prompt_style(config),
                    "config": "" if config is None else config,
                    "value": "" if not math.isfinite(score.value) else f"{score.value:.8g}",
                    "ci95": f"{score.ci95:.8g}",
                    "valid_rows": score.valid_rows,
                    "total_rows": score.total_rows,
                    "valid_rate": "" if not math.isfinite(score.valid_rate) else f"{score.valid_rate:.8g}",
                    "source": score.source,
                    "source_path": "" if score.source_path is None else str(score.source_path),
                }
            )
    print(f"[write] {out_path}")


def main() -> None:
    args = parse_args()
    if args.names_only:
        scores = collect_names_only_scores(args)
        prompt_suffix = prompt_style_title(args)
        title = f"Base vs fine-tuned Qwen3-4B names only{prompt_suffix}"
        plot(
            scores,
            args,
            title=title,
            stem=names_only_stem(args),
            legend_labels={
                "base_data": "Base Qwen3-4B names only",
                "ft_data": "Fine-tuned Qwen3-4B names only",
            },
        )
        write_names_only_sources(scores, args)
        return

    scores = collect_scores(args)
    plot(scores, args)
    write_sources(scores, args)
    if not args.no_appendix:
        config_rows = paired_config_scores(args)
        plot_config_by_config(config_rows, args)
        write_config_by_config(config_rows, args)


if __name__ == "__main__":
    main()
