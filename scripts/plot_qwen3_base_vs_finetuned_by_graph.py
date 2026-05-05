#!/usr/bin/env python3
"""Plot Qwen3-4B base vs fine-tuned performance by graph."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
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
DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "out" / "qwen3_base_vs_finetuned"
BASE_MODEL = "Qwen3-4B-Thinking-2507"
FT_MODEL = "qwen3_4b_cd_format_v5_rerun_2gpu_checkpoint-100_merged"
MODEL_LABELS = {
    "base": "Base Qwen3-4B",
    "ft": "Fine-tuned",
}
MODEL_COLORS = {
    "base": "#7b5ea7",
    "ft": "#59a14f",
}
GRAPH_ORDER = ["asia", "cancer", "earthquake", "sachs"]
GRAPH_LABELS = {
    "asia": "Asia",
    "cancer": "Cancer",
    "earthquake": "Earthquake",
    "sachs": "Sachs",
}
LOW_VALID_THRESHOLD = 0.60
LOWER_BETTER_METRICS = {"shd", "nhd", "nhd_ratio"}
CONFIG_MODEL_MARKERS = ("_matrix_", "_summary_", "_colRandom_")


@dataclass(frozen=True)
class BarStats:
    value: float
    ci95: float
    valid_rate: float
    n_configs: int
    n_metric_rows: int
    n_total_rows: int


@dataclass(frozen=True)
class ParsedResponseFile:
    graph: str
    config: str
    model: str
    path: Path


@dataclass(frozen=True)
class DeltaStats:
    model: str
    value: float
    ci95: float
    n_metric_pairs: int
    n_total_pairs: int
    win_rate: float
    valid_delta: float
    n_valid_pairs: int
    n_graphs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-dir", type=Path, default=DEFAULT_RESPONSES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metric", default="f1", help="Per-row metric to plot; default: f1.")
    parser.add_argument(
        "--all-finetuned",
        action="store_true",
        help=(
            "Compare every evaluated Qwen3-4B fine-tuned checkpoint against the base model. "
            "Writes a ranking dot plot, condition heatmap, validity-vs-quality scatter, and CSV summary."
        ),
    )
    parser.add_argument(
        "--max-heatmap-cols",
        type=int,
        default=0,
        help="Optionally cap the all-finetuned heatmap to the most-covered N conditions; 0 keeps all.",
    )
    parser.add_argument(
        "--pairing",
        choices=["normalized", "exact"],
        default="normalized",
        help=(
            "Use exact filename configuration matches, or normalize prompt-wrapper tokens "
            "such as wrapchat_fmthint before pairing. Normalized gives all four graphs."
        ),
    )
    parser.add_argument(
        "--anonymization",
        choices=["all", "real", "anon"],
        default="all",
        help="Filter response configs by anonymization condition. 'real' excludes *_anon_* configs.",
    )
    parser.add_argument(
        "--obs",
        type=int,
        default=None,
        help="Only include configs with this observational sample budget, e.g. --obs 5000.",
    )
    parser.add_argument(
        "--inter",
        type=int,
        default=None,
        help="Only include configs with this interventional sample budget, e.g. --inter 200.",
    )
    parser.add_argument(
        "--prompt-form",
        choices=["all", "summary", "matrix", "names_only"],
        default="all",
        help="Only include a prompt/output form. names_only has no obs/int budget in the filename.",
    )
    parser.add_argument("--formats", nargs="*", default=["pdf", "png"], choices=["pdf", "png", "svg"])
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
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


def config_key(
    path: Path,
    model: str,
    responses_dir: Path,
    pairing: str,
    anonymization: str,
    obs: int | None,
    inter: int | None,
    prompt_form: str,
) -> tuple[str, str] | None:
    rel = path.relative_to(responses_dir)
    graph = rel.parts[0]
    suffix = ".csv.per_row.csv"
    if not path.name.endswith(suffix):
        return None
    stem = path.name[: -len(suffix)]
    model_suffix = f"_{model}"
    if not stem.endswith(model_suffix):
        return None
    config = stem[: -len(model_suffix)]
    is_anon = "_anon_" in config
    if anonymization == "real" and is_anon:
        return None
    if anonymization == "anon" and not is_anon:
        return None
    if not config_matches_budget(config, obs=obs, inter=inter, prompt_form=prompt_form):
        return None
    if pairing == "normalized":
        for token in ("_wrapchat_fmthint", "_wrapchat", "_fmthint"):
            config = config.replace(token, "")
    return graph, config


def config_matches_budget(config: str, *, obs: int | None, inter: int | None, prompt_form: str) -> bool:
    is_names_only = "names_only" in config
    is_summary = "summary" in config
    is_matrix = "matrix" in config

    if prompt_form == "names_only" and not is_names_only:
        return False
    if prompt_form == "summary" and not is_summary:
        return False
    if prompt_form == "matrix" and not is_matrix:
        return False
    if prompt_form == "all" and not (is_names_only or is_summary or is_matrix):
        return False

    parts = config.split("_")
    config_obs = next((part[3:] for part in parts if part.startswith("obs")), None)
    config_inter = next((part[3:] for part in parts if part.startswith("int")), None)

    if obs is not None:
        if config_obs is None or not config_obs.isdigit() or int(config_obs) != obs:
            return False
    if inter is not None:
        if config_inter is None or not config_inter.isdigit() or int(config_inter) != inter:
            return False
    return True


def paired_files(
    responses_dir: Path,
    pairing: str,
    anonymization: str,
    obs: int | None,
    inter: int | None,
    prompt_form: str,
) -> dict[tuple[str, str], dict[str, Path]]:
    models = {"base": BASE_MODEL, "ft": FT_MODEL}
    pairs: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    for model_key, model_name in models.items():
        for path in responses_dir.glob(f"**/*{model_name}.csv.per_row.csv"):
            key = config_key(
                path,
                model_name,
                responses_dir,
                pairing,
                anonymization,
                obs,
                inter,
                prompt_form,
            )
            if key is not None:
                pairs[key][model_key] = path
    return {key: value for key, value in pairs.items() if set(value) == {"base", "ft"}}


def parse_response_file(
    path: Path,
    responses_dir: Path,
    pairing: str,
    anonymization: str,
    obs: int | None,
    inter: int | None,
    prompt_form: str,
) -> ParsedResponseFile | None:
    suffix = ".csv.per_row.csv"
    if not path.name.endswith(suffix):
        return None
    try:
        rel = path.relative_to(responses_dir)
    except ValueError:
        return None
    if len(rel.parts) < 2:
        return None

    graph = rel.parts[0]
    stem = path.name[: -len(suffix)]
    marker_matches = [(stem.rfind(marker), marker) for marker in CONFIG_MODEL_MARKERS]
    marker_matches = [(idx, marker) for idx, marker in marker_matches if idx >= 0]
    if not marker_matches:
        return None

    idx, marker = max(marker_matches, key=lambda item: item[0])
    config = stem[: idx + len(marker) - 1]
    model = stem[idx + len(marker) :]
    if not model:
        return None

    is_anon = "_anon_" in config
    if anonymization == "real" and is_anon:
        return None
    if anonymization == "anon" and not is_anon:
        return None
    if not config_matches_budget(config, obs=obs, inter=inter, prompt_form=prompt_form):
        return None
    if pairing == "normalized":
        for token in ("_wrapchat_fmthint", "_wrapchat", "_fmthint"):
            config = config.replace(token, "")
    return ParsedResponseFile(graph=graph, config=config, model=model, path=path)


def response_file_index(
    responses_dir: Path,
    pairing: str,
    anonymization: str,
    obs: int | None,
    inter: int | None,
    prompt_form: str,
) -> dict[tuple[str, str], dict[str, tuple[Path, ...]]]:
    index: dict[tuple[str, str], dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in responses_dir.glob("**/*.csv.per_row.csv"):
        parsed = parse_response_file(
            path,
            responses_dir,
            pairing,
            anonymization,
            obs,
            inter,
            prompt_form,
        )
        if parsed is not None:
            index[(parsed.graph, parsed.config)][parsed.model].append(parsed.path)
    return {
        key: {model: tuple(paths) for model, paths in model_paths.items()}
        for key, model_paths in index.items()
    }


def is_qwen3_4b_finetune(model: str) -> bool:
    lower = model.lower()
    if model == BASE_MODEL:
        return False
    return "qwen3_4b" in lower or "grpo_from_qwen3_4b" in lower


def all_finetuned_pairs(
    responses_dir: Path,
    pairing: str,
    anonymization: str,
    obs: int | None,
    inter: int | None,
    prompt_form: str,
) -> dict[str, dict[tuple[str, str], dict[str, tuple[Path, ...]]]]:
    index = response_file_index(
        responses_dir,
        pairing,
        anonymization,
        obs,
        inter,
        prompt_form,
    )
    pairs_by_model: dict[str, dict[tuple[str, str], dict[str, tuple[Path, ...]]]] = defaultdict(dict)
    for key, model_paths in index.items():
        base_paths = model_paths.get(BASE_MODEL)
        if not base_paths:
            continue
        for model, ft_paths in model_paths.items():
            if is_qwen3_4b_finetune(model):
                pairs_by_model[model][key] = {"base": base_paths, "ft": ft_paths}
    return dict(pairs_by_model)


def numeric_values(path: Path, metric: str) -> tuple[list[float], int]:
    values: list[float] = []
    n_rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n_rows += 1
            raw = row.get(metric, "")
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value):
                values.append(value)
    return values, n_rows


def mean_for_paths(paths: tuple[Path, ...], metric: str) -> tuple[float, int, int]:
    file_means: list[float] = []
    n_metric_rows = 0
    n_total_rows = 0
    for path in paths:
        values, n_rows = numeric_values(path, metric)
        n_total_rows += n_rows
        if values:
            file_means.append(mean(values))
            n_metric_rows += len(values)
    if not file_means:
        return math.nan, n_metric_rows, n_total_rows
    return mean(file_means), n_metric_rows, n_total_rows


def paired_delta(base_value: float, ft_value: float, metric: str) -> float:
    if metric in LOWER_BETTER_METRICS:
        return base_value - ft_value
    return ft_value - base_value


def summarize(pairs: dict[tuple[str, str], dict[str, Path]], metric: str) -> dict[str, dict[str, BarStats]]:
    by_graph: dict[str, dict[str, list[tuple[float, int, int]]]] = defaultdict(lambda: defaultdict(list))
    for (graph, _config), files in pairs.items():
        for model_key, path in files.items():
            values, n_rows = numeric_values(path, metric)
            if values:
                by_graph[graph][model_key].append((mean(values), len(values), n_rows))
            else:
                by_graph[graph][model_key].append((math.nan, 0, n_rows))

    stats: dict[str, dict[str, BarStats]] = defaultdict(dict)
    for graph, model_rows in by_graph.items():
        for model_key, rows in model_rows.items():
            config_values = [value for value, _, _ in rows if math.isfinite(value)]
            n_metric_rows = sum(n for value, n, _ in rows if math.isfinite(value))
            n_total_rows = sum(total for _, _, total in rows)
            valid_rate = n_metric_rows / n_total_rows if n_total_rows else math.nan
            if config_values:
                value = mean(config_values)
                ci95 = (
                    1.96 * stdev(config_values) / math.sqrt(len(config_values))
                    if len(config_values) > 1
                    else 0.0
                )
            else:
                value = math.nan
                ci95 = 0.0
            stats[graph][model_key] = BarStats(
                value=value,
                ci95=ci95,
                valid_rate=valid_rate,
                n_configs=len(rows),
                n_metric_rows=n_metric_rows,
                n_total_rows=n_total_rows,
            )
    return stats


def metric_label(metric: str) -> str:
    labels = {
        "f1": "Directed-edge F1",
        "skeleton_f1": "Skeleton F1",
        "ancestor_f1": "Ancestor F1",
        "accuracy": "Directed-edge accuracy",
        "format_ok": "Format-valid rate",
        "nhd_ratio": "Normalized Hamming distance ratio",
    }
    return labels.get(metric, metric.replace("_", " ").title())


def delta_metric_label(metric: str) -> str:
    if metric in LOWER_BETTER_METRICS:
        return f"{metric_label(metric)} improvement vs base (base - fine-tuned)"
    return f"{metric_label(metric)} improvement vs base (fine-tuned - base)"


def compact_delta_metric_label(metric: str) -> str:
    return f"{metric_label(metric)} improvement vs base"


def model_display_name(model: str) -> str:
    exact = {
        "qwen3_4b_cd_format_v5_rerun_2gpu": "CD v5 2gpu",
        "qwen3_4b_cd_format_v5_then_sft_5way_v4_small": "CD to SFT small",
    }
    if model in exact:
        return exact[model]

    label = model
    replacements = [
        ("grpo_from_qwen3_4b_cd_format_v5_rerun_no_cancer_full_", "GRPO CD no-cancer "),
        ("grpo_from_qwen3_4b_sft_mix_guide_v2_lenfix_2gpu_", "GRPO SFT mix v2 "),
        ("grpo_from_qwen3_4b_sft_5way_v4_ckpt200_free_probe_", "GRPO SFT free-probe "),
        ("grpo_from_qwen3_4b_", "GRPO "),
        ("qwen3_4b_", ""),
        ("cd_format_v5_rerun_2gpu_", "CD v5 "),
        ("cd_format_v5_rerun_", "CD v5 "),
        ("cd_format_v5_then_", "CD v5 then "),
        ("sft_5way_v4_2gpu_", "SFT 5way "),
        ("sft_5way_v4_", "SFT 5way "),
        ("checkpoint-", "ckpt-"),
        ("_merged", " merged"),
        ("_", " "),
    ]
    for old, new in replacements:
        label = label.replace(old, new)
    return " ".join(label.split())


def plot(
    stats: dict[str, dict[str, BarStats]],
    out_dir: Path,
    metric: str,
    pairing: str,
    anonymization: str,
    obs: int | None,
    inter: int | None,
    prompt_form: str,
    formats: list[str],
) -> None:
    configure_style()
    graphs = [graph for graph in GRAPH_ORDER if graph in stats]
    x = np.arange(len(graphs))
    width = 0.30

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)

    for offset, model_key in [(-width / 2, "base"), (width / 2, "ft")]:
        for idx, graph in enumerate(graphs):
            bar = stats[graph].get(model_key)
            if bar is None:
                continue
            value = 0.0 if not math.isfinite(bar.value) else bar.value
            hatch = "///" if bar.valid_rate < LOW_VALID_THRESHOLD else None
            rects = ax.bar(
                x[idx] + offset,
                value,
                width=width,
                color=MODEL_COLORS[model_key],
                edgecolor="#333333",
                linewidth=0.8,
                hatch=hatch,
                label=MODEL_LABELS[model_key] if idx == 0 else None,
                zorder=3,
            )
            if math.isfinite(bar.value) and bar.ci95 > 0:
                ax.errorbar(
                    x[idx] + offset,
                    bar.value,
                    yerr=bar.ci95,
                    color="#333333",
                    linewidth=0.8,
                    capsize=3,
                    zorder=4,
                )
            label = "n/a" if not math.isfinite(bar.value) else f"{bar.value:.2f}"
            ax.text(
                x[idx] + offset,
                value + 0.025,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#222222",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([GRAPH_LABELS.get(graph, graph.title()) for graph in graphs])
    ax.set_ylabel(metric_label(metric))
    ax.set_ylim(0.0, 1.05 if metric != "shd" else None)
    condition_label = {
        "all": "all naming conditions",
        "real": "real names",
        "anon": "anonymized names",
    }[anonymization]
    budget_label = budget_description(obs=obs, inter=inter, prompt_form=prompt_form)
    ax.set_title(
        f"Qwen3-4B base vs fine-tuned performance by graph ({condition_label}, {budget_label})",
        pad=10,
    )

    handles = [
        Patch(facecolor=MODEL_COLORS["base"], edgecolor="#333333", label=MODEL_LABELS["base"]),
        Patch(facecolor=MODEL_COLORS["ft"], edgecolor="#333333", label=MODEL_LABELS["ft"]),
        Patch(facecolor="white", edgecolor="#333333", hatch="///", label="<60% valid runs"),
    ]
    ax.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17), frameon=False)

    subtitle = (
        f"Mean over paired response configurations ({pairing} pairing, {condition_label}, {budget_label}); "
        "error bars are 95% CIs over configs."
    )
    fig.text(0.5, -0.015, subtitle, ha="center", va="top", fontsize=8.5)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"qwen3_base_vs_finetuned_by_graph_{metric}_{pairing}_{anonymization}"
        f"_{budget_slug(obs=obs, inter=inter, prompt_form=prompt_form)}"
    )
    for fmt in formats:
        out_path = out_dir / f"{stem}.{fmt}"
        fig.savefig(out_path)
        print(f"[write] {out_path}")
    plt.close(fig)


def summarize_all_finetuned(
    pairs_by_model: dict[str, dict[tuple[str, str], dict[str, tuple[Path, ...]]]],
    metric: str,
) -> tuple[list[DeltaStats], dict[str, dict[tuple[str, str], float]], dict[str, dict[tuple[str, str], float]]]:
    condition_metric_deltas: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    condition_valid_deltas: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    rows: list[DeltaStats] = []

    for model, pairs in pairs_by_model.items():
        metric_deltas: list[float] = []
        valid_deltas: list[float] = []
        graphs: set[str] = set()
        for key, files in pairs.items():
            graph, _config = key
            graphs.add(graph)

            base_value, _, _ = mean_for_paths(files["base"], metric)
            ft_value, _, _ = mean_for_paths(files["ft"], metric)
            if math.isfinite(base_value) and math.isfinite(ft_value):
                delta = paired_delta(base_value, ft_value, metric)
                metric_deltas.append(delta)
                condition_metric_deltas[model][key] = delta

            base_valid, _, _ = mean_for_paths(files["base"], "format_ok")
            ft_valid, _, _ = mean_for_paths(files["ft"], "format_ok")
            if math.isfinite(base_valid) and math.isfinite(ft_valid):
                valid_delta = paired_delta(base_valid, ft_valid, "format_ok")
                valid_deltas.append(valid_delta)
                condition_valid_deltas[model][key] = valid_delta

        if not metric_deltas:
            continue
        value = mean(metric_deltas)
        ci95 = 1.96 * stdev(metric_deltas) / math.sqrt(len(metric_deltas)) if len(metric_deltas) > 1 else 0.0
        rows.append(
            DeltaStats(
                model=model,
                value=value,
                ci95=ci95,
                n_metric_pairs=len(metric_deltas),
                n_total_pairs=len(pairs),
                win_rate=sum(delta > 0 for delta in metric_deltas) / len(metric_deltas),
                valid_delta=mean(valid_deltas) if valid_deltas else math.nan,
                n_valid_pairs=len(valid_deltas),
                n_graphs=len(graphs),
            )
        )

    rows.sort(key=lambda row: (row.value, row.n_metric_pairs), reverse=True)
    return rows, dict(condition_metric_deltas), dict(condition_valid_deltas)


def condition_sort_key(key: tuple[str, str]) -> tuple[int, int, int, int, str]:
    graph, config = key
    graph_idx = GRAPH_ORDER.index(graph) if graph in GRAPH_ORDER else len(GRAPH_ORDER)
    form_idx = 0 if "names_only" in config else 1 if "summary" in config else 2 if "matrix" in config else 3
    anon_idx = 1 if "_anon_" in config else 0

    parts = config.split("_")
    obs_raw = next((part[3:] for part in parts if part.startswith("obs")), "")
    inter_raw = next((part[3:] for part in parts if part.startswith("int")), "")
    obs_idx = int(obs_raw) if obs_raw.isdigit() else -1
    inter_idx = int(inter_raw) if inter_raw.isdigit() else -1
    return graph_idx, form_idx, anon_idx, obs_idx, f"{inter_idx:08d}_{config}"


def condition_display_name(key: tuple[str, str]) -> str:
    graph, config = key
    graph_label = GRAPH_LABELS.get(graph, graph.title())
    naming = "anon" if "_anon_" in config else "real"
    if "names_only" in config:
        return f"{graph_label}\n{naming} names"

    form = "summary" if "summary" in config else "matrix" if "matrix" in config else "other"
    parts = config.split("_")
    obs_raw = next((part[3:] for part in parts if part.startswith("obs")), "?")
    inter_raw = next((part[3:] for part in parts if part.startswith("int")), "?")
    return f"{graph_label}\n{naming} {form}\nobs{obs_raw}/int{inter_raw}"


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = out_dir / f"{stem}.{fmt}"
        fig.savefig(out_path)
        print(f"[write] {out_path}")


def all_finetuned_stem(
    plot_kind: str,
    metric: str,
    pairing: str,
    anonymization: str,
    obs: int | None,
    inter: int | None,
    prompt_form: str,
) -> str:
    return (
        f"qwen3_all_finetuned_{plot_kind}_{metric}_{pairing}_{anonymization}"
        f"_{budget_slug(obs=obs, inter=inter, prompt_form=prompt_form)}"
    )


def plot_all_finetuned_ranking(
    rows: list[DeltaStats],
    out_dir: Path,
    metric: str,
    pairing: str,
    anonymization: str,
    obs: int | None,
    inter: int | None,
    prompt_form: str,
    formats: list[str],
) -> None:
    configure_style()
    labels = [model_display_name(row.model) for row in rows]
    values = np.array([row.value for row in rows], dtype=float)
    errors = np.array([row.ci95 for row in rows], dtype=float)
    y = np.arange(len(rows))
    colors = np.where(values >= 0, "#4E79A7", "#C44E52")

    fig_height = max(3.8, 0.34 * len(rows) + 1.6)
    fig, ax = plt.subplots(figsize=(7.4, fig_height))
    ax.set_facecolor("white")
    ax.grid(axis="x", color="#d8d8d8", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    ax.axvline(0, color="#222222", linewidth=0.9, linestyle="--", zorder=1)
    ax.errorbar(values, y, xerr=errors, fmt="none", ecolor="#333333", elinewidth=0.8, capsize=3, zorder=2)
    ax.scatter(values, y, s=34, color=colors, edgecolor="#333333", linewidth=0.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(delta_metric_label(metric))
    ax.set_title("All Qwen3-4B fine-tuned checkpoints vs base", pad=10)

    condition_label = {"all": "all naming conditions", "real": "real names", "anon": "anonymized names"}[
        anonymization
    ]
    budget_label = budget_description(obs=obs, inter=inter, prompt_form=prompt_form)
    fig.text(
        0.5,
        -0.01,
        (
            f"Mean paired change over graph/configuration cells ({pairing} pairing, "
            f"{condition_label}, {budget_label}); error bars are 95% CIs over cells."
        ),
        ha="center",
        va="top",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.31, right=0.98, top=0.91, bottom=0.18)

    stem = all_finetuned_stem("delta_rank", metric, pairing, anonymization, obs, inter, prompt_form)
    save_figure(fig, out_dir, stem, formats)
    plt.close(fig)


def plot_all_finetuned_heatmap(
    rows: list[DeltaStats],
    condition_metric_deltas: dict[str, dict[tuple[str, str], float]],
    out_dir: Path,
    metric: str,
    pairing: str,
    anonymization: str,
    obs: int | None,
    inter: int | None,
    prompt_form: str,
    max_cols: int,
    formats: list[str],
) -> None:
    configure_style()
    models = [row.model for row in rows]
    all_conditions = sorted(
        {condition for deltas in condition_metric_deltas.values() for condition in deltas},
        key=condition_sort_key,
    )
    if max_cols > 0 and len(all_conditions) > max_cols:
        all_conditions = sorted(
            all_conditions,
            key=lambda condition: (
                -sum(condition in condition_metric_deltas.get(model, {}) for model in models),
                condition_sort_key(condition),
            ),
        )[:max_cols]
        all_conditions = sorted(all_conditions, key=condition_sort_key)

    data = np.full((len(models), len(all_conditions)), np.nan, dtype=float)
    for row_idx, model in enumerate(models):
        for col_idx, condition in enumerate(all_conditions):
            value = condition_metric_deltas.get(model, {}).get(condition)
            if value is not None and math.isfinite(value):
                data[row_idx, col_idx] = value

    finite = data[np.isfinite(data)]
    if finite.size:
        limit = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))))
    else:
        limit = 1.0
    limit = max(limit, 1e-6)
    norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#eeeeee")

    fig_width = max(8.0, 0.36 * len(all_conditions) + 2.2)
    fig_height = max(4.5, 0.34 * len(models) + 2.1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([model_display_name(model) for model in models])
    ax.set_xticks(np.arange(len(all_conditions)))
    ax.set_xticklabels(
        [condition_display_name(condition) for condition in all_conditions],
        rotation=90,
        ha="center",
        va="top",
        fontsize=6.0 if len(all_conditions) > 50 else 7.0,
    )
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Paired graph/configuration")
    ax.set_title("Fine-tuned checkpoint improvement over base by condition", pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
    cbar.ax.set_ylabel(compact_delta_metric_label(metric), rotation=90, va="center", labelpad=10)
    fig.subplots_adjust(left=0.26, right=0.90, top=0.92, bottom=0.35)

    stem = all_finetuned_stem("delta_heatmap", metric, pairing, anonymization, obs, inter, prompt_form)
    save_figure(fig, out_dir, stem, formats)
    plt.close(fig)


def plot_all_finetuned_validity_scatter(
    rows: list[DeltaStats],
    out_dir: Path,
    metric: str,
    pairing: str,
    anonymization: str,
    obs: int | None,
    inter: int | None,
    prompt_form: str,
    formats: list[str],
) -> None:
    configure_style()
    plot_rows = [row for row in rows if math.isfinite(row.valid_delta)]
    if not plot_rows:
        return

    x_values = np.array([row.valid_delta for row in plot_rows], dtype=float)
    y_values = np.array([row.value for row in plot_rows], dtype=float)
    sizes = np.array([36 + 7 * math.sqrt(row.n_metric_pairs) for row in plot_rows], dtype=float)
    colors = np.where(y_values >= 0, "#4E79A7", "#C44E52")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_facecolor("white")
    ax.grid(color="#d8d8d8", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#222222", linewidth=0.9, linestyle="--", zorder=1)
    ax.axvline(0, color="#222222", linewidth=0.9, linestyle="--", zorder=1)
    ax.scatter(x_values, y_values, s=sizes, color=colors, edgecolor="#333333", linewidth=0.55, alpha=0.9, zorder=3)

    ax.margins(x=0.10, y=0.12)
    top_quality = set(sorted(range(len(plot_rows)), key=lambda idx: y_values[idx], reverse=True)[:1])
    top_validity = set(sorted(range(len(plot_rows)), key=lambda idx: x_values[idx], reverse=True)[:2])
    bottom_quality = set(sorted(range(len(plot_rows)), key=lambda idx: y_values[idx])[:2])
    label_indices = top_quality | top_validity | bottom_quality
    ordered_for_labels = sorted(label_indices, key=lambda idx: y_values[idx], reverse=True)
    label_rank = {idx: rank for rank, idx in enumerate(ordered_for_labels)}
    offsets = [7, -9, 15, -17, 23, -25]
    for idx, (row, x_value, y_value) in enumerate(zip(plot_rows, x_values, y_values)):
        if idx not in label_indices:
            continue
        x_offset = -6 if x_value > np.nanpercentile(x_values, 80) else 6
        ha = "right" if x_offset < 0 else "left"
        ax.annotate(
            model_display_name(row.model),
            xy=(x_value, y_value),
            xytext=(x_offset, offsets[label_rank[idx] % len(offsets)]),
            textcoords="offset points",
            fontsize=6.8,
            ha=ha,
            va="center",
            color="#333333",
            annotation_clip=False,
        )

    ax.set_xlabel("Format-valid rate improvement vs base")
    ax.set_ylabel(compact_delta_metric_label(metric))
    ax.set_title("Validity vs graph-quality change over base", pad=10)
    fig.subplots_adjust(left=0.26, right=0.98, top=0.90, bottom=0.16)

    stem = all_finetuned_stem("validity_quality_scatter", metric, pairing, anonymization, obs, inter, prompt_form)
    save_figure(fig, out_dir, stem, formats)
    plt.close(fig)


def write_all_finetuned_summary_csv(
    rows: list[DeltaStats],
    out_dir: Path,
    metric: str,
    pairing: str,
    anonymization: str,
    obs: int | None,
    inter: int | None,
    prompt_form: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = all_finetuned_stem("summary", metric, pairing, anonymization, obs, inter, prompt_form)
    out_path = out_dir / f"{stem}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "display_name",
                "metric",
                "mean_improvement",
                "ci95",
                "n_metric_pairs",
                "n_total_pairs",
                "win_rate",
                "format_ok_improvement",
                "n_format_ok_pairs",
                "n_graphs",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row.model,
                    "display_name": model_display_name(row.model),
                    "metric": metric,
                    "mean_improvement": f"{row.value:.8g}",
                    "ci95": f"{row.ci95:.8g}",
                    "n_metric_pairs": row.n_metric_pairs,
                    "n_total_pairs": row.n_total_pairs,
                    "win_rate": f"{row.win_rate:.8g}",
                    "format_ok_improvement": "" if not math.isfinite(row.valid_delta) else f"{row.valid_delta:.8g}",
                    "n_format_ok_pairs": row.n_valid_pairs,
                    "n_graphs": row.n_graphs,
                }
            )
    print(f"[write] {out_path}")


def plot_all_finetuned(
    responses_dir: Path,
    out_dir: Path,
    metric: str,
    pairing: str,
    anonymization: str,
    obs: int | None,
    inter: int | None,
    prompt_form: str,
    max_heatmap_cols: int,
    formats: list[str],
) -> None:
    pairs_by_model = all_finetuned_pairs(responses_dir, pairing, anonymization, obs, inter, prompt_form)
    if not pairs_by_model:
        raise SystemExit("No paired base/fine-tuned response files found.")

    rows, condition_metric_deltas, _condition_valid_deltas = summarize_all_finetuned(pairs_by_model, metric)
    if not rows:
        raise SystemExit(f"No paired numeric values found for metric {metric!r}.")

    plot_all_finetuned_ranking(
        rows,
        out_dir,
        metric,
        pairing,
        anonymization,
        obs,
        inter,
        prompt_form,
        formats,
    )
    plot_all_finetuned_heatmap(
        rows,
        condition_metric_deltas,
        out_dir,
        metric,
        pairing,
        anonymization,
        obs,
        inter,
        prompt_form,
        max_heatmap_cols,
        formats,
    )
    plot_all_finetuned_validity_scatter(
        rows,
        out_dir,
        metric,
        pairing,
        anonymization,
        obs,
        inter,
        prompt_form,
        formats,
    )
    write_all_finetuned_summary_csv(rows, out_dir, metric, pairing, anonymization, obs, inter, prompt_form)


def budget_description(*, obs: int | None, inter: int | None, prompt_form: str) -> str:
    pieces: list[str] = []
    if prompt_form != "all":
        pieces.append(prompt_form.replace("_", " "))
    if obs is not None:
        pieces.append(f"obs={obs}")
    if inter is not None:
        pieces.append(f"int={inter}")
    return ", ".join(pieces) if pieces else "all budgets/forms"


def budget_slug(*, obs: int | None, inter: int | None, prompt_form: str) -> str:
    pieces: list[str] = [prompt_form]
    if obs is not None:
        pieces.append(f"obs{obs}")
    if inter is not None:
        pieces.append(f"int{inter}")
    return "_".join(pieces)


def main() -> None:
    args = parse_args()
    if args.all_finetuned:
        plot_all_finetuned(
            args.responses_dir,
            args.out_dir,
            args.metric,
            args.pairing,
            args.anonymization,
            args.obs,
            args.inter,
            args.prompt_form,
            args.max_heatmap_cols,
            args.formats,
        )
        return

    pairs = paired_files(
        args.responses_dir,
        args.pairing,
        args.anonymization,
        args.obs,
        args.inter,
        args.prompt_form,
    )
    if not pairs:
        raise SystemExit("No paired response files found.")
    stats = summarize(pairs, args.metric)
    plot(
        stats,
        args.out_dir,
        args.metric,
        args.pairing,
        args.anonymization,
        args.obs,
        args.inter,
        args.prompt_form,
        args.formats,
    )


if __name__ == "__main__":
    main()
