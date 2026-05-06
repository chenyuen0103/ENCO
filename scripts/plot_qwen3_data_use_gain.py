#!/usr/bin/env python3
"""Plot whether fine-tuning increases data-use gain over names-only.

For each graph and fine-tuned checkpoint, this computes

    (F1_ft(real + data) - F1_ft(names only))
  - (F1_base(real + data) - F1_base(names only)).

Positive values mean the fine-tuned model extracts more incremental causal
discovery value from the supplied data than the base model does.
"""

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

from plot_qwen3_finetuned_ladder import (
    BASE_MODEL,
    DEFAULT_OUT_DIR,
    DEFAULT_RESPONSES_DIR,
    GRAPH_LABELS,
    GRAPH_ORDER,
    budget_arg,
    budget_matches,
    config_prompt_style,
    configure_style,
    end_to_end_values,
    is_qwen3_4b_finetune,
    metric_label,
    model_display_name,
    parse_response_file,
    prompt_style_label,
    resolve_graphs,
)


LOW_VALID_THRESHOLD = 0.60


MODEL_ALIASES = {
    "grpo_from_qwen3_4b_cd_format_v5_rerun_no_cancer_full_checkpoint-1200":
    "grpo_from_qwen3_4b_cd_format_v5_rerun_no_cancer_full_checkpoint-1200_merged",
}


def canonical_model_id(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


@dataclass(frozen=True)
class SelectedScore:
    model: str
    graph: str
    slot: str
    data_form: str
    value: float
    valid_rows: int
    total_rows: int
    config: str
    path: Path

    @property
    def valid_rate(self) -> float:
        return self.valid_rows / self.total_rows if self.total_rows else math.nan


@dataclass(frozen=True)
class GainRow:
    model: str
    value: float
    ci95: float
    graph_coverage: int
    graph_total: int
    positive_graphs: int
    valid_rows: int
    total_rows: int
    ft_gain: float
    base_gain: float

    @property
    def valid_rate(self) -> float:
        return self.valid_rows / self.total_rows if self.total_rows else math.nan


@dataclass(frozen=True)
class DetailRow:
    model: str
    graph: str
    base_names: SelectedScore
    base_data: SelectedScore
    ft_names: SelectedScore
    ft_data: SelectedScore
    base_gain: float
    ft_gain: float
    delta_gain: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-dir", type=Path, default=DEFAULT_RESPONSES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--graphs", nargs="*", default=GRAPH_ORDER, help="Graphs to average, or 'all'.")
    parser.add_argument("--obs", type=budget_arg, default="all")
    parser.add_argument("--inter", type=budget_arg, default="all")
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--anonymization", choices=["real", "anon", "all"], default="real")
    parser.add_argument("--data-form", choices=["best", "summary", "matrix"], default="best")
    parser.add_argument(
        "--prompt-style",
        choices=["all", "plain", "wrapchat", "fmthint", "wrapchat-fmthint"],
        default="all",
    )
    parser.add_argument("--models", nargs="*", default=None, help="Optional exact fine-tuned model ids to include.")
    parser.add_argument("--top-models", type=int, default=0, help="Keep only the top N checkpoints by delta gain.")
    parser.add_argument(
        "--best-per-dataset",
        action="store_true",
        help="Plot one bar per dataset, selecting the fine-tuned method with the largest data-use gain on that dataset.",
    )
    parser.add_argument(
        "--shared-ft-model",
        default=None,
        help=(
            "With --best-per-dataset, use the same fine-tuned model on every dataset. "
            "Pass an exact model id, or 'best' to choose the best complete-coverage model."
        ),
    )
    parser.add_argument(
        "--gain-components",
        action="store_true",
        help=(
            "With --best-per-dataset, plot [F1_FT(data)-F1_FT(names)] and "
            "[F1_base(data)-F1_base(names)] as side-by-side bars."
        ),
    )
    parser.add_argument(
        "--min-graph-coverage",
        type=int,
        default=1,
        help="Drop checkpoints with fewer complete graph contrasts than this.",
    )
    parser.add_argument("--formats", nargs="*", default=["pdf", "png"], choices=["pdf", "png", "svg"])
    return parser.parse_args()


def condition_from_config(config: str) -> str | None:
    if config.startswith("responses_names_only"):
        return "names_only"
    if "_summary" in config:
        return "summary"
    if "_matrix" in config:
        return "matrix"
    return None


def names_allowed(config: str, args: argparse.Namespace) -> bool:
    if condition_from_config(config) != "names_only":
        return False
    if args.prompt_style != "all" and config_prompt_style(config) != args.prompt_style:
        return False
    return True


def data_allowed(config: str, args: argparse.Namespace) -> bool:
    condition = condition_from_config(config)
    if condition not in {"summary", "matrix"}:
        return False
    if args.data_form != "best" and condition != args.data_form:
        return False
    if args.prompt_style != "all" and config_prompt_style(config) != args.prompt_style:
        return False
    is_anon = "_anon_" in config
    if args.anonymization == "real" and is_anon:
        return False
    if args.anonymization == "anon" and not is_anon:
        return False
    if not budget_matches(config, args.obs, args.inter):
        return False
    return True


def score_file(model: str, graph: str, slot: str, config: str, path: Path, metric: str) -> SelectedScore | None:
    values, valid_rows, total_rows = end_to_end_values(path, metric)
    if not values:
        return None
    return SelectedScore(
        model=model,
        graph=graph,
        slot=slot,
        data_form=condition_from_config(config) or "",
        value=mean(values),
        valid_rows=valid_rows,
        total_rows=total_rows,
        config=config,
        path=path,
    )


def choose_best(scores: list[SelectedScore]) -> SelectedScore | None:
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


def collect_selected(args: argparse.Namespace, graphs: list[str]) -> dict[tuple[str, str, str], SelectedScore]:
    candidates: dict[tuple[str, str, str], list[SelectedScore]] = defaultdict(list)
    requested_models = {canonical_model_id(model) for model in args.models or []}
    for path in args.responses_dir.glob("*/*.csv.per_row.csv"):
        parsed = parse_response_file(path, args.responses_dir)
        if parsed is None or parsed.graph not in graphs:
            continue
        model = canonical_model_id(parsed.model)

        if model == BASE_MODEL:
            pass
        elif requested_models:
            if model not in requested_models:
                continue
        elif not is_qwen3_4b_finetune(parsed.model):
            continue

        if names_allowed(parsed.config, args):
            score = score_file(model, parsed.graph, "names_only", parsed.config, parsed.path, args.metric)
        elif data_allowed(parsed.config, args):
            score = score_file(model, parsed.graph, "data", parsed.config, parsed.path, args.metric)
        else:
            score = None
        if score is not None:
            candidates[(model, parsed.graph, score.slot)].append(score)

    selected: dict[tuple[str, str, str], SelectedScore] = {}
    for key, scores in candidates.items():
        best = choose_best(scores)
        if best is not None:
            selected[key] = best
    return selected


def build_details(
    selected: dict[tuple[str, str, str], SelectedScore],
    graphs: list[str],
) -> list[DetailRow]:
    models = sorted({model for model, _, _ in selected if model != BASE_MODEL})
    details: list[DetailRow] = []
    for model in models:
        for graph in graphs:
            base_names = selected.get((BASE_MODEL, graph, "names_only"))
            base_data = selected.get((BASE_MODEL, graph, "data"))
            ft_names = selected.get((model, graph, "names_only"))
            ft_data = selected.get((model, graph, "data"))
            if base_names is None or base_data is None or ft_names is None or ft_data is None:
                continue
            base_gain = base_data.value - base_names.value
            ft_gain = ft_data.value - ft_names.value
            details.append(
                DetailRow(
                    model=model,
                    graph=graph,
                    base_names=base_names,
                    base_data=base_data,
                    ft_names=ft_names,
                    ft_data=ft_data,
                    base_gain=base_gain,
                    ft_gain=ft_gain,
                    delta_gain=ft_gain - base_gain,
                )
            )
    return details


def summarize(details: list[DetailRow], graphs: list[str], min_graph_coverage: int) -> list[GainRow]:
    by_model: dict[str, list[DetailRow]] = defaultdict(list)
    for row in details:
        by_model[row.model].append(row)

    rows: list[GainRow] = []
    for model, model_rows in by_model.items():
        if len(model_rows) < min_graph_coverage:
            continue
        deltas = [row.delta_gain for row in model_rows]
        ft_gains = [row.ft_gain for row in model_rows]
        base_gains = [row.base_gain for row in model_rows]
        valid_rows = sum(
            row.base_names.valid_rows + row.base_data.valid_rows + row.ft_names.valid_rows + row.ft_data.valid_rows
            for row in model_rows
        )
        total_rows = sum(
            row.base_names.total_rows + row.base_data.total_rows + row.ft_names.total_rows + row.ft_data.total_rows
            for row in model_rows
        )
        rows.append(
            GainRow(
                model=model,
                value=mean(deltas),
                ci95=1.96 * stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0,
                graph_coverage=len(deltas),
                graph_total=len(graphs),
                positive_graphs=sum(delta > 0 for delta in deltas),
                valid_rows=valid_rows,
                total_rows=total_rows,
                ft_gain=mean(ft_gains),
                base_gain=mean(base_gains),
            )
        )
    return rows


def order_models(rows: list[GainRow], top_models: int) -> list[str]:
    ordered = sorted(rows, key=lambda row: (row.value, row.graph_coverage, row.positive_graphs, row.model), reverse=True)
    if top_models > 0:
        ordered = ordered[:top_models]
    return [row.model for row in ordered]


def budget_slug(args: argparse.Namespace) -> str:
    obs = "all" if args.obs is None else str(args.obs)
    inter = "all" if args.inter is None else str(args.inter)
    return f"obs{obs}_int{inter}"


def budget_label(args: argparse.Namespace) -> str:
    obs = "*" if args.obs is None else str(args.obs)
    inter = "*" if args.inter is None else str(args.inter)
    if args.obs is None or args.inter is None:
        return f"best over N={obs}, M={inter}"
    return f"N={obs}, M={inter}"


def output_stem(args: argparse.Namespace) -> str:
    top = "" if args.top_models <= 0 else f"_top{args.top_models}"
    prompt = "" if args.prompt_style == "all" else f"_{args.prompt_style}"
    per_dataset = "_best_per_dataset" if args.best_per_dataset else ""
    quantity = "components" if args.gain_components else "delta"
    shared = ""
    if args.shared_ft_model:
        shared_id = "best" if args.shared_ft_model == "best" else safe_slug(args.shared_ft_model)
        shared = f"_sharedft_{shared_id}"
    return (
        f"qwen3_data_use_gain_{quantity}{per_dataset}_{args.metric}_{args.anonymization}_{args.data_form}_"
        f"{budget_slug(args)}{prompt}{top}{shared}"
    )


def safe_slug(text: str) -> str:
    keep = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("-")
    return "".join(keep).strip("-")


def detail_valid_rate(row: DetailRow) -> float:
    valid_rows = row.base_names.valid_rows + row.base_data.valid_rows + row.ft_names.valid_rows + row.ft_data.valid_rows
    total_rows = row.base_names.total_rows + row.base_data.total_rows + row.ft_names.total_rows + row.ft_data.total_rows
    return valid_rows / total_rows if total_rows else math.nan


def pair_valid_rate(first: SelectedScore, second: SelectedScore) -> float:
    total_rows = first.total_rows + second.total_rows
    valid_rows = first.valid_rows + second.valid_rows
    return valid_rows / total_rows if total_rows else math.nan


def best_details_by_graph(details: list[DetailRow], graphs: list[str]) -> list[DetailRow]:
    by_graph: dict[str, list[DetailRow]] = defaultdict(list)
    for row in details:
        by_graph[row.graph].append(row)
    best_rows: list[DetailRow] = []
    for graph in graphs:
        candidates = by_graph.get(graph, [])
        if not candidates:
            continue
        best_rows.append(
            sorted(
                candidates,
                key=lambda row: (
                    row.delta_gain,
                    row.ft_gain,
                    detail_valid_rate(row) if math.isfinite(detail_valid_rate(row)) else -math.inf,
                    row.model,
                ),
                reverse=True,
            )[0]
        )
    return best_rows


def shared_model_details(details: list[DetailRow], graphs: list[str], requested_model: str) -> tuple[str, list[DetailRow]]:
    by_model_graph: dict[tuple[str, str], DetailRow] = {(row.model, row.graph): row for row in details}
    if requested_model != "best":
        model = canonical_model_id(requested_model)
        rows = [by_model_graph[(model, graph)] for graph in graphs if (model, graph) in by_model_graph]
        if not rows:
            raise SystemExit(f"No complete contrasts found for shared FT model: {model}")
        return model, rows

    summaries = summarize(details, graphs, min_graph_coverage=len(graphs))
    if not summaries:
        print("[warn] No fine-tuned model has complete coverage on all requested graphs; using highest-coverage model.")
        summaries = summarize(details, graphs, min_graph_coverage=1)
    if not summaries:
        raise SystemExit("No complete base/fine-tuned names-only/data contrasts found.")
    model = order_models(summaries, top_models=1)[0]
    rows = [by_model_graph[(model, graph)] for graph in graphs if (model, graph) in by_model_graph]
    return model, rows


def dataset_details(details: list[DetailRow], args: argparse.Namespace, graphs: list[str]) -> tuple[str | None, list[DetailRow]]:
    if args.shared_ft_model:
        return shared_model_details(details, graphs, args.shared_ft_model)
    return None, best_details_by_graph(details, graphs)


def short_model_label(model: str) -> str:
    label = model_display_name(model)
    replacements = [
        ("GRPO CD no-cancer", "GRPO no-cancer"),
        ("GRPO SFT mix v2", "GRPO mix v2"),
        ("checkpoint", "ckpt"),
        (" merged", ""),
    ]
    for old, new in replacements:
        label = label.replace(old, new)
    return " ".join(label.split())


def plot_best_per_dataset(details: list[DetailRow], args: argparse.Namespace, graphs: list[str]) -> None:
    shared_model, best_rows = dataset_details(details, args, graphs)
    if not best_rows:
        raise SystemExit("No complete per-dataset contrasts found.")
    configure_style()
    values = [row.delta_gain for row in best_rows]
    colors = ["#59A14F" if value >= 0 else "#E15759" for value in values]
    x = np.arange(len(best_rows))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#333333", linewidth=0.9)

    bars = ax.bar(x, values, width=0.48, color=colors, edgecolor="#333333", linewidth=0.75, zorder=3)
    any_low_valid = False
    for bar, row in zip(bars, best_rows):
        if detail_valid_rate(row) < LOW_VALID_THRESHOLD:
            bar.set_hatch("///")
            any_low_valid = True

    y_min = min([0.0] + values)
    y_max = max([0.0] + values)
    pad = max(0.08, 0.16 * (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)

    for xpos, value, row in zip(x, values, best_rows):
        label = f"{value:+.2f}\n{short_model_label(row.model)}"
        y = value + 0.025 if value >= 0 else value - 0.025
        ax.text(
            xpos,
            y,
            label,
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=7.1,
            linespacing=0.92,
            color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([GRAPH_LABELS.get(row.graph, row.graph.title()) for row in best_rows])
    ax.set_ylabel("Additional data-use gain vs base")
    prompt = "" if args.prompt_style == "all" else f", {prompt_style_label(args.prompt_style)} prompts"
    selection = (
        f"shared FT: {short_model_label(shared_model)}"
        if shared_model is not None
        else "best FT selected per dataset"
    )
    ax.set_title(
        f"Fine-tuned data-use gain per dataset ({args.anonymization}, {args.data_form} data, {budget_label(args)}{prompt})",
        fontsize=10.5,
        pad=8,
    )
    fig.text(
        0.02,
        0.02,
        f"{selection}. Bar = [FT data gain - base data gain].",
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#333333",
    )
    handles = [
        Patch(facecolor="#59A14F", edgecolor="#333333", label="FT gain > base gain"),
        Patch(facecolor="#E15759", edgecolor="#333333", label="FT gain < base gain"),
    ]
    if any_low_valid:
        handles.append(Patch(facecolor="white", edgecolor="#333333", hatch="///", label="<60% parsed rows"))
    ax.legend(handles=handles, ncol=min(3, len(handles)), loc="upper center", bbox_to_anchor=(0.5, 1.20), frameon=False)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.78, bottom=0.24)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args)
    for fmt in args.formats:
        out_path = args.out_dir / f"{stem}.{fmt}"
        fig.savefig(out_path)
        print(f"[write] {out_path}")
    plt.close(fig)


def plot_best_per_dataset_components(details: list[DetailRow], args: argparse.Namespace, graphs: list[str]) -> None:
    shared_model, best_rows = dataset_details(details, args, graphs)
    if not best_rows:
        raise SystemExit("No complete per-dataset contrasts found.")
    configure_style()

    x = np.arange(len(best_rows))
    width = 0.30
    base_values = [row.base_gain for row in best_rows]
    ft_values = [row.ft_gain for row in best_rows]
    all_values = base_values + ft_values + [0.0]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#333333", linewidth=0.9)

    base_bars = ax.bar(
        x - width / 2,
        base_values,
        width=width,
        color="#7B5EA7",
        edgecolor="#333333",
        linewidth=0.75,
        label="Base data gain",
        zorder=3,
    )
    ft_bars = ax.bar(
        x + width / 2,
        ft_values,
        width=width,
        color="#59A14F",
        edgecolor="#333333",
        linewidth=0.75,
        label="Fine-tuned data gain",
        zorder=3,
    )

    any_low_valid = False
    for bar, row in zip(base_bars, best_rows):
        if pair_valid_rate(row.base_names, row.base_data) < LOW_VALID_THRESHOLD:
            bar.set_hatch("///")
            any_low_valid = True
    for bar, row in zip(ft_bars, best_rows):
        if pair_valid_rate(row.ft_names, row.ft_data) < LOW_VALID_THRESHOLD:
            bar.set_hatch("///")
            any_low_valid = True

    y_min = min(all_values)
    y_max = max(all_values)
    pad = max(0.10, 0.18 * (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)

    for xpos, value in zip(x - width / 2, base_values):
        y = value + 0.025 if value >= 0 else value - 0.025
        ax.text(
            xpos,
            y,
            f"{value:+.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=7.4,
            color="#222222",
        )
    for xpos, value, row in zip(x + width / 2, ft_values, best_rows):
        y = value + 0.025 if value >= 0 else value - 0.025
        label = f"{value:+.2f}" if shared_model is not None else f"{value:+.2f}\n{short_model_label(row.model)}"
        ax.text(
            xpos,
            y,
            label,
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=7.1,
            linespacing=0.92,
            color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([GRAPH_LABELS.get(row.graph, row.graph.title()) for row in best_rows])
    ax.set_ylabel("Data gain over names-only F1")
    prompt = "" if args.prompt_style == "all" else f", {prompt_style_label(args.prompt_style)} prompts"
    selection = (
        f"shared FT: {short_model_label(shared_model)}"
        if shared_model is not None
        else "FT method selected per dataset"
    )
    ax.set_title(
        f"Base vs fine-tuned data gain per dataset ({args.anonymization}, {args.data_form} data, {budget_label(args)}{prompt})",
        fontsize=10.5,
        pad=8,
    )
    fig.text(
        0.02,
        0.02,
        "Base bar = F1_base(data) - F1_base(names). "
        f"FT bar = F1_FT(data) - F1_FT(names); {selection}.",
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#333333",
    )

    handles, labels = ax.get_legend_handles_labels()
    if any_low_valid:
        handles.append(Patch(facecolor="white", edgecolor="#333333", hatch="///", label="<60% parsed rows"))
        labels.append("<60% parsed rows")
    ax.legend(handles, labels, ncol=min(3, len(handles)), loc="upper center", bbox_to_anchor=(0.5, 1.20), frameon=False)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.78, bottom=0.25)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args)
    for fmt in args.formats:
        out_path = args.out_dir / f"{stem}.{fmt}"
        fig.savefig(out_path)
        print(f"[write] {out_path}")
    plt.close(fig)


def plot(rows: list[GainRow], args: argparse.Namespace, graphs: list[str]) -> None:
    configure_style()
    row_by_model = {row.model: row for row in rows}
    models = order_models(rows, args.top_models)
    values = [row_by_model[model].value for model in models]
    errors = [row_by_model[model].ci95 for model in models]
    colors = ["#59A14F" if value >= 0 else "#E15759" for value in values]
    x = np.arange(len(models))
    fig_width = max(7.2, 0.56 * len(models) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 4.55))
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#333333", linewidth=0.9)

    bars = ax.bar(x, values, width=0.58, color=colors, edgecolor="#333333", linewidth=0.75, zorder=3)
    any_partial = False
    any_low_valid = False
    for bar, model in zip(bars, models):
        row = row_by_model[model]
        if row.graph_coverage < row.graph_total:
            bar.set_alpha(0.58)
            any_partial = True
        if row.valid_rate < LOW_VALID_THRESHOLD:
            bar.set_hatch("///")
            any_low_valid = True

    show_err = np.asarray(errors) > 1e-4
    if show_err.any():
        ax.errorbar(
            x[show_err],
            np.asarray(values)[show_err],
            yerr=np.asarray(errors)[show_err],
            fmt="none",
            ecolor="#333333",
            elinewidth=0.75,
            capsize=2.4,
            zorder=4,
        )

    show_labels = len(models) <= 12
    if show_labels:
        for xpos, value, err, model in zip(x, values, errors, models):
            row = row_by_model[model]
            label = f"{value:+.2f}"
            if row.graph_coverage < row.graph_total:
                label = f"{label}\n{row.graph_coverage}/{row.graph_total}"
            y = value + err + 0.018 if value >= 0 else value - err - 0.018
            ax.text(
                xpos,
                y,
                label,
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7,
                linespacing=0.85,
                color="#222222",
            )

    y_min = min([0.0] + [value - err for value, err in zip(values, errors)])
    y_max = max([0.0] + [value + err for value, err in zip(values, errors)])
    pad = max(0.08, 0.16 * (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xticks(x)
    ax.set_xticklabels([model_display_name(model) for model in models], rotation=35, ha="right")
    ax.set_ylabel("Additional data-use gain vs base")
    prompt = "" if args.prompt_style == "all" else f", {prompt_style_label(args.prompt_style)} prompts"
    ax.set_title(
        f"Fine-tuning data-use gain ({args.anonymization}, {args.data_form} data, {budget_label(args)}{prompt})",
        fontsize=10.8,
        pad=8,
    )
    fig.text(
        0.02,
        0.02,
        "Bar = [F1_FT(data) - F1_FT(names)] - [F1_base(data) - F1_base(names)]. "
        "Positive means the fine-tuned model benefits more from data.",
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#333333",
    )

    handles = [
        Patch(facecolor="#59A14F", edgecolor="#333333", label="FT gain > base gain"),
        Patch(facecolor="#E15759", edgecolor="#333333", label="FT gain < base gain"),
    ]
    if any_low_valid:
        handles.append(Patch(facecolor="white", edgecolor="#333333", hatch="///", label="<60% parsed rows"))
    if any_partial:
        handles.append(Patch(facecolor="#777777", edgecolor="#333333", alpha=0.45, label="Partial graph coverage"))
    ax.legend(
        handles=handles,
        ncol=1,
        loc="upper right",
        frameon=False,
        fontsize=7.6,
        handlelength=1.7,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.13, right=0.99, top=0.87, bottom=0.42)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args)
    for fmt in args.formats:
        out_path = args.out_dir / f"{stem}.{fmt}"
        fig.savefig(out_path)
        print(f"[write] {out_path}")
    plt.close(fig)


def write_csvs(rows: list[GainRow], details: list[DetailRow], args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args)
    plotted_models = set(order_models(rows, args.top_models))
    summary_path = args.out_dir / f"{stem}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "model_label",
                "delta_data_use_gain",
                "ci95",
                "ft_data_gain",
                "base_data_gain",
                "graph_coverage",
                "graph_total",
                "positive_graphs",
                "valid_rows",
                "total_rows",
                "valid_rate",
            ],
        )
        writer.writeheader()
        for row in rows:
            if row.model not in plotted_models:
                continue
            writer.writerow(
                {
                    "model": row.model,
                    "model_label": model_display_name(row.model),
                    "delta_data_use_gain": f"{row.value:.8g}",
                    "ci95": f"{row.ci95:.8g}",
                    "ft_data_gain": f"{row.ft_gain:.8g}",
                    "base_data_gain": f"{row.base_gain:.8g}",
                    "graph_coverage": row.graph_coverage,
                    "graph_total": row.graph_total,
                    "positive_graphs": row.positive_graphs,
                    "valid_rows": row.valid_rows,
                    "total_rows": row.total_rows,
                    "valid_rate": "" if not math.isfinite(row.valid_rate) else f"{row.valid_rate:.8g}",
                }
            )
    print(f"[write] {summary_path}")

    detail_path = args.out_dir / f"{stem}.selected.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "model_label",
                "graph",
                "delta_data_use_gain",
                "ft_data_gain",
                "base_data_gain",
                "base_names_f1",
                "base_data_f1",
                "ft_names_f1",
                "ft_data_f1",
                "base_names_config",
                "base_data_config",
                "ft_names_config",
                "ft_data_config",
                "base_names_path",
                "base_data_path",
                "ft_names_path",
                "ft_data_path",
            ],
        )
        writer.writeheader()
        for row in sorted(details, key=lambda item: (item.model, item.graph)):
            if row.model not in plotted_models:
                continue
            writer.writerow(
                {
                    "model": row.model,
                    "model_label": model_display_name(row.model),
                    "graph": row.graph,
                    "delta_data_use_gain": f"{row.delta_gain:.8g}",
                    "ft_data_gain": f"{row.ft_gain:.8g}",
                    "base_data_gain": f"{row.base_gain:.8g}",
                    "base_names_f1": f"{row.base_names.value:.8g}",
                    "base_data_f1": f"{row.base_data.value:.8g}",
                    "ft_names_f1": f"{row.ft_names.value:.8g}",
                    "ft_data_f1": f"{row.ft_data.value:.8g}",
                    "base_names_config": row.base_names.config,
                    "base_data_config": row.base_data.config,
                    "ft_names_config": row.ft_names.config,
                    "ft_data_config": row.ft_data.config,
                    "base_names_path": row.base_names.path,
                    "base_data_path": row.base_data.path,
                    "ft_names_path": row.ft_names.path,
                    "ft_data_path": row.ft_data.path,
                }
            )
    print(f"[write] {detail_path}")


def write_best_per_dataset_csv(details: list[DetailRow], args: argparse.Namespace, graphs: list[str]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shared_model, best_rows = dataset_details(details, args, graphs)
    out_path = args.out_dir / f"{output_stem(args)}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "selection_mode",
                "shared_ft_model",
                "graph",
                "best_model",
                "best_model_label",
                "delta_data_use_gain",
                "ft_data_gain",
                "base_data_gain",
                "base_names_f1",
                "base_data_f1",
                "ft_names_f1",
                "ft_data_f1",
                "valid_rate",
                "base_names_config",
                "base_data_config",
                "ft_names_config",
                "ft_data_config",
                "base_names_path",
                "base_data_path",
                "ft_names_path",
                "ft_data_path",
            ],
        )
        writer.writeheader()
        for row in best_rows:
            writer.writerow(
                {
                    "selection_mode": "shared_ft_model" if shared_model is not None else "best_per_dataset",
                    "shared_ft_model": "" if shared_model is None else shared_model,
                    "graph": row.graph,
                    "best_model": row.model,
                    "best_model_label": model_display_name(row.model),
                    "delta_data_use_gain": f"{row.delta_gain:.8g}",
                    "ft_data_gain": f"{row.ft_gain:.8g}",
                    "base_data_gain": f"{row.base_gain:.8g}",
                    "base_names_f1": f"{row.base_names.value:.8g}",
                    "base_data_f1": f"{row.base_data.value:.8g}",
                    "ft_names_f1": f"{row.ft_names.value:.8g}",
                    "ft_data_f1": f"{row.ft_data.value:.8g}",
                    "valid_rate": (
                        "" if not math.isfinite(detail_valid_rate(row)) else f"{detail_valid_rate(row):.8g}"
                    ),
                    "base_names_config": row.base_names.config,
                    "base_data_config": row.base_data.config,
                    "ft_names_config": row.ft_names.config,
                    "ft_data_config": row.ft_data.config,
                    "base_names_path": row.base_names.path,
                    "base_data_path": row.base_data.path,
                    "ft_names_path": row.ft_names.path,
                    "ft_data_path": row.ft_data.path,
                }
            )
    print(f"[write] {out_path}")


def main() -> None:
    args = parse_args()
    graphs = resolve_graphs(args.graphs)
    selected = collect_selected(args, graphs)
    details = build_details(selected, graphs)
    if args.best_per_dataset:
        if args.gain_components:
            plot_best_per_dataset_components(details, args, graphs)
        else:
            plot_best_per_dataset(details, args, graphs)
        write_best_per_dataset_csv(details, args, graphs)
        return

    rows = summarize(details, graphs, args.min_graph_coverage)
    if not rows:
        raise SystemExit("No complete base/fine-tuned names-only/data contrasts found.")
    plot(rows, args, graphs)
    write_csvs(rows, details, args)


if __name__ == "__main__":
    main()
