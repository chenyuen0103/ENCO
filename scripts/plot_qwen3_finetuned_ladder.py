#!/usr/bin/env python3
"""Plot an evidence ladder for Qwen3-4B fine-tuned checkpoints.

The ladder searches the best available names-only, summary, and matrix
configuration for each fine-tuned model. Invalid/unparseable responses count as
zero F1, matching the end-to-end scoring used in the post-training figures.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
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
import pandas as pd
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSES_DIR = REPO_ROOT / "scripts" / "responses"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "out" / "qwen3_posttraining_maintext"
BASE_MODEL = "Qwen3-4B-Thinking-2507"
BASE_MODEL_ALIASES = {BASE_MODEL, "Qwen3-4B", "Qwen/Qwen3-4B-Thinking-2507"}
GRAPH_ORDER = ["cancer", "earthquake", "asia", "sachs"]
GRAPH_LABELS = {
    "cancer": "Cancer",
    "earthquake": "Earthquake",
    "asia": "Asia",
    "sachs": "Sachs",
}
CONDITION_ORDER_REAL = ["names_only", "summary", "matrix"]
CONDITION_ORDER_ANON = ["summary", "matrix"]
CONDITION_LABELS = {
    "names_only": "Names only",
    "summary": "Summary",
    "matrix": "Matrix",
}
CONDITION_COLORS = {
    "names_only": "#7B5EA7",
    "summary": "#59A14F",
    "matrix": "#4E79A7",
}
LOW_VALID_THRESHOLD = 0.60
BUDGET_RE = re.compile(r"^responses_obs(?P<obs>\d+)_int(?P<inter>\d+)_")
BUDGET_ANY = {"*", "all", "any"}
PARSE_MARKERS = ("_matrix_", "_summary_", "_colRandom_")


@dataclass(frozen=True)
class ParsedFile:
    graph: str
    config: str
    model: str
    path: Path


@dataclass(frozen=True)
class FileScore:
    graph: str
    model: str
    condition: str
    config: str
    value: float
    valid_rows: int
    total_rows: int
    path: Path

    @property
    def valid_rate(self) -> float:
        return self.valid_rows / self.total_rows if self.total_rows else math.nan


@dataclass(frozen=True)
class SummaryScore:
    model: str
    condition: str
    value: float
    ci95: float
    graph_coverage: int
    graph_total: int
    valid_rows: int
    total_rows: int

    @property
    def valid_rate(self) -> float:
        return self.valid_rows / self.total_rows if self.total_rows else math.nan


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-dir", type=Path, default=DEFAULT_RESPONSES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--best-configs-csv",
        type=Path,
        default=None,
        help="Optional best_prompt_configs_by_valid_rate.csv to use instead of scanning per-row response files.",
    )
    parser.add_argument("--graphs", nargs="*", default=GRAPH_ORDER, help="Graphs to average, or 'all'.")
    parser.add_argument("--obs", type=budget_arg, default="all")
    parser.add_argument("--inter", type=budget_arg, default="all")
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--anonymization", choices=["real", "anon", "all"], default="real")
    parser.add_argument(
        "--prompt-style",
        choices=["all", "plain", "wrapchat", "fmthint", "wrapchat-fmthint"],
        default="all",
    )
    parser.add_argument("--models", nargs="*", default=None, help="Optional exact model ids to include.")
    parser.add_argument("--include-base", action="store_true", help="Include the base Qwen3-4B model in the ladder.")
    parser.add_argument("--top-models", type=int, default=0, help="Keep only the top N models by best ladder score.")
    parser.add_argument("--formats", nargs="*", default=["pdf", "png"], choices=["pdf", "png", "svg"])
    return parser.parse_args()


def resolve_graphs(raw_graphs: list[str]) -> list[str]:
    if len(raw_graphs) == 1 and raw_graphs[0].lower() == "all":
        return GRAPH_ORDER
    graphs = [graph.lower() for graph in raw_graphs]
    unknown = [graph for graph in graphs if graph not in GRAPH_ORDER]
    if unknown:
        raise SystemExit(f"Unknown graph(s): {', '.join(unknown)}. Choose from {', '.join(GRAPH_ORDER)}.")
    return graphs


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 9.5,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
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


def end_to_end_values(path: Path, metric: str) -> tuple[list[float], int, int]:
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


def parse_response_file(path: Path, responses_dir: Path) -> ParsedFile | None:
    suffix = ".csv.per_row.csv"
    if not path.name.endswith(suffix):
        return None
    try:
        rel = path.relative_to(responses_dir)
    except ValueError:
        return None
    if len(rel.parts) < 2:
        return None
    stem = path.name[: -len(suffix)]
    marker_matches = [(stem.rfind(marker), marker) for marker in PARSE_MARKERS]
    marker_matches = [(idx, marker) for idx, marker in marker_matches if idx >= 0]
    if not marker_matches:
        return None
    idx, marker = max(marker_matches, key=lambda item: item[0])
    config = stem[: idx + len(marker) - 1]
    model = stem[idx + len(marker) :]
    if not model:
        return None
    return ParsedFile(graph=rel.parts[0], config=config, model=model, path=path)


def is_qwen3_4b_finetune(model: str) -> bool:
    lower = model.lower()
    if model in BASE_MODEL_ALIASES:
        return False
    return "qwen3_4b" in lower or "grpo_from_qwen3_4b" in lower


def canonical_model_id(model: str) -> str:
    if model in BASE_MODEL_ALIASES:
        return BASE_MODEL
    return model


def budget_from_config(config: str) -> tuple[int, int] | None:
    match = BUDGET_RE.match(config)
    if not match:
        return None
    return int(match.group("obs")), int(match.group("inter"))


def budget_matches(config: str, obs: int | None, inter: int | None) -> bool:
    if config.startswith("responses_names_only"):
        return True
    budget = budget_from_config(config)
    if budget is None:
        return False
    config_obs, config_inter = budget
    if obs is not None and config_obs != obs:
        return False
    if inter is not None and config_inter != inter:
        return False
    return True


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


def condition_from_config(config: str) -> str | None:
    if config.startswith("responses_names_only"):
        return "names_only"
    if "_summary" in config:
        return "summary"
    if "_matrix" in config:
        return "matrix"
    return None


def config_allowed(config: str, condition: str, args: argparse.Namespace) -> bool:
    found_condition = condition_from_config(config)
    if found_condition != condition:
        return False
    if args.prompt_style != "all" and config_prompt_style(config) != args.prompt_style:
        return False
    if condition != "names_only":
        is_anon = "_anon_" in config
        if args.anonymization == "real" and is_anon:
            return False
        if args.anonymization == "anon" and not is_anon:
            return False
        if not budget_matches(config, args.obs, args.inter):
            return False
    return True


def score_file(parsed: ParsedFile, condition: str, metric: str) -> FileScore | None:
    values, valid_rows, total_rows = end_to_end_values(parsed.path, metric)
    if not values:
        return None
    return FileScore(
        graph=parsed.graph,
        model=canonical_model_id(parsed.model),
        condition=condition,
        config=parsed.config,
        value=mean(values),
        valid_rows=valid_rows,
        total_rows=total_rows,
        path=parsed.path,
    )


def choose_best(scores: list[FileScore]) -> FileScore | None:
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


def collect_best_files(args: argparse.Namespace, graphs: list[str], conditions: list[str]) -> list[FileScore]:
    candidates: dict[tuple[str, str, str], list[FileScore]] = defaultdict(list)
    requested_models = {canonical_model_id(model) for model in args.models or []}
    for path in args.responses_dir.glob("*/*.csv.per_row.csv"):
        parsed = parse_response_file(path, args.responses_dir)
        if parsed is None or parsed.graph not in graphs:
            continue
        model = canonical_model_id(parsed.model)
        if requested_models and model not in requested_models:
            continue
        if model == BASE_MODEL:
            if not args.include_base and not requested_models:
                continue
        elif not is_qwen3_4b_finetune(parsed.model):
            continue
        for condition in conditions:
            if config_allowed(parsed.config, condition, args):
                score = score_file(parsed, condition, args.metric)
                if score is not None:
                    candidates[(model, condition, parsed.graph)].append(score)
                break

    selected: list[FileScore] = []
    for key in sorted(candidates):
        best = choose_best(candidates[key])
        if best is not None:
            selected.append(best)
    return selected


def _best_config_path(path: str | float | None, response_basename: str) -> Path:
    if path is not None and not (isinstance(path, float) and math.isnan(path)) and str(path):
        return Path(str(path))
    return Path(str(response_basename))


def collect_best_config_rows(args: argparse.Namespace, graphs: list[str], conditions: list[str]) -> list[FileScore]:
    if args.best_configs_csv is None:
        return []
    csv_path = args.best_configs_csv if args.best_configs_csv.is_absolute() else REPO_ROOT / args.best_configs_csv
    df = pd.read_csv(csv_path)
    requested_models = {canonical_model_id(model) for model in args.models or []}
    candidates: dict[tuple[str, str, str], list[FileScore]] = defaultdict(list)

    for _, row in df.iterrows():
        graph = str(row.get("graph", "")).lower()
        if graph not in graphs:
            continue
        condition = str(row.get("prompt_style", ""))
        if condition not in conditions:
            continue
        model = canonical_model_id(str(row.get("model_raw") or row.get("model") or ""))
        if requested_models and model not in requested_models:
            continue
        if model == BASE_MODEL:
            if not args.include_base and not requested_models:
                continue
        elif not is_qwen3_4b_finetune(model):
            continue
        if args.prompt_style != "all":
            wrapper = str(row.get("wrapper_mode", "") or "")
            hint = int(float(row.get("append_format_hint", 0) or 0))
            found_prompt_style = "wrapchat-fmthint" if wrapper == "chat" and hint else "wrapchat" if wrapper == "chat" else "fmthint" if hint else "plain"
            if found_prompt_style != args.prompt_style:
                continue
        if condition != "names_only":
            anonymize = int(float(row.get("anonymize", 0) or 0))
            if args.anonymization == "real" and anonymize:
                continue
            if args.anonymization == "anon" and not anonymize:
                continue
            obs = int(float(row.get("obs_n", 0)))
            inter = int(float(row.get("int_n", 0)))
            if args.obs is not None and obs != args.obs:
                continue
            if args.inter is not None and inter != args.inter:
                continue

        valid_rows = int(float(row.get("valid_rows", 0) or 0))
        total_rows = int(float(row.get("num_rows", 0) or 0))
        valid_rate = row.get("valid_rate", math.nan)
        if not total_rows and math.isfinite(float(valid_rate or math.nan)):
            total_rows = 1
            valid_rows = int(round(float(valid_rate)))
        value = finite_float(str(row.get("avg_f1", "")))
        if value is None:
            value = 0.0
        score = FileScore(
            graph=graph,
            model=model,
            condition=condition,
            config=str(row.get("config", "")),
            value=value,
            valid_rows=valid_rows,
            total_rows=total_rows,
            path=_best_config_path(row.get("response_csv"), str(row.get("response_basename", ""))),
        )
        candidates[(model, condition, graph)].append(score)

    selected: list[FileScore] = []
    for key in sorted(candidates):
        best = choose_best(candidates[key])
        if best is not None:
            selected.append(best)
    return selected


def summarize(selected: list[FileScore], graphs: list[str], conditions: list[str]) -> list[SummaryScore]:
    by_key: dict[tuple[str, str], list[FileScore]] = defaultdict(list)
    for score in selected:
        by_key[(score.model, score.condition)].append(score)

    models = sorted({score.model for score in selected})
    rows: list[SummaryScore] = []
    for model in models:
        for condition in conditions:
            scores = by_key.get((model, condition), [])
            values = [score.value for score in scores if math.isfinite(score.value)]
            valid_rows = sum(score.valid_rows for score in scores)
            total_rows = sum(score.total_rows for score in scores)
            if values:
                value = mean(values)
                ci95 = 1.96 * stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
            else:
                value = math.nan
                ci95 = 0.0
            rows.append(
                SummaryScore(
                    model=model,
                    condition=condition,
                    value=value,
                    ci95=ci95,
                    graph_coverage=len(values),
                    graph_total=len(graphs),
                    valid_rows=valid_rows,
                    total_rows=total_rows,
                )
            )
    return rows


def model_display_name(model: str) -> str:
    if model == BASE_MODEL:
        return "Base Qwen3-4B"
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
    label = model
    for old, new in replacements:
        label = label.replace(old, new)
    return " ".join(label.split())


def metric_label(metric: str) -> str:
    labels = {
        "f1": "End-to-end directed-edge F1",
        "skeleton_f1": "End-to-end skeleton F1",
        "ancestor_f1": "End-to-end ancestor F1",
        "format_ok": "Format-valid rate",
    }
    return labels.get(metric, metric.replace("_", " ").title())


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


def prompt_slug(args: argparse.Namespace) -> str:
    return "" if args.prompt_style == "all" else f"_{args.prompt_style}"


def output_stem(args: argparse.Namespace) -> str:
    top = "" if args.top_models <= 0 else f"_top{args.top_models}"
    source = "_bestconfigs" if args.best_configs_csv is not None else ""
    base = "_withbase" if args.include_base else ""
    return (
        f"qwen3_candidate_evidence_ladder{source}{base}_{args.metric}_{args.anonymization}_"
        f"{budget_slug(args)}{prompt_slug(args)}{top}"
    )


def order_models(rows: list[SummaryScore], top_models: int) -> list[str]:
    by_model: dict[str, list[SummaryScore]] = defaultdict(list)
    for row in rows:
        by_model[row.model].append(row)

    def rank_key(model: str) -> tuple[float, int, str]:
        model_rows = by_model[model]
        values = [row.value for row in model_rows if math.isfinite(row.value)]
        coverage = sum(row.graph_coverage for row in model_rows)
        return (max(values) if values else -math.inf, coverage, model)

    models = sorted(by_model, key=rank_key, reverse=True)
    if top_models > 0:
        models = models[:top_models]
    return models


def plot_ladder(rows: list[SummaryScore], args: argparse.Namespace, graphs: list[str], conditions: list[str]) -> None:
    configure_style()
    models = order_models(rows, args.top_models)
    rows_by_key = {(row.model, row.condition): row for row in rows}
    x = np.arange(len(models))
    width = min(0.24, 0.72 / max(len(conditions), 1))
    fig_width = max(8.0, 0.62 * len(models) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 3.8))
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)

    any_low_valid = False
    any_partial = False
    label_ceiling = 0.0
    show_bar_labels = len(models) <= 10
    for cond_idx, condition in enumerate(conditions):
        offset = (cond_idx - (len(conditions) - 1) / 2) * width
        values: list[float] = []
        errors: list[float] = []
        bars_rows: list[SummaryScore | None] = []
        for model in models:
            row = rows_by_key.get((model, condition))
            bars_rows.append(row)
            values.append(0.0 if row is None or not math.isfinite(row.value) else row.value)
            errors.append(0.0 if row is None or not math.isfinite(row.value) else row.ci95)

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=CONDITION_COLORS[condition],
            edgecolor="#333333",
            linewidth=0.7,
            label=CONDITION_LABELS[condition],
            zorder=3,
        )
        for bar, row in zip(bars, bars_rows):
            if row is None:
                bar.set_alpha(0.22)
                any_partial = True
                continue
            if row.graph_coverage < row.graph_total:
                bar.set_alpha(0.55)
                any_partial = True
            if row.valid_rate < LOW_VALID_THRESHOLD:
                bar.set_hatch("///")
                any_low_valid = True

        show_err = np.asarray(errors) > 1e-4
        if show_err.any():
            ax.errorbar(
                (x + offset)[show_err],
                np.asarray(values)[show_err],
                yerr=np.asarray(errors)[show_err],
                fmt="none",
                ecolor="#333333",
                elinewidth=0.75,
                capsize=2.3,
                zorder=4,
            )

        for xpos, value, err, row in zip(x + offset, values, errors, bars_rows):
            if row is None or not math.isfinite(row.value):
                continue
            label_ceiling = max(label_ceiling, value + max(err, 0.0))
            if not show_bar_labels:
                continue
            text = f"{value:.2f}"
            if row.graph_coverage < row.graph_total:
                text = f"{text}\n{row.graph_coverage}/{row.graph_total}"
            ax.text(
                xpos,
                value + max(err, 0.0) + 0.018,
                text,
                ha="center",
                va="bottom",
                fontsize=6.7,
                linespacing=0.85,
                color="#222222",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([model_display_name(model) for model in models], rotation=35, ha="right")
    ax.set_ylabel(metric_label(args.metric))
    y_upper = max(1.08, label_ceiling + (0.14 if show_bar_labels else 0.06))
    if args.metric in {"f1", "skeleton_f1", "ancestor_f1", "format_ok"}:
        y_upper = min(y_upper, 1.35)
    ax.set_ylim(0, y_upper)
    graph_label = ", ".join(GRAPH_LABELS[graph] for graph in graphs)
    prompt = "" if args.prompt_style == "all" else f", {prompt_style_label(args.prompt_style)} prompts"
    source = "best configs" if args.best_configs_csv is not None else "best per-row files"
    title = f"Qwen3-4B evidence ladder ({source}; {args.anonymization}, {budget_label(args)}{prompt})"
    ax.set_title(title, pad=8, fontsize=11)
    fig.text(
        0.01,
        -0.02,
        f"Mean over selected graphs: {graph_label}. Invalid outputs score 0; dim bars have partial graph coverage.",
        ha="left",
        va="top",
        fontsize=7.4,
    )

    handles, labels = ax.get_legend_handles_labels()
    if any_low_valid:
        handles.append(Patch(facecolor="white", edgecolor="#333333", hatch="///", label="<60% parsed rows"))
        labels.append("<60% parsed rows")
    if any_partial:
        handles.append(Patch(facecolor="#777777", edgecolor="#333333", alpha=0.45, label="Partial graph coverage"))
        labels.append("Partial graph coverage")
    ax.legend(handles, labels, ncol=min(5, len(handles)), loc="upper center", bbox_to_anchor=(0.5, 1.33), frameon=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.70, bottom=0.35)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args)
    for fmt in args.formats:
        out_path = args.out_dir / f"{stem}.{fmt}"
        fig.savefig(out_path)
        print(f"[write] {out_path}")
    plt.close(fig)


def write_csvs(rows: list[SummaryScore], selected: list[FileScore], args: argparse.Namespace) -> None:
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
                "condition",
                "value",
                "ci95",
                "graph_coverage",
                "graph_total",
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
                    "condition": row.condition,
                    "value": "" if not math.isfinite(row.value) else f"{row.value:.8g}",
                    "ci95": f"{row.ci95:.8g}",
                    "graph_coverage": row.graph_coverage,
                    "graph_total": row.graph_total,
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
                "condition",
                "value",
                "valid_rows",
                "total_rows",
                "valid_rate",
                "config",
                "source_path",
            ],
        )
        writer.writeheader()
        for row in sorted(selected, key=lambda item: (item.model, item.condition, item.graph)):
            if row.model not in plotted_models:
                continue
            writer.writerow(
                {
                    "model": row.model,
                    "model_label": model_display_name(row.model),
                    "graph": row.graph,
                    "condition": row.condition,
                    "value": f"{row.value:.8g}",
                    "valid_rows": row.valid_rows,
                    "total_rows": row.total_rows,
                    "valid_rate": "" if not math.isfinite(row.valid_rate) else f"{row.valid_rate:.8g}",
                    "config": row.config,
                    "source_path": row.path,
                }
            )
    print(f"[write] {detail_path}")


def main() -> None:
    args = parse_args()
    graphs = resolve_graphs(args.graphs)
    conditions = CONDITION_ORDER_ANON if args.anonymization == "anon" else CONDITION_ORDER_REAL
    selected = (
        collect_best_config_rows(args, graphs, conditions)
        if args.best_configs_csv is not None
        else collect_best_files(args, graphs, conditions)
    )
    if not selected:
        raise SystemExit("No matching ladder rows found.")
    rows = summarize(selected, graphs, conditions)
    plot_ladder(rows, args, graphs, conditions)
    write_csvs(rows, selected, args)


if __name__ == "__main__":
    main()
