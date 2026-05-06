#!/usr/bin/env python3
"""Plot the Sachs observational-budget Figure 1 for MICAD.

This version fixes the interventional budget at M=0 and sweeps observational
budget N over 0, 1000, 2000, 3000, 4000, and 5000.  The N=0 point is the
names-only semantic baseline for the LLM curves.  Classical observational
baselines are PC and GES.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INPUT = REPO_ROOT / "benchmark_runs" / "sachs_figure1" / "figure1_summary.csv"
DEFAULT_RESPONSE_SUMMARY = REPO_ROOT / "experiments" / "responses" / "sachs" / "sachs_summary.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "benchmark_runs" / "sachs_figure1"
OBS_BUDGETS = [0, 1000, 2000, 3000, 4000, 5000]
CLASSICAL_OBS_BUDGETS = [1000, 2000, 3000, 4000, 5000]
OBS_POS = {budget: index for index, budget in enumerate(OBS_BUDGETS)}
INT_BUDGETS = [0, 50, 100, 200]
CLASSICAL_INT_BUDGETS = [50, 100, 200]
INT_POS = {budget: index for index, budget in enumerate(INT_BUDGETS)}
SEMANTIC_FLOOR_F1 = 0.50


@dataclass(frozen=True)
class SummaryRow:
    line_id: str
    system: str
    naming_regime: str
    obs_n: int
    int_n: int
    n_valid: int
    f1_mean: float
    f1_ci95_halfwidth: float
    shd_mean: float
    shd_ci95_halfwidth: float


def _configure_matplotlib() -> None:
    if not os.getenv("MPLCONFIGDIR"):
        mpl_dir = Path("/tmp") / f"matplotlib_{os.getuid()}"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)


def _read_summary(path: Path) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            n_valid = int(float(raw["n_valid"]))
            shd_sd = float(raw.get("shd_sd") or 0.0)
            rows.append(
                SummaryRow(
                    line_id=str(raw["line_id"]),
                    system=str(raw["system"]),
                    naming_regime=str(raw["naming_regime"]),
                    obs_n=int(float(raw["obs_n"])),
                    int_n=int(float(raw["int_n"])),
                    n_valid=n_valid,
                    f1_mean=float(raw["f1_mean"]),
                    f1_ci95_halfwidth=float(raw.get("f1_ci95_halfwidth") or 0.0),
                    shd_mean=float(raw["shd_mean"]),
                    shd_ci95_halfwidth=1.96 * shd_sd / math.sqrt(max(n_valid, 1)),
                )
            )
    return rows


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _ci95_halfwidth(sd: float | None, n_valid: int) -> float:
    if sd is None or n_valid <= 1:
        return 0.0
    return 1.96 * sd / math.sqrt(n_valid)


def _candidate_rank(row: dict[str, str], *, model: str, kind: str) -> tuple[int, int, str]:
    response_csv = row.get("response_csv", "")
    valid_rows = int(float(row.get("valid_rows") or 0))
    if kind == "names_only":
        exact = f"responses_names_only_p5_{model}.csv"
        if response_csv.endswith(exact):
            return (0, -valid_rows, response_csv)
        if f"responses_names_only_p5_" in response_csv and response_csv.endswith(f"_{model}.csv"):
            return (1, -valid_rows, response_csv)
        return (2, -valid_rows, response_csv)
    if kind == "llm_real":
        exact = f"shuf1_p5_summary_joint_{model}.csv"
        if response_csv.endswith(exact):
            return (0, -valid_rows, response_csv)
        if "shuf1_p5" in response_csv and "summary_joint" in response_csv:
            return (1, -valid_rows, response_csv)
        if "summary_joint" in response_csv:
            return (2, -valid_rows, response_csv)
        return (3, -valid_rows, response_csv)
    if kind == "enco":
        if f"_ENCO_seed" in response_csv:
            return (0, -valid_rows, response_csv)
        return (1, -valid_rows, response_csv)
    return (99, -valid_rows, response_csv)


def _summary_row_from_response(row: dict[str, str], *, line_id: str, system: str, naming_regime: str) -> SummaryRow:
    n_valid = int(float(row.get("valid_rows") or row.get("num_rows") or 0))
    f1_mean = _float_or_none(row.get("avg_f1"))
    shd_mean = _float_or_none(row.get("avg_shd"))
    if f1_mean is None or shd_mean is None or n_valid <= 0:
        raise ValueError("Cannot convert incomplete response-summary row.")
    f1_sd = _float_or_none(row.get("var_f1_sd"))
    shd_sd = _float_or_none(row.get("var_shd_sd"))
    return SummaryRow(
        line_id=line_id,
        system=system,
        naming_regime=naming_regime,
        obs_n=int(float(row.get("obs_n") or 0)),
        int_n=int(float(row.get("int_n") or 0)),
        n_valid=n_valid,
        f1_mean=f1_mean,
        f1_ci95_halfwidth=_ci95_halfwidth(f1_sd, n_valid),
        shd_mean=shd_mean,
        shd_ci95_halfwidth=_ci95_halfwidth(shd_sd, n_valid),
    )


def _read_intervention_summary(path: Path, *, llm_model: str) -> list[SummaryRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))

    rows: list[SummaryRow] = []
    semantic_candidates = [
        row
        for row in raw_rows
        if row.get("model") == llm_model
        and row.get("prompt_style") == "names_only"
        and int(float(row.get("valid_rows") or 0)) > 0
        and _float_or_none(row.get("avg_f1")) is not None
    ]
    if not semantic_candidates:
        raise ValueError(f"Missing names-only semantic row for {llm_model}.")
    semantic = sorted(semantic_candidates, key=lambda row: _candidate_rank(row, model=llm_model, kind="names_only"))[0]
    rows.append(
        _summary_row_from_response(
            semantic,
            line_id="semantic_floor",
            system=llm_model,
            naming_regime="names_only",
        )
    )

    for budget in CLASSICAL_INT_BUDGETS:
        llm_candidates = [
            row
            for row in raw_rows
            if row.get("model") == llm_model
            and row.get("prompt_style") == "summary"
            and row.get("naming_regime") == "real"
            and int(float(row.get("obs_n") or -1)) == 0
            and int(float(row.get("int_n") or -1)) == budget
            and int(float(row.get("valid_rows") or 0)) > 0
            and _float_or_none(row.get("avg_f1")) is not None
        ]
        if llm_candidates:
            rows.append(
                _summary_row_from_response(
                    sorted(llm_candidates, key=lambda row: _candidate_rank(row, model=llm_model, kind="llm_real"))[0],
                    line_id="llm_real",
                    system=llm_model,
                    naming_regime="real",
                )
            )

        enco_candidates = [
            row
            for row in raw_rows
            if row.get("model") == "ENCO"
            and int(float(row.get("obs_n") or -1)) == 0
            and int(float(row.get("int_n") or -1)) == budget
            and int(float(row.get("valid_rows") or 0)) > 0
            and _float_or_none(row.get("avg_f1")) is not None
        ]
        if enco_candidates:
            rows.append(
                _summary_row_from_response(
                    sorted(enco_candidates, key=lambda row: _candidate_rank(row, model=llm_model, kind="enco"))[0],
                    line_id="enco_anchor",
                    system="ENCO",
                    naming_regime="anonymized",
                )
            )
    return rows


def _select(rows: Iterable[SummaryRow], line_id: str, *, int_n: int = 0) -> list[SummaryRow]:
    return sorted(
        (row for row in rows if row.line_id == line_id and row.int_n == int_n),
        key=lambda row: row.obs_n,
    )


def _select_by_obs(rows: Iterable[SummaryRow], line_id: str, *, obs_n: int = 0) -> list[SummaryRow]:
    return sorted(
        (row for row in rows if row.line_id == line_id and row.obs_n == obs_n),
        key=lambda row: row.int_n,
    )


def _by_obs(rows: Iterable[SummaryRow]) -> dict[int, SummaryRow]:
    result: dict[int, SummaryRow] = {}
    for row in rows:
        if row.obs_n in result:
            raise ValueError(f"Duplicate row for {row.line_id} at obs_n={row.obs_n}, int_n={row.int_n}.")
        result[row.obs_n] = row
    return result


def _by_int(rows: Iterable[SummaryRow]) -> dict[int, SummaryRow]:
    result: dict[int, SummaryRow] = {}
    for row in rows:
        if row.int_n in result:
            raise ValueError(f"Duplicate row for {row.line_id} at obs_n={row.obs_n}, int_n={row.int_n}.")
        result[row.int_n] = row
    return result


def _metric(row: SummaryRow, metric: str) -> tuple[float, float]:
    if metric == "f1":
        return row.f1_mean, row.f1_ci95_halfwidth
    if metric == "shd":
        return row.shd_mean, row.shd_ci95_halfwidth
    raise ValueError(f"Unsupported metric: {metric}")


def _metric_config(metric: str, all_rows: list[SummaryRow]) -> dict[str, object]:
    if metric == "f1":
        return {
            "ylabel": "F1",
            "ylim": (-0.035, 1.05),
            "yticks": [0.0, 0.25, 0.50, 0.75, 1.0],
            "yticklabels": ["0.00", "0.25", "0.50", "0.75", "1.00"],
            "semantic_label": "Semantic floor",
        }
    if metric == "shd":
        max_y = max(row.shd_mean for row in all_rows)
        y_max = max(22.0, math.ceil((max_y + 1.0) / 2.0) * 2.0)
        return {
            "ylabel": "SHD",
            "ylim": (0.0, y_max),
            "yticks": list(range(0, int(y_max) + 1, 5)),
            "yticklabels": None,
            "semantic_label": "Semantic baseline",
        }
    raise ValueError(f"Unsupported metric: {metric}")


def _write_formats(fig, out_dir: Path, basename: str, formats: list[str]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        out_path = out_dir / f"{basename}.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        written.append(out_path)
    return written


def _display_model_name(model: str) -> str:
    if model == "gpt-5-mini":
        return "GPT-5 mini"
    if model.lower() == "gpt-5.2-pro":
        return "GPT-5.2-Pro"
    return model


def _model_slug(model: str) -> str:
    return (
        model.lower()
        .replace("/", "-")
        .replace(" ", "-")
        .replace(".", "")
        .replace("_", "-")
    )


def _series_for_obs(
    row_map: dict[int, SummaryRow],
    budgets: list[int],
    *,
    metric: str,
    strict: bool,
    label: str,
) -> tuple[list[int], list[float], list[float]]:
    missing = [budget for budget in budgets if budget not in row_map]
    if missing:
        print(f"[warn] Missing {label} rows for obs budgets {missing} at int_n=0.", file=sys.stderr)
    if strict and missing:
        raise ValueError(f"Missing {label} rows for obs budgets {missing} at int_n=0.")
    xs: list[int] = []
    ys: list[float] = []
    yerr: list[float] = []
    for budget in budgets:
        row = row_map.get(budget)
        if row is None:
            continue
        value, err = _metric(row, metric)
        xs.append(budget)
        ys.append(value)
        yerr.append(err)
    return xs, ys, yerr


def _series_for_int(
    row_map: dict[int, SummaryRow],
    budgets: list[int],
    *,
    metric: str,
    strict: bool,
    label: str,
) -> tuple[list[int], list[float], list[float]]:
    missing = [budget for budget in budgets if budget not in row_map]
    if missing:
        print(f"[warn] Missing {label} rows for interventional budgets {missing} at obs_n=0.", file=sys.stderr)
    if strict and missing:
        raise ValueError(f"Missing {label} rows for interventional budgets {missing} at obs_n=0.")
    xs: list[int] = []
    ys: list[float] = []
    yerr: list[float] = []
    for budget in budgets:
        row = row_map.get(budget)
        if row is None:
            continue
        value, err = _metric(row, metric)
        xs.append(budget)
        ys.append(value)
        yerr.append(err)
    return xs, ys, yerr


def _last_present(row_map: dict[int, SummaryRow], budgets: list[int]) -> SummaryRow:
    for budget in reversed(budgets):
        if budget in row_map:
            return row_map[budget]
    raise ValueError("No rows available for requested budgets.")


def _plot_intervention_metric(
    response_summary_csv: Path,
    out_dir: Path,
    basename: str,
    formats: list[str],
    metric: str,
    *,
    strict: bool,
    llm_model: str,
) -> list[Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FixedFormatter

    rows = _read_intervention_summary(response_summary_csv, llm_model=llm_model)
    semantic = _by_int(_select_by_obs(rows, "semantic_floor"))
    real_data = _by_int(_select_by_obs(rows, "llm_real"))
    enco_data = _by_int(_select_by_obs(rows, "enco_anchor"))

    if 0 not in semantic:
        raise ValueError("Missing semantic_floor row for obs_n=0, int_n=0.")

    real_with_floor = {0: semantic[0], **real_data}
    real_x, real_y, real_err = _series_for_int(
        real_with_floor,
        INT_BUDGETS,
        metric=metric,
        strict=strict,
        label="llm_real/semantic_floor",
    )
    enco_x, enco_y, _ = _series_for_int(
        enco_data,
        CLASSICAL_INT_BUDGETS,
        metric=metric,
        strict=strict,
        label="ENCO",
    )

    if not real_x:
        raise ValueError("No LLM rows available for requested interventional budgets.")
    if not enco_x:
        raise ValueError("No ENCO rows available for requested interventional budgets.")

    llm_label = _display_model_name(_last_present(real_with_floor, INT_BUDGETS).system)
    cfg = _metric_config(metric, rows)

    semantic_value, _ = _metric(semantic[0], metric)
    semantic_plot_value = SEMANTIC_FLOOR_F1 if metric == "f1" else semantic_value
    if metric == "f1":
        real_y = [SEMANTIC_FLOOR_F1 if x == 0 else y for x, y in zip(real_x, real_y)]
        real_err = [0.0 if x == 0 else err for x, err in zip(real_x, real_err)]
        if 0 not in enco_x:
            enco_x = [0, *enco_x]
            enco_y = [0.0, *enco_y]
    semantic_color = "#E07B39"
    mixed_color = "#0072B2"
    enco_color = "#333333"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    real_pos = [INT_POS[x] for x in real_x]
    ax.fill_between(
        real_pos,
        [max(0.0, y - err) for y, err in zip(real_y, real_err)],
        [min(float(cfg["ylim"][1]), y + err) for y, err in zip(real_y, real_err)],
        color=mixed_color,
        alpha=0.12,
        linewidth=0,
        zorder=3,
    )
    ax.plot(
        real_pos,
        real_y,
        color=mixed_color,
        marker="o",
        markersize=6.5,
        linewidth=2.6,
        label=f"{llm_label}: Real names",
        zorder=5,
    )
    ax.plot(
        [INT_POS[x] for x in enco_x],
        enco_y,
        color=enco_color,
        marker="X",
        markersize=8.0,
        linewidth=2.6,
        linestyle=(0, (2, 2)),
        label="ENCO",
        zorder=4,
    )

    ax.axhline(
        semantic_plot_value,
        color=semantic_color,
        linestyle="--",
        linewidth=1.6,
        label=f"{cfg['semantic_label']} ({semantic_plot_value:.2f})",
        zorder=1,
    )

    ax.set_xlabel("Interventional budget M (fixed N = 0)", fontsize=11)
    ax.set_ylabel(cfg["ylabel"], fontsize=11)
    if metric == "f1":
        ax.set_title(
            "Interventions Help LLMs More Than ENCO Without Observations",
            fontsize=12,
            fontweight="bold",
            pad=9,
        )
    ax.set_xlim(-0.12, len(INT_BUDGETS) - 0.78)
    ax.set_ylim(*cfg["ylim"])
    ax.xaxis.set_major_locator(FixedLocator(list(range(len(INT_BUDGETS)))))
    ax.xaxis.set_major_formatter(FixedFormatter([str(budget) for budget in INT_BUDGETS]))
    ax.yaxis.set_major_locator(FixedLocator(cfg["yticks"]))
    if cfg["yticklabels"] is not None:
        ax.yaxis.set_major_formatter(FixedFormatter(cfg["yticklabels"]))

    ax.grid(alpha=0.25, color="gray", linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", labelsize=10)

    legend_handles = [
        Line2D([0], [0], color=enco_color, marker="X", markersize=8.0, linewidth=2.6, linestyle=(0, (2, 2)), label="ENCO"),
        Line2D([0], [0], color=mixed_color, marker="o", markersize=6.5, linewidth=2.6, label=f"{llm_label}: Real names"),
        Line2D(
            [0],
            [0],
            color=semantic_color,
            linestyle="--",
            linewidth=1.6,
            label=f"{cfg['semantic_label']} ({semantic_plot_value:.2f})",
        ),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        framealpha=0.0,
        facecolor="white",
        edgecolor="#DDDDDD",
        fontsize=8.5,
        columnspacing=1.1,
        handlelength=1.8,
    )
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    written = _write_formats(fig, out_dir, basename, formats)
    plt.close(fig)
    return written


def _plot_metric(
    summary_csv: Path,
    out_dir: Path,
    basename: str,
    formats: list[str],
    metric: str,
    *,
    strict: bool,
) -> list[Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FixedFormatter

    rows = _read_summary(summary_csv)
    semantic = _by_obs(_select(rows, "semantic_floor"))
    real_data = _by_obs(_select(rows, "llm_real"))
    ges_data = _by_obs(_select(rows, "ges_anchor"))

    if 0 not in semantic:
        raise ValueError("Missing semantic_floor row for obs_n=0, int_n=0.")

    # Treat names-only as the N=0 point for the real-name LLM curve only.
    real_with_floor = {0: semantic[0], **real_data}

    real_x, real_y, real_err = _series_for_obs(
        real_with_floor,
        OBS_BUDGETS,
        metric=metric,
        strict=strict,
        label="llm_real/semantic_floor",
    )
    ges_x, ges_y, _ = _series_for_obs(
        ges_data,
        CLASSICAL_OBS_BUDGETS,
        metric=metric,
        strict=strict,
        label="GES",
    )

    if not real_x:
        raise ValueError("No LLM rows available for requested observational budgets.")
    if not ges_x:
        raise ValueError("No GES rows available for requested observational budgets.")

    llm_label = _display_model_name(_last_present(real_with_floor, OBS_BUDGETS).system)
    cfg = _metric_config(metric, rows)

    semantic_value, _ = _metric(semantic[0], metric)
    semantic_plot_value = SEMANTIC_FLOOR_F1 if metric == "f1" else semantic_value
    if metric == "f1":
        real_y = [SEMANTIC_FLOOR_F1 if x == 0 else y for x, y in zip(real_x, real_y)]
        real_err = [0.0 if x == 0 else err for x, err in zip(real_x, real_err)]
        if 0 not in ges_x:
            ges_x = [0, *ges_x]
            ges_y = [0.0, *ges_y]
    semantic_color = "#E07B39"
    mixed_color = "#0072B2"
    ges_color = "#333333"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    real_pos = [OBS_POS[x] for x in real_x]
    ax.fill_between(
        real_pos,
        [max(0.0, y - err) for y, err in zip(real_y, real_err)],
        [min(float(cfg["ylim"][1]), y + err) for y, err in zip(real_y, real_err)],
        color=mixed_color,
        alpha=0.12,
        linewidth=0,
        zorder=3,
    )
    ax.plot(
        real_pos,
        real_y,
        color=mixed_color,
        marker="o",
        markersize=6.5,
        linewidth=2.6,
        label=f"{llm_label}: Real names",
        zorder=5,
    )
    ax.plot(
        [OBS_POS[x] for x in ges_x],
        ges_y,
        color=ges_color,
        marker="X",
        markersize=8.0,
        linewidth=2.6,
        linestyle=(0, (2, 2)),
        label="GES",
        zorder=4,
    )

    ax.axhline(
        semantic_plot_value,
        color=semantic_color,
        linestyle="--",
        linewidth=1.6,
        label=f"{cfg['semantic_label']} ({semantic_plot_value:.2f})",
        zorder=1,
    )

    ax.set_xlabel("Observational budget N (fixed M = 0)", fontsize=11)
    ax.set_ylabel(cfg["ylabel"], fontsize=11)
    if metric == "f1":
        ax.set_title(
            "LLMs Show Unstable Gains with More Data",
            fontsize=12,
            fontweight="bold",
            pad=9,
        )
    ax.set_xlim(-0.12, len(OBS_BUDGETS) - 0.78)
    ax.set_ylim(*cfg["ylim"])
    ax.xaxis.set_major_locator(FixedLocator(list(range(len(OBS_BUDGETS)))))
    ax.xaxis.set_major_formatter(FixedFormatter([str(budget) for budget in OBS_BUDGETS]))
    ax.yaxis.set_major_locator(FixedLocator(cfg["yticks"]))
    if cfg["yticklabels"] is not None:
        ax.yaxis.set_major_formatter(FixedFormatter(cfg["yticklabels"]))

    ax.grid(alpha=0.25, color="gray", linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", labelsize=10)

    if metric == "f1":
        ax.annotate(
            "GES scales\nwith data",
            xy=(OBS_POS[ges_x[-1]], ges_y[-1]),
            xytext=(3.2, 0.91),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=8.5,
            color=ges_color,
            arrowprops=dict(arrowstyle="-", color=ges_color, lw=0.9),
        )
        ax.annotate(
            "LLM stays near\nsemantic floor",
            xy=(OBS_POS[1000], real_y[real_x.index(1000)] if 1000 in real_x else real_y[0]),
            xytext=(0.28, 0.68),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=8.5,
            color=mixed_color,
            arrowprops=dict(arrowstyle="-", color=mixed_color, lw=0.9),
        )

    legend_handles = [
        Line2D([0], [0], color=ges_color, marker="X", markersize=8.0, linewidth=2.6, linestyle=(0, (2, 2)), label="GES"),
        Line2D([0], [0], color=mixed_color, marker="o", markersize=6.5, linewidth=2.6, label=f"{llm_label}: Real names"),
        Line2D(
            [0],
            [0],
            color=semantic_color,
            linestyle="--",
            linewidth=1.6,
            label=f"{cfg['semantic_label']} ({semantic_plot_value:.2f})",
        ),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        framealpha=0.0,
        facecolor="white",
        edgecolor="#DDDDDD",
        fontsize=8.5,
        columnspacing=1.1,
        handlelength=1.8,
    )
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    written = _write_formats(fig, out_dir, basename, formats)
    plt.close(fig)
    return written


def plot(summary_csv: Path, out_dir: Path, basename: str, formats: list[str], *, strict: bool) -> list[Path]:
    written = _plot_metric(summary_csv, out_dir, basename, formats, metric="f1", strict=strict)
    written.extend(_plot_metric(summary_csv, out_dir, f"{basename}_shd", formats, metric="shd", strict=strict))
    return written


def plot_intervention(
    response_summary_csv: Path,
    out_dir: Path,
    basename: str,
    formats: list[str],
    *,
    strict: bool,
    llm_model: str,
) -> list[Path]:
    written = _plot_intervention_metric(
        response_summary_csv,
        out_dir,
        basename,
        formats,
        metric="f1",
        strict=strict,
        llm_model=llm_model,
    )
    written.extend(
        _plot_intervention_metric(
            response_summary_csv,
            out_dir,
            f"{basename}_shd",
            formats,
            metric="shd",
            strict=strict,
            llm_model=llm_model,
        )
    )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", choices=["obs", "int"], default="obs")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--response-summary-csv", type=Path, default=DEFAULT_RESPONSE_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--basename", default=None)
    parser.add_argument("--formats", nargs="+", default=["pdf", "png", "svg"], choices=["pdf", "png", "svg"])
    parser.add_argument("--llm-model", default="gpt-5.2-pro")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested budget is missing for the selected sweep.",
    )
    args = parser.parse_args()

    if args.sweep == "obs":
        basename = args.basename or "sachs_figure1_obs_sweep_int0_gpt-52-pro"
        written = plot(args.summary_csv, args.out_dir, basename, args.formats, strict=args.strict)
    else:
        basename = args.basename or f"sachs_figure1_int_sweep_obs0_enco_{_model_slug(args.llm_model)}"
        written = plot_intervention(
            args.response_summary_csv,
            args.out_dir,
            basename,
            args.formats,
            strict=args.strict,
            llm_model=args.llm_model,
        )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
