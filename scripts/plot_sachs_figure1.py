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
DEFAULT_OUT_DIR = REPO_ROOT / "benchmark_runs" / "sachs_figure1"
OBS_BUDGETS = [0, 1000, 2000, 3000, 4000, 5000]
CLASSICAL_OBS_BUDGETS = [1000, 2000, 3000, 4000, 5000]
OBS_POS = {budget: index for index, budget in enumerate(OBS_BUDGETS)}
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


def _select(rows: Iterable[SummaryRow], line_id: str, *, int_n: int = 0) -> list[SummaryRow]:
    return sorted(
        (row for row in rows if row.line_id == line_id and row.int_n == int_n),
        key=lambda row: row.obs_n,
    )


def _by_obs(rows: Iterable[SummaryRow]) -> dict[int, SummaryRow]:
    result: dict[int, SummaryRow] = {}
    for row in rows:
        if row.obs_n in result:
            raise ValueError(f"Duplicate row for {row.line_id} at obs_n={row.obs_n}, int_n={row.int_n}.")
        result[row.obs_n] = row
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


def _last_present(row_map: dict[int, SummaryRow], budgets: list[int]) -> SummaryRow:
    for budget in reversed(budgets):
        if budget in row_map:
            return row_map[budget]
    raise ValueError("No rows available for requested budgets.")


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--basename", default="sachs_figure1_obs_sweep_int0")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png", "svg"], choices=["pdf", "png", "svg"])
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested obs budget is missing for LLM, PC, or GES.",
    )
    args = parser.parse_args()

    written = plot(args.summary_csv, args.out_dir, args.basename, args.formats, strict=args.strict)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
