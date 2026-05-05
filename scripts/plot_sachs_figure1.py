#!/usr/bin/env python3
"""Plot the Sachs observational-budget Figure 1 for MICAD.

This version fixes the interventional budget at M=0 and sweeps observational
budget N over 0, 100, 500, 1000, and 5000.  The N=0 point is the names-only
semantic baseline for the LLM curves.  Classical observational baselines are PC
and GES.
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
OBS_BUDGETS = [0, 100, 500, 1000, 5000]
CLASSICAL_OBS_BUDGETS = [100, 500, 1000, 5000]
OBS_POS = {budget: index for index, budget in enumerate(OBS_BUDGETS)}


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
            "legend_loc": "lower right",
        }
    if metric == "shd":
        max_y = max(row.shd_mean for row in all_rows)
        y_max = max(22.0, math.ceil((max_y + 1.0) / 2.0) * 2.0)
        return {
            "ylabel": "SHD",
            "ylim": (0.0, y_max),
            "yticks": list(range(0, int(y_max) + 1, 5)),
            "yticklabels": None,
            "legend_loc": "upper right",
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
    anon_data = _by_obs(_select(rows, "llm_anonymized"))
    pc_data = _by_obs(_select(rows, "pc_anchor"))
    ges_data = _by_obs(_select(rows, "ges_anchor"))

    if 0 not in semantic:
        raise ValueError("Missing semantic_floor row for obs_n=0, int_n=0.")

    # Treat names-only as the N=0 point for both LLM curves.
    real_with_floor = {0: semantic[0], **real_data}
    anon_with_floor = {0: semantic[0], **anon_data}

    real_x, real_y, real_err = _series_for_obs(
        real_with_floor,
        OBS_BUDGETS,
        metric=metric,
        strict=strict,
        label="llm_real/semantic_floor",
    )
    anon_x, anon_y, anon_err = _series_for_obs(
        anon_with_floor,
        OBS_BUDGETS,
        metric=metric,
        strict=strict,
        label="llm_anonymized/semantic_floor",
    )
    pc_x, pc_y, pc_err = _series_for_obs(
        pc_data,
        CLASSICAL_OBS_BUDGETS,
        metric=metric,
        strict=strict,
        label="PC",
    )
    ges_x, ges_y, ges_err = _series_for_obs(
        ges_data,
        CLASSICAL_OBS_BUDGETS,
        metric=metric,
        strict=strict,
        label="GES",
    )

    if not real_x and not anon_x:
        raise ValueError("No LLM rows available for requested observational budgets.")
    if not pc_x and not ges_x:
        raise ValueError("No PC/GES rows available for requested observational budgets.")

    llm_label = _display_model_name(_last_present(real_with_floor, OBS_BUDGETS).system)
    cfg = _metric_config(metric, rows)

    semantic_color = "#D55E00"
    mixed_color = "#0072B2"
    pc_color = "#6F6F6F"
    ges_color = "#333333"

    fig, ax = plt.subplots(figsize=(7.3, 4.4))
    ax.set_facecolor("white")

    ax.errorbar(
        [OBS_POS[x] for x in real_x],
        real_y,
        yerr=real_err,
        color=mixed_color,
        marker="o",
        markersize=5,
        linewidth=2.0,
        capsize=2.5,
        label=f"{llm_label}: Real names",
        zorder=5,
    )
    ax.errorbar(
        [OBS_POS[x] for x in anon_x],
        anon_y,
        yerr=anon_err,
        color=mixed_color,
        marker="s",
        markersize=5,
        linewidth=2.0,
        linestyle=(0, (4, 2)),
        capsize=2.5,
        label=f"{llm_label}: Anonymized",
        zorder=5,
    )
    ax.errorbar(
        [OBS_POS[x] for x in pc_x],
        pc_y,
        yerr=pc_err,
        color=pc_color,
        marker="D",
        markersize=5.2,
        linewidth=1.8,
        linestyle=(0, (5, 3)),
        capsize=2.5,
        label="PC",
        zorder=4,
    )
    ax.errorbar(
        [OBS_POS[x] for x in ges_x],
        ges_y,
        yerr=ges_err,
        color=ges_color,
        marker="X",
        markersize=5.8,
        linewidth=1.8,
        linestyle=(0, (2, 2)),
        capsize=2.5,
        label="GES",
        zorder=4,
    )

    semantic_value, _ = _metric(semantic[0], metric)
    ax.axhline(
        semantic_value,
        color=semantic_color,
        linestyle=(0, (5, 3)),
        linewidth=1.5,
        zorder=2,
    )
    ax.annotate(
        "names-only",
        xy=(OBS_POS[OBS_BUDGETS[-1]], semantic_value),
        xytext=(-4, 7),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.7,
        color=semantic_color,
    )

    ax.set_xlabel("Observational budget $N$ (fixed $M=0$)", fontsize=10.5)
    ax.set_ylabel(cfg["ylabel"], fontsize=10.5)
    ax.set_xlim(-0.12, len(OBS_BUDGETS) - 0.78)
    ax.set_ylim(*cfg["ylim"])
    ax.xaxis.set_major_locator(FixedLocator(list(range(len(OBS_BUDGETS)))))
    ax.xaxis.set_major_formatter(FixedFormatter(["0", "100", "500", "1000", "5000"]))
    ax.yaxis.set_major_locator(FixedLocator(cfg["yticks"]))
    if cfg["yticklabels"] is not None:
        ax.yaxis.set_major_formatter(FixedFormatter(cfg["yticklabels"]))

    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.6, alpha=0.55)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", labelsize=9)

    legend_handles = [
        Line2D([0], [0], color=mixed_color, marker="o", markersize=5, linewidth=2.0, label=f"{llm_label}: Real names"),
        Line2D(
            [0],
            [0],
            color=mixed_color,
            marker="s",
            markersize=5,
            linewidth=2.0,
            linestyle=(0, (4, 2)),
            label=f"{llm_label}: Anonymized",
        ),
        Line2D([0], [0], color=pc_color, marker="D", markersize=5.2, linewidth=1.8, linestyle=(0, (5, 3)), label="PC"),
        Line2D([0], [0], color=ges_color, marker="X", markersize=5.8, linewidth=1.8, linestyle=(0, (2, 2)), label="GES"),
        Line2D([0], [0], color=semantic_color, linestyle=(0, (5, 3)), linewidth=1.5, label="Names-only"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc=cfg["legend_loc"],
        ncol=2,
        frameon=True,
        framealpha=0.97,
        facecolor="white",
        edgecolor="#DDDDDD",
        fontsize=7.8,
        columnspacing=1.2,
        handlelength=2.0,
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
