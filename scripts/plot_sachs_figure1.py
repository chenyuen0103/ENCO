#!/usr/bin/env python3
"""Plot the Sachs falsifiability-target figure for the MICAD paper."""

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
MAX_PLOTTED_INTERVENTIONS = 200


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


def _read_summary(path: Path) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                SummaryRow(
                    line_id=str(raw["line_id"]),
                    system=str(raw["system"]),
                    naming_regime=str(raw["naming_regime"]),
                    obs_n=int(float(raw["obs_n"])),
                    int_n=int(float(raw["int_n"])),
                    n_valid=int(float(raw["n_valid"])),
                    f1_mean=float(raw["f1_mean"]),
                    f1_ci95_halfwidth=float(raw.get("f1_ci95_halfwidth") or 0.0),
                    shd_mean=float(raw["shd_mean"]),
                    shd_ci95_halfwidth=(
                        1.96
                        * float(raw.get("shd_sd") or 0.0)
                        / math.sqrt(max(int(float(raw["n_valid"])), 1))
                    ),
                )
            )
    return rows


def _select(rows: Iterable[SummaryRow], line_id: str) -> list[SummaryRow]:
    return sorted((row for row in rows if row.line_id == line_id), key=lambda row: (row.int_n, row.obs_n))


def _single(rows: Iterable[SummaryRow], line_id: str) -> SummaryRow:
    matches = _select(rows, line_id)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one `{line_id}` row, found {len(matches)}.")
    return matches[0]


def _values(rows: list[SummaryRow], metric: str) -> tuple[list[int], list[float], list[float]]:
    if metric == "f1":
        return (
            [row.int_n for row in rows],
            [row.f1_mean for row in rows],
            [row.f1_ci95_halfwidth for row in rows],
        )
    if metric == "shd":
        return (
            [row.int_n for row in rows],
            [row.shd_mean for row in rows],
            [row.shd_ci95_halfwidth for row in rows],
        )
    raise ValueError(f"Unsupported metric: {metric}")


def _value(row: SummaryRow, metric: str) -> float:
    if metric == "f1":
        return row.f1_mean
    if metric == "shd":
        return row.shd_mean
    raise ValueError(f"Unsupported metric: {metric}")


def _metric_config(metric: str, floor: SummaryRow, rows: list[SummaryRow]) -> dict[str, object]:
    if metric == "f1":
        return {
            "ylabel": "F1",
            "ylim": (-0.035, 1.05),
            "yticks": [0.0, 0.25, 0.50, 0.75, 1.0],
            "yticklabels": ["0.00", "0.25", "0.50", "0.75", "1.00"],
            "semantic_label": "",
            "legend_semantic": f"Semantic-Only ({floor.f1_mean:.2f})",
            "lower_region": (0.0, floor.f1_mean),
            "upper_region": (floor.f1_mean, 1.0),
            "bad_region_label": "Below Semantic-Only:\nNo mixed-information gain",
            "bad_region_xy": (145, floor.f1_mean * 0.44),
            "gap_label": "Headroom:\nENCO - LLM",
            "better": "higher",
            "legend_loc": "lower left",
            "legend_bbox": (0.01, 0.02),
        }

    if metric == "shd":
        max_y = max(_value(row, "shd") for row in rows)
        y_max = max(24.0, math.ceil((max_y + 1.0) / 2.0) * 2.0)
        return {
            "ylabel": "SHD",
            "ylim": (0.0, y_max),
            "yticks": [0, 5, 10, 15, 20] if y_max <= 22 else list(range(0, int(y_max) + 1, 5)),
            "yticklabels": None,
            "semantic_label": "Semantic-only baseline",
            "legend_semantic": f"Semantic baseline ({floor.shd_mean:.1f})",
            "lower_region": (0.0, floor.shd_mean),
            "upper_region": (floor.shd_mean, y_max),
            "bad_region_label": "Above baseline:\nworse than semantic-only",
            "bad_region_xy": (145, floor.shd_mean + (y_max - floor.shd_mean) * 0.42),
            "gap_label": "Gap:\nLLM - ENCO",
            "better": "lower",
            "legend_loc": "upper right",
            "legend_bbox": (0.99, 0.99),
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


def _configure_matplotlib() -> None:
    if not os.getenv("MPLCONFIGDIR"):
        mpl_dir = Path("/tmp") / f"matplotlib_{os.getuid()}"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)


def _plot_metric(summary_csv: Path, out_dir: Path, basename: str, formats: list[str], metric: str) -> list[Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FixedFormatter

    rows = _read_summary(summary_csv)
    enco = [row for row in _select(rows, "enco_ceiling") if row.int_n <= MAX_PLOTTED_INTERVENTIONS]
    real = [row for row in _select(rows, "llm_real") if row.int_n <= MAX_PLOTTED_INTERVENTIONS]
    anon = [row for row in _select(rows, "llm_anonymized") if row.int_n <= MAX_PLOTTED_INTERVENTIONS]
    floor = _single(rows, "semantic_floor")
    pc = _single(rows, "pc_anchor")
    ges = _single(rows, "ges_anchor")

    if not enco or not real or not anon:
        raise ValueError("Summary CSV must contain ENCO, llm_real, and llm_anonymized rows.")
    llm_label = _display_model_name(real[0].system)
    cfg = _metric_config(metric, floor, rows)
    floor_y = _value(floor, metric)
    pc_y = _value(pc, metric)
    ges_y = _value(ges, metric)

    semantic_color = "#D55E00"
    mixed_color = "#0072B2"
    data_color = "#111111"
    anchor_color = "#6F6F6F"
    below_floor_fill = "#F3C78A"
    target_fill = "#B9D9EC"

    fig, ax = plt.subplots(figsize=(7.3, 4.65))
    ax.set_facecolor("white")

    all_m = sorted({row.int_n for row in enco + real + anon} | {0})
    x_min, x_max = -20, min(max(all_m), MAX_PLOTTED_INTERVENTIONS) + 32
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(*cfg["ylim"])

    lower_region = cfg["lower_region"]
    upper_region = cfg["upper_region"]
    if cfg["better"] == "higher":
        ax.axhspan(*lower_region, color=below_floor_fill, alpha=0.13, zorder=0)
        ax.axhspan(*upper_region, color=target_fill, alpha=0.07, zorder=0)
    else:
        ax.axhspan(*lower_region, color=target_fill, alpha=0.07, zorder=0)
        ax.axhspan(*upper_region, color=below_floor_fill, alpha=0.13, zorder=0)

    ax.axhline(
        floor_y,
        color=semantic_color,
        linestyle=(0, (5, 3)),
        linewidth=1.8,
        zorder=2,
    )

    enco_x, enco_y, _ = _values(enco, metric)
    real_x, real_y, _ = _values(real, metric)
    anon_x, anon_y, _ = _values(anon, metric)

    ax.plot(
        enco_x,
        enco_y,
        color=data_color,
        marker="o",
        markersize=5.5,
        linewidth=2.4,
        zorder=4,
    )
    ax.plot(
        real_x,
        real_y,
        color=mixed_color,
        marker="o",
        markersize=5,
        linewidth=2.0,
        zorder=5,
    )
    ax.plot(
        anon_x,
        anon_y,
        color=mixed_color,
        marker="s",
        markersize=5,
        linewidth=2.0,
        linestyle=(0, (4, 2)),
        zorder=5,
    )

    ax.scatter([0], [pc_y], marker="D", s=34, color=anchor_color, zorder=6)
    ax.scatter([0], [ges_y], marker="X", s=42, color=anchor_color, zorder=6)
    ax.annotate(
        "PC",
        xy=(0, pc_y),
        xytext=(-9, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8.5,
        color=anchor_color,
    )
    ax.annotate(
        "GES",
        xy=(0, ges_y),
        xytext=(-9, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8.5,
        color=anchor_color,
    )

    ax.annotate(
        cfg["semantic_label"],
        xy=(132, floor_y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.8,
        color=semantic_color,
    )
    ax.annotate(
        cfg["bad_region_label"],
        xy=cfg["bad_region_xy"],
        ha="center",
        va="center",
        fontsize=8.8,
        color="#6B4B20",
    )
    headroom_x = 176
    llm_reference_y = max(real_y[-1], anon_y[-1]) if cfg["better"] == "higher" else min(real_y[-1], anon_y[-1])
    headroom_mid_y = (enco_y[-1] + llm_reference_y) * 0.5
    ax.annotate(
        "",
        xy=(headroom_x, enco_y[-1]),
        xytext=(headroom_x, llm_reference_y),
        arrowprops=dict(arrowstyle="<->", color=data_color, lw=1.1),
    )
    ax.annotate(
        cfg["gap_label"],
        xy=(headroom_x, headroom_mid_y),
        xytext=(-7, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8.9,
        color=data_color,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88),
    )

    def _endpoint_label(
        xs: list[int],
        ys: list[float],
        *,
        text: str,
        color: str,
        dx: float,
        dy: float,
        ha: str = "left",
        linestyle: str = "solid",
        marker: str = "o",
    ) -> None:
        ax.annotate(
            text,
            xy=(xs[-1], ys[-1]),
            xytext=(dx, dy),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.9, shrinkA=2, shrinkB=2),
            ha=ha,
            va="center",
            fontsize=8.9,
            color=color,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.9),
        )
        ax.plot(
            [xs[-1], xs[-1] + 0.001],
            [ys[-1], ys[-1]],
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=0,
            alpha=0.0,
        )

    _endpoint_label(enco_x, enco_y, text="ENCO", color=data_color, dx=8, dy=0)

    ax.set_xlabel("Intervention budget $M$ (LLM curves at fixed $N=1000$)", fontsize=10.5)
    ax.set_ylabel(cfg["ylabel"], fontsize=10.5)
    ax.xaxis.set_major_locator(FixedLocator([0, 50, 100, 200]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0", "50", "100", "200"]))
    ax.yaxis.set_major_locator(FixedLocator(cfg["yticks"]))
    if cfg["yticklabels"] is not None:
        ax.yaxis.set_major_formatter(FixedFormatter(cfg["yticklabels"]))

    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.6, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", labelsize=9)

    legend_handles = [
        Line2D([0], [0], color=data_color, marker="o", markersize=5.5, linewidth=2.4, label="ENCO"),
        Line2D([0], [0], color=mixed_color, marker="o", markersize=5, linewidth=2.0, label=f"{llm_label}: Real Names"),
        Line2D([0], [0], color=mixed_color, marker="s", markersize=5, linewidth=2.0, linestyle=(0, (4, 2)), label=f"{llm_label}: Anonymized"),
        Line2D([0], [0], color=semantic_color, linestyle=(0, (5, 3)), linewidth=1.8, label=cfg["legend_semantic"]),
        Line2D([0], [0], marker="D", markersize=6.5, color=anchor_color, linewidth=0, label="PC"),
        Line2D([0], [0], marker="X", markersize=7, color=anchor_color, linewidth=0, label="GES"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc=cfg["legend_loc"],
        bbox_to_anchor=cfg["legend_bbox"],
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

    # fig.text(
    #     0.5,
    #     0.02,
    #     # "Caption note: at M=0, ENCO collapses to the observational-only setting; this point is omitted from the plotted ENCO curve.",
    #     ha="center",
    #     va="bottom",
    #     fontsize=8.0,
    #     color="#333333",
    # )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    written = _write_formats(fig, out_dir, basename, formats)
    plt.close(fig)
    return written


def plot(summary_csv: Path, out_dir: Path, basename: str, formats: list[str]) -> list[Path]:
    written = _plot_metric(summary_csv, out_dir, basename, formats, metric="f1")
    written.extend(_plot_metric(summary_csv, out_dir, f"{basename}_shd", formats, metric="shd"))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--basename", default="sachs_figure1_falsifiability_target")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png", "svg"], choices=["pdf", "png", "svg"])
    args = parser.parse_args()

    written = plot(args.summary_csv, args.out_dir, args.basename, args.formats)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
