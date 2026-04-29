#!/usr/bin/env python3
"""Plot the Sachs falsifiability-target figure for the MICAD paper."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
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


def _values(rows: list[SummaryRow]) -> tuple[list[int], list[float], list[float]]:
    return (
        [row.int_n for row in rows],
        [row.f1_mean for row in rows],
        [row.f1_ci95_halfwidth for row in rows],
    )


def _display_model_name(model: str) -> str:
    if model == "gpt-5-mini":
        return "GPT-5 mini"
    return model


def _configure_matplotlib() -> None:
    if not os.getenv("MPLCONFIGDIR"):
        mpl_dir = Path("/tmp") / f"matplotlib_{os.getuid()}"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)


def plot(summary_csv: Path, out_dir: Path, basename: str, formats: list[str]) -> list[Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter

    rows = _read_summary(summary_csv)
    enco = [row for row in _select(rows, "enco_ceiling") if row.int_n <= MAX_PLOTTED_INTERVENTIONS]
    enco_obs = _select(rows, "enco_observational")
    real = [row for row in _select(rows, "llm_real") if row.int_n <= MAX_PLOTTED_INTERVENTIONS]
    anon = [row for row in _select(rows, "llm_anonymized") if row.int_n <= MAX_PLOTTED_INTERVENTIONS]
    floor = _single(rows, "semantic_floor")
    pc = _single(rows, "pc_anchor")
    ges = _single(rows, "ges_anchor")

    if not enco or not real or not anon:
        raise ValueError("Summary CSV must contain ENCO, llm_real, and llm_anonymized rows.")
    llm_label = _display_model_name(real[0].system)

    semantic_color = "#D55E00"
    mixed_color = "#0072B2"
    data_color = "#111111"
    anchor_color = "#6F6F6F"
    below_floor_fill = "#F3C78A"
    target_fill = "#B9D9EC"

    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    ax.set_facecolor("white")

    all_m = sorted({row.int_n for row in enco + real + anon} | {0})
    x_min, x_max = -22, min(max(all_m), MAX_PLOTTED_INTERVENTIONS) + 38
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.035, 1.05)

    ax.axhspan(0.0, floor.f1_mean, color=below_floor_fill, alpha=0.16, zorder=0)
    ax.axhspan(floor.f1_mean, 1.0, color=target_fill, alpha=0.11, zorder=0)

    ax.axhline(
        floor.f1_mean,
        color=semantic_color,
        linestyle=(0, (5, 3)),
        linewidth=1.8,
        label=f"Semantic-only floor ({floor.f1_mean:.2f})",
        zorder=2,
    )

    enco_x, enco_y, _ = _values(enco)
    real_x, real_y, real_err = _values(real)
    anon_x, anon_y, anon_err = _values(anon)

    ax.plot(
        enco_x,
        enco_y,
        color=data_color,
        marker="o",
        markersize=5.5,
        linewidth=2.4,
        label="ENCO data-only ceiling",
        zorder=4,
    )
    if enco_obs:
        offsets = [-14.0, 14.0] if len(enco_obs) == 2 else [0.0 for _ in enco_obs]
        obs_x = [row.int_n + float(offset) for row, offset in zip(enco_obs, offsets)]
        obs_y = [row.f1_mean for row in enco_obs]
        ax.scatter(
            obs_x,
            obs_y,
            marker="^",
            s=46,
            facecolor="white",
            edgecolor=data_color,
            linewidth=1.4,
            zorder=7,
            clip_on=False,
            label="ENCO obs-only at M=0",
        )
        for x, row in zip(obs_x, enco_obs):
            label_offset = -4 if row.obs_n <= 1000 else 4
            ax.annotate(
                f"N={row.obs_n}",
                xy=(x, row.f1_mean),
                xytext=(x + label_offset, row.f1_mean + 0.06),
                fontsize=7.5,
                color=data_color,
                ha="right" if label_offset < 0 else "left",
                va="center",
            )
    ax.errorbar(
        real_x,
        real_y,
        yerr=real_err,
        color=mixed_color,
        marker="o",
        markersize=5,
        linewidth=2.0,
        elinewidth=1.0,
        capsize=2.5,
        label=f"{llm_label}, real names",
        zorder=5,
    )
    ax.errorbar(
        anon_x,
        anon_y,
        yerr=anon_err,
        color=mixed_color,
        marker="s",
        markersize=5,
        linewidth=2.0,
        linestyle=(0, (4, 2)),
        elinewidth=1.0,
        capsize=2.5,
        label=f"{llm_label}, anonymized",
        zorder=5,
    )

    ax.scatter(
        [0, 0],
        [pc.f1_mean, ges.f1_mean],
        marker="D",
        s=34,
        color=anchor_color,
        zorder=6,
        label="PC/GES at M=0",
    )
    ax.annotate(
        "PC",
        xy=(0, pc.f1_mean),
        xytext=(-7, pc.f1_mean - 0.055),
        ha="right",
        fontsize=8.5,
        color=anchor_color,
    )
    ax.annotate(
        "GES",
        xy=(0, ges.f1_mean),
        xytext=(-7, ges.f1_mean + 0.035),
        ha="right",
        fontsize=8.5,
        color=anchor_color,
    )

    ax.annotate(
        "No information\nbelow floor",
        xy=(150, floor.f1_mean * 0.42),
        ha="center",
        va="center",
        fontsize=8.8,
        color="#6B4B20",
    )
    ax.annotate(
        "Real names:\nsemantic prior + data",
        xy=(145, 0.62),
        xytext=(160, 0.86),
        arrowprops=dict(arrowstyle="->", color=mixed_color, lw=1.0),
        ha="center",
        va="center",
        fontsize=9,
        color=mixed_color,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
    )
    ax.annotate(
        "Anonymized:\ndata-driven signal",
        xy=(200, 0.76),
        xytext=(96, 0.92),
        arrowprops=dict(arrowstyle="->", color=mixed_color, lw=1.0),
        ha="center",
        va="center",
        fontsize=9,
        color=mixed_color,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
    )
    ax.annotate(
        "Target zone for\nfuture MICAD methods",
        xy=(158, max(floor.f1_mean + 0.08, 0.62)),
        xytext=(202, 0.70),
        arrowprops=dict(arrowstyle="->", color=data_color, lw=1.2),
        ha="center",
        va="center",
        fontsize=9.5,
        color=data_color,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
    )

    ax.set_xlabel("Intervention budget M; LLM curves at fixed N = 1000", fontsize=10.5)
    ax.set_ylabel("F1", fontsize=10.5)
    ax.xaxis.set_major_locator(FixedLocator([0, 50, 100, 200]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0", "50", "100", "200"]))
    ax.yaxis.set_major_locator(FixedLocator([0.0, 0.25, 0.50, 0.75, 1.0]))
    ax.yaxis.set_major_formatter(FixedFormatter(["0.00", "0.25", "0.50", "0.75", "1.00"]))

    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.6, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="both", labelsize=9)

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#DDDDDD",
        fontsize=8.0,
        columnspacing=1.0,
        handlelength=2.0,
    )
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        out_path = out_dir / f"{basename}.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        written.append(out_path)
    plt.close(fig)
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
