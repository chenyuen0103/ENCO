#!/usr/bin/env python3
"""Plot a dataset-composition donut for the MICAD benchmark-data release."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "benchmark_data" / "summaries" / "data_summary.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "benchmark_runs" / "dataset_composition"


@dataclass(frozen=True)
class DatasetRow:
    dataset: str
    nodes: int
    rows: int


def _configure_matplotlib() -> None:
    if not os.getenv("MPLCONFIGDIR"):
        mpl_dir = Path("/tmp") / f"matplotlib_{os.getuid()}"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)


def _read_rows(path: Path) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                DatasetRow(
                    dataset=str(raw["dataset"]),
                    nodes=int(float(raw["nodes"])),
                    rows=int(float(raw["rows"])),
                )
            )
    if not rows:
        raise ValueError(f"No dataset rows found in {path}")
    return rows


def _display_name(dataset: str) -> str:
    if dataset.startswith("synthetic_"):
        parts = dataset.split("_")
        if len(parts) >= 3:
            return f"{parts[1].capitalize()}-25"
    return dataset.replace("_", " ").title()


def _family(dataset: str, nodes: int) -> str:
    if dataset.startswith("synthetic_"):
        return "Synthetic topology controls"
    if dataset in {"diabetes", "pigs"} or nodes >= 100:
        return "Large challenge graphs"
    return "Real reference suite"


def _colors_for_labels(labels: list[str], palette: list[str]) -> list[str]:
    return [palette[idx % len(palette)] for idx, _ in enumerate(labels)]


def _write_formats(fig, out_dir: Path, basename: str, formats: list[str]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        out_path = out_dir / f"{basename}.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        written.append(out_path)
    return written


def plot(summary_csv: Path, out_dir: Path, basename: str, formats: list[str]) -> list[Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = _read_rows(summary_csv)
    family_order = ["Real reference suite", "Synthetic topology controls", "Large challenge graphs"]
    family_rows: dict[str, list[DatasetRow]] = {family: [] for family in family_order}
    for row in rows:
        family_rows.setdefault(_family(row.dataset, row.nodes), []).append(row)

    for family in family_rows:
        family_rows[family].sort(key=lambda row: (row.nodes, row.dataset))

    outer_labels: list[str] = []
    outer_values: list[int] = []
    outer_colors: list[str] = []
    inner_labels: list[str] = []
    inner_values: list[int] = []
    inner_colors: list[str] = []

    palettes = {
        "Real reference suite": ["#B7D3EA", "#9FC5E3", "#7FB3D5", "#5DA5CF", "#3C8DBC", "#21618C"],
        "Synthetic topology controls": ["#F8CFAE", "#F6BE98", "#F4AD82", "#F09C6C", "#EA8B56", "#D97842"],
        "Large challenge graphs": ["#D7D7D7", "#BEBEBE", "#A6A6A6"],
    }
    inner_color_map = {
        "Real reference suite": "#7FB3D5",
        "Synthetic topology controls": "#F4AD82",
        "Large challenge graphs": "#BEBEBE",
    }

    for family in family_order:
        members = family_rows.get(family, [])
        if not members:
            continue
        inner_labels.append(family)
        inner_values.append(sum(max(row.rows, 1) for row in members))
        inner_colors.append(inner_color_map[family])
        outer_labels.extend(_display_name(row.dataset) for row in members)
        outer_values.extend(max(row.rows, 1) for row in members)
        outer_colors.extend(_colors_for_labels([row.dataset for row in members], palettes[family]))

    fig, ax = plt.subplots(figsize=(6.7, 5.2), subplot_kw={"aspect": "equal"})
    ax.set_facecolor("white")

    outer_wedges, outer_texts = ax.pie(
        outer_values,
        radius=1.0,
        labels=outer_labels,
        labeldistance=1.05,
        colors=outer_colors,
        startangle=110,
        counterclock=False,
        wedgeprops=dict(width=0.34, edgecolor="white", linewidth=0.9),
        textprops=dict(fontsize=8.4, color="#222222"),
    )
    for text in outer_texts:
        text.set_rotation(0)

    ax.pie(
        inner_values,
        radius=0.66,
        labels=None,
        colors=inner_colors,
        startangle=110,
        counterclock=False,
        wedgeprops=dict(width=0.34, edgecolor="white", linewidth=0.9),
    )
    ax.add_artist(plt.Circle((0, 0), 0.30, color="white"))
    ax.text(0, 0.04, "MICAD", ha="center", va="center", fontsize=12.0, fontweight="bold", color="#222222")
    ax.text(0, -0.12, "dataset", ha="center", va="center", fontsize=10.0, color="#222222")

    legend_handles = [Patch(facecolor=inner_color_map[family], edgecolor="none", label=family) for family in family_order]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=1,
        frameon=False,
        fontsize=8.6,
    )
    ax.set_title("Dataset composition", fontsize=13.0, fontweight="bold", pad=12)
    fig.text(
        0.5,
        0.02,
        "Slice area is proportional to generated prompt rows per graph family.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#333333",
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.98))
    written = _write_formats(fig, out_dir, basename, formats)
    plt.close(fig)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot dataset-composition donut from benchmark data summary CSV.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--basename", default="dataset_composition")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png", "svg"], choices=["pdf", "png", "svg"])
    args = parser.parse_args()
    written = plot(args.summary_csv, args.out_dir, args.basename, args.formats)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
