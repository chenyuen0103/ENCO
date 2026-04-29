#!/usr/bin/env python3
"""Plot the MICAD-Bench specification-to-cells schematic for the MICAD paper."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "benchmark_runs" / "sachs_figure2"


def _configure_matplotlib() -> None:
    if not os.getenv("MPLCONFIGDIR"):
        mpl_dir = Path("/tmp") / f"matplotlib_{os.getuid()}"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)


def plot(out_dir: Path, basename: str, formats: list[str]) -> list[Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    mixed_fill = "#D9ECF7"
    mixed_edge = "#0072B2"
    semantic_fill = "#FBE5CF"
    semantic_edge = "#D55E00"
    classical_fill = "#ECECEC"
    classical_edge = "#111111"
    neutral_fill = "#F7F7F7"
    neutral_edge = "#AAAAAA"
    text_dark = "#1F1F1F"

    fig, ax = plt.subplots(figsize=(11.6, 5.2))
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        edge: str,
        title: str,
        lines: list[str],
        title_size: float = 9.2,
        body_size: float = 8.0,
        dashed: bool = False,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.6,
            edgecolor=edge,
            facecolor=fill,
            linestyle=(0, (4, 2)) if dashed else "solid",
        )
        ax.add_patch(patch)
        ax.text(x + 0.16, y + h - 0.28, title, fontsize=title_size, fontweight="bold", color=text_dark, va="top")
        for i, line in enumerate(lines):
            ax.text(x + 0.16, y + h - 0.58 - 0.28 * i, line, fontsize=body_size, color=text_dark, va="top")

    # Left: one specification
    spec = FancyBboxPatch(
        (0.40, 1.00),
        3.55,
        6.35,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        linewidth=1.8,
        edgecolor="#555555",
        facecolor="white",
    )
    ax.add_patch(spec)
    ax.text(0.80, 7.05, "One MICAD-Bench", fontsize=12.0, fontweight="bold", color=text_dark, va="top")
    ax.text(0.80, 6.72, "specification", fontsize=12.0, fontweight="bold", color=text_dark, va="top")
    spec_lines = [
        "Graph: Sachs",
        r"Data conditions: $(N,M)=(1000,0),(1000,50)$",
        "Naming: real, anonymized",
        "Prompt representations:",
        "  summary, matrix, names_only",
        "Model: gpt-5-mini",
        "Baselines: PC, GES, ENCO",
    ]
    for i, line in enumerate(spec_lines):
        ax.text(0.80, 6.05 - 0.58 * i, line, fontsize=9.2, color=text_dark, va="top")

    ax.annotate(
        "",
        xy=(4.75, 4.02),
        xytext=(4.05, 4.02),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#555555"),
    )

    # Center: LLM cell matrix
    ax.text(8.35, 8.08, "LLM cells from the same specification", fontsize=12.4, fontweight="bold", ha="center")
    ax.text(6.55, 7.55, "summary", fontsize=10.1, fontweight="bold", ha="center")
    ax.text(8.35, 7.55, "matrix", fontsize=10.1, fontweight="bold", ha="center")
    ax.text(10.15, 7.55, "names_only", fontsize=10.1, fontweight="bold", ha="center")
    ax.text(5.38, 6.30, "real", fontsize=10.2, fontweight="bold", ha="right", va="center")
    ax.text(5.38, 4.50, "anonymized", fontsize=10.2, fontweight="bold", ha="right", va="center")

    cell_w, cell_h = 1.60, 1.22
    xs = [5.45, 7.30, 9.15]
    ys = [5.72, 3.92]

    box(xs[0], ys[0], cell_w, cell_h, mixed_fill, mixed_edge, "real / summary", ["mixed-info cell"])
    box(xs[1], ys[0], cell_w, cell_h, mixed_fill, mixed_edge, "real / matrix", ["mixed-info cell"])
    box(xs[0], ys[1], cell_w, cell_h, mixed_fill, mixed_edge, "anon / summary", ["mixed-info cell"])
    box(xs[1], ys[1], cell_w, cell_h, mixed_fill, mixed_edge, "anon / matrix", ["mixed-info cell"])
    box(xs[2], ys[0], cell_w, cell_h, semantic_fill, semantic_edge, "semantic floor", ["names_only reference"])
    box(
        xs[2],
        ys[1],
        cell_w,
        cell_h,
        neutral_fill,
        neutral_edge,
        "not instantiated",
        ["no second names_only cell"],
        dashed=True,
    )

    ax.text(8.35, 2.95, "Naming regime and prompt representation are the controlled axes.", fontsize=9.4, ha="center")

    # Right: classical references
    panel = FancyBboxPatch(
        (12.15, 2.05),
        3.35,
        5.60,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        linewidth=1.8,
        edgecolor=classical_edge,
        facecolor="white",
    )
    ax.add_patch(panel)
    ax.text(13.82, 7.32, "Matched classical", fontsize=11.0, fontweight="bold", ha="center", va="top")
    ax.text(13.82, 6.98, "references", fontsize=11.0, fontweight="bold", ha="center", va="top")
    box(12.50, 5.82, 2.55, 1.10, classical_fill, classical_edge, "PC / GES", ["observational anchor"])
    box(12.50, 4.20, 2.55, 1.10, classical_fill, classical_edge, "ENCO", ["interventional anchor"])
    box(12.50, 2.58, 2.55, 1.10, semantic_fill, semantic_edge, "shared anchors", ["scores are read against", "these references"])

    ax.annotate(
        "",
        xy=(11.95, 4.85),
        xytext=(11.10, 4.85),
        arrowprops=dict(arrowstyle="->", lw=1.8, color=classical_edge),
    )

    # Bottom legend / interpretation
    legend_y = 0.46
    ax.add_patch(Rectangle((5.15, legend_y), 0.34, 0.24, facecolor=mixed_fill, edgecolor=mixed_edge, linewidth=1.3))
    ax.text(5.60, legend_y + 0.12, "mixed-information cells", va="center", fontsize=9.1)
    ax.add_patch(Rectangle((8.35, legend_y), 0.34, 0.24, facecolor=semantic_fill, edgecolor=semantic_edge, linewidth=1.3))
    ax.text(8.80, legend_y + 0.12, "semantic-only reference", va="center", fontsize=9.1)
    ax.add_patch(Rectangle((11.40, legend_y), 0.34, 0.24, facecolor=classical_fill, edgecolor=classical_edge, linewidth=1.3))
    ax.text(11.85, legend_y + 0.12, "classical references", va="center", fontsize=9.1)

    ax.text(
        8.35,
        1.46,
        "One specification yields the semantic floor, the mixed-information cells, and the matched classical references.",
        ha="center",
        va="center",
        fontsize=8.9,
        color="#333333",
    )

    fig.tight_layout(pad=0.4)
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
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--basename", default="sachs_figure2_spec_to_cells")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png", "svg"], choices=["pdf", "png", "svg"])
    args = parser.parse_args()

    written = plot(args.out_dir, args.basename, args.formats)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
