#!/usr/bin/env python3
"""Horizontal lollipop chart of LLM-CD audit results vs. PC / GES / semantic floor.

Two panels side-by-side: F1 (higher is better) and SHD (lower is better).
Methods are sorted by F1 descending.  Reference lines mark PC, GES, and the
semantic-only floor so readers can see at a glance which methods clear each bar.

Usage (from repo root):
    python3 scripts/plot_llm_cd_audit.py
    python3 scripts/plot_llm_cd_audit.py --obs 5000 --out-dir benchmark_runs/figures
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

csv.field_size_limit(10_000_000)

DEFAULT_OUT_DIR = REPO_ROOT / "benchmark_runs" / "figures"

# ── display labels ────────────────────────────────────────────────────────────
DISPLAY = {
    "JiralerspongBFS":      "Jiralerspong BFS",
    "JiralerspongPairwise": "Jiralerspong Pairwise",
    "TakayamaSCP":          "Takayama SCP",
    "CausalLLMData":        "Roy CausalLLM + Data",
    "CausalLLMPrompt":      "Roy CausalLLM",
}


# ── colours (consistent with existing paper figures) ─────────────────────────
COL_LLM      = "#0072B2"   # blue  – LLM-CD methods
COL_SEMANTIC = "#D55E00"   # orange – semantic floor
COL_ANCHOR   = "#6F6F6F"   # grey  – PC / GES anchors
COL_STEM     = "#BBBBBB"   # light grey stems


@dataclass
class MethodRow:
    name: str
    label: str
    f1: float
    shd: float
    kind: str   # "llm" | "reference" | "floor"


# ── data loading ─────────────────────────────────────────────────────────────

def _read_eval_summary(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _find_row(rows: list[dict], *, system: str, obs_n: int | None = None,
              naming_regime: str | None = None) -> dict | None:
    for r in rows:
        if r.get("system") != system:
            continue
        if obs_n is not None and int(float(r.get("obs_n", -1))) != obs_n:
            continue
        if naming_regime is not None and r.get("naming_regime") != naming_regime:
            continue
        return r
    return None


def _f1_shd(row: dict) -> tuple[float, float]:
    return float(row["avg_f1"]), float(row["avg_shd"])


def load_data(obs: int) -> tuple[list[MethodRow], dict[str, float], dict[str, float]]:
    """Return (method_rows, f1_refs, shd_refs) for the given obs budget."""

    full_grid   = _read_eval_summary(REPO_ROOT / "benchmark_runs" / "sachs_full_grid"          / "evaluation_summary.csv")
    llm_base    = _read_eval_summary(REPO_ROOT / "benchmark_runs" / "sachs_llm_baselines_obs1000" / "evaluation_summary.csv")

    # ── references ────────────────────────────────────────────────────────────
    pc_row  = _find_row(full_grid, system="PC",  obs_n=obs, naming_regime="anonymized")
    ges_row = _find_row(full_grid, system="GES", obs_n=obs, naming_regime="anonymized")

    if pc_row is None or ges_row is None:
        raise SystemExit(f"PC or GES not found at obs={obs} in sachs_full_grid evaluation_summary.")

    pc_f1,  pc_shd  = _f1_shd(pc_row)
    ges_f1, ges_shd = _f1_shd(ges_row)

    # semantic floor: names-only CausalLLMPrompt (no data)
    floor_row = (_find_row(llm_base, system="CausalLLMPrompt", naming_regime="names_only")
                 or _find_row(full_grid, system="CausalLLMPrompt", naming_regime="names_only"))
    if floor_row is None:
        raise SystemExit("CausalLLMPrompt (names-only) row not found.")
    floor_f1, floor_shd = _f1_shd(floor_row)

    f1_refs  = {"PC": pc_f1,  "GES": ges_f1,  "Semantic floor": floor_f1}
    shd_refs = {"PC": pc_shd, "GES": ges_shd, "Semantic floor": floor_shd}

    # ── LLM-CD methods ────────────────────────────────────────────────────────
    # Prefer full_grid (single run, seed 42) at the requested obs; do NOT fall
    # back to a different obs — stale data in a labelled obs=N plot is misleading.
    sources = [
        ("JiralerspongBFS",      full_grid,  dict(obs_n=obs, naming_regime="real"),  False),
        ("TakayamaSCP",          full_grid,  dict(obs_n=obs, naming_regime="real"),  False),
        ("CausalLLMPrompt",      llm_base,   dict(naming_regime="names_only"),        False),
        ("JiralerspongPairwise", llm_base,   dict(obs_n=obs, naming_regime="real"),  True),
        ("CausalLLMData",        llm_base,   dict(obs_n=obs, naming_regime="real"),  True),
    ]

    methods: list[MethodRow] = []
    for name, pool, kw, obs_fallback in sources:
        row = _find_row(pool, system=name, **kw)
        if row is None:
            # Try obs=1000 fallback only for methods that don't have an obs-specific run,
            # and mark them so the plot can render a dagger (†) on the label.
            if obs_fallback and obs != 1000:
                row = _find_row(pool, system=name,
                                obs_n=1000, naming_regime=kw.get("naming_regime", "real"))
            if row is None:
                print(f"[warn] {name} not found at obs={obs} – skipping", file=sys.stderr)
                continue
            fallback = True
        else:
            fallback = False
        f1, shd = _f1_shd(row)
        kind = "floor" if name == "CausalLLMPrompt" else "llm"
        label = DISPLAY[name] + (" †" if fallback else "")
        methods.append(MethodRow(name=name, label=label, f1=f1, shd=shd, kind=kind))

    # Sort by F1 ascending so highest-F1 method plots at the top (y = n-1)
    methods.sort(key=lambda m: m.f1)
    return methods, f1_refs, shd_refs


# ── plotting ─────────────────────────────────────────────────────────────────

def _configure_matplotlib() -> None:
    if not os.getenv("MPLCONFIGDIR"):
        mpl_dir = Path("/tmp") / f"matplotlib_{os.getuid()}"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)


def plot(methods: list[MethodRow],
         f1_refs: dict[str, float],
         shd_refs: dict[str, float],
         *,
         obs: int,
         out_dir: Path,
         basename: str,
         formats: list[str]) -> list[Path]:

    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    n = len(methods)
    ys = list(range(n))
    labels = [m.label for m in methods]

    # ── reference line styles ─────────────────────────────────────────────────
    ref_styles = {
        "PC":             dict(color=COL_ANCHOR,   linestyle=(0, (4, 3)), linewidth=1.3),
        "GES":            dict(color=COL_ANCHOR,   linestyle=(0, (2, 2)), linewidth=1.3),
        "Semantic floor": dict(color=COL_SEMANTIC, linestyle=(0, (5, 3)), linewidth=1.6),
    }

    fig_h = max(3.2, 0.6 * n + 1.8)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, fig_h), sharey=True)
    fig.subplots_adjust(wspace=0.06, top=0.82, bottom=0.30)

    panels = [
        ("F1",  [m.f1  for m in methods], f1_refs,  None, True),
        ("SHD", [m.shd for m in methods], shd_refs, None, False),
    ]

    for ax, (metric, vals, refs, x_ceil, higher_better) in zip(axes, panels):
        data_max = max(max(vals), max(refs.values()))
        x_max = data_max * 1.18

        # reference lines — draw line then place label to the right of it,
        # staggered vertically so close lines (PC/GES/floor) never overlap
        ref_items = list(refs.items())
        for i, (ref_name, ref_val) in enumerate(ref_items):
            ax.axvline(ref_val, zorder=1, **ref_styles[ref_name])
            y_data = -0.1 - 0.38 * i          # below the lowest row, stacked downward
            ax.text(ref_val, y_data, ref_name,
                    ha="center", va="top", fontsize=7.2,
                    color=ref_styles[ref_name]["color"],
                    transform=ax.transData, clip_on=False)

        # lollipops
        for y, m, val in zip(ys, methods, vals):
            dot_color = COL_SEMANTIC if m.kind == "floor" else COL_LLM
            ax.plot([0, val], [y, y], color=COL_STEM, linewidth=1.4, zorder=2)
            ax.plot(val, y, "o", color=dot_color, markersize=8, zorder=3)
            offset = 0.018 * x_max
            ax.text(val + offset, y, f"{val:.2f}",
                    va="center", ha="left", fontsize=8.0, color=dot_color)

        ax.set_xlim(0, x_max * 1.2)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlabel(metric, fontsize=10.5)
        ax.set_yticks(ys)
        ax.grid(axis="x", color="#E0E0E0", linewidth=0.7, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#333333")
        ax.tick_params(axis="both", labelsize=9)

    # y-axis labels only on left panel
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].set_ylabel("")

    # footnote for fallback obs
    if any("†" in m.label for m in methods):
        fig.text(0.5, 0.155, "† obs=1000 run shown; obs=5000 not yet available.",
                 ha="center", va="top", fontsize=7.5, color="#555555",
                 style="italic")

    # ── legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=COL_LLM,     marker="o", markersize=7, linewidth=0,
               label="LLM-CD method"),
        Line2D([0], [0], color=COL_SEMANTIC, marker="o", markersize=7, linewidth=0,
               label="Semantic-only (names-only)"),
        Line2D([0], [0], color=ref_styles["PC"]["color"],
               linestyle=ref_styles["PC"]["linestyle"],
               linewidth=ref_styles["PC"]["linewidth"], label="PC"),
        Line2D([0], [0], color=ref_styles["GES"]["color"],
               linestyle=ref_styles["GES"]["linestyle"],
               linewidth=ref_styles["GES"]["linewidth"], label="GES"),
        Line2D([0], [0], color=ref_styles["Semantic floor"]["color"],
               linestyle=ref_styles["Semantic floor"]["linestyle"],
               linewidth=ref_styles["Semantic floor"]["linewidth"],
               label="Semantic floor"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=5, fontsize=8.0, frameon=True,
               framealpha=0.95, facecolor="white", edgecolor="#DDDDDD",
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(f"LLM-CD Methods — Sachs (obs={obs})",
                 fontsize=14)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        p = out_dir / f"{basename}.{fmt}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        written.append(p)
    plt.close(fig)
    return written


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--obs",      type=int,  default=1000,
                        help="Observational budget to plot (default: 1000)")
    parser.add_argument("--out-dir",  type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--basename", default=None,
                        help="Output file stem (default: llm_cd_audit_obs{obs})")
    parser.add_argument("--formats",  nargs="+", default=["pdf", "png"],
                        choices=["pdf", "png", "svg"])
    args = parser.parse_args()

    basename = args.basename or f"llm_cd_audit_obs{args.obs}"
    methods, f1_refs, shd_refs = load_data(args.obs)
    written = plot(methods, f1_refs, shd_refs,
                   obs=args.obs,
                   out_dir=args.out_dir,
                   basename=basename,
                   formats=args.formats)
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
