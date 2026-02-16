#!/usr/bin/env python3
"""
Generate a LaTeX table for an ENCO baseline sweep (table-range grid).

Expected inputs:
  experiments/responses/<dataset>/predictions_obs{N}_int{M}_ENCO.csv

Each CSV is expected to contain a single row with at least 'f1' and 'SHD' columns.
The ENCO sweep scripts in experiments/run_scripts/ create these files.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable


_PRED_RE = re.compile(r"^predictions_obs(?P<obs>\d+)_int(?P<int>\d+)_ENCO\.csv$")
_CKPT_RE = re.compile(r"^obs(?P<obs>\d+)_int(?P<int>\d+)_seed(?P<seed>\d+)$")


def _parse_int_list(values: list[str]) -> list[int] | None:
    if not values:
        return None
    out: list[int] = []
    for part in values:
        for tok in part.replace(",", " ").split():
            if tok:
                out.append(int(tok))
    return out


def _read_single_row_csv(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected 1 row in {path}, found {len(rows)}.")
    return rows[0]


def _discover_seed(checkpoints_dir: Path) -> int | None:
    if not checkpoints_dir.exists():
        return None
    seeds: set[int] = set()
    for child in checkpoints_dir.iterdir():
        if not child.is_dir():
            continue
        m = _CKPT_RE.match(child.name)
        if not m:
            continue
        seeds.add(int(m.group("seed")))
    if len(seeds) == 1:
        return next(iter(seeds))
    return None


def _fmt_cell(f1: float, shd: int) -> str:
    return f"{f1:.2f} ({shd})"


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _render_table(
    *,
    dataset: str,
    obs_sizes: Iterable[int],
    int_sizes: Iterable[int],
    values: dict[tuple[int, int], tuple[float, int]],
    seed: int | None,
    label: str,
) -> str:
    obs_sizes = list(obs_sizes)
    int_sizes = list(int_sizes)

    col_spec = "l" + ("c" * len(int_sizes))
    header_cells = [r"Obs $N$ / Int $M$"] + [str(m) for m in int_sizes]

    lines: list[str] = []
    lines.append(r"% Requires \usepackage{booktabs}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for n in obs_sizes:
        row = [str(n)]
        for m in int_sizes:
            v = values.get((n, m))
            row.append("--" if v is None else _fmt_cell(*v))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    seed_str = f", seed={seed}" if seed is not None else ""
    caption = (
        f"ENCO baseline (table-range sweep) on {_latex_escape(dataset)}{seed_str}. "
        r"Cells report F1 (higher is better) with SHD in parentheses (lower is better)."
    )
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sachs", help="Dataset name under experiments/responses/<dataset>/")
    p.add_argument(
        "--responses-dir",
        type=Path,
        default=None,
        help="Optional override for the responses directory (default: experiments/responses/<dataset>).",
    )
    p.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=None,
        help="Optional override for the checkpoints directory (default: experiments/checkpoints/enco_table_range/<dataset>).",
    )
    p.add_argument("--obs-sizes", nargs="*", default=[], help="Optional explicit list (space/comma separated).")
    p.add_argument("--int-sizes", nargs="*", default=[], help="Optional explicit list (space/comma separated).")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .tex path (default: experiments/out/baselines/<dataset>_enco_baseline_table.tex).",
    )
    p.add_argument("--label", default=None, help="LaTeX label (default: tab:<dataset>-enco-table-range).")
    args = p.parse_args()

    dataset = args.dataset
    responses_dir = args.responses_dir or Path("experiments") / "responses" / dataset
    checkpoints_dir = args.checkpoints_dir or Path("experiments") / "checkpoints" / "enco_table_range" / dataset
    out_path = args.out or Path("experiments") / "out" / "baselines" / f"{dataset}_enco_baseline_table.tex"
    label = args.label or f"tab:{dataset}-enco-table-range"

    values: dict[tuple[int, int], tuple[float, int]] = {}
    discovered_obs: set[int] = set()
    discovered_int: set[int] = set()

    for csv_path in sorted(responses_dir.glob("predictions_obs*_int*_ENCO.csv")):
        m = _PRED_RE.match(csv_path.name)
        if not m:
            continue
        obs_n = int(m.group("obs"))
        int_m = int(m.group("int"))
        row = _read_single_row_csv(csv_path)
        try:
            f1 = float(row["f1"])
            shd = int(float(row["SHD"]))
        except KeyError as e:
            raise KeyError(f"Missing expected column in {csv_path}: {e}") from e
        values[(obs_n, int_m)] = (f1, shd)
        discovered_obs.add(obs_n)
        discovered_int.add(int_m)

    if not values:
        raise FileNotFoundError(
            f"No ENCO predictions found under {responses_dir} (expected predictions_obs*_int*_ENCO.csv)."
        )

    obs_sizes = _parse_int_list(args.obs_sizes) or sorted(discovered_obs)
    int_sizes = _parse_int_list(args.int_sizes) or sorted(discovered_int)
    seed = _discover_seed(checkpoints_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tex = _render_table(
        dataset=dataset,
        obs_sizes=obs_sizes,
        int_sizes=int_sizes,
        values=values,
        seed=seed,
        label=label,
    )
    out_path.write_text(tex, encoding="utf-8")
    print(f"[done] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
