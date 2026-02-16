#!/usr/bin/env python3
"""
Backfill the `_p{num_prompts}` tag into legacy response filenames.

We infer `num_prompts` from the number of rows in the response CSV:
  num_prompts ~= (num_rows / shuffles_per_graph)

This script renames:
  - the response CSV itself: responses_obs*_int*_shuf*_..._MODEL.csv
  - sidecars that share the same prefix, e.g.:
      <csv>.summary.json
      <csv>.per_row.csv
      <csv>.consensus_tau0.70.json
      <stem>.probplot.pdf

Default is dry-run; pass --apply to actually rename.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_RESP_RE = re.compile(
    r"^responses_obs(?P<obs>\d+)_int(?P<int>\d+)_shuf(?P<shuf>\d+)(?P<tags>.*?)(?:_(?P<model>[^_]+))?$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class RenamePlan:
    src: Path
    dst: Path
    rows: int
    shuf: int
    inferred_p: int
    ok: bool
    reason: str


def _count_rows(csv_path: Path) -> int:
    n = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for _ in r:
            n += 1
    return n


def _infer_p(rows: int, shuf: int) -> tuple[int, bool, str]:
    if rows <= 0:
        return 0, False, "no rows"
    if shuf <= 0:
        return rows, False, "shuf<=0 (cannot normalize); using rows"
    if rows % shuf != 0:
        return rows, False, f"rows ({rows}) not divisible by shuf ({shuf}); using rows"
    return rows // shuf, True, "ok"


def _insert_p_tag(*, stem: str, p: int) -> str | None:
    m = _RESP_RE.match(stem)
    if not m:
        return None
    shuf = int(m.group("shuf"))
    tags = (m.group("tags") or "")
    model = (m.group("model") or "")
    if re.search(r"(?:^|_)p\d+(?:_|$)", tags, flags=re.IGNORECASE):
        return None
    base = f"responses_obs{m.group('obs')}_int{m.group('int')}_shuf{shuf}_p{int(p)}{tags}"
    if model:
        base += f"_{model}"
    return base


def _iter_candidate_csvs(roots: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("responses_obs*_int*_shuf*.csv")):
            # Exclude derived per-row metrics files (they end with ".per_row.csv" but also match "*.csv").
            if p.name.endswith(".per_row.csv"):
                continue
            out.append(p)
    # De-dup in case roots overlap
    uniq = {p.resolve(): p for p in out}
    return [uniq[k] for k in sorted(uniq.keys())]


def _plan_for_csv(csv_path: Path) -> RenamePlan:
    stem = csv_path.stem
    m = _RESP_RE.match(stem)
    if not m:
        return RenamePlan(csv_path, csv_path, 0, 0, 0, False, "stem does not match expected pattern")
    tags = (m.group("tags") or "")
    if re.search(r"(?:^|_)p\d+(?:_|$)", tags, flags=re.IGNORECASE):
        return RenamePlan(csv_path, csv_path, 0, int(m.group("shuf")), 0, False, "already has _p* tag")

    rows = _count_rows(csv_path)
    shuf = int(m.group("shuf"))
    p, ok, why = _infer_p(rows, shuf)
    new_stem = _insert_p_tag(stem=stem, p=p)
    if not new_stem:
        return RenamePlan(csv_path, csv_path, rows, shuf, p, False, "could not build new name")
    dst = csv_path.with_name(new_stem + csv_path.suffix)
    if dst.exists():
        return RenamePlan(csv_path, dst, rows, shuf, p, False, "target already exists")
    return RenamePlan(csv_path, dst, rows, shuf, p, ok, why)


def _rename_with_sidecars(plan: RenamePlan, *, apply: bool) -> None:
    src_csv = plan.src
    dst_csv = plan.dst

    src_stem = src_csv.stem
    dst_stem = dst_csv.stem

    src_csv_name = src_csv.name  # <stem>.csv
    dst_csv_name = dst_csv.name

    dirp = src_csv.parent
    items = sorted(dirp.iterdir())

    renames: list[tuple[Path, Path]] = []
    for p in items:
        name = p.name
        if name == src_csv_name:
            renames.append((p, dirp / dst_csv_name))
            continue
        if name.startswith(src_csv_name + "."):
            renames.append((p, dirp / (dst_csv_name + name[len(src_csv_name) :])))
            continue
        if name.startswith(src_stem + "."):
            renames.append((p, dirp / (dst_stem + name[len(src_stem) :])))
            continue

    # Safety: avoid collisions inside this directory.
    dsts = [b for _, b in renames]
    if len(set(dsts)) != len(dsts):
        raise RuntimeError(f"rename collision in {dirp}: {src_csv.name}")
    for _, b in renames:
        if b.exists():
            raise RuntimeError(f"target already exists: {b}")

    for a, b in renames:
        if apply:
            a.rename(b)
        else:
            print(f"[dry-run] {a} -> {b}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        nargs="*",
        default=["responses", "experiments/responses"],
        help="Root directories to search under (default: responses experiments/responses).",
    )
    ap.add_argument("--apply", action="store_true", help="Actually rename files.")
    ap.add_argument("--only-dataset", default=None, help="Optional dataset filter (e.g., sachs).")
    args = ap.parse_args()

    roots = [Path(r) for r in args.roots]
    csvs = _iter_candidate_csvs(roots)
    if args.only_dataset:
        d = str(args.only_dataset)
        csvs = [p for p in csvs if p.parent.name == d]

    plans = [_plan_for_csv(p) for p in csvs]
    todo = [pl for pl in plans if pl.src != pl.dst and pl.dst.suffix == ".csv"]

    if not todo:
        print("[info] No legacy response CSVs found that need a _p* tag.")
        return 0

    try:
        for pl in todo:
            status = "OK" if pl.ok else "WARN"
            print(f"[plan] {status} rows={pl.rows} shuf={pl.shuf} -> p={pl.inferred_p} ({pl.reason})")
            print(f"       {pl.src}")
            print(f"    -> {pl.dst}")
    except BrokenPipeError:
        # e.g., piping to `head`. Exit cleanly.
        try:
            sys.stdout.close()
        finally:
            return 0

    if not args.apply:
        print("\n[info] Dry-run only. Re-run with --apply to rename.")
        return 0

    for pl in todo:
        _rename_with_sidecars(pl, apply=True)

    print("[done] Renamed files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
