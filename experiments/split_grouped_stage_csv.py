#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_key(row: dict[str, str], group_cols: list[str]) -> Tuple[str, ...]:
    return tuple(str(row.get(col, "")) for col in group_cols)


def split_rows(
    *,
    rows: list[dict[str, str]],
    group_cols: list[str],
    eval_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    if not rows:
        raise ValueError("input CSV has no rows")
    if not (0.0 < eval_fraction < 1.0):
        raise ValueError("--eval-fraction must be between 0 and 1")

    groups: Dict[Tuple[str, ...], List[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row, group_cols)].append(row)

    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    eval_groups_n = max(1, int(round(len(group_keys) * eval_fraction)))
    eval_groups_n = min(eval_groups_n, len(group_keys) - 1)

    eval_group_keys = set(group_keys[:eval_groups_n])
    train_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []
    for key in group_keys:
        if key in eval_group_keys:
            eval_rows.extend(groups[key])
        else:
            train_rows.extend(groups[key])

    if not train_rows or not eval_rows:
        raise ValueError("split produced an empty train or eval partition")

    summary: dict[str, object] = {
        "rows_total": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "groups_total": len(group_keys),
        "train_groups": len(group_keys) - eval_groups_n,
        "eval_groups": eval_groups_n,
        "group_cols": group_cols,
        "seed": seed,
        "eval_fraction": eval_fraction,
    }
    return train_rows, eval_rows, summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Split a curriculum stage CSV into train/eval partitions by grouped keys to avoid leakage."
    )
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--eval-csv", type=Path, required=True)
    p.add_argument(
        "--group-col",
        action="append",
        default=[],
        help="Group key column. Repeatable. Default: data_idx and shuffle_idx.",
    )
    p.add_argument("--eval-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summary-json", type=Path, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    group_cols = list(args.group_col or ["data_idx", "shuffle_idx"])

    fieldnames, rows = _read_rows(args.input_csv.resolve())
    train_rows, eval_rows, summary = split_rows(
        rows=rows,
        group_cols=group_cols,
        eval_fraction=float(args.eval_fraction),
        seed=int(args.seed),
    )
    _write_rows(args.train_csv.resolve(), fieldnames, train_rows)
    _write_rows(args.eval_csv.resolve(), fieldnames, eval_rows)

    if args.summary_json is not None:
        args.summary_json.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.resolve().write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
