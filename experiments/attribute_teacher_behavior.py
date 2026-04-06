#!/usr/bin/env python3
"""
Behaviorally attribute an LLM run to one or more teacher runs.

This script compares an evaluated LLM response CSV against one or more teacher CSVs
that contain graph predictions for the same instances. It focuses on two analyses:

1. Nearest-teacher matching: which teacher is most similar to the LLM per row.
2. Disagreement-set analysis: the same question, restricted to rows where the
   candidate teachers do not all make the same prediction.

Teacher CSVs are expected to use the same row granularity as the LLM CSV when
possible. Alignment is attempted in this order:
  - (data_idx, shuffle_idx)
  - data_idx (only if unique in the teacher CSV)
  - ground-truth graph hash parsed from answer/answer_path
  - single-row fallback (optional)

The tool reuses the existing parsing/evaluation helpers from experiments/evaluate.py,
so it works with the current response formats in this repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.evaluate import (  # noqa: E402
    eval_pair,
    extract_adjacency_from_response,
    extract_adjacency_matrix,
    load_gt_from_cell,
)


def _mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return float(sum(vals) / len(vals)) if vals else None


def _matrix_hash(mat: Optional[np.ndarray]) -> Optional[str]:
    if mat is None:
        return None
    arr = np.asarray(mat, dtype=np.int8)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _resolve_roots(csv_path: Path) -> list[Path]:
    roots: list[Path] = [csv_path.parent]
    try:
        if len(csv_path.parents) >= 2:
            roots.append(csv_path.parents[1])
        if len(csv_path.parents) >= 3:
            roots.append(csv_path.parents[2])
    except Exception:
        pass
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve())
        if key not in seen:
            out.append(root)
            seen.add(key)
    return out


def _answer_col(fieldnames: list[str], explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    for candidate in ("answer", "answer_path"):
        if candidate in fieldnames:
            return candidate
    return None


def _teacher_name_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("responses_"):
        return stem.removeprefix("responses_")
    if stem.startswith("predictions_"):
        return stem.removeprefix("predictions_")
    return stem


def _normalize_pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _skeleton_edges(adj: np.ndarray) -> set[tuple[int, int]]:
    n = adj.shape[0]
    out: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] or adj[j, i]:
                out.add((i, j))
    return out


def _colliders(adj: np.ndarray) -> set[tuple[int, int, int]]:
    """
    Return colliders as (min_parent, child, max_parent) index triples.
    """
    n = adj.shape[0]
    out: set[tuple[int, int, int]] = set()
    for child in range(n):
        parents = [src for src in range(n) if src != child and adj[src, child] == 1]
        if len(parents) < 2:
            continue
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                a = parents[i]
                b = parents[j]
                # Collider requires the parents to be non-adjacent.
                if adj[a, b] or adj[b, a]:
                    continue
                x, z = _normalize_pair(a, b)
                out.add((x, child, z))
    return out


def _set_prf(pred: set[Any], true: set[Any]) -> dict[str, Optional[float]]:
    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else None
    rec = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2.0 * prec * rec / (prec + rec)
        if prec is not None and rec is not None and (prec + rec) > 0
        else None
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def pair_metrics(a_ref: np.ndarray, a_pred: np.ndarray) -> dict[str, Optional[float]]:
    """
    Compare a_pred to a_ref.

    `a_ref` is the teacher graph when used for attribution scoring, so the
    precision/recall style metrics are directional with the teacher as reference.
    """
    basic = eval_pair(a_ref, a_pred)
    skel = _set_prf(_skeleton_edges(a_pred), _skeleton_edges(a_ref))
    col = _set_prf(_colliders(a_pred), _colliders(a_ref))

    return {
        "pair_shd": float(basic["shd"]),
        "pair_edge_precision": basic["precision"],
        "pair_edge_recall": basic["recall"],
        "pair_edge_f1": basic["f1"],
        "pair_orient_acc": basic["orient_acc"],
        "pair_skeleton_precision": skel["precision"],
        "pair_skeleton_recall": skel["recall"],
        "pair_skeleton_f1": skel["f1"],
        "pair_collider_precision": col["precision"],
        "pair_collider_recall": col["recall"],
        "pair_collider_f1": col["f1"],
    }


def _gt_metrics(gt: Optional[np.ndarray], pred: Optional[np.ndarray], prefix: str) -> dict[str, Optional[float]]:
    if gt is None or pred is None or gt.shape != pred.shape:
        return {
            f"{prefix}_vs_gt_f1": None,
            f"{prefix}_vs_gt_shd": None,
            f"{prefix}_vs_gt_orient_acc": None,
        }
    met = eval_pair(gt, pred)
    return {
        f"{prefix}_vs_gt_f1": met["f1"],
        f"{prefix}_vs_gt_shd": float(met["shd"]),
        f"{prefix}_vs_gt_orient_acc": met["orient_acc"],
    }


@dataclass(frozen=True)
class RunRow:
    row_idx: int
    data_idx: Optional[str]
    shuffle_idx: Optional[str]
    gt_hash: Optional[str]
    prediction: Optional[np.ndarray]
    gt: Optional[np.ndarray]


@dataclass
class RunTable:
    name: str
    path: Path
    rows: list[RunRow]
    by_pair: dict[tuple[str, str], list[RunRow]]
    by_data: dict[str, list[RunRow]]
    by_gt_hash: dict[str, list[RunRow]]


def _load_run_table(
    path: Path,
    *,
    name: Optional[str],
    pred_col: str,
    answer_col: Optional[str],
) -> RunTable:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    try:
        csv.field_size_limit(10_000_000)
    except OverflowError:
        csv.field_size_limit(1_000_000)

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows_in = list(reader)
        fieldnames = list(reader.fieldnames or [])

    ans_col = _answer_col(fieldnames, answer_col)
    resolve_roots = _resolve_roots(path)
    loaded_rows: list[RunRow] = []
    by_pair: dict[tuple[str, str], list[RunRow]] = defaultdict(list)
    by_data: dict[str, list[RunRow]] = defaultdict(list)
    by_gt_hash: dict[str, list[RunRow]] = defaultdict(list)

    for row_idx, raw in enumerate(rows_in):
        data_idx = (raw.get("data_idx") or "").strip() or None
        shuffle_idx = (raw.get("shuffle_idx") or "").strip() or None

        gt = None
        if ans_col is not None:
            gt_cell = raw.get(ans_col, "") or ""
            gt, fallback_variables = load_gt_from_cell(gt_cell, resolve_roots=resolve_roots)
        else:
            fallback_variables = None

        pred = None
        pred_s = (raw.get(pred_col) or "").strip()
        if pred_s:
            pred = extract_adjacency_matrix(pred_s, fallback_variables=fallback_variables)
        if pred is None:
            raw_s = raw.get("raw_response", "") or ""
            if raw_s:
                pred = extract_adjacency_from_response(raw_s, fallback_variables=fallback_variables)

        gt_hash = _matrix_hash(gt)
        run_row = RunRow(
            row_idx=row_idx,
            data_idx=data_idx,
            shuffle_idx=shuffle_idx,
            gt_hash=gt_hash,
            prediction=pred,
            gt=gt,
        )
        loaded_rows.append(run_row)

        if data_idx is not None and shuffle_idx is not None:
            by_pair[(data_idx, shuffle_idx)].append(run_row)
        if data_idx is not None:
            by_data[data_idx].append(run_row)
        if gt_hash is not None:
            by_gt_hash[gt_hash].append(run_row)

    return RunTable(
        name=name or _teacher_name_from_path(path),
        path=path,
        rows=loaded_rows,
        by_pair=dict(by_pair),
        by_data=dict(by_data),
        by_gt_hash=dict(by_gt_hash),
    )


def _align_row(
    llm_row: RunRow,
    teacher: RunTable,
    *,
    allow_single_row_fallback: bool,
) -> tuple[Optional[RunRow], Optional[str]]:
    if llm_row.data_idx is not None and llm_row.shuffle_idx is not None:
        matches = teacher.by_pair.get((llm_row.data_idx, llm_row.shuffle_idx), [])
        if len(matches) == 1:
            return matches[0], "data_idx+shuffle_idx"

    if llm_row.data_idx is not None:
        matches = teacher.by_data.get(llm_row.data_idx, [])
        if len(matches) == 1:
            return matches[0], "data_idx"

    if llm_row.gt_hash is not None:
        matches = teacher.by_gt_hash.get(llm_row.gt_hash, [])
        if len(matches) == 1:
            return matches[0], "gt_hash"

    if allow_single_row_fallback and len(teacher.rows) == 1:
        return teacher.rows[0], "single_row_fallback"

    return None, None


def _metric_value(metric_name: str, record: dict[str, Optional[float]]) -> Optional[float]:
    val = record.get(metric_name)
    return None if val is None else float(val)


def _pick_winners(metric_name: str, rows: list[dict[str, Any]]) -> list[str]:
    scored: list[tuple[str, float]] = []
    for row in rows:
        value = _metric_value(metric_name, row)
        if value is None or math.isnan(value):
            continue
        scored.append((str(row["teacher"]), value))
    if not scored:
        return []

    if metric_name == "pair_shd":
        best = min(v for _, v in scored)
        return sorted(name for name, v in scored if v == best)

    best = max(v for _, v in scored)
    return sorted(name for name, v in scored if v == best)


def _count_fractional(counter: dict[str, float], winners: list[str]) -> None:
    if not winners:
        return
    weight = 1.0 / len(winners)
    for winner in winners:
        counter[winner] = counter.get(winner, 0.0) + weight


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Attribute LLM graph outputs to one or more teacher runs using pairwise graph similarity."
    )
    ap.add_argument("--llm-csv", required=True, help="Evaluated LLM response CSV.")
    ap.add_argument(
        "--teacher-csv",
        action="append",
        required=True,
        help="Teacher CSV. Pass multiple times for multiple teachers.",
    )
    ap.add_argument(
        "--teacher-name",
        action="append",
        default=[],
        help="Optional display name for a teacher CSV. Order must match --teacher-csv.",
    )
    ap.add_argument("--pred-col", default="prediction", help="Prediction column shared by all CSVs.")
    ap.add_argument(
        "--answer-col",
        default=None,
        help="Optional answer/answer_path column override shared by all CSVs.",
    )
    ap.add_argument(
        "--metric",
        default="pair_shd",
        choices=[
            "pair_shd",
            "pair_edge_f1",
            "pair_skeleton_f1",
            "pair_orient_acc",
            "pair_collider_f1",
        ],
        help="Primary attribution metric used for nearest-teacher matching.",
    )
    ap.add_argument(
        "--out-prefix",
        default=None,
        help="Output prefix. Defaults to <llm_csv>.teacher_attr in the same directory.",
    )
    ap.add_argument(
        "--allow-single-row-fallback",
        action="store_true",
        help="Allow a single teacher row to align to every LLM row when no better key is available.",
    )
    args = ap.parse_args()

    llm_path = Path(args.llm_csv)
    teacher_paths = [Path(p) for p in args.teacher_csv]
    teacher_names = list(args.teacher_name or [])
    if teacher_names and len(teacher_names) != len(teacher_paths):
        raise SystemExit("--teacher-name must be omitted or passed once per --teacher-csv.")

    llm = _load_run_table(
        llm_path,
        name="llm",
        pred_col=args.pred_col,
        answer_col=args.answer_col,
    )
    teachers = [
        _load_run_table(
            path,
            name=(teacher_names[i] if i < len(teacher_names) else None),
            pred_col=args.pred_col,
            answer_col=args.answer_col,
        )
        for i, path in enumerate(teacher_paths)
    ]

    out_prefix = (
        Path(args.out_prefix)
        if args.out_prefix
        else llm_path.with_suffix(llm_path.suffix + ".teacher_attr")
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pair_rows: list[dict[str, Any]] = []
    row_summaries: list[dict[str, Any]] = []
    alignment_stats: dict[str, Counter[str]] = {teacher.name: Counter() for teacher in teachers}
    teacher_metric_rows: dict[str, list[dict[str, Any]]] = {teacher.name: [] for teacher in teachers}

    nearest_counts_all: dict[str, float] = {}
    nearest_counts_disagreement: dict[str, float] = {}
    rows_with_all_teachers = 0
    disagreement_rows = 0

    for llm_row in llm.rows:
        per_teacher_rows: list[dict[str, Any]] = []
        aligned_teacher_preds: list[np.ndarray] = []

        for teacher in teachers:
            teacher_row, matched_by = _align_row(
                llm_row,
                teacher,
                allow_single_row_fallback=args.allow_single_row_fallback,
            )
            if matched_by is None or teacher_row is None:
                alignment_stats[teacher.name]["unmatched"] += 1
                continue

            alignment_stats[teacher.name][matched_by] += 1
            if llm_row.prediction is None or teacher_row.prediction is None:
                metrics = {
                    "pair_shd": None,
                    "pair_edge_precision": None,
                    "pair_edge_recall": None,
                    "pair_edge_f1": None,
                    "pair_orient_acc": None,
                    "pair_skeleton_precision": None,
                    "pair_skeleton_recall": None,
                    "pair_skeleton_f1": None,
                    "pair_collider_precision": None,
                    "pair_collider_recall": None,
                    "pair_collider_f1": None,
                }
            elif llm_row.prediction.shape != teacher_row.prediction.shape:
                metrics = {
                    "pair_shd": None,
                    "pair_edge_precision": None,
                    "pair_edge_recall": None,
                    "pair_edge_f1": None,
                    "pair_orient_acc": None,
                    "pair_skeleton_precision": None,
                    "pair_skeleton_recall": None,
                    "pair_skeleton_f1": None,
                    "pair_collider_precision": None,
                    "pair_collider_recall": None,
                    "pair_collider_f1": None,
                }
            else:
                metrics = pair_metrics(teacher_row.prediction, llm_row.prediction)
                aligned_teacher_preds.append(teacher_row.prediction)

            record = {
                "llm_row_idx": llm_row.row_idx,
                "data_idx": llm_row.data_idx,
                "shuffle_idx": llm_row.shuffle_idx,
                "teacher": teacher.name,
                "teacher_row_idx": teacher_row.row_idx,
                "matched_by": matched_by,
                **metrics,
                **_gt_metrics(llm_row.gt, llm_row.prediction, "llm"),
                **_gt_metrics(llm_row.gt, teacher_row.prediction, "teacher"),
            }
            pair_rows.append(record)
            per_teacher_rows.append(record)
            teacher_metric_rows[teacher.name].append(record)

        full_row = len(per_teacher_rows) == len(teachers) and len(teachers) > 0
        winners = _pick_winners(args.metric, per_teacher_rows) if full_row else []
        if full_row:
            rows_with_all_teachers += 1
            _count_fractional(nearest_counts_all, winners)

        disagreement = False
        if full_row and aligned_teacher_preds:
            unique_teacher_preds = {_matrix_hash(pred) for pred in aligned_teacher_preds}
            disagreement = len(unique_teacher_preds) > 1
            if disagreement:
                disagreement_rows += 1
                _count_fractional(nearest_counts_disagreement, winners)

        row_summaries.append(
            {
                "llm_row_idx": llm_row.row_idx,
                "data_idx": llm_row.data_idx,
                "shuffle_idx": llm_row.shuffle_idx,
                "num_teachers_aligned": len(per_teacher_rows),
                "has_all_teachers": int(full_row),
                "teacher_disagreement": int(disagreement),
                "nearest_teachers": "|".join(winners),
                "nearest_tie_size": len(winners),
            }
        )

    rows_csv_path = Path(str(out_prefix) + ".rows.csv")
    with rows_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "llm_row_idx",
            "data_idx",
            "shuffle_idx",
            "teacher",
            "teacher_row_idx",
            "matched_by",
            "pair_shd",
            "pair_edge_precision",
            "pair_edge_recall",
            "pair_edge_f1",
            "pair_orient_acc",
            "pair_skeleton_precision",
            "pair_skeleton_recall",
            "pair_skeleton_f1",
            "pair_collider_precision",
            "pair_collider_recall",
            "pair_collider_f1",
            "llm_vs_gt_f1",
            "llm_vs_gt_shd",
            "llm_vs_gt_orient_acc",
            "teacher_vs_gt_f1",
            "teacher_vs_gt_shd",
            "teacher_vs_gt_orient_acc",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in pair_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    row_summary_path = Path(str(out_prefix) + ".row_summary.csv")
    with row_summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "llm_row_idx",
            "data_idx",
            "shuffle_idx",
            "num_teachers_aligned",
            "has_all_teachers",
            "teacher_disagreement",
            "nearest_teachers",
            "nearest_tie_size",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_summaries:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    teacher_means: dict[str, dict[str, Optional[float]]] = {}
    for teacher in teachers:
        records = teacher_metric_rows[teacher.name]
        teacher_means[teacher.name] = {
            "aligned_rows": float(len(records)),
            "mean_pair_shd": _mean(r.get("pair_shd") for r in records),
            "mean_pair_edge_f1": _mean(r.get("pair_edge_f1") for r in records),
            "mean_pair_skeleton_f1": _mean(r.get("pair_skeleton_f1") for r in records),
            "mean_pair_orient_acc": _mean(r.get("pair_orient_acc") for r in records),
            "mean_pair_collider_f1": _mean(r.get("pair_collider_f1") for r in records),
            "mean_teacher_vs_gt_f1": _mean(r.get("teacher_vs_gt_f1") for r in records),
            "mean_teacher_vs_gt_shd": _mean(r.get("teacher_vs_gt_shd") for r in records),
        }

    summary = {
        "llm_csv": str(llm_path),
        "teacher_csvs": {teacher.name: str(teacher.path) for teacher in teachers},
        "metric": args.metric,
        "allow_single_row_fallback": bool(args.allow_single_row_fallback),
        "n_llm_rows": len(llm.rows),
        "n_teachers": len(teachers),
        "rows_with_all_teachers": rows_with_all_teachers,
        "rows_with_teacher_disagreement": disagreement_rows,
        "teacher_alignment": {
            teacher.name: {
                "n_rows": len(teacher.rows),
                "strategy_counts": dict(alignment_stats[teacher.name]),
            }
            for teacher in teachers
        },
        "teacher_mean_metrics": teacher_means,
        "nearest_teacher": {
            "counts_all_rows": nearest_counts_all,
            "rates_all_rows": {
                name: (count / rows_with_all_teachers if rows_with_all_teachers > 0 else None)
                for name, count in nearest_counts_all.items()
            },
            "counts_disagreement_rows": nearest_counts_disagreement,
            "rates_disagreement_rows": {
                name: (count / disagreement_rows if disagreement_rows > 0 else None)
                for name, count in nearest_counts_disagreement.items()
            },
        },
        "outputs": {
            "pair_rows_csv": str(rows_csv_path),
            "row_summary_csv": str(row_summary_path),
        },
    }

    summary_path = Path(str(out_prefix) + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")

    print(f"[done] Wrote pair rows: {rows_csv_path}")
    print(f"[done] Wrote row summary: {row_summary_path}")
    print(f"[done] Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
