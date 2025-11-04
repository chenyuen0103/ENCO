#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def extract_adj_from_json(s: str) -> Optional[np.ndarray]:
    """
    Parse an adjacency matrix from a JSON string.

    Accepts either:
      - {"variables": [...], "adjacency_matrix": [[...], ...]}
      - or just [[...], ...] directly.

    Returns an (N,N) np.ndarray of ints or None on failure.
    """
    if not s:
        return None
    try:
        obj = json.loads(s)
    except Exception:
        return None

    if isinstance(obj, dict) and "adjacency_matrix" in obj:
        mat = obj["adjacency_matrix"]
    else:
        mat = obj

    try:
        arr = np.asarray(mat, dtype=int)
    except Exception:
        return None

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return None
    return arr


def eval_pair(
    A_true: np.ndarray,
    A_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute per-row metrics between two directed adjacency matrices.

    - A_true, A_pred: (N,N) binary (0/1) numpy arrays.

    Metrics:
      * tp, tn, fp, fn (on directed edges, excluding diagonal)
      * accuracy, precision, recall, f1
      * shd: fp + fn (directed SHD)
      * orientation_*: orientation evaluation on correctly-predicted skeletons.
    """
    if A_true.shape != A_pred.shape:
        raise ValueError(f"Shape mismatch: true {A_true.shape}, pred {A_pred.shape}")

    n = A_true.shape[0]
    mask_offdiag = ~np.eye(n, dtype=bool)

    t = (A_true == 1)
    p = (A_pred == 1)

    tp = int(np.sum(t & p & mask_offdiag))
    tn = int(np.sum((~t) & (~p) & mask_offdiag))
    fp = int(np.sum((~t) & p & mask_offdiag))
    fn = int(np.sum(t & (~p) & mask_offdiag))

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else None
    prec = tp / (tp + fp) if (tp + fp) > 0 else None
    rec = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None

    # Directed SHD = #false positives + #false negatives
    shd = fp + fn

    # Orientation metric
    orient_eval = 0
    orient_tp = 0
    orient_fn = 0

    # Skeletons (undirected presence)
    skel_true = ((A_true + A_true.T) > 0)
    skel_pred = ((A_pred + A_pred.T) > 0)

    for i in range(n):
        for j in range(i + 1, n):
            gt_skel = skel_true[i, j]
            pr_skel = skel_pred[i, j]
            if not (gt_skel and pr_skel):
                continue

            gt_ij = A_true[i, j]
            gt_ji = A_true[j, i]
            pr_ij = A_pred[i, j]
            pr_ji = A_pred[j, i]

            # Only evaluate orientation if both GT and prediction have exactly one direction
            if (gt_ij + gt_ji) == 1 and (pr_ij + pr_ji) == 1:
                orient_eval += 1
                if (gt_ij == 1 and pr_ij == 1) or (gt_ji == 1 and pr_ji == 1):
                    orient_tp += 1
                else:
                    orient_fn += 1

    orient_acc = (orient_tp / orient_eval) if orient_eval > 0 else None

    return {
        "n_vars": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "shd": shd,
        "orient_eval_pairs": orient_eval,
        "orient_tp": orient_tp,
        "orient_fn": orient_fn,
        "orient_acc": orient_acc,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate adjacency predictions vs ground truth (including orientation metric)."
    )
    ap.add_argument(
        "--csv",
        required=True,
        help="CSV with columns 'answer' (ground truth JSON) and 'prediction' (predicted adjacency JSON).",
    )
    ap.add_argument(
        "--answer-col",
        default="answer",
        help="Column containing ground-truth graph JSON (default: 'answer').",
    )
    ap.add_argument(
        "--pred-col",
        default="prediction",
        help="Column containing predicted adjacency JSON (default: 'prediction').",
    )
    ap.add_argument(
        "--per-row-out",
        default=None,
        help=(
            "Optional path to write per-row metrics CSV. "
            "If omitted, metrics are appended as new columns to the *same* CSV."
        ),
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    # Read entire CSV
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = list(reader)
        orig_fieldnames = list(reader.fieldnames or [])

    # Per-row metrics & global aggregates
    per_row_metrics: List[Dict[str, Any]] = []

    # Lists to accumulate per-row metrics for averaging
    tp_list: List[int] = []
    tn_list: List[int] = []
    fp_list: List[int] = []
    fn_list: List[int] = []
    shd_list: List[int] = []

    acc_list: List[float] = []
    prec_list: List[Optional[float]] = []
    rec_list: List[Optional[float]] = []
    f1_list: List[Optional[float]] = []

    orient_eval_list: List[int] = []
    orient_tp_list: List[int] = []
    orient_fn_list: List[int] = []
    orient_acc_list: List[Optional[float]] = []

    valid_rows = 0

    for row in rows:
        ans_s = row.get(args.answer_col, "") or ""
        pred_s = row.get(args.pred_col, "") or ""

        A_true = extract_adj_from_json(ans_s)
        A_pred = extract_adj_from_json(pred_s)

        if A_true is None or A_pred is None:
            # No metrics for this row
            met = {
                "n_vars": None,
                "tp": None,
                "tn": None,
                "fp": None,
                "fn": None,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "shd": None,
                "orient_eval_pairs": None,
                "orient_tp": None,
                "orient_fn": None,
                "orient_acc": None,
            }
        else:
            met = eval_pair(A_true, A_pred)
            valid_rows += 1

            tp_list.append(met["tp"])
            tn_list.append(met["tn"])
            fp_list.append(met["fp"])
            fn_list.append(met["fn"])
            shd_list.append(met["shd"])

            acc_list.append(met["accuracy"])
            prec_list.append(met["precision"])
            rec_list.append(met["recall"])
            f1_list.append(met["f1"])

            orient_eval_list.append(met["orient_eval_pairs"])
            orient_tp_list.append(met["orient_tp"])
            orient_fn_list.append(met["orient_fn"])
            orient_acc_list.append(met["orient_acc"])

        # Attach identifiers if present
        for k in ("data_idx", "shuffle_idx"):
            if k in row:
                met[k] = row[k]

        per_row_metrics.append(met)

    # --------- Global summary as averages over valid rows ---------
    if valid_rows > 0:
        def _mean_or_none(xs: List[Optional[float]]) -> Optional[float]:
            vals = [x for x in xs if x is not None]
            return float(sum(vals) / len(vals)) if vals else None

        avg_TP  = float(sum(tp_list) / valid_rows)
        avg_TN  = float(sum(tn_list) / valid_rows)
        avg_FP  = float(sum(fp_list) / valid_rows)
        avg_FN  = float(sum(fn_list) / valid_rows)
        avg_SHD = float(sum(shd_list) / valid_rows)

        avg_accuracy  = float(sum(acc_list) / valid_rows)
        avg_precision = _mean_or_none(prec_list)
        avg_recall    = _mean_or_none(rec_list)
        avg_f1        = _mean_or_none(f1_list)

        avg_orient_eval = float(sum(orient_eval_list) / valid_rows)
        avg_orient_TP   = float(sum(orient_tp_list) / valid_rows)
        avg_orient_FN   = float(sum(orient_fn_list) / valid_rows)
        avg_orient_acc  = _mean_or_none(orient_acc_list)
    else:
        avg_TP = avg_TN = avg_FP = avg_FN = avg_SHD = 0.0
        avg_accuracy = avg_precision = avg_recall = avg_f1 = None
        avg_orient_eval = avg_orient_TP = avg_orient_FN = 0.0
        avg_orient_acc = None

    summary = {
        "num_rows": len(rows),
        "valid_rows": valid_rows,
        "avg_TP": avg_TP,
        "avg_TN": avg_TN,
        "avg_FP": avg_FP,
        "avg_FN": avg_FN,
        "avg_accuracy": avg_accuracy,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "avg_shd": avg_SHD,
        "avg_orientation_eval_pairs": avg_orient_eval,
        "avg_orientation_TP": avg_orient_TP,
        "avg_orientation_FN": avg_orient_FN,
        "avg_orientation_accuracy": avg_orient_acc,
    }

    # Pretty-print summary to stdout
    print("=== Global metrics (averages per row) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # --------- Save summary JSON ---------
    summary_path = csv_path.with_suffix(csv_path.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2)
    print(f"\nSaved summary to: {summary_path}")

    # --------- Per-row output handling ---------

    metric_keys = [
        "n_vars",
        "tp",
        "tn",
        "fp",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "shd",
        "orient_eval_pairs",
        "orient_tp",
        "orient_fn",
        "orient_acc",
    ]

    if args.per_row_out:
        # 1) WRITE METRICS ONLY to separate CSV
        per_path = Path(args.per_row_out)
        # Build fieldnames: identifiers + metric keys
        per_fieldnames = []
        for k in ("data_idx", "shuffle_idx"):
            if any(k in m for m in per_row_metrics):
                per_fieldnames.append(k)
        per_fieldnames.extend(metric_keys)

        with per_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=per_fieldnames, extrasaction="ignore")
            writer.writeheader()
            for m in per_row_metrics:
                writer.writerow(m)

        print(f"Per-row metrics written to: {per_path}")

    else:
        # 2) APPEND METRICS TO THE SAME CSV (atomic replace)
        for row, m in zip(rows, per_row_metrics):
            for k in metric_keys:
                val = m.get(k, None)
                row[k] = "" if val is None else val

        fieldnames = orig_fieldnames[:]
        for k in metric_keys:
            if k not in fieldnames:
                fieldnames.append(k)

        tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        os.replace(tmp_path, csv_path)
        print(f"Metrics appended to original CSV: {csv_path}")

if __name__ == "__main__":
    main()
