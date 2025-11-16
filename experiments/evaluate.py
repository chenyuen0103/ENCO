#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
import sys
sys.path.append("../")
from causal_graphs.graph_visualization import graph_to_image
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
def save_pred_vs_true_graph(A_true, A_pred, var_names, out_path):
    """
    Draws a side-by-side figure:

        left  = true graph
        right = predicted graph

    using a single Graphviz layout (like your cancer graph example),
    with the same visual style (grey nodes, black edges, etc.).
    """
    n = len(var_names)
    # Make sure A_true / A_pred are numpy arrays
    A_true = np.asarray(A_true)
    A_pred = np.asarray(A_pred)

    # --- Build DiGraphs with integer nodes 0..n-1 ---
    G_true = nx.DiGraph()
    G_pred = nx.DiGraph()
    G_true.add_nodes_from(range(n))
    G_pred.add_nodes_from(range(n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if A_true[i, j] == 1:
                G_true.add_edge(i, j)
            if A_pred[i, j] == 1:
                G_pred.add_edge(i, j)

    # --- One shared Graphviz layout (like your example) ---
    pos = graphviz_layout(G_true, prog="dot")

    # --- Figure + axes ---
    figsize = max(3, n ** 0.7)
    fig, axes = plt.subplots(1, 2, figsize=(2 * figsize, figsize))

    # Label dict: int node -> name
    labels = {i: var_names[i] for i in range(n)}

    draw_kwargs = dict(
        arrows=True,
        node_color="lightgrey",
        edgecolors="black",
        node_size=600,
        arrowstyle="-|>",
        arrowsize=16,
    )

    # Left: TRUE graph
    ax = axes[0]
    nx.draw(G_true, pos, ax=ax, with_labels=False, **draw_kwargs)
    nx.draw_networkx_labels(G_true, pos, labels=labels, font_weight="bold", ax=ax)
    ax.set_title("True graph")
    ax.set_axis_off()
    ax.margins(0.2)

    # Right: PREDICTED graph
    ax = axes[1]
    nx.draw(G_pred, pos, ax=ax, with_labels=False, **draw_kwargs)
    nx.draw_networkx_labels(G_pred, pos, labels=labels, font_weight="bold", ax=ax)
    ax.set_title("Predicted graph")
    ax.set_axis_off()
    ax.margins(0.2)

    plt.tight_layout(pad=1.0)
    fig.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close(fig)

def scan_given_edges_df(df: pd.DataFrame):
    """
    Look through the 'given_edges' column (if present).

    Returns:
        has_given_edges (0/1 flag),
        max_given_edge_count (max number of edges given in any row)
    """
    if "given_edges" not in df.columns:
        return 0, 0

    has_any = 0
    max_count = 0

    for raw in df["given_edges"].dropna():
        s = str(raw).strip()
        if not s:
            continue

        try:
            edges = json.loads(s)
        except json.JSONDecodeError:
            # try a light clean-up if quoting is weird
            cleaned = s.replace("''", '"').replace('""', '"')
            try:
                edges = json.loads(cleaned)
            except json.JSONDecodeError:
                continue

        if isinstance(edges, list):
            has_any = 1
            max_count = max(max_count, len(edges))

    return has_any, max_count


def get_true_num_edges_from_answers(df: pd.DataFrame) -> int | None:
    """
    Use the first non-empty 'answer' JSON to infer the true number of edges
    by summing the adjacency_matrix entries.

    Supports both:
      - {"variables": [...], "adjacency_matrix": [[...], ...]}
      - [[...], [...], ...]  (plain list-of-lists)
    """
    if "answer" not in df.columns:
        return None

    for raw in df["answer"].dropna():
        s = str(raw).strip()
        if not s:
            continue
        try:
            ans_obj = json.loads(s)
        except json.JSONDecodeError:
            continue

        # Support both dict and bare matrix
        if isinstance(ans_obj, dict):
            mat = ans_obj.get("adjacency_matrix")
        else:
            mat = ans_obj

        try:
            A = np.asarray(mat)
        except Exception:
            continue

        if A.ndim == 2:
            return int(A.sum())

    return None



def extract_adjacency_matrix(text: str) -> Optional[np.ndarray]:
    """
    Robustly extract a square adjacency matrix from a messy LLM response.
    Returns a (N, N) numpy array of ints, or None.
    """

    if not text:
        return None

    # Helper: convert list-of-lists to square numpy array
    def _normalize_matrix(mat: Any) -> Optional[np.ndarray]:
        if not isinstance(mat, list) or not mat:
            return None
        try:
            rows: List[List[int]] = [[int(x) for x in row] for row in mat]
        except Exception:
            return None
        n = len(rows)
        if any(len(r) != n for r in rows):
            return None
        try:
            arr = np.asarray(rows, dtype=int)
        except Exception:
            return None
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            return None
        return arr

    def _from_obj(obj: Any) -> Optional[np.ndarray]:
        if isinstance(obj, dict) and "adjacency_matrix" in obj:
            return _normalize_matrix(obj["adjacency_matrix"])
        return _normalize_matrix(obj)

    def _try_one_variant(txt: str) -> Optional[np.ndarray]:
        txt = txt.strip()

        # 1) Whole-string JSON
        try:
            obj = json.loads(txt)
            m = _from_obj(obj)
            if m is not None:
                return m
        except Exception:
            pass

        # 2) Objects that start with "variables"
        for m in re.finditer(r'\{\s*"variables"\s*:[\s\S]*?\}', txt):
            frag = m.group(0)
            try:
                obj = json.loads(frag)
                mat = _from_obj(obj)
                if mat is not None:
                    return mat
            except Exception:
                continue

        # 3) Generic { ... } blocks
        for m in re.finditer(r"\{[\s\S]*?\}", txt):
            frag = m.group(0)
            try:
                obj = json.loads(frag)
                mat = _from_obj(obj)
                if mat is not None:
                    return mat
            except Exception:
                continue

        # 4) `"adjacency_matrix": [ ... ]` via bracket-balancing
        for m in re.finditer(r'"adjacency_matrix"\s*:', txt):
            start = m.end()
            lb = txt.find("[", start)
            if lb == -1:
                continue
            depth = 0
            for i in range(lb, len(txt)):
                ch = txt[i]
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        candidate = txt[lb:i+1]
                        try:
                            obj = json.loads(candidate)
                            mat = _normalize_matrix(obj)
                            if mat is not None:
                                return mat
                        except Exception:
                            pass
                        break  # done with this "adjacency_matrix" block

        # 5) Any [[...]]-style matrix
        for m in re.finditer(r"\[\s*\[", txt):
            lb = m.start()
            depth = 0
            seen_inner = False
            for i in range(lb, len(txt)):
                ch = txt[i]
                if ch == "[":
                    depth += 1
                    if depth > 1:
                        seen_inner = True
                elif ch == "]":
                    depth -= 1
                    if depth == 0 and seen_inner:
                        candidate = txt[lb:i+1]
                        try:
                            obj = json.loads(candidate)
                            mat = _normalize_matrix(obj)
                            if mat is not None:
                                return mat
                        except Exception:
                            pass
                        break

        return None

    # First try the text as-is
    mat = _try_one_variant(text)
    if mat is not None:
        return mat

    # If we have literal "\n" sequences (backslash+n) but few real newlines,
    # try again with "\n" converted to actual newlines.
    if "\\n" in text:
        alt = text.replace("\\n", "\n")
        if alt != text:
            mat = _try_one_variant(alt)
            if mat is not None:
                return mat

    return None


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

def save_prediction_graph(A_pred: np.ndarray,
                          var_names,
                          out_path: Path,
                          figsize=(3, 3)) -> None:
    """
    Turn an adjacency matrix + variable names into a PNG.

    A_pred: (N, N) numpy array with 0/1 entries
    var_names: list of length N
    out_path: path to save the image
    """
    G = nx.DiGraph()
    G.add_nodes_from(var_names)
    n = A_pred.shape[0]
    for i in range(n):
        for j in range(n):
            if A_pred[i, j] == 1:
                G.add_edge(var_names[i], var_names[j])

    graph_to_image(
        G,
        filename=str(out_path),
        show_plot=False,
        layout="graphviz",
        figsize=figsize,
    )


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
    # save 
    viz_dir = csv_path.parent / (csv_path.stem + "_pred_graphs")
    # Read entire CSV
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = list(reader)
        orig_fieldnames = list(reader.fieldnames or [])
        # --------- Pre-pass: recompute prediction & valid from raw_response ---------
        RAW_COL = "raw_response"  # adjust if your column name differs

        if RAW_COL in orig_fieldnames:
            for row_idx, row in enumerate(rows):
                raw = row.get(RAW_COL, "") or ""
                mat = extract_adjacency_matrix(raw)

                if mat is not None:
                    # store as JSON list-of-lists
                    row[args.pred_col] = json.dumps(mat.tolist(), ensure_ascii=False)
                    row["valid"] = 1
                else:
                    row[args.pred_col] = ""
                    row["valid"] = 0
        else:
            # no raw_response column; don't touch prediction/valid
            pass

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
    pred_edges_list: List[int] = []
    for row_idx, row in enumerate(rows):
        ans_s = row.get(args.answer_col, "") or ""
        pred_s = row.get(args.pred_col, "") or ""

        A_true = extract_adjacency_matrix(ans_s)
        A_pred = extract_adjacency_matrix(pred_s)

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
            # ===== NEW: count predicted edges (off-diagonal) =====
            # (A_pred is 0/1; we exclude self-loops / diagonal just in case)
            n = A_pred.shape[0]
            mask_offdiag = ~np.eye(n, dtype=bool)
            num_pred_edges = int(A_pred[mask_offdiag].sum())
            pred_edges_list.append(num_pred_edges)

            # ---------- Optional: visualize this prediction ----------
            var_names = None
            try:
                ans_obj = json.loads(ans_s)
                if isinstance(ans_obj, dict) and "variables" in ans_obj:
                    var_names = ans_obj["variables"]
            except Exception:
                pass

            if var_names is None:
                # Fallback: X1, X2, ..., Xn
                n_vars = A_true.shape[0]
                var_names = [f"X{i+1}" for i in range(n_vars)]

            # Same directory as CSV, same base name + "_row{idx}_pred_graph.pdf"
            base_no_suffix = csv_path.with_suffix("")  # drop ".csv"
            fig_name = f"{base_no_suffix.name}_row{row_idx}_pred_graph.pdf"
            fig_path = base_no_suffix.with_name(fig_name)

            save_pred_vs_true_graph(A_true, A_pred, var_names, fig_path)
        # Attach identifiers if present
        for k in ("data_idx", "shuffle_idx"):
            if k in row:
                met[k] = row[k]

        per_row_metrics.append(met)
    df = pd.DataFrame(rows)
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
        avg_pred_edges = float(sum(pred_edges_list) / valid_rows)
    else:
        avg_TP = avg_TN = avg_FP = avg_FN = avg_SHD = 0.0
        avg_accuracy = avg_precision = avg_recall = avg_f1 = None
        avg_orient_eval = avg_orient_TP = avg_orient_FN = 0.0
        avg_orient_acc = None
        avg_pred_edges = None

    has_given_edges, given_edge_count = scan_given_edges_df(df)
    true_num_edges = get_true_num_edges_from_answers(df)

    if true_num_edges and true_num_edges > 0 and given_edge_count > 0:
        given_edge_frac = given_edge_count / float(true_num_edges)
    else:
        given_edge_frac = None
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
        "num_pred_edges": avg_pred_edges,           # <-- NEW COLUMN
        "true_num_edges": int(true_num_edges) if true_num_edges is not None else None,
        "given_edges": int(has_given_edges),         # 0/1 flag
        "given_edge_count": int(given_edge_count),   # absolute number of edges given
        "given_edge_frac": (
            float(given_edge_frac) if given_edge_frac is not None else None
        ),
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
