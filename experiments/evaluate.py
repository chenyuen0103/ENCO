#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from scipy.stats import binomtest

# If your project layout needs this:
import sys
sys.path.append("../")
from causal_graphs.graph_visualization import graph_to_image  # noqa: E402

# Optional: graphviz layout; guarded with try/except in _safe_layout
from networkx.drawing.nx_pydot import graphviz_layout  # type: ignore


# ------------------------ Utility: dispersion ------------------------ #

from matplotlib.patches import FancyArrowPatch
import numpy as np


import numpy as np

def nhd(G_true: np.ndarray, G_pred: np.ndarray, use_m2: bool = True) -> float:
    Gt = G_true.astype(int).copy()
    Gp = G_pred.astype(int).copy()
    m = Gt.shape[0]
    np.fill_diagonal(Gt, 0)
    np.fill_diagonal(Gp, 0)
    diff = (Gt != Gp).sum()
    denom = m*m if use_m2 else m*(m-1)
    return diff / denom

def nhd_baseline(G_true: np.ndarray, k_pred: int, use_m2: bool = True) -> float:
    Gt = G_true.astype(int).copy()
    m = Gt.shape[0]
    np.fill_diagonal(Gt, 0)
    e_true = int(Gt.sum())
    denom = m*m if use_m2 else m*(m-1)
    return (e_true + k_pred) / denom

def nhd_ratio(G_true: np.ndarray, G_pred: np.ndarray, use_m2: bool = True) -> float:
    nhd_val = nhd(G_true, G_pred, use_m2=use_m2)
    k_pred = int(G_pred.astype(int).sum())
    base = nhd_baseline(G_true, k_pred, use_m2=use_m2)
    return nhd_val / base if base > 0 else np.nan

def brier_edgewise(P: np.ndarray, Y: np.ndarray) -> float:
    """Directed-edge Brier over i!=j."""
    assert P.shape == Y.shape
    n = P.shape[0]
    mask = ~np.eye(n, dtype=bool)
    diff2 = (P - Y)**2
    return float(diff2[mask].mean())

def brier_skeleton(P: np.ndarray, Y: np.ndarray) -> float:
    """
    Unordered skeleton Brier:
      score = max(P_ij, P_ji)
      label = 1 iff Y_ij or Y_ji is 1
    Computed over i<j.
    """
    assert P.shape == Y.shape
    S = np.maximum(P, P.T)
    Y_skel = ((Y + Y.T) > 0).astype(float)
    iu = np.triu_indices_from(S, k=1)  # i < j
    return float(((S[iu] - Y_skel[iu])**2).mean())



def _perp_unit(a, b, eps=1e-9):
    """Unit vector perpendicular to segment a->b (rotate (dx,dy) by 90°)."""
    ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    n = np.hypot(dx, dy)
    if n < eps:
        # nodes at same/super-close position -> arbitrary perpendicular
        return np.array([0.0, 1.0])
    # perpendicular (−dy, dx) and normalize
    return np.array([-dy, dx]) / n

def _offset_segment(a, b, delta):
    """Shift both endpoints by 'delta' along a perpendicular to get a parallel segment."""
    off = _perp_unit(a, b) * delta
    return (a + off, b + off)

def _draw_straight_arrow(ax, p0, p1, width=1.5, arrowsize=16, color="k", z=2):
    """Draw a straight arrow from p0 to p1 (both np.array([x,y]))."""
    patch = FancyArrowPatch(
        posA=(p0[0], p0[1]),
        posB=(p1[0], p1[1]),
        arrowstyle="-|>",
        mutation_scale=arrowsize,
        linewidth=width,
        color=color,
        shrinkA=12, shrinkB=12,  # leave small gap at nodes
        clip_on=False,
        zorder=z,
    )
    ax.add_patch(patch)
def _draw_probability_graph(ax, pos, var_names, P, width_scale=5.0, delta=0.05):
    """
    Draw straight, non-overlapping arrows for both directions by parallel shifting.
    - If both i->j and j->i exist (P>0), i->j is shifted by +delta, j->i by -delta.
    - If only one direction exists, it is unshifted (delta=0) to keep the picture clean.
    - Edges with P == 0 are omitted.
    """
    n = len(var_names)
    # map node index -> coordinate (numpy array)
    coords = {i: np.array(pos[i], dtype=float) for i in range(n)}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pij = float(P[i, j])
            if pij <= 0.0:
                continue  # omit zeros, per your requirement

            # Are we in a bidirectional pair?
            has_back = (P[j, i] > 0.0)

            a = coords[i]; b = coords[j]
            if has_back:
                # opposite offsets for the two directions
                sign = +1.0  # this call is for (i->j); (j->i) will use -1.0 when its turn comes
                p0, p1 = _offset_segment(a, b, sign * delta)
                label_frac, label_dy = 0.4, +0.0  # keep symmetric; labels are on each displaced segment
            else:
                # single direction -> no offset
                p0, p1 = a, b
                label_frac, label_dy = 0.5, 0.0

            # width encodes probability; ensure a minimum width for visibility
            lw = max(0.6, width_scale * pij)

            _draw_straight_arrow(ax, p0, p1, arrowsize=16, color="k", z=2)
            _put_edge_label(ax, p0, p1, f"{pij:.2f}", frac=label_frac, dy=0.0, fs=9)

def _put_edge_label(ax, p0, p1, text, frac=0.5, dy=0.0, fs=9):
    """Place label at a point along segment p0->p1 with a small normal offset dy."""
    mid = p0 + frac * (p1 - p0)
    # push slightly off the segment to avoid sitting on the line
    n = _perp_unit(p0, p1)
    mid = mid + dy * n
    ax.text(mid[0], mid[1], text, fontsize=fs,
            ha="center", va="center",
            bbox=dict(fc="white", alpha=0.8, ec="none"))

def _mean_std_iqr_ci(xs):
    """Return mean, std, IQR, and 95% normal-approx CI for the mean (None if empty)."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, None, (None, None)
    xs = np.array(xs, dtype=float)
    m  = float(xs.mean())
    sd = float(xs.std(ddof=1)) if len(xs) > 1 else 0.0
    q25, q75 = np.percentile(xs, [25, 75])
    iqr = float(q75 - q25)
    z = 1.96
    se = sd / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    ci = (float(m - z*se), float(m + z*se))
    return m, sd, iqr, ci


# ------------------------ Stability / consensus ------------------------ #

def edge_stability(adj_preds: List[np.ndarray]):
    """
    adj_preds: list of (N,N) int arrays (0/1).
    Returns:
      P         : (N,N) selection probabilities
      Var       : (N,N) Bernoulli variances
      Entropy   : (N,N) entropies (nats)
      WilsonLow : (N,N) lower CI
      WilsonHigh: (N,N) upper CI
      K         : number of matrices aggregated
    """
    if not adj_preds:
        return None, None, None, None, None, 0
    A = np.stack(adj_preds, axis=0)           # (K,N,N)
    K, n, _ = A.shape
    P = A.mean(axis=0)                        # selection prob per edge
    Var = P * (1 - P)
    eps = 1e-12
    Ent = -(P*np.log(P+eps) + (1-P)*np.log(1-P+eps))
    WL = np.zeros_like(P)
    WH = np.zeros_like(P)
    S = A.sum(axis=0).astype(int)             # successes per edge
    for i in range(n):
        for j in range(n):
            s = int(S[i, j])
            ci = binomtest(s, K).proportion_ci(method="wilson")
            WL[i, j] = float(ci.low)
            WH[i, j] = float(ci.high)
    return P, Var, Ent, WL, WH, K


def consensus_from_P(P: np.ndarray, tau: float = 0.5) -> np.ndarray:
    return (P >= tau).astype(int)


def pairwise_shd_distribution(adj_preds: List[np.ndarray]) -> np.ndarray:
    mats = [a.astype(int) for a in adj_preds]
    dists = []
    for i in range(len(mats)):
        for j in range(i+1, len(mats)):
            dists.append(np.abs(mats[i] - mats[j]).sum())
    return np.array(dists, dtype=int)


# ------------------------ Plotting helpers ------------------------ #

def _safe_layout(G: nx.DiGraph):
    """Try Graphviz 'dot' layout; fall back to spring_layout if graphviz is unavailable."""
    try:
        return graphviz_layout(G, prog="dot")
    except Exception:
        return nx.spring_layout(G, seed=0)

def save_true_vs_prob_graph(A_true: np.ndarray,
                            P: np.ndarray,
                            var_names: List[str],
                            out_path: Path,
                            prob_thresh: float = 0.0,
                            show_probs: bool = True) -> None:
    """
    Left: true graph (binary edges).
    Right: probabilistic predicted graph drawn with straight, slightly offset arrows for
           both directions. Each drawn edge is labeled by P_ij. Any edge with P_ij <= prob_thresh
           is omitted. One PDF per CSV.
    """
    # --- basic checks ---
    A_true = np.asarray(A_true)
    P = np.asarray(P, dtype=float)
    n = len(var_names)
    assert A_true.shape == (n, n), f"A_true shape {A_true.shape} != ({n},{n})"
    assert P.shape == (n, n), f"P shape {P.shape} != ({n},{n})"

    # --- build graphs (left uses true graph; right uses only nodes; edges drawn manually) ---
    G_true = nx.DiGraph()
    G_true.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and A_true[i, j] == 1:
                G_true.add_edge(i, j)

    # Shared layout from the true graph for side-by-side comparability
    pos = _safe_layout(G_true)  # uses graphviz if available else spring_layout

    # --- figure/axes ---
    figsize = max(3, n ** 0.7)
    fig, axes = plt.subplots(1, 2, figsize=(2 * figsize, figsize))

    labels = {i: var_names[i] for i in range(n)}
    draw_nodes_kwargs = dict(node_color="lightgrey", edgecolors="black", node_size=600)

    # ---------- Left: TRUE ----------
    ax = axes[0]
    nx.draw(G_true, pos, ax=ax, with_labels=False, arrows=True,
            arrowstyle="-|>", arrowsize=16, **draw_nodes_kwargs)
    nx.draw_networkx_labels(G_true, pos, labels=labels, font_weight="bold", ax=ax)
    ax.set_title("True graph")
    ax.set_axis_off()
    ax.margins(0.2)

    # ---------- Right: probabilistic predicted ----------
    ax = axes[1]
    ax.set_title("Predicted")
    ax.set_axis_off()
    ax.margins(0.2)

    # Draw nodes first so custom arrows have visible endpoints
    nx.draw_networkx_nodes(G_true, pos, ax=ax, **draw_nodes_kwargs)
    nx.draw_networkx_labels(G_true, pos, labels=labels, font_weight="bold", ax=ax)

    # Apply threshold by masking P; keep diagonal at zero
    P_masked = P.copy()
    P_masked[P_masked <= prob_thresh] = 0.0
    np.fill_diagonal(P_masked, 0.0)

    # Draw straight, offset arrows for all i->j with P_ij > prob_thresh
    if show_probs:
        _draw_probability_graph(ax, pos, var_names, P_masked, width_scale=5.0, delta=0.06)
    else:
        # draw unlabeled arrows when probabilities are trivially 0/1 (e.g., single row)
        n = len(var_names)
        for i in range(n):
            for j in range(n):
                if i == j or P_masked[i, j] <= 0.0:
                    continue
                _draw_straight_arrow(ax, np.array(pos[i]), np.array(pos[j]), arrowsize=16, color="k", z=2)

    # --- save ---
    plt.tight_layout(pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close(fig)

def save_prediction_graph(A_pred: np.ndarray, var_names, out_path: Path, figsize=(3, 3)) -> None:
    """(Unused in this run; kept for compatibility.)"""
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


# ------------------------ CSV utilities ------------------------ #

def scan_given_edges_df(df: pd.DataFrame):
    if "given_edges" not in df.columns:
        return 0, 0
    has_any, max_count = 0, 0
    for raw in df["given_edges"].dropna():
        s = str(raw).strip()
        if not s:
            continue
        try:
            edges = json.loads(s)
        except json.JSONDecodeError:
            cleaned = s.replace("''", '"').replace('""', '"')
            try:
                edges = json.loads(cleaned)
            except json.JSONDecodeError:
                continue
        if isinstance(edges, list):
            has_any = 1
            max_count = max(max_count, len(edges))
    return has_any, max_count

def get_true_num_edges_from_answers(df: pd.DataFrame, answer_col: str = "answer") -> Optional[int]:
    """
    Determine true #edges from the ground-truth column, which may contain
    inline JSON (old) or a path to an answer JSON (new).
    """
    if answer_col not in df.columns:
        return None
    for raw in df[answer_col].dropna():
        A_true, _ = load_gt_from_cell(str(raw))
        if A_true is not None:
            return int(A_true.sum())
    return None

def _normalize_matrix_for_gt(mat: Any) -> Optional[np.ndarray]:
    """Normalize a matrix object (list-of-lists) to a square int np.ndarray or None."""
    if isinstance(mat, np.ndarray):
        if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
            try:
                return np.asarray(mat, dtype=int)
            except Exception:
                return None
        return None
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


def edges_to_adjacency(edges: Any, variables: Optional[List[str]]) -> Optional[np.ndarray]:
    """
    Convert an edge list (list of [src, dst]) into an adjacency matrix using the given variables order.
    """
    if variables is None or not isinstance(variables, list):
        return None
    if not isinstance(edges, list):
        return None
    try:
        var_to_idx = {v: i for i, v in enumerate(variables)}
        n = len(variables)
        A = np.zeros((n, n), dtype=int)
        for pair in edges:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            src, dst = pair
            if src in var_to_idx and dst in var_to_idx and src != dst:
                A[var_to_idx[src], var_to_idx[dst]] = 1
        return A
    except Exception:
        return None


def load_gt_from_cell(cell: str) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
    For a ground-truth cell, support:
      1) direct JSON (old format), e.g. {"variables": [...], "adjacency_matrix": [...]}
      2) path to a JSON file (new format), e.g. "..._answer.json" containing
         {"answer": {"variables": [...], "adjacency_matrix": [...]}, ...}
    Returns (A_true, variables) where variables may be None.
    """
    s = (cell or "").strip()
    if not s:
        return None, None

    obj = None

    # Try: direct JSON
    try:
        obj = json.loads(s)
    except Exception:
        obj = None

    # If not JSON, try: path to file
    if not isinstance(obj, (dict, list)):
        p = Path(s)
        if p.exists():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                obj = None

    if obj is None:
        return None, None

    variables = None
    mat_raw = None

    if isinstance(obj, dict):
        # Case 1: old style: top-level contains adjacency_matrix / edges
        if "adjacency_matrix" in obj or "edges" in obj:
            mat_raw = obj.get("adjacency_matrix", None)
            variables = obj.get("variables", None)
            if mat_raw is None and "edges" in obj:
                mat_raw = edges_to_adjacency(obj.get("edges"), variables)
        # Case 2: new style: {"answer": { "variables": ..., "adjacency_matrix": ... }, ...}
        elif "answer" in obj and isinstance(obj["answer"], dict):
            ans = obj["answer"]
            variables = ans.get("variables", None)
            mat_raw = ans.get("adjacency_matrix", None)
            if mat_raw is None and "edges" in ans:
                mat_raw = edges_to_adjacency(ans.get("edges"), variables)
        else:
            mat_raw = obj
    else:
        mat_raw = obj

    A_true = _normalize_matrix_for_gt(mat_raw)
    return A_true, variables


# ------------------------ Parsing predictions ------------------------ #

def extract_adjacency_matrix(text: str) -> Optional[np.ndarray]:
    if not text:
        return None

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
        if isinstance(obj, dict):
            if "adjacency_matrix" in obj:
                mat = _normalize_matrix(obj["adjacency_matrix"])
                if mat is not None:
                    return mat
            if "edges" in obj and "variables" in obj:
                return edges_to_adjacency(obj.get("edges"), obj.get("variables"))
        return _normalize_matrix(obj)

    def _try_one_variant(txt: str) -> Optional[np.ndarray]:
        txt = txt.strip()
        # whole-string JSON
        try:
            obj = json.loads(txt)
            m = _from_obj(obj)
            if m is not None:
                return m
        except Exception:
            pass
        # {"variables": ...} blocks
        for m in re.finditer(r'\{\s*"variables"\s*:[\s\S]*?\}', txt):
            frag = m.group(0)
            try:
                obj = json.loads(frag)
                mat = _from_obj(obj)
                if mat is not None:
                    return mat
            except Exception:
                continue
        # generic { ... } blocks
        for m in re.finditer(r"\{[\s\S]*?\}", txt):
            frag = m.group(0)
            try:
                obj = json.loads(frag)
                mat = _from_obj(obj)
                if mat is not None:
                    return mat
            except Exception:
                continue
        # "adjacency_matrix": [ ... ]
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
                        break
        # any [[ ... ]] block
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
    # breakpoint()
    mat = _try_one_variant(text)
    if mat is not None:
        return mat
    if "\\n" in text:
        alt = text.replace("\\n", "\n")
        if alt != text:
            mat = _try_one_variant(alt)
            if mat is not None:
                return mat
    return None


# ------------------------ Scoring ------------------------ #

def eval_pair(A_true: np.ndarray, A_pred: np.ndarray) -> Dict[str, Any]:
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
    rec  = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None

    shd = fp + fn

    # orientation on correctly predicted skeletons
    orient_eval = 0
    orient_tp = 0
    orient_fn = 0
    skel_true = ((A_true + A_true.T) > 0)
    skel_pred = ((A_pred + A_pred.T) > 0)
    for i in range(n):
        for j in range(i+1, n):
            if not (skel_true[i, j] and skel_pred[i, j]):
                continue
            gt_ij, gt_ji = A_true[i, j], A_true[j, i]
            pr_ij, pr_ji = A_pred[i, j], A_pred[j, i]
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


# ------------------------ Main ------------------------ #

def main():
    ap = argparse.ArgumentParser(description="Evaluate predictions, compute stability, and build a consensus graph.")
    ap.add_argument("--csv", required=True,
                    help="CSV with columns 'answer' (GT JSON) and 'prediction' (pred adjacency JSON) or 'raw_response'.")
    ap.add_argument("--answer-col", default=None,
                    help="Ground-truth JSON column. If not set, will try 'answer' then 'answer_path'.")
    ap.add_argument("--pred-col", default="prediction", help="Predicted adjacency JSON column.")
    ap.add_argument("--per-row-out", default=None,
                    help="If set, writes a separate CSV of per-row metrics; otherwise appends to input CSV.")
    # Consensus/stability controls & outputs
    ap.add_argument("--tau", type=float, default=0.7,
                    help="Consensus threshold; include edge if selection prob >= tau.")
    ap.add_argument("--consensus-json", default=None,
                    help="Path to save consensus artifact JSON (P, CIs, consensus adjacency). "
                         "Default: <csv>.consensus_tau{tau}.json")
    ap.add_argument("--save-edge-table", default=None,
                    help="Optional CSV with per-edge probabilities and Wilson CIs.")
    # NEW: single PDF with true vs probabilistic predicted edges
    ap.add_argument("--prob-plot", default=None,
                    help="Path to save the probability-labeled predicted graph vs true. "
                         "Default: <csv>.probplot.pdf")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    # Read all rows
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = list(reader)
        orig_fieldnames = list(reader.fieldnames or [])

    # Pick an answer column to use
    if args.answer_col:
        answer_col = args.answer_col
    else:
        # try both common names
        if "answer" in orig_fieldnames:
            answer_col = "answer"
        elif "answer_path" in orig_fieldnames:
            answer_col = "answer_path"
        else:
            answer_col = "answer_path"  # fall back; will likely yield None later
    print(f"[info] Using answer column: {answer_col}")

    # Optionally rebuild `prediction` from `raw_response`
    RAW_COL = "raw_response"
    if RAW_COL in orig_fieldnames:
        for row in rows:
            raw = row.get(RAW_COL, "") or ""
            mat = extract_adjacency_matrix(raw)
            if mat is not None:
                row[args.pred_col] = json.dumps(mat.tolist(), ensure_ascii=False)
                row["valid"] = 1
            else:
                row[args.pred_col] = ""
                row["valid"] = 0

    # --- Per-row metrics (existing flow) ---
    per_row_metrics: List[Dict[str, Any]] = []

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

    # Collect for consensus/probability plot
    pred_mats_all: List[np.ndarray] = []
    true_mats_all: List[np.ndarray] = []
    variables_first: Optional[List[str]] = None
    nhd_list: List[float] = []
    nhd_ratio_list: List[float] = []

    for row_idx, row in enumerate(rows):
        ans_s = row.get(answer_col, "") or ""
        pred_s = row.get(args.pred_col, "") or ""

        A_true, vars_from_this = load_gt_from_cell(ans_s)
        A_pred = extract_adjacency_matrix(pred_s)

        if variables_first is None and vars_from_this is not None:
            variables_first = vars_from_this

        if A_true is None or A_pred is None:
            met = {k: None for k in [
                "n_vars","tp","tn","fp","fn","accuracy","precision","recall",
                "f1","shd","orient_eval_pairs","orient_tp","orient_fn","orient_acc"
            ]}
        else:
            met = eval_pair(A_true, A_pred)
            valid_rows += 1
            # --- NHD & NHD ratio for this row ---
            nhd_val = nhd(A_true, A_pred, use_m2=True)
            n = A_pred.shape[0]
            off = ~np.eye(n, dtype=bool)
            k_pred = int(A_pred.astype(int)[off].sum())

            base = nhd_baseline(A_true, k_pred, use_m2=True)
            nhd_ratio_val = (nhd_val / base) if base > 0 else float("nan")

            nhd_list.append(nhd_val)
            nhd_ratio_list.append(nhd_ratio_val)

            # also store on this row so it can be written out
            met["nhd"] = nhd_val
            met["nhd_ratio"] = nhd_ratio_val

            tp_list.append(met["tp"]);   tn_list.append(met["tn"])
            fp_list.append(met["fp"]);   fn_list.append(met["fn"])
            shd_list.append(met["shd"])

            acc_list.append(met["accuracy"])
            prec_list.append(met["precision"])
            rec_list.append(met["recall"])
            f1_list.append(met["f1"])

            orient_eval_list.append(met["orient_eval_pairs"])
            orient_tp_list.append(met["orient_tp"])
            orient_fn_list.append(met["orient_fn"])
            orient_acc_list.append(met["orient_acc"])

            # number of predicted (off-diagonal) edges
            n = A_pred.shape[0]
            mask_offdiag = ~np.eye(n, dtype=bool)
            pred_edges_list.append(int(A_pred[mask_offdiag].sum()))

            # collect for probability plot / consensus
            pred_mats_all.append(A_pred.astype(int))
            true_mats_all.append(A_true.astype(int))

        for k in ("data_idx", "shuffle_idx"):
            if k in row:
                met[k] = row[k]
        per_row_metrics.append(met)

    df = pd.DataFrame(rows)

    # --- Global averages (existing flow) ---
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
        avg_pred_edges  = float(sum(pred_edges_list) / valid_rows)
    else:
        avg_TP = avg_TN = avg_FP = avg_FN = avg_SHD = 0.0
        avg_accuracy = avg_precision = avg_recall = avg_f1 = None
        avg_orient_eval = avg_orient_TP = avg_orient_FN = 0.0
        avg_orient_acc = None
        avg_pred_edges = None

    has_given_edges, given_edge_count = scan_given_edges_df(df)
    true_num_edges = get_true_num_edges_from_answers(df, answer_col=args.answer_col)
    if true_num_edges and true_num_edges > 0 and given_edge_count > 0:
        given_edge_frac = given_edge_count / float(true_num_edges)
    else:
        given_edge_frac = None

    # --- Build probabilities/consensus & single PDF ---
    A_true_ref = true_mats_all[0] if true_mats_all else None
    P = Var = Ent = WL = WH = None
    K = 0
    consensus_adj = None
    consensus_num_edges = None
    consensus_metrics = None
    nhd_consensus = None
    nhd_ratio_consensus = None
    brier = None
    brier_skel = None

    if pred_mats_all:
        P, Var, Ent, WL, WH, K = edge_stability(pred_mats_all)
        consensus_adj = consensus_from_P(P, tau=args.tau)
        n = consensus_adj.shape[0]
        consensus_num_edges = int(consensus_adj[~np.eye(n, dtype=bool)].sum())
        if A_true_ref is not None:
            consensus_metrics = eval_pair(A_true_ref, consensus_adj)

        # Consensus artifact JSON
        cj_path = (csv_path.with_suffix(csv_path.suffix + f".consensus_tau{args.tau:.2f}.json")
                   if args.consensus_json is None else Path(args.consensus_json))
        artifact = {
            "variables": (variables_first if variables_first is not None else
                          [f"X{i+1}" for i in range(P.shape[0])]),
            "tau": float(args.tau),
            "K": int(K),
            "adjacency_matrix_consensus": consensus_adj.tolist(),
            "P": P.tolist(),
            "wilson_low": WL.tolist(),
            "wilson_high": WH.tolist(),
        }
        with cj_path.open("w", encoding="utf-8") as f_out:
            json.dump(artifact, f_out, indent=2)
        print(f"Saved consensus artifact to: {cj_path}")

        # Optional: per-edge table
        if args.save_edge_table:
            vars_list = artifact["variables"]
            nvars = len(vars_list)
            rows_edges = []
            for i in range(nvars):
                for j in range(nvars):
                    if i == j:
                        continue
                    rows_edges.append({
                        "src_idx": i,
                        "dst_idx": j,
                        "src": vars_list[i],
                        "dst": vars_list[j],
                        "p_ij": float(P[i, j]),
                        "wilson_low": float(WL[i, j]),
                        "wilson_high": float(WH[i, j]),
                        "in_consensus": int(consensus_adj[i, j]),
                    })
            edge_df = pd.DataFrame(rows_edges)
            Path(args.save_edge_table).parent.mkdir(parents=True, exist_ok=True)
            edge_df.to_csv(args.save_edge_table, index=False)
            print(f"Saved per-edge table to: {args.save_edge_table}")

        # === ONE PDF per CSV: true vs probability-labeled predicted ===
        if A_true_ref is not None:
            var_names = artifact["variables"]
            prob_path = (csv_path.with_suffix(".probplot.pdf")
                         if args.prob_plot is None else Path(args.prob_plot))
            show_probs = (K is not None and K > 1)
            save_true_vs_prob_graph(A_true_ref, P, var_names, prob_path, show_probs=show_probs)
            print(f"Saved probability-labeled predicted graph: {prob_path}")
            # --- after you have P and A_true_ref ---
        # Compute Brier scores if we have probabilities and a GT reference
        if A_true_ref is not None:
            brier = brier_edgewise(P, A_true_ref)
            brier_skel = brier_skeleton(P, A_true_ref)

        # Compute consensus NHD / ratio now that consensus_adj exists
        if consensus_adj is not None and A_true_ref is not None:
            nhd_consensus = nhd(A_true_ref, consensus_adj, use_m2=True)
            off_cons = ~np.eye(consensus_adj.shape[0], dtype=bool)
            k_pred_cons = int(consensus_adj.astype(int)[off_cons].sum())
            base_cons = nhd_baseline(A_true_ref, k_pred_cons, use_m2=True)
            nhd_ratio_consensus = (nhd_consensus / base_cons) if base_cons > 0 else float("nan")



    # ---- Original summary keys (unchanged names) ----
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
        "num_pred_edges": avg_pred_edges,
        "true_num_edges": int(true_num_edges) if true_num_edges is not None else None,
        "given_edges": int(has_given_edges),
        "given_edge_count": int(given_edge_count),
        "given_edge_frac": (float(given_edge_frac) if given_edge_frac is not None else None),
    }
    summary.update({
        "brier": brier,
        "brier_skeleton": brier_skel,
    })
    if brier is not None:
        print(f"Brier (directed): {brier:.6f}")
    if brier_skel is not None:
        print(f"Brier (skeleton): {brier_skel:.6f}")

    # ---- Variability summaries (unique names) ----
    shd_mean, shd_sd, shd_iqr, shd_ci = _mean_std_iqr_ci(shd_list)
    acc_mean, acc_sd, acc_iqr, acc_ci = _mean_std_iqr_ci(acc_list)
    pre_mean, pre_sd, pre_iqr, pre_ci = _mean_std_iqr_ci([x for x in prec_list if x is not None])
    rec_mean, rec_sd, rec_iqr, rec_ci = _mean_std_iqr_ci([x for x in rec_list if x is not None])
    f1_mean,  f1_sd,  f1_iqr,  f1_ci  = _mean_std_iqr_ci([x for x in f1_list  if x is not None])
    edges_mean, edges_sd, edges_iqr, edges_ci = _mean_std_iqr_ci(pred_edges_list)

    summary.update({
        "var_shd_sd": shd_sd,
        "var_shd_iqr": shd_iqr,
        "var_shd_ci95_low": shd_ci[0],
        "var_shd_ci95_high": shd_ci[1],

        "var_accuracy_sd": acc_sd,
        "var_accuracy_iqr": acc_iqr,
        "var_accuracy_ci95_low": acc_ci[0],
        "var_accuracy_ci95_high": acc_ci[1],

        "var_precision_sd": pre_sd,
        "var_precision_iqr": pre_iqr,
        "var_precision_ci95_low": pre_ci[0],
        "var_precision_ci95_high": pre_ci[1],

        "var_recall_sd": rec_sd,
        "var_recall_iqr": rec_iqr,
        "var_recall_ci95_low": rec_ci[0],
        "var_recall_ci95_high": rec_ci[1],

        "var_f1_sd": f1_sd,
        "var_f1_iqr": f1_iqr,
        "var_f1_ci95_low": f1_ci[0],
        "var_f1_ci95_high": f1_ci[1],

        "var_num_pred_edges_sd": edges_sd,
        "var_num_pred_edges_iqr": edges_iqr,
        "var_num_pred_edges_ci95_low": edges_ci[0],
        "var_num_pred_edges_ci95_high": edges_ci[1],
    })

    # ---- Consensus summary (unique names; optional) ----
    if P is not None:
        summary.update({
            "consensus_tau": float(args.tau),
            "consensus_K": int(K),
            "consensus_num_edges": int(consensus_num_edges),
        })
        if consensus_metrics is not None:
            summary.update({
                "consensus_accuracy": consensus_metrics["accuracy"],
                "consensus_precision": consensus_metrics["precision"],
                "consensus_recall": consensus_metrics["recall"],
                "consensus_f1": consensus_metrics["f1"],
                "consensus_shd": consensus_metrics["shd"],
                "consensus_orient_acc": consensus_metrics["orient_acc"],
            })
        # Per-row dispersion for NHD and NHD ratio
        nhd_mean, nhd_sd, nhd_iqr, nhd_ci = _mean_std_iqr_ci(nhd_list)
        nhd_ratio_mean, nhd_ratio_sd, nhd_ratio_iqr, nhd_ratio_ci = _mean_std_iqr_ci(nhd_ratio_list)

        summary.update({
            "nhd_mean": nhd_mean,
            "nhd_sd": nhd_sd,
            "nhd_iqr": nhd_iqr,
            "nhd_ci95_low": nhd_ci[0],
            "nhd_ci95_high": nhd_ci[1],
            "nhd_ratio_mean": nhd_ratio_mean,
            "nhd_ratio_sd": nhd_ratio_sd,
            "nhd_ratio_iqr": nhd_ratio_iqr,
            "nhd_ratio_ci95_low": nhd_ratio_ci[0],
            "nhd_ratio_ci95_high": nhd_ratio_ci[1],
        })


        # Consensus-level NHD
        summary.update({
            "nhd_consensus": nhd_consensus,
            "nhd_ratio_consensus": nhd_ratio_consensus,
        })

    # Print and save summary JSON
    print("=== Global metrics (averages per row) + consensus ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    summary_path = csv_path.with_suffix(csv_path.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2)
    print(f"Saved summary to: {summary_path}")

    # ---- Per-row metrics output handling (unchanged) ----
    metric_keys = [
        "n_vars","tp","tn","fp","fn","accuracy","precision","recall","f1",
        "shd","orient_eval_pairs","orient_tp","orient_fn","orient_acc",
        "nhd","nhd_ratio", 

    ]

    if args.per_row_out:
        per_path = Path(args.per_row_out)
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
        for row, m in zip(rows, per_row_metrics):
            for k in metric_keys:
                row[k] = "" if m.get(k, None) is None else m.get(k)
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
