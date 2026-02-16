#!/usr/bin/env python3
"""
Backfill missing *_answer.json files referenced by prompt CSVs.

Why:
  Some prompt generators historically wrote prompt CSVs that referenced an
  answer_path, but did not actually write the answer JSON file. This breaks
  evaluation (evaluate.py) because it cannot load the ground-truth adjacency.

This script:
  - scans experiments/prompts/experiment1/<dataset>/**/prompts*.csv
  - reads the answer_path from the first non-empty row
  - if the referenced file is missing, reconstructs the GT adjacency from the BIF
    and writes {"answer": {"variables": [...], "adjacency_matrix": [...]}, "given_edges": null}

Supports:
  - anonymization via "_anon" tag
  - column permutations via "_colreverse/_colrandom/_coltopo/_colreverse_topo" tags
    (and defaults to original order otherwise)
  - names-only prompts via prompts_names_only*.csv (handled by the same logic)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _parse_bif_edges(bif_path: Path) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Minimal BIF parser: variable names + parent->child edges via 'probability (child | parents)' blocks.
    """
    text = bif_path.read_text(encoding="utf-8", errors="ignore")
    var_matches = re.findall(r"variable\s+([^\s\{]+)\s*\{", text)
    variables = [m.strip() for m in var_matches]

    edges: List[Tuple[str, str]] = []
    prob_blocks = re.findall(r"probability\s+.*?\{[^}]*\}", text, flags=re.S)
    for blk in prob_blocks:
        try:
            inside = blk.split("probability (", 1)[1].split(")", 1)[0]
        except Exception:
            continue
        if "|" in inside:
            child = inside.split("|", 1)[0].strip()
            parents = [s.strip() for s in inside.split("|", 1)[1].split(",") if s.strip()]
            edges.extend((p, child) for p in parents)
    return variables, edges


def _topo_sort(adj: List[List[int]]) -> List[int]:
    n = len(adj)
    indeg = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i][j] == 1:
                indeg[j] += 1
    q = [i for i in range(n) if indeg[i] == 0]
    q.sort()
    out: List[int] = []
    indeg_cur = indeg[:]
    while q:
        u = q.pop(0)
        out.append(u)
        for v in range(n):
            if adj[u][v] == 1:
                indeg_cur[v] -= 1
                if indeg_cur[v] == 0:
                    q.append(v)
        q.sort()
    if len(out) != n:
        return list(range(n))
    return out


def _adj_from_edges(variables: List[str], edges: List[Tuple[str, str]]) -> List[List[int]]:
    idx = {v: i for i, v in enumerate(variables)}
    n = len(variables)
    adj = [[0] * n for _ in range(n)]
    for src, dst in edges:
        if src in idx and dst in idx:
            adj[idx[src]][idx[dst]] = 1
    return adj


@dataclass(frozen=True)
class AnswerSpec:
    anonymize: bool
    col_order: str  # original|reverse|random|topo|reverse_topo


def _infer_answer_spec(path_str: str) -> AnswerSpec:
    s = path_str.lower()
    anonymize = "_anon" in s
    col_order = "original"
    m = re.search(r"_col([a-z0-9]+)", s)
    if m:
        col_order = m.group(1)
    else:
        # also support tags embedded in the base_name: colreverse, coltopo, colreverse_topo, colrandom
        m2 = re.search(r"col(reverse_topo|reverse|topo|random)", s)
        if m2:
            col_order = m2.group(1)
    return AnswerSpec(anonymize=anonymize, col_order=col_order)


def _permute_by_col_order(variables: List[str], adj: List[List[int]], *, col_order: str) -> Tuple[List[str], List[List[int]]]:
    n = len(variables)
    col_indices = list(range(n))
    if col_order == "reverse":
        col_indices.reverse()
    elif col_order == "topo":
        col_indices = _topo_sort(adj)
    elif col_order == "reverse_topo":
        col_indices = _topo_sort(adj)
        col_indices.reverse()
    elif col_order == "random":
        # Deterministic fallback: keep original (random requires knowing the seed)
        col_indices = list(range(n))

    perm_vars = [variables[i] for i in col_indices]
    perm_adj = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            old_r, old_c = col_indices[r], col_indices[c]
            perm_adj[r][c] = adj[old_r][old_c]
    return perm_vars, perm_adj


def _apply_anonymize(variables: List[str], *, anonymize: bool) -> List[str]:
    if not anonymize:
        return variables
    return [f"X{i+1}" for i in range(len(variables))]


def _read_first_answer_path(prompt_csv: Path) -> Optional[str]:
    with prompt_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ap = (row.get("answer_path") or "").strip()
            if ap:
                return ap
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--bif-file",
        default=None,
        help="Override BIF file path (default: causal_graphs/real_data/small_graphs/<dataset>.bif).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    experiments_dir = repo_root / "experiments"
    prompts_root = experiments_dir / "prompts" / "experiment1" / args.dataset
    if not prompts_root.exists():
        raise SystemExit(f"Prompts directory not found: {prompts_root}")

    bif_path = Path(args.bif_file) if args.bif_file else (repo_root / "causal_graphs" / "real_data" / "small_graphs" / f"{args.dataset}.bif")
    bif_path = bif_path.resolve(strict=True)

    variables, edges = _parse_bif_edges(bif_path)
    base_adj = _adj_from_edges(variables, edges)

    prompt_csvs = sorted(prompts_root.rglob("prompts*.csv"))
    if not prompt_csvs:
        print(f"[info] No prompt CSVs found under {prompts_root}")
        return 0

    created = 0
    skipped = 0
    missing_answer_path = 0

    for pcsv in prompt_csvs:
        ans_rel = _read_first_answer_path(pcsv)
        if not ans_rel:
            missing_answer_path += 1
            continue
        ans_path = (experiments_dir / ans_rel).resolve()
        if ans_path.exists():
            skipped += 1
            continue

        spec = _infer_answer_spec(ans_rel)
        perm_vars, perm_adj = _permute_by_col_order(variables, base_adj, col_order=spec.col_order)
        vars_out = _apply_anonymize(perm_vars, anonymize=spec.anonymize)

        payload = {"answer": {"variables": vars_out, "adjacency_matrix": perm_adj}, "given_edges": None}
        if args.dry_run:
            print(f"[dry-run] Would write {ans_path} (from {pcsv})")
        else:
            ans_path.parent.mkdir(parents=True, exist_ok=True)
            ans_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            created += 1

    print(f"[done] dataset={args.dataset} created={created} skipped_existing={skipped} missing_answer_path={missing_answer_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

