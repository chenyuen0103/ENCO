#!/usr/bin/env python3
"""
generate_staged_sft_data.py

Programmatically generates per-row staged SFT data from an exported CD CSV.

For each row the script:
  1. Reads adj_bin from the 'answer' column
  2. Infers the variable names from the prompt (X1..Xn or real names)
  3. Computes Stage 1 (Skeleton), Stage 2 (V-structures), Stage 3 (Orientation)
     from the ground-truth adjacency matrix
  4. Writes a JSONL record with the staged think text as the completion

Usage:
    python experiments/generate_staged_sft_data.py \
        --input-csv experiments/data/cancer_obs100_int10_anon_train.csv \
        --output-jsonl experiments/data/cancer_staged_sft.jsonl

    # Multiple CSVs → one mixed JSONL
    python experiments/generate_staged_sft_data.py \
        --input-csv experiments/data/cancer_obs100_int10_anon_train.csv \
                    experiments/data/earthquake_obs100_int10_anon_train.csv \
                    experiments/data/asia_obs100_int10_anon_train.csv \
        --output-jsonl experiments/data/mixed_staged_sft.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Helpers imported from existing pipeline
# ---------------------------------------------------------------------------
try:
    from cd_training_format import ensure_assistant_think_prefill, validate_sft_example
except ModuleNotFoundError:
    from experiments.cd_training_format import ensure_assistant_think_prefill, validate_sft_example


def _set_csv_field_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10_000_000)


def _load_adj(answer_raw: str) -> Optional[List[List[int]]]:
    """Parse adjacency matrix from the answer column JSON."""
    try:
        obj = json.loads(str(answer_raw or "").strip())
        if isinstance(obj, dict) and "adjacency_matrix" in obj:
            mat = obj["adjacency_matrix"]
        else:
            mat = obj
        rows = [[int(x) for x in r] for r in mat]
        n = len(rows)
        if any(len(r) != n for r in rows):
            return None
        return rows
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Variable name extraction
# ---------------------------------------------------------------------------

_VAR_BLOCK_RE = re.compile(
    r"---\s*VARIABLES\s*---\s*\n(.*?)(?=\n---|\Z)", re.DOTALL
)
_VAR_LINE_RE = re.compile(r"^\s*\d+\s*:\s*(\S+)", re.MULTILINE)


def _extract_variables(prompt: str) -> Optional[List[str]]:
    """Extract ordered variable names from the VARIABLES block in the prompt."""
    m = _VAR_BLOCK_RE.search(prompt or "")
    if not m:
        return None
    names = _VAR_LINE_RE.findall(m.group(1))
    return names if names else None


# ---------------------------------------------------------------------------
# Stage computation
# ---------------------------------------------------------------------------

def _skeleton_edges(adj: List[List[int]]) -> List[Tuple[int, int]]:
    """Undirected pairs (i < j) where an edge exists in either direction."""
    n = len(adj)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 1 or adj[j][i] == 1:
                edges.append((i, j))
    return edges


def _vstructs(adj: List[List[int]]) -> List[Tuple[int, int, int]]:
    """
    Unshielded colliders: triples (i, k, j) where i -> k <- j
    and i, j are NOT adjacent (neither adj[i][j] nor adj[j][i]).
    Returns canonical tuples with i < j.
    """
    n = len(adj)
    result = []
    for k in range(n):
        parents = [i for i in range(n) if adj[i][k] == 1]
        for pi, i in enumerate(parents):
            for j in parents[pi + 1:]:
                if adj[i][j] == 0 and adj[j][i] == 0:  # unshielded
                    result.append((i, k, j))
    return result


def _directed_edges(adj: List[List[int]]) -> List[Tuple[int, int]]:
    """All directed edges (i -> j)."""
    n = len(adj)
    return [(i, j) for i in range(n) for j in range(n) if adj[i][j] == 1]


def build_staged_think(
    adj: List[List[int]],
    variables: List[str],
) -> str:
    """
    Build the three-stage think text for a given adjacency matrix and variable list.

    Stage 1: undirected skeleton
    Stage 2: unshielded colliders (v-structures)
    Stage 3: all directed edges (orientation)
    """
    n = len(adj)
    if len(variables) != n:
        raise ValueError(
            f"variables length {len(variables)} != adjacency matrix size {n}"
        )

    # Stage 1 — Skeleton: one "X -- Y" per line
    skel = _skeleton_edges(adj)
    if skel:
        lines = "\n".join(f"{variables[i]} -- {variables[j]}" for i, j in skel)
        stage1 = f"Stage 1 (Skeleton):\n{lines}"
    else:
        stage1 = "Stage 1 (Skeleton):\nNone"

    # Stage 2 — V-structures: one "(parent1, collider, parent2)" triple per line
    vs = _vstructs(adj)
    if vs:
        lines = "\n".join(f"({variables[i]}, {variables[k]}, {variables[j]})" for i, k, j in vs)
        stage2 = f"Stage 2 (V-structures):\n{lines}"
    else:
        stage2 = "Stage 2 (V-structures):\nNone"

    # Stage 3 — Orientation: one "X -> Y" per line
    directed = _directed_edges(adj)
    if directed:
        lines = "\n".join(f"{variables[i]} -> {variables[j]}" for i, j in directed)
        stage3 = f"Stage 3 (Orientation):\n{lines}"
    else:
        stage3 = "Stage 3 (Orientation):\nNone"

    return f"{stage1}\n\n{stage2}\n\n{stage3}"


# ---------------------------------------------------------------------------
# Main build loop
# ---------------------------------------------------------------------------

def build_staged_jsonl(
    input_csvs: List[Path],
    output_jsonl: Path,
    prompt_col: str,
    answer_col: str,
) -> Tuple[int, int]:
    _set_csv_field_limit()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    skipped = 0

    with output_jsonl.open("w", encoding="utf-8") as fout:
        for csv_path in input_csvs:
            rows = list(
                csv.DictReader(csv_path.open("r", encoding="utf-8", newline=""))
            )
            for i, row in enumerate(rows):
                try:
                    prompt = (row.get(prompt_col) or "").strip()
                    if not prompt:
                        raise ValueError(f"empty prompt (col={prompt_col!r})")

                    answer_raw = (row.get(answer_col) or "").strip()
                    if not answer_raw:
                        raise ValueError("empty answer")

                    adj = _load_adj(answer_raw)
                    if adj is None:
                        raise ValueError("failed to parse adjacency_matrix from answer")

                    variables = _extract_variables(prompt)
                    if variables is None or len(variables) != len(adj):
                        # Fallback: generic X1..Xn names
                        n = len(adj)
                        variables = [f"X{k+1}" for k in range(n)]

                    think_text = build_staged_think(adj, variables)

                    # Ensure prompt ends with <think>\n prefill
                    prompt_prefilled = ensure_assistant_think_prefill(prompt)

                    completion = (
                        f"{think_text}</think>"
                        f"<answer>{json.dumps({'adjacency_matrix': adj}, ensure_ascii=False)}</answer>"
                    )

                    issues = validate_sft_example(prompt_prefilled, completion)
                    if issues:
                        raise ValueError("; ".join(issues))

                    rec = {
                        "prompt": prompt_prefilled,
                        "answer": completion,
                        "text": prompt_prefilled + "\n\n" + completion,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    wrote += 1

                except Exception as exc:
                    skipped += 1
                    print(
                        f"[warn] skip {csv_path.name} row {i}: {exc}",
                        file=sys.stderr,
                    )

    return wrote, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate staged SFT JSONL from exported CD CSVs."
    )
    ap.add_argument(
        "--input-csv",
        nargs="+",
        required=True,
        type=Path,
        help="One or more exported CD CSV files (prompt, answer columns).",
    )
    ap.add_argument(
        "--output-jsonl",
        required=True,
        type=Path,
        help="Output JSONL path.",
    )
    ap.add_argument(
        "--prompt-col",
        default="prompt",
        help="CSV column containing the full chat-formatted prompt (default: prompt).",
    )
    ap.add_argument(
        "--answer-col",
        default="answer",
        help="CSV column containing the ground-truth adjacency matrix JSON (default: answer).",
    )
    args = ap.parse_args()

    for p in args.input_csv:
        if not p.exists():
            sys.exit(f"ERROR: input CSV not found: {p}")

    wrote, skipped = build_staged_jsonl(
        input_csvs=args.input_csv,
        output_jsonl=args.output_jsonl,
        prompt_col=args.prompt_col,
        answer_col=args.answer_col,
    )

    print(
        json.dumps(
            {
                "output": str(args.output_jsonl),
                "wrote": wrote,
                "skipped": skipped,
                "input_csvs": [str(p) for p in args.input_csv],
            },
            indent=2,
        )
    )
    if wrote == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
