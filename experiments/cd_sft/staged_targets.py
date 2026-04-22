"""
staged_targets.py

Programmatically generates per-row staged SFT data from an exported CD CSV.

For each row the script:
  1. Reads adj_bin from the 'answer' column
  2. Infers the variable names from the prompt (X1..Xn or real names)
  3. Computes Stage 1 (Skeleton), Stage 2 (V-structures), Stage 3 (Orientation)
     from the ground-truth adjacency matrix
  4. Writes a JSONL record with the staged think text as the completion

Usage:
    python experiments/cd_sft/staged_targets.py \
        --input-csv experiments/data/cancer_obs100_int10_anon_train.csv \
        --output-jsonl experiments/data/cancer_staged_sft.jsonl

    # Multiple CSVs → one mixed JSONL
    python experiments/cd_sft/staged_targets.py \
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
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Helpers imported from existing pipeline
# ---------------------------------------------------------------------------
try:
    from cd_generation.format import (
        ensure_assistant_think_prefill,
        validate_sft_example,
    )
except ModuleNotFoundError:
    from experiments.cd_generation.format import (
        ensure_assistant_think_prefill,
        validate_sft_example,
    )


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
_OBS_BLOCK_RE = re.compile(
    r"---\s*OBSERVATIONAL DATA\s*---\s*\n.*?observational_data=(\{.*?\})(?=\n(?:---|Output:|Formatting requirement:|assistant\b)|\Z)",
    re.DOTALL,
)
_INT_BLOCK_RE = re.compile(
    r"---\s*INTERVENTIONAL DATA\s*---\s*\n.*?interventional_data=(\{.*?\})(?=\n(?:---|Output:|Formatting requirement:|assistant\b)|\Z)",
    re.DOTALL,
)
_DO_KEY_RE = re.compile(r"do\((.+?)=([^)]+)\)")


def _extract_variables(prompt: str) -> Optional[List[str]]:
    """Extract ordered variable names from the VARIABLES block in the prompt."""
    m = _VAR_BLOCK_RE.search(prompt or "")
    if not m:
        return None
    names = _VAR_LINE_RE.findall(m.group(1))
    return names if names else None


def _extract_prompt_distribution_blocks(prompt: str) -> Tuple[Optional[dict], Optional[dict]]:
    def _decode_after_marker(text: str, marker: str) -> Optional[dict]:
        start = str(text or "").find(marker)
        if start < 0:
            return None
        value_start = start + len(marker)
        try:
            decoder = json.JSONDecoder()
            value, _ = decoder.raw_decode(str(text or ""), value_start)
        except Exception:
            return None
        return value if isinstance(value, dict) else None

    text = str(prompt or "")
    obs_payload = _decode_after_marker(text, "observational_data=")
    int_payload = _decode_after_marker(text, "interventional_data=")
    return obs_payload, int_payload


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


def _tv_distance(lhs: List[float], rhs: List[float]) -> float:
    return 0.5 * sum(abs(float(a) - float(b)) for a, b in zip(lhs, rhs))


def _fmt_probs(values: List[float], *, limit: int = 3) -> str:
    clipped = [float(v) for v in values[:limit]]
    rendered = ",".join(f"{value:.2f}" for value in clipped)
    if len(values) > limit:
        rendered += ",..."
    return f"[{rendered}]"


def _build_edge_effect_table(
    prompt: str,
    variables: List[str],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    obs_payload, int_payload = _extract_prompt_distribution_blocks(prompt)
    if not isinstance(obs_payload, dict) or not isinstance(int_payload, dict):
        return {}

    obs_marginals = obs_payload.get("marginals")
    if not isinstance(obs_marginals, list):
        return {}

    variable_to_idx = {str(name): idx for idx, name in enumerate(variables)}
    effect_table: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for do_key, payload in int_payload.items():
        if not isinstance(payload, dict):
            continue
        match = _DO_KEY_RE.fullmatch(str(do_key))
        if not match:
            continue
        cause_name = str(match.group(1))
        intervention_value = str(match.group(2))
        cause_idx = variable_to_idx.get(cause_name)
        if cause_idx is None:
            continue

        do_marginals = payload.get("marginals")
        if not isinstance(do_marginals, list):
            continue

        for effect_idx, (obs_dist, do_dist) in enumerate(zip(obs_marginals, do_marginals)):
            if effect_idx == cause_idx:
                continue
            if not isinstance(obs_dist, list) or not isinstance(do_dist, list):
                continue
            if len(obs_dist) != len(do_dist) or not obs_dist:
                continue

            tv = _tv_distance(obs_dist, do_dist)
            key = (cause_idx, effect_idx)
            current = effect_table.get(key)
            if current is None or tv > float(current["tv"]):
                effect_table[key] = {
                    "cause_name": variables[cause_idx],
                    "effect_name": variables[effect_idx],
                    "intervention_value": intervention_value,
                    "obs_dist": [float(v) for v in obs_dist],
                    "do_dist": [float(v) for v in do_dist],
                    "tv": float(tv),
                }

    return effect_table


def _summarize_effect(effect: Dict[str, Any]) -> str:
    return (
        f"do({effect['cause_name']}={effect['intervention_value']}) shifts {effect['effect_name']} "
        f"{_fmt_probs(effect['obs_dist'])}->{_fmt_probs(effect['do_dist'])} "
        f"(TV={float(effect['tv']):.2f})"
    )


def _inject_stage_evidence(
    stage_text: str,
    evidence_summaries: List[str],
) -> str:
    if not evidence_summaries:
        return stage_text
    lines = stage_text.splitlines()
    if len(lines) <= 1:
        return stage_text
    injected = [lines[0], f"Evidence: {'; '.join(evidence_summaries)}", *lines[1:]]
    return "\n".join(injected)


def build_evidence_grounded_sections(
    prompt: str,
    adj: List[List[int]],
    variables: List[str],
) -> Tuple[str, str, str]:
    stage1, stage2, stage3 = build_staged_sections(adj, variables)
    effect_table = _build_edge_effect_table(prompt, variables)
    if not effect_table:
        return stage1, stage2, stage3

    skeleton_candidates: List[Dict[str, Any]] = []
    for i, j in _skeleton_edges(adj):
        forward = effect_table.get((i, j))
        backward = effect_table.get((j, i))
        best = forward
        if backward is not None and (best is None or float(backward["tv"]) > float(best["tv"])):
            best = backward
        if best is not None:
            skeleton_candidates.append(best)

    skeleton_candidates.sort(key=lambda item: float(item["tv"]), reverse=True)
    stage1_evidence = [_summarize_effect(item) for item in skeleton_candidates[:3]]

    vstruct_evidence: List[str] = []
    for i, k, j in _vstructs(adj):
        left = effect_table.get((i, k))
        right = effect_table.get((j, k))
        pieces: List[str] = []
        if left is not None:
            pieces.append(f"{variables[i]}->{variables[k]} TV={float(left['tv']):.2f}")
        if right is not None:
            pieces.append(f"{variables[j]}->{variables[k]} TV={float(right['tv']):.2f}")
        if pieces:
            vstruct_evidence.append(f"{variables[k]} is supported by {' and '.join(pieces)}")
    stage2_evidence = vstruct_evidence[:2]

    orientation_candidates = [
        effect_table[(i, j)]
        for i, j in _directed_edges(adj)
        if (i, j) in effect_table
    ]
    orientation_candidates.sort(key=lambda item: float(item["tv"]), reverse=True)
    stage3_evidence = [_summarize_effect(item) for item in orientation_candidates[:5]]

    return (
        _inject_stage_evidence(stage1, stage1_evidence),
        _inject_stage_evidence(stage2, stage2_evidence),
        _inject_stage_evidence(stage3, stage3_evidence),
    )


def build_staged_sections(
    adj: List[List[int]],
    variables: List[str],
) -> Tuple[str, str, str]:
    """
    Build the three gold reasoning sections for a given adjacency matrix and variable list.

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

    return stage1, stage2, stage3


def build_staged_think(
    prompt: str,
    adj: List[List[int]],
    variables: List[str],
) -> str:
    """Build the full gold Stage 1/2/3 think text for one example."""
    stage1, stage2, stage3 = build_evidence_grounded_sections(prompt, adj, variables)
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
                    prompt_raw = (row.get(prompt_col) or "").strip()
                    if not prompt_raw:
                        raise ValueError(f"empty prompt (col={prompt_col!r})")
                    prompt = prompt_raw

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

                    stage1_text, stage2_text, stage3_text = build_evidence_grounded_sections(
                        prompt,
                        adj,
                        variables,
                    )
                    think_text = f"{stage1_text}\n\n{stage2_text}\n\n{stage3_text}"

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
                        "gold_think": think_text,
                        "gold_stage1": stage1_text,
                        "gold_stage2": stage2_text,
                        "gold_stage3": stage3_text,
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
