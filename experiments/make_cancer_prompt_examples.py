#!/usr/bin/env python3
"""
Create paper-ready prompt examples for the cancer graph without requiring torch.

This script:
  1) Copies an existing matrix-style prompt (already generated in this repo).
  2) Derives a summary-statistics prompt from the matrix prompt by parsing the
     observational + interventional sample blocks and computing simple summary
     statistics over numeric codes.

Outputs are written to:
  experiments/prompts/cancer/example_prompts/
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / float(len(xs)) if xs else float("nan")


def _corr_matrix(rows: List[List[float]]) -> List[List[float]]:
    # Pearson correlation over columns (population version).
    if not rows:
        return []
    n = len(rows)
    m = len(rows[0])
    cols = [[float(r[j]) for r in rows] for j in range(m)]
    means = [_mean(c) for c in cols]
    vars_ = []
    for j in range(m):
        v = _mean([(x - means[j]) ** 2 for x in cols[j]])
        vars_.append(v)

    out: List[List[float]] = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i == j:
                out[i][j] = 1.0
                continue
            vi = vars_[i]
            vj = vars_[j]
            if vi <= 0.0 or vj <= 0.0:
                out[i][j] = 0.0
                continue
            cov = _mean([(cols[i][k] - means[i]) * (cols[j][k] - means[j]) for k in range(n)])
            out[i][j] = cov / math.sqrt(vi * vj)
    return out


def _round_mat(mat: List[List[float]], decimals: int) -> List[List[float]]:
    return [[round(float(x), decimals) for x in row] for row in mat]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_matrix_prompt(path: Path) -> Tuple[List[str], List[List[str]], Dict[Tuple[str, str], List[List[str]]]]:
    """
    Parse a CausalBench-style matrix prompt produced by this repo.

    Returns:
      variables, obs_rows, int_groups
    """
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Find observational header (variables line)
    obs_header_idx = None
    for i, line in enumerate(text):
        if line.strip() == "--- TRAINING DATA MATRIX (observational samples) ---":
            obs_header_idx = i
            break
    if obs_header_idx is None:
        raise RuntimeError(f"Failed to locate observational block in: {path}")

    # Header line occurs a couple lines after the block header.
    # We scan forward for the first '|' separated header.
    var_line_idx = None
    for i in range(obs_header_idx, min(obs_header_idx + 20, len(text))):
        if "|" in text[i] and "Each row is" not in text[i]:
            var_line_idx = i
            break
    if var_line_idx is None:
        raise RuntimeError(f"Failed to locate variable header line in: {path}")

    variables = [v.strip() for v in text[var_line_idx].split("|")]
    if not variables or any(not v for v in variables):
        raise RuntimeError(f"Parsed empty variables from header: {text[var_line_idx]!r}")

    def parse_row(line: str) -> List[str]:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != len(variables):
            raise ValueError(f"Bad row width: got {len(parts)} expected {len(variables)}: {line!r}")
        return parts

    # Observational rows continue until the interventional block header.
    obs_rows: List[List[str]] = []
    i = var_line_idx + 1
    while i < len(text):
        line = text[i].strip()
        if not line:
            i += 1
            continue
        if line == "--- TRAINING DATA MATRICES (interventional samples) ---":
            break
        if line.startswith("--- "):
            # unexpected section; stop to avoid mixing
            break
        # data row
        if "|" in line:
            obs_rows.append(parse_row(text[i]))
        i += 1

    # Interventions: blocks marked as [Intervention: do(X = v)]
    int_groups: Dict[Tuple[str, str], List[List[str]]] = {}
    re_do = re.compile(r"^\[Intervention:\s*do\(([^=]+)=([^\)]+)\)\]\s*$")

    while i < len(text):
        m = re_do.match(text[i].strip())
        if not m:
            i += 1
            continue
        ivar = m.group(1).strip()
        ival = m.group(2).strip()
        key = (ivar, ival)
        i += 1
        # Skip description lines until header line
        while i < len(text) and ("|" not in text[i] or text[i].strip().startswith("Columns follow")):
            if text[i].strip() == variables[0] or "|" in text[i]:
                break
            i += 1
        # Expect header line next (variables)
        while i < len(text) and "|" not in text[i]:
            i += 1
        if i >= len(text):
            break
        # Skip the header line
        i += 1
        rows: List[List[str]] = []
        while i < len(text):
            line = text[i].strip()
            if not line:
                i += 1
                continue
            if line.startswith("[Intervention:"):
                break
            if line.startswith("---"):
                break
            if "|" in line:
                rows.append(parse_row(text[i]))
            i += 1
        int_groups[key] = rows
    return variables, obs_rows, int_groups


def _build_codebook(variables: List[str], obs_rows: List[List[str]], int_groups: Dict[Tuple[str, str], List[List[str]]]) -> List[List[str]]:
    """
    Build per-variable state lists in a stable order (first-seen).
    """
    seen: List[Dict[str, int]] = [{ } for _ in variables]
    states: List[List[str]] = [[] for _ in variables]

    def ingest_row(row: List[str]) -> None:
        for j, val in enumerate(row):
            if val not in seen[j]:
                seen[j][val] = len(states[j])
                states[j].append(val)

    for r in obs_rows:
        ingest_row(r)
    for rows in int_groups.values():
        for r in rows:
            ingest_row(r)

    # Make sure we always have at least 2 states for booleans, if present
    for j in range(len(variables)):
        if "True" in seen[j] or "False" in seen[j]:
            for v in ["False", "True"]:
                if v not in seen[j]:
                    seen[j][v] = len(states[j])
                    states[j].append(v)

    return states


def _to_numeric_rows(rows: List[List[str]], codebook: List[List[str]]) -> List[List[float]]:
    idx: List[Dict[str, int]] = [{s: k for k, s in enumerate(codebook[j])} for j in range(len(codebook))]
    out: List[List[float]] = []
    for r in rows:
        out.append([float(idx[j][r[j]]) for j in range(len(r))])
    return out


def _summary_prompt_from_parsed(
    *,
    dataset_name: str,
    variables: List[str],
    codebook: List[List[str]],
    obs_rows_num: List[List[float]],
    int_groups_num: Dict[Tuple[str, str], List[List[float]]],
    decimals: int = 4,
) -> str:
    m = len(variables)
    obs_means = [_mean([r[j] for r in obs_rows_num]) for j in range(m)]
    obs_corr = _corr_matrix(obs_rows_num) if len(obs_rows_num) >= 2 else [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]

    payload: Dict[str, Any] = {
        "dataset": dataset_name,
        "variables_order": variables,
        "numeric_code_definition": "Each variable is encoded by its state index as listed in `state_names`.",
        "state_names": {variables[j]: codebook[j] for j in range(m)},
        "observational": {
            "n": len(obs_rows_num),
            "obs_mean_numeric_codes": [round(float(x), decimals) for x in obs_means],
            "obs_corr_numeric_codes": _round_mat(obs_corr, decimals),
        },
        "interventional": {},
    }

    for (ivar, ival), rows_num in sorted(int_groups_num.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        means = [_mean([r[j] for r in rows_num]) for j in range(m)] if rows_num else [float("nan")] * m
        delta = [round(float(means[j] - obs_means[j]), decimals) for j in range(m)]
        key = f"do({ivar}={ival})"
        payload["interventional"][key] = {
            "n": int(len(rows_num)),
            "mean_numeric_codes": [round(float(x), decimals) for x in means],
            "delta_from_obs_mean": delta,
        }

    method = (
        "You are an expert in causal discovery from observational and interventional data.\n"
        f"All data are sampled from a Bayesian network named {dataset_name}.\n"
        "Assume causal sufficiency, a DAG, and perfect do-interventions.\n\n"
        "NOTES:\n"
        "- Variables are discrete. Each value is encoded as a numeric state index.\n"
        "- `obs_corr_numeric_codes` is Pearson correlation computed on these numeric codes.\n\n"
        "TASK:\n"
        "Infer the directed causal graph over the variables.\n\n"
        "--- OUTPUT INSTRUCTIONS ---\n"
        "Respond with a single valid JSON object and nothing else.\n"
        "The object must have exactly two keys: \"variables\" and \"adjacency_matrix\".\n"
        "- \"variables\": the ordered list of variable names given in variables_order.\n"
        "- \"adjacency_matrix\": an N x N list of lists of 0/1 integers, where [i][j] = 1 iff there is a directed edge from variables[i] to variables[j], else 0.\n"
        "Your first character MUST be \"{\" and your last character MUST be \"}\".\n\n"
        "--- SUMMARY STATISTICS (JSON) ---\n"
    )
    return method + json.dumps(payload, ensure_ascii=False, indent=2)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Write cancer prompt examples (matrix + derived summary).")
    ap.add_argument("--obs-n", type=int, default=10, help="Number of observational rows to include.")
    ap.add_argument(
        "--int-k",
        type=int,
        default=3,
        help="Max number of rows to keep per intervention block (approx. int-per-combo for examples).",
    )
    ap.add_argument(
        "--src-matrix",
        type=Path,
        default=(
            repo_root
            / "experiments"
            / "prompts"
            / "experiment1"
            / "cancer"
            / "matrix_real_obs100_int50"
            / "prompt_txt"
            / "prompts_obs100_int50_shuf1_matrix_data0_shuf0.txt"
        ),
        help="Path to an existing matrix prompt file to downsample.",
    )
    args = ap.parse_args()

    src_matrix = Path(args.src_matrix)
    if not src_matrix.exists():
        raise SystemExit(f"Matrix prompt not found: {src_matrix}")

    out_dir = repo_root / "experiments" / "prompts" / "cancer" / "example_prompts"
    _ensure_dir(out_dir)

    variables, obs_rows_all, int_groups_all = _parse_matrix_prompt(src_matrix)

    obs_n = max(0, int(args.obs_n))
    int_k = max(0, int(args.int_k))
    obs_rows = obs_rows_all[:obs_n] if obs_n > 0 else []
    int_groups = {k: (rows[:int_k] if int_k > 0 else []) for k, rows in int_groups_all.items()}

    # Rewrite a smaller matrix prompt by editing the original file text, so the
    # example matches the exact repo prompt format.
    out_matrix = out_dir / f"cancer_matrix_obs{obs_n}_int{int_k}_example.txt"
    text = src_matrix.read_text(encoding="utf-8", errors="ignore").splitlines()
    out_lines: List[str] = []
    in_obs = False
    obs_header_written = False
    obs_emitted = 0
    in_int = False
    current_emitted = 0
    re_do = re.compile(r"^\[Intervention:\s*do\(([^=]+)=([^\)]+)\)\]\s*$")

    for line in text:
        if line.strip() == "--- TRAINING DATA MATRIX (observational samples) ---":
            in_obs = True
            in_int = False
            out_lines.append(line)
            continue
        if line.strip() == "--- TRAINING DATA MATRICES (interventional samples) ---":
            in_int = True
            in_obs = False
            out_lines.append(line)
            continue

        if in_obs:
            if line.strip().startswith("--- TRAINING DATA MATRICES"):
                in_obs = False
                in_int = True
                out_lines.append(line)
                continue
            if "|" in line and not obs_header_written:
                obs_header_written = True
                out_lines.append(line)
                continue
            if "|" in line and obs_header_written:
                if obs_emitted < obs_n:
                    out_lines.append(line)
                    obs_emitted += 1
                continue
            out_lines.append(line)
            continue

        if in_int:
            if re_do.match(line.strip()):
                current_emitted = 0
                out_lines.append(line)
                continue
            if "|" in line:
                # Keep the per-block header line (variable names) unconditionally.
                if current_emitted == 0 and line.strip().split("|")[0].strip() == variables[0]:
                    out_lines.append(line)
                    continue
                if current_emitted < int_k:
                    out_lines.append(line)
                    current_emitted += 1
                continue
            out_lines.append(line)
            continue

        out_lines.append(line)

    out_matrix.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    codebook = _build_codebook(variables, obs_rows, int_groups)
    obs_rows_num = _to_numeric_rows(obs_rows, codebook)
    int_groups_num = {k: _to_numeric_rows(v, codebook) for k, v in int_groups.items()}

    summary = _summary_prompt_from_parsed(
        dataset_name="cancer",
        variables=variables,
        codebook=codebook,
        obs_rows_num=obs_rows_num,
        int_groups_num=int_groups_num,
        decimals=4,
    )
    out_summary = out_dir / f"cancer_summary_obs{obs_n}_int{int_k}_example.txt"
    out_summary.write_text(summary, encoding="utf-8")

    print(f"[done] Wrote {out_matrix}")
    print(f"[done] Wrote {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
