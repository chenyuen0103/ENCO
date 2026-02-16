#!/usr/bin/env python3
"""
Print examples of each prompt style (cases, matrix, summary, summary_probs)
using the cancer graph, without requiring torch.

We reuse already-generated prompt_txt files for cases/matrix and derive
summary/summary_probs from the same cases prompt by parsing the data section.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _excerpt(text: str, *, head: int, tail: int) -> str:
    lines = text.splitlines()
    if head + tail >= len(lines):
        return text
    out = []
    out.extend(lines[:head])
    out.append("... (snip) ...")
    out.extend(lines[-tail:])
    return "\n".join(out)


def parse_bif_categories(bif_path: Path) -> Dict[str, List[str]]:
    text = _read_text(bif_path)
    mapping: Dict[str, List[str]] = {}
    pattern = re.compile(
        r"variable\s+([^\s\{]+)\s*\{[^\{\}]*\{\s*([^\}]*)\s*\}[^\}]*\}",
        flags=re.S,
    )
    for name, states_str in pattern.findall(text):
        states = [s.strip() for s in states_str.split(",") if s.strip()]
        mapping[name] = states
    if not mapping:
        raise RuntimeError(f"Failed to parse categories from BIF: {bif_path}")
    return mapping


def _parse_system_variables(prompt_text: str) -> List[str]:
    m = re.search(r"--- SYSTEM VARIABLES.*?---\n(.*?)\n(?:The rows below|---)", prompt_text, flags=re.S)
    if not m:
        raise RuntimeError("Could not find SYSTEM VARIABLES section in cases prompt.")
    block = m.group(1).strip().splitlines()
    variables: List[str] = []
    for line in block:
        line = line.strip()
        if not line:
            continue
        # "0: Pollution"
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        variables.append(parts[1].strip())
    if not variables:
        raise RuntimeError("Parsed no variables from SYSTEM VARIABLES.")
    return variables


def _parse_case_assignments(case_line: str) -> Dict[str, str]:
    # "Case 1: Pollution is low, Smoker is True, ... ."
    _, rhs = case_line.split(":", 1)
    rhs = rhs.strip()
    if rhs.endswith("."):
        rhs = rhs[:-1]
    out: Dict[str, str] = {}
    for part in rhs.split(","):
        part = part.strip()
        if not part:
            continue
        if " is " not in part:
            continue
        var, val = part.split(" is ", 1)
        out[var.strip()] = val.strip()
    return out


def _parse_obs_and_int_from_cases_prompt(
    prompt_text: str,
    *,
    variables: List[str],
) -> tuple[List[Dict[str, str]], Dict[Tuple[str, str], List[Dict[str, str]]]]:
    obs_rows: List[Dict[str, str]] = []
    int_groups: Dict[Tuple[str, str], List[Dict[str, str]]] = {}

    lines = prompt_text.splitlines()
    i = 0
    in_obs = False
    in_int = False
    current_do: Tuple[str, str] | None = None

    while i < len(lines):
        line = lines[i].strip()

        if line == "--- OBSERVATIONAL DATA ---":
            in_obs = True
            in_int = False
            current_do = None
            i += 1
            continue

        if line == "--- INTERVENTIONAL DATA ---":
            in_obs = False
            in_int = True
            current_do = None
            i += 1
            continue

        if line == "--- END OF DATA ---":
            break

        if in_int:
            m = re.match(r"^When an intervention sets (.+?) to (.+?), the following cases were observed:$", line)
            if m:
                current_do = (m.group(1).strip(), m.group(2).strip())
                int_groups.setdefault(current_do, [])
                i += 1
                continue

        if (in_obs or in_int) and line.startswith("Case "):
            row = _parse_case_assignments(line)
            # Ensure we only keep variables we care about
            row = {v: row[v] for v in variables if v in row}
            if len(row) != len(variables):
                # Skip malformed rows (shouldn't happen for generated prompts)
                i += 1
                continue
            if in_obs:
                obs_rows.append(row)
            elif in_int and current_do is not None:
                int_groups[current_do].append(row)
            i += 1
            continue

        i += 1

    return obs_rows, int_groups


def _rows_to_numeric(
    rows: List[Dict[str, str]],
    *,
    variables: List[str],
    codebook: Dict[str, List[str]],
) -> List[List[float]]:
    out: List[List[float]] = []
    for r in rows:
        vec: List[float] = []
        for v in variables:
            val = r[v]
            states = codebook.get(v)
            if not states:
                raise RuntimeError(f"Missing states for variable {v}.")
            try:
                idx = states.index(val)
            except ValueError as e:
                raise RuntimeError(f"Value {val!r} not found in BIF states for {v}: {states}") from e
            vec.append(float(idx))
        out.append(vec)
    return out


def _corr_matrix_py(rows: List[List[float]]) -> List[List[float]]:
    n = len(rows)
    if n < 2:
        raise ValueError(f"Need at least 2 rows for correlation; got n={n}.")
    m = len(rows[0])
    if any(len(r) != m for r in rows):
        raise ValueError("Ragged rows: all rows must have the same number of columns.")

    means = [0.0] * m
    for r in rows:
        for j, x in enumerate(r):
            means[j] += float(x)
    means = [s / n for s in means]

    var = [0.0] * m
    for r in rows:
        for j, x in enumerate(r):
            d = float(x) - means[j]
            var[j] += d * d
    denom = float(n - 1)
    std = [(v / denom) ** 0.5 for v in var]

    corr: List[List[float]] = [[0.0] * m for _ in range(m)]
    for i in range(m):
        corr[i][i] = 1.0
        for j in range(i + 1, m):
            cov = 0.0
            for r in rows:
                cov += (float(r[i]) - means[i]) * (float(r[j]) - means[j])
            cov /= denom
            c = 0.0 if std[i] == 0.0 or std[j] == 0.0 else cov / (std[i] * std[j])
            corr[i][j] = c
            corr[j][i] = c
    return corr


def _mean_vec_py(rows: List[List[float]]) -> List[float]:
    n = len(rows)
    if n <= 0:
        raise ValueError("Need at least 1 row for mean.")
    m = len(rows[0])
    if any(len(r) != m for r in rows):
        raise ValueError("Ragged rows: all rows must have the same number of columns.")
    sums = [0.0] * m
    for r in rows:
        for j, x in enumerate(r):
            sums[j] += float(x)
    return [s / n for s in sums]


def _marginals_py(rows: List[List[float]], num_states: List[int]) -> List[List[float]]:
    n = len(rows)
    if n <= 0:
        raise ValueError("Need at least 1 row for marginals.")
    m = len(rows[0])
    if any(len(r) != m for r in rows):
        raise ValueError("Ragged rows: all rows must have the same number of columns.")
    if len(num_states) != m:
        raise ValueError(f"num_states must have length {m}; got {len(num_states)}.")

    counts: List[List[int]] = [[0 for _ in range(k)] for k in num_states]
    for r in rows:
        for j, x in enumerate(r):
            k = num_states[j]
            idx = int(round(float(x)))
            if 0 <= idx < k:
                counts[j][idx] += 1
    probs: List[List[float]] = []
    for j, k in enumerate(num_states):
        denom = float(n)
        probs.append([c / denom for c in counts[j]])
    return probs


def _format_summary(
    *,
    dataset_name: str,
    variables: List[str],
    state_names: List[List[str]],
    obs_rows_num: List[List[float]],
    int_groups_num: Dict[Tuple[str, str], List[List[float]]],
    decimals: int = 4,
) -> str:
    lines: List[str] = []
    lines.append("You are a highly intelligent question-answering bot with profound knowledge of causal inference and causal discovery.")
    lines.append(f"The following are summary statistics computed from data sampled from a Bayesian network named {dataset_name}.")
    lines.append("Infer the directed causal graph over the variables.")
    lines.append("\n--- OUTPUT INSTRUCTIONS ---")
    lines.extend([
        'Respond with a single valid JSON object and nothing else.',
        'The object must have exactly two keys: "variables" and "adjacency_matrix".',
        '- "variables": the ordered list of variable names given below.',
        '- "adjacency_matrix": an N x N list of lists of 0/1 integers, where [i][j] = 1 iff there is a directed edge from variables[i] to variables[j], else 0.',
        'Any text, explanation, or markdown outside this JSON object makes the answer invalid.',
        'Your first character MUST be "{" and your last character MUST be "}".',
    ])
    lines.append("\n--- VARIABLES (ORDER MATTERS) ---")
    for i, v in enumerate(variables):
        mapping = {str(s): str(name) for s, name in enumerate(state_names[i])}
        lines.append(f"{i}: {v} states=" + json.dumps(mapping, separators=(",", ":"), ensure_ascii=False))

    lines.append("\n--- OBSERVATIONAL SUMMARY ---")
    obs_mean = _mean_vec_py(obs_rows_num)
    obs_corr = _corr_matrix_py(obs_rows_num) if len(obs_rows_num) >= 2 else None
    obs_mean_r = [round(float(x), decimals) for x in obs_mean]
    obs_corr_r = (
        [[round(float(x), decimals) for x in row] for row in obs_corr] if obs_corr is not None else None
    )
    lines.append(f"obs_n={len(obs_rows_num)}")
    lines.append("obs_mean_numeric_codes=" + json.dumps(obs_mean_r, separators=(",", ":"), ensure_ascii=False))
    if obs_corr_r is not None:
        lines.append("obs_corr_numeric_codes=" + json.dumps(obs_corr_r, separators=(",", ":"), ensure_ascii=False))

    lines.append("\n--- INTERVENTIONAL SUMMARY ---")
    for (ivar, ival) in sorted(int_groups_num.keys(), key=lambda kv: (str(kv[0]), str(kv[1]))):
        rows = int_groups_num[(ivar, ival)]
        mu = _mean_vec_py(rows)
        mu_r = [round(float(x), decimals) for x in mu]
        delta_r = [round(float(mu[j] - obs_mean[j]), decimals) for j in range(len(mu))]
        payload = {"n": len(rows), "mean_numeric_codes": mu_r, "delta_from_obs_mean": delta_r}
        lines.append(f"do({ivar}={ival}): " + json.dumps(payload, separators=(",", ":"), ensure_ascii=False))

    lines.append("\n--- END OF SUMMARY ---")
    return "\n".join(lines)


def _format_summary_probs(
    *,
    dataset_name: str,
    variables: List[str],
    state_names: List[List[str]],
    obs_rows_num: List[List[float]],
    int_groups_num: Dict[Tuple[str, str], List[List[float]]],
    decimals: int = 4,
    top_k_effects: int = 5,
) -> str:
    lines: List[str] = []
    lines.append("You are a highly intelligent question-answering bot with profound knowledge of causal inference and causal discovery.")
    lines.append(f"The following are summary statistics computed from data sampled from a Bayesian network named {dataset_name}.")
    lines.append("Infer the directed causal graph over the variables.")
    lines.append("\n--- OUTPUT INSTRUCTIONS ---")
    lines.extend([
        'Respond with a single valid JSON object and nothing else.',
        'The object must have exactly two keys: "variables" and "adjacency_matrix".',
        '- "variables": the ordered list of variable names given below.',
        '- "adjacency_matrix": an N x N list of lists of 0/1 integers, where [i][j] = 1 iff there is a directed edge from variables[i] to variables[j], else 0.',
        'Any text, explanation, or markdown outside this JSON object makes the answer invalid.',
        'Your first character MUST be "{" and your last character MUST be "}".',
    ])
    lines.append("\n--- VARIABLES (ORDER MATTERS) ---")
    for i, v in enumerate(variables):
        mapping = {str(s): str(name) for s, name in enumerate(state_names[i])}
        lines.append(f"{i}: {v} states=" + json.dumps(mapping, separators=(",", ":"), ensure_ascii=False))

    num_states = [len(s) for s in state_names]
    obs_probs = _marginals_py(obs_rows_num, num_states)
    obs_probs_r = [[round(float(x), decimals) for x in row] for row in obs_probs]
    obs_corr = _corr_matrix_py(obs_rows_num) if len(obs_rows_num) >= 2 else None
    obs_corr_r = (
        [[round(float(x), decimals) for x in row] for row in obs_corr] if obs_corr is not None else None
    )

    lines.append("\n--- OBSERVATIONAL SUMMARY ---")
    lines.append(f"obs_n={len(obs_rows_num)}")
    payload = {variables[j]: obs_probs_r[j] for j in range(len(variables))}
    lines.append("obs_marginals=" + json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    if obs_corr_r is not None:
        lines.append("obs_corr_numeric_codes=" + json.dumps(obs_corr_r, separators=(",", ":"), ensure_ascii=False))

    lines.append("\n--- INTERVENTIONAL SUMMARY ---")
    for (ivar, ival) in sorted(int_groups_num.keys(), key=lambda kv: (str(kv[0]), str(kv[1]))):
        rows = int_groups_num[(ivar, ival)]
        do_probs = _marginals_py(rows, num_states)
        do_probs_r = [[round(float(x), decimals) for x in row] for row in do_probs]

        tv_scores: List[Tuple[int, float]] = []
        for j in range(len(variables)):
            if variables[j] == ivar:
                continue
            p = obs_probs_r[j]
            q = do_probs_r[j]
            tv = 0.5 * sum(abs(float(p[s]) - float(q[s])) for s in range(min(len(p), len(q))))
            tv_scores.append((j, tv))
        tv_scores.sort(key=lambda t: t[1], reverse=True)
        top = tv_scores[: max(0, int(top_k_effects))]

        top_effects = [[variables[j], round(float(tv), decimals)] for (j, tv) in top]
        top_marginals = {variables[j]: do_probs_r[j] for (j, _) in top}

        payload = {"n": len(rows), "tv_top_effects": top_effects, "do_marginals_top": top_marginals}
        lines.append(f"do({ivar}={ival}): " + json.dumps(payload, separators=(",", ":"), ensure_ascii=False))

    lines.append("\n--- END OF SUMMARY ---")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bif-file",
        type=Path,
        default=Path("causal_graphs/real_data/small_graphs/cancer.bif"),
    )
    ap.add_argument(
        "--cases-prompt",
        type=Path,
        default=Path("experiments/prompts/cancer/prompt_txt/prompts_obs200_int3_shuf3_data0_shuf0.txt"),
    )
    ap.add_argument(
        "--matrix-prompt",
        type=Path,
        default=Path("experiments/prompts/cancer/prompt_txt/prompts_obs200_int3_shuf3_matrix_data0_shuf0.txt"),
    )
    ap.add_argument("--full", action="store_true", help="Print full cases/matrix prompts (can be very long).")
    ap.add_argument("--head", type=int, default=60, help="Excerpt head lines for long prompts.")
    ap.add_argument("--tail", type=int, default=60, help="Excerpt tail lines for long prompts.")
    ap.add_argument("--decimals", type=int, default=4)
    ap.add_argument("--top-k-effects", type=int, default=5)
    args = ap.parse_args()

    codebook = parse_bif_categories(args.bif_file)
    cases_text = _read_text(args.cases_prompt)
    matrix_text = _read_text(args.matrix_prompt)
    variables = _parse_system_variables(cases_text)
    state_names = [codebook[v] for v in variables]

    obs_rows, int_groups = _parse_obs_and_int_from_cases_prompt(cases_text, variables=variables)
    obs_rows_num = _rows_to_numeric(obs_rows, variables=variables, codebook=codebook)
    int_groups_num: Dict[Tuple[str, str], List[List[float]]] = {}
    for (ivar, ival), rows in int_groups.items():
        int_groups_num[(ivar, ival)] = _rows_to_numeric(rows, variables=variables, codebook=codebook)

    dataset_name = args.bif_file.stem
    summary = _format_summary(
        dataset_name=dataset_name,
        variables=variables,
        state_names=state_names,
        obs_rows_num=obs_rows_num,
        int_groups_num=int_groups_num,
        decimals=int(args.decimals),
    )
    summary_probs = _format_summary_probs(
        dataset_name=dataset_name,
        variables=variables,
        state_names=state_names,
        obs_rows_num=obs_rows_num,
        int_groups_num=int_groups_num,
        decimals=int(args.decimals),
        top_k_effects=int(args.top_k_effects),
    )

    print("===== cases =====")
    print(cases_text if args.full else _excerpt(cases_text, head=int(args.head), tail=int(args.tail)))
    print("\n===== matrix =====")
    print(matrix_text if args.full else _excerpt(matrix_text, head=int(args.head), tail=int(args.tail)))
    print("\n===== summary (generated) =====")
    print(summary)
    print("\n===== summary_probs (generated) =====")
    print(summary_probs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

