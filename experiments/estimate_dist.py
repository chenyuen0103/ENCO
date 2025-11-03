#!/usr/bin/env python3
"""
Compute empirical marginal, conditional, and interventional probabilities
from the JSONL produced by generate_prompts.py.

Input:  JSONL where each line is a record of the form
    {
      "data_idx": int,
      "shuffle_idx": int,
      "prompt": str,
      "answer": {
        "variables": [...],
        "adjacency_matrix": [...]
      },
      "rows": [
        {
          "intervened_variable": "Observational" or "<var>",
          "intervened_value": null or "<value>",
          "<var1>": "<value1>",
          "<var2>": "<value2>",
          ...
        },
        ...
      ]
    }

Output: JSONL where each line is a summary of the form
    {
      "data_idx": int,
      "shuffle_idx": int,
      "variables": [...],
      "adjacency_matrix": [...],
      "regimes": {
        "observational": { "N": ..., "marginals": {...}, "conditionals": {...} },
        "do(X1=0)": { ... },
        "do(X1=1)": { ... },
        ...
      }
    }

Values are kept exactly as strings as they appear in the prompt data;
if your generator anonymizes variables, values will be numeric strings.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def compute_regime_stats(rows: List[Dict[str, Any]],
                         variables: List[str]) -> Dict[str, Any]:
    """
    Compute empirical marginals and pairwise conditionals for a given regime.

    rows: list of dicts, each with keys:
        - "intervened_variable"
        - "intervened_value"
        - plus one key per variable in `variables` with the observed value
    variables: ordered list of variable names (e.g., ["X1","X2",...])

    Returns:
        {
          "N": int,
          "marginals": { var: { value: prob } },
          "conditionals": {
            given_var: {
              given_value: {
                target_var: { target_value: prob }
              }
            }
          }
        }
    """
    N = len(rows)
    if N == 0:
        return {
            "N": 0,
            "marginals": {},
            "conditionals": {}
        }

    # ---------- Marginals ----------
    # Count occurrences for each variable
    marg_counts: Dict[str, Counter] = {v: Counter() for v in variables}

    # ---------- Conditionals ----------
    # given_counts[X][x] = #rows with X=x
    # cond_counts[X][x][Y][y] = #rows with X=x and Y=y
    given_counts: Dict[str, Counter] = {v: Counter() for v in variables}
    cond_counts: Dict[str, Dict[str, Dict[str, Counter]]] = {
        gv: defaultdict(lambda: defaultdict(Counter))
        for gv in variables
    }

    for row in rows:
        # Values are strings (or None); convert to str to be consistent
        vals = {v: str(row[v]) for v in variables}

        # marginals
        for v, val in vals.items():
            marg_counts[v][val] += 1

        # conditionals
        for given_var in variables:
            gv_val = vals[given_var]
            given_counts[given_var][gv_val] += 1
            for target_var in variables:
                if target_var == given_var:
                    continue
                tv_val = vals[target_var]
                cond_counts[given_var][gv_val][target_var][tv_val] += 1

    # normalize marginals
    marginals: Dict[str, Dict[str, float]] = {}
    for v in variables:
        total = float(N)
        marginals[v] = {}
        for val, c in marg_counts[v].items():
            marginals[v][val] = c / total

    # normalize conditionals
    conditionals: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for given_var in variables:
        conditionals[given_var] = {}
        for gv_val, gv_count in given_counts[given_var].items():
            if gv_count == 0:
                continue
            gv_total = float(gv_count)
            conditionals[given_var][gv_val] = {}
            for target_var, tv_counter in cond_counts[given_var][gv_val].items():
                conditionals[given_var][gv_val][target_var] = {}
                for tv_val, cnt in tv_counter.items():
                    conditionals[given_var][gv_val][target_var][tv_val] = cnt / gv_total

    return {
        "N": N,
        "marginals": marginals,
        "conditionals": conditionals,
    }


def summarize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take one record from the prompts JSONL and produce the
    regimes-style summary object.
    """
    data_idx = record.get("data_idx")
    shuffle_idx = record.get("shuffle_idx")

    ans = record.get("answer", {})
    variables: List[str] = ans.get("variables", [])
    adjacency_matrix = ans.get("adjacency_matrix", [])

    rows: List[Dict[str, Any]] = record.get("rows", [])

    # Split into observational vs interventional regimes
    obs_rows: List[Dict[str, Any]] = []
    interventional_groups: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        ivar = row.get("intervened_variable")
        ival = row.get("intervened_value")
        if ivar == "Observational":
            obs_rows.append(row)
        else:
            regime_name = f"do({ivar}={ival})"
            interventional_groups.setdefault(regime_name, []).append(row)

    regimes: Dict[str, Any] = {}

    # Observational regime
    regimes["observational"] = compute_regime_stats(obs_rows, variables)

    # Each interventional regime
    for reg_name, rlist in interventional_groups.items():
        regimes[reg_name] = compute_regime_stats(rlist, variables)

    summary = {
        "data_idx": data_idx,
        "shuffle_idx": shuffle_idx,
        "variables": variables,
        "adjacency_matrix": adjacency_matrix,
        "regimes": regimes,
    }

    # You can optionally keep the original prompt if you like:
    # if "prompt" in record:
    #     summary["prompt"] = record["prompt"]

    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Compute marginal/conditional/interventional probabilities from prompts JSONL."
    )
    ap.add_argument(
        "--input-jsonl",
        default="out/cancer/prompts_obs200_int3_int-stat1_shuf5_anon.jsonl",
        help="Path to the prompts JSONL produced by generate_prompts.py",
    )
    ap.add_argument(
        "--output-jsonl",
        help="Where to write the regime stats JSONL. "
             "If omitted, will append '_stats.jsonl' to the input basename.",
    )
    args = ap.parse_args()

    in_path = Path(args.input_jsonl)
    if args.output_jsonl:
        out_path = Path(args.output_jsonl)
    else:
        out_path = in_path.with_name(in_path.stem + "_stats.jsonl")

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            summary = summarize_record(record)
            fout.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Wrote regime stats to {out_path}")


if __name__ == "__main__":
    main()
