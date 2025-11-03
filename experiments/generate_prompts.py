#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

# Allow running from experiments/ with repo root one level up
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from causal_graphs.graph_real_world import load_graph_file  # type: ignore


# ------------------------ Utility: variable names ------------------------ #

def normalize_variable_names(graph) -> List[str]:
    """
    Returns the list of variable names in the order used by the graph.
    Adjust here if your graph uses a different naming convention.
    """
    return [v.name for v in graph.variables]


# ------------------------ BIF parsing and codebook ------------------------ #

def parse_bif_categories(filename: str) -> Dict[str, List[str]]:
    """
    Parse state names from a .bif file.

    Uses the same pattern as your working query_bif.py:
      variable X { ... { state1, state2, ... } ... }
    """
    text = Path(filename).read_text(encoding="utf-8", errors="ignore")
    mapping: Dict[str, List[str]] = {}
    pattern = re.compile(
        r"variable\s+([^\s\{]+)\s*\{[^\{\}]*\{\s*([^\}]*)\s*\}[^\}]*\}",
        flags=re.S,
    )
    for name, states_str in pattern.findall(text):
        states = [s.strip() for s in states_str.split(",") if s.strip()]
        mapping[name] = states
    if not mapping:
        raise RuntimeError("Failed to parse categories from BIF. Check file formatting.")
    return mapping


def _canon(s: str) -> str:
    """Canonicalize for case/whitespace-insensitive matching."""
    return re.sub(r"\s+", "", s).lower()


def build_codebook_from_bif(bif_path: str, variables: List[str]) -> Dict[str, List[str] | None]:
    """
    Map BIF variable names to the exact names in `variables`, using
    case/whitespace-insensitive matching.
    """
    raw = parse_bif_categories(bif_path)  # e.g., {"Pollution": ["low","high"], ...}
    raw_canon = {_canon(k): v for k, v in raw.items()}
    codebook: Dict[str, List[str] | None] = {}
    for v in variables:
        codebook[v] = raw_canon.get(_canon(v), None)
    return codebook


def build_codebook(graph, variables: List[str], bif_path: str) -> Dict[str, List[str]]:
    """
    Final codebook: prefer BIF categories, then fall back to graph cardinality,
    else default to ['0','1'].
    """
    cb_bif = build_codebook_from_bif(bif_path, variables)
    out: Dict[str, List[str]] = {}
    for i, v in enumerate(variables):
        names = cb_bif.get(v)
        if not names:
            # Fallback: infer number of categories from graph if possible
            prob_dist = getattr(graph.variables[i], "prob_dist", None)
            k = getattr(prob_dist, "num_categs", None)
            if isinstance(k, int) and k > 0:
                names = [str(j) for j in range(k)]
            else:
                names = ["0", "1"]
        out[v] = names
    return out


# ------------------------ Interventional sampling helper ------------------------ #

def _try_sample_interventional_api(
    graph: Any,
    batch_size: int,
    var_idx: int,
    var_name: str,
    state_idx: int,
    as_array: bool = True,
) -> np.ndarray:
    """
    Try several possible interventional sampling APIs.

    If your graph exposes a different method, adjust this function accordingly.
    On failure, raises RuntimeError with a clear message.
    """
    # 1) Preferred path for this repo: CausalDAG.sample(interventions=...)
    if hasattr(graph, "sample"):
        try:
            import numpy as _np
            values = _np.full((batch_size,), int(state_idx), dtype=_np.int32)
            return graph.sample(
                interventions={var_name: values},
                batch_size=batch_size,
                as_array=as_array,
            )
        except TypeError:
            pass

    # 2) Fallback: ENCO-like API: graph.sample_interventional(...)
    if hasattr(graph, "sample_interventional"):
        fn = graph.sample_interventional
        # (a) by index dict
        try:
            return fn(
                batch_size=batch_size,
                interventions={var_idx: state_idx},
                as_array=as_array,
            )
        except TypeError:
            pass
        # (b) by name dict
        try:
            return fn(
                batch_size=batch_size,
                interventions={var_name: state_idx},
                as_array=as_array,
            )
        except TypeError:
            pass
        # (c) positional style
        try:
            return fn(var_idx, state_idx, batch_size=batch_size, as_array=as_array)
        except TypeError:
            pass

    raise RuntimeError(
        f"Graph does not expose a supported interventional sampling API. "
        f"Cannot generate interventional data for {var_name}={state_idx}. "
        f"Please adapt _try_sample_interventional_api() to your graph implementation."
    )


# ------------------------ Prompt formatter ------------------------ #

def format_prompt_with_interventions(variables: List[str],
                                     all_rows: List[Dict[str, Any]],
                                     variable_map: Dict[str, str] | None = None,
                                     include_causal_rules: bool = False,
                                     include_give_steps: bool = False) -> str:
    """
    Turns mixed observational and interventional data into a comprehensive prompt
    asking for a JSON adjacency matrix.

    `variables`     : display variable names in system order.
    `all_rows`      : each row has keys:
                        - "intervened_variable": "Observational" or var name (display)
                        - "intervened_value"   : None or the value used in the do(...)
                        - plus one entry per variable in `variables`, with the observed value.
    `variable_map`  : optional mapping from original names to anonymized names; only used
                      for display in the intervention section.
    """
    obs_rows: List[Dict[str, Any]] = []
    interventions: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}

    for row in all_rows:
        intervened_var = row.get("intervened_variable")
        if intervened_var == "Observational":
            obs_rows.append(row)
        else:
            intervened_val = row.get("intervened_value")
            key = (intervened_var, intervened_val)
            if key not in interventions:
                interventions[key] = []
            interventions[key].append(row)

    lines: List[str] = []
    lines.append(
        "You are a causal discovery assistant. Your task is to infer a directed causal graph "
        "from the mixed observational and interventional data provided."
    )

    lines.append("\n--- OUTPUT INSTRUCTIONS ---")
    lines.append("Your response MUST be a single JSON object. Do NOT include any text before or after the JSON.")
    lines.append('- The JSON object must have two keys: "variables" and "adjacency_matrix".')
    lines.append('- "variables" MUST be the ordered list of variable names provided below.')
    lines.append('- "adjacency_matrix" MUST be an N x N list of lists where an entry at (row i, column j) '
                 'is 1 if a directed edge exists from `variables[i]` to `variables[j]`, and 0 otherwise.')

    # Optional: Causal inference reminders
    if include_causal_rules:
        lines.append("\n--- CAUSAL INFERENCE REMINDERS ---")
        lines.append("- Confounder: causes both X and Y. Adjust for confounders that are not effects of X.")
        lines.append("- Mediator: lies on the path X -> M -> Y. Do not adjust mediators if estimating the total effect of X on Y.")
        lines.append("- Collider: a common effect of two (or more) variables. Avoid conditioning on colliders or their descendants.")
        lines.append("- Backdoor adjustment: to estimate effect of X on Y, adjust for a set Z that blocks all backdoor paths into X and excludes descendants of X.")
        lines.append("- Front-door adjustment: if a mediator M transmits X -> M -> Y, with all backdoor paths from X to M blocked and all backdoor paths from M to Y blocked by X, the effect is identifiable via front-door.")
        lines.append("- Interventions: when a variable is intervened on, incoming edges to that variable are severed; use resulting changes to orient edges (e.g., if do(X) changes Y but do(Y) does not change X, prefer X -> Y).")
        lines.append("- Output a DAG consistent with the data (no directed cycles).")

    lines.append("\n--- SYSTEM VARIABLES (in order) ---")
    for i, var in enumerate(variables):
        lines.append(f"{i}: {var}")

    if obs_rows:
        lines.append("\n--- OBSERVATIONAL DATA ---")
        lines.append("The following cases were observed without any intervention:")
        for i, r in enumerate(obs_rows):
            clauses = [f"{h} was {r[h]}" for h in variables]
            sentence = f"Case {i+1}: " + ", and ".join(clauses) + "."
            lines.append(sentence)

    if interventions:
        lines.append("\n--- INTERVENTIONAL DATA ---")
        lines.append("The following cases were observed after specific interventions:")
        for (var, val), inter_rows in interventions.items():
            display_var = variable_map.get(var, var) if variable_map else var
            lines.append(
                f"\nWhen an intervention was performed to set '{display_var}' to '{val}', "
                f"the following outcomes were observed:"
            )
            for i, r in enumerate(inter_rows):
                clauses = [f"{h} was {r[h]}" for h in variables]
                sentence = f"Case {i+1}: " + ", and ".join(clauses) + "."
                lines.append(sentence)

    lines.append("\n--- END OF DATA ---")
    if include_give_steps:
        lines.append(
            "\nBased on ALL the data provided (both observational and interventional), think step-by-step: "
            "1. Compute the conditional probability using the observational data, which gives us the markov equivalent class of possible causal graphs "
            "2. Compute the interventional probability using the interventional, which gives us the exact graph inside the markov equivalent class."
        )
    lines.append("\nBased on ALL the data provided (both observational and interventional), generate the JSON response:")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Generate N prompt,answer pairs with optional interventional rows.")
    ap.add_argument("--bif-file", default="/home/yuen_chen/ENCO/causal_graphs/real_data/small_graphs/cancer.bif",
                    help="Path to .bif Bayesian network file.")
    ap.add_argument("--num-prompts", type=int, default=10, help="Number of pairs to generate.")
    ap.add_argument("--obs-per-prompt", type=int, default=200, help="Observational samples per prompt.")
    ap.add_argument("--int-per-combo", type=int, default=3,
                    help="Interventional samples PER (variable,state) combo. 0 disables interventions.")
    ap.add_argument("--intervene-vars", default="all",
                    help='Comma-separated variable names, or "all", or "none".')
    ap.add_argument("--states-per-var", type=int, default=1,
                    help="How many distinct intervention values to sample per variable.")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed; prompt i uses seed+i.")
    ap.add_argument("--anonymize", action="store_true", help="Replace variable names with X1,X2,... in prompt/answer.")
    ap.add_argument("--causal-rules", action="store_true",
                    help="Include causal inference reminders in the prompt.")
    ap.add_argument("--give-steps", action="store_true",
                    help="Append step-by-step guidance on using observational and interventional data.")
    ap.add_argument("--out-dir", default="./out/cancer", help="Output directory.")
    ap.add_argument("--shuffles-per-graph", type=int, default=5,
                    help="How many independent shuffles to generate per graph.")

    args = ap.parse_args()

    # ---------- Load graph & basic info ----------
    bif_abs = Path(args.bif_file).resolve(strict=True)
    graph = load_graph_file(str(bif_abs))
    variables = normalize_variable_names(graph)
    nvars = len(variables)

    adj_np = np.asarray(graph.adj_matrix)
    if adj_np.shape != (nvars, nvars):
        raise ValueError(f"adj_matrix shape {adj_np.shape} != ({nvars},{nvars})")
    adj_bin = (adj_np > 0).astype(int).tolist()

    # anonymization map
    vmap = {name: f"X{i+1}" for i, name in enumerate(variables)} if args.anonymize else {}
    variables_out = [vmap.get(v, v) for v in variables]

    # state name codebook
    codebook = build_codebook(graph, variables, str(bif_abs))

    def value_for_display(var_original_name: str, idx: int) -> str:
        idx = int(idx)
        if args.anonymize:
            # when anonymized, values are numeric indices as strings
            return str(idx)
        names = codebook[var_original_name]
        return names[idx] if 0 <= idx < len(names) else str(idx)

    # Resolve intervention targets (ORIGINAL names)
    if args.int_per_combo > 0 and args.intervene_vars.lower() not in {"none", ""}:
        if args.intervene_vars.lower() == "all":
            intervene_var_names = variables
        else:
            intervene_var_names = [s.strip() for s in args.intervene_vars.split(",") if s.strip()]
            unknown = [v for v in intervene_var_names if v not in variables]
            if unknown:
                raise ValueError(f"Unknown intervene-vars: {unknown}")
        intervene_var_idxs = [(variables.index(v), v) for v in intervene_var_names]
    else:
        intervene_var_idxs = []

    # ---------- Outputs ----------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compose suffixes reflecting configuration
    tags = []
    if args.anonymize:
        tags.append("anon")
    if getattr(args, "causal_rules", False):
        tags.append("rules")
    if getattr(args, "give_steps", False):
        tags.append("steps")
    extra_suffix = ("_" + "_".join(tags)) if tags else ""

    base_name = (
        f"prompts_obs{args.obs_per_prompt}"
        f"_int{args.int_per_combo}"
        f"_int-stat{args.states_per_var}"
        f"_shuf{args.shuffles_per_graph}{extra_suffix}"
    )

    jsonl_path = out_dir / f"{base_name}.jsonl"
    csv_path = out_dir / f"{base_name}.csv"

    jsonl_f = jsonl_path.open("w", encoding="utf-8")

    csv_f = csv_path.open("w", newline="", encoding="utf-8")
    fieldnames = ["data_idx", "shuffle_idx", "prompt", "answer"]
    csv_writer = csv.DictWriter(csv_f, fieldnames=fieldnames, extrasaction="ignore")
    csv_writer.writeheader()

    try:
        for i in range(args.num_prompts):
            # ---------- FIXED DATA PER data_idx ----------
            # Use a base seed that does NOT depend on rep
            seed_data = args.seed + i * 1000
            np.random.seed(seed_data)

            # Observational: sample once per data_idx
            arr_obs = graph.sample(batch_size=args.obs_per_prompt, as_array=True)

            # Build *base* observational rows (no shuffling here)
            obs_rows_base: List[Dict[str, Any]] = []
            for r in arr_obs:
                row_orig = {
                    variables[j]: value_for_display(variables[j], r[j])
                    for j in range(nvars)
                }
                row_disp = {vmap.get(k, k): v for k, v in row_orig.items()}
                row_disp = {
                    "intervened_variable": "Observational",
                    "intervened_value": None,
                    **row_disp,
                }
                obs_rows_base.append(row_disp)

            # Build *base* interventional rows ONCE per data_idx
            interventional_rows_base: List[Dict[str, Any]] = []
            if intervene_var_idxs and args.int_per_combo > 0:
                rng_states = np.random.default_rng(seed_data + 10_000)

                for var_idx, var_name in intervene_var_idxs:
                    n_states = len(codebook[var_name]) if var_name in codebook else 2
                    max_states = min(args.states_per_var, n_states)

                    chosen_states = rng_states.choice(
                        n_states,
                        size=max_states,
                        replace=False,
                    )

                    for s_idx in chosen_states:
                        arr_int = _try_sample_interventional_api(
                            graph,
                            args.int_per_combo,
                            var_idx,
                            var_name,
                            int(s_idx),
                            as_array=True,
                        )
                        for r in arr_int:
                            row_orig = {
                                variables[j]: value_for_display(variables[j], r[j])
                                for j in range(nvars)
                            }
                            row_disp = {vmap.get(k, k): v for k, v in row_orig.items()}
                            ivar_out = vmap.get(var_name, var_name)

                            interventional_rows_base.append({
                                "intervened_variable": ivar_out,
                                "intervened_value": (
                                    str(int(s_idx)) if args.anonymize
                                    else value_for_display(var_name, s_idx)
                                ),
                                **row_disp,
                            })

            # ---------- NOW ONLY SHUFFLE FOR EACH shuffle_idx ----------
            for rep in range(args.shuffles_per_graph):
                seed_ir = seed_data + rep

                # copy & shuffle observational rows
                obs_rows = [row.copy() for row in obs_rows_base]
                rng_obs = np.random.default_rng(seed_ir)
                rng_obs.shuffle(obs_rows)

                # copy interventional base rows
                interventional_rows = [row.copy() for row in interventional_rows_base]

                # group by (ivar, val)
                group_keys: List[Tuple[str, Any]] = []
                buckets: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
                for row in interventional_rows:
                    key = (row["intervened_variable"], row["intervened_value"])
                    if key not in buckets:
                        buckets[key] = []
                        group_keys.append(key)
                    buckets[key].append(row)

                # shuffle group order
                rng_groups = np.random.default_rng(seed_ir + 20_000)
                rng_groups.shuffle(group_keys)

                # shuffle rows within group
                rng_rows = np.random.default_rng(seed_ir + 30_000)
                interventional_rows_shuffled: List[Dict[str, Any]] = []
                for key in group_keys:
                    rows = buckets[key]
                    rng_rows.shuffle(rows)
                    interventional_rows_shuffled.extend(rows)

                interventional_rows = interventional_rows_shuffled

                # ---------- Full tabular data ----------
                all_rows = interventional_rows + obs_rows

                # ---------- Build prompt ----------
                prompt_text = format_prompt_with_interventions(
                    variables_out,
                    all_rows,
                    variable_map=vmap,
                    include_causal_rules=args.causal_rules,
                    include_give_steps=args.give_steps,
                )

                record = {
                    "data_idx": i,
                    "shuffle_idx": rep,
                    "prompt": prompt_text,
                    "answer": {
                        "variables": variables_out,
                        "adjacency_matrix": adj_bin,
                    },
                    "rows": all_rows,
                }
                jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # CSV: simple view (no rows)
                csv_writer.writerow({
                    "data_idx": i,
                    "shuffle_idx": rep,
                    "prompt": record["prompt"],
                    "answer": json.dumps(record["answer"], ensure_ascii=False),
                })

        print(f"Generated {args.num_prompts * max(1, args.shuffles_per_graph)} prompt,answer pairs.")
        print(f"- JSONL: {jsonl_path}")
        print(f"- CSV:   {csv_path}")
    finally:
        try:
            jsonl_f.close()
        except Exception:
            pass
        try:
            csv_f.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
