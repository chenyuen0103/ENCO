#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

# Allow running from experiments/ with repo root one level up
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from causal_graphs.graph_real_world import load_graph_file  # type: ignore


# ------------------------ Utility: variable names ------------------------ #

def get_topological_sort(adj_matrix: List[List[int]]) -> List[int]:
    """
    Returns a list of node indices in topological order (Causes -> Effects).
    Uses Kahn's Algorithm.
    """
    n = len(adj_matrix)
    in_degree = [0] * n
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] == 1:
                in_degree[j] += 1
    
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    
    # Standard Kahn's Algo
    # (Using a copy of in_degree to avoid mutating if we needed it later, 
    # though here it's fine)
    in_degree_curr = list(in_degree)
    
    while queue:
        # Sort queue to ensure deterministic tie-breaking for identical structures
        queue.sort() 
        u = queue.pop(0)
        topo_order.append(u)
        
        for v in range(n):
            if adj_matrix[u][v] == 1:
                in_degree_curr[v] -= 1
                if in_degree_curr[v] == 0:
                    queue.append(v)
                    
    if len(topo_order) != n:
        # Fallback for cycles (shouldn't happen in DAGs)
        return list(range(n))
        
    return topo_order

def sample_interventional_values_vec(
    graph: Any,
    var_idx: int,
    var_name: str,
    values_vec: np.ndarray,
    as_array: bool = True,
) -> np.ndarray:
    """
    ENCO-style helper: sample interventional data where the clamped value
    can differ per sample.

    values_vec: shape (N,), integers in [0, num_states-1].
    If graph.sample(interventions={var_name: values_vec}) works, we use that
    (this is exactly what ENCO does).
    Otherwise we fall back to grouping by state and calling
    _try_sample_interventional_api(...) per state.
    """
    values_vec = np.asarray(values_vec, dtype=np.int32)
    batch_size = values_vec.shape[0]

    # 1) ENCO-style path: CausalDAG.sample with vector interventions
    if hasattr(graph, "sample"):
        try:
            return graph.sample(
                interventions={var_name: values_vec},
                batch_size=batch_size,
                as_array=as_array,
            )
        except TypeError:
            # fall through to grouped fallback
            pass

    # 2) Fallback: group by state and call _try_sample_interventional_api
    unique_states = np.unique(values_vec)
    arr_all: Optional[np.ndarray] = None

    for s in unique_states:
        idxs = np.where(values_vec == s)[0]
        arr_s = _try_sample_interventional_api(
            graph,
            batch_size=len(idxs),
            var_idx=var_idx,
            var_name=var_name,
            state_idx=int(s),
            as_array=as_array,
        )
        if arr_all is None:
            arr_all = np.zeros((batch_size, arr_s.shape[1]), dtype=arr_s.dtype)
        arr_all[idxs] = arr_s

    if arr_all is None:
        raise RuntimeError("sample_interventional_values_vec: no samples generated.")
    return arr_all

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


def build_codebook_from_bif(bif_path: str, variables: List[str]) -> Dict[str, Optional[List[str]]]:
    """
    Map BIF variable names to the exact names in `variables`, using
    case/whitespace-insensitive matching.
    """
    raw = parse_bif_categories(bif_path)  # e.g., {"Pollution": ["low","high"], ...}
    raw_canon = {_canon(k): v for k, v in raw.items()}
    codebook: Dict[str, Optional[List[str]]] = {}
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

def format_prompt_cb_matrix(
    variables: List[str],
    all_rows: List[Dict[str, Any]],
    dataset_name: str,
    include_causal_rules: bool = False,
    include_give_steps: bool = False,
    given_edges: Optional[List[Tuple[str, str]]] = None,
    include_def_int: bool = False,
) -> str:
    """
    CausalBench-style prompt: variable names + training data matrix.

    - Variables are given as a header row: v1 | v2 | ... | vN
    - Observational samples are in a single matrix block.
    - Interventional samples are grouped by (variable, value) and
      shown as separate matrix blocks, with do(X = v) in the
      descriptive text (NOT in the column names).
    - We still ask for a JSON adjacency_matrix, so it's comparable to
      our other prompts.

    all_rows: list of dicts, each like
        {
            "intervened_variable": "Observational" or var_name,
            "intervened_value": None or value,
            v1: val1,
            v2: val2,
            ...
        }
    """

    # ---------- split into observational vs interventional ----------
    obs_rows: List[Dict[str, Any]] = []
    int_buckets: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}

    for row in all_rows:
        ivar = row.get("intervened_variable", "Observational")
        ival = row.get("intervened_value", None)
        if ivar == "Observational" or ivar is None:
            obs_rows.append(row)
        else:
            key = (ivar, ival)
            int_buckets.setdefault(key, []).append(row)

    lines: List[str] = []

    # --- High-level task description (CausalBench-style) ---
    lines.append(
        "You are a highly intelligent question-answering bot with profound "
        "knowledge of causal inference and causal discovery."
    )
    lines.append(
        f"The following matrices are training data sampled from a Bayesian "
        f"network named {dataset_name}. Columns denote variables and rows denote observed cases."
    )
    lines.append(
        "From these data, infer the directed causal graph over the variables."
    )

    # --- OUTPUT INSTRUCTIONS (same JSON spec as before) ---
    lines.append("\n--- OUTPUT INSTRUCTIONS ---")
    if not include_give_steps:
        lines.extend([
            'Respond with a single valid JSON object and nothing else.',
            'The object must have exactly two keys: "variables" and "adjacency_matrix".',
            '- "variables": the ordered list of variable names given in the VARIABLE ORDER section below.',
            '- "adjacency_matrix": an N x N list of lists of 0/1 integers, where [i][j] = 1 '
            'iff there is a directed edge from variables[i] to variables[j], else 0.',
            'Any text, explanation, or markdown outside this JSON object makes the answer invalid.',
            'Your first character MUST be "{" and your last character MUST be "}".',
        ])
    else:
        lines.extend([
            'You may optionally include a brief explanation first.',
            'At the end, you MUST output a single JSON object with exactly two keys: "variables" and "adjacency_matrix".',
            '- "variables": the ordered list of variable names given in the VARIABLE ORDER section below.',
            '- "adjacency_matrix": an N x N list of lists of 0/1 integers, where [i][j] = 1 '
            'iff there is a directed edge from variables[i] to variables[j], else 0.',
            'The JSON must be syntactically valid (starts with "{" and ends with "}"), '
            'must appear on its own line at the end of your answer, and nothing may follow it.',
        ])

    # --- Optional causal reminders ---
    if include_causal_rules:
        lines.extend([
            "\n--- CAUSAL INFERENCE REMINDERS ---",
            "- Confounder: a variable that causes two others.",
            "- Mediator: lies on a path X -> M -> Y.",
            "- Collider: a common effect of two variables; avoid conditioning on colliders.",
            "- Use (conditional) independencies in the data to constrain the DAG.",
            "- The final output must be a DAG (no directed cycles).",
        ])

    if include_def_int:
        lines.extend([
            "\n--- INTERVENTION NOTES ---",
            "- An intervention do(X = v) sets X externally and replaces its usual causal mechanism.",
            "- In the intervened causal graph, all incoming edges into X are removed.",
            "- Only descendants of X can be causally affected by this intervention; non-descendants are not causally affected (though statistical associations may remain).",
        ])

    # --- Known edges (optional) ---
    if given_edges:
        lines.append("\n--- KNOWN DIRECT CAUSAL EDGES ---")
        lines.append("The following directed causal edges are guaranteed to be present in the true graph:")
        for src, dst in given_edges:
            lines.append(f"- {src} -> {dst}")
        lines.append(
            "In your adjacency_matrix output, the entry [i][j] MUST be 1 whenever "
            "variables[i] = src and variables[j] = dst for one of these edges."
        )

    # --- Variable order (for adjacency_matrix and matrices) ---
    # lines.append("\n--- VARIABLE ORDER (columns of every matrix) ---")
    # for i, v in enumerate(variables):
    #     lines.append(f"{i}: {v}")

    # Common header row
    header = " | ".join(variables)

    # --- Observational matrix ---
    if obs_rows:
        lines.append("\n--- TRAINING DATA MATRIX (observational samples) ---")
        lines.append(
            "Each row is one observed case without intervention."
        )
        lines.append(header)
        for row in obs_rows:
            vals = [str(row[v]) for v in variables]
            lines.append(" | ".join(vals))

    # --- Interventional matrices (grouped by do(X = value)) ---
    if int_buckets:
        lines.append("\n--- TRAINING DATA MATRICES (interventional samples) ---")
        lines.append(
            "Each block below corresponds to samples collected under a specific intervention do(X = value)."
        )

        # deterministic order for reproducibility
        for (ivar, ival) in sorted(int_buckets.keys(), key=lambda kv: (str(kv[0]), str(kv[1]))):
            rows = int_buckets[(ivar, ival)]
            lines.append(
                f"\n[Intervention: do({ivar} = {ival})]"
            )
            lines.append(
                "Columns follow the VARIABLE ORDER above. Each row is one observed case under this intervention."
            )
            lines.append(header)
            for row in rows:
                vals = [str(row[v]) for v in variables]
                lines.append(" | ".join(vals))

    lines.append("\n--- END OF DATA ---")

    # Optional internal reasoning hints
    if include_give_steps:
        lines.extend([
            "\n(You may follow these steps silently.)",
            "1) Use patterns of dependence/independence between columns in the observational matrix.",
            "2) Use differences between observational and interventional blocks do(X = value) "
            "to orient edges and distinguish causes from effects.",
            "3) Choose a directed acyclic graph consistent with all these constraints.",
            "Then output the JSON as specified above.",
        ])
    else:
        lines.append("\nBased on all of these data (observational and interventional), output the required JSON.")

    return "\n".join(lines)

def format_prompt_with_interventions(
    variables: List[str],
    all_rows: List[Dict[str, Any]],
    variable_map: Optional[Dict[str, str]] = None,
    include_causal_rules: bool = False,
    include_give_steps: bool = False,
    given_edges: Optional[List[Tuple[str, str]]] = None,
    copula_mode: str = "auto",  # NEW: 'auto' | '=' | 'is'
    include_def_int: bool = False,
) -> str:
    """
    Build a prompt that presents mixed observational + interventional data
    and asks for a JSON adjacency matrix over `variables`.

    If copula_mode == 'auto', we use '=' when variables look anonymized (X1, X2, ...),
    otherwise we use 'is'. You can force '=' or 'is' by passing the mode explicitly.
    """

    def _is_anonymized(names: List[str]) -> bool:
        return all(re.fullmatch(r"X\d+", v) for v in names)

    def _choose_copula(names: List[str], mode: str) -> str:
        if mode in {"=", "is"}:
            return mode
        return "=" if _is_anonymized(names) else "is"

    def _case_line(names: List[str], row: Dict[str, Any], mode: str) -> str:
        cop = _choose_copula(names, mode)
        return ", ".join(f"{v} {cop} {row[v]}" for v in names) + "."

    # --- Split rows into observational vs. interventional buckets ---
    obs_rows: List[Dict[str, Any]] = []
    interventions: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
    for row in all_rows:
        ivar = row.get("intervened_variable")
        if ivar == "Observational":
            obs_rows.append(row)
        else:
            ival = row.get("intervened_value")
            key = (ivar, ival)
            interventions.setdefault(key, []).append(row)

    lines: List[str] = []
    lines.append(
        "You are a causal discovery assistant. From the data below, infer a directed causal graph "
        "over the given variables."
    )

    # ---------- OUTPUT INSTRUCTIONS ----------
    lines.append("\n--- OUTPUT INSTRUCTIONS ---")
    if not include_give_steps:
        lines.extend([
            'Respond with a single valid JSON object and nothing else.',
            'The object must have exactly two keys: "variables" and "adjacency_matrix".',
            '- "variables": the ordered list of variable names given in the SYSTEM VARIABLES section below.',
            '- "adjacency_matrix": an N x N list of lists of 0/1 integers, where [i][j] = 1 '
            'iff there is a directed edge from variables[i] to variables[j], else 0.',
            'Any text, explanation, or markdown outside this JSON object makes the answer invalid.',
            'Your first character MUST be "{" and your last character MUST be "}".',
        ])
    else:
        lines.extend([
            'You may optionally include a brief explanation first.',
            'At the end, you MUST output a single JSON object with exactly two keys: "variables" and "adjacency_matrix".',
            '- "variables": the ordered list of variable names given in the SYSTEM VARIABLES section below.',
            '- "adjacency_matrix": an N x N list of lists of 0/1 integers, where [i][j] = 1 '
            'iff there is a directed edge from variables[i] to variables[j], else 0.',
            'The JSON must be syntactically valid (starts with "{" and ends with "}"), '
            'must appear on its own line at the end of your answer, and nothing may follow it.',
        ])

    # ---------- Optional causal reminders ----------
    if include_causal_rules:
        lines.extend([
            "\n--- CAUSAL INFERENCE REMINDERS ---",
            "- Confounder: a variable that causes two others.",
            "- Mediator: lies on a path X -> M -> Y.",
            "- Collider: a common effect of two variables; avoid conditioning on colliders.",
            "- Backdoor paths: block backdoor paths into a cause when estimating its effect.",
            "- Interventions: do(X) cuts all incoming edges into X; use changes in other variables to orient edges.",
            "- The final output must be a DAG (no directed cycles).",
        ])

    if include_def_int:
        lines.extend([
            "\n--- INTERVENTION NOTES ---",
            "- Each do(X = v) forces X to the chosen value and severs incoming causal arrows into X.",
            "- Descendants of X may change because X is fixed; ancestors of X are unaffected by the intervention itself.",
        ])

    # ---------- Known edges section (optional) ----------
    if given_edges:
        lines.append("\n--- KNOWN DIRECT CAUSAL EDGES ---")
        lines.append("You are told that the following directed causal relationships are definitely present in the true causal graph:")
        for src, dst in given_edges:
            lines.append(f"- {src} -> {dst}")

    # ---------- System variables ----------
    lines.append("\n--- SYSTEM VARIABLES (in order) ---")
    for i, var in enumerate(variables):
        lines.append(f"{i}: {var}")
    lines.append(
        "The rows below are in random order and must be treated as an unordered set."
    )
    # ---------- Observational data ----------
    if obs_rows:
        lines.append("\n--- OBSERVATIONAL DATA ---")
        
        lines.append("Cases observed:")
        for i, r in enumerate(obs_rows):
            lines.append(f"Case {i+1}: " + _case_line(variables, r, copula_mode))

    # ---------- Interventional data ----------
    if interventions:
        lines.append("\n--- INTERVENTIONAL DATA ---")
        lines.append("Cases observed under specific interventions:")
        for (var, val), inter_rows in interventions.items():
            display_var = variable_map.get(var, var) if variable_map else var
            lines.append(f"\nWhen an intervention sets {display_var} to {val}, the following cases were observed:")
            for i, r in enumerate(inter_rows):
                lines.append(f"Case {i+1}: " + _case_line(variables, r, copula_mode))

    lines.append("\n--- END OF DATA ---")

    # ---------- High-level reasoning hints (optional) ----------
    has_obs = bool(obs_rows)
    has_interv = bool(interventions)
    if include_give_steps:
        if has_obs and has_interv:
            lines.extend([
                "\n(You may follow these steps silently.)",
                "1) Use observational data to infer a plausible Markov equivalence class.",
                "2) Use interventional data to identify a single DAG within that class.",
                "Then output the JSON as specified above.",
            ])
        elif has_obs:
            lines.extend([
                "\n(You may follow these steps silently.)",
                "1) Use observational conditional (in)dependencies to constrain the DAG.",
                "Then output the JSON as specified above.",
            ])
        elif has_interv:
            lines.extend([
                "\n(You may follow these steps silently.)",
                "1) For each do(X=x), treat incoming edges into X as cut; use changes in other variables to orient edges.",
                "2) Combine across interventions into a single DAG.",
                "Then output the JSON as specified above.",
            ])
        else:
            lines.append("\nThen output the JSON as specified above.")
    else:
        if has_obs and has_interv:
            lines.append("\nBased on ALL the data (observational and interventional), output the required JSON.")
        elif has_obs:
            lines.append("\nBased on the observational data, output the required JSON.")
        elif has_interv:
            lines.append("\nBased on the interventional data, output the required JSON.")
        else:
            lines.append("\nOutput the required JSON.")

    return "\n".join(lines)


def choose_given_edges(
    adj_bin: List[List[int]],
    frac: float,
    seed: int = 0,
    max_per_node: int = 2,
) -> List[Tuple[int, int]]:
    """
    Scheme 1: choose a small, roughly uniform subset of true edges.

    adj_bin: binary adjacency matrix (list of list of 0/1)
    frac:    fraction of *all* edges to reveal (e.g. 0.2 = 20%)
    seed:    RNG seed
    max_per_node: optional cap on how many given-edges can touch each node.
                  If None, don't cap.

    Returns list of (i, j) index pairs where adj_bin[i][j] == 1.
    """
    n = len(adj_bin)
    edges: List[Tuple[int, int]] = [
        (i, j)
        for i in range(n)
        for j in range(n)
        if adj_bin[i][j] == 1
    ]

    if frac <= 0.0 or not edges:
        return []

    import numpy as _np
    rng = _np.random.default_rng(seed)
    edges_arr = _np.array(edges, dtype=int)
    rng.shuffle(edges_arr)
    edges = [tuple(map(int, e)) for e in edges_arr]

    k_target = max(1, int(round(len(edges) * frac)))

    if max_per_node is None:
        # Simple: just take first k_target edges
        return edges[:k_target]

    node_counts = [0] * n
    chosen: List[Tuple[int, int]] = []

    for (i, j) in edges:
        if len(chosen) >= k_target:
            break
        if node_counts[i] >= max_per_node and node_counts[j] >= max_per_node:
            continue
        chosen.append((i, j))
        node_counts[i] += 1
        node_counts[j] += 1

    return chosen


# def main():
#     ap = argparse.ArgumentParser(description="Generate N prompt,answer pairs with optional interventional rows.")
#     ap.add_argument("--bif-file", default="../causal_graphs/real_data/small_graphs/asia.bif",
#                     help="Path to .bif Bayesian network file.")
#     ap.add_argument("--num-prompts", type=int, default=5, help="Number of pairs to generate.")
#     ap.add_argument("--obs-per-prompt", type=int, default=5000, help="Observational samples per prompt.")
#     ap.add_argument("--int-per-combo", type=int, default=200,
#                     help="Interventional samples PER (variable,state) combo. 0 disables interventions.")
#     ap.add_argument("--intervene-vars", default="all",
#                     help='Comma-separated variable names, or "all", or "none".')
#     ap.add_argument("--seed", type=int, default=0, help="Base RNG seed; prompt i uses seed+i.")
#     ap.add_argument("--anonymize", action="store_true", help="Replace variable names with X1,X2,... in prompt/answer.")
#     ap.add_argument("--causal-rules", action="store_true",
#                     help="Include causal inference reminders in the prompt.")
#     ap.add_argument("--give-steps", action="store_true",
#                     help="Append step-by-step guidance on using observational and interventional data.")
#     # ap.add_argument("--out-dir", default="./prompts/asia", help="Output directory.")
#     ap.add_argument("--shuffles-per-graph", type=int, default=3,
#                     help="How many independent shuffles to generate per graph.")
#     ap.add_argument(
#         "--given-edge-frac",
#         type=float,
#         default=0.0,
#         help="Fraction of true edges to reveal as known ground-truth edges (Scheme 1). "
#              "0.0 means no edges are given.",
#     )
#     ap.add_argument(
#     "--prompt-style",
#     choices=["cases", "matrix"],
#     default="cases",
#     help=(
#         "Prompt layout: 'cases' = per-case natural language; "
#         "'matrix' = CausalBench-style header + numeric matrix."
#     ),
#     )
#         # --- NEW ARGS FOR ROBUSTNESS ---
#     ap.add_argument("--row-order", choices=["random", "sorted", "reverse"], default="random")
#     ap.add_argument("--col-order", choices=["original", "reverse", "random"], default="original")


#     args = ap.parse_args()
#     args.out_dir = os.path.join(
#         os.path.dirname(__file__),
#         "prompts",
#         os.path.splitext(os.path.basename(args.bif_file))[0],
#     )
#     # ---------- Load graph & basic info ----------
#     bif_abs = Path(args.bif_file).resolve(strict=True)
#     graph = load_graph_file(str(bif_abs))
#     variables = normalize_variable_names(graph)
#     nvars = len(variables)

#     adj_np = np.asarray(graph.adj_matrix)
#     if adj_np.shape != (nvars, nvars):
#         raise ValueError(f"adj_matrix shape {adj_np.shape} != ({nvars},{nvars})")
#     adj_bin = (adj_np > 0).astype(int).tolist()

#     # anonymization map
#     vmap = {name: f"X{i+1}" for i, name in enumerate(variables)} if args.anonymize else {}
#     variables_out = [vmap.get(v, v) for v in variables]
#     # ------------------ Scheme 1: choose given edges ------------------
#     if args.given_edge_frac > 0.0:
#         given_edges_idx = choose_given_edges(
#             adj_bin,
#             frac=args.given_edge_frac,
#             seed=args.seed,
#             max_per_node=2,  # tweak if you like
#         )
#         # convert from indices to displayed variable names (respecting anonymization)
#         given_edges_named: List[Tuple[str, str]] = [
#             (variables_out[i], variables_out[j]) for (i, j) in given_edges_idx
#         ]
#     else:
#         given_edges_idx = []
#         given_edges_named = []

#     if args.given_edge_frac > 0.0:
#         # Option A: encode fraction as percentage, e.g. _gedge20 for 0.2
#         frac_pct = int(round(100 * args.given_edge_frac))
#         edge_tag = f"_gedge{frac_pct}"
#         # Option B (alternative): encode absolute # of revealed edges:
#         # edge_tag = f"_gedge{len(given_edges_idx)}"
#     else:
#         edge_tag = ""
#     answer_obj = {
#         "variables": variables_out,
#         "adjacency_matrix": adj_bin,
#     }
#     # state name codebook
#     codebook = build_codebook(graph, variables, str(bif_abs))

#     def value_for_display(var_original_name: str, idx: int) -> str:
#         idx = int(idx)
#         if args.anonymize:
#             # when anonymized, values are numeric indices as strings
#             return str(idx)
#         names = codebook[var_original_name]
#         return names[idx] if 0 <= idx < len(names) else str(idx)

#     # Resolve intervention targets (ORIGINAL names)
#     if args.int_per_combo > 0 and args.intervene_vars.lower() not in {"none", ""}:
#         if args.intervene_vars.lower() == "all":
#             intervene_var_names = variables
#         else:
#             intervene_var_names = [s.strip() for s in args.intervene_vars.split(",") if s.strip()]
#             unknown = [v for v in intervene_var_names if v not in variables]
#             if unknown:
#                 raise ValueError(f"Unknown intervene-vars: {unknown}")
#         intervene_var_idxs = [(variables.index(v), v) for v in intervene_var_names]
#     else:
#         intervene_var_idxs = []

#     # ---------- Outputs ----------
#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # NEW: directory to store prompt text files
#     prompt_txt_dir = out_dir / "prompt_txt"
#     prompt_txt_dir.mkdir(parents=True, exist_ok=True)
#     # Compose suffixes reflecting configuration
#     tags = []
#     if args.anonymize:
#         tags.append("anon")
#     if getattr(args, "causal_rules", False):
#         tags.append("rules")
#     if getattr(args, "give_steps", False):
#         tags.append("steps")

#     # NEW: encode prompt style
#     if hasattr(args, "prompt_style"):
#         if args.prompt_style in {"matrix"}:
#             tags.append(f"{args.prompt_style}")

#     extra_suffix = ("_" + "_".join(tags)) if tags else ""

#     base_name = (
#         f"prompts_obs{args.obs_per_prompt}"
#         f"_int{args.int_per_combo}"
#         f"_shuf{args.shuffles_per_graph}{edge_tag}{extra_suffix}"
#     )


#     jsonl_path = out_dir / f"{base_name}.jsonl"
#     csv_path = out_dir / f"{base_name}.csv"
#     answer_path = out_dir / f"{base_name}_answer.json"
#     with answer_path.open("w", encoding="utf-8") as f_ans:
#         json.dump(
#             {
#                 "answer": answer_obj,
#                 "given_edges": given_edges_named if args.given_edge_frac > 0.0 else None,
#             },
#             f_ans,
#             ensure_ascii=False,
#             indent=2,
#         )
#     jsonl_f = jsonl_path.open("w", encoding="utf-8")

#     csv_f = csv_path.open("w", newline="", encoding="utf-8")
#     # We now store only a reference to the shared answer JSON
#     fieldnames = ["data_idx", "shuffle_idx", "prompt_path", "answer_path", "given_edges"]
#     csv_writer = csv.DictWriter(csv_f, fieldnames=fieldnames, extrasaction="ignore")
#     csv_writer.writeheader()


#     try:
#         for i in range(args.num_prompts):
#             # ---------- FIXED DATA PER data_idx ----------
#             # Use a base seed that does NOT depend on rep
#             seed_data = args.seed + i * 1000
#             np.random.seed(seed_data)

#             # Observational: sample once per data_idx
#             arr_obs = graph.sample(batch_size=args.obs_per_prompt, as_array=True)

#             # Build *base* observational rows (no shuffling here)
#             obs_rows_base: List[Dict[str, Any]] = []
#             for r in arr_obs:
#                 row_orig = {
#                     variables[j]: value_for_display(variables[j], r[j])
#                     for j in range(nvars)
#                 }
#                 row_disp = {vmap.get(k, k): v for k, v in row_orig.items()}
#                 row_disp = {
#                     "intervened_variable": "Observational",
#                     "intervened_value": None,
#                     **row_disp,
#                 }
#                 obs_rows_base.append(row_disp)

#             # Build *base* interventional rows ONCE per data_idx
#             # Build *base* interventional rows ONCE per data_idx
#             interventional_rows_base: List[Dict[str, Any]] = []
#             if intervene_var_idxs and args.int_per_combo > 0:
#                 rng_int = np.random.default_rng(seed_data + 10_000)

#                 # For fair comparison with ENCO:
#                 # - Treat int_per_combo as "dataset_size" per variable.
#                 # - For each sample, draw the clamped value uniformly over states.
#                 for var_idx, var_name in intervene_var_idxs:
#                     # Determine number of categories for this variable
#                     prob_dist = getattr(graph.variables[var_idx], "prob_dist", None)
#                     num_categs = getattr(prob_dist, "num_categs", None)
#                     if not isinstance(num_categs, int) or num_categs <= 0:
#                         num_categs = len(codebook.get(var_name, [])) or 2

#                     dataset_size = args.int_per_combo  # ENCO-style: #samples per variable

#                     values_vec = rng_int.integers(
#                         low=0,
#                         high=num_categs,
#                         size=dataset_size,
#                         dtype=np.int32,
#                     )

#                     # Sample interventional data with *per-sample* clamped values
#                     arr_int = sample_interventional_values_vec(
#                         graph,
#                         var_idx=var_idx,
#                         var_name=var_name,
#                         values_vec=values_vec,
#                         as_array=True,
#                     )

#                     # Build rows; note each row's "intervened_value" can differ
#                     for sample_idx, r in enumerate(arr_int):
#                         s_idx = int(values_vec[sample_idx])

#                         row_orig = {
#                             variables[j]: value_for_display(variables[j], r[j])
#                             for j in range(nvars)
#                         }
#                         row_disp = {vmap.get(k, k): v for k, v in row_orig.items()}
#                         ivar_out = vmap.get(var_name, var_name)

#                         interventional_rows_base.append({
#                             "intervened_variable": ivar_out,
#                             "intervened_value": (
#                                 str(s_idx) if args.anonymize
#                                 else value_for_display(var_name, s_idx)
#                             ),
#                             **row_disp,
#                         })

#             # ---------- NOW ONLY SHUFFLE FOR EACH shuffle_idx ----------
#             for rep in range(args.shuffles_per_graph):
#                 seed_ir = seed_data + rep

#                 # copy & shuffle observational rows
#                 obs_rows = [row.copy() for row in obs_rows_base]
#                 rng_obs = np.random.default_rng(seed_ir)
#                 rng_obs.shuffle(obs_rows)

#                 # copy interventional base rows
#                 interventional_rows = [row.copy() for row in interventional_rows_base]

#                 # group by (ivar, val)
#                 group_keys: List[Tuple[str, Any]] = []
#                 buckets: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
#                 for row in interventional_rows:
#                     key = (row["intervened_variable"], row["intervened_value"])
#                     if key not in buckets:
#                         buckets[key] = []
#                         group_keys.append(key)
#                     buckets[key].append(row)

#                 # shuffle group order
#                 rng_groups = np.random.default_rng(seed_ir + 20_000)
#                 rng_groups.shuffle(group_keys)

#                 # shuffle rows within group
#                 rng_rows = np.random.default_rng(seed_ir + 30_000)
#                 interventional_rows_shuffled: List[Dict[str, Any]] = []
#                 for key in group_keys:
#                     rows = buckets[key]
#                     rng_rows.shuffle(rows)
#                     interventional_rows_shuffled.extend(rows)

#                 interventional_rows = interventional_rows_shuffled

#                 # ---------- Full tabular data ----------
#                 all_rows = interventional_rows + obs_rows

#                 # ---------- Build prompt ----------
#                 # Build the rows to feed the unified formatter
#                 if args.int_per_combo > 0:
#                     all_rows_final = all_rows  # already includes interventional + observational
#                 else:
#                     # observational-only: tag rows so the formatter knows there are no interventions
#                     all_rows_final = [
#                         {
#                             "intervened_variable": "Observational",
#                             "intervened_value": None,
#                             **row,
#                         }
#                         for row in obs_rows_base
#                     ]

#                 if args.prompt_style == "matrix":
#                     # For CausalBench-style matrix prompts, we only use observational rows,
#                     # since CausalBench's "training data" is purely observational.
#                     obs_rows_for_matrix = [
#                         r for r in obs_rows_base
#                     ]

#                     dataset_name = os.path.splitext(os.path.basename(args.bif_file))[0]

#                     prompt_text = format_prompt_cb_matrix(
#                         variables_out,
#                         obs_rows_for_matrix,
#                         dataset_name=dataset_name,
#                         include_causal_rules=args.causal_rules,
#                         include_give_steps=args.give_steps,
#                         given_edges=given_edges_named if args.given_edge_frac > 0.0 else None,
#                     )
#                 else:
#                     # Default: "cases" style prompt with interventional rows
#                     prompt_text = format_prompt_with_interventions(
#                         variables_out,
#                         all_rows_final,
#                         variable_map=vmap,
#                         include_causal_rules=args.causal_rules,
#                         include_give_steps=args.give_steps,
#                         given_edges=given_edges_named if args.given_edge_frac > 0.0 else None,
#                     )

#                 # ---------- Write prompt text to its own .txt file ----------
#                 prompt_filename = f"{base_name}_data{i}_shuf{rep}.txt"
#                 prompt_path = prompt_txt_dir / prompt_filename
#                 prompt_path.write_text(prompt_text, encoding="utf-8")

#                 # ---------- CSV: store only path to the prompt file ----------
#                 csv_writer.writerow({
#                     "data_idx": i,
#                     "shuffle_idx": rep,
#                     "prompt_path": str(prompt_path),
#                     "answer_path": str(answer_path),
#                     "given_edges": json.dumps(given_edges_named, ensure_ascii=False),
#                 })



#         print(f"Generated {args.num_prompts * max(1, args.shuffles_per_graph)} prompt,answer pairs.")
#         # print(f"- JSONL: {jsonl_path}")
#         print(f"- CSV:   {csv_path}")
#     finally:
#         try:
#             jsonl_f.close()
#         except Exception:
#             pass
#         try:
#             csv_f.close()
#         except Exception:
#             pass

def main():
    ap = argparse.ArgumentParser(description="Generate N prompt,answer pairs with optional interventional rows.")
    ap.add_argument("--bif-file", default="../causal_graphs/real_data/small_graphs/cancer.bif")
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--obs-per-prompt", type=int, default=100)
    ap.add_argument("--int-per-combo", type=int, default=0)
    ap.add_argument("--intervene-vars", default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--anonymize", action="store_true")
    ap.add_argument("--causal-rules", action="store_true")
    ap.add_argument("--give-steps", action="store_true")
    ap.add_argument("--def-int", action="store_true", help="Include a brief definition of interventions when interventional data are present.")
    ap.add_argument("--shuffles-per-graph", type=int, default=1)
    ap.add_argument("--given-edge-frac", type=float, default=0.0)
    ap.add_argument("--prompt-style", choices=["cases", "matrix"], default="cases")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Directory for outputs; defaults to experiments/prompts/<bif basename>",
    )
    
    # --- UPDATED ROBUSTNESS ARGS ---
    ap.add_argument("--row-order", choices=["random", "sorted", "reverse"], default="random")
    ap.add_argument("--col-order", choices=["original", "reverse", "random", "topo", "reverse_topo"], default="original")

    args = ap.parse_args()
    default_out_dir = os.path.join(
        os.path.dirname(__file__),
        "prompts",
        os.path.splitext(os.path.basename(args.bif_file))[0],
    )
    args.out_dir = args.out_dir or default_out_dir

    # 1. Load Graph
    bif_abs = Path(args.bif_file).resolve(strict=True)
    graph = load_graph_file(str(bif_abs))
    base_variables = normalize_variable_names(graph)
    nvars = len(base_variables)
    codebook = build_codebook(graph, base_variables, str(bif_abs))

    # 2. Base Adjacency
    adj_np = np.asarray(graph.adj_matrix)
    base_adj_bin = (adj_np > 0).astype(int).tolist()

    # 3. Determine Column Order
    col_indices = list(range(nvars))
    
    if args.col_order == "reverse":
        col_indices.reverse()
    elif args.col_order == "random":
        rng_col = np.random.default_rng(args.seed + 999)
        rng_col.shuffle(col_indices)
    elif args.col_order == "topo":
        # Sort Cause -> Effect
        col_indices = get_topological_sort(base_adj_bin)
    elif args.col_order == "reverse_topo":
        # Sort Effect -> Cause
        topo = get_topological_sort(base_adj_bin)
        topo.reverse()
        col_indices = topo
    
    # Apply Column Permutation
    permuted_real_names = [base_variables[i] for i in col_indices]
    
    # Permute Adjacency Matrix
    adj_bin = [[0]*nvars for _ in range(nvars)]
    for r in range(nvars):
        for c in range(nvars):
            old_r, old_c = col_indices[r], col_indices[c]
            adj_bin[r][c] = base_adj_bin[old_r][old_c]

    # 4. Anonymization Map
    vmap = {}
    if args.anonymize:
        for i, name in enumerate(permuted_real_names):
            vmap[name] = f"X{i+1}"
    else:
        for name in permuted_real_names:
            vmap[name] = name

    variables_out = [vmap[name] for name in permuted_real_names]

    # 5. Answer Object
    answer_obj = {
        "variables": variables_out,
        "adjacency_matrix": adj_bin,
    }

    # Helper for Interventions
    if args.int_per_combo > 0 and args.intervene_vars.lower() not in {"none", ""}:
        if args.intervene_vars.lower() == "all":
            intervene_var_names = base_variables
        else:
            intervene_var_names = [s.strip() for s in args.intervene_vars.split(",") if s.strip()]
        intervene_var_idxs = [(base_variables.index(v), v) for v in intervene_var_names]
    else:
        intervene_var_idxs = []
    include_def_int = bool(args.def_int and args.int_per_combo > 0)

    def value_for_display(var_original_name: str, idx: int) -> str:
        idx = int(idx)
        if args.anonymize: return str(idx)
        names = codebook.get(var_original_name, [])
        return names[idx] if 0 <= idx < len(names) else str(idx)

    # 6. Setup Output & FILENAME GENERATION
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_txt_dir = out_dir / "prompt_txt"
    prompt_txt_dir.mkdir(parents=True, exist_ok=True)

    # --- Construct Filename Tags ---
    tags = []
    if args.anonymize: tags.append("anon")
    if getattr(args, "causal_rules", False): tags.append("rules")
    if getattr(args, "give_steps", False): tags.append("steps")
    if hasattr(args, "prompt_style") and args.prompt_style == "matrix":
        tags.append(args.prompt_style)
    
    # ROBUSTNESS TAGS (Explicitly added to filename)
    if args.row_order != "random": tags.append(f"row{args.row_order}")
    if args.col_order != "original": tags.append(f"col{args.col_order}")

    extra_suffix = ("_" + "_".join(tags)) if tags else ""
    
    base_name = (
        f"prompts_obs{args.obs_per_prompt}"
        f"_int{args.int_per_combo}"
        f"_shuf{args.shuffles_per_graph}{extra_suffix}"
    )

    csv_path = out_dir / f"{base_name}.csv"
    answer_path = out_dir / f"{base_name}_answer.json"

    # Write Answer
    # with answer_path.open("w", encoding="utf-8") as f_ans:
    #     json.dump(
    #         {"answer": answer_obj, "given_edges": None}, 
    #         f_ans, ensure_ascii=False, indent=2
    #     )

    # Write CSV Header
    csv_f = csv_path.open("w", newline="", encoding="utf-8")
    fieldnames = ["data_idx", "shuffle_idx", "prompt_path", "answer_path", "given_edges"]
    csv_writer = csv.DictWriter(csv_f, fieldnames=fieldnames, extrasaction="ignore")
    csv_writer.writeheader()

    print(f"Generating -> {csv_path} ...")

    # 7. Generation Loop
    try:
        for i in range(args.num_prompts):
            seed_data = args.seed + i * 1000
            np.random.seed(seed_data)

            # Sample Obs (Base Order)
            arr_obs = graph.sample(batch_size=args.obs_per_prompt, as_array=True)
            obs_rows_base = []
            for r in arr_obs:
                row_orig = {
                    base_variables[j]: value_for_display(base_variables[j], r[j])
                    for j in range(nvars)
                }
                row_disp = {vmap.get(k, k): v for k, v in row_orig.items()}
                row_disp["intervened_variable"] = "Observational"
                row_disp["intervened_value"] = None
                obs_rows_base.append(row_disp)

            # Sample Int (Base Order)
            interventional_rows_base = []
            if intervene_var_idxs:
                rng_int = np.random.default_rng(seed_data + 10_000)
                for var_idx, var_name in intervene_var_idxs:
                    prob_dist = getattr(graph.variables[var_idx], "prob_dist", None)
                    num_categs = getattr(prob_dist, "num_categs", None)
                    if not isinstance(num_categs, int) or num_categs <= 0:
                        num_categs = len(codebook.get(var_name, [])) or 2
                    
                    dataset_size = args.int_per_combo
                    values_vec = rng_int.integers(low=0, high=num_categs, size=dataset_size, dtype=np.int32)
                    arr_int = sample_interventional_values_vec(graph, var_idx, var_name, values_vec)
                    
                    for sample_idx, r in enumerate(arr_int):
                        s_idx = int(values_vec[sample_idx])
                        row_orig = {
                            base_variables[j]: value_for_display(base_variables[j], r[j])
                            for j in range(nvars)
                        }
                        row_disp = {vmap.get(k, k): v for k, v in row_orig.items()}
                        ivar_out = vmap.get(var_name, var_name)
                        interventional_rows_base.append({
                            "intervened_variable": ivar_out,
                            "intervened_value": (str(s_idx) if args.anonymize else value_for_display(var_name, s_idx)),
                            **row_disp,
                        })

            # Shuffle & Sort Logic per Replicate
            for rep in range(args.shuffles_per_graph):
                seed_ir = seed_data + rep
                
                # --- Row Ordering ---
                obs_rows = [r.copy() for r in obs_rows_base]
                rng_obs = np.random.default_rng(seed_ir)

                if args.row_order == "random":
                    rng_obs.shuffle(obs_rows)
                elif args.row_order == "reverse":
                    obs_rows.reverse()
                elif args.row_order == "sorted":
                    # Sort by first displayed variable
                    key_var = variables_out[0]
                    obs_rows.sort(key=lambda x: str(x.get(key_var, "")))

                # Intervention Rows
                int_rows_final = []
                if interventional_rows_base:
                    if args.row_order == "random":
                        # Standard bucket shuffle
                        tmp_rows = [r.copy() for r in interventional_rows_base]
                        buckets = {}
                        for r in tmp_rows:
                            k = (r["intervened_variable"], r["intervened_value"])
                            buckets.setdefault(k, []).append(r)
                        keys = list(buckets.keys())
                        np.random.default_rng(seed_ir+1).shuffle(keys)
                        for k in keys:
                            batch = buckets[k]
                            np.random.default_rng(seed_ir+2).shuffle(batch)
                            int_rows_final.extend(batch)
                    elif args.row_order == "reverse":
                        int_rows_final = interventional_rows_base[::-1]
                    elif args.row_order == "sorted":
                        int_rows_final = sorted(
                            interventional_rows_base, 
                            key=lambda x: str(x.get(variables_out[0], ""))
                        )

                # Final Rows
                if args.int_per_combo > 0:
                    rows_for_prompt = int_rows_final + obs_rows
                else:
                    rows_for_prompt = obs_rows
                
                # For matrix prompts, include whatever data exist (obs + int)
                rows_text_source = rows_for_prompt

                # Text Generation
                dataset_name = os.path.splitext(os.path.basename(args.bif_file))[0]
                if args.prompt_style == "matrix":
                    prompt_text = format_prompt_cb_matrix(
                        variables_out, rows_text_source, dataset_name,
                        args.causal_rules, args.give_steps, include_def_int=include_def_int
                    )
                else:
                    prompt_text = format_prompt_with_interventions(
                        variables_out, rows_text_source, vmap,
                        args.causal_rules, args.give_steps, include_def_int=include_def_int
                    )
                
                # Write Prompt File
                p_filename = f"{base_name}_data{i}_shuf{rep}.txt"
                p_path = prompt_txt_dir / p_filename
                p_path.write_text(prompt_text, encoding="utf-8")

                csv_writer.writerow({
                    "data_idx": i,
                    "shuffle_idx": rep,
                    "prompt_path": str(p_path),
                    "answer_path": str(answer_path),
                    "given_edges": None
                })

    finally:
        csv_f.close()
    
    print("Done.")
if __name__ == "__main__":
    main()
