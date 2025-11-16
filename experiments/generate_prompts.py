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
    arr_all: np.ndarray | None = None

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

def format_prompt_with_interventions(
    variables: List[str],
    all_rows: List[Dict[str, Any]],
    variable_map = None,
    include_causal_rules: bool = False,
    include_give_steps: bool = False,
    given_edges = None,
) -> str:

    """
    Build a prompt that presents mixed observational + interventional data
    and asks for a JSON adjacency matrix over `variables`.
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
        "You are a causal discovery assistant. From the data below, infer a directed causal graph "
        "over the given variables."
    )

    # ---------- OUTPUT INSTRUCTIONS ----------
    lines.append("\n--- OUTPUT INSTRUCTIONS ---")
    if not include_give_steps:
        # STRICT JSON-ONLY MODE (good for eval)
        lines.append(
            'Respond with a single valid JSON object and nothing else.'
        )
        lines.append(
            'The object must have exactly two keys: "variables" and "adjacency_matrix".'
        )
        lines.append(
            '- "variables": the ordered list of variable names given in the SYSTEM VARIABLES section below.'
        )
        lines.append(
            '- "adjacency_matrix": an N x N list of lists of 0/1 integers, where [i][j] = 1 '
            'iff there is a directed edge from variables[i] to variables[j], else 0.'
        )
        lines.append(
            'Any text, explanation, or markdown outside this JSON object makes the answer invalid.'
        )
        lines.append(
            'Your first character MUST be "{" and your last character MUST be "}".'
        )
    else:
        # EXPLANATION + JSON MODE (brief reasoning allowed, JSON at the end)
        lines.append(
            'You may optionally include a brief explanation first.'
        )
        lines.append(
            'At the end, you MUST output a single JSON object with exactly two keys: "variables" and "adjacency_matrix".'
        )
        lines.append(
            '- "variables": the ordered list of variable names given in the SYSTEM VARIABLES section below.'
        )
        lines.append(
            '- "adjacency_matrix": an N x N list of lists of 0/1 integers, where [i][j] = 1 '
            'iff there is a directed edge from variables[i] to variables[j], else 0.'
        )
        lines.append(
            'The JSON must be syntactically valid (starts with "{" and ends with "}"), '
            'must appear on its own line at the end of your answer, and nothing may follow it.'
        )

    # ---------- Optional causal reminders ----------
    if include_causal_rules:
        lines.append("\n--- CAUSAL INFERENCE REMINDERS ---")
        lines.append("- Confounder: a variable that causes two others.")
        lines.append("- Mediator: lies on a path X -> M -> Y.")
        lines.append("- Collider: a common effect of two variables; avoid conditioning on colliders.")
        lines.append("- Backdoor paths: block backdoor paths into a cause when estimating its effect.")
        lines.append("- Interventions: do(X) cuts all incoming edges into X; use changes in other variables to orient edges.")
        lines.append("- The final output must be a DAG (no directed cycles).")


    # ---------- Known edges section (optional) ----------
    if given_edges:
        lines.append("\n--- KNOWN DIRECT CAUSAL EDGES ---")
        lines.append(
            "You are told that the following directed causal relationships "
            "are definitely present in the true causal graph:"
        )
        for src, dst in given_edges:
            lines.append(f"- {src} -> {dst}")


    # ---------- System variables ----------
    lines.append("\n--- SYSTEM VARIABLES (in order) ---")
    for i, var in enumerate(variables):
        lines.append(f"{i}: {var}")

    # ---------- Observational data ----------
    if obs_rows:
        lines.append("\n--- OBSERVATIONAL DATA ---")
        lines.append("Cases observed without intervention:")
        for i, r in enumerate(obs_rows):
            clauses = [f"{h} = {r[h]}" for h in variables]
            sentence = f"Case {i+1}: " + ", ".join(clauses) + "."
            lines.append(sentence)

    # ---------- Interventional data ----------
    if interventions:
        lines.append("\n--- INTERVENTIONAL DATA ---")
        lines.append("Cases observed under specific interventions:")
        for (var, val), inter_rows in interventions.items():
            display_var = variable_map.get(var, var) if variable_map else var
            lines.append(
                f"\nWhen an intervention sets {display_var} to {val}, the following cases were observed:"
            )
            for i, r in enumerate(inter_rows):
                clauses = [f"{h} = {r[h]}" for h in variables]
                sentence = f"Case {i+1}: " + ", ".join(clauses) + "."
                lines.append(sentence)

    lines.append("\n--- END OF DATA ---")

    # ---------- High-level reasoning hints (optional) ----------
    has_obs = bool(obs_rows)
    has_interv = bool(interventions)

    if include_give_steps:
        if has_obs and has_interv:
            lines.append(
                "\nWhen deciding on the causal graph, you may follow these steps "
                "(you do NOT need to describe them explicitly):\n"
                "1. Use the observational data to infer conditional (in)dependencies and identify a plausible Markov equivalence class.\n"
                "2. Use the interventional data to distinguish between graphs in that class and select a single directed graph.\n"
                "Then produce the JSON object described in the OUTPUT INSTRUCTIONS section."
            )
        elif has_obs and not has_interv:
            lines.append(
                "\nWhen deciding on the causal graph, you may follow these steps "
                "(you do NOT need to describe them explicitly):\n"
                "1. Use the observational data to infer conditional (in)dependencies among the variables.\n"
                "2. Choose a directed acyclic graph (DAG) consistent with these (in)dependencies.\n"
                "Then produce the JSON object described in the OUTPUT INSTRUCTIONS section."
            )
        elif has_interv and not has_obs:
            lines.append(
                "\nWhen deciding on the causal graph, you may follow these steps "
                "(you do NOT need to describe them explicitly):\n"
                "1. For each intervention do(X = x), treat incoming edges into X as cut and use the changes in other variables "
                "to orient edges into or out of X.\n"
                "2. Combine information from all interventions into a single DAG.\n"
                "Then produce the JSON object described in the OUTPUT INSTRUCTIONS section."
            )
        else:
            lines.append(
                "\nThen produce the JSON object described in the OUTPUT INSTRUCTIONS section."
            )
    else:
        if has_obs and has_interv:
            lines.append(
                "\nBased on ALL the data (observational and interventional), "
                "produce the JSON object described in the OUTPUT INSTRUCTIONS section."
            )
        elif has_obs and not has_interv:
            lines.append(
                "\nBased on the observational data, "
                "produce the JSON object described in the OUTPUT INSTRUCTIONS section."
            )
        elif has_interv and not has_obs:
            lines.append(
                "\nBased on the interventional data, "
                "produce the JSON object described in the OUTPUT INSTRUCTIONS section."
            )
        else:
            lines.append(
                "\nProduce the JSON object described in the OUTPUT INSTRUCTIONS section."
            )

    return "\n".join(lines)




def format_prompt_without_intervention(
    variables: List[str],
    obs_rows: List[Dict[str, Any]],
    include_causal_rules: bool = False,
    include_give_steps: bool = False,
    given_edges = None,
) -> str:
    """
    Build a prompt that presents purely observational data and asks
    for a JSON adjacency matrix over `variables`.

    `variables` : ordered list of variable names (as they should appear in JSON).
    `obs_rows`  : list of rows; each row is a dict mapping each variable name
                  to its observed value. Extra keys (like 'intervened_variable')
                  are ignored.
    """
    lines: List[str] = []

    # ---------- Task description ----------
    lines.append(
        "You are a causal discovery assistant. From the observational data below, "
        "infer a directed causal graph over the given variables."
    )

    # ---------- OUTPUT INSTRUCTIONS ----------
    lines.append("\n--- OUTPUT INSTRUCTIONS ---")
    if not include_give_steps:
        # STRICT JSON-ONLY MODE (good for eval)
        lines.append(
            'Respond with a single valid JSON object and nothing else.'
        )
        lines.append(
            'The object must have exactly two keys: "variables" and "adjacency_matrix".'
        )
        lines.append(
            '- "variables": the ordered list of variable names given in the SYSTEM VARIABLES section below.'
        )
        lines.append(
            '- "adjacency_matrix": an N x N list of lists of 0/1 integers, where [i][j] = 1 '
            'iff there is a directed edge from variables[i] to variables[j], else 0.'
        )
        lines.append(
            'Any text, explanation, or markdown outside this JSON object makes the answer invalid.'
        )
        lines.append(
            'Your first character MUST be "{" and your last character MUST be "}".'
        )
    else:
        # EXPLANATION + JSON MODE (brief reasoning allowed, JSON at the end)
        lines.append(
            'You may optionally include a brief explanation first.'
        )
        lines.append(
            'At the end, you MUST output a single JSON object with exactly two keys: "variables" and "adjacency_matrix".'
        )
        lines.append(
            '- "variables": the ordered list of variable names given in the SYSTEM VARIABLES section below.'
        )
        lines.append(
            '- "adjacency_matrix": an N x N list of lists of 0/1 integers, where [i][j] = 1 '
            'iff there is a directed edge from variables[i] to variables[j], else 0.'
        )
        lines.append(
            'The JSON must be syntactically valid (starts with "{" and ends with "}"), '
            'must appear on its own line at the end of your answer, and nothing may follow it.'
        )

    # ---------- Optional causal reminders ----------
    if include_causal_rules:
        lines.append("\n--- CAUSAL INFERENCE REMINDERS ---")
        lines.append("- Confounder: a variable that causes two others.")
        lines.append("- Mediator: lies on a path X -> M -> Y.")
        lines.append("- Collider: a common effect of two variables; avoid conditioning on colliders.")
        lines.append("- Use conditional independencies to constrain the possible DAGs.")
        lines.append("- The final output must be a DAG (no directed cycles).")

    # ---------- Known edges section (optional) ----------
    if given_edges:
        lines.append("\n--- KNOWN DIRECT CAUSAL EDGES ---")
        lines.append(
            "You are told that the following directed causal relationships "
            "are definitely present in the true causal graph:"
        )
        for src, dst in given_edges:
            lines.append(f"- {src} -> {dst}")
        lines.append(
            "In your adjacency_matrix output, the entry [i][j] MUST be 1 whenever "
            "variables[i] = src and variables[j] = dst for one of these edges."
        )

    # ---------- System variables ----------
    lines.append("\n--- SYSTEM VARIABLES (in order) ---")
    for i, var in enumerate(variables):
        lines.append(f"{i}: {var}")

    # ---------- Observational data ----------
    lines.append("\n--- OBSERVATIONAL DATA ---")
    if obs_rows:
        lines.append("Cases observed without intervention:")
        for i, r in enumerate(obs_rows):
            clauses = [f"{h} = {r[h]}" for h in variables]
            sentence = f"Case {i+1}: " + ", ".join(clauses) + "."
            lines.append(sentence)
    else:
        lines.append("No observational cases were provided.")

    lines.append("\n--- END OF DATA ---")

    # ---------- High-level reasoning hints (optional) ----------
    if include_give_steps:
        lines.append(
            "\nWhen deciding on the causal graph, you may follow these steps "
            "(you do NOT need to describe them explicitly):\n"
            "1. Use the observational data to infer conditional (in)dependencies among the variables.\n"
            "2. Choose a directed acyclic graph (DAG) consistent with these (in)dependencies.\n"
            "Then produce the JSON object described in the OUTPUT INSTRUCTIONS section."
        )
    else:
        lines.append(
            "\nBased on the observational data, "
            "produce the JSON object described in the OUTPUT INSTRUCTIONS section."
        )

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


def main():
    ap = argparse.ArgumentParser(description="Generate N prompt,answer pairs with optional interventional rows.")
    ap.add_argument("--bif-file", default="../causal_graphs/real_data/small_graphs/cancer.bif",
                    help="Path to .bif Bayesian network file.")
    ap.add_argument("--num-prompts", type=int, default=10, help="Number of pairs to generate.")
    ap.add_argument("--obs-per-prompt", type=int, default=5000, help="Observational samples per prompt.")
    ap.add_argument("--int-per-combo", type=int, default=200,
                    help="Interventional samples PER (variable,state) combo. 0 disables interventions.")
    ap.add_argument("--intervene-vars", default="all",
                    help='Comma-separated variable names, or "all", or "none".')
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed; prompt i uses seed+i.")
    ap.add_argument("--anonymize", action="store_true", help="Replace variable names with X1,X2,... in prompt/answer.")
    ap.add_argument("--causal-rules", action="store_true",
                    help="Include causal inference reminders in the prompt.")
    ap.add_argument("--give-steps", action="store_true",
                    help="Append step-by-step guidance on using observational and interventional data.")
    # ap.add_argument("--out-dir", default="./prompts/cancer", help="Output directory.")
    ap.add_argument("--shuffles-per-graph", type=int, default=3,
                    help="How many independent shuffles to generate per graph.")
    ap.add_argument(
        "--given-edge-frac",
        type=float,
        default=0.0,
        help="Fraction of true edges to reveal as known ground-truth edges (Scheme 1). "
             "0.0 means no edges are given.",
    )

    args = ap.parse_args()
    args.out_dir = os.path.join(
        os.path.dirname(__file__),
        "prompts",
        os.path.splitext(os.path.basename(args.bif_file))[0],
    )
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
    # ------------------ Scheme 1: choose given edges ------------------
    if args.given_edge_frac > 0.0:
        given_edges_idx = choose_given_edges(
            adj_bin,
            frac=args.given_edge_frac,
            seed=args.seed,
            max_per_node=2,  # tweak if you like
        )
        # convert from indices to displayed variable names (respecting anonymization)
        given_edges_named: List[Tuple[str, str]] = [
            (variables_out[i], variables_out[j]) for (i, j) in given_edges_idx
        ]
    else:
        given_edges_idx = []
        given_edges_named = []

    if args.given_edge_frac > 0.0:
        # Option A: encode fraction as percentage, e.g. _gedge20 for 0.2
        frac_pct = int(round(100 * args.given_edge_frac))
        edge_tag = f"_gedge{frac_pct}"
        # Option B (alternative): encode absolute # of revealed edges:
        # edge_tag = f"_gedge{len(given_edges_idx)}"
    else:
        edge_tag = ""
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

    # NEW: directory to store prompt text files
    prompt_txt_dir = out_dir / "prompt_txt"
    prompt_txt_dir.mkdir(parents=True, exist_ok=True)
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
        f"_shuf{args.shuffles_per_graph}{edge_tag}{extra_suffix}"
    )

    jsonl_path = out_dir / f"{base_name}.jsonl"
    csv_path = out_dir / f"{base_name}.csv"

    jsonl_f = jsonl_path.open("w", encoding="utf-8")

    csv_f = csv_path.open("w", newline="", encoding="utf-8")
    fieldnames = ["data_idx", "shuffle_idx", "prompt_path", "answer", "given_edges"]
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
            # Build *base* interventional rows ONCE per data_idx
            interventional_rows_base: List[Dict[str, Any]] = []
            if intervene_var_idxs and args.int_per_combo > 0:
                rng_int = np.random.default_rng(seed_data + 10_000)

                # For fair comparison with ENCO:
                # - Treat int_per_combo as "dataset_size" per variable.
                # - For each sample, draw the clamped value uniformly over states.
                for var_idx, var_name in intervene_var_idxs:
                    # Determine number of categories for this variable
                    prob_dist = getattr(graph.variables[var_idx], "prob_dist", None)
                    num_categs = getattr(prob_dist, "num_categs", None)
                    if not isinstance(num_categs, int) or num_categs <= 0:
                        num_categs = len(codebook.get(var_name, [])) or 2

                    dataset_size = args.int_per_combo  # ENCO-style: #samples per variable

                    values_vec = rng_int.integers(
                        low=0,
                        high=num_categs,
                        size=dataset_size,
                        dtype=np.int32,
                    )

                    # Sample interventional data with *per-sample* clamped values
                    arr_int = sample_interventional_values_vec(
                        graph,
                        var_idx=var_idx,
                        var_name=var_name,
                        values_vec=values_vec,
                        as_array=True,
                    )

                    # Build rows; note each row's "intervened_value" can differ
                    for sample_idx, r in enumerate(arr_int):
                        s_idx = int(values_vec[sample_idx])

                        row_orig = {
                            variables[j]: value_for_display(variables[j], r[j])
                            for j in range(nvars)
                        }
                        row_disp = {vmap.get(k, k): v for k, v in row_orig.items()}
                        ivar_out = vmap.get(var_name, var_name)

                        interventional_rows_base.append({
                            "intervened_variable": ivar_out,
                            "intervened_value": (
                                str(s_idx) if args.anonymize
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
                # ---------- Build prompt ----------
                if args.int_per_combo > 0:
                    prompt_text = format_prompt_with_interventions(
                        variables_out,
                        all_rows,
                        variable_map=vmap,
                        include_causal_rules=args.causal_rules,
                        include_give_steps=args.give_steps,
                        given_edges=given_edges_named if args.given_edge_frac > 0.0 else None,
                    )
                else:
                    prompt_text = format_prompt_without_intervention(
                        variables_out,
                        obs_rows_base,
                        include_causal_rules=args.causal_rules,
                        include_give_steps=args.give_steps,
                        given_edges=given_edges_named if args.given_edge_frac > 0.0 else None,
                    )

                # ---------- Write prompt text to its own .txt file ----------
                prompt_filename = f"{base_name}_data{i}_shuf{rep}.txt"
                prompt_path = prompt_txt_dir / prompt_filename
                prompt_path.write_text(prompt_text, encoding="utf-8")

                # ---------- JSONL record (can still include full prompt) ----------
                record = {
                    "data_idx": i,
                    "shuffle_idx": rep,
                    "prompt": prompt_text,  # keep full text in JSONL if you want
                    "answer": {
                        "variables": variables_out,
                        "adjacency_matrix": adj_bin,
                    },
                    "rows": all_rows,
                    "given_edges": given_edges_named if args.given_edge_frac > 0.0 else None,
                }
                jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # ---------- CSV: store only path to the prompt file ----------
                csv_writer.writerow({
                    "data_idx": i,
                    "shuffle_idx": rep,
                    "prompt_path": str(prompt_path),
                    "answer": json.dumps(record["answer"], ensure_ascii=False),
                    "given_edges": json.dumps(given_edges_named, ensure_ascii=False),
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
