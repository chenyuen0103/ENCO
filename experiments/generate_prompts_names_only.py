#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Any, Tuple, Optional, Iterator

import numpy as np

# Allow running from experiments/ with repo root one level up
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from causal_graphs.graph_real_world import load_graph_file as _load_graph_file_full  # type: ignore
except Exception:
    _load_graph_file_full = None


def _load_graph_file_light(filename: str) -> SimpleNamespace:
    """
    Minimal BIF loader that only extracts variable names and adjacency.
    Avoids heavyweight deps (e.g., torch) when running names-only prompts.
    """
    text = Path(filename).read_text(encoding="utf-8", errors="ignore")

    var_matches = re.findall(r"variable\s+([^\s\{]+)\s*\{", text)
    variables = [m.strip() for m in var_matches]

    edges: List[Tuple[str, str]] = []
    prob_str = re.findall(r"probability\s+.*?\{[^}]*\}", text, flags=re.S)
    for p_str in prob_str:
        bracks = p_str.split("probability (")[1].split(")")[0]
        if "|" in bracks:
            out = bracks.split("|")[0].strip()
            inputs = [s.strip() for s in bracks.split("|")[1].split(",")]
            edges.extend((inp, out) for inp in inputs)

    name_to_idx = {v: i for i, v in enumerate(variables)}
    n = len(variables)
    adj = [[0] * n for _ in range(n)]
    for src, dst in edges:
        if src in name_to_idx and dst in name_to_idx:
            adj[name_to_idx[src]][name_to_idx[dst]] = 1

    vars_objs = [SimpleNamespace(name=v) for v in variables]
    return SimpleNamespace(variables=vars_objs, adj_matrix=adj)


def load_graph_file(filename: str) -> SimpleNamespace:
    if _load_graph_file_full is not None:
        return _load_graph_file_full(filename)
    return _load_graph_file_light(filename)


# # ------------------------ Utility: variable names ------------------------ #


# def get_topological_sort(adj_matrix: List[List[int]]) -> List[int]:
#     """Returns a list of node indices in topological order (Causes -> Effects)."""
#     n = len(adj_matrix)
#     in_degree = [0] * n
#     for i in range(n):
#         for j in range(n):
#             if adj_matrix[i][j] == 1:
#                 in_degree[j] += 1
    
#     queue = [i for i in range(n) if in_degree[i] == 0]
#     queue.sort() # Deterministic tie-breaking
#     topo_order = []
    
#     in_degree_curr = list(in_degree)
#     while queue:
#         u = queue.pop(0)
#         topo_order.append(u)
#         for v in range(n):
#             if adj_matrix[u][v] == 1:
#                 in_degree_curr[v] -= 1
#                 if in_degree_curr[v] == 0:
#                     queue.append(v)
    
#     if len(topo_order) != n: return list(range(n)) # Cycle fallback
#     return topo_order
# def normalize_variable_names(graph) -> List[str]:
#     """
#     Returns the list of variable names in the order used by the graph.
#     Adjust here if your graph uses a different naming convention.
#     """
#     return [v.name for v in graph.variables]


# # ------------------------ BIF parsing and codebook (kept minimal) ------------------------ #

# def parse_bif_categories(filename: str) -> Dict[str, List[str]]:
#     """
#     Parse state names from a .bif file.

#     Uses the same pattern as your working query_bif.py:
#       variable X { ... { state1, state2, ... } ... }
#     """
#     text = Path(filename).read_text(encoding="utf-8", errors="ignore")
#     mapping: Dict[str, List[str]] = {}
#     pattern = re.compile(
#         r"variable\s+([^\s\{]+)\s*\{[^\{\}]*\{\s*([^\}]*)\s*\}[^\}]*\}",
#         flags=re.S,
#     )
#     for name, states_str in pattern.findall(text):
#         states = [s.strip() for s in states_str.split(",") if s.strip()]
#         mapping[name] = states
#     return mapping


# # ------------------------ Baseline prompt: NAMES ONLY ------------------------ #

# def format_prompt_names_only(
#     variables: List[str],
#     dataset_name: str,
#     include_causal_rules: bool = False,
#     include_give_steps: bool = False,
#     given_edges: Optional[List[Tuple[str, str]]] = None,
#     use_edge_list: bool = False,
# ) -> str:
#     """
#     Baseline prompt: ONLY variable names (no data).
#     LLM must infer a DAG purely from semantic information in the names.

#     Output spec switches by graph size:
#       - If N <= 20: output adjacency_matrix (N x N list of 0/1)
#       - If N > 20:  output edges as list of ["source","target"] pairs
#     """
#     lines: List[str] = []

#     # --- Task description ---
#     lines.append(
#         "You are a causal discovery assistant. Your task is to infer a directed causal graph "
#         f"over the variables of a Bayesian network named {dataset_name}."
#     )
#     lines.append(
#         "You are given only the variable names (no numerical data). Use your general scientific "
#         "and commonsense knowledge about the semantics of the variable names to infer plausible "
#         "direct causal relationships."
#     )

#     # --- OUTPUT INSTRUCTIONS ---
#     lines.append("\n--- OUTPUT INSTRUCTIONS ---")
#     if use_edge_list:
#         if not include_give_steps:
#             lines.extend([
#                 'Respond with a single valid JSON object and nothing else.',
#                 'The object must have exactly two keys: "variables" and "edges".',
#                 '- "variables": the ordered list of variable names given in the SYSTEM VARIABLES section below.',
#                 '- "edges": a list of directed edges, each as ["source","target"], using the SAME spellings as in "variables".',
#                 'Any text, explanation, or markdown outside this JSON object makes the answer invalid.',
#                 'Your first character MUST be "{" and your last character MUST be "}".',
#             ])
#         else:
#             lines.extend([
#                 'You may optionally include a brief explanation first.',
#                 'At the end, you MUST output a single JSON object with exactly two keys: "variables" and "edges".',
#                 '- "variables": the ordered list of variable names given in the SYSTEM VARIABLES section below.',
#                 '- "edges": a list of directed edges, each as ["source","target"], using the SAME spellings as in "variables".',
#                 'The JSON must be syntactically valid (starts with "{" and ends with "}"), '
#                 'must appear on its own line at the end of your answer, and nothing may follow it.',
#             ])
#     else:
#         if not include_give_steps:
#             lines.extend([
#                 'Respond with a single valid JSON object and nothing else.',
#                 'The object must have exactly two keys: "variables" and "adjacency_matrix".',
#                 '- "variables": the ordered list of variable names given in the SYSTEM VARIABLES section below.',
#                 '- "adjacency_matrix": an N x N list of lists of 0/1 integers, where [i][j] = 1 '
#                 'iff there is a directed edge from variables[i] to variables[j], else 0.',
#                 'Any text, explanation, or markdown outside this JSON object makes the answer invalid.',
#                 'Your first character MUST be "{" and your last character MUST be "}".',
#             ])
#         else:
#             lines.extend([
#                 'You may optionally include a brief explanation first.',
#                 'At the end, you MUST output a single JSON object with exactly two keys: "variables" and "adjacency_matrix".',
#                 '- "variables": the ordered list of variable names given in the SYSTEM VARIABLES section below.',
#                 '- "adjacency_matrix": an N x N list of lists of 0/1 integers, where [i][j] = 1 '
#                 'iff there is a directed edge from variables[i] to variables[j], else 0.',
#                 'The JSON must be syntactically valid (starts with "{" and ends with "}"), '
#                 'must appear on its own line at the end of your answer, and nothing may follow it.',
#             ])

#     # --- Optional causal reminders ---
#     if include_causal_rules:
#         lines.extend([
#             "\n--- CAUSAL INFERENCE REMINDERS ---",
#             "- A cause should temporally or logically precede its effect.",
#             "- A variable representing an innate attribute (e.g., age, sex, genetic factors) "
#             "is usually a cause rather than an effect.",
#             "- Avoid cycles: the final causal graph must be a directed acyclic graph (DAG).",
#             "- Some variables may act as confounders, mediators, or colliders; reason about "
#             "which direction is more plausible given their semantics.",
#         ])

#     # --- Known edges section (optional; for ablations) ---
#     if given_edges:
#         lines.append("\n--- KNOWN DIRECT CAUSAL EDGES ---")
#         lines.append(
#             "You are told that the following directed causal relationships are definitely present in the true causal graph:"
#         )
#         for src, dst in given_edges:
#             lines.append(f"- {src} -> {dst}")
#         lines.append(
#             "In your adjacency_matrix output, the entry [i][j] MUST be 1 whenever "
#             "variables[i] = src and variables[j] = dst for one of these edges."
#         )

#     # --- System variables ---
#     lines.append("\n--- SYSTEM VARIABLES (in order) ---")
#     for i, var in enumerate(variables):
#         lines.append(f"{i}: {var}")

#     lines.append(
#         "\nBased only on the semantics of these variable names (and any known edges above), "
#         "output the required JSON object describing a plausible directed acyclic causal graph."
#     )

#     return "\n".join(lines)


# # ------------------------ Edge subsampling (optional) ------------------------ #

# def choose_given_edges(
#     adj_bin: List[List[int]],
#     frac: float,
#     seed: int = 0,
#     max_per_node: int = 2,
# ) -> List[Tuple[int, int]]:
#     """
#     Scheme 1: choose a small, roughly uniform subset of true edges.

#     adj_bin: binary adjacency matrix (list of list of 0/1)
#     frac:    fraction of *all* edges to reveal (e.g. 0.2 = 20%)
#     seed:    RNG seed
#     max_per_node: optional cap on how many given-edges can touch each node.
#                   If None, don't cap.

#     Returns list of (i, j) index pairs where adj_bin[i][j] == 1.
#     """
#     n = len(adj_bin)
#     edges: List[Tuple[int, int]] = [
#         (i, j)
#         for i in range(n)
#         for j in range(n)
#         if adj_bin[i][j] == 1
#     ]

#     if frac <= 0.0 or not edges:
#         return []

#     rng = np.random.default_rng(seed)
#     edges_arr = np.array(edges, dtype=int)
#     rng.shuffle(edges_arr)
#     edges = [tuple(map(int, e)) for e in edges_arr]

#     k_target = max(1, int(round(len(edges) * frac)))

#     if max_per_node is None:
#         # Simple: just take first k_target edges
#         return edges[:k_target]

#     node_counts = [0] * n
#     chosen: List[Tuple[int, int]] = []

#     for (i, j) in edges:
#         if len(chosen) >= k_target:
#             break
#         if node_counts[i] >= max_per_node and node_counts[j] >= max_per_node:
#             continue
#         chosen.append((i, j))
#         node_counts[i] += 1
#         node_counts[j] += 1

#     return chosen


# # ------------------------ MAIN: names-only baseline generator ------------------------ #

# def main():
#     ap = argparse.ArgumentParser(
#         description="Generate prompt,answer pairs for names-only causal discovery."
#     )
#     ap.add_argument(
#         "--bif-file",
#         default="../causal_graphs/real_data/small_graphs/sachs.bif",
#         help="Path to .bif Bayesian network file.",
#     )
#     ap.add_argument(
#         "--num-prompts",
#         type=int,
#         default=1,
#         help="Number of prompt variants to generate.",
#     )
#     ap.add_argument(
#         "--shuffles-per-graph",
#         type=int,
#         default=1,
#         help="How many prompts to generate per graph instance (usually 1 for names-only).",
#     )
#     ap.add_argument(
#         "--seed",
#         type=int,
#         default=0,
#         help="Base RNG seed (used only for subsampling revealed edges).",
#     )
#     ap.add_argument(
#         "--anonymize",
#         action="store_true",
#         help="Replace variable names with X1,X2,... in prompt/answer.",
#     )
#     ap.add_argument(
#         "--causal-rules",
#         action="store_true",
#         help="Include causal inference reminders in the prompt.",
#     )
#     ap.add_argument(
#         "--give-steps",
#         action="store_true",
#         help="Allow the model to provide explanation before the final JSON.",
#     )
#     ap.add_argument(
#         "--given-edge-frac",
#         type=float,
#         default=0.0,
#         help="Fraction of true edges to reveal as known ground-truth edges. "
#              "0.0 means no edges are given (pure names-only baseline).",
#     )

#     args = ap.parse_args()

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

#     # ------------------ Optional: choose given edges ------------------
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

#     # Tag for filenames
#     if args.given_edge_frac > 0.0:
#         frac_pct = int(round(100 * args.given_edge_frac))
#         edge_tag = f"_gedge{frac_pct}"
#     else:
#         edge_tag = ""

#     # Directed edge list for convenience
#     edge_list = [
#         (variables_out[i], variables_out[j])
#         for i in range(nvars)
#         for j in range(nvars)
#         if adj_bin[i][j] == 1
#     ]

#     use_edge_list = nvars > 20

#     # Ground-truth answer object
#     answer_obj = {
#         "variables": variables_out,
#         "edges": edge_list if use_edge_list else None,
#         "adjacency_matrix": adj_bin if not use_edge_list else None,
#     }

#     # ---------- Outputs ----------
#     out_dir = Path(
#         os.path.join(
#             os.path.dirname(__file__),
#             "prompts_names_only",
#             os.path.splitext(os.path.basename(args.bif_file))[0],
#         )
#     )
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # Directory to store prompt text files
#     prompt_txt_dir = out_dir / "prompt_txt"
#     prompt_txt_dir.mkdir(parents=True, exist_ok=True)

#     # Suffix tags for configuration
#     tags = []
#     if args.anonymize:
#         tags.append("anon")
#     if getattr(args, "causal_rules", False):
#         tags.append("rules")
#     if getattr(args, "give_steps", False):
#         tags.append("steps")

#     extra_suffix = ("_" + "_".join(tags)) if tags else ""

#     base_name = (
#         f"prompts_names_only"
#         f"_shuf{args.shuffles_per_graph}{edge_tag}{extra_suffix}"
#     )

#     # jsonl_path = out_dir / f"{base_name}.jsonl"
#     csv_path = out_dir / f"{base_name}.csv"
#     answer_path = out_dir / f"{base_name}_answer.json"

#     # Save shared answer JSON (true DAG)
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

#     # jsonl_f = jsonl_path.open("w", encoding="utf-8")

#     csv_f = csv_path.open("w", newline="", encoding="utf-8")
#     fieldnames = ["data_idx", "shuffle_idx", "prompt_path", "answer_path", "given_edges"]
#     csv_writer = csv.DictWriter(csv_f, fieldnames=fieldnames, extrasaction="ignore")
#     csv_writer.writeheader()

#     dataset_name = os.path.splitext(os.path.basename(args.bif_file))[0]

#     try:
#         for i in range(args.num_prompts):
#             # No data sampling: each data_idx just corresponds to a prompt variant.
#             for rep in range(args.shuffles_per_graph):
#                 prompt_text = format_prompt_names_only(
#                     variables_out,
#                     dataset_name=dataset_name,
#                     include_causal_rules=args.causal_rules,
#                     include_give_steps=args.give_steps,
#                     given_edges=given_edges_named if args.given_edge_frac > 0.0 else None,
#                     use_edge_list=use_edge_list,
#                 )

#                 prompt_filename = f"{base_name}_data{i}_shuf{rep}.txt"
#                 prompt_path = prompt_txt_dir / prompt_filename
#                 prompt_path.write_text(prompt_text, encoding="utf-8")

#                 # record: Dict[str, Any] = {
#                 #     "data_idx": i,
#                 #     "shuffle_idx": rep,
#                 #     "prompt": prompt_text,
#                 #     "answer": answer_obj,
#                 #     "given_edges": given_edges_named if args.given_edge_frac > 0.0 else None,
#                 # }
#                 # jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")

#                 csv_writer.writerow({
#                     "data_idx": i,
#                     "shuffle_idx": rep,
#                     "prompt_path": str(prompt_path),
#                     "answer_path": str(answer_path),
#                     "given_edges": json.dumps(given_edges_named, ensure_ascii=False),
#                 })

#         print(f"Generated {args.num_prompts * max(1, args.shuffles_per_graph)} names-only prompt,answer pairs.")
#         # print(f"- JSONL: {jsonl_path}")
#         print(f"- CSV:   {csv_path}")
#     finally:
#         # try:
#         #     jsonl_f.close()
#         # except Exception:
#         #     pass
#         try:
#             csv_f.close()
#         except Exception:
#             pass


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# import argparse
# import json
# import os
# import re
# import sys
# from pathlib import Path
# from typing import List, Dict, Optional

# import numpy as np

# ------------------------ Helpers ------------------------ #

def get_topological_sort(adj_matrix: List[List[int]]) -> List[int]:
    """Returns a list of node indices in topological order (Causes -> Effects)."""
    n = len(adj_matrix)
    in_degree = [0] * n
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] == 1:
                in_degree[j] += 1
    
    queue = [i for i in range(n) if in_degree[i] == 0]
    queue.sort() # Deterministic tie-breaking
    topo_order = []
    
    in_degree_curr = list(in_degree)
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in range(n):
            if adj_matrix[u][v] == 1:
                in_degree_curr[v] -= 1
                if in_degree_curr[v] == 0:
                    queue.append(v)
    
    if len(topo_order) != n: return list(range(n)) # Cycle fallback
    return topo_order

def normalize_variable_names(graph) -> List[str]:
    return [v.name for v in graph.variables]

def format_names_only_prompt(
    variables: List[str],
    dataset_name: str,
    include_causal_rules: bool = False
) -> str:
    lines = []
    lines.append("You are a highly intelligent causal discovery expert.")
    lines.append(f"We are studying a system called '{dataset_name}'.")
    lines.append("Based on your background knowledge of the world and these variables, infer the directed causal graph.")
    
    lines.append("\n--- VARIABLES ---")
    # Explicitly show the order
    lines.append(f"The variables in the system are: {', '.join(variables)}")
    
    lines.append("\n--- OUTPUT INSTRUCTIONS ---")
    lines.append('Respond with a single valid JSON object and nothing else.')
    lines.append('The object must have exactly two keys: "variables" and "adjacency_matrix".')
    lines.append('- "variables": the list of variable names exactly as shown above.')
    lines.append('- "adjacency_matrix": an N x N list of lists of 0/1 integers.')
    
    if include_causal_rules:
        lines.append("\n--- REMINDER ---")
        lines.append("Recall that X->Y implies a causal mechanism where manipulating X changes Y.")

    lines.append("\nOutput the JSON now.")
    return "\n".join(lines)


def iter_names_only_prompts_in_memory(
    *,
    bif_file: str,
    num_prompts: int,
    seed: int,
    col_order: str,
    anonymize: bool,
    causal_rules: bool,
) -> tuple[str, dict[str, Any], Iterator[dict[str, Any]]]:
    """
    Generate names-only prompts in-memory.
    Returns (base_name, answer_obj, iterator of rows with prompt_text).
    """
    bif_abs = Path(bif_file).resolve(strict=True)
    graph = load_graph_file(str(bif_abs))
    base_variables = normalize_variable_names(graph)
    nvars = len(base_variables)

    adj_np = np.asarray(graph.adj_matrix)
    base_adj_bin = (adj_np > 0).astype(int).tolist()

    col_indices = list(range(nvars))
    if col_order == "reverse":
        col_indices.reverse()
    elif col_order == "random":
        rng_col = np.random.default_rng(seed + 999)
        rng_col.shuffle(col_indices)
    elif col_order == "topo":
        col_indices = get_topological_sort(base_adj_bin)
    elif col_order == "reverse_topo":
        topo = get_topological_sort(base_adj_bin)
        topo.reverse()
        col_indices = topo

    permuted_real_names = [base_variables[i] for i in col_indices]

    adj_bin = [[0] * nvars for _ in range(nvars)]
    for r in range(nvars):
        for c in range(nvars):
            old_r, old_c = col_indices[r], col_indices[c]
            adj_bin[r][c] = base_adj_bin[old_r][old_c]

    vmap: Dict[str, str] = {}
    if anonymize:
        for i, name in enumerate(permuted_real_names):
            vmap[name] = f"X{i+1}"
    else:
        for name in permuted_real_names:
            vmap[name] = name

    variables_out = [vmap[name] for name in permuted_real_names]
    answer_obj = {
        "variables": variables_out,
        "adjacency_matrix": adj_bin,
    }

    base_name = "prompts_names_only"
    if col_order != "original":
        base_name += f"_col{col_order.capitalize()}"

    dataset_name = os.path.splitext(os.path.basename(bif_file))[0]
    prompt_text = format_names_only_prompt(variables_out, dataset_name, causal_rules)

    def _iter() -> Iterator[dict[str, Any]]:
        for i in range(num_prompts):
            yield {
                "data_idx": i,
                "shuffle_idx": 0,
                "prompt_text": prompt_text,
                "given_edges": None,
            }

    return base_name, answer_obj, _iter()

# ------------------------ Main ------------------------ #

def main():
    ap = argparse.ArgumentParser(description="Generate 'Names Only' prompts (No Data).")
    
    # --- Critical Arguments (Used) ---
    ap.add_argument("--bif-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--col-order", choices=["original", "reverse", "random", "topo", "reverse_topo"], default="original")
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--anonymize", action="store_true")
    ap.add_argument("--causal-rules", action="store_true")
    
    # --- Dummy Arguments (Accepted to ignore) ---
    # These are passed by the orchestrator but irrelevant for names-only
    ap.add_argument("--prompt-style", default="cases")
    ap.add_argument("--obs-per-prompt", type=int, default=0)
    ap.add_argument("--int-per-combo", type=int, default=0)
    ap.add_argument("--row-order", default="random")
    ap.add_argument("--intervene-vars", default="none")
    ap.add_argument("--shuffles-per-graph", type=int, default=1)
    ap.add_argument("--given-edge-frac", type=float, default=0.0)
    ap.add_argument("--give-steps", action="store_true")

    args = ap.parse_args()

    # 1. Load Graph
    bif_abs = Path(args.bif_file).resolve(strict=True)
    graph = load_graph_file(str(bif_abs))
    base_variables = normalize_variable_names(graph)
    nvars = len(base_variables)
    
    adj_np = np.asarray(graph.adj_matrix)
    base_adj_bin = (adj_np > 0).astype(int).tolist()

    # 2. Determine Column Order (This affects the prompt list and answer key)
    col_indices = list(range(nvars))
    
    if args.col_order == "reverse":
        col_indices.reverse()
    elif args.col_order == "random":
        rng_col = np.random.default_rng(args.seed + 999)
        rng_col.shuffle(col_indices)
    elif args.col_order == "topo":
        col_indices = get_topological_sort(base_adj_bin)
    elif args.col_order == "reverse_topo":
        topo = get_topological_sort(base_adj_bin)
        topo.reverse()
        col_indices = topo

    # Permute variables and matrix
    permuted_real_names = [base_variables[i] for i in col_indices]
    
    adj_bin = [[0]*nvars for _ in range(nvars)]
    for r in range(nvars):
        for c in range(nvars):
            old_r, old_c = col_indices[r], col_indices[c]
            adj_bin[r][c] = base_adj_bin[old_r][old_c]

    # 3. Anonymization (If requested, though names-only on anon vars is random guessing)
    vmap = {}
    if args.anonymize:
        for i, name in enumerate(permuted_real_names):
            vmap[name] = f"X{i+1}"
    else:
        for name in permuted_real_names:
            vmap[name] = name

    variables_out = [vmap[name] for name in permuted_real_names]

    # 4. Output Setup
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_txt_dir = out_dir / "prompt_txt"
    prompt_txt_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"prompts_names_only"
    if args.col_order != "original": base_name += f"_col{args.col_order.capitalize()}"
    
    csv_path = out_dir / f"{base_name}.csv"
    answer_path = out_dir / f"{base_name}_answer.json"

    # Write Answer Key
    # with answer_path.open("w", encoding="utf-8") as f_ans:
    #     json.dump({
    #         "answer": {"variables": variables_out, "adjacency_matrix": adj_bin},
    #         "given_edges": None
    #     }, f_ans, indent=2)

    # Write CSV
    csv_f = csv_path.open("w", newline="", encoding="utf-8")
    fieldnames = ["data_idx", "shuffle_idx", "prompt_path", "answer_path", "given_edges"]
    csv_writer = csv.DictWriter(csv_f, fieldnames=fieldnames, extrasaction="ignore")
    csv_writer.writeheader()

    # 5. Generate Prompts (Since there's no data, the prompt is identical for all 'replicates')
    dataset_name = os.path.splitext(os.path.basename(args.bif_file))[0]
    prompt_text = format_names_only_prompt(variables_out, dataset_name, args.causal_rules)
    
    print(f"Generating 'Names Only' prompts into {out_dir}...")

    for i in range(args.num_prompts):
        p_filename = f"{base_name}_idx{i}.txt"
        p_path = prompt_txt_dir / p_filename
        p_path.write_text(prompt_text, encoding="utf-8")

        csv_writer.writerow({
            "data_idx": i,
            "shuffle_idx": 0,
            "prompt_path": str(p_path),
            "answer_path": str(answer_path),
            "given_edges": None
        })

    csv_f.close()
    print("Done.")

if __name__ == "__main__":
    main()
