#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

# Allow running from experiments/ with repo root one level up
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# NOTE: We intentionally avoid importing torch-dependent graph loaders at module import time.
# This file also contains prompt formatting utilities that are useful without torch installed.
# When sampling from BIF graphs, we lazy-import the loader inside the relevant functions.

LARGE_GRAPH_EDGE_LIST_THRESHOLD = 100


def _use_edge_list_output(variables: List[str]) -> bool:
    return len(variables) > LARGE_GRAPH_EDGE_LIST_THRESHOLD

def _load_graph_file(path: str):  # type: ignore
    from causal_graphs.graph_real_world import load_graph_file  # type: ignore
    return load_graph_file(path)


def _build_output_contract_lines(
    *,
    output_edge_list: bool,
    require_think_answer_blocks: bool = True,
) -> List[str]:
    if output_edge_list:
        json_key = "edges"
        json_field_desc = '- "edges": [["source","target"], ...] using exact variable names.'
    else:
        json_key = "adjacency_matrix"
        json_field_desc = '- "adjacency_matrix": N x N 0/1 matrix in declared variable order.'

    if require_think_answer_blocks:
        return [
            "Output exactly: <think>...</think><answer>...</answer>.",
            "Keep <think> concise (minimal necessary reasoning only).",
            f'Inside <answer>, output exactly one JSON object with key "{json_key}".',
            json_field_desc,
            "No extra text before, between, or after the two blocks.",
            'The JSON in <answer> must start with "{" and end with "}".',
        ]

    return [
        "Output exactly one JSON object and nothing else.",
        f'The object must contain exactly one key: "{json_key}".',
        json_field_desc,
        "No explanation, markdown, or extra text.",
        'First char must be "{", last char must be "}".',
    ]


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


def iter_prompts_in_memory(
    *,
    bif_file: str,
    num_prompts: int,
    shuffles_per_graph: int,
    seed: int,
    prompt_style: str,
    obs_per_prompt: int,
    int_per_combo: int,
    row_order: str,
    col_order: str,
    anonymize: bool,
    causal_rules: bool,
    give_steps: bool,
    def_int: bool,
    intervene_vars: str,
    thinking_tags: bool = True,
) -> tuple[str, dict[str, Any], Iterator[dict[str, Any]]]:
    """
    Generate prompts in-memory (no prompt files, no prompt CSV).
    Returns (base_name, answer_obj, iterator of rows with prompt_text).
    """
    bif_abs = Path(bif_file).resolve(strict=True)
    graph = _load_graph_file(str(bif_abs))
    base_variables = normalize_variable_names(graph)
    nvars = len(base_variables)
    codebook = build_codebook(graph, base_variables, str(bif_abs))

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

    if int_per_combo > 0 and intervene_vars.lower() not in {"none", ""}:
        if intervene_vars.lower() == "all":
            intervene_var_names = base_variables
        else:
            intervene_var_names = [s.strip() for s in intervene_vars.split(",") if s.strip()]
        intervene_var_idxs = [(base_variables.index(v), v) for v in intervene_var_names]
    else:
        intervene_var_idxs = []
    include_def_int = bool(def_int and int_per_combo > 0)

    def value_for_display(var_original_name: str, idx: int) -> str:
        idx = int(idx)
        if anonymize:
            return str(idx)
        names = codebook.get(var_original_name, [])
        return names[idx] if 0 <= idx < len(names) else str(idx)

    tags = []
    if anonymize:
        tags.append("anon")
    if causal_rules:
        tags.append("rules")
    if give_steps:
        tags.append("steps")
    if thinking_tags:
        tags.append("thinktags")
    if prompt_style in {"matrix", "summary_joint"}:
        tags.append(prompt_style)
    if row_order != "random":
        tags.append(f"row{row_order}")
    if col_order != "original":
        tags.append(f"col{col_order}")
    extra_suffix = ("_" + "_".join(tags)) if tags else ""

    base_name = (
        f"prompts_obs{obs_per_prompt}"
        f"_int{int_per_combo}"
        f"_shuf{shuffles_per_graph}"
        f"_p{num_prompts}{extra_suffix}"
    )

    def _iter() -> Iterator[dict[str, Any]]:
        for i in range(num_prompts):
            seed_data = seed + i * 1000
            np.random.seed(seed_data)

            arr_obs = graph.sample(batch_size=obs_per_prompt, as_array=True)
            obs_rows_base = []
            obs_rows_num: List[List[float]] = []
            for r in arr_obs:
                row_orig = {
                    base_variables[j]: value_for_display(base_variables[j], r[j])
                    for j in range(nvars)
                }
                row_disp = {vmap.get(k, k): v for k, v in row_orig.items()}
                row_disp["intervened_variable"] = "Observational"
                row_disp["intervened_value"] = None
                obs_rows_base.append(row_disp)
                if obs_per_prompt > 0:
                    obs_rows_num.append([float(r[idx]) for idx in col_indices])

            interventional_rows_base = []
            int_groups_num: Dict[Tuple[str, str], List[List[float]]] = {}
            if intervene_var_idxs:
                rng_int = np.random.default_rng(seed_data + 10_000)
                for var_idx, var_name in intervene_var_idxs:
                    prob_dist = getattr(graph.variables[var_idx], "prob_dist", None)
                    num_categs = getattr(prob_dist, "num_categs", None)
                    if not isinstance(num_categs, int) or num_categs <= 0:
                        num_categs = len(codebook.get(var_name, [])) or 2

                    dataset_size = int_per_combo
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
                        ival_out = (str(s_idx) if anonymize else value_for_display(var_name, s_idx))
                        interventional_rows_base.append({
                            "intervened_variable": ivar_out,
                            "intervened_value": ival_out,
                            **row_disp,
                        })
                        int_groups_num.setdefault((ivar_out, str(ival_out)), []).append([float(r[idx]) for idx in col_indices])

            for rep in range(shuffles_per_graph):
                seed_ir = seed_data + rep
                obs_rows = [r.copy() for r in obs_rows_base]
                rng_obs = np.random.default_rng(seed_ir)

                if row_order == "random":
                    rng_obs.shuffle(obs_rows)
                elif row_order == "reverse":
                    obs_rows.reverse()
                elif row_order == "sorted":
                    key_var = variables_out[0]
                    obs_rows.sort(key=lambda x: str(x.get(key_var, "")))

                int_rows_final = []
                if interventional_rows_base:
                    if row_order == "random":
                        tmp_rows = [r.copy() for r in interventional_rows_base]
                        buckets = {}
                        for r in tmp_rows:
                            k = (r["intervened_variable"], r["intervened_value"])
                            buckets.setdefault(k, []).append(r)
                        keys = list(buckets.keys())
                        np.random.default_rng(seed_ir + 1).shuffle(keys)
                        for k in keys:
                            batch = buckets[k]
                            np.random.default_rng(seed_ir + 2).shuffle(batch)
                            int_rows_final.extend(batch)
                    elif row_order == "reverse":
                        int_rows_final = interventional_rows_base[::-1]
                    elif row_order == "sorted":
                        int_rows_final = sorted(
                            interventional_rows_base,
                            key=lambda x: str(x.get(variables_out[0], "")),
                        )

                if int_per_combo > 0:
                    rows_for_prompt = int_rows_final + obs_rows
                else:
                    rows_for_prompt = obs_rows

                rows_text_source = rows_for_prompt

                dataset_name = os.path.splitext(os.path.basename(bif_file))[0]
                use_edge_list_output = _use_edge_list_output(variables_out)
                is_names_only_cfg = (obs_per_prompt == 0 and int_per_combo == 0)
                if is_names_only_cfg:
                    # (obs=0,int=0) should always use the names-only prompt format, regardless of style.
                    from generate_prompts_names_only import format_names_only_prompt
                    prompt_text = format_names_only_prompt(
                        variables_out, dataset_name, causal_rules, output_edge_list=use_edge_list_output
                    )
                elif prompt_style == "summary_joint":
                    state_names = []
                    for orig_name in permuted_real_names:
                        states = codebook.get(orig_name, []) or []
                        if anonymize:
                            state_names.append([str(i) for i in range(len(states))] if states else [])
                        else:
                            state_names.append([str(s) for s in states] if states else [])
                    prompt_text = format_prompt_summary_full_joint(
                        variables_out,
                        dataset_name=dataset_name,
                        obs_rows_num=obs_rows_num,
                        int_groups_num=int_groups_num,
                        state_names=state_names if state_names else None,
                        include_causal_rules=causal_rules,
                        include_give_steps=give_steps,
                        include_def_int=include_def_int,
                        anonymize=anonymize,
                        include_probabilities=False,
                        sort_hist_by="count_desc",
                        require_think_answer_blocks=thinking_tags,
                        output_edge_list=use_edge_list_output,
                    )
                elif prompt_style == "matrix":
                    prompt_text = format_prompt_cb_matrix(
                        variables_out, rows_text_source, dataset_name,
                        causal_rules, give_steps, include_def_int=include_def_int,
                        require_think_answer_blocks=thinking_tags,
                        output_edge_list=use_edge_list_output,
                    )
                else:
                    prompt_text = format_prompt_with_interventions(
                        variables_out, rows_text_source, vmap,
                        causal_rules, give_steps, include_def_int=include_def_int,
                        require_think_answer_blocks=thinking_tags,
                        output_edge_list=use_edge_list_output,
                    )

                yield {
                    "data_idx": i,
                    "shuffle_idx": rep,
                    "prompt_text": prompt_text,
                    "given_edges": None,
                }

    return base_name, answer_obj, _iter()


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

def format_prompt_payload_json(
    variables: List[str],
    *,
    dataset_name: str,
    obs_rows_num: List[List[float]],
    int_groups_num: Dict[Tuple[str, str], List[List[float]]],
    state_names: Optional[List[List[str]]] = None,
    epsilon: float = 0.02,
    decimals: int = 4,
    anonymize: bool = False,   # <-- add this
    require_think_answer_blocks: bool = False,
    output_edge_list: bool = False,
) -> str:
    """
    New prompt format:
      - One JSON payload under 'INPUT JSON:'.
      - Includes full observational marginals + corr.
      - Includes full do-marginals for each intervention group.
      - Includes TV distance-to-observational for ALL variables (excluding intervened var).
    """
    import json

    # ---------- build variable specs ----------
    # Determine number of states per variable (prefer state_names if provided)
    m = len(variables)
    num_states: List[int] = []
    for j in range(m):
        if state_names and j < len(state_names) and state_names[j]:
            num_states.append(len(state_names[j]))
        else:
            # infer from data
            max_seen = -1
            for r in obs_rows_num:
                if j < len(r):
                    max_seen = max(max_seen, int(round(float(r[j]))))
            for rows in int_groups_num.values():
                for r in rows:
                    if j < len(r):
                        max_seen = max(max_seen, int(round(float(r[j]))))
            num_states.append(max(2, max_seen + 1))

    variables_spec = []
    for j, v in enumerate(variables):
        if state_names and j < len(state_names) and state_names[j]:
            mapping = {str(s): str(name) for s, name in enumerate(state_names[j])}
        else:
            mapping = {str(s): str(s) for s in range(num_states[j])}
        variables_spec.append({"name": v, "states": mapping, "index": j})

    # ---------- observational summaries ----------
    if not obs_rows_num:
        raise ValueError("obs_rows_num is empty; cannot build observational summary payload.")

    obs_marginals_list = _marginals_py(obs_rows_num, num_states)
    obs_marginals_list_r = [[round(float(x), decimals) for x in row] for row in obs_marginals_list]
    obs_marginals = {variables[j]: obs_marginals_list_r[j] for j in range(m)}

    obs_corr = _corr_matrix_py(obs_rows_num) if len(obs_rows_num) >= 2 else None
    obs_corr_r = (
        [[round(float(x), decimals) for x in row] for row in obs_corr] if obs_corr is not None else None
    )

    # ---------- interventions: full marginals + full TV dict ----------
    interventions_payload = []
    for (ivar, ival) in sorted(int_groups_num.keys(), key=lambda kv: (str(kv[0]), str(kv[1]))):
        rows = int_groups_num[(ivar, ival)]
        do_marginals_list = _marginals_py(rows, num_states)
        do_marginals_list_r = [[round(float(x), decimals) for x in row] for row in do_marginals_list]
        do_marginals = {variables[j]: do_marginals_list_r[j] for j in range(m)}

        # TV distance of each variable's marginal vs observational
        tv_dict: Dict[str, float] = {}
        for j in range(m):
            vname = variables[j]
            if vname == ivar:
                continue
            p = obs_marginals_list_r[j]
            q = do_marginals_list_r[j]
            tv = 0.5 * sum(abs(float(p[s]) - float(q[s])) for s in range(min(len(p), len(q))))
            tv_dict[vname] = round(float(tv), decimals)

        interventions_payload.append({
            "do": {"var": ivar, "value": ival},
            "n": len(rows),
            "marginals": do_marginals,
            "distance_to_observational_marginals": tv_dict,
            "distance_metric": "total_variation_on_marginals",
        })

    # ---------- final payload ----------
    payload: Dict[str, Any] = {
        "assumptions": {
            "causal_sufficiency": True,
            "dag": True,
            "perfect_do_interventions": True,
        },
        "variables": variables_spec,
        "observational": {
            "n": len(obs_rows_num),
            "marginals": obs_marginals,
        },
        "interventions": interventions_payload,
        "hyperparams_for_decisions": {
            "marginal_change_epsilon": epsilon,
            "prefer_sparse_graph": True,
        },
        "output_format": {
            "variables_key_order": variables,
            "adjacency_matrix_definition": "[i][j]=1 iff variables[i] -> variables[j]",
        },
    }
    if output_edge_list:
        payload["output_format"] = {
            "variables_key_order": variables,
            "edges_definition": "[[source,target], ...] using exact variable names",
        }

    if not anonymize:
        payload["network_name"] = dataset_name

    if obs_corr_r is not None:
        payload["observational"]["dependence"] = {
            "type": "pearson_corr_on_numeric_codes",
            "numeric_code_definition": "state index as listed above (0/1/...)",
            "matrix": obs_corr_r,
        }

    # ---------- wrap with strict output instructions + payload ----------
    # (Single-file prompt: include “system” guidance inline since you write .txt prompts)
    method_header = (
        "METHOD (write this reasoning in the <think>...</think> block):\n"
        if require_think_answer_blocks else
        "METHOD (follow internally, do not output reasoning):\n"
    )
    method_block = (
        "ROLE: You are an expert in causal discovery from observational and interventional data.\n"
        "ASSUMPTIONS: causal sufficiency, DAG, perfect do-interventions.\n"
        + method_header +
        "1) From each do(X=x), treat variables whose marginals change (TV > epsilon) as descendants of X.\n"
        "2) Use observational dependence only to suggest adjacencies (skeleton), not directions.\n"
        "3) Prefer X->Y if Y changes under do(X) but X does not change under do(Y).\n"
        "4) Enforce acyclicity and choose the sparsest graph consistent with the evidence.\n"
    )

    out_contract = "OUTPUT INSTRUCTIONS:\n" + "\n".join(
        _build_output_contract_lines(
            output_edge_list=output_edge_list,
            require_think_answer_blocks=require_think_answer_blocks,
        )
    ) + "\n"

    return method_block + "\nINPUT JSON:\n" + json.dumps(payload, ensure_ascii=False) + "\n\n" + out_contract


def format_prompt_payload_topk_json(
    variables: List[str],
    *,
    dataset_name: str,
    obs_rows_num: List[List[float]],
    int_groups_num: Dict[Tuple[str, str], List[List[float]]],
    state_names: Optional[List[List[str]]] = None,
    epsilon: float = 0.02,
    decimals: int = 4,
    top_k_effects: int = 6,
    include_network_name: bool = True,
    require_think_answer_blocks: bool = False,
    output_edge_list: bool = False,
) -> str:
    """
    Compact payload prompt (v2):
      - No Pearson correlation matrix (can be misleading on discrete numeric codes).
      - Observational: marginals only.
      - Each do-block: TV top-K effects + list of changed vars + marginals only for top-K vars.
      - Includes explicit method text telling the model to use precomputed fields only.
    """
    import json

    m = len(variables)

    # ---------- infer number of states per variable ----------
    num_states: List[int] = []
    for j in range(m):
        if state_names and j < len(state_names) and state_names[j]:
            num_states.append(len(state_names[j]))
            continue

        max_seen = -1
        for r in obs_rows_num:
            if j < len(r):
                max_seen = max(max_seen, int(round(float(r[j]))))
        for rows in int_groups_num.values():
            for r in rows:
                if j < len(r):
                    max_seen = max(max_seen, int(round(float(r[j]))))
        num_states.append(max(2, max_seen + 1))

    # ---------- variable specs ----------
    variables_spec: List[Dict[str, Any]] = []
    for j, v in enumerate(variables):
        if state_names and j < len(state_names) and state_names[j]:
            mapping = {str(s): str(name) for s, name in enumerate(state_names[j])}
        else:
            mapping = {str(s): str(s) for s in range(num_states[j])}
        variables_spec.append({"name": v, "states": mapping, "index": j})

    if not obs_rows_num:
        raise ValueError("obs_rows_num is empty; cannot build payload.")

    # ---------- observational marginals ----------
    obs_marginals_list = _marginals_py(obs_rows_num, num_states)
    obs_marginals_list_r = [[round(float(x), decimals) for x in row] for row in obs_marginals_list]
    obs_marginals = {variables[j]: obs_marginals_list_r[j] for j in range(m)}

    # ---------- interventions: compute TV for all vars, surface top-K ----------
    interventions_payload: List[Dict[str, Any]] = []
    k = max(0, int(top_k_effects))
    for (ivar, ival) in sorted(int_groups_num.keys(), key=lambda kv: (str(kv[0]), str(kv[1]))):
        rows = int_groups_num[(ivar, ival)]
        do_marginals_list = _marginals_py(rows, num_states)
        do_marginals_list_r = [[round(float(x), decimals) for x in row] for row in do_marginals_list]

        tv_scores: List[Tuple[str, float]] = []
        for j in range(m):
            vname = variables[j]
            if vname == ivar:
                continue
            p = obs_marginals_list_r[j]
            q = do_marginals_list_r[j]
            tv = 0.5 * sum(abs(float(p[s]) - float(q[s])) for s in range(min(len(p), len(q))))
            tv_scores.append((vname, round(float(tv), decimals)))

        tv_scores.sort(key=lambda t: t[1], reverse=True)
        changed_vars = [v for (v, tv) in tv_scores if tv > float(epsilon)]

        top = tv_scores[:k]
        tv_top_effects = [[v, tv] for (v, tv) in top]
        do_marginals_top = {v: do_marginals_list_r[variables.index(v)] for (v, _) in top}

        interventions_payload.append(
            {
                "do": {"var": ivar, "value": ival},
                "n": len(rows),
                "tv_top_effects": tv_top_effects,
                "changed_vars": changed_vars,
                "do_marginals_top": do_marginals_top,
                "tv_metric": "total_variation_on_marginals",
            }
        )

    payload: Dict[str, Any] = {
        "assumptions": {
            "causal_sufficiency": True,
            "dag": True,
            "perfect_do_interventions": True,
        },
        "variables": variables_spec,
        "observational": {
            "n": len(obs_rows_num),
            "marginals": obs_marginals,
        },
        "interventions": interventions_payload,
        "hyperparams_for_decisions": {
            "marginal_change_epsilon": float(epsilon),
            "prefer_sparse_graph": True,
            "top_k_effects": k,
        },
        "output_format": {
            "variables_key_order": variables,
            "adjacency_matrix_definition": "[i][j]=1 iff variables[i] -> variables[j]",
        },
    }
    if output_edge_list:
        payload["output_format"] = {
            "variables_key_order": variables,
            "edges_definition": "[[source,target], ...] using exact variable names",
        }

    if include_network_name:
        payload["network_name"] = dataset_name

    method_header = (
        "METHOD (write this reasoning in the <think>...</think> block):\n"
        if require_think_answer_blocks else
        "METHOD (follow internally; do not output reasoning):\n"
    )
    method_block = (
        "ROLE: You are an expert in causal discovery from observational and interventional data.\n"
        "ASSUMPTIONS: causal sufficiency, DAG, perfect do-interventions.\n"
        "EVIDENCE SEMANTICS:\n"
        "- For each intervention entry, `changed_vars` are variables whose marginals changed under do(X=v) (TV > epsilon).\n"
        "- Only descendants of X can change under do(X=v) in a perfect intervention setting.\n"
        + method_header +
        "1) For each intervention do(X=v), treat `changed_vars` as a superset of descendants of X.\n"
        "2) Prefer X->Y if Y appears in changed_vars under do(X=*) but X does NOT appear in changed_vars under do(Y=*).\n"
        "3) Do NOT infer direction from observational marginals alone.\n"
        "4) Output the sparsest DAG consistent with the intervention constraints.\n"
    )

    out_contract = "OUTPUT INSTRUCTIONS:\n" + "\n".join(
        _build_output_contract_lines(
            output_edge_list=output_edge_list,
            require_think_answer_blocks=require_think_answer_blocks,
        )
    ) + "\n"

    return (
        method_block
        + "\nINPUT JSON:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n\n"
        + out_contract
    )


def format_prompt_cb_matrix(
    variables: List[str],
    all_rows: List[Dict[str, Any]],
    dataset_name: str,
    include_causal_rules: bool = False,
    include_give_steps: bool = False,
    given_edges: Optional[List[Tuple[str, str]]] = None,
    include_def_int: bool = False,
    state_names: Optional[Dict[str, Dict[str, str]]] = None,  # optional: {var: {"0":"low","1":"high"}}
    anonymize: bool = False,  # controls whether to omit network_name
    require_think_answer_blocks: bool = False,
    output_edge_list: bool = False,
) -> str:
    """
    Data-given prompt: matrix blocks (observational + grouped interventions).

    state_names (optional) lets you explicitly describe categorical codes.
    If anonymize=True, omit network_name to avoid leakage.
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

    # --- Role / task ---
    lines.append("ROLE: You are an expert in causal discovery from observational and interventional data.")
    lines.append("TASK: Infer the directed causal graph over the variables.")

    if anonymize:
        lines.append("The following are empirical distributions computed from data sampled from an anonymized Bayesian network.")
    else:
        lines.append(f"The following are empirical distributions computed from data sampled from a Bayesian network named {dataset_name}.")


    # --- Assumptions (high impact, short) ---
    lines.append("ASSUMPTIONS:")
    lines.append("- The true graph is a DAG (no directed cycles).")
    lines.append("- Causal sufficiency holds (no unobserved confounders among these variables).")
    lines.append("- Interventions are perfect do-interventions (surgical): do(X=v) cuts all incoming edges into X.")

    # --- Optional causal reminders ---
    if include_causal_rules:
        lines.extend([
            "\n--- CAUSAL DISCOVERY REMINDERS ---",
            "- Use observational dependence/independence to suggest which pairs may be connected (skeleton).",
            "- Use interventions to orient edges: if Y changes under do(X=v), that supports X being an ancestor of Y.",
            "- Prefer directions that are consistent across all intervention blocks and keep the graph acyclic.",
        ])

    # --- Intervention notes (stronger + operational) ---
    if include_def_int:
        lines.extend([
            "\n--- INTERVENTION SEMANTICS ---",
            "- Each block [Intervention: do(X = v)] contains samples where X is externally set to v.",
            "- In that block, X's usual causes are disabled (incoming edges into X are cut).",
            "- Only descendants of X can change in distribution because of do(X=v); non-descendants should remain invariant up to sampling noise.",
            "- Treat values as categorical labels (do NOT assume numeric ordering of codes).",
        ])

    # --- Known edges (optional) ---
    if given_edges:
        lines.append("\n--- KNOWN DIRECT CAUSAL EDGES ---")
        lines.append("The following directed causal edges are guaranteed to be present in the true graph:")
        for src, dst in given_edges:
            lines.append(f"- {src} -> {dst}")

    # --- Variable order ---
    lines.append("\n--- VARIABLE ORDER (ORDER MATTERS) ---")
    for i, v in enumerate(variables):
        if state_names and v in state_names and state_names[v]:
            lines.append(f"{i}: {v} states={state_names[v]}")
        else:
            lines.append(f"{i}: {v}")

    # Common header row
    header = " | ".join(variables)

    # --- Observational matrix ---
    if obs_rows:
        lines.append("\n--- OBSERVATIONAL DATA (no intervention) ---")
        lines.append("Each row is one observed case.")
        lines.append(header)
        for row in obs_rows:
            vals = [str(row[v]) for v in variables]
            lines.append(" | ".join(vals))

    # --- Interventional matrices ---
    if int_buckets:
        lines.append("\n--- INTERVENTIONAL DATA ---")
        lines.append("Each block corresponds to samples collected under a perfect intervention do(X = value).")

        for (ivar, ival) in sorted(int_buckets.keys(), key=lambda kv: (str(kv[0]), str(kv[1]))):
            rows = int_buckets[(ivar, ival)]
            lines.append(f"\n[Intervention: do({ivar} = {ival})]")
            lines.append(header)
            for row in rows:
                vals = [str(row[v]) for v in variables]
                lines.append(" | ".join(vals))

    lines.append("\n--- END OF DATA ---")

    # --- Minimal internal method (optional) ---
    if include_give_steps:
        method_header = (
            "\nMETHOD (write this reasoning in the <think>...</think> block):"
            if require_think_answer_blocks else
            "\nMETHOD (follow internally; do not output reasoning):"
        )
        lines.extend([
            method_header,
            "1) For each do(X=v) block, mark variables whose distributions differ from observational as descendants of X.",
            "2) Use asymmetry across interventions to orient: if Y changes under do(X) but X does not change under do(Y), prefer X -> Y.",
            "3) Use observational data only to suggest adjacency (avoid overfitting); interventions decide directions when possible.",
            "4) Choose the sparsest DAG consistent with all blocks and known edges.",
        ])

    # Put strict formatting constraints at the very end for stronger adherence.
    lines.append("\n--- OUTPUT INSTRUCTIONS ---")
    lines.extend(
        _build_output_contract_lines(
            output_edge_list=output_edge_list,
            require_think_answer_blocks=require_think_answer_blocks,
        )
    )

    return "\n".join(lines)


def _corr_matrix_py(rows: List[List[float]]) -> List[List[float]]:
    """
    Pure-Python Pearson correlation over columns (sample correlation, ddof=1).
    Avoids numpy/BLAS/OpenMP issues on some clusters.
    """
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
            if std[i] == 0.0 or std[j] == 0.0:
                c = 0.0
            else:
                c = cov / (std[i] * std[j])
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


def format_prompt_summary_stats(
    variables: List[str],
    *,
    dataset_name: str,
    obs_rows_num: List[List[float]],
    int_groups_num: Dict[Tuple[str, str], List[List[float]]],
    include_causal_rules: bool = False,
    include_give_steps: bool = False,
    include_def_int: bool = False,
    decimals: int = 4,
    require_think_answer_blocks: bool = False,
    output_edge_list: bool = False,
) -> str:
    """
    Summary-statistics prompt: small token footprint.

    We embed:
      - variable order
      - observational correlation matrix (computed on numeric codes)
      - optional intervention group means / mean-shifts (computed on numeric codes)

    NOTE: For discrete variables this is a lossy summary; it's meant as a lightweight baseline.
    """
    import json

    lines: List[str] = []
    lines.append(
        "You are a highly intelligent question-answering bot with profound "
        "knowledge of causal inference and causal discovery."
    )
    lines.append(
        f"The following are summary statistics computed from data sampled from a Bayesian "
        f"network named {dataset_name}."
    )
    lines.append("Infer the directed causal graph over the variables.")

    if include_causal_rules:
        lines.extend([
            "\n--- CAUSAL INFERENCE REMINDERS ---",
            "- Confounder: a variable that causes two others.",
            "- Mediator: lies on a path X -> M -> Y.",
            "- Collider: a common effect of two variables; avoid conditioning on colliders.",
            "- The final output must be a DAG (no directed cycles).",
        ])

    if include_def_int and int_groups_num:
        lines.extend([
            "\n--- INTERVENTION NOTES ---",
            "- An intervention do(X = v) sets X externally and replaces its usual causal mechanism.",
            "- In the intervened causal graph, all incoming edges into X are removed.",
            "- Only descendants of X can be causally affected by this intervention.",
        ])

    lines.append("\n--- VARIABLES (ORDER MATTERS) ---")
    for i, v in enumerate(variables):
        lines.append(f"{i}: {v}")

    # Observational summaries
    lines.append("\n--- OBSERVATIONAL SUMMARY ---")
    if obs_rows_num:
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
    else:
        obs_mean = None
        lines.append("obs_n=0")

    # Interventional summaries
    lines.append("\n--- INTERVENTIONAL SUMMARY ---")
    if not int_groups_num:
        lines.append("(none)")
    else:
        # Stable order
        for (ivar, ival) in sorted(int_groups_num.keys(), key=lambda kv: (str(kv[0]), str(kv[1]))):
            rows = int_groups_num[(ivar, ival)]
            mu = _mean_vec_py(rows)
            mu_r = [round(float(x), decimals) for x in mu]
            if obs_mean is not None:
                delta_r = [round(float(mu[j] - obs_mean[j]), decimals) for j in range(len(mu))]
            else:
                delta_r = None
            payload = {
                "n": len(rows),
                "mean_numeric_codes": mu_r,
                "delta_from_obs_mean": delta_r,
            }
            lines.append(f"do({ivar}={ival}): " + json.dumps(payload, separators=(",", ":"), ensure_ascii=False))

    lines.append("\n--- END OF SUMMARY ---")
    if include_give_steps:
        steps_header = (
            "\n(Use this in your <think>...</think> block.)"
            if require_think_answer_blocks else
            "\n(You may follow these steps silently.)"
        )
        lines.extend([
            steps_header,
            "1) Use obs_corr and intervention deltas to constrain and orient edges.",
            "2) Choose a directed acyclic graph consistent with the constraints.",
            "Then output the JSON as specified above.",
        ])

    lines.append("\n--- OUTPUT INSTRUCTIONS ---")
    lines.extend(
        _build_output_contract_lines(
            output_edge_list=output_edge_list,
            require_think_answer_blocks=require_think_answer_blocks,
        )
    )

    return "\n".join(lines)


def _marginals_py(rows: List[List[float]], num_states: List[int]) -> List[List[float]]:
    """
    Compute per-column discrete marginals over integer-coded states.
    Returns probs[j][s] = P(X_j = s).
    """
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
            if k <= 0:
                continue
            idx = int(round(float(x)))
            if 0 <= idx < k:
                counts[j][idx] += 1
    probs: List[List[float]] = []
    for j, k in enumerate(num_states):
        if k <= 0:
            probs.append([])
            continue
        denom = float(n)
        probs.append([c / denom for c in counts[j]])
    return probs


def _pairwise_joints_py(
    rows: List[List[float]],
    num_states: List[int],
    *,
    pairs: Optional[List[Tuple[int, int]]] = None,
) -> Dict[Tuple[int, int], List[List[float]]]:
    """
    Compute pairwise joint distributions over integer-coded states.

    Returns a dict mapping (i,j) -> probs, where probs[a][b] = P(X_i=a, X_j=b).
    Only pairs in `pairs` are computed; if None, computes all i<j.
    """
    n = len(rows)
    if n <= 0:
        raise ValueError("Need at least 1 row for pairwise joints.")
    m = len(rows[0])
    if any(len(r) != m for r in rows):
        raise ValueError("Ragged rows: all rows must have the same number of columns.")
    if len(num_states) != m:
        raise ValueError(f"num_states must have length {m}; got {len(num_states)}.")

    if pairs is None:
        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    else:
        # Ensure consistent (i<j) ordering.
        pairs = [(i, j) if i < j else (j, i) for (i, j) in pairs]
        pairs = sorted(set(pairs))

    counts: Dict[Tuple[int, int], List[List[int]]] = {}
    for i, j in pairs:
        ki = int(num_states[i])
        kj = int(num_states[j])
        if ki <= 0 or kj <= 0:
            continue
        counts[(i, j)] = [[0 for _ in range(kj)] for _ in range(ki)]

    for r in rows:
        for i, j in pairs:
            table = counts.get((i, j))
            if table is None:
                continue
            ki = len(table)
            kj = len(table[0]) if table else 0
            if ki <= 0 or kj <= 0:
                continue
            xi = int(round(float(r[i])))
            xj = int(round(float(r[j])))
            if 0 <= xi < ki and 0 <= xj < kj:
                table[xi][xj] += 1

    probs: Dict[Tuple[int, int], List[List[float]]] = {}
    denom = float(n)
    for (i, j), table in counts.items():
        probs[(i, j)] = [[c / denom for c in row] for row in table]
    return probs


def _histogram_rows_full_assignments_py(
    rows: List[List[float]],
    num_states: List[int],
) -> Dict[Tuple[int, ...], int]:
    """
    Histogram over full assignments (unique rows) for integer-coded discrete states.

    Returns a dict mapping x (length-m tuple of ints) -> count.
    """
    n = len(rows)
    if n <= 0:
        raise ValueError("Need at least 1 row for histogram.")
    m = len(rows[0])
    if any(len(r) != m for r in rows):
        raise ValueError("Ragged rows: all rows must have the same number of columns.")
    if len(num_states) != m:
        raise ValueError(f"num_states must have length {m}; got {len(num_states)}.")

    hist: Dict[Tuple[int, ...], int] = {}
    for r in rows:
        x: List[int] = []
        ok = True
        for j in range(m):
            k = int(num_states[j])
            if k <= 0:
                ok = False
                break
            v = int(round(float(r[j])))
            if not (0 <= v < k):
                ok = False
                break
            x.append(v)
        if not ok:
            continue
        key = tuple(x)
        hist[key] = hist.get(key, 0) + 1
    return hist


def _encode_assignment_mixed_radix_str(x: Tuple[int, ...], bases: List[int]) -> str:
    """
    Encode a full assignment x into a single non-negative integer string using mixed radix.

    code = sum_{j=0..m-1} x[j] * prod_{t<j} bases[t]
    """
    if len(x) != len(bases):
        raise ValueError("x and bases must have the same length.")
    code = 0
    mult = 1
    for j, v in enumerate(x):
        b = int(bases[j])
        if b <= 0:
            raise ValueError("All bases must be positive.")
        vv = int(v)
        if not (0 <= vv < b):
            raise ValueError(f"x[{j}]={vv} out of range for base {b}.")
        code += vv * mult
        mult *= b
    return str(code)


def format_prompt_summary_probs(
    variables: List[str],
    *,
    dataset_name: str,
    obs_rows_num: List[List[float]],
    int_groups_num: Dict[Tuple[str, str], List[List[float]]],
    state_names: Optional[List[List[str]]] = None,
    include_causal_rules: bool = False,
    include_give_steps: bool = False,
    include_def_int: bool = False,
    decimals: int = 4,
    top_k_effects: int = 5,
    require_think_answer_blocks: bool = False,
    output_edge_list: bool = False,
) -> str:
    """
    Probability-based summary prompt (token-efficient, more informative than means).

    We embed:
      - variable order (+ optional state-name mapping)
      - observational marginals over integer-coded discrete states
      - optional observational correlation matrix over numeric codes (coarse)
      - intervention effects summarized via top-K TV-distance shifts and marginals
    """
    import json

    lines: List[str] = []
    lines.append(
        "You are a highly intelligent question-answering bot with profound "
        "knowledge of causal inference and causal discovery."
    )
    lines.append(
        f"The following are summary statistics computed from data sampled from a Bayesian "
        f"network named {dataset_name}."
    )
    lines.append("Infer the directed causal graph over the variables.")

    if include_causal_rules:
        lines.extend([
            "\n--- CAUSAL INFERENCE REMINDERS ---",
            "- Confounder: a variable that causes two others.",
            "- Mediator: lies on a path X -> M -> Y.",
            "- Collider: a common effect of two variables; avoid conditioning on colliders.",
            "- The final output must be a DAG (no directed cycles).",
        ])

    if include_def_int and int_groups_num:
        lines.extend([
            "\n--- INTERVENTION NOTES ---",
            "- An intervention do(X = v) sets X externally and replaces its usual causal mechanism.",
            "- In the intervened causal graph, all incoming edges into X are removed.",
            "- Only descendants of X can be causally affected by this intervention.",
        ])

    lines.append("\n--- VARIABLES (ORDER MATTERS) ---")
    for i, v in enumerate(variables):
        if state_names and i < len(state_names) and state_names[i]:
            mapping = {str(s): str(name) for s, name in enumerate(state_names[i])}
            lines.append(f"{i}: {v} states=" + json.dumps(mapping, separators=(",", ":"), ensure_ascii=False))
        else:
            lines.append(f"{i}: {v}")

    # Determine number of states per variable.
    # Prefer state_names lengths if provided, else infer from observed + interventional rows.
    m = len(variables)
    num_states: List[int] = []
    for j in range(m):
        if state_names and j < len(state_names) and state_names[j]:
            num_states.append(len(state_names[j]))
        else:
            max_seen = -1
            for r in obs_rows_num:
                if j < len(r):
                    max_seen = max(max_seen, int(round(float(r[j]))))
            for rows in int_groups_num.values():
                for r in rows:
                    if j < len(r):
                        max_seen = max(max_seen, int(round(float(r[j]))))
            num_states.append(max(2, max_seen + 1))

    lines.append("\n--- OBSERVATIONAL SUMMARY ---")
    if obs_rows_num:
        obs_probs = _marginals_py(obs_rows_num, num_states)
        obs_probs_r = [[round(float(x), decimals) for x in row] for row in obs_probs]
        obs_corr = _corr_matrix_py(obs_rows_num) if len(obs_rows_num) >= 2 else None
        obs_corr_r = (
            [[round(float(x), decimals) for x in row] for row in obs_corr] if obs_corr is not None else None
        )
        lines.append(f"obs_n={len(obs_rows_num)}")
        payload = {variables[j]: obs_probs_r[j] for j in range(m)}
        lines.append("obs_marginals=" + json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
        if obs_corr_r is not None:
            lines.append("obs_corr_numeric_codes=" + json.dumps(obs_corr_r, separators=(",", ":"), ensure_ascii=False))
    else:
        obs_probs_r = None
        lines.append("obs_n=0")

    lines.append("\n--- INTERVENTIONAL SUMMARY ---")
    if not int_groups_num:
        lines.append("(none)")
    else:
        # Stable order by (intervened variable, intervened value)
        for (ivar, ival) in sorted(int_groups_num.keys(), key=lambda kv: (str(kv[0]), str(kv[1]))):
            rows = int_groups_num[(ivar, ival)]
            do_probs = _marginals_py(rows, num_states)
            do_probs_r = [[round(float(x), decimals) for x in row] for row in do_probs]

            tv_scores: List[Tuple[int, float]] = []
            if obs_probs_r is not None:
                for j in range(m):
                    if variables[j] == ivar:
                        continue
                    p = obs_probs_r[j]
                    q = do_probs_r[j]
                    if not p or not q:
                        continue
                    tv = 0.5 * sum(abs(float(p[s]) - float(q[s])) for s in range(min(len(p), len(q))))
                    tv_scores.append((j, tv))
                tv_scores.sort(key=lambda t: t[1], reverse=True)

            top = tv_scores[: max(0, int(top_k_effects))]
            top_effects = [[variables[j], round(float(tv), decimals)] for (j, tv) in top]
            top_marginals = {variables[j]: do_probs_r[j] for (j, _) in top}

            payload = {
                "n": len(rows),
                "tv_top_effects": top_effects,
                "do_marginals_top": top_marginals,
            }
            lines.append(f"do({ivar}={ival}): " + json.dumps(payload, separators=(",", ":"), ensure_ascii=False))

    lines.append("\n--- END OF SUMMARY ---")
    if include_give_steps:
        steps_header = (
            "\n(Use this in your <think>...</think> block.)"
            if require_think_answer_blocks else
            "\n(You may follow these steps silently.)"
        )
        lines.extend([
            steps_header,
            "1) Use obs_marginals, obs_corr, and intervention TV effects to constrain and orient edges.",
            "2) Choose a directed acyclic graph consistent with the constraints.",
            "Then output the JSON as specified above.",
        ])

    lines.append("\n--- OUTPUT INSTRUCTIONS ---")
    lines.extend(
        _build_output_contract_lines(
            output_edge_list=output_edge_list,
            require_think_answer_blocks=require_think_answer_blocks,
        )
    )

    return "\n".join(lines)






def format_prompt_summary_full_joint(
    variables: List[str],
    *,
    dataset_name: str,
    obs_rows_num: List[List[float]],
    int_groups_num: Dict[Tuple[str, str], List[List[float]]],
    state_names: Optional[List[List[str]]] = None,
    include_causal_rules: bool = False,
    include_give_steps: bool = False,
    include_def_int: bool = False,
    # formatting controls
    decimals: int = 6,
    omitted_are_zero_prob: bool = False,
    include_probabilities: bool = True,
    # readability / size controls
    sort_hist_by: str = "prob_desc",  # "prob_desc" | "count_desc" | "lex"
    include_marginals: bool = True,
    include_state_legend: bool = True,  # only used when NOT anonymized and mapping is non-identity
    # token budget controls
    max_hist_entries: Optional[int] = None,  # truncate hist per regime (after sorting). None => full
    max_intervention_regimes: Optional[int] = None,
    # anonymization
    anonymize: bool = True,
    # pretty printing controls
    pretty_json: bool = True,
    json_indent: int = 2,
    # hybrid formatting (readable dict, compact big arrays)
    compact_array_keys: Tuple[str, ...] = ("hist", "marginals"),
    # NEW: make interventional payloads compact even when nested
    compact_interventions: bool = True,
    require_think_answer_blocks: bool = False,
    output_edge_list: bool = False,
) -> str:
    """
    Full empirical joint prompt (sparse histogram over full assignments), using clearer block names:

      - observational_data: {n, hist, (optional) marginals}
      - interventional_data: { "do(X=v)": {n, hist, (optional) marginals}, ... }

    Hybrid formatting:
      - observational_data: normal hybrid (top-level arrays compact)
      - interventional_data: if compact_interventions=True, each do()-payload is compacted (one-line hist/marginals)
        while keeping the top-level dict pretty (do-keys easy to scan).
    """
    import json

    # -----------------------------
    # helpers
    # -----------------------------
    def _safe_int(v: float) -> int:
        return int(round(float(v)))

    def _roundp(x: float) -> float:
        return round(float(x), int(decimals))

    def _infer_num_states() -> List[int]:
        m = len(variables)
        bases: List[int] = []
        for j in range(m):
            if state_names and j < len(state_names) and state_names[j]:
                bases.append(len(state_names[j]))
                continue
            max_seen = -1
            for r in obs_rows_num:
                if j < len(r):
                    max_seen = max(max_seen, _safe_int(r[j]))
            for rows in int_groups_num.values():
                for r in rows:
                    if j < len(r):
                        max_seen = max(max_seen, _safe_int(r[j]))
            bases.append(max(2, max_seen + 1))
        return bases

    def _state_maps() -> Dict[str, Dict[int, str]]:
        """code -> name map per variable (only meaningful when not anonymized)."""
        out: Dict[str, Dict[int, str]] = {}
        if not state_names:
            return out
        for i, v in enumerate(variables):
            if i < len(state_names) and state_names[i]:
                out[v] = {int(s): str(name) for s, name in enumerate(state_names[i])}
        return out

    def _is_identity_state_map(sm: Dict[int, str]) -> bool:
        """Detect maps like {0:"0",1:"1",...} which are redundant."""
        for k, name in sm.items():
            if str(k) != str(name):
                return False
        return True

    def _to_hist_items(rows: List[List[float]], num_states: List[int]) -> Tuple[int, List[List[Any]]]:
        """
        Returns (n, items) where each item is [x, count] or [x, count, prob].
        """
        n = len(rows)
        if n <= 0:
            return 0, []
        hist = _histogram_rows_full_assignments_py(rows, num_states)  # Dict[Tuple[int,...], int]
        items: List[List[Any]] = []
        for x, c in hist.items():
            x_list = [int(v) for v in x]
            c_int = int(c)
            if include_probabilities and n > 0:
                items.append([x_list, c_int, _roundp(c_int / float(n))])
            else:
                items.append([x_list, c_int])
        return n, items

    def _sort_items(items: List[List[Any]]) -> List[List[Any]]:
        if sort_hist_by == "lex":
            return sorted(items, key=lambda t: t[0])
        if sort_hist_by == "count_desc":
            return sorted(items, key=lambda t: (-int(t[1]), t[0]))
        if include_probabilities:
            return sorted(items, key=lambda t: (-float(t[2]), -int(t[1]), t[0]))
        return sorted(items, key=lambda t: (-int(t[1]), t[0]))

    def _maybe_truncate(items: List[List[Any]]) -> List[List[Any]]:
        if max_hist_entries is None:
            return items
        return items[: max(0, int(max_hist_entries))]

    def _marginals_from_items(num_states: List[int], items_full: List[List[Any]], n: int) -> List[List[float]]:
        """
        marginals[j][s] = P(X_j=s) computed from COUNTS in the (full) items list.
        """
        m = len(num_states)
        marg = [[0.0 for _ in range(num_states[j])] for j in range(m)]
        if n <= 0:
            return marg
        for it in items_full:
            x = it[0]
            c = float(it[1])
            for j in range(m):
                v = int(x[j])
                if 0 <= v < num_states[j]:
                    marg[j][v] += c
        inv = 1.0 / float(n)
        for j in range(m):
            for s in range(num_states[j]):
                marg[j][s] = _roundp(marg[j][s] * inv)
        return marg

    def _format_do_label(ivar: str, ival: str, state_map: Dict[str, Dict[int, str]]) -> str:
        if anonymize:
            return f"do({ivar}={ival})"
        sm = state_map.get(ivar)
        if sm is not None:
            try:
                code = int(ival)
                if code in sm:
                    return f"do({ivar}={sm[code]})"
            except Exception:
                pass
        return f"do({ivar}={ival})"

    def _compact_json(x: Any) -> str:
        return json.dumps(x, separators=(",", ":"), ensure_ascii=False)

    def _dumps_hybrid_top_level_dict(d: Dict[str, Any]) -> str:
        """
        Hybrid at TOP LEVEL only: arrays under compact_array_keys become one-line.
        """
        if not pretty_json:
            return _compact_json(d)

        placeholders: Dict[str, str] = {}
        tmp = dict(d)
        for k in compact_array_keys:
            if k in tmp and tmp[k] is not None:
                token = f"__{k.upper()}_PLACEHOLDER__"
                placeholders[token] = _compact_json(tmp[k])
                tmp[k] = token

        s = json.dumps(tmp, indent=int(json_indent), ensure_ascii=False)
        for token, compact in placeholders.items():
            s = s.replace(json.dumps(token), compact)
        return s

    def _dumps_interventional_hybrid(interv: Dict[str, Any]) -> str:
        """
        Pretty-print the outer dict, but inject each do()-payload as compact JSON
        so nested hist/marginals are one-line.
        """
        if not pretty_json:
            return _compact_json(interv)
        if not compact_interventions:
            return json.dumps(interv, indent=int(json_indent), ensure_ascii=False)

        injected: Dict[str, str] = {}
        outer: Dict[str, Any] = {}
        for do_k, payload in interv.items():
            token = f"__PAYLOAD_{len(injected)}__"
            outer[do_k] = token
            injected[token] = _compact_json(payload)  # payload itself compact => hist/marginals one-line

        s = json.dumps(outer, indent=int(json_indent), ensure_ascii=False)
        for token, payload_str in injected.items():
            s = s.replace(json.dumps(token), payload_str)  # replace quoted token with JSON object
        return s

    # -----------------------------
    # Build prompt
    # -----------------------------
    num_states = _infer_num_states()
    state_map = _state_maps()

    lines: List[str] = []
    lines.append("You are a question-answering assistant with knowledge of causal inference and causal discovery.")
    if anonymize:
        lines.append("The following are empirical distributions computed from data sampled from an anonymized Bayesian network.")
    else:
        lines.append(f"The following are empirical distributions computed from data sampled from a Bayesian network named {dataset_name}.")
    lines.append("Infer the directed causal graph over the variables.")

    if include_causal_rules:
        lines.extend([
            "\n--- REMINDERS ---",
            "- Confounder causes two variables; collider is a common effect; mediator lies on a path.",
        ])

    if include_def_int and int_groups_num:
        lines.extend([
            "\n--- INTERVENTIONS ---",
            "- do(X=v) sets X externally; incoming edges into X are removed in the intervened graph.",
        ])

    lines.append("\n--- VARIABLES ---")
    for i, v in enumerate(variables):
        lines.append(f"{i}: {v}")

    lines.append("\n--- ASSIGNMENT FORMAT ---")
    lines.append("x = [" + ",".join(variables) + "]")
    lines.append("num_states=" + _compact_json(num_states))
    lines.append("Each histogram entry is [x, count{}].".format(", prob" if include_probabilities else ""))
    lines.append("do(X=v) means intervened data where X is forcibly set to value v.")
    if include_marginals:
        lines.append("marginals[j][s] := P(variables[j]=s) in VARIABLES order; each marginals[j] sums to 1.")

    lines.append("\n--- OMISSIONS ---")
    if omitted_are_zero_prob:
        lines.append("Unlisted assignments have probability 0.")
    else:
        lines.append("Unlisted assignments may be missing due to finite sampling (not necessarily probability 0).")

    if (not anonymize) and include_state_legend and state_map:
        kept_vars: List[str] = []
        for v in variables:
            sm = state_map.get(v)
            if sm and (not _is_identity_state_map(sm)):
                kept_vars.append(v)
        if kept_vars:
            lines.append("\n--- STATE LEGEND ---")
            for v in kept_vars:
                sm = state_map[v]
                parts = [f"{k}->{sm[k]}" for k in sorted(sm.keys())]
                lines.append(f"{v}: " + ", ".join(parts))

    # -----------------------------
    # OBSERVATIONAL DATA
    # -----------------------------
    lines.append("\n--- OBSERVATIONAL DATA ---")
    obs_n, obs_items_full = _to_hist_items(obs_rows_num, num_states)
    obs_items_full = _sort_items(obs_items_full)
    obs_items_out = _maybe_truncate(obs_items_full)

    obs_payload: Dict[str, Any] = {"n": obs_n, "hist": obs_items_out}
    if include_marginals and (max_hist_entries is None):
        obs_payload["marginals"] = _marginals_from_items(num_states, obs_items_full, obs_n)

    # hybrid top-level is enough here
    lines.append("observational_data=" + _dumps_hybrid_top_level_dict(obs_payload))

    # -----------------------------
    # INTERVENTIONAL DATA
    # -----------------------------
    lines.append("\n--- INTERVENTIONAL DATA ---")
    if not int_groups_num:
        lines.append("interventional_data={}")
    else:
        keys = sorted(int_groups_num.keys(), key=lambda kv: (str(kv[0]), str(kv[1])))
        if max_intervention_regimes is not None:
            keys = keys[: max(0, int(max_intervention_regimes))]

        interventional_dict: Dict[str, Any] = {}
        for (ivar, ival) in keys:
            n_i, items_full = _to_hist_items(int_groups_num[(ivar, ival)], num_states)
            items_full = _sort_items(items_full)
            items_out = _maybe_truncate(items_full)

            payload_i: Dict[str, Any] = {"n": n_i, "hist": items_out}
            if include_marginals and (max_hist_entries is None):
                payload_i["marginals"] = _marginals_from_items(num_states, items_full, n_i)

            label = _format_do_label(str(ivar), str(ival), state_map)
            interventional_dict[label] = payload_i

        # IMPORTANT: use the special dumper so nested arrays become compact
        lines.append("interventional_data=" + _dumps_interventional_hybrid(interventional_dict))

    if include_give_steps:
        lines.append("\n--- (THINKING STEPS) ---" if require_think_answer_blocks else "\n--- (SILENT STEPS) ---")
        lines.append("1) Use interventional shifts to identify descendants/orient edges; use joint patterns for colliders.")
        lines.append("2) Output a DAG as JSON.")

    lines.append("\n--- OUTPUT INSTRUCTIONS ---")
    lines.extend(
        _build_output_contract_lines(
            output_edge_list=output_edge_list,
            require_think_answer_blocks=require_think_answer_blocks,
        )
    )
    lines.append("Must be a DAG (acyclic).")

    lines.append("\n--- END ---")
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
    require_think_answer_blocks: bool = False,
    output_edge_list: bool = False,
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
        steps_header = (
            "\n(Use this in your <think>...</think> block.)"
            if require_think_answer_blocks else
            "\n(You may follow these steps silently.)"
        )
        if has_obs and has_interv:
            lines.extend([
                steps_header,
                "1) Use observational data to infer a plausible Markov equivalence class.",
                "2) Use interventional data to identify a single DAG within that class.",
                "Then output the JSON as specified above.",
            ])
        elif has_obs:
            lines.extend([
                steps_header,
                "1) Use observational conditional (in)dependencies to constrain the DAG.",
                "Then output the JSON as specified above.",
            ])
        elif has_interv:
            lines.extend([
                steps_header,
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

    lines.append("\n--- OUTPUT INSTRUCTIONS ---")
    lines.extend(
        _build_output_contract_lines(
            output_edge_list=output_edge_list,
            require_think_answer_blocks=require_think_answer_blocks,
        )
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
    ap.add_argument("--bif-file", default="../causal_graphs/real_data/small_graphs/cancer.bif")
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--obs-per-prompt", type=int, default=100)
    ap.add_argument("--int-per-combo", type=int, default=0)
    ap.add_argument("--intervene-vars", default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--anonymize", action="store_true")
    ap.add_argument("--causal-rules", action="store_true")
    ap.add_argument("--give-steps", action="store_true")
    ap.add_argument(
        "--thinking-tags",
        action="store_true",
        help="Require model outputs to use <think>...</think> and <answer>...</answer> blocks.",
    )
    ap.add_argument(
        "--no-thinking-tags",
        dest="thinking_tags",
        action="store_false",
        help="Disable <think>...</think><answer>...</answer> output contract and use JSON-only output.",
    )
    ap.set_defaults(thinking_tags=True)
    ap.add_argument("--def-int", action="store_true", help="Include a brief definition of interventions when interventional data are present.")
    ap.add_argument("--shuffles-per-graph", type=int, default=1)
    ap.add_argument("--given-edge-frac", type=float, default=0.0)
    ap.add_argument(
        "--prompt-style",
        choices=["cases", "matrix", "summary_joint"],
        default="cases",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Directory for outputs; defaults to experiments/prompts/<bif basename>",
    )
    
    # --- UPDATED ROBUSTNESS ARGS ---
    ap.add_argument("--row-order", choices=["random", "sorted", "reverse"], default="random")
    ap.add_argument("--col-order", choices=["original", "reverse", "random", "topo", "reverse_topo"], default="original")

    args = ap.parse_args()
    if args.int_per_combo > 0:
        iv = str(args.intervene_vars).strip().lower()
        if iv not in {"all", "none", ""}:
            print(
                f"[warn] --intervene-vars={args.intervene_vars!r} will generate interventions for only a subset of "
                f"variables (not all). If you expected do(X=*) rows for every variable, pass --intervene-vars all.",
                file=sys.stderr,
            )
    default_out_dir = os.path.join(
        os.path.dirname(__file__),
        "prompts",
        os.path.splitext(os.path.basename(args.bif_file))[0],
    )
    args.out_dir = args.out_dir or default_out_dir

    # 1. Load Graph
    bif_abs = Path(args.bif_file).resolve(strict=True)
    graph = _load_graph_file(str(bif_abs))
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
    if getattr(args, "thinking_tags", False): tags.append("thinktags")
    if hasattr(args, "prompt_style") and args.prompt_style in {"matrix", "summary_joint"}:
        tags.append(args.prompt_style)
    
    # ROBUSTNESS TAGS (Explicitly added to filename)
    if args.row_order != "random": tags.append(f"row{args.row_order}")
    if args.col_order != "original": tags.append(f"col{args.col_order}")

    extra_suffix = ("_" + "_".join(tags)) if tags else ""
    
    base_name = (
        f"prompts_obs{args.obs_per_prompt}"
        f"_int{args.int_per_combo}"
        f"_shuf{args.shuffles_per_graph}"
        f"_p{args.num_prompts}{extra_suffix}"
    )

    csv_path = out_dir / f"{base_name}.csv"
    answer_path = out_dir / f"{base_name}_answer.json"

    # Write Answer
    with answer_path.open("w", encoding="utf-8") as f_ans:
        json.dump(
            {"answer": answer_obj, "given_edges": None},
            f_ans,
            ensure_ascii=False,
            indent=2,
        )

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
            # Numeric rows in *permuted* column order (for summary prompts)
            obs_rows_num: List[List[float]] = []
            if args.obs_per_prompt > 0:
                for r in arr_obs:
                    obs_rows_num.append([float(r[idx]) for idx in col_indices])
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
            int_groups_num: Dict[Tuple[str, str], List[List[float]]] = {}
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
                        # Numeric vector in permuted order for summary aggregation
                        # always store code in the key
                        int_groups_num.setdefault((ivar_out, str(s_idx)), []).append([float(r[idx]) for idx in col_indices])

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
                use_edge_list_output = _use_edge_list_output(variables_out)
                is_names_only_cfg = (args.obs_per_prompt == 0 and args.int_per_combo == 0)
                if is_names_only_cfg:
                    # (obs=0,int=0) should always use the names-only prompt format, regardless of style.
                    from generate_prompts_names_only import format_names_only_prompt
                    prompt_text = format_names_only_prompt(
                        variables_out, dataset_name, args.causal_rules, output_edge_list=use_edge_list_output
                    )
                elif args.prompt_style == "summary_joint":
                    state_names = []
                    for orig_name in permuted_real_names:
                        states = codebook.get(orig_name, []) or []
                        if args.anonymize:
                            state_names.append([str(i) for i in range(len(states))] if states else [])
                        else:
                            state_names.append([str(s) for s in states] if states else [])
                    prompt_text = format_prompt_summary_full_joint(
                        variables_out,
                        dataset_name=dataset_name,
                        obs_rows_num=obs_rows_num,
                        int_groups_num=int_groups_num,
                        state_names=state_names if state_names else None,
                        include_causal_rules=args.causal_rules,
                        include_give_steps=args.give_steps,
                        include_def_int=include_def_int,
                        anonymize=args.anonymize,
                        include_probabilities=False,
                        sort_hist_by="count_desc",
                        require_think_answer_blocks=args.thinking_tags,
                        output_edge_list=use_edge_list_output,
                    )
                elif args.prompt_style == "matrix":
                    prompt_text = format_prompt_cb_matrix(
                        variables_out,
                        rows_text_source,
                        dataset_name,
                        include_causal_rules=args.causal_rules,
                        include_give_steps=args.give_steps,
                        include_def_int=include_def_int,
                        anonymize=args.anonymize,
                        require_think_answer_blocks=args.thinking_tags,
                        state_names=None,  # or build mapping if you want
                        output_edge_list=use_edge_list_output,
                    )
                else:
                    prompt_text = format_prompt_with_interventions(
                        variables_out, rows_text_source, vmap,
                        args.causal_rules, args.give_steps, include_def_int=include_def_int,
                        require_think_answer_blocks=args.thinking_tags,
                        output_edge_list=use_edge_list_output,
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
