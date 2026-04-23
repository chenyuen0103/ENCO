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

try:
    from .prompt_utils import (
        LARGE_GRAPH_EDGE_LIST_THRESHOLD,
        build_causal_discovery_reminder_lines,
        build_causal_graph_assumption_lines,
        _build_output_contract_lines,
        get_topological_sort,
        normalize_variable_names,
        render_prompt_text,
        resolve_wrapper_mode,
    )
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from experiments.cd_generation.prompt_utils import (
        LARGE_GRAPH_EDGE_LIST_THRESHOLD,
        build_causal_discovery_reminder_lines,
        build_causal_graph_assumption_lines,
        _build_output_contract_lines,
        get_topological_sort,
        normalize_variable_names,
        render_prompt_text,
        resolve_wrapper_mode,
    )
try:
    from benchmark_builder.graph_io import load_causal_graph as _load_graph_file_full  # type: ignore
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


def format_names_only_prompt(
    variables: List[str],
    dataset_name: str,
    include_causal_rules: bool = False,
    output_edge_list: bool = False,
    require_think_answer_blocks: bool = False,
    anonymize: bool = False,
) -> str:
    lines = []
    lines.append("ROLE: You are an expert in causal discovery from variable semantics and background knowledge.")
    lines.append("TASK: Infer the directed causal graph over the variables.")
    if anonymize:
        lines.append("We are studying an anonymized causal system.")
    else:
        lines.append(f"We are studying a system called '{dataset_name}'.")
    lines.append("No observational or interventional data are provided for this case. Use only the variable meanings and relevant background causal knowledge.")
    lines.append("ASSUMPTIONS:")
    lines.extend(build_causal_graph_assumption_lines(include_intervention_assumption=False))
    
    lines.append("\n--- VARIABLE ORDER (ORDER MATTERS) ---")
    for i, var in enumerate(variables):
        lines.append(f"{i}: {var}")
    
    if include_causal_rules:
        lines.append("\n--- CAUSAL DISCOVERY REMINDERS ---")
        lines.extend(build_causal_discovery_reminder_lines())

    lines.append("\n--- AVAILABLE EVIDENCE ---")
    lines.append("No sampled data are provided. The graph must be inferred from the variable names and domain knowledge alone.")

    lines.append("\n--- OUTPUT INSTRUCTIONS ---")
    lines.extend(
        _build_output_contract_lines(
            output_edge_list=output_edge_list,
            require_think_answer_blocks=require_think_answer_blocks,
        )
    )

    return "\n".join(lines)


def iter_names_only_prompts_in_memory(
    *,
    bif_file: str,
    num_prompts: int,
    seed: int,
    col_order: str,
    anonymize: bool,
    causal_rules: bool,
    wrapper_mode: Optional[str] = None,
    append_format_hint: Optional[bool] = None,
    reasoning_guidance: Optional[str] = None,
    prefill_think: Optional[bool] = None,
    prefill_answer: bool = False,
) -> tuple[str, dict[str, Any], Iterator[dict[str, Any]]]:
    """
    Generate names-only prompts in-memory.
    Returns (base_name, answer_obj, iterator of rows with prompt_text).
    """
    graph_abs = Path(bif_file).resolve(strict=True)
    graph = load_graph_file(str(graph_abs))
    base_variables = normalize_variable_names(graph)
    nvars = len(base_variables)
    resolved_wrapper_mode = resolve_wrapper_mode(wrapper_mode=wrapper_mode)
    resolved_append_format_hint = bool(append_format_hint)
    require_think_answer_blocks = True

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
    use_edge_list_output = nvars > LARGE_GRAPH_EDGE_LIST_THRESHOLD
    edge_list = [
        [variables_out[i], variables_out[j]]
        for i in range(nvars)
        for j in range(nvars)
        if adj_bin[i][j] == 1
    ]
    answer_obj: Dict[str, Any] = {"variables": variables_out}
    if use_edge_list_output:
        answer_obj["edges"] = edge_list
    else:
        answer_obj["adjacency_matrix"] = adj_bin

    base_name = f"prompts_names_only_p{num_prompts}"
    if resolved_wrapper_mode != "plain":
        base_name += f"_wrap{resolved_wrapper_mode}"
    if resolved_append_format_hint:
        base_name += "_fmthint"
    if reasoning_guidance and str(reasoning_guidance).strip().lower() != "staged":
        base_name += f"_reason{str(reasoning_guidance).strip().lower()}"
    if col_order != "original":
        base_name += f"_col{col_order.capitalize()}"

    dataset_name = os.path.splitext(os.path.basename(bif_file))[0]
    prompt_text = format_names_only_prompt(
        variables_out,
        dataset_name,
        causal_rules,
        output_edge_list=use_edge_list_output,
        require_think_answer_blocks=require_think_answer_blocks,
        anonymize=anonymize,
    )
    prompt_text = render_prompt_text(
        prompt_text,
        task="causal_discovery",
        wrapper_mode=resolved_wrapper_mode,
        append_format_hint=resolved_append_format_hint,
        reasoning_guidance=str(reasoning_guidance or "staged"),
        prefill_think=prefill_think,
        prefill_answer=prefill_answer,
    )

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
    ap.add_argument("--bif-file", default="../causal_graphs/real_data/small_graphs/cancer.bif")
    ap.add_argument(
        "--graph-file",
        default=None,
        help="Generic graph path (.bif or .pt). If set, overrides --bif-file.",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--col-order", choices=["original", "reverse", "random", "topo", "reverse_topo"], default="original")
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--anonymize", action="store_true")
    ap.add_argument("--causal-rules", action="store_true")
    ap.add_argument(
        "--wrapper-mode",
        choices=["plain", "chat"],
        default=None,
        help="Prompt transport: plain text or system/user/assistant chat wrapper.",
    )
    ap.add_argument(
        "--append-format-hint",
        action="store_true",
        help=(
            "Append the canonical Formatting requirement line. For causal discovery this "
            "adds the optional stage-by-stage reasoning instructions."
        ),
    )
    ap.add_argument(
        "--reasoning-guidance",
        choices=["staged", "concise", "none"],
        default="staged",
        help="Control how much reasoning structure the prompt asks for.",
    )
    
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
    resolved_wrapper_mode = resolve_wrapper_mode(wrapper_mode=args.wrapper_mode)
    require_think_answer_blocks = True

    # 1. Load Graph
    graph_file = args.graph_file or args.bif_file
    graph_abs = Path(graph_file).resolve(strict=True)
    graph = load_graph_file(str(graph_abs))
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

    base_name = f"prompts_names_only_p{args.num_prompts}"
    if resolved_wrapper_mode != "plain":
        base_name += f"_wrap{resolved_wrapper_mode}"
    if args.append_format_hint:
        base_name += "_fmthint"
    if args.col_order != "original": base_name += f"_col{args.col_order.capitalize()}"
    
    csv_path = out_dir / f"{base_name}.csv"
    answer_path = out_dir / f"{base_name}_answer.json"

    use_edge_list_output = nvars > LARGE_GRAPH_EDGE_LIST_THRESHOLD
    edge_list = [
        [variables_out[i], variables_out[j]]
        for i in range(nvars)
        for j in range(nvars)
        if adj_bin[i][j] == 1
    ]
    answer_obj: Dict[str, Any] = {"variables": variables_out}
    if use_edge_list_output:
        answer_obj["edges"] = edge_list
    else:
        answer_obj["adjacency_matrix"] = adj_bin

    # Write Answer Key
    with answer_path.open("w", encoding="utf-8") as f_ans:
        json.dump({"answer": answer_obj, "given_edges": None}, f_ans, ensure_ascii=False, indent=2)

    # Write CSV
    csv_f = csv_path.open("w", newline="", encoding="utf-8")
    fieldnames = ["data_idx", "shuffle_idx", "prompt_path", "answer_path", "given_edges"]
    csv_writer = csv.DictWriter(csv_f, fieldnames=fieldnames, extrasaction="ignore")
    csv_writer.writeheader()

    # 5. Generate Prompts (Since there's no data, the prompt is identical for all 'replicates')
    dataset_name = os.path.splitext(os.path.basename(graph_file))[0]
    prompt_text = format_names_only_prompt(
        variables_out,
        dataset_name,
        args.causal_rules,
        output_edge_list=use_edge_list_output,
        require_think_answer_blocks=require_think_answer_blocks,
        anonymize=args.anonymize,
    )
    prompt_text = render_prompt_text(
        prompt_text,
        task="causal_discovery",
        wrapper_mode=resolved_wrapper_mode,
        append_format_hint=bool(args.append_format_hint),
        reasoning_guidance=args.reasoning_guidance,
    )
    
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
