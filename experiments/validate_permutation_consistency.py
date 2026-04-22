#!/usr/bin/env python3
"""
validate_permutation_consistency.py

Validate that column permutations from iter_prompts_in_memory propagate
consistently through:

1. Raw simulated data:
   - variable order
   - adjacency matrix
   - observational rows
   - interventional rows / keys
   - state metadata

2. Downstream descendant-task artifacts:
   - summary prompts
   - matrix prompts
   - descendant reasoning (CoT)

The validator compares an original-order run against a permuted run produced
from the same graph and random seed. The expected invariant is:

  permuted_artifact == original_artifact with the same permutation applied

This checks the actual source-permutation path used by
collect_descendant_sft_data.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from build_cd_descendant_tasks import _descendants_from_adj
from collect_descendant_sft_data import _compute_tv_from_rows, build_descendant_think
from generate_prompts import (
    format_prompt_descendants_matrix,
    format_prompt_descendants_summary,
    iter_prompts_in_memory,
)


_SECTION_RE = re.compile(r"^--- ([^-].*?) ---$", re.MULTILINE)
_THINK_STAGE1_RE = re.compile(
    r"^Stage 1 \(Shift Detection\) under do\(([^)]+)\), do_n=(\d+):\n\s*(.*)$",
    re.MULTILINE,
)
_THINK_STAGE3_DESC_RE = re.compile(
    r"^Stage 3 \(Conclusion\):\n\s*Descendants of ([^:]+): (.*)$",
    re.MULTILINE,
)
_THINK_STAGE3_NONE_RE = re.compile(
    r"^Stage 3 \(Conclusion\):\n\s*([^\s]+) has no descendants in this graph\.$",
    re.MULTILINE,
)


def _check(condition: bool, label: str, *, context: str = "") -> None:
    if not condition:
        suffix = f" [{context}]" if context else ""
        raise AssertionError(f"{label}{suffix}")


def _check_equal(actual, expected, label: str, *, context: str = "") -> None:
    if actual != expected:
        suffix = f" [{context}]" if context else ""
        raise AssertionError(
            f"{label}{suffix}\nACTUAL:   {actual!r}\nEXPECTED: {expected!r}"
        )


def _permute_rows(rows: List[List[float]], perm: List[int]) -> List[List[float]]:
    return [[row[perm[i]] for i in range(len(perm))] for row in rows]


def _permute_adj(adj: List[List[int]], perm: List[int]) -> List[List[int]]:
    n = len(perm)
    return [[adj[perm[i]][perm[j]] for j in range(n)] for i in range(n)]


def _permute_state_names(
    state_names: Optional[List[List[str]]],
    perm: List[int],
) -> Optional[List[List[str]]]:
    if state_names is None:
        return None
    return [state_names[perm[i]] for i in range(len(perm))]


def _infer_perm(original_variables: List[str], permuted_variables: List[str]) -> List[int]:
    index_by_var = {name: i for i, name in enumerate(original_variables)}
    return [index_by_var[name] for name in permuted_variables]


def _invert_perm(perm: List[int]) -> List[int]:
    inv = [0] * len(perm)
    for new_i, old_i in enumerate(perm):
        inv[old_i] = new_i
    return inv


def _expected_anon_int_groups(
    original_int_groups: Dict[Tuple[str, str], List[List[float]]],
    perm: List[int],
) -> Dict[Tuple[str, str], List[List[float]]]:
    inv_perm = _invert_perm(perm)
    result: Dict[Tuple[str, str], List[List[float]]] = {}
    for (ivar, ival), rows in original_int_groups.items():
        old_idx = int(str(ivar)[1:]) - 1
        new_idx = inv_perm[old_idx]
        result[(f"X{new_idx + 1}", str(ival))] = _permute_rows(rows, perm)
    return result


def _collect_examples(
    *,
    bif_file: Path,
    seed: int,
    col_order: str,
    anonymize: bool,
    num_prompts: int,
    obs_per_prompt: int,
    int_per_combo: int,
) -> Tuple[dict, List[dict]]:
    _base_name, answer_obj, prompt_iter = iter_prompts_in_memory(
        bif_file=str(bif_file),
        num_prompts=num_prompts,
        shuffles_per_graph=1,
        seed=seed,
        prompt_style="summary",
        obs_per_prompt=obs_per_prompt,
        int_per_combo=int_per_combo,
        row_order="random",
        col_order=col_order,
        anonymize=anonymize,
        causal_rules=False,
        give_steps=False,
        def_int=True,
        intervene_vars="all",
    )
    return answer_obj, list(prompt_iter)


def _iter_section_lines(text: str, section_name: str) -> List[str]:
    lines = text.splitlines()
    out: List[str] = []
    inside = False
    header = f"--- {section_name} ---"
    for line in lines:
        if line == header:
            inside = True
            continue
        if inside and line.startswith("--- ") and line.endswith(" ---"):
            break
        if inside:
            out.append(line)
    return out


def _parse_variable_order(prompt_text: str) -> List[str]:
    result: List[str] = []
    for line in _iter_section_lines(prompt_text, "VARIABLE ORDER"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        _, rhs = line.split(":", 1)
        rhs = rhs.strip()
        if " states=" in rhs:
            rhs = rhs.split(" states=", 1)[0].strip()
        result.append(rhs)
    return result


def _parse_json_value(prompt_text: str, key: str):
    prefix = f"{key}="
    for line in prompt_text.splitlines():
        if line.startswith(prefix):
            raw = line[len(prefix):].strip()
            if key == "tv_change_vs_obs" and "  #" in raw:
                raw = raw.split("  #", 1)[0].rstrip()
            return json.loads(raw)
    raise ValueError(f"Missing {key} in prompt")


def _parse_int_value(prompt_text: str, key: str) -> int:
    prefix = f"{key}="
    for line in prompt_text.splitlines():
        if line.startswith(prefix):
            return int(line[len(prefix):].strip())
    raise ValueError(f"Missing {key} in prompt")


def _parse_intervention_line(prompt_text: str) -> Tuple[str, str]:
    prefix = "intervention=do("
    for line in prompt_text.splitlines():
        if not line.startswith(prefix) or not line.endswith(")"):
            continue
        body = line[len(prefix):-1]
        target, value = body.split("=", 1)
        return target, value
    raise ValueError("Missing intervention line in prompt")


def _parse_matrix_prompt(prompt_text: str) -> dict:
    lines = prompt_text.splitlines()
    result = {
        "variables": _parse_variable_order(prompt_text),
        "obs_rows": [],
        "int_rows": [],
        "target": None,
        "value": None,
    }

    obs_header = "--- OBSERVATIONAL DATA (no intervention) ---"
    int_header = "--- INTERVENTIONAL DATA ---"
    int_label_re = re.compile(r"^\[Intervention: do\(([^=]+) = ([^)]+)\)\]$")

    i = 0
    while i < len(lines):
        line = lines[i]
        if line == obs_header:
            i += 3  # description + header row
            while i < len(lines) and lines[i] and not lines[i].startswith("--- "):
                result["obs_rows"].append(
                    [int(x.strip()) for x in lines[i].split("|")]
                )
                i += 1
            continue
        if line == int_header:
            i += 1
            m = int_label_re.match(lines[i])
            if not m:
                raise ValueError("Malformed intervention label in matrix prompt")
            result["target"], result["value"] = m.group(1), m.group(2)
            i += 2  # move to first row after header row
            while i < len(lines) and lines[i] and not lines[i].startswith("--- "):
                result["int_rows"].append(
                    [int(x.strip()) for x in lines[i].split("|")]
                )
                i += 1
            continue
        i += 1

    return result


def _parse_think(think_text: str) -> dict:
    m1 = _THINK_STAGE1_RE.search(think_text)
    if not m1:
        raise ValueError("Missing Stage 1 in think text")
    target = m1.group(1)
    do_n = int(m1.group(2))
    ranking_text = m1.group(3).strip()
    rankings: List[Tuple[str, float]] = []
    if ranking_text != "No intervention data available.":
        for part in ranking_text.split("  |  "):
            name, score = part.split(":", 1)
            rankings.append((name.strip(), float(score.strip())))

    m3 = _THINK_STAGE3_DESC_RE.search(think_text)
    if m3:
        desc_target = m3.group(1).strip()
        descendants = [s.strip() for s in m3.group(2).split(",") if s.strip()]
        return {
            "target": target,
            "do_n": do_n,
            "rankings": rankings,
            "stage3_target": desc_target,
            "descendants": descendants,
        }

    m3_none = _THINK_STAGE3_NONE_RE.search(think_text)
    if not m3_none:
        raise ValueError("Missing Stage 3 in think text")
    return {
        "target": target,
        "do_n": do_n,
        "rankings": rankings,
        "stage3_target": m3_none.group(1).strip(),
        "descendants": [],
    }


def _validate_summary_prompt(
    *,
    prompt_text: str,
    variables: List[str],
    target: str,
    intervention_value: str,
    obs_rows_num: List[List[float]],
    intervention_rows_num: List[List[float]],
    context: str,
) -> None:
    _check_equal(_parse_variable_order(prompt_text), variables, "summary prompt variable order", context=context)
    _check_equal(_parse_int_value(prompt_text, "obs_n"), len(obs_rows_num), "summary prompt obs_n", context=context)

    parsed_target, parsed_value = _parse_intervention_line(prompt_text)
    _check_equal((parsed_target, parsed_value), (target, str(intervention_value)), "summary prompt intervention", context=context)
    _check_equal(_parse_int_value(prompt_text, "do_n"), len(intervention_rows_num), "summary prompt do_n", context=context)

    m = len(variables)
    observed_scores, _ = _compute_tv_from_rows(
        variables,
        target,
        obs_rows_num,
        intervention_rows_num,
    )

    # Recompute marginals with the same logic used by the prompt formatter.
    num_states: List[int] = []
    for j in range(m):
        max_seen = -1
        for rows in (obs_rows_num, intervention_rows_num):
            for row in rows:
                max_seen = max(max_seen, int(round(float(row[j]))))
        num_states.append(max(2, max_seen + 1))

    def _marginals(rows: List[List[float]]) -> List[List[float]]:
        out: List[List[float]] = []
        for j in range(m):
            counts = [0.0] * num_states[j]
            for row in rows:
                idx = int(round(float(row[j])))
                counts[idx] += 1.0
            denom = float(len(rows)) if rows else 1.0
            out.append([round(c / denom, 6) for c in counts])
        return out

    obs_expected = {variables[j]: _marginals(obs_rows_num)[j] for j in range(m)}
    do_expected = {variables[j]: _marginals(intervention_rows_num)[j] for j in range(m)}

    _check_equal(_parse_json_value(prompt_text, "obs_marginals"), obs_expected, "summary prompt obs_marginals", context=context)
    _check_equal(_parse_json_value(prompt_text, "do_marginals"), do_expected, "summary prompt do_marginals", context=context)
    _check_equal(_parse_json_value(prompt_text, "tv_change_vs_obs"), [[v, s] for v, s in observed_scores], "summary prompt tv_change_vs_obs", context=context)


def _validate_matrix_prompt(
    *,
    prompt_text: str,
    variables: List[str],
    target: str,
    intervention_value: str,
    obs_rows_num: List[List[float]],
    intervention_rows_num: List[List[float]],
    context: str,
) -> None:
    parsed = _parse_matrix_prompt(prompt_text)
    _check_equal(parsed["variables"], variables, "matrix prompt variable order", context=context)
    _check_equal((parsed["target"], parsed["value"]), (target, str(intervention_value)), "matrix prompt intervention", context=context)
    obs_expected = [[int(round(float(x))) for x in row] for row in obs_rows_num]
    int_expected = [[int(round(float(x))) for x in row] for row in intervention_rows_num]
    _check_equal(parsed["obs_rows"], obs_expected, "matrix prompt observational rows", context=context)
    _check_equal(parsed["int_rows"], int_expected, "matrix prompt intervention rows", context=context)


def _validate_think(
    *,
    variables: List[str],
    adj: List[List[int]],
    target: str,
    obs_rows_num: List[List[float]],
    intervention_rows_num: List[List[float]],
    context: str,
) -> None:
    target_idx = variables.index(target)
    descendants = [variables[j] for j in _descendants_from_adj(adj, target_idx)]
    tv_scores, do_n = _compute_tv_from_rows(
        variables,
        target,
        obs_rows_num,
        intervention_rows_num,
    )
    think_text = build_descendant_think(
        target=target,
        variables=variables,
        descendants=descendants,
        tv_changes=tv_scores,
        do_n=do_n,
    )
    parsed = _parse_think(think_text)
    expected_rankings = [(name, round(score, 2)) for name, score in tv_scores]
    actual_rankings = [(name, round(score, 2)) for name, score in parsed["rankings"]]
    _check_equal(parsed["target"], target, "think Stage 1 target", context=context)
    _check_equal(parsed["do_n"], do_n, "think Stage 1 do_n", context=context)
    _check_equal(actual_rankings, expected_rankings, "think Stage 1 ranking", context=context)
    _check_equal(parsed["stage3_target"], target, "think Stage 3 target", context=context)
    _check_equal(sorted(parsed["descendants"]), sorted(descendants), "think Stage 3 descendants", context=context)


def _validate_real_permutation(
    *,
    original_answer: dict,
    permuted_answer: dict,
    original_items: Sequence[dict],
    permuted_items: Sequence[dict],
    perm: List[int],
) -> None:
    context = "real-name answer"
    _check_equal(
        permuted_answer["adjacency_matrix"],
        _permute_adj(original_answer["adjacency_matrix"], perm),
        "permuted adjacency matrix",
        context=context,
    )
    _check_equal(
        permuted_answer["variables"],
        [original_answer["variables"][i] for i in perm],
        "permuted variable order",
        context=context,
    )

    _check_equal(len(original_items), len(permuted_items), "item count", context="real-name items")
    for original_item, permuted_item in zip(original_items, permuted_items):
        item_context = f"real data_idx={original_item['data_idx']}"
        _check_equal(
            permuted_item["variables"],
            [original_item["variables"][i] for i in perm],
            "permuted item variables",
            context=item_context,
        )
        _check_equal(
            permuted_item["obs_rows_num"],
            _permute_rows(original_item["obs_rows_num"], perm),
            "permuted observational rows",
            context=item_context,
        )
        _check_equal(
            permuted_item["state_names"],
            _permute_state_names(original_item.get("state_names"), perm),
            "permuted state names",
            context=item_context,
        )

        _check_equal(
            set(permuted_item["int_groups_num"].keys()),
            set(original_item["int_groups_num"].keys()),
            "real-name intervention key set",
            context=item_context,
        )
        for key, rows in original_item["int_groups_num"].items():
            _check_equal(
                permuted_item["int_groups_num"][key],
                _permute_rows(rows, perm),
                f"permuted intervention rows for {key}",
                context=item_context,
            )

        variables = [str(v) for v in permuted_item["variables"]]
        adj = permuted_answer["adjacency_matrix"]
        for (target, value), int_rows in sorted(permuted_item["int_groups_num"].items()):
            prompt_summary = format_prompt_descendants_summary(
                variables,
                dataset_name=str(permuted_item.get("dataset_name", "")),
                intervention_target=str(target),
                intervention_value=str(value),
                intervention_rows_num=list(int_rows),
                obs_rows_num=list(permuted_item["obs_rows_num"]),
                state_names=permuted_item.get("state_names"),
                include_causal_rules=False,
                include_def_int=True,
                anonymize=False,
            )
            prompt_matrix = format_prompt_descendants_matrix(
                variables,
                dataset_name=str(permuted_item.get("dataset_name", "")),
                intervention_target=str(target),
                intervention_value=str(value),
                intervention_rows_num=list(int_rows),
                obs_rows_num=list(permuted_item["obs_rows_num"]),
                state_names=permuted_item.get("state_names"),
                include_causal_rules=False,
                include_def_int=True,
                anonymize=False,
            )
            prompt_context = f"{item_context} target={target} value={value}"
            _validate_summary_prompt(
                prompt_text=prompt_summary,
                variables=variables,
                target=str(target),
                intervention_value=str(value),
                obs_rows_num=list(permuted_item["obs_rows_num"]),
                intervention_rows_num=list(int_rows),
                context=prompt_context,
            )
            _validate_matrix_prompt(
                prompt_text=prompt_matrix,
                variables=variables,
                target=str(target),
                intervention_value=str(value),
                obs_rows_num=list(permuted_item["obs_rows_num"]),
                intervention_rows_num=list(int_rows),
                context=prompt_context,
            )
            _validate_think(
                variables=variables,
                adj=adj,
                target=str(target),
                obs_rows_num=list(permuted_item["obs_rows_num"]),
                intervention_rows_num=list(int_rows),
                context=prompt_context,
            )


def _validate_anonymized_permutation(
    *,
    original_answer: dict,
    permuted_answer: dict,
    original_items: Sequence[dict],
    permuted_items: Sequence[dict],
    perm: List[int],
) -> None:
    n = len(perm)
    anon_vars = [f"X{i + 1}" for i in range(n)]
    context = "anon answer"
    _check_equal(permuted_answer["variables"], anon_vars, "anon variable order", context=context)
    _check_equal(
        permuted_answer["adjacency_matrix"],
        _permute_adj(original_answer["adjacency_matrix"], perm),
        "anon permuted adjacency matrix",
        context=context,
    )

    _check_equal(len(original_items), len(permuted_items), "item count", context="anon items")
    for original_item, permuted_item in zip(original_items, permuted_items):
        item_context = f"anon data_idx={original_item['data_idx']}"
        _check_equal(permuted_item["variables"], anon_vars, "anon item variables", context=item_context)
        _check_equal(
            permuted_item["obs_rows_num"],
            _permute_rows(original_item["obs_rows_num"], perm),
            "anon permuted observational rows",
            context=item_context,
        )
        _check_equal(
            permuted_item["state_names"],
            _permute_state_names(original_item.get("state_names"), perm),
            "anon permuted state names",
            context=item_context,
        )
        _check_equal(
            permuted_item["int_groups_num"],
            _expected_anon_int_groups(original_item["int_groups_num"], perm),
            "anon permuted intervention groups",
            context=item_context,
        )

        variables = [str(v) for v in permuted_item["variables"]]
        adj = permuted_answer["adjacency_matrix"]
        for (target, value), int_rows in sorted(permuted_item["int_groups_num"].items()):
            prompt_summary = format_prompt_descendants_summary(
                variables,
                dataset_name=str(permuted_item.get("dataset_name", "")),
                intervention_target=str(target),
                intervention_value=str(value),
                intervention_rows_num=list(int_rows),
                obs_rows_num=list(permuted_item["obs_rows_num"]),
                state_names=permuted_item.get("state_names"),
                include_causal_rules=False,
                include_def_int=True,
                anonymize=True,
            )
            prompt_matrix = format_prompt_descendants_matrix(
                variables,
                dataset_name=str(permuted_item.get("dataset_name", "")),
                intervention_target=str(target),
                intervention_value=str(value),
                intervention_rows_num=list(int_rows),
                obs_rows_num=list(permuted_item["obs_rows_num"]),
                state_names=permuted_item.get("state_names"),
                include_causal_rules=False,
                include_def_int=True,
                anonymize=True,
            )
            prompt_context = f"{item_context} target={target} value={value}"
            _validate_summary_prompt(
                prompt_text=prompt_summary,
                variables=variables,
                target=str(target),
                intervention_value=str(value),
                obs_rows_num=list(permuted_item["obs_rows_num"]),
                intervention_rows_num=list(int_rows),
                context=prompt_context,
            )
            _validate_matrix_prompt(
                prompt_text=prompt_matrix,
                variables=variables,
                target=str(target),
                intervention_value=str(value),
                obs_rows_num=list(permuted_item["obs_rows_num"]),
                intervention_rows_num=list(int_rows),
                context=prompt_context,
            )
            _validate_think(
                variables=variables,
                adj=adj,
                target=str(target),
                obs_rows_num=list(permuted_item["obs_rows_num"]),
                intervention_rows_num=list(int_rows),
                context=prompt_context,
            )


def _print_debug_examples(
    *,
    perm: List[int],
    original_answer_real: dict,
    permuted_answer_real: dict,
    original_answer_anon: dict,
    permuted_answer_anon: dict,
    original_items_real: Sequence[dict],
    permuted_items_real: Sequence[dict],
    show_prompts: bool,
) -> None:
    print("\n=== Variable Order ===")
    print("real original :", original_answer_real["variables"])
    print("real permuted :", permuted_answer_real["variables"])
    print("anon original :", original_answer_anon["variables"])
    print("anon permuted :", permuted_answer_anon["variables"])
    print("perm(new_idx -> old_idx):", perm)

    if not original_items_real or not permuted_items_real:
        return

    original_item = original_items_real[0]
    permuted_item = permuted_items_real[0]

    print("\n=== First Item ===")
    print("data_idx:", original_item.get("data_idx"))
    print("dataset :", original_item.get("dataset_name"))
    print("original variables:", original_item.get("variables"))
    print("permuted variables:", permuted_item.get("variables"))
    print("original obs_rows_num:")
    print(json.dumps(original_item.get("obs_rows_num"), indent=2))
    print("permuted obs_rows_num:")
    print(json.dumps(permuted_item.get("obs_rows_num"), indent=2))
    print("original int_groups_num keys:", sorted(original_item.get("int_groups_num", {}).keys()))
    print("permuted int_groups_num keys:", sorted(permuted_item.get("int_groups_num", {}).keys()))

    orig_int_groups = original_item.get("int_groups_num", {})
    perm_int_groups = permuted_item.get("int_groups_num", {})
    if orig_int_groups and perm_int_groups:
        orig_key = sorted(orig_int_groups.keys(), key=lambda kv: (str(kv[0]), str(kv[1])))[0]
        perm_key = sorted(perm_int_groups.keys(), key=lambda kv: (str(kv[0]), str(kv[1])))[0]
        print(f"original rows for {orig_key}:")
        print(json.dumps(orig_int_groups[orig_key], indent=2))
        print(f"permuted rows for {perm_key}:")
        print(json.dumps(perm_int_groups[perm_key], indent=2))

        if show_prompts:
            orig_prompt = format_prompt_descendants_summary(
                [str(v) for v in original_item["variables"]],
                dataset_name=str(original_item.get("dataset_name", "")),
                intervention_target=str(orig_key[0]),
                intervention_value=str(orig_key[1]),
                intervention_rows_num=list(orig_int_groups[orig_key]),
                obs_rows_num=list(original_item["obs_rows_num"]),
                state_names=original_item.get("state_names"),
                include_causal_rules=False,
                include_def_int=True,
                anonymize=False,
            )
            perm_prompt = format_prompt_descendants_summary(
                [str(v) for v in permuted_item["variables"]],
                dataset_name=str(permuted_item.get("dataset_name", "")),
                intervention_target=str(perm_key[0]),
                intervention_value=str(perm_key[1]),
                intervention_rows_num=list(perm_int_groups[perm_key]),
                obs_rows_num=list(permuted_item["obs_rows_num"]),
                state_names=permuted_item.get("state_names"),
                include_causal_rules=False,
                include_def_int=True,
                anonymize=False,
            )
            print("\n=== First Summary Prompt Pair ===")
            print("\n--- original prompt ---")
            print(orig_prompt)
            print("\n--- permuted prompt ---")
            print(perm_prompt)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate that source-level column permutations propagate consistently through sampled data and descendant artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--bif-file",
        type=Path,
        default=Path("causal_graphs/real_data/small_graphs/cancer.bif"),
        help="BIF file to test.",
    )
    ap.add_argument(
        "--col-order",
        choices=["reverse", "random", "topo", "reverse_topo"],
        default="random",
        help="Permutation scheme to validate against original order.",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--num-prompts", type=int, default=2)
    ap.add_argument("--obs-per-prompt", type=int, default=20)
    ap.add_argument("--int-per-combo", type=int, default=4)
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print original/permuted variable orders and the first paired item.",
    )
    ap.add_argument(
        "--show-prompts",
        action="store_true",
        help="With --verbose, also print the first original/permuted summary prompt pair.",
    )
    args = ap.parse_args()

    if not args.bif_file.exists():
        sys.exit(f"ERROR: BIF file not found: {args.bif_file}")

    original_answer_real, original_items_real = _collect_examples(
        bif_file=args.bif_file,
        seed=args.seed,
        col_order="original",
        anonymize=False,
        num_prompts=args.num_prompts,
        obs_per_prompt=args.obs_per_prompt,
        int_per_combo=args.int_per_combo,
    )
    permuted_answer_real, permuted_items_real = _collect_examples(
        bif_file=args.bif_file,
        seed=args.seed,
        col_order=args.col_order,
        anonymize=False,
        num_prompts=args.num_prompts,
        obs_per_prompt=args.obs_per_prompt,
        int_per_combo=args.int_per_combo,
    )
    original_answer_anon, original_items_anon = _collect_examples(
        bif_file=args.bif_file,
        seed=args.seed,
        col_order="original",
        anonymize=True,
        num_prompts=args.num_prompts,
        obs_per_prompt=args.obs_per_prompt,
        int_per_combo=args.int_per_combo,
    )
    permuted_answer_anon, permuted_items_anon = _collect_examples(
        bif_file=args.bif_file,
        seed=args.seed,
        col_order=args.col_order,
        anonymize=True,
        num_prompts=args.num_prompts,
        obs_per_prompt=args.obs_per_prompt,
        int_per_combo=args.int_per_combo,
    )

    perm = _infer_perm(
        [str(v) for v in original_answer_real["variables"]],
        [str(v) for v in permuted_answer_real["variables"]],
    )

    _validate_real_permutation(
        original_answer=original_answer_real,
        permuted_answer=permuted_answer_real,
        original_items=original_items_real,
        permuted_items=permuted_items_real,
        perm=perm,
    )
    _validate_anonymized_permutation(
        original_answer=original_answer_anon,
        permuted_answer=permuted_answer_anon,
        original_items=original_items_anon,
        permuted_items=permuted_items_anon,
        perm=perm,
    )

    if args.verbose:
        _print_debug_examples(
            perm=perm,
            original_answer_real=original_answer_real,
            permuted_answer_real=permuted_answer_real,
            original_answer_anon=original_answer_anon,
            permuted_answer_anon=permuted_answer_anon,
            original_items_real=original_items_real,
            permuted_items_real=permuted_items_real,
            show_prompts=args.show_prompts,
        )

    print("Permutation consistency checks passed.")
    print(f"BIF: {args.bif_file}")
    print(f"col_order: {args.col_order}")
    print(f"seed: {args.seed}")
    print(f"num_prompts: {args.num_prompts}")
    print(f"obs_per_prompt: {args.obs_per_prompt}")
    print(f"int_per_combo: {args.int_per_combo}")
    print(f"perm(new_idx -> old_idx): {perm}")


if __name__ == "__main__":
    main()
