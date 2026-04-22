#!/usr/bin/env python3
"""
collect_format_sft_data.py

Collect SFT examples that teach the model to start reasoning immediately
after <think> in the correct staged format.

Three modes
-----------
Mode A  --graphs graph1 [graph2 ...]
    Generate data in-memory from BIF files.  Each (graph, obs, int) config
    is fed through iter_prompts_in_memory with --col-perms random column
    orderings so variable order is varied automatically.

Mode B  (default — no --graphs and no --perm-csv)
    Discover *_obs100_int10_anon_train.csv and *_randcol_seed*.csv files
    under --data-dir (or read explicit --csv files) and build one SFT record
    per CSV row.

Mode C  --perm-csv
    Like Mode B (reads the same CSV sources) but instead of one record per
    row, enumerates up to --max-perms variable-order permutations for each of
    --rows-per-source rows.  The prompt text is rewritten in-place: VARIABLES,
    OBSERVATIONAL DATA, INTERVENTIONAL DATA, num_states, and tv_change_vs_obs
    are all permuted consistently.  Supersedes the former
    collect_permuted_sft_data.py.

Usage examples
--------------
    # Mode A — in-memory BIF, 5 column permutations each
    python experiments/collect_format_sft_data.py \\
        --graphs cancer earthquake asia sachs \\
        --col-perms 5 --num-prompts-per-config 500

    # Mode B — CSV discovery, 100 rows per source
    python experiments/collect_format_sft_data.py \\
        --data-dir experiments/data --n-per-source 100

    # Mode C — exhaustive permutation from existing CSV rows
    python experiments/collect_format_sft_data.py \\
        --perm-csv --rows-per-source 5 --max-perms 500

Output JSONL schema (one JSON object per line):
    {
      "prompt":  "<system>...<think>\\n",
      "answer":  "Stage 1 ...\\n\\n...Stage 3 ...</think><answer>{...}</answer>",
      "source":  "cancer_obs100_int10_anon_train",
      "graph":   "cancer"
    }

Compatible with run_sft.py --sft-jsonl.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

try:
    from cd_generation.format import validate_sft_example  # noqa: E402
    from cd_sft.staged_targets import (  # noqa: E402
        _load_adj,
        _extract_variables,
        build_evidence_grounded_sections,
        build_staged_sections,
    )
    from cd_generation.prompt_utils import render_prompt_text  # noqa: E402
except ImportError:
    from experiments.cd_generation.format import validate_sft_example  # noqa: E402
    from experiments.cd_sft.staged_targets import (  # noqa: E402
        _load_adj,
        _extract_variables,
        build_evidence_grounded_sections,
        build_staged_sections,
    )
    from experiments.cd_generation.prompt_utils import render_prompt_text  # noqa: E402
from generate_prompts import iter_prompts_in_memory  # noqa: E402


# ---------------------------------------------------------------------------
# CSV helpers (shared across all modes)
# ---------------------------------------------------------------------------

def _set_csv_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10_000_000)


def _iter_csv_rows(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def _sample_rows(path: Path, n: int, rng: random.Random) -> List[dict]:
    """Reservoir-sample n rows from a potentially large CSV."""
    reservoir: List[dict] = []
    for i, row in enumerate(_iter_csv_rows(path)):
        if i < n:
            reservoir.append(row)
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = row
    rng.shuffle(reservoir)
    return reservoir


def _build_think_sections(
    *,
    prompt_text: str,
    adj: List[List[int]],
    variables: List[str],
    think_style: str,
) -> Tuple[str, str, str]:
    if think_style == "strict":
        return build_staged_sections(adj, variables)
    if think_style == "evidence":
        return build_evidence_grounded_sections(prompt_text, adj, variables)
    raise ValueError(f"unsupported think_style={think_style!r}")


def _build_reasoning_text(
    *,
    prompt_text: str,
    adj: List[List[int]],
    variables: List[str],
    reasoning_target: str,
) -> Tuple[str, str, str, str]:
    if reasoning_target == "answer_only":
        return "", "", "", ""
    if reasoning_target == "stages":
        stage1_text, stage2_text, stage3_text = _build_think_sections(
            prompt_text=prompt_text,
            adj=adj,
            variables=variables,
            think_style="strict",
        )
    elif reasoning_target == "stages_evidence":
        stage1_text, stage2_text, stage3_text = _build_think_sections(
            prompt_text=prompt_text,
            adj=adj,
            variables=variables,
            think_style="evidence",
        )
    else:
        raise ValueError(f"unsupported reasoning_target={reasoning_target!r}")
    return stage1_text, stage2_text, stage3_text, f"{stage1_text}\n\n{stage2_text}\n\n{stage3_text}"


def _reasoning_style_label(reasoning_target: str) -> str:
    if reasoning_target == "stages":
        return "strict"
    if reasoning_target == "stages_evidence":
        return "evidence"
    return reasoning_target


# ---------------------------------------------------------------------------
# Mode B: record builder
# ---------------------------------------------------------------------------

def _build_record(
    row: dict,
    source_name: str,
    graph_name: str,
    prompt_col: str,
    answer_col: str,
    reasoning_target: str,
    wrapper_mode: str,
) -> Optional[dict]:
    """Return a validated SFT record or None if anything fails."""
    prompt_raw = (row.get(prompt_col) or "").strip()
    answer_raw = (row.get(answer_col) or "").strip()

    if not prompt_raw or not answer_raw:
        return None

    adj = _load_adj(answer_raw)
    if adj is None:
        return None

    variables = _extract_variables(prompt_raw)
    if variables is None or len(variables) != len(adj):
        n = len(adj)
        variables = [f"X{k+1}" for k in range(n)]

    try:
        stage1_text, stage2_text, stage3_text, think_text = _build_reasoning_text(
            prompt_text=prompt_raw,
            adj=adj,
            variables=variables,
            reasoning_target=reasoning_target,
        )
    except ValueError:
        return None

    prompt = render_prompt_text(
        prompt_raw,
        task="causal_discovery",
        wrapper_mode=wrapper_mode,
        prefill_think=True,
    )
    completion = (
        f"{think_text}</think>"
        f"<answer>{json.dumps({'adjacency_matrix': adj}, ensure_ascii=False)}</answer>"
    )

    issues = validate_sft_example(prompt, completion)
    if issues:
        return None

    row_graph_name = _resolve_graph_name_from_row(row, fallback=graph_name)

    return {
        "prompt": prompt,
        "answer": completion,
        "gold_think": think_text,
        "gold_stage1": stage1_text,
        "gold_stage2": stage2_text,
        "gold_stage3": stage3_text,
        "source": source_name,
        "graph": row_graph_name,
        "think_style": _reasoning_style_label(reasoning_target),
    }


def _resolve_variables_for_record(
    *,
    prompt_raw: str,
    adj: List[List[int]],
    item_variables: Optional[List[str]] = None,
    fallback_variables: Optional[List[str]] = None,
) -> List[str]:
    """Resolve the variable order for one record from the permuted prompt first."""
    n = len(adj)

    prompt_variables = _extract_variables(prompt_raw)
    if prompt_variables is not None and len(prompt_variables) == n:
        return [str(v) for v in prompt_variables]

    if item_variables is not None and len(item_variables) == n:
        return [str(v) for v in item_variables]

    if fallback_variables is not None and len(fallback_variables) == n:
        return [str(v) for v in fallback_variables]

    return [f"X{k+1}" for k in range(n)]


def _resolve_graph_name_from_row(row: dict, *, fallback: str) -> str:
    """Prefer row-level graph metadata over file-level source naming."""
    dataset = (row.get("dataset") or "").strip()
    if dataset:
        return dataset

    graph = (row.get("graph") or "").strip()
    if graph:
        return graph

    bif_file = (row.get("bif_file") or "").strip()
    if bif_file:
        return Path(bif_file).stem

    return fallback


# ---------------------------------------------------------------------------
# Mode A: in-memory generation from BIF files
# ---------------------------------------------------------------------------

def _build_records_in_memory(
    *,
    bif_file: Path,
    graph_name: str,
    prompt_style: str,
    obs_per_prompt: int,
    int_per_combo: int,
    num_prompts: int,
    seed: int,
    anonymize: bool,
    col_perms: int,
    reasoning_target: str,
    wrapper_mode: str,
) -> List[dict]:
    """
    Generate SFT records directly from a BIF file without writing intermediate CSVs.

    col_perms controls column-order diversity:
      - col_perms=1  : original column order only (seed used as-is)
      - col_perms=N  : 1 original + (N-1) random column permutations
                       Each permutation uses seed+i so the column shuffle differs.
    """
    source_tag = f"{graph_name}_obs{obs_per_prompt}_int{int_per_combo}"
    records: List[dict] = []

    for perm_idx in range(col_perms):
        col_order = "original" if perm_idx == 0 else "random"
        perm_seed = seed + perm_idx

        try:
            _base_name, answer_obj, prompt_iter = iter_prompts_in_memory(
                bif_file=str(bif_file),
                num_prompts=num_prompts,
                shuffles_per_graph=1,
                seed=perm_seed,
                prompt_style=prompt_style,
                obs_per_prompt=obs_per_prompt,
                int_per_combo=int_per_combo,
                row_order="random",
                col_order=col_order,
                anonymize=anonymize,
                causal_rules=False,
                give_steps=False,
                def_int=False,
                intervene_vars="all",
                wrapper_mode="plain",
            )
        except Exception as e:
            print(f"  [warn] {source_tag} perm_idx={perm_idx}: iter_prompts_in_memory failed: {e}",
                  file=sys.stderr)
            continue

        adj = answer_obj["adjacency_matrix"]
        fallback_variables = [str(v) for v in answer_obj["variables"]]

        for item in prompt_iter:
            prompt_raw = (item.get("prompt_text") or "").strip()
            if not prompt_raw:
                continue
            variables = _resolve_variables_for_record(
                prompt_raw=prompt_raw,
                adj=adj,
                item_variables=[str(v) for v in (item.get("variables") or [])],
                fallback_variables=fallback_variables,
            )

            try:
                stage1_text, stage2_text, stage3_text, think_text = _build_reasoning_text(
                    prompt_text=prompt_raw,
                    adj=adj,
                    variables=variables,
                    reasoning_target=reasoning_target,
                )
            except ValueError:
                continue

            prompt = render_prompt_text(
                prompt_raw,
                task="causal_discovery",
                wrapper_mode=wrapper_mode,
                prefill_think=True,
            )
            completion = (
                f"{think_text}</think>"
                f"<answer>{json.dumps({'adjacency_matrix': adj}, ensure_ascii=False)}</answer>"
            )

            if validate_sft_example(prompt, completion):
                continue

            records.append({
                "prompt": prompt,
                "answer": completion,
                "gold_think": think_text,
                "gold_stage1": stage1_text,
                "gold_stage2": stage2_text,
                "gold_stage3": stage3_text,
                "source": source_tag,
                "graph": graph_name,
                "think_style": _reasoning_style_label(reasoning_target),
            })

    return records


# ---------------------------------------------------------------------------
# Source discovery (Mode B and C)
# ---------------------------------------------------------------------------

def _discover_sources(data_dir: Path) -> List[Tuple[Path, str, str]]:
    """
    Return list of (csv_path, source_name, graph_name) tuples.

    Priority order:
      1. *_obs100_int10_anon_train.csv  — the canonical per-graph train splits
      2. *_randcol_seed*.csv            — column-permuted variants (larger)
    """
    sources: List[Tuple[Path, str, str]] = []

    for p in sorted(data_dir.glob("*_obs100_int10_anon_train.csv")):
        graph = p.stem.replace("_obs100_int10_anon_train", "")
        sources.append((p, p.stem, graph))

    for p in sorted(data_dir.glob("*_randcol_seed*.csv")):
        parts = p.stem.split("_randcol_seed")
        graph = parts[0]
        sources.append((p, p.stem, graph))

    return sources


# ---------------------------------------------------------------------------
# Mode C: prompt rewriting helpers
# ---------------------------------------------------------------------------

_VAR_BLOCK_RE = re.compile(r"(--- VARIABLES ---\n)(.*?)(\n---)", re.DOTALL)
_VAR_LINE_RE = re.compile(r"^(\s*\d+\s*:\s*)(\S+)", re.MULTILINE)
_OBS_BLOCK_RE = re.compile(
    r"(--- OBSERVATIONAL DATA ---\n.*?observational_data=\{)(.*?)(\}\s*\n---)", re.DOTALL
)
_INT_BLOCK_RE = re.compile(
    r"(--- INTERVENTIONAL DATA ---\n.*?interventional_data=\{)(.*?)(\}\s*(?:\n---|$))", re.DOTALL
)
_NUM_STATES_RE = re.compile(r"num_states=\[([^\]]+)\]")


def _extract_varnames_from_prompt(prompt: str) -> Optional[List[str]]:
    """Return variable name list parsed from the VARIABLES block, or None."""
    m = _VAR_BLOCK_RE.search(prompt)
    if not m:
        return None
    pairs = _VAR_LINE_RE.findall(m.group(2))
    return [name for _, name in pairs] if pairs else None


def _permute_assignment(assignment: List[int], perm: List[int]) -> List[int]:
    """new_assignment[i] = assignment[perm[i]]"""
    return [assignment[perm[i]] for i in range(len(perm))]


def _permute_obs_data(obs_data: dict, perm: List[int]) -> dict:
    new_hist = [[_permute_assignment(e[0], perm), e[1]] for e in obs_data["hist"]]
    new_marginals = [obs_data["marginals"][perm[i]] for i in range(len(perm))]
    return {"n": obs_data["n"], "hist": new_hist, "marginals": new_marginals}


def _permute_int_data(int_data: dict, perm: List[int], var_names: List[str]) -> dict:
    """Permute interventional data: rename do(Xi=v) keys and reorder hist slots."""
    inv_perm = [0] * len(perm)
    for new_i, old_i in enumerate(perm):
        inv_perm[old_i] = new_i

    new_int: dict = {}
    for key, val in int_data.items():
        m = re.match(r"do\((\w+)=(\d+)\)", key)
        if not m:
            new_int[key] = val
            continue
        old_var_name, v = m.group(1), m.group(2)
        if old_var_name not in var_names:
            new_int[key] = val
            continue
        new_idx = inv_perm[var_names.index(old_var_name)]
        new_key = f"do(X{new_idx + 1}={v})"
        new_hist = [[_permute_assignment(e[0], perm), e[1]] for e in val["hist"]]
        new_marginals = [val["marginals"][perm[i]] for i in range(len(perm))]
        new_int[new_key] = {"n": val["n"], "hist": new_hist, "marginals": new_marginals}
    return new_int


def _permute_adj(adj: List[List[int]], perm: List[int]) -> List[List[int]]:
    n = len(perm)
    return [[adj[perm[i]][perm[j]] for j in range(n)] for i in range(n)]


def _fmt_float(x: float) -> str:
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _obs_to_str(obs: dict) -> str:
    hist_parts = ", ".join(f"[{json.dumps(e[0])},{e[1]}]" for e in obs["hist"])
    marg_parts = ", ".join(
        f"[{', '.join(_fmt_float(v) for v in m)}]" for m in obs["marginals"]
    )
    return f'{{"n": {obs["n"]}, "hist": [{hist_parts}], "marginals": [{marg_parts}]}}'


def _int_to_str(int_data: dict) -> str:
    parts = []
    for key in sorted(int_data.keys()):
        val = int_data[key]
        hist_parts = ", ".join(f"[{json.dumps(e[0])},{e[1]}]" for e in val["hist"])
        marg_parts = ", ".join(
            f"[{', '.join(_fmt_float(v) for v in m)}]" for m in val["marginals"]
        )
        v_str = f'{{"n": {val["n"]}, "hist": [{hist_parts}], "marginals": [{marg_parts}]}}'
        parts.append(f'"{key}": {v_str}')
    return "{" + ", ".join(parts) + "}"


def _get_permutations(n: int, max_perms: int, rng: random.Random) -> List[List[int]]:
    """All n! permutations if n! <= max_perms, else sample max_perms distinct ones."""
    total = math.factorial(n)
    base = list(range(n))
    if total <= max_perms:
        return [list(p) for p in itertools.permutations(base)]
    seen: set = set()
    result: List[List[int]] = []
    attempts = 0
    while len(result) < max_perms and attempts < max_perms * 20:
        p = base[:]
        rng.shuffle(p)
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            result.append(p)
        attempts += 1
    return result


def _rewrite_prompt(
    prompt: str,
    perm: List[int],
    var_names: List[str],
) -> Optional[str]:
    """
    Rewrite a CD prompt with permuted variable order.

    perm[new_idx] = old_idx — new position i gets the variable at old_idx.
    var_names: original variable names (X1, X2, ...) in original order.
    """
    n = len(perm)

    # 1. Rewrite VARIABLES block
    def _replace_var_block(m: re.Match) -> str:
        lines = [f"{new_i}: X{new_i + 1}" for new_i in range(n)]
        return m.group(1) + "\n".join(lines) + m.group(3)

    prompt2 = _VAR_BLOCK_RE.sub(_replace_var_block, prompt)

    # 2. Rewrite OBSERVATIONAL DATA
    obs_m = _OBS_BLOCK_RE.search(prompt2)
    if not obs_m:
        return None
    try:
        obs_data = json.loads("{" + obs_m.group(2) + "}")
    except json.JSONDecodeError:
        try:
            obs_data = json.loads(obs_m.group(2).strip())
        except Exception:
            return None

    obs_str = _obs_to_str(_permute_obs_data(obs_data, perm))
    prompt3 = prompt2[:obs_m.start(2)] + obs_str + prompt2[obs_m.end(2):]

    # 3. Rewrite INTERVENTIONAL DATA
    int_m = _INT_BLOCK_RE.search(prompt3)
    if not int_m:
        return None
    try:
        int_data = json.loads("{" + int_m.group(2).strip() + "}")
    except json.JSONDecodeError:
        try:
            int_data = json.loads(int_m.group(2).strip())
        except Exception:
            return None

    int_str = _int_to_str(_permute_int_data(int_data, perm, var_names))
    prompt4 = prompt3[:int_m.start(2)] + int_str + prompt3[int_m.end(2):]

    # 4. Update num_states order
    ns_m = _NUM_STATES_RE.search(prompt4)
    if ns_m:
        orig_states = [int(x.strip()) for x in ns_m.group(1).split(",")]
        new_states = [orig_states[perm[i]] for i in range(n)]
        prompt4 = (
            prompt4[:ns_m.start(1)]
            + ",".join(str(s) for s in new_states)
            + prompt4[ns_m.end(1):]
        )

    # 5. Update tv_change_vs_obs variable names.
    # Format: tv_change_vs_obs=[["X3",0.25],["X1",0.10],...]
    inv_perm = [0] * n
    for new_i, old_i in enumerate(perm):
        inv_perm[old_i] = new_i

    _TV_PREFIX = "tv_change_vs_obs="
    tv_pos = prompt4.find(_TV_PREFIX)
    if tv_pos != -1:
        val_start = tv_pos + len(_TV_PREFIX)
        try:
            decoder = json.JSONDecoder()
            tv_list, end_offset = decoder.raw_decode(prompt4, val_start)
            new_tv = []
            for entry in tv_list:
                var_name, tv_val = entry[0], entry[1]
                if (isinstance(var_name, str)
                        and var_name.startswith("X")
                        and var_name[1:].isdigit()):
                    old_i = int(var_name[1:]) - 1
                    var_name = f"X{inv_perm[old_i] + 1}"
                new_tv.append([var_name, tv_val])
            tv_json = json.dumps(new_tv, separators=(",", ":"), ensure_ascii=False)
            prompt4 = prompt4[:val_start] + tv_json + prompt4[val_start + end_offset:]
        except Exception:
            pass  # leave unchanged on parse failure

    return prompt4


def _build_records_perm_csv(
    csv_path: Path,
    source_name: str,
    graph_name: str,
    rows_per_source: int,
    max_perms: int,
    rng: random.Random,
    prompt_col: str = "prompt",
    answer_col: str = "answer",
    reasoning_target: str = "stages_evidence",
    wrapper_mode: str = "chat",
) -> Tuple[List[dict], int, int]:
    """
    Build SFT records from one CSV by enumerating variable-order permutations.

    Returns (records, n_built, n_skipped).
    """
    rows = _sample_rows(csv_path, rows_per_source, rng)
    if not rows:
        return [], 0, 0

    var_names_0 = _extract_varnames_from_prompt((rows[0].get(prompt_col) or "").strip())
    if not var_names_0:
        return [], 0, len(rows)

    n = len(var_names_0)
    n_total = math.factorial(n)
    perms = _get_permutations(n, max_perms, rng)
    print(f"  {source_name}: n={n}, {n_total:,} total perms, "
          f"using {len(perms)} | rows={len(rows)}")

    records: List[dict] = []
    built = skipped = 0

    for row in rows:
        prompt_raw = (row.get(prompt_col) or "").strip()
        answer_raw = (row.get(answer_col) or "").strip()
        adj_orig = _load_adj(answer_raw)
        if adj_orig is None or len(adj_orig) != n:
            skipped += 1
            continue

        row_var_names = _extract_varnames_from_prompt(prompt_raw) or var_names_0
        row_graph_name = _resolve_graph_name_from_row(row, fallback=graph_name)
        row_seen: set = set()

        for perm in perms:
            adj_perm = _permute_adj(adj_orig, perm)
            adj_key = str(adj_perm)
            if adj_key in row_seen:
                continue  # automorphism: same graph under a different label

            prompt_perm = _rewrite_prompt(prompt_raw, perm, row_var_names)
            if prompt_perm is None:
                skipped += 1
                continue

            new_var_names = _extract_varnames_from_prompt(prompt_perm)
            if new_var_names is None or len(new_var_names) != n:
                skipped += 1
                continue

            prompt_final = render_prompt_text(
                prompt_perm,
                task="causal_discovery",
                wrapper_mode=wrapper_mode,
                prefill_think=True,
            )
            try:
                stage1_text, stage2_text, stage3_text, think_text = _build_reasoning_text(
                    prompt_text=prompt_perm,
                    adj=adj_perm,
                    variables=new_var_names,
                    reasoning_target=reasoning_target,
                )
            except ValueError:
                skipped += 1
                continue
            completion = (
                f"{think_text}</think>"
                f"<answer>{json.dumps({'adjacency_matrix': adj_perm}, ensure_ascii=False)}</answer>"
            )

            if validate_sft_example(prompt_final, completion):
                skipped += 1
                continue

            row_seen.add(adj_key)
            records.append({
                "prompt": prompt_final,
                "answer": completion,
                "gold_think": think_text,
                "gold_stage1": stage1_text,
                "gold_stage2": stage2_text,
                "gold_stage3": stage3_text,
                "source": source_name,
                "graph": row_graph_name,
                "perm": perm,
                "think_style": _reasoning_style_label(reasoning_target),
            })
            built += 1

    return records, built, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Collect format-teaching SFT examples for causal discovery. "
            "Mode A: in-memory BIF (--graphs). "
            "Mode B: CSV discovery [default]. "
            "Mode C: exhaustive permutation from CSV rows (--perm-csv)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("experiments/data/format_sft.jsonl"),
        help="Output JSONL path.",
    )

    # --- Mode A: in-memory BIF ---
    ap.add_argument(
        "--graphs", nargs="+", default=None,
        help=(
            "Graph names to generate from BIF files (Mode A). "
            "When set, data is generated in-memory — no CSV files needed."
        ),
    )
    ap.add_argument(
        "--graphs-dir", type=Path,
        default=Path("causal_graphs/real_data/small_graphs"),
        help="Directory containing *.bif files (Mode A).",
    )
    ap.add_argument(
        "--prompt-style",
        choices=["summary", "summary_joint", "matrix"],
        default="summary",
        help="Prompt style for in-memory generation (Mode A only).",
    )
    ap.add_argument(
        "--obs-values", nargs="+", type=int, default=[100],
        help="obs_per_prompt values (Mode A only).",
    )
    ap.add_argument(
        "--int-values", nargs="+", type=int, default=[10],
        help="int_per_combo values (Mode A only).",
    )
    ap.add_argument(
        "--num-prompts-per-config", type=int, default=500,
        help="Prompts per (graph, obs, int) config (Mode A only).",
    )
    ap.add_argument(
        "--col-perms", type=int, default=5,
        help=(
            "Column-order permutations per config (Mode A only). "
            "1 = original order only. N = 1 original + (N-1) random shuffles."
        ),
    )
    ap.add_argument(
        "--anonymize", action="store_true", default=False,
        help="Anonymize variable names to X1, X2, ... (Mode A only).",
    )
    ap.add_argument(
        "--wrapper-mode",
        choices=["plain", "chat"],
        default="chat",
        help="Prompt transport used for stored SFT prompts.",
    )

    # --- Mode B / C: CSV-based ---
    ap.add_argument(
        "--data-dir", type=Path, default=Path("experiments/data"),
        help="Directory containing source CSV files (Mode B/C).",
    )
    ap.add_argument(
        "--n-per-source", type=int, default=100,
        help="Max rows to sample per CSV source (Mode B only).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-col", default="prompt")
    ap.add_argument("--answer-col", default="answer")
    ap.add_argument(
        "--reasoning-target",
        choices=["answer_only", "stages", "stages_evidence"],
        default="stages_evidence",
        help="Supervised completion target style.",
    )
    ap.add_argument(
        "--csv", action="append", default=[], metavar="PATH[:GRAPH]",
        help=(
            "Explicit CSV file to include (repeatable). "
            "Optionally append :GRAPH_NAME (e.g. data/cancer.csv:cancer). "
            "When provided, auto-discovery from --data-dir is skipped unless "
            "--also-discover is also set."
        ),
    )
    ap.add_argument(
        "--also-discover", action="store_true",
        help="When --csv is given, also run auto-discovery from --data-dir.",
    )
    ap.add_argument(
        "--graph-filter", nargs="*",
        help="Only include sources whose graph name matches one of these.",
    )
    ap.add_argument(
        "--sources-only", action="store_true",
        help="List discovered sources and exit without writing output.",
    )

    # --- Mode C: perm-csv ---
    ap.add_argument(
        "--perm-csv", action="store_true",
        help=(
            "Enable exhaustive variable-order permutation mode (Mode C). "
            "Reads CSV sources (same discovery logic as Mode B) and generates "
            "up to --max-perms permutations for each of --rows-per-source rows."
        ),
    )
    ap.add_argument(
        "--rows-per-source", type=int, default=5,
        help="CSV rows to sample per source (Mode C only).",
    )
    ap.add_argument(
        "--max-perms", type=int, default=500,
        help=(
            "Max permutations per row (Mode C only). "
            "All n! permutations are used when n! <= this value."
        ),
    )

    args = ap.parse_args()
    if args.prompt_style == "summary_joint":
        args.prompt_style = "summary"
    _set_csv_limit()
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_records: List[dict] = []

    # ------------------------------------------------------------------ #
    # Mode A: in-memory BIF generation                                    #
    # ------------------------------------------------------------------ #
    if args.graphs:
        graph_filter = set(args.graph_filter) if args.graph_filter else None
        config_seed = args.seed
        for graph_name in args.graphs:
            if graph_filter and graph_name not in graph_filter:
                continue
            bif_file = args.graphs_dir / f"{graph_name}.bif"
            if not bif_file.exists():
                print(f"  [warn] BIF not found: {bif_file}", file=sys.stderr)
                continue
            for obs_n in args.obs_values:
                for int_n in args.int_values:
                    recs = _build_records_in_memory(
                        bif_file=bif_file,
                        graph_name=graph_name,
                        prompt_style=args.prompt_style,
                        obs_per_prompt=obs_n,
                        int_per_combo=int_n,
                        num_prompts=args.num_prompts_per_config,
                        seed=config_seed,
                        anonymize=args.anonymize,
                        col_perms=args.col_perms,
                        reasoning_target=args.reasoning_target,
                        wrapper_mode=args.wrapper_mode,
                    )
                    config_seed += 1000
                    all_records.extend(recs)
                    print(
                        f"  {graph_name:12s}  obs={obs_n:4d}  int={int_n:3d}"
                        f"  col_perms={args.col_perms}  → {len(recs)} records"
                    )

    # ------------------------------------------------------------------ #
    # Mode B / C: CSV-based                                               #
    # ------------------------------------------------------------------ #
    else:
        # Resolve CSV sources
        sources: List[Tuple[Path, str, str]] = []
        explicit_csvs = args.csv or []

        if explicit_csvs and not args.also_discover:
            for spec in explicit_csvs:
                if ":" in spec:
                    path_str, graph = spec.rsplit(":", 1)
                else:
                    path_str = spec
                    graph = Path(spec).stem.split("_")[0]
                p = Path(path_str)
                if not p.exists():
                    sys.exit(f"ERROR: --csv file not found: {p}")
                sources.append((p, p.stem, graph))
        else:
            data_dir = args.data_dir
            if not data_dir.is_dir():
                sys.exit(f"ERROR: data directory not found: {data_dir}")
            sources = _discover_sources(data_dir)
            for spec in explicit_csvs:
                if ":" in spec:
                    path_str, graph = spec.rsplit(":", 1)
                else:
                    path_str = spec
                    graph = Path(spec).stem.split("_")[0]
                p = Path(path_str)
                if not p.exists():
                    sys.exit(f"ERROR: --csv file not found: {p}")
                sources.append((p, p.stem, graph))

        if not sources:
            sys.exit("ERROR: no CSV sources found (check --data-dir or --csv)")

        if args.graph_filter:
            keep = set(args.graph_filter)
            sources = [(p, s, g) for p, s, g in sources if g in keep]
            if not sources:
                sys.exit(f"ERROR: no sources match graph filter {args.graph_filter}")

        if args.sources_only:
            for p, s, g in sources:
                print(f"  graph={g:20s}  source={s:45s}  path={p}")
            print(f"\n{len(sources)} sources total")
            return

        # ---- Mode C: exhaustive permutation ----
        if args.perm_csv:
            print(f"Mode C (perm-csv): {len(sources)} sources, "
                  f"rows_per_source={args.rows_per_source}, max_perms={args.max_perms}\n")
            unique_answers: set = set()
            for csv_path, source_name, graph_name in sources:
                recs, built, skipped = _build_records_perm_csv(
                    csv_path=csv_path,
                    source_name=source_name,
                    graph_name=graph_name,
                    rows_per_source=args.rows_per_source,
                    max_perms=args.max_perms,
                    rng=rng,
                    prompt_col=args.prompt_col,
                    answer_col=args.answer_col,
                    reasoning_target=args.reasoning_target,
                    wrapper_mode=args.wrapper_mode,
                )
                all_records.extend(recs)
                for rec in recs:
                    unique_answers.add(rec["answer"].split("<answer>")[-1])
                print(f"    → built={built}, skipped={skipped}, "
                      f"unique answers so far={len(unique_answers)}")

        # ---- Mode B: one record per CSV row ----
        else:
            print(f"Mode B (csv): {len(sources)} sources, "
                  f"n_per_source={args.n_per_source}, seed={args.seed}\n")
            for csv_path, source_name, graph_name in sources:
                sampled = _sample_rows(csv_path, args.n_per_source, rng)
                built = skipped = 0
                for row in sampled:
                    rec = _build_record(
                        row,
                        source_name=source_name,
                        graph_name=graph_name,
                        prompt_col=args.prompt_col,
                        answer_col=args.answer_col,
                        reasoning_target=args.reasoning_target,
                        wrapper_mode=args.wrapper_mode,
                    )
                    if rec is None:
                        skipped += 1
                    else:
                        all_records.append(rec)
                        built += 1
                print(f"  {source_name:50s}  built={built:4d}  skipped={skipped}")

    # Final shuffle + write
    rng.shuffle(all_records)

    with args.output.open("w", encoding="utf-8") as fout:
        for rec in all_records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_graph: dict[str, int] = {}
    for rec in all_records:
        by_graph[rec["graph"]] = by_graph.get(rec["graph"], 0) + 1

    print(f"\nWrote {len(all_records)} records to {args.output}")
    print("\nBreakdown by graph:")
    for g, cnt in sorted(by_graph.items()):
        print(f"  {g:20s}  {cnt}")

    if all_records:
        ex = all_records[0]
        print("\n--- Example completion (first record) ---")
        print(ex["answer"][:500])
        print("..." if len(ex["answer"]) > 500 else "")

    if not all_records:
        sys.exit("ERROR: no records written — check CSV format and column names")


if __name__ == "__main__":
    main()
