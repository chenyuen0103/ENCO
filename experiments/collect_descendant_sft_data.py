#!/usr/bin/env python3
"""
collect_descendant_sft_data.py

Generate SFT examples for the cd_descendants task from BIF files.

Each record teaches the model to:
  1. Rank variables by TV(obs, do) shift
  2. Apply a noise-aware threshold to classify shifted vs stable
  3. Output the ground-truth descendants from the adj matrix

Think is generated in three stages (analogous to the CD staged format):
  Stage 1 (Shift Detection): rank all variables by TV change
  Stage 2 (Threshold Analysis): classify each as shifted/stable
  Stage 3 (Conclusion): list final descendants

Variable-order diversity is achieved by passing col_order="random" with
different seeds to iter_prompts_in_memory, so the permuted adjacency matrix
and variable list are the ground truth from the start — no post-hoc rewriting.

Output JSONL schema:
  {
    "prompt":  "...assistant\\n<think>\\n",
    "answer":  "Stage 1...\\n\\nStage 3...\\n</think><answer>{...}</answer>",
    "graph":   "sachs",
    "source":  "sachs_obs100_int10",
    "target":  "Akt",
    "n_descendants": 2
  }

Usage:
    cd /u/chenyuen0103/ENCO
    python experiments/collect_descendant_sft_data.py \\
        --output experiments/data/descendant_sft.jsonl \\
        --graphs cancer earthquake asia sachs \\
        --obs-values 50 100 \\
        --int-values 10 50 \\
        --num-prompts 10 \\
        --col-perms 6 \\
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

try:
    from cd_generation.format import (
        DEFAULT_DESCENDANT_FORMAT_HINT_TEXT,
        canonicalize_cd_prompt,
        validate_sft_example,
    )
except ModuleNotFoundError:
    from experiments.cd_generation.format import (
        DEFAULT_DESCENDANT_FORMAT_HINT_TEXT,
        canonicalize_cd_prompt,
        validate_sft_example,
    )
from generate_prompts import (
    format_prompt_descendants_matrix,
    format_prompt_descendants_summary,
    iter_prompts_in_memory,
)
from build_cd_descendant_tasks import _descendants_from_adj


def _compute_tv_from_rows(
    variables: List[str],
    intervention_target: str,
    obs_rows_num: List[List[float]],
    intervention_rows_num: List[List[float]],
) -> Tuple[List[Tuple[str, float]], int]:
    """Compute TV(obs, do) scores directly from raw rows (used for matrix-style prompts)."""
    n = len(variables)
    do_n = len(intervention_rows_num)

    if not obs_rows_num or not intervention_rows_num:
        return [], do_n

    num_states = []
    for j in range(n):
        max_val = -1
        for rows in (obs_rows_num, intervention_rows_num):
            for row in rows:
                if j < len(row):
                    max_val = max(max_val, int(round(float(row[j]))))
        num_states.append(max(2, max_val + 1))

    def _marginals(rows: List[List[float]]) -> List[List[float]]:
        k = len(rows)
        result = []
        for j in range(n):
            counts = [0.0] * num_states[j]
            for row in rows:
                if j < len(row):
                    idx = int(round(float(row[j])))
                    if 0 <= idx < num_states[j]:
                        counts[idx] += 1.0
            result.append([c / k for c in counts])
        return result

    obs_probs = _marginals(obs_rows_num)
    do_probs = _marginals(intervention_rows_num)

    tv_scores: List[Tuple[str, float]] = []
    for j, var in enumerate(variables):
        if var == intervention_target:
            continue
        p, q = obs_probs[j], do_probs[j]
        tv = round(0.5 * sum(abs(p[s] - q[s]) for s in range(min(len(p), len(q)))), 6)
        tv_scores.append((var, tv))
    tv_scores.sort(key=lambda x: x[1], reverse=True)
    return tv_scores, do_n

# ---------------------------------------------------------------------------
# Step-by-step think generation
# ---------------------------------------------------------------------------

def _noise_threshold(do_n: int) -> float:
    """Heuristic TV threshold: below this is likely sampling noise."""
    if do_n <= 0:
        return 0.20
    if do_n <= 5:
        return 0.15
    if do_n <= 20:
        return 0.10
    if do_n <= 50:
        return 0.07
    return 0.05


def build_descendant_think(
    target: str,
    variables: List[str],
    descendants: List[str],
    tv_changes: List[Tuple[str, float]],
    do_n: int,
) -> str:
    """
    Generate a three-stage think for the cd_descendants task.

    Stage 1 (Shift Detection): rank all variables by TV change.
    Stage 2 (Threshold Analysis): classify each variable as shifted/stable,
        annotating with ground-truth labels so the reasoning is always correct.
    Stage 3 (Conclusion): list the final descendants.
    """
    desc_set = set(descendants)
    threshold = _noise_threshold(do_n)
    non_target_vars = [v for v in variables if v != target]

    # Stage 1
    if tv_changes:
        ranking = "  |  ".join(f"{v}: {tv:.2f}" for v, tv in tv_changes)
        stage1 = (
            f"Stage 1 (Shift Detection) under do({target}), do_n={do_n}:\n"
            f"  {ranking}"
        )
    else:
        stage1 = (
            f"Stage 1 (Shift Detection) under do({target}):\n"
            f"  No intervention data available."
        )

    # Stage 2
    tv_map: Dict[str, float] = dict(tv_changes)
    reliability = (
        " [low confidence: do_n < 10, estimates noisy]" if do_n < 10 else
        " [moderate confidence: do_n < 30]" if do_n < 30 else ""
    )
    lines2 = [f"Stage 2 (Threshold Analysis), noise threshold ≈ {threshold:.2f}{reliability}:"]

    shifted = [(v, tv_map[v]) for v, _ in tv_changes if tv_map[v] > threshold]
    stable  = [(v, tv_map[v]) for v, _ in tv_changes if tv_map[v] <= threshold]
    missing = [v for v in non_target_vars if v not in tv_map]

    for var, tv in shifted:
        annotation = (
            "shifted → descendant" if var in desc_set
            else "shifted → not descendant (spurious; sampling noise)"
        )
        lines2.append(f"  {var}: {tv:.2f} → {annotation}")

    for var, tv in stable:
        annotation = (
            "stable → descendant (weak signal; small do_n)" if var in desc_set
            else "stable → not descendant"
        )
        lines2.append(f"  {var}: {tv:.2f} → {annotation}")

    for var in missing:
        annotation = "no data → descendant" if var in desc_set else "no data → not descendant"
        lines2.append(f"  {var}: — → {annotation}")

    stage2 = "\n".join(lines2)

    # Stage 3
    if descendants:
        desc_str = ", ".join(sorted(descendants))
        stage3 = f"Stage 3 (Conclusion):\n  Descendants of {target}: {desc_str}"
    else:
        stage3 = f"Stage 3 (Conclusion):\n  {target} has no descendants in this graph."

    return f"{stage1}\n\n{stage2}\n\n{stage3}"


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def _build_records(
    *,
    bif_file: Path,
    graph_name: str,
    obs_per_prompt: int,
    int_per_combo: int,
    num_prompts: int,
    seed: int,
    anonymize: bool,
    col_perms: int,
    prompt_style: str = "summary",
) -> List[dict]:
    """
    Generate SFT records for one (graph, obs, int) configuration.

    Variable-order diversity comes from col_perms independent calls to
    iter_prompts_in_memory, each with a different seed and col_order="random"
    (except perm_idx=0 which uses the original order).  The permuted
    adjacency matrix and variable list returned by iter_prompts_in_memory
    are the ground truth — no post-hoc data rewriting is needed.
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
        except Exception as e:
            print(f"  [warn] {source_tag} perm_idx={perm_idx}: {e}", file=sys.stderr)
            continue

        # adj and variables are already in the permuted order — treat as ground truth.
        adj = answer_obj["adjacency_matrix"]
        variables_base = [str(v) for v in answer_obj["variables"]]

        for item in prompt_iter:
            item_variables = [str(v) for v in item.get("variables", variables_base)]
            n = len(item_variables)
            if len(adj) != n:
                continue

            dataset_name = str(item.get("dataset_name", graph_name))
            obs_rows_num = list(item.get("obs_rows_num") or [])
            state_names = item.get("state_names") or None
            int_groups_num: Dict[Tuple[str, str], List[List[float]]] = (
                item.get("int_groups_num") or {}
            )

            for (ivar, ival), intervention_rows_num in sorted(
                int_groups_num.items(),
                key=lambda kv: (str(kv[0][0]), str(kv[0][1])),
            ):
                target = str(ivar)
                try:
                    target_idx = item_variables.index(target)
                except ValueError:
                    continue

                descendants = [
                    item_variables[j]
                    for j in _descendants_from_adj(adj, target_idx)
                ]

                int_rows = list(intervention_rows_num)
                if prompt_style == "matrix":
                    prompt_raw = format_prompt_descendants_matrix(
                        item_variables,
                        dataset_name=dataset_name,
                        intervention_target=target,
                        intervention_value=str(ival),
                        intervention_rows_num=int_rows,
                        obs_rows_num=obs_rows_num,
                        state_names=state_names,
                        include_causal_rules=False,
                        include_def_int=True,
                        anonymize=anonymize,
                    )
                    tv_changes, do_n = _compute_tv_from_rows(
                        item_variables, target, obs_rows_num, int_rows
                    )
                else:
                    prompt_raw = format_prompt_descendants_summary(
                        item_variables,
                        dataset_name=dataset_name,
                        intervention_target=target,
                        intervention_value=str(ival),
                        intervention_rows_num=int_rows,
                        obs_rows_num=obs_rows_num,
                        state_names=state_names,
                        include_causal_rules=False,
                        include_def_int=True,
                        anonymize=anonymize,
                    )
                    tv_changes, do_n = _compute_tv_from_rows(
                        item_variables, target, obs_rows_num, int_rows
                    )

                think_text = build_descendant_think(
                    target=target,
                    variables=item_variables,
                    descendants=descendants,
                    tv_changes=tv_changes,
                    do_n=do_n,
                )

                prompt_canonical = canonicalize_cd_prompt(
                    prompt_raw,
                    task="cd_descendants",
                    wrap_system_prompt=True,
                    append_format_hint=True,
                    format_hint_text=DEFAULT_DESCENDANT_FORMAT_HINT_TEXT,
                    prefill_think=True,
                )

                answer_payload = json.dumps(
                    {"target": target, "descendants": sorted(descendants)},
                    ensure_ascii=False,
                )
                completion = f"{think_text}</think><answer>{answer_payload}</answer>"

                if validate_sft_example(prompt_canonical, completion):
                    continue

                records.append({
                    "prompt": prompt_canonical,
                    "answer": completion,
                    "graph": graph_name,
                    "source": source_tag,
                    "target": target,
                    "n_descendants": len(descendants),
                })

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate cd_descendants SFT data with step-by-step think.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--output", "-o", type=Path,
        default=Path("experiments/data/descendant_sft.jsonl"),
    )
    ap.add_argument(
        "--graphs-dir", type=Path,
        default=Path("causal_graphs/real_data/small_graphs"),
    )
    ap.add_argument(
        "--graphs", nargs="+",
        default=["cancer", "earthquake", "asia", "sachs"],
        help="Graph names (BIF stem) to include.",
    )
    ap.add_argument(
        "--obs-values", nargs="+", type=int,
        default=[50, 100],
        help="obs_per_prompt values to generate.",
    )
    ap.add_argument(
        "--int-values", nargs="+", type=int,
        default=[10, 50],
        help="int_per_combo values to generate.",
    )
    ap.add_argument(
        "--num-prompts", type=int, default=10,
        help="Number of independent data samples per (graph, obs, int, col_perm) config.",
    )
    ap.add_argument(
        "--col-perms", type=int, default=1,
        help=(
            "Number of variable-order permutations per config. "
            "1 = original order only. N = 1 original + (N-1) random column shuffles, "
            "each generated with a different seed so the permutation and data differ."
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--anonymize", action="store_true", default=False,
        help="Use anonymous variable names (X1, X2, ...) instead of real names.",
    )
    ap.add_argument(
        "--no-shuffle", action="store_true",
        help="Do not shuffle output records.",
    )
    ap.add_argument(
        "--prompt-style",
        choices=["summary", "matrix"],
        default="summary",
        help=(
            "'summary' (default): marginal/TV format. "
            "'matrix': raw case rows format."
        ),
    )
    # GRPO output options
    ap.add_argument(
        "--grpo-train", type=Path, default=None,
        metavar="PATH",
        help=(
            "If set, write a GRPO-ready CSV to this path. "
            "The 'answer' column contains raw JSON payload only (no CoT). "
            "When --grpo-eval is also set, only the train split is written here."
        ),
    )
    ap.add_argument(
        "--grpo-eval", type=Path, default=None,
        metavar="PATH",
        help="If set, write a GRPO eval CSV to this path (requires --grpo-train).",
    )
    ap.add_argument(
        "--grpo-eval-frac", type=float, default=0.1,
        help="Fraction of records to use as eval split (default: 0.1).",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_records: List[dict] = []
    config_seed = args.seed

    for graph_name in args.graphs:
        bif_file = args.graphs_dir / f"{graph_name}.bif"
        if not bif_file.exists():
            print(f"  [warn] BIF not found: {bif_file}", file=sys.stderr)
            continue

        for obs_n in args.obs_values:
            for int_n in args.int_values:
                recs = _build_records(
                    bif_file=bif_file,
                    graph_name=graph_name,
                    obs_per_prompt=obs_n,
                    int_per_combo=int_n,
                    num_prompts=args.num_prompts,
                    seed=config_seed,
                    anonymize=args.anonymize,
                    col_perms=args.col_perms,
                    prompt_style=args.prompt_style,
                )
                config_seed += 1000
                all_records.extend(recs)
                print(
                    f"  {graph_name:12s}  obs={obs_n:4d}  int={int_n:3d}"
                    f"  col_perms={args.col_perms}  → {len(recs)} records"
                )

    if not args.no_shuffle:
        rng.shuffle(all_records)

    with args.output.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # GRPO CSV output
    if args.grpo_train or args.grpo_eval:
        grpo_rows = []
        for rec in all_records:
            completion = rec["answer"]
            try:
                payload = completion.split("<answer>")[1].split("</answer>")[0].strip()
            except IndexError:
                payload = completion
            grpo_rows.append({
                "prompt_text": rec["prompt"],
                "answer": payload,
                "dataset": rec["graph"],
                "source": rec["source"],
                "target": rec["target"],
                "n_descendants": rec["n_descendants"],
                "prompt_style": args.prompt_style,
            })

        _GRPO_FIELDS = [
            "prompt_text", "answer", "dataset", "source",
            "target", "n_descendants", "prompt_style",
        ]

        if args.grpo_train and not args.grpo_eval:
            args.grpo_train.parent.mkdir(parents=True, exist_ok=True)
            with args.grpo_train.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=_GRPO_FIELDS)
                w.writeheader()
                w.writerows(grpo_rows)
            print(f"[grpo] wrote {len(grpo_rows)} rows → {args.grpo_train}")

        elif args.grpo_train and args.grpo_eval:
            n_eval = max(1, int(len(grpo_rows) * args.grpo_eval_frac))
            eval_rows = grpo_rows[:n_eval]
            train_rows = grpo_rows[n_eval:]
            for path, rows, label in [
                (args.grpo_train, train_rows, "train"),
                (args.grpo_eval, eval_rows, "eval"),
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=_GRPO_FIELDS)
                    w.writeheader()
                    w.writerows(rows)
                print(f"[grpo] {label}: wrote {len(rows)} rows → {path}")

        elif args.grpo_eval and not args.grpo_train:
            print("[warn] --grpo-eval requires --grpo-train; skipping.", file=sys.stderr)

    # Summary
    by_graph: Dict[str, int] = {}
    by_desc: Dict[int, int] = {}
    for rec in all_records:
        by_graph[rec["graph"]] = by_graph.get(rec["graph"], 0) + 1
        n = rec["n_descendants"]
        by_desc[n] = by_desc.get(n, 0) + 1

    print(f"\nWrote {len(all_records)} records → {args.output}")
    print("\nBy graph:")
    for g, cnt in sorted(by_graph.items()):
        print(f"  {g:20s}  {cnt}")
    print("\nBy descendant count:")
    for n, cnt in sorted(by_desc.items()):
        print(f"  n_descendants={n:2d}  {cnt}")

    if all_records:
        ex = all_records[0]
        print(f"\n--- Example (graph={ex['graph']}, target={ex['target']}) ---")
        print("PROMPT:\n" + ex["prompt"])
        print("THINK:\n" + ex["answer"].split("</think>")[0])
        print("ANSWER: " + ex["answer"].split("</think>")[1])


if __name__ == "__main__":
    main()
