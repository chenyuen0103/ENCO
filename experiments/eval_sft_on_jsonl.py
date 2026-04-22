#!/usr/bin/env python3
"""
eval_sft_on_jsonl.py

Validate a (LoRA) SFT model on a JSONL/CSV eval set.

Each JSONL record must have:
  - "prompt":  the raw prefill text, ending with "assistant\\n<think>\\n"
  - "answer":  ground-truth completion (think content + </think><answer>...</answer>)
  - "source":  CSV source name   (for per-source breakdown)
  - "graph":   graph name        (for per-graph breakdown)

CSV records may use: "prompt", "prompt_text", "prompt_path", "answer", "answer_path",
"graph" (or "dataset"), "source".

Metrics reported per example and aggregated:
    - tags_correct:         1 if completion has the expected think/answer tag structure
    - adj_matrix_present:   1 if adjacency matrix extracted from completion answer payload
    - exact_match:          1 if predicted adj == ground truth adj (element-wise)
    - edge_f1:              F1 over directed edges (pred vs. truth)
    - prompt_copy:          1 if completion looks like it re-printed the prompt (hallucination)

GRPO reward components are computed by default (generation stops at </answer>).
Pass --no-compute-rewards to skip reward computation and run quality metrics only.

Usage:
    cd /u/chenyuen0103/ENCO

    # Evaluate generation quality on a dataset
    python experiments/eval_sft_on_jsonl.py \\
        --model experiments/checkpoints/causal_sft_matrix_v1 \\
        --jsonl experiments/data/permuted_sft.jsonl \\
        --n 200 \\
        --max-new-tokens 4096 \\
        --output-jsonl experiments/checkpoints/causal_sft_matrix_v1/eval_permuted_sft.jsonl

    # Skip GRPO reward computation (quality metrics only, faster):
    python experiments/eval_sft_on_jsonl.py \\
        --model experiments/checkpoints/causal_sft_matrix_v1 \\
        --jsonl experiments/data/permuted_sft.jsonl \\
        --n 200 \\
        --no-compute-rewards

    # Check GRPO multi-GPU compatibility only (no generation, fast):
    python experiments/eval_sft_on_jsonl.py \\
        --model experiments/checkpoints/causal_sft_matrix_v1 \\
        --jsonl experiments/data/permuted_sft.jsonl \\
        --check-grpo-compat

    # To eval a base model (no LoRA adapter):
    python experiments/eval_sft_on_jsonl.py \\
        --model Qwen/Qwen3-4B-Thinking-2507 \\
        --jsonl experiments/data/permuted_sft.jsonl \\
        --n 100
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from verifier_cd import (
    build_cd_cot_structure_reward,
    build_cd_descendant_cot_structure_reward,
    build_cd_descendant_f1_reward,
    build_cd_descendant_partial_format_reward,
    build_cd_descendant_shift_ranking_reward,
    build_cd_descendant_variable_classification_reward,
    build_cd_edge_f1_reward,
    build_cd_format_reward,
    build_cd_graph_reward,
    build_cd_low_shd_reward,
    build_cd_orientation_f1_reward,
    build_cd_skeleton_f1_reward,
    build_cd_stage_targets,
    build_cd_vstruct_f1_reward,
    build_length_penalty_reward,
    extract_adjacency_matrix,
    extract_answer_text,
    extract_descendant_payload,
    format_ok as _format_ok,
    _looks_like_prompt_copy,
    _set_f1,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path, n: Optional[int], rng: random.Random) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if n is not None and n < len(records):
        records = rng.sample(records, n)
    return records


def _load_records(path: Path, n: Optional[int], rng: random.Random) -> List[dict]:
    """Load JSONL or CSV into a unified list of dicts."""
    if path.suffix.lower() in {".jsonl", ".jsonlines"}:
        return _load_jsonl(path, n=n, rng=rng)

    records: List[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            prompt = (row.get("prompt") or row.get("prompt_text") or "").strip()
            answer = (row.get("answer") or "").strip()

            if not prompt:
                prompt_path_raw = (row.get("prompt_path") or "").strip()
                if prompt_path_raw:
                    prompt_path = Path(prompt_path_raw)
                    if not prompt_path.is_absolute():
                        prompt_path = (path.parent / prompt_path).resolve()
                    prompt = prompt_path.read_text(encoding="utf-8", errors="ignore")

            if not answer:
                answer_path_raw = (row.get("answer_path") or "").strip()
                if answer_path_raw:
                    answer_path = Path(answer_path_raw)
                    if not answer_path.is_absolute():
                        answer_path = (path.parent / answer_path).resolve()
                    answer = answer_path.read_text(encoding="utf-8", errors="ignore")

            if not prompt or not answer:
                continue

            records.append({
                "prompt": prompt,
                "answer": answer,
                "graph": row.get("graph", row.get("dataset", "")),
                "source": row.get("source", ""),
            })

    if n is not None and n < len(records):
        records = rng.sample(records, n)
    return records


# ---------------------------------------------------------------------------
# Task detection
# ---------------------------------------------------------------------------

def _detect_task(records: List[dict]) -> str:
    """Infer task type from answer field: 'cd_descendants' or 'cd'."""
    for rec in records[:10]:
        ans = rec.get("answer", "")
        payload = extract_descendant_payload(ans)
        if payload is not None:
            return "cd_descendants"
    return "cd"


# ---------------------------------------------------------------------------
# Quality-metric scoring
# ---------------------------------------------------------------------------

def _adj_from_answer_field(raw: str) -> Optional[List[List[int]]]:
    return extract_adjacency_matrix(raw)


def _edge_f1(pred: List[List[int]], truth: List[List[int]]) -> float:
    n = len(truth)
    pred_edges = {(i, j) for i in range(n) for j in range(n) if pred[i][j]}
    true_edges = {(i, j) for i in range(n) for j in range(n) if truth[i][j]}
    if not true_edges and not pred_edges:
        return 1.0
    if not true_edges or not pred_edges:
        return 0.0
    tp = len(pred_edges & true_edges)
    precision = tp / len(pred_edges)
    recall = tp / len(true_edges)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _score_cd(completion: str, ground_truth_answer: str) -> dict:
    fmt = _format_ok(completion)
    prompt_copy = int(_looks_like_prompt_copy(completion))

    ans_text = extract_answer_text(completion)
    truth_adj = _adj_from_answer_field(ground_truth_answer)
    expected_n = len(truth_adj) if truth_adj is not None else None

    pred_adj = extract_adjacency_matrix(ans_text, expected_n=expected_n)
    if pred_adj is None:
        pred_adj = extract_adjacency_matrix(completion, expected_n=expected_n)

    parse_ok = int(pred_adj is not None)
    tags_correct = int(fmt)
    exact_match = 0
    edge_f1 = 0.0

    if pred_adj is not None and truth_adj is not None:
        exact_match = int(pred_adj == truth_adj)
        edge_f1 = _edge_f1(pred_adj, truth_adj)

    return {
        "tags_correct": tags_correct,
        "payload_present": parse_ok,
        "exact_match": exact_match,
        "primary_f1": edge_f1,
        "prompt_copy": prompt_copy,
        "answer_text": ans_text,
    }


def _score_descendant(completion: str, ground_truth_answer: str) -> dict:
    fmt = _format_ok(completion)
    prompt_copy = int(_looks_like_prompt_copy(completion))

    ans_text = extract_answer_text(completion)

    truth_payload = extract_descendant_payload(ground_truth_answer)
    pred_payload = extract_descendant_payload(ans_text)
    if pred_payload is None:
        pred_payload = extract_descendant_payload(completion)

    parse_ok = int(pred_payload is not None)
    tags_correct = int(fmt)
    exact_match = 0
    descendant_f1 = 0.0

    if pred_payload is not None and truth_payload is not None:
        target_ok = pred_payload["target"] == truth_payload["target"]
        if target_ok:
            descendant_f1 = _set_f1(pred_payload["descendants"], truth_payload["descendants"])
            exact_match = int(set(pred_payload["descendants"]) == set(truth_payload["descendants"]))

    return {
        "tags_correct": tags_correct,
        "payload_present": parse_ok,
        "exact_match": exact_match,
        "primary_f1": descendant_f1,
        "prompt_copy": prompt_copy,
        "answer_text": ans_text,
    }


def _score(completion: str, ground_truth_answer: str, task: str = "cd") -> dict:
    if task == "cd_descendants":
        return _score_descendant(completion, ground_truth_answer)
    return _score_cd(completion, ground_truth_answer)


# ---------------------------------------------------------------------------
# Aggregate stats / readiness helpers
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.pstdev(values)


def _has_tokenizer_assets(path: Path) -> bool:
    asset_names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
    }
    return any((path / name).exists() for name in asset_names)


def _normalize_model_ref(model_ref: str) -> tuple[str, bool]:
    path = Path(str(model_ref)).expanduser()
    if path.exists():
        return str(path.resolve()), True
    return str(model_ref), False


def _resolve_tokenizer_source(model_path: str, base_model_id: str) -> str:
    model_dir = Path(model_path)
    if model_dir.exists():
        candidates = [model_dir]
        if model_dir.name.startswith("checkpoint-") and model_dir.parent.exists():
            candidates.append(model_dir.parent)
        for candidate in candidates:
            if _has_tokenizer_assets(candidate):
                return str(candidate)
    return base_model_id


def _prepare_tokenizer_source(tokenizer_source: str, base_model_id: str) -> str:
    """
    Return a tokenizer source path safe for AutoTokenizer.from_pretrained().

    Some local Qwen checkpoint tokenizer_config.json files store
    `extra_special_tokens` as a list, while newer Transformers expects a dict.
    When that shape is detected, create a temporary sanitized tokenizer directory:
      - copy tokenizer assets from the base model when available
      - overlay adapter/chat-template files from the local checkpoint
      - if needed, rewrite tokenizer_config.json with a dict-valued
        `extra_special_tokens`
    """
    src_path = Path(str(tokenizer_source)).expanduser()
    if not src_path.exists():
        return tokenizer_source

    cfg_path = src_path / "tokenizer_config.json"
    if not cfg_path.exists():
        return tokenizer_source

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return tokenizer_source

    if not isinstance(cfg.get("extra_special_tokens"), list):
        return tokenizer_source

    base_path = Path(str(base_model_id)).expanduser()
    base_has_assets = base_path.exists() and _has_tokenizer_assets(base_path)
    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f"enco_eval_tok_{src_path.name}_", dir="/tmp")
    )
    asset_names = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "spiece.model",
    ]

    if base_has_assets:
        for name in asset_names:
            src = base_path / name
            if src.exists():
                shutil.copy2(src, tmp_dir / name)
        if (src_path / "chat_template.jinja").exists():
            shutil.copy2(src_path / "chat_template.jinja", tmp_dir / "chat_template.jinja")
        if (src_path / "tokenizer.json").exists():
            shutil.copy2(src_path / "tokenizer.json", tmp_dir / "tokenizer.json")
    else:
        for name in asset_names:
            src = src_path / name
            if src.exists():
                shutil.copy2(src, tmp_dir / name)

    tmp_cfg_path = tmp_dir / "tokenizer_config.json"
    if tmp_cfg_path.exists():
        try:
            tmp_cfg = json.loads(tmp_cfg_path.read_text(encoding="utf-8"))
            if isinstance(tmp_cfg.get("extra_special_tokens"), list):
                tmp_cfg["extra_special_tokens"] = {}
                tmp_cfg_path.write_text(json.dumps(tmp_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    print(f"[tokenizer] sanitized tokenizer assets -> {tmp_dir}", flush=True)
    return str(tmp_dir)


def _build_reward_kwargs(prompt: str, ground_truth: str, task: str) -> dict[str, list[Any]]:
    reward_kwargs: dict[str, list[Any]] = {
        "prompt": [prompt],
        "answer": [ground_truth],
    }
    if task == "cd":
        stage_targets = build_cd_stage_targets(prompt, ground_truth) or {}
        reward_kwargs.update({
            "target_stage1_skeleton_edges": [stage_targets.get("target_stage1_skeleton_edges", [])],
            "target_stage2_vstructures": [stage_targets.get("target_stage2_vstructures", [])],
            "target_stage3_directed_edges": [stage_targets.get("target_stage3_directed_edges", [])],
        })
    return reward_kwargs


def _summarize_rollouts(
    rollouts: list[dict[str, Any]],
    *,
    reward_names: list[str],
) -> dict[str, Any]:
    totals = [float(item.get("reward_total", 0.0)) for item in rollouts]
    f1s = [float(item["primary_f1"]) for item in rollouts]
    unique_completions = len({str(item.get("completion", "")) for item in rollouts})
    scoreable = [
        int(bool(item["tags_correct"]) and bool(item["payload_present"]) and not bool(item["prompt_copy"]))
        for item in rollouts
    ]

    summary: dict[str, Any] = {
        "num_rollouts": len(rollouts),
        "tags_correct_rate": _mean([float(item["tags_correct"]) for item in rollouts]),
        "payload_present_rate": _mean([float(item["payload_present"]) for item in rollouts]),
        "exact_match_rate": _mean([float(item["exact_match"]) for item in rollouts]),
        "primary_f1_mean": _mean(f1s),
        "primary_f1_std": _std(f1s),
        "prompt_copy_rate": _mean([float(item["prompt_copy"]) for item in rollouts]),
        "scoreable_rollout_rate": _mean([float(v) for v in scoreable]),
        "all_rollouts_scoreable": int(bool(rollouts) and all(scoreable)),
        "any_rollout_scoreable": int(any(scoreable)),
        "unique_completion_count": unique_completions,
        "completion_collapse": int(unique_completions <= 1),
        "gen_seconds_mean": _mean([float(item.get("gen_seconds", 0.0)) for item in rollouts]),
    }

    if totals:
        positive_rewards = [t for t in totals if t > 0.0]
        summary.update({
            "reward_total_mean": _mean(totals),
            "reward_total_std": _std(totals),
            "reward_total_min": min(totals),
            "reward_total_max": max(totals),
            "reward_total_span": max(totals) - min(totals),
            "reward_total_nonzero_variance": int((max(totals) - min(totals)) > 1e-9),
            "reward_total_unique_count": len({round(t, 8) for t in totals}),
            "all_zero_reward": int(all(abs(t) <= 1e-9 for t in totals)),
            "any_positive_reward": int(bool(positive_rewards)),
            "positive_reward_rate": _mean([float(t > 0.0) for t in totals]),
        })
        reward_component_stats: dict[str, dict[str, float]] = {}
        for reward_name in reward_names:
            values = [float(item.get("rewards", {}).get(reward_name, 0.0)) for item in rollouts]
            reward_component_stats[reward_name] = {
                "mean": _mean(values),
                "std": _std(values),
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
            }
        summary["reward_component_stats"] = reward_component_stats

    return summary


def _print_overall_summary(
    prompt_results: list[dict[str, Any]],
    *,
    task: str,
    payload_label: str,
    f1_label: str,
    reward_names: list[str],
    num_rollouts: int,
) -> None:
    all_rollouts = [rollout for result in prompt_results for rollout in result.get("rollouts", [])]
    n_prompts = len(prompt_results)
    n_rollouts = len(all_rollouts)

    print("\n" + "=" * 60)
    print(f"OVERALL ({n_prompts} prompts, {n_rollouts} rollouts)  [task={task}]", flush=True)
    print(f"  tags_correct:     {_mean([float(r['tags_correct']) for r in all_rollouts]):.2%}")
    print(f"  {payload_label+':':<28} {_mean([float(r['payload_present']) for r in all_rollouts]):.2%}")
    print(f"  exact_match:      {_mean([float(r['exact_match']) for r in all_rollouts]):.2%}")
    print(f"  {f1_label+':':<28} {_mean([float(r['primary_f1']) for r in all_rollouts]):.4f}")
    print(f"  prompt_copy:      {_mean([float(r['prompt_copy']) for r in all_rollouts]):.2%}")

    if reward_names:
        totals = [float(r.get("reward_total", 0.0)) for r in all_rollouts]
        print(f"\n  reward_total (mean): {_mean(totals):.4f}")
        print(f"  reward_total (std):  {_std(totals):.4f}")
        for reward_name in reward_names:
            values = [float(r.get("rewards", {}).get(reward_name, 0.0)) for r in all_rollouts]
            print(f"    {reward_name+':':<45} mean={_mean(values):.4f}  std={_std(values):.4f}")

    readiness = [result["rollout_summary"] for result in prompt_results]
    print("\nGRPO readiness:")
    print(f"  prompts_all_rollouts_scoreable: {_mean([float(x['all_rollouts_scoreable']) for x in readiness]):.2%}")
    print(f"  prompts_any_rollout_scoreable:  {_mean([float(x['any_rollout_scoreable']) for x in readiness]):.2%}")
    print(f"  prompts_completion_collapsed:   {_mean([float(x['completion_collapse']) for x in readiness]):.2%}")
    print(f"  prompts_reward_nonzero_var:     {_mean([float(x.get('reward_total_nonzero_variance', 0)) for x in readiness]):.2%}")
    if reward_names:
        print(f"  prompts_all_zero_reward:        {_mean([float(x.get('all_zero_reward', 0)) for x in readiness]):.2%}")
        print(f"  prompts_any_positive_reward:    {_mean([float(x.get('any_positive_reward', 0)) for x in readiness]):.2%}")
        print(f"  reward_total_std_per_prompt:    {_mean([float(x.get('reward_total_std', 0.0)) for x in readiness]):.4f}")
        print(f"  unique_reward_totals_per_prompt:{_mean([float(x.get('reward_total_unique_count', 0)) for x in readiness]):.2f}")
    if num_rollouts > 1:
        print(f"  unique_completions_per_prompt:  {_mean([float(x['unique_completion_count']) for x in readiness]):.2f}")


def _print_group_breakdown(
    prompt_results: list[dict[str, Any]],
    *,
    key: str,
    label: str,
    f1_label: str,
    reward_names: list[str],
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in prompt_results:
        grouped.setdefault(str(result.get(key, "")), []).append(result)

    if len(grouped) <= 1:
        return

    print(f"\nPer-{label} breakdown:")
    for group_name in sorted(grouped):
        items = grouped[group_name]
        all_rollouts = [rollout for result in items for rollout in result.get("rollouts", [])]
        readiness = [result["rollout_summary"] for result in items]
        reward_part = ""
        variance_part = ""
        if reward_names:
            reward_part = f"  reward={_mean([float(r.get('reward_total', 0.0)) for r in all_rollouts]):.3f}"
            variance_part = f"  reward_std/prompt={_mean([float(x.get('reward_total_std', 0.0)) for x in readiness]):.3f}"
        print(
            f"  {group_name:20s}  prompts={len(items):4d}  "
            f"tags={_mean([float(r['tags_correct']) for r in all_rollouts]):.2%}  "
            f"exact={_mean([float(r['exact_match']) for r in all_rollouts]):.2%}  "
            f"{f1_label}={_mean([float(r['primary_f1']) for r in all_rollouts]):.3f}"
            f"{reward_part}{variance_part}"
        )


# ---------------------------------------------------------------------------
# GRPO reward functions
# ---------------------------------------------------------------------------

def _build_reward_funcs(args, tokenizer, task: str) -> list[tuple[str, object]]:
    reward_funcs: list[tuple[str, object]] = []
    if task == "cd_descendants":
        if float(args.cd_format_reward_scale) > 0.0:
            reward_funcs.append(
                ("cd_format_reward", build_cd_format_reward(scale=float(args.cd_format_reward_scale)))
            )
        if float(args.cd_partial_format_reward_scale) > 0.0:
            reward_funcs.append((
                "cd_descendant_partial_format_reward",
                build_cd_descendant_partial_format_reward(scale=float(args.cd_partial_format_reward_scale)),
            ))
        if float(args.cd_descendant_cot_structure_reward_scale) > 0.0:
            reward_funcs.append((
                "cd_descendant_cot_structure_reward",
                build_cd_descendant_cot_structure_reward(scale=float(args.cd_descendant_cot_structure_reward_scale)),
            ))
        if float(args.cd_descendant_shift_ranking_reward_scale) > 0.0:
            reward_funcs.append((
                "cd_descendant_shift_ranking_reward",
                build_cd_descendant_shift_ranking_reward(scale=float(args.cd_descendant_shift_ranking_reward_scale)),
            ))
        if float(args.cd_descendant_variable_classification_reward_scale) > 0.0:
            reward_funcs.append((
                "cd_descendant_variable_classification_reward",
                build_cd_descendant_variable_classification_reward(
                    scale=float(args.cd_descendant_variable_classification_reward_scale)
                ),
            ))
        reward_funcs.append(
            ("cd_descendant_f1_reward", build_cd_descendant_f1_reward(scale=float(args.cd_graph_reward_scale)))
        )
    else:
        reward_funcs.extend([
            ("cd_format_reward", build_cd_format_reward(scale=float(args.cd_format_reward_scale))),
            ("cd_cot_structure_reward", build_cd_cot_structure_reward(scale=float(args.cd_cot_structure_reward_scale))),
            ("cd_skeleton_f1_reward", build_cd_skeleton_f1_reward(scale=float(args.cd_skeleton_f1_reward_scale))),
            ("cd_vstruct_f1_reward", build_cd_vstruct_f1_reward(scale=float(args.cd_vstruct_f1_reward_scale))),
            ("cd_orientation_f1_reward", build_cd_orientation_f1_reward(scale=float(args.cd_orientation_f1_reward_scale))),
            ("cd_edge_f1_reward", build_cd_edge_f1_reward(scale=float(args.cd_edge_f1_reward_scale))),
            ("cd_low_shd_reward", build_cd_low_shd_reward(scale=float(args.cd_low_shd_reward_scale))),
            ("cd_graph_reward", build_cd_graph_reward(
                require_dag=not bool(args.no_cd_reward_require_dag),
                dag_penalty=float(args.cd_reward_dag_penalty),
                shd_weight=float(args.cd_reward_shd_weight),
                scale=float(args.cd_graph_reward_scale),
            )),
        ])
    if float(args.length_penalty_coef) > 0.0:
        reward_funcs.append((
            "length_penalty_reward",
            build_length_penalty_reward(
                tokenizer,
                coef=float(args.length_penalty_coef),
                target_tokens=int(args.length_penalty_target_tokens),
                max_abs=float(args.length_penalty_max_abs),
            ),
        ))
    return reward_funcs


# ---------------------------------------------------------------------------
# GRPO multi-GPU compatibility check
# ---------------------------------------------------------------------------

def check_grpo_compat(model_path: str, base_model_override: Optional[str] = None) -> bool:
    """
    Mirror the exact load path that grpo.py uses for multi-GPU torchrun.
    Returns True if all checks pass, False otherwise.
    """
    import inspect

    issues: list[str] = []
    passes: list[str] = []

    print("\n" + "="*60)
    print(f"GRPO MULTI-GPU COMPATIBILITY CHECK: {model_path}")
    print("="*60)

    adapter_cfg_path = Path(model_path) / "adapter_config.json"
    is_adapter = adapter_cfg_path.exists()
    if is_adapter:
        passes.append("adapter_config.json found")
        adapter_cfg = json.loads(adapter_cfg_path.read_text())
    else:
        passes.append("no adapter_config.json — will apply fresh LoRA (r=8, q_proj+v_proj)")
        adapter_cfg = {}

    base_model_id = base_model_override or adapter_cfg.get("base_model_name_or_path", model_path)
    base_is_local = Path(base_model_id).exists()
    base_is_hf = not Path(base_model_id).is_absolute() and "/" in base_model_id
    if base_is_local or base_is_hf:
        passes.append(f"base model reachable: {base_model_id}")
    else:
        issues.append(f"base model path not found locally and not a HF id: {base_model_id}")

    if is_adapter:
        has_weights = (Path(model_path) / "adapter_model.safetensors").exists() or \
                      (Path(model_path) / "adapter_model.bin").exists()
        if has_weights:
            passes.append("adapter weight file found (adapter_model.safetensors or .bin)")
        else:
            issues.append("no adapter_model.safetensors or adapter_model.bin in checkpoint dir")

    if is_adapter:
        target_modules = adapter_cfg.get("target_modules", [])
        if target_modules:
            passes.append(f"target_modules: {target_modules}")
        else:
            issues.append("target_modules is empty in adapter_config.json")

    try:
        import flash_attn  # type: ignore  # noqa: F401
        passes.append(f"flash_attn installed: {flash_attn.__version__}")
    except ImportError:
        issues.append(
            "flash_attn not installed — grpo.py uses attn_implementation='flash_attention_2'; "
            "install with: pip install flash-attn --no-build-isolation"
        )

    try:
        from peft import AutoPeftModelForCausalLM
        params = inspect.signature(AutoPeftModelForCausalLM.from_pretrained).parameters
        if "is_trainable" in params:
            passes.append("PEFT supports is_trainable=True in from_pretrained")
        else:
            issues.append(
                "PEFT version does not support is_trainable= in AutoPeftModelForCausalLM.from_pretrained"
            )
    except ImportError:
        issues.append("peft not installed")

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            passes.append("bfloat16 supported on this GPU")
        else:
            issues.append("bfloat16 NOT supported on this GPU")
    else:
        issues.append("no CUDA device visible — cannot check bfloat16 / flash attention")

    print("\n-- Attempting distributed-style load (no device_map) --")
    load_ok = False
    try:
        if is_adapter:
            from peft import AutoPeftModelForCausalLM, PeftModel
            from transformers import AutoModelForCausalLM as _AMCLM

            model_load_kwargs: dict = {
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else "auto",
            }
            try:
                import flash_attn  # noqa: F401
                model_load_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                print("  [skip] flash_attn not installed — skipping attn_implementation check")

            if base_model_override:
                base = _AMCLM.from_pretrained(base_model_override, **model_load_kwargs)
                model = PeftModel.from_pretrained(base, model_path, is_trainable=True)
            else:
                adapter_load_kwargs = dict(model_load_kwargs)
                params = inspect.signature(AutoPeftModelForCausalLM.from_pretrained).parameters
                if "is_trainable" in params:
                    adapter_load_kwargs["is_trainable"] = True
                model = AutoPeftModelForCausalLM.from_pretrained(model_path, **adapter_load_kwargs)

            if hasattr(model, "set_adapter"):
                try:
                    model.set_adapter("default")
                    passes.append("set_adapter('default') succeeded")
                except Exception as e:
                    issues.append(f"set_adapter('default') failed: {e}")

            if hasattr(model, "enable_input_require_grads"):
                try:
                    model.enable_input_require_grads()
                    passes.append("enable_input_require_grads() succeeded")
                except Exception as e:
                    issues.append(f"enable_input_require_grads() failed: {e}")
            else:
                issues.append("model has no enable_input_require_grads() method")

            load_ok = True
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            passes.append("base model (no adapter) — grpo.py applies fresh LoRA, always compatible")
            load_ok = True
    except Exception as e:
        issues.append(f"distributed-style load FAILED: {type(e).__name__}: {e}")

    print()
    for p in passes:
        print(f"  [PASS] {p}")
    for issue in issues:
        print(f"  [FAIL] {issue}")

    overall = load_ok and not issues
    print()
    if overall:
        print("RESULT: COMPATIBLE with multi-GPU GRPO training")
    else:
        print("RESULT: ISSUES FOUND — review [FAIL] items above before launching GRPO")
    print("="*60 + "\n")
    return overall


# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------

class _StopOnSuffix(StoppingCriteria):
    """Stop generation as soon as the last N token IDs match stop_ids."""
    def __init__(self, stop_ids: list[int]):
        self.stop_ids = list(stop_ids)

    def __call__(self, input_ids, scores, **kwargs):
        if not self.stop_ids:
            return False
        if input_ids.shape[1] < len(self.stop_ids):
            return False
        tail = input_ids[0, -len(self.stop_ids):].tolist()
        return tail == self.stop_ids


# ---------------------------------------------------------------------------
# Model loading / generation
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(
    model_path: str,
    dtype_str: str,
    device_map: str,
    base_model_override: Optional[str] = None,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel  # type: ignore

    dtype_map = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(dtype_str, "auto")

    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        cfg = json.loads(adapter_config.read_text())
        base_model_id, base_is_local = _normalize_model_ref(
            base_model_override or cfg["base_model_name_or_path"]
        )
        print(f"Loading base model: {base_model_id}", flush=True)
        tokenizer_source = _resolve_tokenizer_source(model_path, base_model_id)
        tokenizer_source = _prepare_tokenizer_source(tokenizer_source, base_model_id)
        print(f"Loading tokenizer from: {tokenizer_source}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": device_map,
            "torch_dtype": dtype,
        }
        if base_is_local:
            model_kwargs["local_files_only"] = True
        base = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
        print(f"Loading LoRA adapter from: {model_path}", flush=True)
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model_id, model_is_local = _normalize_model_ref(model_path)
        print(f"Loading model: {model_id}", flush=True)
        tokenizer_source = _prepare_tokenizer_source(model_id, model_id)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
            "torch_dtype": dtype,
        }
        if model_is_local:
            model_kwargs["local_files_only"] = True
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    return model, tokenizer


def _generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_ids: Optional[list[int]] = None,
) -> tuple[str, str]:
    """Returns (completion, prompt_model_input).

    If stop_ids is provided, generation halts as soon as that token suffix is
    emitted (used by --compute-rewards to stop at </answer>).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    prompt_model_input = tokenizer.decode(inputs["input_ids"][0])

    gen_kwargs: dict = dict(
        max_new_tokens=max_new_tokens,
        max_length=None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if stop_ids:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_StopOnSuffix(stop_ids)])
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_ids = output_ids[0][input_len:]
    completion = tokenizer.decode(new_ids, skip_special_tokens=True)
    return completion, prompt_model_input


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate a SFT/LoRA model on a JSONL/CSV eval set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", required=True, help="Path to LoRA adapter dir or HF model id.")
    ap.add_argument(
        "--jsonl", type=Path,
        default=Path(__file__).parent / "data" / "permuted_sft.jsonl",
        help="JSONL or CSV eval set. Defaults to experiments/data/permuted_sft.jsonl.",
    )
    ap.add_argument("--n", type=int, default=20, help="Random subset of examples to evaluate.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling for generation. Only used when temperature > 0.",
    )
    ap.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="Number of completions to sample per prompt. Use >1 to measure GRPO reward spread.",
    )
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument(
        "--output-jsonl", type=Path, default=None,
        help="Path to write per-example results. Defaults to {model_dir}/eval_permuted_{n}.jsonl.",
    )
    ap.add_argument(
        "--graph-filter", nargs="*",
        help="Only evaluate examples from these graphs (e.g. cancer asia).",
    )
    ap.add_argument(
        "--task",
        choices=["auto", "causal_discovery", "cd_descendants"],
        default="auto",
        help="Override task detection. 'auto' infers from the answer payload.",
    )
    ap.add_argument(
        "--check-grpo-compat",
        action="store_true",
        help="Run GRPO multi-GPU compatibility checks (no generation).",
    )
    ap.add_argument(
        "--base-model-override",
        default=None,
        help="Full-precision base model path when adapter was trained on a quantized base.",
    )

    # --- GRPO reward evaluation ---
    ap.add_argument(
        "--no-compute-rewards",
        action="store_true",
        help=(
            "Skip GRPO reward computation. By default rewards are computed and "
            "generation stops at </answer>."
        ),
    )
    ap.add_argument("--cd-format-reward-scale", type=float, default=0.05)
    ap.add_argument("--cd-partial-format-reward-scale", type=float, default=0.0)
    ap.add_argument("--cd-cot-structure-reward-scale", type=float, default=0.05)
    ap.add_argument("--cd-descendant-cot-structure-reward-scale", type=float, default=0.0)
    ap.add_argument("--cd-descendant-shift-ranking-reward-scale", type=float, default=0.0)
    ap.add_argument("--cd-descendant-variable-classification-reward-scale", type=float, default=0.0)
    ap.add_argument("--cd-skeleton-f1-reward-scale", type=float, default=0.10)
    ap.add_argument("--cd-vstruct-f1-reward-scale", type=float, default=0.10)
    ap.add_argument("--cd-orientation-f1-reward-scale", type=float, default=0.10)
    ap.add_argument("--cd-edge-f1-reward-scale", type=float, default=0.30)
    ap.add_argument("--cd-low-shd-reward-scale", type=float, default=0.20)
    ap.add_argument("--cd-graph-reward-scale", type=float, default=0.50)
    ap.add_argument("--cd-reward-shd-weight", type=float, default=0.0)
    ap.add_argument("--cd-reward-dag-penalty", type=float, default=0.1)
    ap.add_argument("--no-cd-reward-require-dag", action="store_true")
    ap.add_argument("--length-penalty-coef", type=float, default=0.0)
    ap.add_argument("--length-penalty-target-tokens", type=int, default=0)
    ap.add_argument("--length-penalty-max-abs", type=float, default=1.0)

    args = ap.parse_args()

    if args.output_jsonl is None and not args.check_grpo_compat:
        model_dir = Path(args.model)
        if not model_dir.exists():
            safe_name = Path(args.model).name.replace("/", "_")
            model_dir = Path(__file__).parent / "checkpoints" / safe_name
        n_tag = str(args.n) if args.n else "all"
        suffix = "cd_rewards" if not args.no_compute_rewards else "permuted"
        args.output_jsonl = model_dir / f"eval_{suffix}_{n_tag}.jsonl"

    rng = random.Random(args.seed)

    if args.check_grpo_compat:
        ok = check_grpo_compat(args.model, base_model_override=args.base_model_override)
        sys.exit(0 if ok else 1)

    if args.num_rollouts < 1:
        raise ValueError("--num-rollouts must be >= 1")

    print(f"Loading eval set: {args.jsonl}", flush=True)
    records = _load_records(args.jsonl, n=None, rng=rng)

    if args.graph_filter:
        keep = set(args.graph_filter)
        records = [r for r in records if r.get("graph") in keep]
        print(f"After graph filter {args.graph_filter}: {len(records)} examples", flush=True)

    if args.n and args.n < len(records):
        records = rng.sample(records, args.n)

    # Task detection
    if args.task == "auto":
        task = _detect_task(records)
    else:
        task = "cd" if args.task == "causal_discovery" else args.task

    payload_label = "descendant_payload_present" if task == "cd_descendants" else "adj_matrix_present"
    f1_label = "descendant_f1" if task == "cd_descendants" else "edge_f1"
    print(f"Task detected: {task}", flush=True)
    print(
        f"Evaluating {len(records)} prompts  num_rollouts={args.num_rollouts}  "
        f"temperature={args.temperature}  top_p={args.top_p}  "
        f"compute_rewards={not args.no_compute_rewards}",
        flush=True,
    )
    if args.num_rollouts > 1 and args.temperature <= 0.0:
        print(
            "[warn] num_rollouts > 1 with temperature <= 0 will usually collapse to identical completions; "
            "use temperature > 0 to measure GRPO reward spread.",
            flush=True,
        )

    model, tokenizer = _load_model_and_tokenizer(
        args.model,
        args.dtype,
        args.device_map,
        base_model_override=args.base_model_override,
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Stop-on-answer token IDs (only used when --compute-rewards)
    stop_ids: Optional[list[int]] = None
    reward_funcs: list[tuple[str, object]] = []
    reward_names: list[str] = []
    if not args.no_compute_rewards:
        stop_ids = tokenizer.encode("</answer>", add_special_tokens=False)
        reward_funcs = _build_reward_funcs(args, tokenizer, task)
        reward_names = [name for name, _ in reward_funcs]

    # Truncate output file if computing rewards (we stream-append)
    if not args.no_compute_rewards and args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if args.output_jsonl.exists():
            args.output_jsonl.unlink()

    results: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        prompt = str(rec["prompt"])
        ground_truth = str(rec["answer"])
        reward_kwargs = _build_reward_kwargs(prompt, ground_truth, task) if reward_funcs else {}

        prompt_model_input: Optional[str] = None
        rollouts: list[dict[str, Any]] = []
        for rollout_idx in range(args.num_rollouts):
            t0 = time.time()
            completion, prompt_model_input_cur = _generate_completion(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stop_ids=stop_ids,
            )
            gen_seconds = round(time.time() - t0, 3)
            if prompt_model_input is None:
                prompt_model_input = prompt_model_input_cur

            scores = _score(completion, ground_truth, task=task)
            rollout_result: dict[str, Any] = {
                "rollout_idx": rollout_idx,
                **scores,
                "completion_snippet": completion[:300],
                "completion": completion,
                "gen_seconds": gen_seconds,
            }
            if reward_funcs:
                rewards = {
                    name: float(fn([completion], **reward_kwargs)[0])
                    for name, fn in reward_funcs
                }
                rollout_result["rewards"] = rewards
                rollout_result["reward_total"] = float(sum(rewards.values()))
            rollouts.append(rollout_result)

        first_rollout = rollouts[0]
        rollout_summary = _summarize_rollouts(rollouts, reward_names=reward_names)
        result = {
            "idx": i,
            "graph": rec.get("graph", ""),
            "source": rec.get("source", ""),
            "num_rollouts": len(rollouts),
            "prompt_model_input": prompt_model_input or "",
            "target_answer": ground_truth,
            "rollout_summary": rollout_summary,
            "rollouts": rollouts,
            # Backward-compatible top-level fields mirror the first rollout.
            "tags_correct": first_rollout["tags_correct"],
            "payload_present": first_rollout["payload_present"],
            "exact_match": first_rollout["exact_match"],
            "primary_f1": first_rollout["primary_f1"],
            "prompt_copy": first_rollout["prompt_copy"],
            "answer_text": first_rollout["answer_text"],
            "completion_snippet": first_rollout["completion_snippet"],
            "gen_seconds": first_rollout["gen_seconds"],
        }
        if reward_funcs:
            result["completion"] = first_rollout["completion"]
            result["rewards"] = first_rollout["rewards"]
            result["reward_total"] = first_rollout["reward_total"]

        results.append(result)

        # Stream-write when computing rewards so partial results are not lost
        if not args.no_compute_rewards and args.output_jsonl:
            with args.output_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Progress logging
        if (i + 1) % 10 == 0 or i == 0:
            so_far_rollouts = [rollout for item in results for rollout in item["rollouts"]]
            readiness_so_far = [item["rollout_summary"] for item in results]
            n_so_far = len(so_far_rollouts)
            reward_str = (
                f"  reward_total={_mean([float(r.get('reward_total', 0.0)) for r in so_far_rollouts]):.3f}"
                if reward_funcs else ""
            )
            variance_str = (
                f"  reward_std/prompt={_mean([float(x.get('reward_total_std', 0.0)) for x in readiness_so_far]):.3f}"
                if reward_funcs and args.num_rollouts > 1 else ""
            )
            print(
                f"[{i+1:4d}/{len(records)} prompts] "
                f"tags_correct={_mean([float(r['tags_correct']) for r in so_far_rollouts]):.2%}  "
                f"{payload_label}={_mean([float(r['payload_present']) for r in so_far_rollouts]):.2%}  "
                f"exact_match={_mean([float(r['exact_match']) for r in so_far_rollouts]):.2%}  "
                f"{f1_label}={_mean([float(r['primary_f1']) for r in so_far_rollouts]):.3f}  "
                f"prompt_copy={_mean([float(r['prompt_copy']) for r in so_far_rollouts]):.2%}"
                f"{reward_str}",
                flush=True,
            )
            if variance_str:
                print(
                    f"             prompts_all_scoreable={_mean([float(x['all_rollouts_scoreable']) for x in readiness_so_far]):.2%}"
                    f"{variance_str}",
                    flush=True,
                )
        for rollout in rollouts:
            if rollout["tags_correct"] and not rollout["payload_present"]:
                print(
                    f"  [parse_fail idx={i} rollout={rollout['rollout_idx']}] <answer>: "
                    f"{repr(rollout['answer_text'])}",
                    flush=True,
                )

    _print_overall_summary(
        results,
        task=task,
        payload_label=payload_label,
        f1_label=f1_label,
        reward_names=reward_names,
        num_rollouts=args.num_rollouts,
    )

    _print_group_breakdown(
        results,
        key="graph",
        label="graph",
        f1_label=f1_label,
        reward_names=reward_names,
    )
    _print_group_breakdown(
        results,
        key="source",
        label="source",
        f1_label=f1_label,
        reward_names=reward_names,
    )

    # Write output JSONL (non-reward mode: write all at once; reward mode: already streamed)
    if args.output_jsonl and args.no_compute_rewards:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(results)} prompt records to {args.output_jsonl}")
    elif args.output_jsonl and not args.no_compute_rewards:
        print(f"\nWrote {len(results)} prompt records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
