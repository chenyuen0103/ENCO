#!/usr/bin/env python3
from __future__ import annotations

"""Sample prompt rows, generate vLLM rollouts, and score them with GRPO rewards."""

import argparse
import csv
import difflib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
os.environ.setdefault("ENCO_PATCH_VLLM_TQDM", "1")
os.environ.setdefault("ENCO_PATCH_VLLM_TOKENIZER", "1")
_repo_root = _HERE.parent
_py_path_parts = [
    str(_HERE),
    str(_repo_root),
]
_existing_pythonpath = str(os.environ.get("PYTHONPATH") or "")
if _existing_pythonpath:
    _py_path_parts.append(_existing_pythonpath)
os.environ["PYTHONPATH"] = os.pathsep.join(_py_path_parts)

from verifier_cd import (
    build_cd_acyclic_reward,
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
    build_cd_partial_format_reward,
    build_cd_skeleton_f1_reward,
    build_cd_stage_targets,
    build_cd_vstruct_f1_reward,
    build_length_penalty_reward,
    extract_descendant_payload,
    score_cd_completion,
    score_cd_descendants_completion,
)
DEFAULT_CSVS = [
    str(_HERE / "data" / "grpo_mix_anon.csv"),
    str(_HERE / "data" / "grpo_mix_named.csv"),
]


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    return slug.strip("._-") or "model"


def _is_local_model_arg(value: str) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    if raw.startswith((".", "/", "~")):
        return True
    return any(sep in raw for sep in (os.sep, "/"))


def _iter_checkpoint_dirs() -> list[str]:
    base = _HERE / "checkpoints"
    if not base.is_dir():
        return []
    return sorted(
        str(path.relative_to(_repo_root))
        for path in base.iterdir()
        if path.is_dir()
    )


def _resolve_model_arg(model_arg: str) -> str:
    raw = str(model_arg or "").strip()
    if not raw:
        raise ValueError("--model must not be empty")
    if not _is_local_model_arg(raw):
        return raw

    expanded = Path(raw).expanduser()
    candidates = [expanded]
    if not expanded.is_absolute():
        candidates.append((Path.cwd() / expanded).resolve())
        candidates.append((_repo_root / expanded).resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return str(resolved)

    checkpoint_dirs = _iter_checkpoint_dirs()
    suggestion_pool = checkpoint_dirs + [Path(item).name for item in checkpoint_dirs]
    suggestions = difflib.get_close_matches(raw, suggestion_pool, n=3, cutoff=0.45)
    if not suggestions:
        suggestions = difflib.get_close_matches(Path(raw).name, suggestion_pool, n=3, cutoff=0.45)

    message = (
        f"Local model path not found: {raw}. "
        f"Checked: {', '.join(str(path) for path in seen)}."
    )
    if suggestions:
        message += " Did you mean: " + ", ".join(sorted(dict.fromkeys(suggestions))) + "?"
    raise FileNotFoundError(message)


def _is_peft_adapter_dir(path: Path) -> bool:
    path = Path(path)
    return path.is_dir() and (path / "adapter_config.json").exists()


def _is_full_model_dir(path: Path) -> bool:
    path = Path(path)
    if not path.is_dir() or not (path / "config.json").exists():
        return False
    model_markers = [
        path / "model.safetensors",
        path / "model.safetensors.index.json",
        path / "pytorch_model.bin",
    ]
    if any(marker.exists() for marker in model_markers):
        return True
    return any(path.glob("model-*.safetensors"))


def _resolve_runtime_model_arg(model_arg: str) -> str:
    resolved = _resolve_model_arg(model_arg)
    candidate = Path(resolved)
    if not _is_peft_adapter_dir(candidate):
        return resolved

    merged_dir = candidate.parent / f"{candidate.name}_merged"
    if _is_full_model_dir(merged_dir):
        print(f"[info] using existing merged model for adapter checkpoint: {merged_dir}", flush=True)
        return str(merged_dir)

    print(f"[info] merging PEFT adapter checkpoint for vLLM: {candidate} -> {merged_dir}", flush=True)
    from run_sft import merge_sft_adapter

    merged_path = merge_sft_adapter(
        sft_model_dir=candidate,
        merged_output_dir=merged_dir,
    )
    return str(merged_path)


def _looks_like_inline_text(raw: str) -> bool:
    text = str(raw or "")
    if not text:
        return False
    if len(text) > 240:
        return True
    if any(ch in text for ch in ("\n", "\r", "\t", "{", "}", "[", "]")):
        return True
    if text.startswith(("system\n", "user\n", "assistant\n", "<think>", "<answer>")):
        return True
    return False


def _read_text_maybe_path(value: str, *, csv_path: Path) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if _looks_like_inline_text(raw):
        return raw
    try:
        candidate = Path(raw)
        if candidate.exists() and candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="ignore")
        if not candidate.is_absolute():
            candidate = (csv_path.parent / candidate).resolve()
            if candidate.exists() and candidate.is_file():
                return candidate.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return raw
    return raw


def _load_records_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row_idx, row in enumerate(csv.DictReader(handle)):
            prompt = _read_text_maybe_path(
                row.get("prompt_text") or row.get("prompt") or row.get("prompt_path") or "",
                csv_path=csv_path,
            )
            answer = _read_text_maybe_path(
                row.get("answer") or row.get("answer_path") or "",
                csv_path=csv_path,
            )
            if not prompt or not answer:
                continue
            record = dict(row)
            record["prompt"] = prompt
            record["answer"] = answer
            record["source_csv"] = str(csv_path)
            record["source_name"] = csv_path.name
            record["row_index"] = row_idx
            records.append(record)
    return records


def _sample_records(
    csv_paths: list[str],
    *,
    samples_per_csv: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for csv_path_str in csv_paths:
        csv_path = Path(csv_path_str)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        records = _load_records_from_csv(csv_path)
        if not records:
            raise ValueError(f"No usable prompt/answer rows found in {csv_path}")
        if samples_per_csv > 0 and samples_per_csv < len(records):
            picked = rng.sample(records, samples_per_csv)
        else:
            picked = records
        picked = sorted(picked, key=lambda row: int(row["row_index"]))
        for picked_idx, record in enumerate(picked):
            record = dict(record)
            record["sample_in_source"] = picked_idx
            sampled.append(record)
    return sampled


def _detect_task(records: list[dict[str, Any]], override: str) -> str:
    if override != "auto":
        return override
    saw_descendants = False
    saw_cd = False
    for record in records[:20]:
        answer = str(record.get("answer") or "")
        if extract_descendant_payload(answer) is not None:
            saw_descendants = True
        else:
            saw_cd = True
    if saw_descendants and saw_cd:
        raise ValueError("Mixed causal_discovery and cd_descendants rows are not supported in one run.")
    return "cd_descendants" if saw_descendants else "causal_discovery"


def _build_reward_funcs(args, tokenizer, task: str) -> list[tuple[str, Any]]:
    reward_funcs: list[tuple[str, Any]] = []
    if task == "cd_descendants":
        if float(args.cd_format_reward_scale) > 0.0:
            reward_funcs.append(
                ("cd_format_reward", build_cd_format_reward(scale=float(args.cd_format_reward_scale)))
            )
        if float(args.cd_partial_format_reward_scale) > 0.0:
            reward_funcs.append(
                (
                    "cd_descendant_partial_format_reward",
                    build_cd_descendant_partial_format_reward(scale=float(args.cd_partial_format_reward_scale)),
                )
            )
        if float(args.cd_descendant_cot_structure_reward_scale) > 0.0:
            reward_funcs.append(
                (
                    "cd_descendant_cot_structure_reward",
                    build_cd_descendant_cot_structure_reward(
                        scale=float(args.cd_descendant_cot_structure_reward_scale)
                    ),
                )
            )
        if float(args.cd_descendant_shift_ranking_reward_scale) > 0.0:
            reward_funcs.append(
                (
                    "cd_descendant_shift_ranking_reward",
                    build_cd_descendant_shift_ranking_reward(
                        scale=float(args.cd_descendant_shift_ranking_reward_scale)
                    ),
                )
            )
        if float(args.cd_descendant_variable_classification_reward_scale) > 0.0:
            reward_funcs.append(
                (
                    "cd_descendant_variable_classification_reward",
                    build_cd_descendant_variable_classification_reward(
                        scale=float(args.cd_descendant_variable_classification_reward_scale)
                    ),
                )
            )
        reward_funcs.append(
            ("cd_descendant_f1_reward", build_cd_descendant_f1_reward(scale=float(args.cd_graph_reward_scale)))
        )
    else:
        if float(args.cd_format_reward_scale) > 0.0:
            reward_funcs.append(
                ("cd_format_reward", build_cd_format_reward(scale=float(args.cd_format_reward_scale)))
            )
        if float(args.cd_partial_format_reward_scale) > 0.0:
            reward_funcs.append(
                (
                    "cd_partial_format_reward",
                    build_cd_partial_format_reward(scale=float(args.cd_partial_format_reward_scale)),
                )
            )
        if float(args.cd_edge_f1_reward_scale) > 0.0:
            reward_funcs.append(
                ("cd_edge_f1_reward", build_cd_edge_f1_reward(scale=float(args.cd_edge_f1_reward_scale)))
            )
        if float(args.cd_low_shd_reward_scale) > 0.0:
            reward_funcs.append(
                ("cd_low_shd_reward", build_cd_low_shd_reward(scale=float(args.cd_low_shd_reward_scale)))
            )
        if float(args.cd_acyclic_reward_scale) > 0.0:
            reward_funcs.append(
                ("cd_acyclic_reward", build_cd_acyclic_reward(scale=float(args.cd_acyclic_reward_scale)))
            )
        if float(args.cd_cot_structure_reward_scale) > 0.0:
            reward_funcs.append(
                (
                    "cd_cot_structure_reward",
                    build_cd_cot_structure_reward(scale=float(args.cd_cot_structure_reward_scale)),
                )
            )
        if float(args.cd_skeleton_f1_reward_scale) > 0.0:
            reward_funcs.append(
                (
                    "cd_skeleton_f1_reward",
                    build_cd_skeleton_f1_reward(scale=float(args.cd_skeleton_f1_reward_scale)),
                )
            )
        if float(args.cd_vstruct_f1_reward_scale) > 0.0:
            reward_funcs.append(
                (
                    "cd_vstruct_f1_reward",
                    build_cd_vstruct_f1_reward(scale=float(args.cd_vstruct_f1_reward_scale)),
                )
            )
        if float(args.cd_orientation_f1_reward_scale) > 0.0:
            reward_funcs.append(
                (
                    "cd_orientation_f1_reward",
                    build_cd_orientation_f1_reward(scale=float(args.cd_orientation_f1_reward_scale)),
                )
            )
        reward_funcs.append(
            (
                "cd_graph_reward",
                build_cd_graph_reward(
                    require_dag=bool(args.cd_reward_require_dag),
                    dag_penalty=float(args.cd_reward_dag_penalty),
                    shd_weight=float(args.cd_reward_shd_weight),
                    scale=float(args.cd_graph_reward_scale),
                ),
            )
        )

    if float(args.length_penalty_coef) > 0.0:
        reward_funcs.append(
            (
                "length_penalty_reward",
                build_length_penalty_reward(
                    tokenizer,
                    coef=float(args.length_penalty_coef),
                    target_tokens=int(args.length_penalty_target_tokens),
                    max_abs=float(args.length_penalty_max_abs),
                ),
            )
        )
    return reward_funcs


def _make_generation_prompt(record: dict[str, Any], args, task: str) -> str:
    raw_prompt = str(record["prompt"])
    if task not in {"causal_discovery", "cd_descendants"}:
        return raw_prompt
    return raw_prompt


def _score_completion(task: str, completion: str, answer: str, args) -> dict[str, Any]:
    if task == "cd_descendants":
        return score_cd_descendants_completion(completion, answer)
    return score_cd_completion(
        completion,
        answer,
        require_dag=bool(args.cd_reward_require_dag),
        dag_penalty=float(args.cd_reward_dag_penalty),
        shd_weight=float(args.cd_reward_shd_weight),
    )


def _build_reward_inputs(
    task: str,
    sampled_records: list[dict[str, Any]],
    rollout_outputs: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    prompts: list[str] = []
    answers: list[str] = []
    stage1_targets: list[list[Any]] = []
    stage2_targets: list[list[Any]] = []
    stage3_targets: list[list[Any]] = []

    stage_cache: dict[int, dict[str, Any]] = {}
    for sample_idx, record in enumerate(sampled_records):
        if task == "causal_discovery":
            stage_cache[sample_idx] = build_cd_stage_targets(record["prompt"], record["answer"]) or {}
        else:
            stage_cache[sample_idx] = {}

    for output in rollout_outputs:
        sample_idx = int(output["sample_idx"])
        record = sampled_records[sample_idx]
        prompts.append(str(record["prompt"]))
        answers.append(str(record["answer"]))
        stage_targets = stage_cache[sample_idx]
        if task == "causal_discovery":
            stage1_targets.append(stage_targets.get("target_stage1_skeleton_edges", []))
            stage2_targets.append(stage_targets.get("target_stage2_vstructures", []))
            stage3_targets.append(stage_targets.get("target_stage3_directed_edges", []))

    reward_inputs: dict[str, list[Any]] = {
        "prompt": prompts,
        "answer": answers,
    }
    if task == "causal_discovery":
        reward_inputs["target_stage1_skeleton_edges"] = stage1_targets
        reward_inputs["target_stage2_vstructures"] = stage2_targets
        reward_inputs["target_stage3_directed_edges"] = stage3_targets
    return reward_inputs


def _summarize_results(
    rollout_outputs: list[dict[str, Any]],
    reward_names: list[str],
) -> dict[str, Any]:
    totals = [float(item["reward_total"]) for item in rollout_outputs]
    completions = [str(item.get("completion") or "") for item in rollout_outputs]
    summary: dict[str, Any] = {
        "num_rollouts": len(rollout_outputs),
        "reward_total_mean": statistics.fmean(totals) if totals else 0.0,
        "reward_total_min": min(totals) if totals else 0.0,
        "reward_total_max": max(totals) if totals else 0.0,
    }
    for reward_name in reward_names:
        vals = [float(item["rewards"][reward_name]) for item in rollout_outputs]
        summary[f"{reward_name}_mean"] = statistics.fmean(vals) if vals else 0.0
    if completions:
        format_progress = {
            "has_close_think": sum("</think>" in text for text in completions),
            "has_open_answer": sum("<answer>" in text for text in completions),
            "has_close_answer": sum("</answer>" in text for text in completions),
            "has_adjacency_matrix": sum("adjacency_matrix" in text for text in completions),
            "strict_format_ok": sum(bool(item.get("score_meta", {}).get("format_ok")) for item in rollout_outputs),
            "parse_ok": sum(bool(item.get("score_meta", {}).get("parse_ok")) for item in rollout_outputs),
        }
        summary["format_progress"] = {
            key: {
                "count": value,
                "rate": (float(value) / len(completions)),
            }
            for key, value in format_progress.items()
        }

    by_source: dict[str, list[dict[str, Any]]] = {}
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for item in rollout_outputs:
        by_source.setdefault(str(item["source_name"]), []).append(item)
        by_dataset.setdefault(str(item["dataset"]), []).append(item)

    summary["by_source"] = {
        key: {
            "num_rollouts": len(items),
            "reward_total_mean": statistics.fmean(float(item["reward_total"]) for item in items),
        }
        for key, items in sorted(by_source.items())
    }
    summary["by_dataset"] = {
        key: {
            "num_rollouts": len(items),
            "reward_total_mean": statistics.fmean(float(item["reward_total"]) for item in items),
        }
        for key, items in sorted(by_dataset.items())
    }
    return summary


def _build_grouped_rollout_records(
    sampled_records: list[dict[str, Any]],
    rollout_outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped_records: list[dict[str, Any]] = []
    rollouts_by_sample: dict[int, list[dict[str, Any]]] = {}
    for item in rollout_outputs:
        rollouts_by_sample.setdefault(int(item["sample_idx"]), []).append(item)

    meta_keys = [
        "source_csv",
        "source_name",
        "row_index",
        "dataset",
        "bif_file",
        "prompt_style",
        "anonymize",
        "obs_per_prompt",
        "int_per_combo",
        "data_idx",
        "shuffle_idx",
    ]

    for sample_idx, record in enumerate(sampled_records):
        grouped_rollouts = sorted(
            rollouts_by_sample.get(sample_idx, []),
            key=lambda item: int(item["rollout_idx"]),
        )
        grouped_record: dict[str, Any] = {
            "sample_idx": sample_idx,
            "prompt": record["prompt"],
            "prompt_model_input": record.get("prompt_model_input", record["prompt"]),
            "answer": record["answer"],
            "num_completions": len(grouped_rollouts),
        }
        for key in meta_keys:
            grouped_record[key] = record.get(key, "")

        for rollout_idx, rollout in enumerate(grouped_rollouts):
            grouped_record[f"completion_{rollout_idx}"] = rollout["completion"]
            grouped_record[f"score_meta_{rollout_idx}"] = rollout["score_meta"]
            grouped_record[f"rewards_{rollout_idx}"] = rollout.get("rewards", {})
            grouped_record[f"reward_total_{rollout_idx}"] = rollout.get("reward_total", 0.0)

        grouped_records.append(grouped_record)

    return grouped_records


def _get_tokenizer_declared_max_length(tokenizer) -> Optional[int]:
    raw_value = getattr(tokenizer, "model_max_length", None)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return None
    # Transformers uses very large sentinels when the limit is effectively unset.
    if value <= 0 or value >= 1_000_000:
        return None
    return value


def _install_vllm_tokenizer_compat_shims(tokenizer) -> list[str]:
    patched: list[str] = []
    tokenizer_cls = tokenizer.__class__

    if not hasattr(tokenizer_cls, "all_special_tokens_extended"):
        setattr(
            tokenizer_cls,
            "all_special_tokens_extended",
            property(lambda self: list(getattr(self, "all_special_tokens", []) or [])),
        )
        patched.append("all_special_tokens_extended")

    if not hasattr(tokenizer_cls, "special_tokens_map_extended"):
        setattr(
            tokenizer_cls,
            "special_tokens_map_extended",
            property(lambda self: dict(getattr(self, "special_tokens_map", {}) or {})),
        )
        patched.append("special_tokens_map_extended")

    if not hasattr(tokenizer_cls, "num_special_tokens_to_add"):
        def _num_special_tokens_to_add(self, pair: bool = False) -> int:
            build_inputs = getattr(self, "build_inputs_with_special_tokens", None)
            if callable(build_inputs):
                try:
                    built = build_inputs([], []) if pair else build_inputs([])
                    return max(0, len(built))
                except Exception:
                    pass
            return 0

        setattr(tokenizer_cls, "num_special_tokens_to_add", _num_special_tokens_to_add)
        patched.append("num_special_tokens_to_add")

    return patched


def _patch_tqdm_for_vllm() -> Optional[str]:
    """Patch vLLM's DisabledTqdm helper for newer huggingface_hub versions.

    vLLM 0.11.0 defines DisabledTqdm as:
        super().__init__(*args, **kwargs, disable=True)
    but newer huggingface_hub.snapshot_download already passes disable=..., which
    triggers "got multiple values for keyword argument 'disable'".
    """
    try:
        import vllm.model_executor.model_loader.weight_utils as weight_utils  # type: ignore
    except Exception:
        return None

    disabled_tqdm_cls = getattr(weight_utils, "DisabledTqdm", None)
    if disabled_tqdm_cls is None:
        return None
    if getattr(disabled_tqdm_cls, "_enco_disable_patch", False):
        return None

    base_tqdm_cls = disabled_tqdm_cls.__mro__[1] if len(disabled_tqdm_cls.__mro__) > 1 else None
    if base_tqdm_cls is None:
        return None

    class PatchedDisabledTqdm(base_tqdm_cls):  # type: ignore[misc, valid-type]
        _enco_disable_patch = True

        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    PatchedDisabledTqdm.__name__ = getattr(disabled_tqdm_cls, "__name__", "PatchedDisabledTqdm")
    PatchedDisabledTqdm.__qualname__ = getattr(disabled_tqdm_cls, "__qualname__", PatchedDisabledTqdm.__name__)
    weight_utils.DisabledTqdm = PatchedDisabledTqdm
    return (
        "patched vLLM DisabledTqdm for huggingface_hub compatibility "
        f"(base={base_tqdm_cls.__module__}.{base_tqdm_cls.__name__})"
    )


def _infer_vllm_max_model_len(
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> tuple[int, int, int, Optional[int], Optional[str]]:
    tokenized = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]
    max_prompt_tokens = max((len(ids) for ids in tokenized), default=0)
    buffer_tokens = max(256, min(2048, max(64, max_prompt_tokens // 8)))
    inferred = int(max_prompt_tokens + int(max_new_tokens) + buffer_tokens)
    declared_max_length = _get_tokenizer_declared_max_length(tokenizer)
    warning: Optional[str] = None
    if declared_max_length is not None and inferred > declared_max_length:
        remaining_completion_budget = max(0, declared_max_length - max_prompt_tokens)
        warning = (
            f"[warn] capping inferred vLLM max_model_len from {inferred} to tokenizer.model_max_length="
            f"{declared_max_length}. Longest sampled prompt uses {max_prompt_tokens} tokens, leaving at most "
            f"{remaining_completion_budget} tokens for generation before hitting the declared context limit."
        )
        inferred = declared_max_length
    return inferred, max_prompt_tokens, buffer_tokens, declared_max_length, warning


def _auto_adjust_vllm_gpu_mem_util(
    requested_util: float,
    tensor_parallel_size: int,
) -> tuple[float, Optional[str]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return float(requested_util), None

    rows: list[tuple[int, int, int]] = []
    for physical_idx, line in enumerate(completed.stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            free_mb = int(float(parts[0]))
            total_mb = int(float(parts[1]))
        except ValueError:
            continue
        if total_mb > 0:
            rows.append((physical_idx, free_mb, total_mb))
    if not rows:
        return float(requested_util), None

    visible_spec = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if visible_spec and visible_spec != "-1":
        remapped_rows: list[tuple[int, int, int]] = []
        requested_devices = [token.strip() for token in visible_spec.split(",") if token.strip()]
        for local_idx, token in enumerate(requested_devices):
            if not token.isdigit():
                remapped_rows = []
                break
            physical_idx = int(token)
            if 0 <= physical_idx < len(rows):
                _, free_mb, total_mb = rows[physical_idx]
                remapped_rows.append((local_idx, free_mb, total_mb))
        if remapped_rows:
            rows = remapped_rows

    world = min(int(tensor_parallel_size), len(rows))
    free_ratios: list[float] = []
    details: list[str] = []
    for device_idx, free_mb, total_mb in rows[:world]:
        free_ratio = float(free_mb) / float(total_mb)
        free_ratios.append(free_ratio)
        details.append(
            f"cuda:{device_idx} free={free_mb / 1024.0:.2f}GiB total={total_mb / 1024.0:.2f}GiB"
        )

    if not free_ratios:
        return float(requested_util), None

    min_free_ratio = min(free_ratios)
    safe_util = max(0.01, min(0.98, min_free_ratio * 0.92))
    if float(requested_util) <= safe_util:
        return float(requested_util), None

    message = (
        f"[warn] lowering --vllm-gpu-mem-util from {float(requested_util):.3f} to {safe_util:.3f} "
        f"based on currently free GPU memory across the first {world} visible device(s): "
        + "; ".join(details)
    )
    return float(safe_util), message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a few GRPO-style vLLM rollouts on prompt CSV rows and score each rollout."
    )
    parser.add_argument("--model", required=True, help="HF model id or local model path.")
    parser.add_argument(
        "--csv",
        action="append",
        default=None,
        help="Prompt CSV to sample from. Repeatable. Defaults to grpo_mix_anon.csv and grpo_mix_named.csv.",
    )
    parser.add_argument(
        "--samples-per-csv",
        type=int,
        default=3,
        help="Number of prompt rows to sample from each CSV.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--task",
        choices=["auto", "causal_discovery", "cd_descendants"],
        default="auto",
        help="Task override. Auto-detects from the answer payload.",
    )
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--stop-sequence",
        action="append",
        default=None,
        help="Stop string to preserve in the output. Repeatable. Defaults to </answer>.",
    )

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument("--vllm-gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.set_defaults(trust_remote_code=True)

    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Per-rollout JSONL output path. Defaults under experiments/out/grpo_vllm_rollouts/.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Summary JSON output path. Defaults next to --output-jsonl.",
    )
    parser.add_argument(
        "--include-prompts",
        action="store_true",
        help="Include the full prompt text in flat per-rollout output records.",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write one JSONL row per rollout instead of the default grouped-by-prompt format.",
    )

    # Reward defaults match grpo.py.
    parser.add_argument("--cd-reward-shd-weight", type=float, default=0.0)
    parser.add_argument("--cd-reward-dag-penalty", type=float, default=0.1)
    parser.add_argument("--cd-reward-require-dag", dest="cd_reward_require_dag", action="store_true")
    parser.add_argument("--no-cd-reward-require-dag", dest="cd_reward_require_dag", action="store_false")
    parser.set_defaults(cd_reward_require_dag=True)
    parser.add_argument("--cd-graph-reward-scale", type=float, default=1.0)
    parser.add_argument("--cd-format-reward-scale", type=float, default=0.2)
    parser.add_argument("--cd-partial-format-reward-scale", type=float, default=0.25)
    parser.add_argument("--cd-edge-f1-reward-scale", type=float, default=0.0)
    parser.add_argument("--cd-low-shd-reward-scale", type=float, default=0.0)
    parser.add_argument("--cd-acyclic-reward-scale", type=float, default=0.0)
    parser.add_argument("--cd-cot-structure-reward-scale", type=float, default=0.0)
    parser.add_argument("--cd-skeleton-f1-reward-scale", type=float, default=0.0)
    parser.add_argument("--cd-vstruct-f1-reward-scale", type=float, default=0.0)
    parser.add_argument("--cd-orientation-f1-reward-scale", type=float, default=0.0)
    parser.add_argument("--cd-descendant-cot-structure-reward-scale", type=float, default=0.0)
    parser.add_argument("--cd-descendant-shift-ranking-reward-scale", type=float, default=0.0)
    parser.add_argument("--cd-descendant-variable-classification-reward-scale", type=float, default=0.0)
    parser.add_argument("--length-penalty-coef", type=float, default=0.0000333333)
    parser.add_argument("--length-penalty-target-tokens", type=int, default=0)
    parser.add_argument("--length-penalty-max-abs", type=float, default=1.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_model = _resolve_runtime_model_arg(args.model)

    csv_paths = list(args.csv or DEFAULT_CSVS)
    sampled_records = _sample_records(
        csv_paths,
        samples_per_csv=int(args.samples_per_csv),
        seed=int(args.seed),
    )
    task = _detect_task(sampled_records, args.task)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_slug = _safe_slug(args.model)
    default_dir = _HERE / "out" / "grpo_vllm_rollouts"
    if args.output_jsonl is None:
        args.output_jsonl = default_dir / f"{model_slug}_{task}_{timestamp}.jsonl"
    if args.summary_json is None:
        args.summary_json = args.output_jsonl.with_suffix(".summary.json")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    stop_sequences = list(args.stop_sequence or ["</answer>"])
    stop_sequences = [value for value in stop_sequences if value]

    print(
        f"[info] sampled {len(sampled_records)} prompts from {len(csv_paths)} CSVs"
        f" | task={task} | rollouts_per_prompt={args.rollouts}",
        flush=True,
    )
    if resolved_model != args.model:
        print(f"[info] resolved local model path: {resolved_model}", flush=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    reward_funcs = _build_reward_funcs(args, tokenizer, task)
    reward_names = [name for name, _ in reward_funcs]

    patched_tokenizer_attrs = _install_vllm_tokenizer_compat_shims(tokenizer)
    if patched_tokenizer_attrs:
        print(
            "[info] installed tokenizer compatibility shims for vLLM on "
            f"{tokenizer.__class__.__name__}: {', '.join(patched_tokenizer_attrs)}",
            flush=True,
        )

    tqdm_patch_message = _patch_tqdm_for_vllm()
    if tqdm_patch_message:
        print(f"[info] {tqdm_patch_message}", flush=True)

    from vllm import LLM, SamplingParams

    prompts = [_make_generation_prompt(record, args, task) for record in sampled_records]
    for record, prompt_model_input in zip(sampled_records, prompts):
        record["prompt_model_input"] = prompt_model_input
    if args.vllm_max_model_len is not None:
        resolved_max_model_len = int(args.vllm_max_model_len)
        max_prompt_tokens = max(
            (
                len(ids)
                for ids in tokenizer(
                    prompts,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                )["input_ids"]
            ),
            default=0,
        )
        buffer_tokens = max(256, min(2048, max(64, max_prompt_tokens // 8)))
        declared_max_model_len = _get_tokenizer_declared_max_length(tokenizer)
        max_len_warning = None
    else:
        (
            resolved_max_model_len,
            max_prompt_tokens,
            buffer_tokens,
            declared_max_model_len,
            max_len_warning,
        ) = _infer_vllm_max_model_len(tokenizer, prompts, int(args.max_new_tokens))
    if max_len_warning:
        print(max_len_warning, flush=True)
    resolved_gpu_mem_util, gpu_mem_message = _auto_adjust_vllm_gpu_mem_util(
        float(args.vllm_gpu_mem_util),
        int(args.tensor_parallel_size),
    )
    if gpu_mem_message:
        print(gpu_mem_message, flush=True)
    print(
        f"[info] using vLLM max_model_len={resolved_max_model_len} "
        f"(longest_prompt_tokens={max_prompt_tokens}, buffer_tokens={buffer_tokens}, "
        f"tokenizer_model_max_length={declared_max_model_len}) and "
        f"gpu_memory_utilization={resolved_gpu_mem_util:.3f}",
        flush=True,
    )

    llm_kwargs: dict[str, Any] = {
        "model": resolved_model,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "dtype": args.vllm_dtype,
        "gpu_memory_utilization": float(resolved_gpu_mem_util),
        "enforce_eager": bool(args.vllm_enforce_eager),
        "trust_remote_code": bool(args.trust_remote_code),
        "max_model_len": int(resolved_max_model_len),
    }
    try:
        llm = LLM(**llm_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize vLLM ({type(exc).__name__}: {exc}). "
            "Try lowering --vllm-gpu-mem-util, reducing --max-new-tokens, "
            "or explicitly setting --vllm-max-model-len. "
            f"Resolved settings were max_model_len={resolved_max_model_len}, "
            f"gpu_memory_utilization={resolved_gpu_mem_util:.3f}."
        ) from exc
    sampling_params = SamplingParams(
        n=int(args.rollouts),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_new_tokens),
        stop=stop_sequences or None,
        include_stop_str_in_output=True,
    )

    t0 = time.time()
    request_outputs = llm.generate(prompts, sampling_params)
    generation_seconds = time.time() - t0
    print(
        f"[info] generated {len(sampled_records) * int(args.rollouts)} rollouts in {generation_seconds:.2f}s",
        flush=True,
    )

    rollout_outputs: list[dict[str, Any]] = []
    for sample_idx, request_output in enumerate(request_outputs):
        record = sampled_records[sample_idx]
        outputs = sorted(request_output.outputs, key=lambda item: item.index)
        for rollout_idx, completion_output in enumerate(outputs):
            completion = str(completion_output.text)
            score_meta = _score_completion(task, completion, record["answer"], args)
            rollout_record: dict[str, Any] = {
                "sample_idx": sample_idx,
                "rollout_idx": rollout_idx,
                "source_csv": record["source_csv"],
                "source_name": record["source_name"],
                "row_index": int(record["row_index"]),
                "dataset": record.get("dataset", ""),
                "bif_file": record.get("bif_file", ""),
                "prompt_style": record.get("prompt_style", ""),
                "anonymize": record.get("anonymize", ""),
                "obs_per_prompt": record.get("obs_per_prompt", ""),
                "int_per_combo": record.get("int_per_combo", ""),
                "data_idx": record.get("data_idx", ""),
                "shuffle_idx": record.get("shuffle_idx", ""),
                "completion": completion,
                "answer": record["answer"],
                "score_meta": score_meta,
            }
            if args.include_prompts:
                rollout_record["prompt"] = record["prompt"]
                rollout_record["prompt_model_input"] = record.get("prompt_model_input", record["prompt"])
            rollout_outputs.append(rollout_record)

    reward_inputs = _build_reward_inputs(task, sampled_records, rollout_outputs)
    completion_texts = [item["completion"] for item in rollout_outputs]
    for reward_name, reward_fn in reward_funcs:
        reward_values = reward_fn(completion_texts, **reward_inputs)
        for item, reward_value in zip(rollout_outputs, reward_values):
            item.setdefault("rewards", {})[reward_name] = float(reward_value)
    for item in rollout_outputs:
        item["reward_total"] = float(sum(float(v) for v in item.get("rewards", {}).values()))

    summary = {
        "model": args.model,
        "task": task,
        "csv_paths": csv_paths,
        "samples_per_csv": int(args.samples_per_csv),
        "num_sampled_prompts": len(sampled_records),
        "rollouts_per_prompt": int(args.rollouts),
        "num_reward_components": len(reward_names),
        "reward_names": reward_names,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "stop_sequences": stop_sequences,
        "generation_seconds": round(generation_seconds, 4),
        "output_jsonl": str(args.output_jsonl),
        "summary_json": str(args.summary_json),
        "output_format": "flat_per_rollout" if bool(args.flat_output) else "grouped_by_prompt",
        "aggregate": _summarize_results(rollout_outputs, reward_names),
    }

    output_records = rollout_outputs if bool(args.flat_output) else _build_grouped_rollout_records(
        sampled_records,
        rollout_outputs,
    )
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for item in output_records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"[done] wrote {len(output_records)} JSONL rows to {args.output_jsonl}"
        f" ({summary['output_format']})",
        flush=True,
    )
    print(
        f"[done] mean reward_total={summary['aggregate']['reward_total_mean']:.4f}"
        f" | min={summary['aggregate']['reward_total_min']:.4f}"
        f" | max={summary['aggregate']['reward_total_max']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
