import os
import re
import argparse
import json
import time
import socket
import gc
import sys
import subprocess
import csv
import io
import inspect
from typing import Any, Dict
from contextlib import redirect_stdout, redirect_stderr

# Unsloth GRPO can fail in TorchDynamo fake-tensor tracing on some stacks.
# Default to eager mode unless user explicitly overrides the env var.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from pathlib import Path
from urllib import request, error
from urllib.parse import urlparse

FastLanguageModel = None
PatchFastRL = None
_HAS_UNSLOTH = False
_UNSLOTH_IMPORT_ERROR = None

try:
    # Import Unsloth before TRL/Transformers/PEFT when available.
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    _HAS_UNSLOTH = True
except Exception as e:
    _UNSLOTH_IMPORT_ERROR = e

try:
    # Optional in some Unsloth versions.
    from unsloth import PatchFastRL
except Exception:
    PatchFastRL = None

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_callback import PrinterCallback
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from verifier_cd import (
    cd_format_reward,
    build_cd_partial_format_reward,
    build_cd_graph_reward,
    build_length_penalty_reward,
    completion_to_text as cd_completion_to_text,
    extract_answer_text as cd_extract_answer_text,
    format_ok as cd_format_ok,
    score_cd_completion,
)

try:
    from math_verify import LatexExtractionConfig, parse, verify
    _HAS_MATH_VERIFY = True
except Exception:
    LatexExtractionConfig = None
    parse = None
    verify = None
    _HAS_MATH_VERIFY = False

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> and </think> tags, and <answer> and </answer> tags, respectively."
)

FORMAT_RE = re.compile(r"(?s)^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$")
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


class CompactMetricsCallback(TrainerCallback):
    """Print a compact metric line instead of full raw log dictionaries."""

    _KEYS = [
        ("epoch", "epoch"),
        ("loss", "loss"),
        ("reward", "reward"),
        ("rewards/accuracy_reward/mean", "acc"),
        ("rewards/cd_partial_format_reward/mean", "pfmt"),
        ("rewards/cd_graph_reward/mean", "cd"),
        ("rewards/format_reward/mean", "fmt"),
        ("rewards/cd_format_reward/mean", "fmt"),
        ("rewards/length_penalty_reward/mean", "len"),
        ("entropy", "ent"),
        ("step_time", "step_s"),
    ]

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero or not logs:
            return

        parts = []
        for key, name in self._KEYS:
            if key not in logs:
                continue
            value = logs[key]
            if isinstance(value, float):
                parts.append(f"{name}={value:.4f}")
            else:
                parts.append(f"{name}={value}")

        if parts:
            print("[train] " + " | ".join(parts))


class JsonlMetricsCallback(TrainerCallback):
    """Persist per-log training metrics as JSONL (one object per line)."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _to_jsonable(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if hasattr(value, "item"):
            try:
                return JsonlMetricsCallback._to_jsonable(value.item())
            except Exception:
                pass
        return str(value)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero or not logs:
            return

        payload = {
            "global_step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "time_unix": time.time(),
        }
        payload.update({k: self._to_jsonable(v) for k, v in logs.items()})

        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")


def build_argparser():
    p = argparse.ArgumentParser(description="GRPO training with TRL + vLLM server mode (Unsloth-ready)")
    p.add_argument("--mode", type=str, default="train", choices=["train", "ab_eval"])
    p.add_argument(
        "--task",
        type=str,
        default="causal_discovery",
        choices=["causal_discovery", "math"],
        help="Training/eval task type. causal_discovery expects prompt CSV inputs.",
    )
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    p.add_argument("--dataset_id", type=str, default="AI-MO/NuminaMath-TIR")
    p.add_argument("--train_split", type=str, default="train[:5%]")
    p.add_argument("--test_split", type=str, default="test[:5%]")
    p.add_argument("--output_dir", type=str, default="Qwen2-0.5B-GRPO-vLLM-server")

    # Prompt / generation
    p.add_argument("--max_prompt_tokens", type=int, default=0)
    p.add_argument("--max_completion_length", type=int, default=0)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument(
        "--stop-sequence",
        action="append",
        default=None,
        help=(
            "Stop sequence for rollout generation (repeatable). "
            "Default is </answer>. Use --stop-sequence '' to disable all stop sequences."
        ),
    )

    # Train
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--use-unsloth", dest="use_unsloth", action="store_true")
    p.add_argument("--no-use-unsloth", dest="use_unsloth", action="store_false")
    p.set_defaults(use_unsloth=True)
    p.add_argument("--unsloth-load-in-4bit", action="store_true", help="Load base model in 4-bit via Unsloth.")
    p.add_argument(
        "--unsloth-fast-inference",
        dest="unsloth_fast_inference",
        action="store_true",
        help="Enable Unsloth fast-inference path where available.",
    )
    p.add_argument("--no-unsloth-fast-inference", dest="unsloth_fast_inference", action="store_false")
    p.set_defaults(unsloth_fast_inference=True)
    p.add_argument(
        "--unsloth-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization hint for Unsloth+vLLM integrations.",
    )
    p.add_argument(
        "--unsloth-max-seq-length",
        type=int,
        default=0,
        help=(
            "Max sequence length used for Unsloth model loading. "
            "0 means auto-derive from prompt+completion limits when possible."
        ),
    )
    p.add_argument(
        "--unsloth-vllm-standby",
        dest="unsloth_vllm_standby",
        action="store_true",
        help=(
            "Set UNSLOTH_VLLM_STANDBY=1 before training setup for shared-memory rollout/training flows."
        ),
    )
    p.add_argument("--no-unsloth-vllm-standby", dest="unsloth_vllm_standby", action="store_false")
    p.set_defaults(unsloth_vllm_standby=True)
    p.add_argument(
        "--grpo-loss-type",
        type=str,
        default="auto",
        choices=["auto", "grpo", "dapo", "dr_grpo"],
        help="Optional GRPO loss variant if supported by the installed TRL.",
    )
    p.add_argument(
        "--grpo-importance-sampling-level",
        type=str,
        default="auto",
        choices=["auto", "token", "sequence"],
        help="Optional importance-sampling level if supported by the installed TRL.",
    )
    p.add_argument(
        "--length_penalty_coef",
        type=float,
        default=0.0,
        help=(
            "Per-token penalty coefficient. 0 disables length penalty reward. "
            "Reward contribution is negative."
        ),
    )
    p.add_argument(
        "--length_penalty_target_tokens",
        type=int,
        default=0,
        help=(
            "Deprecated/ignored. Length penalty now regularizes total completion length."
        ),
    )
    p.add_argument(
        "--length_penalty_max_abs",
        type=float,
        default=1.0,
        help=(
            "Max absolute magnitude of length penalty reward per sample. "
            "Set <=0 for uncapped penalty."
        ),
    )

    # Causal-discovery dataset + verifier options
    p.add_argument(
        "--cd-train-csv",
        action="append",
        default=[],
        help=(
            "Prompt CSV for causal discovery training (repeatable). "
            "Expected columns include prompt_text or prompt_path plus answer or answer_path."
        ),
    )
    p.add_argument(
        "--cd-test-csv",
        action="append",
        default=[],
        help="Optional prompt CSV for held-out eval split in causal discovery mode (repeatable).",
    )
    p.add_argument(
        "--cd-test-fraction",
        type=float,
        default=0.1,
        help="If --cd-test-csv is not provided, reserve this fraction from train as eval.",
    )
    p.add_argument("--cd-max-train-samples", type=int, default=0, help="Cap train samples (0 = no cap).")
    p.add_argument("--cd-max-test-samples", type=int, default=0, help="Cap eval samples (0 = no cap).")
    p.add_argument(
        "--cd-wrap-system-prompt",
        action="store_true",
        help="Wrap causal prompt text using the generic system/user/assistant template.",
    )
    p.add_argument(
        "--no-cd-wrap-system-prompt",
        dest="cd_wrap_system_prompt",
        action="store_false",
        help="Disable wrapping causal prompt text using the generic system/user/assistant template.",
    )
    p.set_defaults(cd_wrap_system_prompt=True)
    p.add_argument("--cd-reward-shd-weight", type=float, default=0.0, help="Weight for normalized SHD penalty.")
    p.add_argument("--cd-reward-dag-penalty", type=float, default=0.1, help="Penalty applied if predicted graph has cycles.")
    p.add_argument(
        "--cd-partial-format-reward-scale",
        type=float,
        default=0.25,
        help="Scale for dense partial-format shaping reward (0 disables).",
    )
    p.add_argument("--cd-reward-require-dag", dest="cd_reward_require_dag", action="store_true")
    p.add_argument("--no-cd-reward-require-dag", dest="cd_reward_require_dag", action="store_false")
    p.set_defaults(cd_reward_require_dag=True)

    # vLLM server
    p.add_argument(
        "--vllm_server_base_url",
        type=str,
        default=os.environ.get("VLLM_SERVER_BASE_URL", "http://127.0.0.1:8000"),
        help="vLLM server URL. Can also be set via VLLM_SERVER_BASE_URL.",
    )
    p.add_argument("--vllm_server_timeout", type=float, default=240.0)
    p.add_argument("--vllm_group_port", type=int, default=51216)
    p.add_argument("--vllm_preflight_timeout", type=float, default=15.0)
    p.add_argument("--use-vllm", dest="use_vllm", action="store_true")
    p.add_argument("--no-use-vllm", dest="use_vllm", action="store_false")
    p.set_defaults(use_vllm=True)
    p.add_argument(
        "--enable_vllm_preflight",
        action="store_true",
        help="Run an early communicator init/close probe before trainer startup.",
    )
    p.add_argument(
        "--auto-launch-vllm-server",
        action="store_true",
        help="Launch TRL vLLM server automatically on first visible GPU and train on the rest.",
    )
    p.add_argument(
        "--vllm_server_model_id",
        type=str,
        default=None,
        help="Model id/path for auto-launched vLLM server. Defaults to --model_id.",
    )
    p.add_argument("--vllm_server_startup_timeout", type=float, default=120.0)
    p.add_argument("--vllm_server_gpu_memory_utilization", type=float, default=0.2)
    p.add_argument(
        "--vllm_server_max_model_len",
        type=int,
        default=None,
        help="Max model length for auto-launched vLLM server (<=0 lets vLLM pick a default).",
    )
    p.add_argument("--vllm_server_log_file", type=str, default="vllm_server.log")

    # Logging
    p.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "none"])
    p.add_argument("--run_name", type=str, default="qwen2-grpo-vllm-server")
    p.add_argument("--wandb_project", type=str, default="enco-grpo")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=1, help="Keep only the most recent N checkpoints.")
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument(
        "--train_log_jsonl",
        type=str,
        default=None,
        help="Optional JSONL file path for per-step trainer logs (written by local rank 0 only).",
    )
    p.add_argument(
        "--save-eval-responses",
        dest="save_eval_responses",
        action="store_true",
        help="Persist every reward-evaluated completion to output_dir/grpo_log as JSONL.",
    )
    p.add_argument("--no-save-eval-responses", dest="save_eval_responses", action="store_false")
    p.set_defaults(save_eval_responses=True)
    p.add_argument(
        "--eval-responses-max-chars",
        type=int,
        default=0,
        help="Optional max chars for prompt/completion/answer fields in eval-response logs (0 = no truncation).",
    )

    # A/B eval (base vs GRPO on frozen set)
    p.add_argument("--ab_base_model", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    p.add_argument("--ab_grpo_model", type=str, default=None)
    p.add_argument("--ab_eval_split", type=str, default="test")
    p.add_argument("--ab_eval_n", type=int, default=200)
    p.add_argument("--ab_eval_seed", type=int, default=1234)
    p.add_argument("--ab_pass_k", type=int, default=1)
    p.add_argument("--ab_max_new_tokens", type=int, default=128)
    p.add_argument("--ab_temperature", type=float, default=0.0)
    p.add_argument("--ab_top_p", type=float, default=0.95)
    p.add_argument("--ab_do_sample", action="store_true")
    p.add_argument("--ab_output_json", type=str, default=None)
    p.add_argument("--ab_debug_csv", type=str, default=None)

    return p


def _filter_supported_kwargs(callable_obj, kwargs: dict) -> dict:
    try:
        params = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def build_prompt(problem: str, tokenizer: Any = None, enable_thinking: bool = True) -> str:
    """
    Build a chat-formatted prompt.
    For Qwen3 tokenizers, use apply_chat_template(enable_thinking=...) when available.
    Falls back to legacy plain-text format if chat templating is unavailable.
    """
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(problem)},
        ]
        kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            params = inspect.signature(tokenizer.apply_chat_template).parameters
            if "enable_thinking" in params:
                kwargs["enable_thinking"] = bool(enable_thinking)
        except (TypeError, ValueError):
            pass
        try:
            return str(tokenizer.apply_chat_template(messages, **kwargs))
        except TypeError:
            # Older tokenizers may not accept some kwargs.
            kwargs.pop("enable_thinking", None)
            try:
                return str(tokenizer.apply_chat_template(messages, **kwargs))
            except Exception:
                pass
        except Exception:
            pass
    return f"system\n{SYSTEM_PROMPT}\nuser\n{problem}\nassistant\n"


def _completion_to_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        if not completion:
            return ""
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(first)
    return str(completion)


def _extract_answer_text(text: str) -> str:
    m = ANSWER_RE.search(text)
    return m.group(1) if m else text


def _require_math_verify():
    if not _HAS_MATH_VERIFY:
        raise RuntimeError(
            "math_verify is not installed, but --task math was requested. "
            "Install math_verify or switch to --task causal_discovery."
        )


def _resolve_existing_path(path_str: str, *, csv_path: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p

    candidates = [
        csv_path.parent / p,
        Path.cwd() / p,
        Path(__file__).resolve().parent.parent / p,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Could not resolve path '{path_str}' from CSV {csv_path}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def _load_cd_rows_from_prompt_csv(
    csv_path: Path,
    *,
    wrap_system_prompt: bool,
    tokenizer: Any = None,
    enable_thinking: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            prompt_raw = (row.get("prompt_text") or "").strip()
            if not prompt_raw:
                prompt_path_raw = (row.get("prompt_path") or "").strip()
                if not prompt_path_raw:
                    raise ValueError(
                        f"{csv_path}: row {i} missing both prompt_text and prompt_path."
                    )
                prompt_path = _resolve_existing_path(prompt_path_raw, csv_path=csv_path)
                prompt_raw = prompt_path.read_text(encoding="utf-8", errors="ignore")

            prompt_text = prompt_raw
            if wrap_system_prompt:
                prompt_text = build_prompt(
                    prompt_text,
                    tokenizer=tokenizer,
                    enable_thinking=enable_thinking,
                )

            answer_raw = (row.get("answer") or "").strip()
            if not answer_raw:
                answer_path_raw = (row.get("answer_path") or "").strip()
                if not answer_path_raw:
                    raise ValueError(
                        f"{csv_path}: row {i} missing both answer and answer_path."
                    )
                answer_path = _resolve_existing_path(answer_path_raw, csv_path=csv_path)
                # Keep path string so verifier can parse answer payload on demand.
                answer_raw = str(answer_path)

            rows.append(
                {
                    "prompt_raw": prompt_raw,
                    "prompt": prompt_text,
                    "answer": answer_raw,
                }
            )
    return rows


def _dataset_from_cd_csvs(
    csv_paths: list[str],
    *,
    wrap_system_prompt: bool,
    tokenizer: Any = None,
    enable_thinking: bool = True,
) -> Dataset:
    datasets_list = []
    for path_str in csv_paths:
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"--cd-train/test-csv file not found: {p}")
        rows = _load_cd_rows_from_prompt_csv(
            p,
            wrap_system_prompt=wrap_system_prompt,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
        )
        if not rows:
            raise ValueError(f"No usable rows found in {p}")
        datasets_list.append(Dataset.from_list(rows))

    if not datasets_list:
        raise ValueError("No causal discovery CSV inputs were provided.")
    if len(datasets_list) == 1:
        return datasets_list[0]
    return concatenate_datasets(datasets_list)


def _is_correct(answer_text: str, solution: str) -> float:
    _require_math_verify()
    gold = parse(
        solution,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    pred = parse(
        answer_text,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )

    if len(gold) == 0:
        return 1.0
    try:
        return float(verify(pred, gold))
    except Exception:
        return 0.0


def _score_answer_with_meta(answer_text: str, solution: str):
    _require_math_verify()
    gold = parse(
        solution,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    pred = parse(
        answer_text,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )

    if len(gold) == 0:
        return 1.0, False, False

    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            score = float(verify(pred, gold))
        log_text = buf.getvalue().lower()
        timed_out = "timeout" in log_text
        return score, timed_out, False
    except Exception as exc:
        msg = f"{exc}".lower()
        log_text = buf.getvalue().lower()
        timed_out = ("timeout" in msg) or ("timeout" in log_text)
        return 0.0, timed_out, True


def _load_eval_model(model_id_or_path: str):
    is_adapter = (Path(model_id_or_path) / "adapter_config.json").exists()
    if is_adapter:
        from peft import AutoPeftModelForCausalLM

        return AutoPeftModelForCausalLM.from_pretrained(
            model_id_or_path,
            dtype="auto",
            device_map="auto",
        ).eval()

    return AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        dtype="auto",
        device_map="auto",
    ).eval()


def _eval_on_dataset(
    model_id_or_path: str,
    eval_dataset,
    task: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    pass_k: int,
    debug_model_label: str = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_eval_model(model_id_or_path)
    input_device = next(model.parameters()).device

    fmt_ok = 0
    acc_sum = 0.0
    pass_k_ok = 0.0
    verify_timeouts = 0
    verify_errors = 0
    debug_rows = []

    for idx, example in enumerate(eval_dataset):
        if task == "math":
            prompt = build_prompt(example["problem"], tokenizer=tokenizer, enable_thinking=True)
        else:
            prompt = str(example["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
        prompt_len = inputs["input_ids"].shape[1]

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": pass_k,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.no_grad():
            out = model.generate(
                **inputs,
                **generation_kwargs,
            )

        completions = [
            tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            for seq in out
        ]
        first_completion = completions[0]
        if task == "math":
            fmt_ok += int(bool(FORMAT_RE.match(first_completion)))
        else:
            fmt_ok += int(cd_format_ok(first_completion))

        sample_scores = []
        for sample_idx, completion_text in enumerate(completions):
            if task == "math":
                answer_text = _extract_answer_text(completion_text)
                score, timed_out, had_error = _score_answer_with_meta(answer_text, example["solution"])
            else:
                answer_text = cd_extract_answer_text(cd_completion_to_text(completion_text))
                meta = score_cd_completion(
                    completion_text=completion_text,
                    target_answer=example["answer"],
                    require_dag=True,
                    dag_penalty=0.0,
                    shd_weight=0.0,
                )
                score = float(meta["reward"])
                timed_out = False
                had_error = False
            sample_scores.append(score)
            verify_timeouts += int(timed_out)
            verify_errors += int(had_error)

            if debug_model_label is not None:
                debug_rows.append(
                    {
                        "model": debug_model_label,
                        "example_idx": idx,
                        "sample_idx": sample_idx,
                        "problem": example.get("problem", example.get("prompt", "")),
                        "solution": example.get("solution", example.get("answer", "")),
                        "completion": completion_text,
                        "answer_text": answer_text,
                        "format_ok": int(bool(FORMAT_RE.match(completion_text))) if task == "math" else int(cd_format_ok(completion_text)),
                        "is_correct": score,
                        "verify_timed_out": int(timed_out),
                        "verify_error": int(had_error),
                    }
                )

        acc_sum += sample_scores[0]
        pass_k_ok += float(max(sample_scores))

    n = len(eval_dataset)
    metrics = {
        "model": model_id_or_path,
        "n": n,
        "format_rate": fmt_ok / n,
        "accuracy": acc_sum / n,
        f"pass@{pass_k}": pass_k_ok / n,
        "verify_timeouts": verify_timeouts,
        "verify_errors": verify_errors,
    }
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, debug_rows


def run_ab_eval(args):
    if not args.ab_grpo_model:
        raise ValueError("--ab_grpo_model is required when --mode ab_eval")
    if args.ab_eval_n <= 0:
        raise ValueError("--ab_eval_n must be > 0")
    if args.ab_pass_k <= 0:
        raise ValueError("--ab_pass_k must be > 0")

    if args.task == "math":
        full_eval = load_dataset(args.dataset_id, split=args.ab_eval_split)
        n = min(args.ab_eval_n, len(full_eval))
        frozen_eval = full_eval.shuffle(seed=args.ab_eval_seed).select(range(n))
    else:
        eval_sources = args.cd_test_csv if args.cd_test_csv else args.cd_train_csv
        if not eval_sources:
            raise ValueError("For --task causal_discovery, pass --cd-train-csv (and optionally --cd-test-csv).")
        full_eval = _dataset_from_cd_csvs(
            eval_sources,
            wrap_system_prompt=bool(args.cd_wrap_system_prompt),
        )
        n = min(args.ab_eval_n, len(full_eval))
        frozen_eval = full_eval.shuffle(seed=args.ab_eval_seed).select(range(n))

    do_sample = args.ab_do_sample or args.ab_temperature > 0 or args.ab_pass_k > 1
    if args.ab_pass_k > 1 and not do_sample:
        raise ValueError("pass@k with k>1 requires sampling; set --ab_do_sample or --ab_temperature > 0")

    base_metrics, base_debug_rows = _eval_on_dataset(
        model_id_or_path=args.ab_base_model,
        eval_dataset=frozen_eval,
        task=args.task,
        max_new_tokens=args.ab_max_new_tokens,
        temperature=args.ab_temperature,
        top_p=args.ab_top_p,
        do_sample=do_sample,
        pass_k=args.ab_pass_k,
        debug_model_label="base" if args.ab_debug_csv else None,
    )
    grpo_metrics, grpo_debug_rows = _eval_on_dataset(
        model_id_or_path=args.ab_grpo_model,
        eval_dataset=frozen_eval,
        task=args.task,
        max_new_tokens=args.ab_max_new_tokens,
        temperature=args.ab_temperature,
        top_p=args.ab_top_p,
        do_sample=do_sample,
        pass_k=args.ab_pass_k,
        debug_model_label="grpo" if args.ab_debug_csv else None,
    )

    pass_k_key = f"pass@{args.ab_pass_k}"
    result = {
        "dataset_id": args.dataset_id,
        "split": args.ab_eval_split,
        "seed": args.ab_eval_seed,
        "n": n,
        "generation": {
            "max_new_tokens": args.ab_max_new_tokens,
            "do_sample": do_sample,
            "temperature": args.ab_temperature if do_sample else 0.0,
            "top_p": args.ab_top_p if do_sample else 1.0,
            "pass_k": args.ab_pass_k,
        },
        "base": base_metrics,
        "grpo": grpo_metrics,
        "delta_grpo_minus_base": {
            "format_rate": grpo_metrics["format_rate"] - base_metrics["format_rate"],
            "accuracy": grpo_metrics["accuracy"] - base_metrics["accuracy"],
            pass_k_key: grpo_metrics[pass_k_key] - base_metrics[pass_k_key],
        },
    }

    print(json.dumps(result, indent=2))
    if args.ab_output_json:
        with open(args.ab_output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print("Saved eval JSON to:", args.ab_output_json)
    if args.ab_debug_csv:
        all_rows = base_debug_rows + grpo_debug_rows
        fieldnames = [
            "model",
            "example_idx",
            "sample_idx",
            "problem",
            "solution",
            "completion",
            "answer_text",
            "format_ok",
            "is_correct",
            "verify_timed_out",
            "verify_error",
        ]
        with open(args.ab_debug_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print("Saved debug CSV to:", args.ab_debug_csv)


def _parse_visible_cuda_devices():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip():
        return [d.strip() for d in cvd.split(",") if d.strip()]
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []


def _wait_for_health(url: str, timeout_s: float, proc=None):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with request.urlopen(url, timeout=2.0) as resp:
                if resp.status < 400:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _build_trl_vllm_serve_cmd(args, host: str, port: int):
    model_id = args.vllm_server_model_id or args.model_id
    cmd = [
        "trl",
        "vllm-serve",
        "--model",
        model_id,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(args.vllm_server_gpu_memory_utilization),
    ]
    # <=0 is treated as "unset" so vLLM can choose a safe default.
    if args.vllm_server_max_model_len > 0:
        cmd.extend(["--max-model-len", str(args.vllm_server_max_model_len)])
    return cmd


def _maybe_launch_local_vllm_and_reexec_train(args):
    if args.mode != "train" or not args.use_vllm or not args.auto_launch_vllm_server:
        return
    if os.environ.get("GRPO_TRAIN_CHILD") == "1":
        return
    if os.environ.get("LOCAL_RANK") is not None:
        raise RuntimeError(
            "--auto-launch-vllm-server should be run with plain python (not inside accelerate launch). "
            "The script will launch accelerate for the training child process automatically."
        )

    visible_gpus = _parse_visible_cuda_devices()
    if len(visible_gpus) < 2:
        raise RuntimeError(
            "Auto vLLM launch needs at least 2 visible GPUs: one for server and one or more for training. "
            f"Visible GPUs: {visible_gpus}"
        )

    server_gpu = visible_gpus[0]
    train_gpus = visible_gpus[1:]
    parsed = urlparse(args.vllm_server_base_url.rstrip("/"))
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    health_url = f"http://{host}:{port}/health"

    server_env = os.environ.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = server_gpu
    log_file = open(args.vllm_server_log_file, "w", encoding="utf-8")

    server_cmd = _build_trl_vllm_serve_cmd(args, host=host, port=port)
    try:
        server_proc = subprocess.Popen(
            server_cmd,
            env=server_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        log_file.close()
        raise RuntimeError(
            "Failed to start auto vLLM server: `trl` command not found in this environment. "
            "Install TRL CLI or start the server manually and omit --auto-launch-vllm-server."
        ) from exc

    try:
        ready = _wait_for_health(health_url, args.vllm_server_startup_timeout, proc=server_proc)
        if not ready:
            exit_code = server_proc.poll()
            if exit_code is not None:
                raise RuntimeError(
                    "Auto-launched vLLM server exited before becoming healthy "
                    f"(exit code {exit_code}). Check logs: {args.vllm_server_log_file}"
                )
            raise RuntimeError(
                "Auto-launched vLLM server did not become healthy in time. "
                f"Check logs: {args.vllm_server_log_file}"
            )

        child_env = os.environ.copy()
        child_env["GRPO_TRAIN_CHILD"] = "1"
        child_env["CUDA_VISIBLE_DEVICES"] = ",".join(train_gpus)
        if len(train_gpus) > 1:
            child_cmd = [
                "accelerate",
                "launch",
                "--num_processes",
                str(len(train_gpus)),
                str(Path(__file__).resolve()),
                *sys.argv[1:],
            ]
        else:
            child_cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]

        print(
            f"[orchestrator] vLLM server on GPU {server_gpu}; training child on GPUs {train_gpus}; "
            f"server url={args.vllm_server_base_url}"
        )
        try:
            rc = subprocess.call(child_cmd, env=child_env)
        except FileNotFoundError as exc:
            if child_cmd and child_cmd[0] == "accelerate":
                raise RuntimeError(
                    "Failed to start training child: `accelerate` command not found. "
                    "Install accelerate or run with only one training GPU."
                ) from exc
            raise
        raise SystemExit(rc)
    finally:
        try:
            server_proc.terminate()
            server_proc.wait(timeout=10)
        except Exception:
            try:
                server_proc.kill()
            except Exception:
                pass
        log_file.close()


def run_train(args):
    args.vllm_server_base_url = args.vllm_server_base_url.rstrip("/")
    if args.max_prompt_tokens <= 0:
        args.max_prompt_tokens = 2048
        print("[warn] max_prompt_tokens <= 0; using safer default 2048.")
    if args.max_completion_length <= 0:
        args.max_completion_length = 512
        print("[warn] max_completion_length <= 0; using safer default 512.")
    if not args.stop_sequence:
        args.stop_sequence = ["</answer>"]
    args.stop_sequence = [s for s in args.stop_sequence if isinstance(s, str) and s]

    # Throughput-oriented defaults for Ampere/Hopper GPUs.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if args.use_vllm:
        # Fail fast if the configured vLLM endpoint is unreachable.
        health_url = f"{args.vllm_server_base_url}/health"
        try:
            with request.urlopen(health_url, timeout=10) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status} from {health_url}")
        except (error.URLError, RuntimeError) as e:
            raise RuntimeError(
                f"Could not reach vLLM server at {health_url}. "
                "Start the server first or set --vllm_server_base_url / VLLM_SERVER_BASE_URL."
            ) from e

    def _json_get(url, timeout=5.0):
        with request.urlopen(url, timeout=timeout) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status} from {url}")
            return json.loads(resp.read().decode("utf-8"))

    def _json_post(url, payload=None, timeout=5.0):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        with request.urlopen(req, timeout=timeout) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status} from {url}")
            body = resp.read()
            return json.loads(body.decode("utf-8")) if body else {}

    def _wait_tcp_ready(host, port, total_timeout):
        deadline = time.time() + total_timeout
        while time.time() < deadline:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return True
            except OSError:
                time.sleep(0.2)
        return False

    def _preflight_vllm_communicator():
        # Only the local rank 0 process probes the communicator to avoid races.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank != 0:
            return

        parsed = urlparse(args.vllm_server_base_url)
        host = parsed.hostname or "127.0.0.1"

        try:
            world = _json_get(f"{args.vllm_server_base_url}/get_world_size/", timeout=5.0)
            vllm_world_size = int(world["world_size"])
        except Exception as e:
            raise RuntimeError(
                "vLLM server does not expose /get_world_size/. "
                "Make sure the server was started with TRL's `trl vllm-serve`."
            ) from e

        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            client_device_uuid = str(torch.cuda.get_device_properties(device_idx).uuid)
        else:
            client_device_uuid = "cpu"

        init_url = f"{args.vllm_server_base_url}/init_communicator/"
        close_url = f"{args.vllm_server_base_url}/close_communicator/"
        try:
            _json_post(
                init_url,
                payload={
                    "host": "0.0.0.0",
                    "port": args.vllm_group_port,
                    "world_size": vllm_world_size + 1,
                    "client_device_uuid": client_device_uuid,
                },
                timeout=5.0,
            )
            ready = _wait_tcp_ready(host, args.vllm_group_port, args.vllm_preflight_timeout)
            if not ready:
                raise RuntimeError(
                    f"Timed out waiting for communicator TCPStore at {host}:{args.vllm_group_port}. "
                    "The vLLM server did not bring up the weight-sync group."
                )
        finally:
            try:
                _json_post(close_url, payload=None, timeout=3.0)
            except Exception:
                pass

    if args.use_vllm and args.enable_vllm_preflight:
        _preflight_vllm_communicator()

    # ---- W&B import + TRL profiling workaround (prevents NameError you saw)
    if args.report_to == "wandb":
        # Configure W&B from CLI flags for reproducible logging behavior.
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        os.environ["WANDB_MODE"] = args.wandb_mode
        import wandb  # noqa: F401
        import trl.extras.profiling as trl_profiling
        trl_profiling.wandb = __import__("wandb")

    # ---- Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Dataset
    if args.task == "math":
        train_dataset, test_dataset = load_dataset(
            args.dataset_id,
            split=[args.train_split, args.test_split],
        )

        def add_prompt(example, max_prompt_tokens=args.max_prompt_tokens):
            text = build_prompt(example["problem"], tokenizer=tokenizer, enable_thinking=True)
            ids = tokenizer(text, truncation=True, max_length=max_prompt_tokens)["input_ids"]
            return {"prompt": tokenizer.decode(ids, skip_special_tokens=True)}

        train_dataset = train_dataset.map(add_prompt)
        test_dataset = test_dataset.map(add_prompt)

        # Keep only what GRPO + reward needs
        keep_cols = ["prompt", "prompt_raw", "solution"]
        train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in keep_cols])
        test_dataset = test_dataset.remove_columns([c for c in test_dataset.column_names if c not in keep_cols])
    else:
        if not args.cd_train_csv:
            raise ValueError(
                "--task causal_discovery requires --cd-train-csv (repeatable) pointing to prompt CSVs."
            )

        raw_train = _dataset_from_cd_csvs(
            args.cd_train_csv,
            wrap_system_prompt=bool(args.cd_wrap_system_prompt),
            tokenizer=tokenizer,
            enable_thinking=True,
        )
        if args.cd_test_csv:
            raw_test = _dataset_from_cd_csvs(
                args.cd_test_csv,
                wrap_system_prompt=bool(args.cd_wrap_system_prompt),
                tokenizer=tokenizer,
                enable_thinking=True,
            )
        else:
            split = raw_train.train_test_split(
                test_size=float(args.cd_test_fraction),
                seed=int(args.ab_eval_seed),
            )
            raw_train = split["train"]
            raw_test = split["test"]

        if args.cd_max_train_samples and args.cd_max_train_samples > 0:
            n_train = min(int(args.cd_max_train_samples), len(raw_train))
            raw_train = raw_train.select(range(n_train))
        if args.cd_max_test_samples and args.cd_max_test_samples > 0:
            n_test = min(int(args.cd_max_test_samples), len(raw_test))
            raw_test = raw_test.select(range(n_test))

        def _truncate_cd_prompt(example, max_prompt_tokens=args.max_prompt_tokens):
            text = str(example["prompt"])
            ids = tokenizer(text, truncation=True, max_length=max_prompt_tokens)["input_ids"]
            return {"prompt": tokenizer.decode(ids, skip_special_tokens=True)}

        train_dataset = raw_train.map(_truncate_cd_prompt)
        test_dataset = raw_test.map(_truncate_cd_prompt)
        keep_cols = ["prompt", "prompt_raw", "answer"]
        train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in keep_cols])
        test_dataset = test_dataset.remove_columns([c for c in test_dataset.column_names if c not in keep_cols])

    # ---- Model + LoRA
    if args.use_unsloth and args.unsloth_vllm_standby:
        os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

    distributed_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.use_unsloth:
        if not _HAS_UNSLOTH:
            detail = f" Import error: {_UNSLOTH_IMPORT_ERROR}" if _UNSLOTH_IMPORT_ERROR is not None else ""
            raise RuntimeError(
                "--use-unsloth was requested, but unsloth is not available in this environment."
                + detail
            )

        if PatchFastRL is not None:
            try:
                PatchFastRL("GRPO", FastLanguageModel)
            except Exception:
                # Some versions may already patch or not require explicit patching.
                pass

        unsloth_max_seq_length = int(args.unsloth_max_seq_length)
        if unsloth_max_seq_length <= 0:
            if args.max_prompt_tokens > 0 and args.max_completion_length > 0:
                unsloth_max_seq_length = int(args.max_prompt_tokens + args.max_completion_length)
            else:
                unsloth_max_seq_length = 2048

        load_kwargs = {
            "model_name": args.model_id,
            "max_seq_length": unsloth_max_seq_length,
            "dtype": None,
            "load_in_4bit": bool(args.unsloth_load_in_4bit),
            "fast_inference": bool(args.unsloth_fast_inference),
            "gpu_memory_utilization": float(args.unsloth_gpu_memory_utilization),
        }
        load_kwargs = _filter_supported_kwargs(FastLanguageModel.from_pretrained, load_kwargs)
        model, unsloth_tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
        tokenizer = unsloth_tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        lora_kwargs = {
            "r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"],
            "use_gradient_checkpointing": "unsloth",
        }
        lora_kwargs = _filter_supported_kwargs(FastLanguageModel.get_peft_model, lora_kwargs)
        model = FastLanguageModel.get_peft_model(model, **lora_kwargs)
    else:
        model_load_kwargs = {
            "dtype": "auto",
        }
        # `device_map="auto"` is incompatible with distributed Accelerate launches.
        if distributed_world_size == 1:
            model_load_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            **model_load_kwargs,
        )

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)

    # Some compiled Unsloth trainer variants call model.for_training()/for_inference().
    if not hasattr(model, "for_training"):
        def _for_training(*args, **kwargs):
            model.train()
            return model
        setattr(model, "for_training", _for_training)
    if not hasattr(model, "for_inference"):
        def _for_inference(*args, **kwargs):
            model.eval()
            return model
        setattr(model, "for_inference", _for_inference)

    # ---- Reward logging
    logs_dir = Path(args.output_dir) / "grpo_log"
    logs_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        for p in logs_dir.glob("*.jsonl"):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        print(f"[log] cleared previous JSONL logs in {logs_dir}")
    eval_response_log_path = logs_dir / f"reward_responses_rank{rank}.jsonl"
    reward_call_counter = {"calls": 0}
    generation_counter = {"calls": 0}
    active_generation_id = {"id": -1}
    pending_reward_rows = {}

    if int(os.environ.get("LOCAL_RANK", "0")) == 0 and args.save_eval_responses:
        print(f"[log] saving reward-evaluated responses to {eval_response_log_path}")

    def _truncate_text_for_log(text: str) -> str:
        s = str(text)
        max_chars = int(args.eval_responses_max_chars)
        if max_chars > 0 and len(s) > max_chars:
            return s[:max_chars] + "...[truncated]"
        return s

    def _value_at(values, idx: int):
        if isinstance(values, (list, tuple)):
            return values[idx] if idx < len(values) else None
        return values

    def _to_jsonable(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [_to_jsonable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _to_jsonable(v) for k, v in value.items()}
        if hasattr(value, "item"):
            try:
                return _to_jsonable(value.item())
            except Exception:
                pass
        return str(value)

    def _normalize_prompt_item(item) -> str:
        if item is None:
            return ""
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            if "content" in item and isinstance(item.get("content"), str):
                return str(item.get("content", ""))
            if "prompt" in item and isinstance(item.get("prompt"), str):
                return str(item.get("prompt", ""))
            if "text" in item and isinstance(item.get("text"), str):
                return str(item.get("text", ""))
            try:
                return json.dumps(item, ensure_ascii=False)
            except Exception:
                return str(item)
        if isinstance(item, (list, tuple)):
            if item and all(isinstance(x, dict) for x in item):
                parts = []
                for msg in item:
                    role = str(msg.get("role", ""))
                    content = str(msg.get("content", ""))
                    if role:
                        parts.append(f"{role}: {content}")
                    else:
                        parts.append(content)
                return "\n".join(parts)
            try:
                return json.dumps(item, ensure_ascii=False)
            except Exception:
                return str(item)
        return str(item)

    def _extract_prompt_texts(kwargs: dict, batch_size: int):
        candidate_keys = ("prompt", "prompts", "messages", "inputs", "input", "query", "queries", "text", "texts")
        for key in candidate_keys:
            value = kwargs.get(key)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                texts = [_normalize_prompt_item(v) for v in value]
                if len(texts) == batch_size:
                    return key, texts
                if len(texts) == 1 and batch_size > 1:
                    return key, texts * batch_size
            else:
                return key, [_normalize_prompt_item(value)] * batch_size
        return "", [""] * batch_size

    def _extract_raw_prompt_texts(kwargs: dict, batch_size: int):
        value = kwargs.get("prompt_raw")
        if value is None:
            return "", [""] * batch_size
        if isinstance(value, (list, tuple)):
            texts = [_normalize_prompt_item(v) for v in value]
            if len(texts) == batch_size:
                return "prompt_raw", texts
            if len(texts) == 1 and batch_size > 1:
                return "prompt_raw", texts * batch_size
            return "prompt_raw", [""] * batch_size
        return "prompt_raw", [_normalize_prompt_item(value)] * batch_size

    def _log_reward_evaluations(
        generation_id: int,
        reward_name: str,
        completions,
        rewards,
        kwargs,
        *,
        flush: bool,
    ):
        if not args.save_eval_responses:
            return

        call_idx = reward_call_counter["calls"]
        reward_call_counter["calls"] += 1
        ts = time.time()
        prompt_key, prompt_texts = _extract_prompt_texts(kwargs, len(completions))
        _, prompt_raw_texts = _extract_raw_prompt_texts(kwargs, len(completions))
        answers = kwargs.get("answer")
        solutions = kwargs.get("solution")

        for i, completion in enumerate(completions):
            completion_text = _completion_to_text(completion)
            answer_text = (
                _extract_answer_text(completion_text)
                if args.task == "math"
                else cd_extract_answer_text(cd_completion_to_text(completion_text))
            )
            key = (generation_id, i)
            row = pending_reward_rows.get(key)
            if row is None:
                row = {
                    "time_unix": ts,
                    "rank": rank,
                    "generation_idx": generation_id,
                    "last_reward_call_idx": call_idx,
                    "sample_idx": i,
                    "prompt_key": prompt_key,
                    "format_ok": (
                        int(bool(FORMAT_RE.match(completion_text)))
                        if args.task == "math"
                        else int(cd_format_ok(completion_text))
                    ),
                    "prompt": _truncate_text_for_log(
                        (
                            prompt_raw_texts[i]
                            if i < len(prompt_raw_texts) and prompt_raw_texts[i]
                            else (prompt_texts[i] if i < len(prompt_texts) else "")
                        )
                    ),
                    "prompt_model_input": _truncate_text_for_log(prompt_texts[i] if i < len(prompt_texts) else ""),
                    "completion": _truncate_text_for_log(completion_text),
                    "answer_text": _truncate_text_for_log(answer_text),
                    "target_answer": _truncate_text_for_log(_value_at(answers, i) or _value_at(solutions, i) or ""),
                    "rewards": {},
                }
                pending_reward_rows[key] = row

            row["time_unix"] = ts
            row["last_reward_call_idx"] = call_idx
            row["rewards"][reward_name] = float(_to_jsonable(_value_at(rewards, i)) or 0.0)

        if flush and pending_reward_rows:
            emit_keys = sorted(k for k in pending_reward_rows.keys() if k[0] == generation_id)
            with eval_response_log_path.open("a", encoding="utf-8") as f:
                for key in emit_keys:
                    row = pending_reward_rows.pop(key)
                    rewards_map = row.get("rewards", {})
                    row["reward_total"] = float(sum(float(v) for v in rewards_map.values()))
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- Reward functions
    def format_reward_math(completions, **kwargs):
        texts = [_completion_to_text(c) for c in completions]
        return [1.0 if FORMAT_RE.match(t) else 0.0 for t in texts]

    def accuracy_reward_math(completions, **kwargs):
        solutions = kwargs["solution"]
        rewards = []
        for completion, solution in zip(completions, solutions):
            text = _completion_to_text(completion)
            answer_text = _extract_answer_text(text)
            rewards.append(_is_correct(answer_text, solution))
        return rewards

    cd_graph_reward = build_cd_graph_reward(
        require_dag=bool(args.cd_reward_require_dag),
        dag_penalty=float(args.cd_reward_dag_penalty),
        shd_weight=float(args.cd_reward_shd_weight),
    )
    reward_funcs = [format_reward_math, accuracy_reward_math] if args.task == "math" else []
    if args.task == "causal_discovery":
        if float(args.cd_partial_format_reward_scale) > 0.0:
            reward_funcs.append(build_cd_partial_format_reward(scale=float(args.cd_partial_format_reward_scale)))
        reward_funcs.extend([cd_format_reward, cd_graph_reward])

    if args.length_penalty_coef < 0:
        raise ValueError("--length_penalty_coef must be >= 0")
    if args.length_penalty_target_tokens < 0:
        raise ValueError("--length_penalty_target_tokens must be >= 0")
    if args.length_penalty_target_tokens != 0:
        print(
            "[warn] --length_penalty_target_tokens is ignored; "
            "length penalty now applies to total completion length."
        )

    if args.length_penalty_coef > 0.0:
        length_penalty_reward = build_length_penalty_reward(
            tokenizer=tokenizer,
            coef=float(args.length_penalty_coef),
            target_tokens=int(args.length_penalty_target_tokens),
            max_abs=float(args.length_penalty_max_abs),
        )
        reward_funcs = [*reward_funcs, length_penalty_reward]

    wrapped_reward_funcs = []

    def _make_logged_reward(inner_fn, inner_name, is_first: bool, is_last: bool):
        def _logged_reward(completions, **kwargs):
            if is_first:
                active_generation_id["id"] = generation_counter["calls"]
                generation_counter["calls"] += 1
            rewards = inner_fn(completions, **kwargs)
            _log_reward_evaluations(
                active_generation_id["id"],
                inner_name,
                completions,
                rewards,
                kwargs,
                flush=is_last,
            )
            return rewards

        _logged_reward.__name__ = inner_name
        return _logged_reward

    for idx, fn in enumerate(reward_funcs):
        reward_name = getattr(fn, "__name__", fn.__class__.__name__)
        wrapped_reward_funcs.append(
            _make_logged_reward(
                fn,
                reward_name,
                idx == 0,
                idx == (len(reward_funcs) - 1),
            )
        )
    reward_funcs = wrapped_reward_funcs

    # ---- GRPO config (TRL 0.28.0)
    report_to = ["wandb"] if args.report_to == "wandb" else "none"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    generation_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    )
    if generation_batch_size % args.num_generations != 0:
        valid_num_generations = [
            str(n)
            for n in range(1, generation_batch_size + 1)
            if generation_batch_size % n == 0
        ]
        raise ValueError(
            "Invalid GRPO settings: "
            f"generation_batch_size={generation_batch_size} "
            "(per_device_train_batch_size * gradient_accumulation_steps * WORLD_SIZE) "
            f"must be divisible by num_generations={args.num_generations}. "
            f"For this launch, choose --num_generations from {{{', '.join(valid_num_generations)}}} "
            "or adjust batch/accum/world size."
        )

    grpo_kwargs = {}
    if args.use_vllm:
        grpo_kwargs.update(
            {
                "vllm_mode": "server",
                "vllm_server_base_url": args.vllm_server_base_url,
                "vllm_server_timeout": args.vllm_server_timeout,
                "vllm_group_port": args.vllm_group_port,
            }
        )

    config_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "remove_unused_columns": False,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        # Avoid DDP autograd hook conflicts with checkpointed LoRA paths.
        "ddp_find_unused_parameters": False,
        "bf16": torch.cuda.is_available(),
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "generation_kwargs": {"stop": args.stop_sequence} if args.stop_sequence else None,
        "report_to": report_to,
        "run_name": args.run_name,
        "logging_steps": args.logging_steps,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": True,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "push_to_hub": False,
        "use_vllm": args.use_vllm,
        **grpo_kwargs,
    }
    print(f"[train] generation stop sequences: {args.stop_sequence if args.stop_sequence else 'none'}")
    if args.grpo_loss_type != "auto":
        config_kwargs["loss_type"] = args.grpo_loss_type
    if args.grpo_importance_sampling_level != "auto":
        config_kwargs["importance_sampling_level"] = args.grpo_importance_sampling_level
    training_args = GRPOConfig(**_filter_supported_kwargs(GRPOConfig.__init__, config_kwargs))

    try:
        callbacks = [CompactMetricsCallback()]
        metrics_jsonl_path = (
            Path(args.train_log_jsonl)
            if args.train_log_jsonl
            else (logs_dir / "train_metrics.jsonl")
        )
        metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(JsonlMetricsCallback(str(metrics_jsonl_path)))
        print(f"[train] JSONL logging enabled: {metrics_jsonl_path}")

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=callbacks,
        )
        trainer.remove_callback(PrinterCallback)
    except RuntimeError as e:
        msg = str(e)
        if "NCCL error: remote process exited or there was a network error" in msg:
            raise RuntimeError(
                "Failed to initialize TRL<->vLLM NCCL communicator. "
                "Most common cause is overlapping GPUs between trainer and vLLM server. "
                "Use disjoint devices (for example, server on GPU 0 and trainer on GPUs 1,2,3), "
                "and ensure --vllm_group_port is free and consistent for this run."
            ) from e
        raise

    try:
        trainer.train()
        trainer.save_model(args.output_dir)
        print("Saved to:", args.output_dir)
    finally:
        # Avoid NCCL resource-leak warnings on normal/early exits in distributed runs.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception as e:
                rank = int(os.environ.get("RANK", "0"))
                if rank == 0:
                    print(f"[warn] destroy_process_group failed: {e}")


def main():
    args = build_argparser().parse_args()
    _maybe_launch_local_vllm_and_reexec_train(args)
    if args.mode == "ab_eval":
        run_ab_eval(args)
        return
    run_train(args)


if __name__ == "__main__":
    main()
