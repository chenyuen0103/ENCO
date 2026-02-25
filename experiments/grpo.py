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
from contextlib import redirect_stdout, redirect_stderr
import torch
from pathlib import Path
from urllib import request, error
from urllib.parse import urlparse

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_callback import PrinterCallback
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
try:
    from torch.distributed.elastic.multiprocessing.errors import record as _elastic_record
except Exception:
    def _elastic_record(fn):
        return fn

from verifier_cd import (
    cd_format_reward,
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
    p = argparse.ArgumentParser(description="GRPO training with TRL + vLLM server mode")
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
    p.add_argument("--max_prompt_tokens", type=int, default=2048)
    p.add_argument("--max_completion_length", type=int, default=512)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument(
        "--sample_completions_every",
        type=int,
        default=10,
        help="Log sampled prompt/completion pairs every N reward calls (<=0 disables).",
    )
    p.add_argument(
        "--sample_completions_k",
        type=int,
        default=2,
        help="How many samples to print each completion log event.",
    )
    p.add_argument(
        "--sample_completions_max_chars",
        type=int,
        default=320,
        help="Max chars per logged prompt/completion sample.",
    )

    # Train
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
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
            "Free completion-token budget before penalty applies. "
            "Only tokens above this threshold are penalized."
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
    p.add_argument("--cd-append-format-hint", dest="cd_append_format_hint", action="store_true")
    p.add_argument("--no-cd-append-format-hint", dest="cd_append_format_hint", action="store_false")
    p.set_defaults(cd_append_format_hint=True)
    p.add_argument(
        "--cd-format-hint-text",
        type=str,
        default=(
            "Use exactly this structure: <think>...</think><answer>...</answer>. "
            "Inside <answer>, output only the final graph answer."
        ),
    )
    p.add_argument("--cd-reward-shd-weight", type=float, default=0.0, help="Weight for normalized SHD penalty.")
    p.add_argument("--cd-reward-dag-penalty", type=float, default=0.1, help="Penalty applied if predicted graph has cycles.")
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
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument(
        "--train_log_jsonl",
        type=str,
        default=None,
        help="Optional JSONL file path for per-step trainer logs (written by local rank 0 only).",
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


def build_prompt(problem: str) -> str:
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


def _sanitize_for_log(text: str, max_chars: int) -> str:
    compact = " ".join(str(text).split())
    if max_chars > 0 and len(compact) > max_chars:
        return compact[:max_chars] + "...[truncated]"
    return compact


def _print_prompt_diagnostics(dataset: Dataset, *, split_name: str):
    if "__prompt_orig_tokens" not in dataset.column_names:
        return
    total = len(dataset)
    if total == 0:
        print(f"[diag] {split_name}: empty dataset.")
        return

    orig = [int(x) for x in dataset["__prompt_orig_tokens"]]
    after = [int(x) for x in dataset["__prompt_final_tokens"]]
    truncated = [int(x) for x in dataset["__prompt_was_truncated"]]
    has_tags = [int(x) for x in dataset["__prompt_has_format_tags"]]

    trunc_count = sum(truncated)
    trunc_ratio = trunc_count / total
    fmt_ratio = sum(has_tags) / total
    print(
        "[diag] "
        f"{split_name}: prompts={total} | "
        f"orig_tokens(mean/min/max)={sum(orig)/total:.1f}/{min(orig)}/{max(orig)} | "
        f"final_tokens(mean/min/max)={sum(after)/total:.1f}/{min(after)}/{max(after)} | "
        f"truncated={trunc_count} ({trunc_ratio:.1%}) | "
        f"prompts_with_<think>_<answer>={fmt_ratio:.1%}"
    )


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
    append_format_hint: bool = False,
    format_hint_text: str = "",
) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            prompt_text = (row.get("prompt_text") or "").strip()
            if not prompt_text:
                prompt_path_raw = (row.get("prompt_path") or "").strip()
                if not prompt_path_raw:
                    raise ValueError(
                        f"{csv_path}: row {i} missing both prompt_text and prompt_path."
                    )
                prompt_path = _resolve_existing_path(prompt_path_raw, csv_path=csv_path)
                prompt_text = prompt_path.read_text(encoding="utf-8", errors="ignore")

            if wrap_system_prompt:
                prompt_text = build_prompt(prompt_text)
            if append_format_hint and format_hint_text:
                prompt_text = (
                    f"{prompt_text}\n\n"
                    f"Formatting requirement: {format_hint_text}"
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
                    "prompt": prompt_text,
                    "answer": answer_raw,
                }
            )
    return rows


def _dataset_from_cd_csvs(
    csv_paths: list[str],
    *,
    wrap_system_prompt: bool,
    append_format_hint: bool = False,
    format_hint_text: str = "",
) -> Dataset:
    datasets_list = []
    for path_str in csv_paths:
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"--cd-train/test-csv file not found: {p}")
        rows = _load_cd_rows_from_prompt_csv(
            p,
            wrap_system_prompt=wrap_system_prompt,
            append_format_hint=append_format_hint,
            format_hint_text=format_hint_text,
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
            prompt = build_prompt(example["problem"])
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
            append_format_hint=bool(args.cd_append_format_hint),
            format_hint_text=str(args.cd_format_hint_text),
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
            text = build_prompt(example["problem"])
            orig_ids = tokenizer(text, truncation=False)["input_ids"]
            ids = tokenizer(text, truncation=True, max_length=max_prompt_tokens)["input_ids"]
            return {
                "prompt": tokenizer.decode(ids, skip_special_tokens=True),
                "__prompt_orig_tokens": len(orig_ids),
                "__prompt_final_tokens": len(ids),
                "__prompt_was_truncated": int(len(orig_ids) > len(ids)),
                "__prompt_has_format_tags": int("<think>" in text and "<answer>" in text),
            }

        train_dataset = train_dataset.map(add_prompt)
        test_dataset = test_dataset.map(add_prompt)
        _print_prompt_diagnostics(train_dataset, split_name="train")
        _print_prompt_diagnostics(test_dataset, split_name="test")

        # Keep only what GRPO + reward needs
        keep_cols = ["prompt", "solution"]
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
            append_format_hint=bool(args.cd_append_format_hint),
            format_hint_text=str(args.cd_format_hint_text),
        )
        if args.cd_test_csv:
            raw_test = _dataset_from_cd_csvs(
                args.cd_test_csv,
                wrap_system_prompt=bool(args.cd_wrap_system_prompt),
                append_format_hint=bool(args.cd_append_format_hint),
                format_hint_text=str(args.cd_format_hint_text),
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
            orig_ids = tokenizer(text, truncation=False)["input_ids"]
            ids = tokenizer(text, truncation=True, max_length=max_prompt_tokens)["input_ids"]
            return {
                "prompt": tokenizer.decode(ids, skip_special_tokens=True),
                "__prompt_orig_tokens": len(orig_ids),
                "__prompt_final_tokens": len(ids),
                "__prompt_was_truncated": int(len(orig_ids) > len(ids)),
                "__prompt_has_format_tags": int("<think>" in text and "<answer>" in text),
            }

        train_dataset = raw_train.map(_truncate_cd_prompt)
        test_dataset = raw_test.map(_truncate_cd_prompt)
        _print_prompt_diagnostics(train_dataset, split_name="train")
        _print_prompt_diagnostics(test_dataset, split_name="test")
        keep_cols = ["prompt", "answer"]
        train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in keep_cols])
        test_dataset = test_dataset.remove_columns([c for c in test_dataset.column_names if c not in keep_cols])

    # ---- Model + LoRA
    distributed_world_size = int(os.environ.get("WORLD_SIZE", "1"))
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
    reward_funcs = (
        [format_reward_math, accuracy_reward_math]
        if args.task == "math"
        else [cd_format_reward, cd_graph_reward]
    )

    completion_log_counter = {"calls": 0}

    def _maybe_log_completion_samples(completions, **kwargs):
        if args.sample_completions_every <= 0:
            return
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank != 0:
            return

        completion_log_counter["calls"] += 1
        if completion_log_counter["calls"] % int(args.sample_completions_every) != 0:
            return

        prompts = kwargs.get("prompt") or []
        print(
            f"[sample] reward_call={completion_log_counter['calls']} "
            f"| batch={len(completions)} | showing={min(len(completions), args.sample_completions_k)}"
        )
        for i, completion in enumerate(completions[: max(int(args.sample_completions_k), 1)]):
            completion_text = _completion_to_text(completion)
            answer_text = _extract_answer_text(completion_text)
            prompt_text = prompts[i] if i < len(prompts) else ""
            format_ok = bool(FORMAT_RE.match(completion_text))
            print(
                f"[sample:{i}] format_ok={format_ok}\n"
                f"  prompt={_sanitize_for_log(prompt_text, int(args.sample_completions_max_chars))}\n"
                f"  completion={_sanitize_for_log(completion_text, int(args.sample_completions_max_chars))}\n"
                f"  answer={_sanitize_for_log(answer_text, int(args.sample_completions_max_chars))}"
            )

    if reward_funcs:
        first_reward = reward_funcs[0]

        def _logged_first_reward(completions, **kwargs):
            _maybe_log_completion_samples(completions, **kwargs)
            return first_reward(completions, **kwargs)

        reward_funcs = [_logged_first_reward, *reward_funcs[1:]]

    if args.length_penalty_coef < 0:
        raise ValueError("--length_penalty_coef must be >= 0")
    if args.length_penalty_target_tokens < 0:
        raise ValueError("--length_penalty_target_tokens must be >= 0")

    if args.length_penalty_coef > 0.0:
        length_penalty_reward = build_length_penalty_reward(
            tokenizer=tokenizer,
            coef=float(args.length_penalty_coef),
            target_tokens=int(args.length_penalty_target_tokens),
            max_abs=float(args.length_penalty_max_abs),
        )
        reward_funcs = [*reward_funcs, length_penalty_reward]

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

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        remove_unused_columns=False,
        num_train_epochs=args.num_train_epochs,

        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ddp_find_unused_parameters=False,

        bf16=torch.cuda.is_available(),

        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,

        report_to=report_to,
        run_name=args.run_name,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,

        save_strategy="steps",
        save_steps=args.save_steps,
        push_to_hub=False,

        # vLLM integration (server mode)
        use_vllm=args.use_vllm,
        **grpo_kwargs,
    )

    try:
        callbacks = [CompactMetricsCallback()]
        if args.train_log_jsonl:
            callbacks.append(JsonlMetricsCallback(args.train_log_jsonl))
            print(f"[train] JSONL logging enabled: {args.train_log_jsonl}")

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


@_elastic_record
def main():
    args = build_argparser().parse_args()
    _maybe_launch_local_vllm_and_reexec_train(args)
    if args.mode == "ab_eval":
        run_ab_eval(args)
        return
    run_train(args)


if __name__ == "__main__":
    main()
