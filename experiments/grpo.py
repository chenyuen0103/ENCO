import os
import re
import argparse
import json
import logging
import time
import socket
import gc
import sys
import subprocess
import csv
import io
import importlib
import inspect
import shlex
from contextlib import redirect_stdout, redirect_stderr
import torch
from pathlib import Path
from tqdm.auto import tqdm
from urllib import request, error
from urllib.parse import urlparse

try:
    from cd_training_format import (
        DEFAULT_FORMAT_HINT_TEXT,
        SYSTEM_PROMPT,
        canonicalize_cd_prompt,
        default_short_think_text,
    )
except ModuleNotFoundError:
    from experiments.cd_training_format import (
        DEFAULT_FORMAT_HINT_TEXT,
        SYSTEM_PROMPT,
        canonicalize_cd_prompt,
        default_short_think_text,
    )

try:
    from torch.nn.parallel import DistributedDataParallel as _DDP
except Exception:
    _DDP = None

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_callback import PrinterCallback
from peft import LoraConfig, get_peft_model
try:
    from torch.distributed.elastic.multiprocessing.errors import record as _elastic_record
except Exception:
    def _elastic_record(fn):
        return fn

from verifier_cd import (
    build_cd_format_reward,
    build_cd_partial_format_reward,
    build_cd_descendant_partial_format_reward,
    build_cd_descendant_answer_tail_partial_format_reward,
    build_cd_graph_reward,
    build_cd_descendant_f1_reward,
    build_cd_edge_f1_reward,
    build_cd_low_shd_reward,
    build_cd_acyclic_reward,
    build_length_penalty_reward,
    completion_to_text as cd_completion_to_text,
    extract_answer_text as cd_extract_answer_text,
    format_ok as cd_format_ok,
    score_cd_completion,
    score_cd_descendants_completion,
)

try:
    from math_verify import LatexExtractionConfig, parse, verify
    _HAS_MATH_VERIFY = True
except Exception:
    LatexExtractionConfig = None
    parse = None
    verify = None
    _HAS_MATH_VERIFY = False

FORMAT_RE = re.compile(r"(?s)^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$")
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def _import_trl_grpo():
    try:
        from trl import GRPOConfig, GRPOTrainer
        return GRPOConfig, GRPOTrainer
    except Exception as e:
        msg = str(e)
        # Some TRL installs try to initialize vLLM during import and fail on
        # systems where NVML/CUDA visibility is unstable. Retry with vLLM disabled.
        if not any(token in msg for token in ("GuidedDecodingParams", "vllm", "NVMLError", "NVML")):
            raise
        try:
            trl_import_utils = importlib.import_module("trl.import_utils")
            setattr(trl_import_utils, "_vllm_available", False)
            from trl import GRPOConfig, GRPOTrainer
            return GRPOConfig, GRPOTrainer
        except Exception:
            raise e


def _load_graph_num_nodes(graph_path: str) -> int | None:
    try:
        from causal_graphs.graph_real_world import load_graph_file  # type: ignore
    except Exception:
        return None
    try:
        graph = load_graph_file(str(graph_path))
    except Exception:
        return None
    variables = getattr(graph, "variables", None)
    if variables is not None:
        try:
            return len(variables)
        except Exception:
            pass
    adj_matrix = getattr(graph, "adj_matrix", None)
    if adj_matrix is not None:
        try:
            return len(adj_matrix)
        except Exception:
            pass
    return None


def _argv_has_flag(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)


def _to_jsonable_config(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable_config(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable_config(v) for k, v in value.items()}
    if hasattr(value, "item"):
        try:
            return _to_jsonable_config(value.item())
        except Exception:
            pass
    return str(value)


def _maybe_enable_small_graph_logging(args, argv: list[str]) -> None:
    if args.task not in {"causal_discovery", "cd_descendants"}:
        return
    if not getattr(args, "cd_bif_file", None):
        return

    num_nodes = _load_graph_num_nodes(args.cd_bif_file)
    if num_nodes is None or num_nodes >= 6:
        return

    explicit_eval_logging = (
        _argv_has_flag(argv, "--save-eval-responses")
        or _argv_has_flag(argv, "--no-save-eval-responses")
    )
    explicit_sample_logging = _argv_has_flag(argv, "--sample_completions_every")

    if not explicit_eval_logging:
        args.save_eval_responses = True
    if not explicit_sample_logging:
        args.sample_completions_every = 1

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(
            f"[log] auto-enabled prompt/response saving for small graph "
            f"({num_nodes} nodes, threshold < 6)"
        )


def _write_training_config_to_model_card(output_dir: str, args, argv: list[str]) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return

    readme_path = Path(output_dir) / "README.md"
    if not readme_path.exists():
        return

    marker = "## Training Config"
    command_text = " ".join(shlex.quote(part) for part in ([sys.executable] + list(argv)))
    args_json = json.dumps(_to_jsonable_config(vars(args)), indent=2, sort_keys=True)
    section = (
        f"\n\n{marker}\n\n"
        "Saved automatically from the GRPO training launch.\n\n"
        "### Command\n\n"
        "```bash\n"
        f"{command_text}\n"
        "```\n\n"
        "### Parsed Arguments\n\n"
        "```json\n"
        f"{args_json}\n"
        "```\n"
    )

    text = readme_path.read_text(encoding="utf-8")
    if marker in text:
        text = text.split(marker, 1)[0].rstrip()
    readme_path.write_text(text + section, encoding="utf-8")


def _patch_ddp_config_attr() -> None:
    """
    Compatibility shim:
    some Unsloth/TRL code paths access model.config directly, but under DDP the
    wrapped model is at model.module.config.
    """
    if _DDP is None:
        return
    if hasattr(_DDP, "config"):
        return

    def _get_config(self):
        module = getattr(self, "module", None)
        return getattr(module, "config", None)

    _DDP.config = property(_get_config)


def _suppress_transformers_attn_mask_warning_bug() -> None:
    # Suppress noisy malformed warning formatting from Transformers in long runs.
    logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)


def _set_csv_field_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10_000_000)


class CompactMetricsCallback(TrainerCallback):
    """Print a compact metric line instead of full raw log dictionaries."""

    _KEYS = [
        ("epoch", "epoch"),
        ("loss", "loss"),
        ("reward", "reward"),
        ("rewards/accuracy_reward/mean", "acc"),
        ("rewards/cd_partial_format_reward/mean", "pfmt"),
        ("rewards/cd_descendant_partial_format_reward/mean", "dpfmt"),
        ("rewards/cd_descendant_answer_tail_partial_format_reward/mean", "dpfmt"),
        ("rewards/cd_descendant_f1_reward/mean", "desc"),
        ("rewards/cd_graph_reward/mean", "cd"),
        ("rewards/cd_edge_f1_reward/mean", "f1"),
        ("rewards/cd_low_shd_reward/mean", "lshd"),
        ("rewards/cd_acyclic_reward/mean", "dag"),
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


class TrainStateSnapshotCallback(TrainerCallback):
    """Keep the latest trainer step/epoch in a mutable dict for auxiliary logs."""

    def __init__(self, state_ref: dict):
        self.state_ref = state_ref

    def on_step_end(self, args, state, control, **kwargs):
        self.state_ref["global_step"] = int(state.global_step)
        self.state_ref["epoch"] = float(state.epoch) if state.epoch is not None else None

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.state_ref["global_step"] = int(state.global_step)
        self.state_ref["epoch"] = float(state.epoch) if state.epoch is not None else None


def build_argparser():
    p = argparse.ArgumentParser(description="GRPO training with TRL + vLLM server mode")
    p.add_argument("--mode", type=str, default="train", choices=["train", "eval", "export_cd_csv"])
    p.add_argument(
        "--task",
        type=str,
        default="causal_discovery",
        choices=["causal_discovery", "cd_descendants", "math"],
        help="Training/eval task type. causal_discovery and cd_descendants expect prompt CSV inputs.",
    )
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    p.add_argument("--dataset_id", type=str, default="AI-MO/NuminaMath-TIR")
    p.add_argument("--train_split", type=str, default="train[:5%]")
    p.add_argument("--test_split", type=str, default="test[:5%]")
    p.add_argument("--output_dir", type=str, default="Qwen2-0.5B-GRPO-vLLM-server")

    # Prompt / generation
    p.add_argument("--max_prompt_tokens", type=int, default=256000)
    p.add_argument("--max_completion_length", type=int, default=8192)
    p.add_argument(
        "--train_temperature",
        type=float,
        default=1.0,
        help="Training rollout temperature for GRPO generation.",
    )
    p.add_argument(
        "--train_top_p",
        type=float,
        default=1.0,
        help="Training rollout top-p for GRPO generation.",
    )
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
    p.add_argument(
        "--sample_completions_every",
        type=int,
        default=0,
        help="Enable sampled prompt/completion logging (<=0 disables). Default is auto-enabled only for graphs with fewer than 6 nodes. When enabled, rows are emitted every save_steps trainer steps.",
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
        "--grpo-beta",
        type=float,
        default=0.02,
        help=(
            "KL anchor strength for GRPO relative to the reference policy. "
            "Use a small positive value to keep rollouts closer to the SFT initializer."
        ),
    )
    p.add_argument(
        "--length_penalty_coef",
        type=float,
        default=0.0000333333,
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
        "--cd-config-file",
        type=str,
        default=None,
        help=(
            "Optional in-memory causal-discovery config JSON (same schema as run_experiment1_in_memory). "
            "When set, prompts are generated in-memory and --cd-train-csv/--cd-test-csv are ignored."
        ),
    )
    p.add_argument(
        "--cd-bif-file",
        type=str,
        default=None,
        help="BIF graph path required when --cd-config-file is used.",
    )
    p.add_argument(
        "--cd-config-num-prompts",
        type=int,
        default=5,
        help="Number of prompts per config row for --cd-config-file mode.",
    )
    p.add_argument(
        "--cd-config-seed",
        type=int,
        default=0,
        help="Random seed for in-memory prompt generation in --cd-config-file mode.",
    )
    p.add_argument(
        "--cd-config-causal-rules",
        action="store_true",
        help="Enable causal-rules text in prompts generated from --cd-config-file.",
    )
    p.add_argument(
        "--cd-config-give-steps",
        action="store_true",
        help="Enable step-by-step instruction text in prompts generated from --cd-config-file.",
    )
    p.add_argument(
        "--cd-config-def-int",
        action="store_true",
        help="Include intervention definitions in prompts generated from --cd-config-file.",
    )
    p.add_argument(
        "--cd-config-intervene-vars",
        type=str,
        default="all",
        help="Intervention variable mode for --cd-config-file generation (default: all).",
    )
    p.add_argument(
        "--cd-config-thinking-tags",
        dest="cd_config_thinking_tags",
        action="store_true",
        help="Include thinking-tags instruction in prompts generated from --cd-config-file (default).",
    )
    p.add_argument(
        "--no-cd-config-thinking-tags",
        dest="cd_config_thinking_tags",
        action="store_false",
        help="Disable thinking-tags instruction in prompts generated from --cd-config-file.",
    )
    p.set_defaults(cd_config_thinking_tags=True)
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
    p.add_argument("--cd-append-format-hint", dest="cd_append_format_hint", action="store_true")
    p.add_argument("--no-cd-append-format-hint", dest="cd_append_format_hint", action="store_false")
    p.set_defaults(cd_append_format_hint=True)
    p.add_argument(
        "--cd-format-hint-text",
        type=str,
        default=DEFAULT_FORMAT_HINT_TEXT,
    )
    p.add_argument(
        "--cd-grpo-prefill-answer",
        dest="cd_grpo_prefill_answer",
        action="store_true",
        help=(
            "For cd_descendants training rollouts, prefill the assistant through the short think "
            "trace and opening <answer>, so GRPO only generates the JSON payload and closing tag."
        ),
    )
    p.add_argument(
        "--no-cd-grpo-prefill-answer",
        dest="cd_grpo_prefill_answer",
        action="store_false",
    )
    p.set_defaults(cd_grpo_prefill_answer=True)
    p.add_argument("--cd-reward-shd-weight", type=float, default=0.0, help="Weight for normalized SHD penalty.")
    p.add_argument("--cd-reward-dag-penalty", type=float, default=0.1, help="Penalty applied if predicted graph has cycles.")
    p.add_argument(
        "--cd-graph-reward-scale",
        type=float,
        default=1.0,
        help="Scale for causal-discovery graph reward (1.0 keeps current behavior).",
    )
    p.add_argument(
        "--cd-format-reward-scale",
        type=float,
        default=0.2,
        help="Scale for strict causal-discovery format reward (0 disables).",
    )
    p.add_argument(
        "--cd-partial-format-reward-scale",
        type=float,
        default=0.25,
        help="Scale for dense partial-format shaping reward (0 disables).",
    )
    p.add_argument(
        "--cd-edge-f1-reward-scale",
        type=float,
        default=0.0,
        help="Optional separate edge-F1 reward scale (0 disables).",
    )
    p.add_argument(
        "--cd-low-shd-reward-scale",
        type=float,
        default=0.0,
        help="Optional separate low-SHD reward scale using (1 - normalized_shd).",
    )
    p.add_argument(
        "--cd-acyclic-reward-scale",
        type=float,
        default=0.0,
        help="Optional separate acyclicity reward scale (0 disables).",
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
        help="Persist every reward-evaluated completion to output_dir/grpo_log as JSONL. Default is auto-enabled only for graphs with fewer than 6 nodes.",
    )
    p.add_argument("--no-save-eval-responses", dest="save_eval_responses", action="store_false")
    p.set_defaults(save_eval_responses=False)
    p.add_argument(
        "--eval-responses-max-chars",
        type=int,
        default=0,
        help="Optional max chars for prompt/completion/answer fields in eval-response logs (0 = no truncation).",
    )

    # Evaluation on a frozen set
    p.add_argument("--eval_split", type=str, default="test")
    p.add_argument("--eval_n", type=int, default=200)
    p.add_argument("--eval_seed", type=int, default=1234)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--eval_pass_k", type=int, default=1)
    p.add_argument("--eval_max_new_tokens", type=int, default=128)
    p.add_argument("--eval_temperature", type=float, default=0.0)
    p.add_argument("--eval_top_p", type=float, default=0.95)
    p.add_argument("--eval_do_sample", action="store_true")
    p.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    p.add_argument("--no-enable-thinking", dest="enable_thinking", action="store_false")
    p.set_defaults(enable_thinking=False)
    p.add_argument("--eval_debug_csv", type=str, default=None)
    p.add_argument("--eval_model", type=str, default=None)
    p.add_argument("--eval_output_json", type=str, default=None)
    p.add_argument(
        "--export_csv",
        type=str,
        default=None,
        help="Output CSV path for --mode export_cd_csv.",
    )
    p.add_argument(
        "--export_limit",
        type=int,
        default=0,
        help="Optional max number of exported rows for --mode export_cd_csv (0 = no cap).",
    )

    return p


def build_prompt(problem: str, tokenizer: AutoTokenizer | None = None, enable_thinking: bool = False) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(problem)},
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            params = inspect.signature(tokenizer.apply_chat_template).parameters
            if enable_thinking and "enable_thinking" in params:
                kwargs["enable_thinking"] = True
            return str(tokenizer.apply_chat_template(messages, **kwargs))
        except Exception:
            try:
                kwargs.pop("enable_thinking", None)
                return str(tokenizer.apply_chat_template(messages, **kwargs))
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
            "Install math_verify or switch to --task causal_discovery or --task cd_descendants."
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
    prefill_answer: bool = False,
    think_text: str = "",
) -> list[dict]:
    rows: list[dict] = []
    _set_csv_field_limit()
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
            prompt_text = canonicalize_cd_prompt(
                prompt_text,
                wrap_system_prompt=bool(wrap_system_prompt),
                append_format_hint=bool(append_format_hint),
                format_hint_text=str(format_hint_text),
                prefill_think=not bool(prefill_answer),
                prefill_answer=bool(prefill_answer),
                think_text=str(think_text),
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
    append_format_hint: bool = False,
    format_hint_text: str = "",
    prefill_answer: bool = False,
    think_text: str = "",
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
            prefill_answer=prefill_answer,
            think_text=think_text,
        )
        if not rows:
            raise ValueError(f"No usable rows found in {p}")
        datasets_list.append(Dataset.from_list(rows))

    if not datasets_list:
        raise ValueError("No causal discovery CSV inputs were provided.")
    if len(datasets_list) == 1:
        return datasets_list[0]
    return concatenate_datasets(datasets_list)


def _dataset_from_cd_config_file(
    *,
    config_file: str,
    bif_file: str,
    num_prompts: int,
    seed: int,
    task: str,
    wrap_system_prompt: bool,
    append_format_hint: bool = False,
    format_hint_text: str = "",
    prefill_answer: bool = False,
    think_text: str = "",
    causal_rules: bool = False,
    give_steps: bool = False,
    def_int: bool = False,
    intervene_vars: str = "all",
    thinking_tags: bool = True,
) -> Dataset:
    try:
        from run_experiment1_in_memory import (  # type: ignore
            _load_configs_from_file as _load_cfgs,
            _iter_prompts_for_config as _iter_cfg_prompts,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to import in-memory prompt helpers from run_experiment1_in_memory.py."
        ) from e

    style_aliases = {"summary_join": "summary_joint"}
    all_styles = ["cases", "matrix", "summary", "summary_joint", "summary_probs", "payload", "payload_topk"]
    allowed_row_orders = {"random", "sorted", "reverse"}
    allowed_col_orders = {"original", "reverse", "random", "topo", "reverse_topo"}

    cfg_path = Path(config_file)
    if not cfg_path.exists():
        raise FileNotFoundError(f"--cd-config-file not found: {cfg_path}")
    bif_path = Path(bif_file)
    if not bif_path.exists():
        raise FileNotFoundError(f"--cd-bif-file not found: {bif_path}")

    configs = _load_cfgs(
        config_file=cfg_path,
        style_aliases=style_aliases,
        allowed_styles=set(all_styles),
        allowed_row_orders=allowed_row_orders,
        allowed_col_orders=allowed_col_orders,
    )

    def _descendants_from_adj(adj: list[list[int]], src_idx: int) -> list[int]:
        seen = set()
        stack = [src_idx]
        while stack:
            u = stack.pop()
            for v, has_edge in enumerate(adj[u]):
                if int(has_edge) != 1 or v in seen or v == src_idx:
                    continue
                seen.add(v)
                stack.append(v)
        return sorted(seen)

    rows: list[dict[str, str]] = []
    for style, anon, obs_n, int_n, row_ord, col_ord, shuf_n in configs:
        if obs_n == 0 and int_n == 0 and int(shuf_n) != 1:
            continue
        _base_name, answer_obj, prompt_iter = _iter_cfg_prompts(
            bif_file=str(bif_path),
            num_prompts=int(num_prompts),
            shuffles_per_graph=int(shuf_n),
            seed=int(seed),
            prompt_style=style,
            obs_per_prompt=int(obs_n),
            int_per_combo=int(int_n),
            row_order=row_ord,
            col_order=col_ord,
            anonymize=bool(anon),
            causal_rules=bool(causal_rules),
            give_steps=bool(give_steps),
            def_int=bool(def_int),
            intervene_vars=str(intervene_vars),
            thinking_tags=bool(thinking_tags),
        )
        answer_raw = json.dumps(answer_obj, ensure_ascii=False)
        adj = answer_obj.get("adjacency_matrix") if isinstance(answer_obj, dict) else None
        variables_out = answer_obj.get("variables") if isinstance(answer_obj, dict) else None
        descendants_map: dict[str, list[str]] = {}
        if (
            task == "cd_descendants"
            and isinstance(adj, list)
            and isinstance(variables_out, list)
        ):
            for idx, name in enumerate(variables_out):
                descendants_map[str(name)] = [str(variables_out[j]) for j in _descendants_from_adj(adj, idx)]
        for item in prompt_iter:
            if task == "cd_descendants":
                try:
                    from generate_prompts import format_prompt_descendants_summary  # type: ignore
                except Exception as e:
                    raise RuntimeError(
                        "Failed to import format_prompt_descendants_summary from generate_prompts.py."
                    ) from e

                item_variables = item.get("variables") or variables_out or []
                int_groups_num = item.get("int_groups_num") or {}
                obs_rows_num = item.get("obs_rows_num") or []
                state_names = item.get("state_names") or None
                dataset_name = str(item.get("dataset_name") or bif_path.stem)
                if not item_variables or not int_groups_num:
                    continue

                for (ivar, ival), intervention_rows_num in sorted(
                    int_groups_num.items(),
                    key=lambda kv: (str(kv[0][0]), str(kv[0][1])),
                ):
                    prompt_raw = format_prompt_descendants_summary(
                        list(item_variables),
                        dataset_name=dataset_name,
                        intervention_target=str(ivar),
                        intervention_value=str(ival),
                        intervention_rows_num=list(intervention_rows_num),
                        obs_rows_num=list(obs_rows_num),
                        state_names=state_names,
                        include_causal_rules=bool(causal_rules),
                        include_def_int=bool(def_int),
                        anonymize=bool(item.get("anonymize", anon)),
                    )
                    prompt_text = prompt_raw
                    prompt_text = canonicalize_cd_prompt(
                        prompt_text,
                        wrap_system_prompt=bool(wrap_system_prompt),
                        append_format_hint=bool(append_format_hint),
                        format_hint_text=str(format_hint_text),
                        prefill_think=not bool(prefill_answer),
                        prefill_answer=bool(prefill_answer),
                        think_text=str(think_text),
                    )
                    rows.append(
                        {
                            "prompt_raw": prompt_raw,
                            "prompt": prompt_text,
                            "answer": json.dumps(
                                {
                                    "target": str(ivar),
                                    "descendants": descendants_map.get(str(ivar), []),
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                continue

            prompt_raw = str(item.get("prompt_text", ""))
            prompt_text = prompt_raw
            prompt_text = canonicalize_cd_prompt(
                prompt_text,
                wrap_system_prompt=bool(wrap_system_prompt),
                append_format_hint=bool(append_format_hint),
                format_hint_text=str(format_hint_text),
                prefill_think=not bool(prefill_answer),
                prefill_answer=bool(prefill_answer),
                think_text=str(think_text),
            )
            rows.append(
                {
                    "prompt_raw": prompt_raw,
                    "prompt": prompt_text,
                    "answer": answer_raw,
                }
            )

    if not rows:
        raise ValueError(
            f"No in-memory rows generated for task={task} from config={cfg_path} and bif={bif_path}."
        )
    return Dataset.from_list(rows)


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

        # Avoid Accelerate's balanced-memory auto-sharding path for PEFT eval loads.
        # In this environment it can fail inside get_balanced_memory() with
        # "TypeError: unhashable type: 'set'". For eval we can place the adapter
        # model on a single CUDA device when available.
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id_or_path,
            dtype="auto",
        ).eval()
        if torch.cuda.is_available():
            model = model.to(f"cuda:{torch.cuda.current_device()}")
        _apply_explicit_generation_config(model)
        return model

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        dtype="auto",
        device_map="auto",
    ).eval()
    _apply_explicit_generation_config(model)
    return model


def _apply_explicit_generation_config(
    model,
    *,
    max_prompt_tokens: int | None = None,
    max_completion_length: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
):
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        return

    if max_prompt_tokens is not None and max_completion_length is not None:
        try:
            gen_cfg.max_length = int(max_prompt_tokens) + int(max_completion_length)
        except Exception:
            pass
    if temperature is not None:
        try:
            gen_cfg.temperature = float(temperature)
        except Exception:
            pass
    if top_p is not None:
        try:
            gen_cfg.top_p = float(top_p)
        except Exception:
            pass


def _wrap_generate_with_eval_mode(model):
    if getattr(model, "_grpo_eval_generate_wrapped", False):
        return model

    original_generate = getattr(model, "generate", None)
    if original_generate is None:
        return model

    def _generate_with_eval_mode(*args, **kwargs):
        was_training = bool(getattr(model, "training", False))
        if was_training:
            model.eval()
        try:
            return original_generate(*args, **kwargs)
        finally:
            if was_training:
                model.train()

    setattr(model, "generate", _generate_with_eval_mode)
    setattr(model, "_grpo_eval_generate_wrapped", True)
    return model


def _write_eval_snapshot(output_json_path: str, payload: dict) -> None:
    if not output_json_path:
        return
    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(out_path)


def _eval_on_dataset(
    model_id_or_path: str,
    eval_dataset,
    task: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    batch_size: int,
    pass_k: int,
    debug_model_label: str = None,
    progress_desc: str = "eval",
    progress_output_json: str = None,
    progress_payload_base: dict | None = None,
    enable_thinking: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = _load_eval_model(model_id_or_path)
    input_device = next(model.parameters()).device

    fmt_ok = 0
    acc_sum = 0.0
    pass_k_ok = 0.0
    verify_timeouts = 0
    verify_errors = 0
    debug_rows = []
    sample_rows = []

    def _build_prompt_for_eval(example):
        if task == "math":
            problem_text = str(example["problem"])
            return build_prompt(
                problem_text,
                tokenizer=tokenizer,
                enable_thinking=bool(enable_thinking),
            )
        return str(example["prompt"])

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": pass_k,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    progress_bar = tqdm(total=len(eval_dataset), desc=progress_desc, unit="ex")
    batch_size = max(1, int(batch_size))
    for batch_start in range(0, len(eval_dataset), batch_size):
        batch_examples = [eval_dataset[i] for i in range(batch_start, min(batch_start + batch_size, len(eval_dataset)))]
        prompts = [_build_prompt_for_eval(example) for example in batch_examples]
        tokenization_kwargs = {
            "return_tensors": "pt",
            "padding": True,
        }
        if max_prompt_tokens and int(max_prompt_tokens) > 0:
            tokenization_kwargs.update(
                {
                    "truncation": True,
                    "max_length": int(max_prompt_tokens),
                }
            )
        inputs = tokenizer(prompts, **tokenization_kwargs).to(input_device)
        padded_input_len = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            out = model.generate(
                **inputs,
                **generation_kwargs,
            )

        for batch_idx, example in enumerate(batch_examples):
            idx = batch_start + batch_idx
            start = batch_idx * pass_k
            end = start + pass_k
            seqs = out[start:end]
            completions = [
                tokenizer.decode(seq[padded_input_len:], skip_special_tokens=True)
                for seq in seqs
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
                elif task == "cd_descendants":
                    answer_text = cd_extract_answer_text(cd_completion_to_text(completion_text))
                    meta = score_cd_descendants_completion(
                        completion_text=completion_text,
                        target_answer=example["answer"],
                    )
                    score = float(meta["reward"])
                    timed_out = False
                    had_error = False
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

                sample_row = {
                    "model": model_id_or_path,
                    "example_idx": idx,
                    "sample_idx": sample_idx,
                    "problem": prompts[batch_idx],
                    "solution": example.get("solution", example.get("answer", "")),
                    "completion": completion_text,
                    "answer_text": answer_text,
                    "format_ok": int(bool(FORMAT_RE.match(completion_text))) if task == "math" else int(cd_format_ok(completion_text)),
                    "reward": float(score),
                    "is_correct": float(score),
                    "verify_timed_out": int(timed_out),
                    "verify_error": int(had_error),
                }
                sample_rows.append(sample_row)

                if debug_model_label is not None:
                    debug_row = dict(sample_row)
                    debug_row["model"] = debug_model_label
                    debug_rows.append(debug_row)

            acc_sum += sample_scores[0]
            pass_k_ok += float(max(sample_scores))
            progress_bar.update(1)
        completed = min(batch_start + len(batch_examples), len(eval_dataset))
        running_metrics = {
            "model": model_id_or_path,
            "n_completed": completed,
            "n_total": len(eval_dataset),
            "format_rate": fmt_ok / completed,
            "accuracy": acc_sum / completed,
            f"pass@{pass_k}": pass_k_ok / completed,
            "verify_timeouts": verify_timeouts,
            "verify_errors": verify_errors,
        }
        progress_bar.set_postfix(
            acc=f"{running_metrics['accuracy']:.4f}",
            fmt=f"{running_metrics['format_rate']:.4f}",
        )
        if progress_output_json:
            snapshot = dict(progress_payload_base or {})
            snapshot["status"] = "running"
            snapshot["completed"] = completed
            snapshot["total"] = len(eval_dataset)
            snapshot["eval"] = running_metrics
            snapshot["samples"] = sample_rows
            _write_eval_snapshot(progress_output_json, snapshot)

    progress_bar.close()

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
    return metrics, debug_rows, sample_rows


def _build_frozen_eval_dataset(args):
    if args.task == "math":
        full_eval = load_dataset(args.dataset_id, split=args.eval_split)
        n = min(args.eval_n, len(full_eval))
        frozen_eval = full_eval.shuffle(seed=args.eval_seed).select(range(n))
        return frozen_eval, n

    if args.cd_config_file:
        if not args.cd_bif_file:
            raise ValueError("--cd-bif-file is required when --cd-config-file is set.")
        full_eval = _dataset_from_cd_config_file(
            config_file=args.cd_config_file,
            bif_file=args.cd_bif_file,
            num_prompts=int(args.cd_config_num_prompts),
            seed=int(args.cd_config_seed),
            task=str(args.task),
            wrap_system_prompt=bool(args.cd_wrap_system_prompt),
            append_format_hint=bool(args.cd_append_format_hint),
            format_hint_text=str(args.cd_format_hint_text),
            causal_rules=bool(args.cd_config_causal_rules),
            give_steps=bool(args.cd_config_give_steps),
            def_int=bool(args.cd_config_def_int),
            intervene_vars=str(args.cd_config_intervene_vars),
            thinking_tags=bool(args.cd_config_thinking_tags),
        )
    else:
        eval_sources = args.cd_test_csv if args.cd_test_csv else args.cd_train_csv
        if not eval_sources:
            raise ValueError(
                f"For --task {args.task}, pass --cd-config-file + --cd-bif-file "
                "or --cd-train-csv (and optionally --cd-test-csv)."
            )
        full_eval = _dataset_from_cd_csvs(
            eval_sources,
            wrap_system_prompt=bool(args.cd_wrap_system_prompt),
            append_format_hint=bool(args.cd_append_format_hint),
            format_hint_text=str(args.cd_format_hint_text),
        )

    n = min(args.eval_n, len(full_eval))
    frozen_eval = full_eval.shuffle(seed=args.eval_seed).select(range(n))
    return frozen_eval, n


def run_eval(args):
    if not args.eval_model:
        raise ValueError("--eval_model is required when --mode eval")
    if args.eval_n <= 0:
        raise ValueError("--eval_n must be > 0")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval_batch_size must be > 0")
    if args.eval_pass_k <= 0:
        raise ValueError("--eval_pass_k must be > 0")

    frozen_eval, n = _build_frozen_eval_dataset(args)

    do_sample = args.eval_do_sample or args.eval_temperature > 0 or args.eval_pass_k > 1
    if args.eval_pass_k > 1 and not do_sample:
        raise ValueError("pass@k with k>1 requires sampling; set --eval_do_sample or --eval_temperature > 0")

    base_result = {
        "dataset_id": args.dataset_id,
        "split": args.eval_split,
        "seed": args.eval_seed,
        "n": n,
        "generation": {
            "max_new_tokens": args.eval_max_new_tokens,
            "do_sample": do_sample,
            "temperature": args.eval_temperature if do_sample else 0.0,
            "top_p": args.eval_top_p if do_sample else 1.0,
            "pass_k": args.eval_pass_k,
        },
    }

    metrics, debug_rows, sample_rows = _eval_on_dataset(
        model_id_or_path=args.eval_model,
        eval_dataset=frozen_eval,
        task=args.task,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.eval_max_new_tokens,
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        do_sample=do_sample,
        batch_size=args.eval_batch_size,
        pass_k=args.eval_pass_k,
        debug_model_label="eval" if args.eval_debug_csv else None,
        progress_desc=f"eval:{Path(args.eval_model).name}",
        progress_output_json=args.eval_output_json,
        progress_payload_base=base_result,
        enable_thinking=bool(args.enable_thinking),
    )

    result = dict(base_result)
    result["status"] = "completed"
    result["completed"] = n
    result["total"] = n
    result["eval"] = metrics
    result["samples"] = sample_rows

    print(json.dumps(result, indent=2))
    if args.eval_output_json:
        _write_eval_snapshot(args.eval_output_json, result)
        print("Saved eval JSON to:", args.eval_output_json)
    if args.eval_debug_csv:
        fieldnames = [
            "model",
            "example_idx",
            "sample_idx",
            "problem",
            "solution",
            "completion",
            "answer_text",
            "format_ok",
            "reward",
            "is_correct",
            "verify_timed_out",
            "verify_error",
        ]
        with open(args.eval_debug_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(debug_rows)
        print("Saved debug CSV to:", args.eval_debug_csv)


def run_export_cd_csv(args) -> None:
    if args.task not in {"causal_discovery", "cd_descendants"}:
        raise ValueError("--mode export_cd_csv only supports --task causal_discovery or cd_descendants.")
    if not args.export_csv:
        raise ValueError("--export_csv is required when --mode export_cd_csv.")

    if args.cd_config_file and args.cd_train_csv:
        raise ValueError(
            "Use either --cd-config-file (in-memory generation) or --cd-train-csv, not both."
        )

    if args.cd_config_file:
        if not args.cd_bif_file:
            raise ValueError("--cd-bif-file is required when --cd-config-file is set.")
        dataset = _dataset_from_cd_config_file(
            config_file=args.cd_config_file,
            bif_file=args.cd_bif_file,
            num_prompts=int(args.cd_config_num_prompts),
            seed=int(args.cd_config_seed),
            task=str(args.task),
            wrap_system_prompt=bool(args.cd_wrap_system_prompt),
            append_format_hint=bool(args.cd_append_format_hint),
            format_hint_text=str(args.cd_format_hint_text),
            causal_rules=bool(args.cd_config_causal_rules),
            give_steps=bool(args.cd_config_give_steps),
            def_int=bool(args.cd_config_def_int),
            intervene_vars=str(args.cd_config_intervene_vars),
            thinking_tags=bool(args.cd_config_thinking_tags),
        )
    else:
        if not args.cd_train_csv:
            raise ValueError(
                f"For --task {args.task}, pass --cd-config-file + --cd-bif-file "
                "or --cd-train-csv."
            )
        dataset = _dataset_from_cd_csvs(
            args.cd_train_csv,
            wrap_system_prompt=bool(args.cd_wrap_system_prompt),
            append_format_hint=bool(args.cd_append_format_hint),
            format_hint_text=str(args.cd_format_hint_text),
        )

    if args.export_limit and args.export_limit > 0:
        dataset = dataset.select(range(min(int(args.export_limit), len(dataset))))

    out_path = Path(args.export_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = dataset.to_list()
    fieldnames = list(dataset.column_names)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(
        json.dumps(
            {
                "mode": "export_cd_csv",
                "task": args.task,
                "rows": len(rows),
                "columns": fieldnames,
                "output_csv": str(out_path.resolve()),
            },
            indent=2,
        )
    )


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


def run_train(args, argv: list[str] | None = None):
    argv = list(argv or [])
    GRPOConfig, GRPOTrainer = _import_trl_grpo()
    _patch_ddp_config_attr()
    _suppress_transformers_attn_mask_warning_bug()

    args.vllm_server_base_url = args.vllm_server_base_url.rstrip("/")
    if args.max_prompt_tokens <= 0:
        args.max_prompt_tokens = 2048
        print("[warn] max_prompt_tokens <= 0; using safer default 2048.")
    if args.max_completion_length <= 0:
        args.max_completion_length = 512
        print("[warn] max_completion_length <= 0; using safer default 512.")
    if args.train_temperature < 0:
        raise ValueError("--train_temperature must be >= 0.")
    if not (0 < args.train_top_p <= 1.0):
        raise ValueError("--train_top_p must be in (0, 1].")
    if args.grpo_beta < 0:
        raise ValueError("--grpo-beta must be >= 0.")
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
    tokenizer.padding_side = "left"

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
        keep_cols = ["prompt", "prompt_raw", "solution"]
        train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in keep_cols])
        test_dataset = test_dataset.remove_columns([c for c in test_dataset.column_names if c not in keep_cols])
    else:
        rollout_prefill_answer = bool(args.task == "cd_descendants" and args.cd_grpo_prefill_answer)
        rollout_think_text = default_short_think_text("cd_descendants") if rollout_prefill_answer else ""
        if args.cd_config_file and args.cd_train_csv:
            raise ValueError(
                "Use either --cd-config-file (in-memory generation) or --cd-train-csv, not both."
            )

        if args.cd_config_file:
            if not args.cd_bif_file:
                raise ValueError("--cd-bif-file is required when --cd-config-file is set.")
            raw_train = _dataset_from_cd_config_file(
                config_file=args.cd_config_file,
                bif_file=args.cd_bif_file,
                num_prompts=int(args.cd_config_num_prompts),
                seed=int(args.cd_config_seed),
                task=str(args.task),
                wrap_system_prompt=bool(args.cd_wrap_system_prompt),
                append_format_hint=bool(args.cd_append_format_hint),
                format_hint_text=str(args.cd_format_hint_text),
                prefill_answer=rollout_prefill_answer,
                think_text=rollout_think_text,
                causal_rules=bool(args.cd_config_causal_rules),
                give_steps=bool(args.cd_config_give_steps),
                def_int=bool(args.cd_config_def_int),
                intervene_vars=str(args.cd_config_intervene_vars),
                thinking_tags=bool(args.cd_config_thinking_tags),
            )
            raw_test = None
        else:
            if not args.cd_train_csv:
                raise ValueError(
                    f"--task {args.task} requires --cd-config-file + --cd-bif-file "
                    "or --cd-train-csv (repeatable) pointing to prompt CSVs."
                )

            raw_train = _dataset_from_cd_csvs(
                args.cd_train_csv,
                wrap_system_prompt=bool(args.cd_wrap_system_prompt),
                append_format_hint=bool(args.cd_append_format_hint),
                format_hint_text=str(args.cd_format_hint_text),
                prefill_answer=rollout_prefill_answer,
                think_text=rollout_think_text,
            )
            if args.cd_test_csv:
                raw_test = _dataset_from_cd_csvs(
                    args.cd_test_csv,
                    wrap_system_prompt=bool(args.cd_wrap_system_prompt),
                    append_format_hint=bool(args.cd_append_format_hint),
                    format_hint_text=str(args.cd_format_hint_text),
                    prefill_answer=False,
                    think_text="",
                )
            else:
                raw_test = None

        if raw_test is None:
            split = raw_train.train_test_split(
                test_size=float(args.cd_test_fraction),
                seed=int(args.eval_seed),
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
        keep_cols = ["prompt", "prompt_raw", "answer"]
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

    is_adapter = (Path(args.model_id) / "adapter_config.json").exists()
    if is_adapter:
        from peft import AutoPeftModelForCausalLM

        adapter_load_kwargs = dict(model_load_kwargs)
        try:
            adapter_params = inspect.signature(AutoPeftModelForCausalLM.from_pretrained).parameters
            if "is_trainable" in adapter_params:
                adapter_load_kwargs["is_trainable"] = True
        except Exception:
            pass
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_id,
            **adapter_load_kwargs,
        )
        if hasattr(model, "train"):
            model.train()
        if hasattr(model, "set_adapter"):
            try:
                model.set_adapter("default")
            except Exception:
                pass
        if hasattr(model, "enable_adapter_layers"):
            try:
                model.enable_adapter_layers()
            except Exception:
                pass
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            **model_load_kwargs,
        )
    _apply_explicit_generation_config(
        model,
        max_prompt_tokens=args.max_prompt_tokens,
        max_completion_length=args.max_completion_length,
        temperature=args.train_temperature,
        top_p=args.train_top_p,
    )

    if not is_adapter:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)

    _wrap_generate_with_eval_mode(model)

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
        scale=float(args.cd_graph_reward_scale),
    )
    reward_funcs = [format_reward_math, accuracy_reward_math] if args.task == "math" else []
    if args.task == "causal_discovery":
        if float(args.cd_format_reward_scale) > 0.0:
            reward_funcs.append(build_cd_format_reward(scale=float(args.cd_format_reward_scale)))
        if float(args.cd_partial_format_reward_scale) > 0.0:
            reward_funcs.append(build_cd_partial_format_reward(scale=float(args.cd_partial_format_reward_scale)))
        if float(args.cd_edge_f1_reward_scale) > 0.0:
            reward_funcs.append(build_cd_edge_f1_reward(scale=float(args.cd_edge_f1_reward_scale)))
        if float(args.cd_low_shd_reward_scale) > 0.0:
            reward_funcs.append(build_cd_low_shd_reward(scale=float(args.cd_low_shd_reward_scale)))
        if float(args.cd_acyclic_reward_scale) > 0.0:
            reward_funcs.append(build_cd_acyclic_reward(scale=float(args.cd_acyclic_reward_scale)))
        reward_funcs.append(cd_graph_reward)
    elif args.task == "cd_descendants":
        if float(args.cd_format_reward_scale) > 0.0:
            reward_funcs.append(build_cd_format_reward(scale=float(args.cd_format_reward_scale)))
        if float(args.cd_partial_format_reward_scale) > 0.0:
            if bool(args.cd_grpo_prefill_answer):
                reward_funcs.append(
                    build_cd_descendant_answer_tail_partial_format_reward(
                        scale=float(args.cd_partial_format_reward_scale)
                    )
                )
            else:
                reward_funcs.append(
                    build_cd_descendant_partial_format_reward(scale=float(args.cd_partial_format_reward_scale))
                )
        reward_funcs.append(build_cd_descendant_f1_reward(scale=float(args.cd_graph_reward_scale)))

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
    sample_log_path = logs_dir / f"sample_completions_rank{rank}.jsonl"
    reward_call_counter = {"calls": 0}
    generation_counter = {"calls": 0}
    active_generation_id = {"id": -1}
    pending_reward_rows = {}
    generation_log_decisions = {}
    train_state_snapshot = {"global_step": 0, "epoch": None}

    if int(os.environ.get("LOCAL_RANK", "0")) == 0 and args.save_eval_responses:
        print(
            f"[log] saving reward-evaluated responses to {eval_response_log_path} "
            "for every rollout on rank0"
        )
    if int(os.environ.get("LOCAL_RANK", "0")) == 0 and args.sample_completions_every > 0:
        print(
            f"[log] saving sampled prompt/completion rows to {sample_log_path} "
            "for every rollout on rank0"
        )

    def _truncate_text_for_log(text: str) -> str:
        s = str(text)
        max_chars = int(args.eval_responses_max_chars)
        if max_chars > 0 and len(s) > max_chars:
            return s[:max_chars] + "...[truncated]"
        return s

    def _truncate_sample_text(text: str) -> str:
        return _sanitize_for_log(text, int(args.sample_completions_max_chars))

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
            # Typical chat-style: [{"role":"user","content":"..."}, ...]
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
        # TRL versions/providers may use different kwarg names.
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

    def _register_generation_logging(completions) -> None:
        generation_id = active_generation_id["id"]
        if generation_id in generation_log_decisions:
            return

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        persist_eval = bool(args.save_eval_responses and local_rank == 0)
        persist_sample = bool(args.sample_completions_every > 0 and local_rank == 0)
        selected = min(len(completions), max(int(args.sample_completions_k), 1)) if persist_sample else 0
        generation_log_decisions[generation_id] = {
            "persist_eval": persist_eval,
            "persist_sample": persist_sample,
            "sampled_indices": set(range(selected)),
        }

    def _log_reward_evaluations(
        generation_id: int,
        reward_name: str,
        completions,
        rewards,
        kwargs,
        *,
        flush: bool,
    ):
        decision = generation_log_decisions.get(generation_id)
        if not decision:
            return
        if not decision.get("persist_eval") and not decision.get("persist_sample"):
            return

        call_idx = reward_call_counter["calls"]
        reward_call_counter["calls"] += 1
        ts = time.time()
        prompt_key, prompt_texts = _extract_prompt_texts(kwargs, len(completions))
        _, prompt_raw_texts = _extract_raw_prompt_texts(kwargs, len(completions))
        answers = kwargs.get("answer")
        solutions = kwargs.get("solution")
        sampled_indices = decision.get("sampled_indices", set())

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
                    "global_step": int(train_state_snapshot.get("global_step") or 0),
                    "epoch": train_state_snapshot.get("epoch"),
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
                    "_sampled_for_log": i in sampled_indices,
                }
                pending_reward_rows[key] = row

            row["time_unix"] = ts
            row["last_reward_call_idx"] = call_idx
            row["global_step"] = int(train_state_snapshot.get("global_step") or 0)
            row["epoch"] = train_state_snapshot.get("epoch")
            row["rewards"][reward_name] = float(_to_jsonable(_value_at(rewards, i)) or 0.0)

        if flush and pending_reward_rows:
            emit_keys = sorted(k for k in pending_reward_rows.keys() if k[0] == generation_id)
            sample_file = sample_log_path.open("a", encoding="utf-8") if decision.get("persist_sample") else None
            eval_file = eval_response_log_path.open("a", encoding="utf-8") if decision.get("persist_eval") else None
            try:
                for key in emit_keys:
                    row = pending_reward_rows.pop(key)
                    rewards_map = row.get("rewards", {})
                    row["reward_total"] = float(sum(float(v) for v in rewards_map.values()))
                    if eval_file is not None:
                        eval_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if sample_file is not None and row.get("_sampled_for_log"):
                        sample_row = {
                            "time_unix": row["time_unix"],
                            "rank": row["rank"],
                            "global_step": row["global_step"],
                            "epoch": row["epoch"],
                            "generation_idx": row["generation_idx"],
                            "reward_call_idx": row["last_reward_call_idx"],
                            "sample_idx": row["sample_idx"],
                            "prompt_key": row["prompt_key"],
                            "format_ok": row["format_ok"],
                            "prompt": _truncate_sample_text(row["prompt"]),
                            "prompt_model_input": _truncate_sample_text(row["prompt_model_input"]),
                            "completion": _truncate_sample_text(row["completion"]),
                            "answer_text": _truncate_sample_text(row["answer_text"]),
                            "target_answer": _truncate_sample_text(row["target_answer"]),
                            "rewards": rewards_map,
                            "reward_total": row["reward_total"],
                        }
                        sample_file.write(json.dumps(sample_row, ensure_ascii=False) + "\n")
            finally:
                if eval_file is not None:
                    eval_file.close()
                if sample_file is not None:
                    sample_file.close()
                generation_log_decisions.pop(generation_id, None)

    if reward_funcs:
        wrapped_reward_funcs = []
        for idx, fn in enumerate(reward_funcs):
            reward_name = getattr(fn, "__name__", fn.__class__.__name__)

            def _make_logged_reward(inner_fn, inner_name, _do_sample_log: bool, is_first: bool, is_last: bool):
                def _logged_reward(completions, **kwargs):
                    if is_first:
                        active_generation_id["id"] = generation_counter["calls"]
                        generation_counter["calls"] += 1
                        _register_generation_logging(completions)
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

            wrapped_reward_funcs.append(
                _make_logged_reward(
                    fn,
                    reward_name,
                    idx == 0,
                    idx == 0,
                    idx == (len(reward_funcs) - 1),
                )
            )
        reward_funcs = wrapped_reward_funcs

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

    train_generation_kwargs = {}
    if args.stop_sequence:
        train_generation_kwargs["stop"] = args.stop_sequence
    if args.train_temperature > 0:
        train_generation_kwargs["temperature"] = args.train_temperature
        train_generation_kwargs["top_p"] = args.train_top_p

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        remove_unused_columns=False,
        num_train_epochs=args.num_train_epochs,
        beta=float(args.grpo_beta),

        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ddp_find_unused_parameters=False,

        bf16=torch.cuda.is_available(),

        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        generation_kwargs=train_generation_kwargs or None,

        report_to=report_to,
        run_name=args.run_name,
        disable_tqdm=False,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        push_to_hub=False,

        # vLLM integration (server mode)
        use_vllm=args.use_vllm,
        **grpo_kwargs,
    )
    print(f"[train] generation stop sequences: {args.stop_sequence if args.stop_sequence else 'none'}")
    print(
        f"[train] generation sampling: temperature={args.train_temperature}, "
        f"top_p={args.train_top_p}"
    )

    try:
        callbacks = [
            CompactMetricsCallback(),
            TrainStateSnapshotCallback(train_state_snapshot),
        ]
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
        _write_training_config_to_model_card(args.output_dir, args, argv)
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
    argv = sys.argv[1:]
    args = build_argparser().parse_args()
    _maybe_enable_small_graph_logging(args, argv)
    _maybe_launch_local_vllm_and_reexec_train(args)
    if args.mode == "eval":
        run_eval(args)
        return
    if args.mode == "export_cd_csv":
        run_export_cd_csv(args)
        return
    run_train(args, argv)


if __name__ == "__main__":
    main()
