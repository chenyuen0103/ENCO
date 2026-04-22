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
import inspect
import shlex
import warnings
import shutil
from typing import Any
from contextlib import redirect_stdout, redirect_stderr
import torch
from pathlib import Path
from tqdm.auto import tqdm
from urllib import request
from urllib.parse import urlparse

try:
    from cd_generation.format import (
        DEFAULT_FORMAT_HINT_TEXT,
        SYSTEM_PROMPT,
        canonicalize_cd_prompt,
        default_format_hint_text,
        default_short_think_text,
    )
except ModuleNotFoundError:
    from experiments.cd_generation.format import (
        DEFAULT_FORMAT_HINT_TEXT,
        SYSTEM_PROMPT,
        canonicalize_cd_prompt,
        default_format_hint_text,
        default_short_think_text,
    )

try:
    from torch.nn.parallel import DistributedDataParallel as _DDP
except Exception:
    _DDP = None

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_utils import get_last_checkpoint
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
    build_cd_descendant_cot_structure_reward,
    build_cd_descendant_shift_ranking_reward,
    build_cd_descendant_variable_classification_reward,
    build_cd_graph_reward,
    build_cd_descendant_f1_reward,
    build_cd_edge_f1_reward,
    build_cd_low_shd_reward,
    build_cd_acyclic_reward,
    build_cd_cot_structure_reward,
    build_cd_skeleton_f1_reward,
    build_cd_vstruct_f1_reward,
    build_cd_orientation_f1_reward,
    build_length_penalty_reward,
    build_cd_stage_targets,
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


def _has_full_model_config(model_dir: Path) -> bool:
    model_dir = Path(model_dir)
    return (model_dir / "config.json").exists() or (model_dir / "params.json").exists()


def _is_adapter_dir(model_dir: Path) -> bool:
    return (Path(model_dir) / "adapter_config.json").exists()


def _merge_adapter_for_vllm(model_dir: Path, *, base_model_override: str | None = None) -> Path:
    """
    Merge a PEFT adapter directory into a full model directory suitable for vLLM.

    The merged checkpoint is cached at `<adapter_dir>_merged_vllm` and reused if it
    already exists with a recognizable config file.
    """
    model_dir = Path(model_dir).resolve()
    merged_dir = model_dir.parent / f"{model_dir.name}_merged_vllm"
    if _has_full_model_config(merged_dir):
        print(f"[vllm] reusing existing merged model for adapter checkpoint: {merged_dir}")
        return merged_dir

    from peft import PeftModel

    adapter_cfg = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
    base_model_name = str(base_model_override or adapter_cfg.get("base_model_name_or_path") or "").strip()
    if not base_model_name:
        raise ValueError(
            f"Cannot merge adapter {model_dir}: no base model override was provided and "
            "adapter_config.json does not declare base_model_name_or_path"
        )
    tokenizer_source = Path(base_model_name) if Path(base_model_name).exists() else Path(str(base_model_name))

    print(f"[vllm] merging adapter for vLLM: {model_dir} -> {merged_dir}")
    base = AutoModelForCausalLM.from_pretrained(
        str(base_model_name),
        torch_dtype="auto",
    )
    model = PeftModel.from_pretrained(base, str(model_dir))
    merged = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))
    # Avoid tokenizer instantiation here: some environment/version combinations
    # trip over adapter tokenizer_config.json (`extra_special_tokens` stored as a list).
    # Copy the base tokenizer assets directly, then overlay adapter-specific files
    # such as the chat template when present.
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
    ]
    for filename in tokenizer_files:
        src = tokenizer_source / filename
        if src.exists():
            shutil.copy2(src, merged_dir / filename)
    chat_template_src = model_dir / "chat_template.jinja"
    if chat_template_src.exists():
        shutil.copy2(chat_template_src, merged_dir / "chat_template.jinja")
    print(f"[vllm] saved merged model -> {merged_dir}")
    return merged_dir


def _maybe_prepare_vllm_model(args) -> None:
    """
    Ensure the model path used by TRL vLLM server mode is a full model directory.

    vLLM cannot load LoRA adapter-only directories because they do not contain a
    base-model `config.json`. When server mode points at such a directory, merge it
    once and redirect vLLM to the merged checkpoint.
    """
    if not getattr(args, "use_vllm", False):
        return
    if getattr(args, "vllm_mode", None) != "server":
        return

    candidate_ref = getattr(args, "vllm_server_model_id", None) or getattr(args, "model_id", None)
    if not candidate_ref:
        return

    candidate_path = Path(str(candidate_ref)).expanduser()
    if not candidate_path.exists():
        return
    if _has_full_model_config(candidate_path):
        return
    if not _is_adapter_dir(candidate_path):
        return

    merged_path = _merge_adapter_for_vllm(
        candidate_path,
        base_model_override=getattr(args, "base_model_override", None),
    )
    args.vllm_server_model_id = str(merged_path)
    print(f"[vllm] server model path set to merged checkpoint: {args.vllm_server_model_id}")


def _ensure_model_warnings_issued(model) -> None:
    """
    TRL >= 0.24 may expect `model.warnings_issued` to exist on the model object.
    Some PEFT-wrapped models do not expose that attribute, so seed it explicitly
    on the wrapper and any reachable base model layers.
    """
    candidates = [model]
    for attr in ("base_model", "model"):
        try:
            child = getattr(model, attr, None)
        except Exception:
            child = None
        if child is not None:
            candidates.append(child)
            for sub_attr in ("model", "base_model"):
                try:
                    sub_child = getattr(child, sub_attr, None)
                except Exception:
                    sub_child = None
                if sub_child is not None:
                    candidates.append(sub_child)
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        ident = id(candidate)
        if ident in seen:
            continue
        seen.add(ident)
        if not hasattr(candidate, "warnings_issued"):
            try:
                setattr(candidate, "warnings_issued", {})
            except Exception:
                pass


def _import_trl_grpo(*, use_vllm: bool = True):
    from transformers.utils import import_utils as tf_import_utils

    orig_is_pkg_available = tf_import_utils._is_package_available
    disabled_packages = {"mergekit", "llm_blender"}
    if not use_vllm:
        disabled_packages.update({"vllm", "vllm_ascend"})

    def _make_is_pkg_available(disable_vllm_probe: bool):
        def _patched_is_package_available(package_name: str, *args, **kwargs):
            return_version = kwargs.get("return_version", args[0] if args else False)
            if package_name in disabled_packages:
                return (False, "0.0.0") if return_version else False
            if disable_vllm_probe and package_name in {"vllm", "vllm_ascend"}:
                return (False, "0.0.0") if return_version else False
            return orig_is_pkg_available(package_name, *args, **kwargs)

        return _patched_is_package_available

    def _patched_is_package_available(package_name: str, *args, **kwargs):
        return_version = kwargs.get("return_version", args[0] if args else False)
        if package_name in disabled_packages:
            return (False, "0.0.0") if return_version else False
        if not use_vllm and package_name in {"vllm", "vllm_ascend"}:
            return (False, "0.0.0") if return_version else False
        return orig_is_pkg_available(package_name, *args, **kwargs)

    def _clear_trl_modules() -> None:
        stale = [name for name in sys.modules if name == "trl" or name.startswith("trl.")]
        for name in stale:
            sys.modules.pop(name, None)

    def _do_import():
        with warnings.catch_warnings():
            # TRL checks vLLM support during import and emits the same advisory on
            # every torchrun worker. Keep it visible on rank 0 only.
            if str(os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or "0") not in {"0", ""}:
                warnings.filterwarnings(
                    "ignore",
                    message=r"TRL currently only supports vLLM version `0\.10\.2`.*",
                    category=UserWarning,
                )
            from trl import GRPOConfig, GRPOTrainer

        return GRPOConfig, GRPOTrainer

    tf_import_utils._is_package_available = _patched_is_package_available
    try:
        _clear_trl_modules()
        try:
            return _do_import()
        except Exception as e:
            msg = str(e)
            # Some TRL installs try to initialize vLLM during import and fail on
            # systems where NVML/CUDA visibility is unstable. Retry with vLLM disabled.
            if not any(token in msg for token in ("GuidedDecodingParams", "vllm", "NVMLError", "NVML")):
                raise
            tf_import_utils._is_package_available = _make_is_pkg_available(True)
            try:
                _clear_trl_modules()
                return _do_import()
            except Exception:
                raise e
    finally:
        tf_import_utils._is_package_available = orig_is_pkg_available


def _filter_supported_kwargs(callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        params = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


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


def _resolve_model_reference_for_attention(args) -> str:
    if getattr(args, "base_model_override", None):
        return str(args.base_model_override)

    model_id = str(getattr(args, "model_id", "") or "")
    adapter_cfg_path = Path(model_id) / "adapter_config.json"
    if adapter_cfg_path.exists():
        try:
            adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
            base_model_name = str(adapter_cfg.get("base_model_name_or_path") or "").strip()
            if base_model_name:
                return base_model_name
        except Exception:
            pass
    return model_id


def _resolve_attn_implementation(args) -> str:
    requested = str(getattr(args, "attn_implementation", "auto") or "auto").strip().lower()
    if requested != "auto":
        return requested

    model_ref = _resolve_model_reference_for_attention(args).lower()
    # Qwen3 has been unreliable with Flash Attention 2 under GRPO's masked/varlen batches.
    if "qwen3" in model_ref:
        return "sdpa"
    return "flash_attention_2"


def _probe_trl_vllm_server(base_url: str, timeout: float) -> tuple[bool, str]:
    base = base_url.rstrip("/")
    probes = (
        ("health", f"{base}/health/", None),
        ("world_size", f"{base}/get_world_size/", "world_size"),
    )
    for probe_name, url, required_key in probes:
        try:
            with request.urlopen(url, timeout=timeout) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}")
                body = resp.read()
            if required_key is not None:
                payload = json.loads(body.decode("utf-8")) if body else {}
                if required_key not in payload:
                    raise RuntimeError(f"missing JSON key {required_key!r}")
        except Exception as e:
            return False, f"{probe_name} probe failed for {url}: {e}"
    return True, ""


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
    """Print a compact, task-aware metric line each logging step.

    Causal discovery  → reward | f1 | lshd | skel
    CD descendants    → reward | f1 (desc) | drank
    Math              → reward | acc
    Common prefix     → epoch | loss
    """

    # Keys always shown when present
    _COMMON = [
        ("epoch", "epoch"),
        ("loss", "loss"),
        ("reward", "reward"),
        ("reward_min", "reward_min"),
        ("reward_max", "reward_max"),
    ]

    # Task-specific keys, detected from whichever key is present in logs
    _CD_KEYS = [
        ("rewards/cd_edge_f1_reward/mean",      "f1"),
        ("rewards/cd_low_shd_reward/mean",       "lshd"),
        ("rewards/cd_skeleton_f1_reward/mean",   "skel"),
    ]

    _DESC_KEYS = [
        ("rewards/cd_descendant_f1_reward/mean",            "f1"),
        ("rewards/cd_descendant_shift_ranking_reward/mean", "drank"),
    ]

    _MATH_KEYS = [
        ("rewards/accuracy_reward/mean", "acc"),
    ]

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero or not logs:
            return

        # Detect task from which reward keys are present
        if "rewards/cd_descendant_f1_reward/mean" in logs:
            task_keys = self._DESC_KEYS
        elif "rewards/cd_edge_f1_reward/mean" in logs:
            task_keys = self._CD_KEYS
        else:
            task_keys = self._MATH_KEYS

        parts = []
        for key, name in self._COMMON + task_keys:
            if key not in logs:
                continue
            value = logs[key]
            parts.append(f"{name}={value:.4f}" if isinstance(value, float) else f"{name}={value}")

        if parts:
            print("[train] " + " | ".join(parts))


class RawMetricsCallback(TrainerCallback):
    """Derive and print/log raw (unscaled) F1 and SHD from logged reward values.

    TRL logs ``rewards/<name>/mean`` as the *scaled* reward average.  This
    callback back-calculates the underlying metric by dividing by the known
    scale so we see the actual F1 (0–1) and normalized SHD (0–1) each step.

    Logged keys written to W&B (when a run is active):
        metrics/raw_edge_f1      – directed-edge F1 averaged over the batch
        metrics/normalized_shd   – normalized SHD  (0 = perfect, 1 = worst)
    """

    def __init__(self, edge_f1_scale: float = 0.0, shd_scale: float = 0.0):
        self.edge_f1_scale = float(edge_f1_scale)
        self.shd_scale = float(shd_scale)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero or not logs:
            return

        parts: list[str] = []
        wandb_payload: dict = {}

        if self.edge_f1_scale > 0.0:
            scaled = logs.get("rewards/cd_edge_f1_reward/mean")
            if scaled is not None:
                raw_f1 = float(scaled) / self.edge_f1_scale
                parts.append(f"raw_f1={raw_f1:.4f}")
                wandb_payload["metrics/raw_edge_f1"] = raw_f1
                logs["metrics/raw_edge_f1"] = raw_f1

        if self.shd_scale > 0.0:
            scaled = logs.get("rewards/cd_low_shd_reward/mean")
            if scaled is not None:
                # reward = (1 - norm_shd) * scale  →  norm_shd = 1 - reward/scale
                norm_shd = 1.0 - float(scaled) / self.shd_scale
                parts.append(f"norm_shd={norm_shd:.4f}")
                wandb_payload["metrics/normalized_shd"] = norm_shd
                logs["metrics/normalized_shd"] = norm_shd

        if parts:
            print("[raw_metrics] " + " | ".join(parts))

        if wandb_payload:
            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    _wandb.log(wandb_payload, step=int(state.global_step))
            except Exception:
                pass


class RewardExtremaCallback(TrainerCallback):
    """Inject per-step reward min/max statistics captured during rollout logging."""

    def __init__(self, stats_ref: dict):
        self.stats_ref = stats_ref

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero or not logs:
            return

        payload = self.stats_ref.get("latest")
        if not payload:
            return

        for key, value in payload.items():
            logs[key] = value

        try:
            import wandb as _wandb
            if _wandb.run is not None:
                # Trainer-managed metrics typically appear under the `train/` namespace
                # in W&B. Mirror the extrema there so they show up alongside the
                # existing mean/std reward series, while also preserving the legacy
                # unprefixed keys for backward compatibility.
                wandb_payload = dict(payload)
                for key, value in payload.items():
                    wandb_payload[f"train/{key}"] = value
                _wandb.log(wandb_payload, step=int(state.global_step))
        except Exception:
            pass


class JsonMetricsCallback(TrainerCallback):
    """Persist per-log training metrics as a JSON array (rewritten on each log)."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []

    @staticmethod
    def _to_jsonable(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if hasattr(value, "item"):
            try:
                return JsonMetricsCallback._to_jsonable(value.item())
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
        self.history.append(payload)
        self.output_path.write_text(
            json.dumps(self.history, ensure_ascii=False),
            encoding="utf-8",
        )


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


def _build_launch_command(argv: list[str]) -> str:
    """Reconstruct the full launch command including script name and torchrun args."""
    script = shlex.quote(sys.argv[0])  # e.g. 'experiments/grpo.py'
    n_gpus = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    script_args = " ".join(shlex.quote(p) for p in argv)
    if n_gpus > 1:
        return f"torchrun --standalone --nproc_per_node={n_gpus} {script} {script_args}"
    else:
        return f"python {script} {script_args}"


class SaveLaunchCommandCallback(TrainerCallback):
    """Write launch_command.sh into every checkpoint subdirectory when it is saved."""

    def __init__(self, argv: list[str]):
        self._command_text = _build_launch_command(argv)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_local_process_zero:
            return
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if ckpt_dir.is_dir():
            (ckpt_dir / "launch_command.sh").write_text(
                "#!/usr/bin/env bash\n" + self._command_text + "\n",
                encoding="utf-8",
            )


def build_argparser():
    p = argparse.ArgumentParser(description="GRPO training with TRL + vLLM server mode")
    p.add_argument("--mode", type=str, default="train", choices=["train", "eval", "export_cd_csv"])
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume training from the latest checkpoint found in --output_dir. "
            "Restores model weights, optimizer, scheduler, and RNG state."
        ),
    )
    p.add_argument(
        "--task",
        type=str,
        default="causal_discovery",
        choices=["causal_discovery", "cd_descendants", "math"],
        help="Training/eval task type. causal_discovery and cd_descendants expect prompt CSV inputs.",
    )
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    p.add_argument(
        "--base-model-override",
        type=str,
        default=None,
        help=(
            "When model_id is a PEFT adapter trained on a quantized base (e.g. BNB 4-bit), "
            "provide the full-precision base model path/id here. The base will be loaded "
            "in bfloat16 without quantization and the adapter applied on top."
        ),
    )
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help=(
            "Attention backend for the training model. "
            "'auto' prefers Flash Attention 2, but falls back to SDPA for Qwen3."
        ),
    )
    p.add_argument("--dataset_id", type=str, default="AI-MO/NuminaMath-TIR")
    p.add_argument("--train_split", type=str, default="train[:5%]")
    p.add_argument("--test_split", type=str, default="test[:5%]")
    p.add_argument("--output_dir", type=str, default="Qwen2-0.5B-GRPO-vLLM-server")

    # Prompt / generation
    p.add_argument("--max_prompt_tokens", type=int, default=4096)
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
        default=10,
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
        default=0,
        help="Max chars per logged prompt/completion sample.",
    )

    # Train
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
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
        "--cd-test-fraction",
        type=float,
        default=0.1,
        help="If --cd-test-csv is not provided, reserve this fraction from train as eval.",
    )
    p.add_argument("--cd-max-train-samples", type=int, default=0, help="Cap train samples (0 = no cap).")
    p.add_argument("--cd-max-test-samples", type=int, default=0, help="Cap eval samples (0 = no cap).")
    p.add_argument(
        "--cd-wrapper-mode",
        choices=["plain", "chat"],
        default=None,
        help="Preferred prompt transport for causal-discovery tasks.",
    )
    p.add_argument(
        "--cd-response-format",
        choices=["think_answer", "json"],
        default="think_answer",
        help="Requested response format for causal-discovery tasks.",
    )
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
        default=default_format_hint_text("causal_discovery"),
    )
    p.add_argument(
        "--cd-grpo-prefill-answer",
        dest="cd_grpo_prefill_answer",
        action="store_true",
        help=(
            "For causal-discovery training rollouts, prefill the assistant through the short think "
            "trace and opening <answer>, so GRPO only generates the answer payload and closing tag."
        ),
    )
    p.add_argument(
        "--no-cd-grpo-prefill-answer",
        dest="cd_grpo_prefill_answer",
        action="store_false",
    )
    p.set_defaults(cd_grpo_prefill_answer=False)
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
    p.add_argument(
        "--cd-cot-structure-reward-scale",
        type=float,
        default=0.0,
        help="Soft reward for mentioning all three staged-reasoning stages in <think> (0 disables).",
    )
    p.add_argument(
        "--cd-skeleton-f1-reward-scale",
        type=float,
        default=0.0,
        help="Hard reward: skeleton F1 parsed from Stage 1 of <think> vs ground truth (0 disables).",
    )
    p.add_argument(
        "--cd-vstruct-f1-reward-scale",
        type=float,
        default=0.0,
        help="Hard reward: v-structure triple F1 parsed from Stage 2 of <think> vs ground truth (0 disables).",
    )
    p.add_argument(
        "--cd-orientation-f1-reward-scale",
        type=float,
        default=0.0,
        help="Hard reward: directed edge F1 parsed from Stage 3 of <think> vs ground truth (0 disables).",
    )
    p.add_argument(
        "--cd-descendant-cot-structure-reward-scale",
        type=float,
        default=0.0,
        help="Soft reward for descendant-task Stage 1/2/3 reasoning structure inside <think> (0 disables).",
    )
    p.add_argument(
        "--cd-descendant-shift-ranking-reward-scale",
        type=float,
        default=0.0,
        help="Reward for ranking variables by intervention shift magnitude in descendant-task Stage 1 (0 disables).",
    )
    p.add_argument(
        "--cd-descendant-variable-classification-reward-scale",
        type=float,
        default=0.0,
        help="Reward for per-variable shifted/stable and descendant/not-descendant labels in descendant-task Stage 2 (0 disables).",
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
        "--vllm-mode",
        type=str,
        choices=["server", "colocate"],
        default="server",
        help="Use a separate TRL vLLM server or colocate vLLM with trainer processes.",
    )
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
    p.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.3,
        help="GPU memory utilization for colocated vLLM (only applies with --vllm-mode colocate).",
    )
    p.add_argument(
        "--vllm_max_model_length",
        type=int,
        default=None,
        help=(
            "Context window for colocated vLLM. Defaults to "
            "--max_prompt_tokens + --max_completion_length."
        ),
    )

    # Logging
    p.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "none"])
    p.add_argument("--run_name", type=str, default=None,
                   help="W&B run name. Defaults to the output_dir basename.")
    p.add_argument("--wandb_project", type=str, default="enco-grpo")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--logging_steps", type=int, default=1)
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


def _middle_truncate_ids(ids: list, max_len: int) -> list:
    """Keep the first half and last half of token ids, dropping the middle.

    Preserves task instructions (head) and format spec (tail) while dropping
    data tokens in the middle when a prompt exceeds max_len.
    """
    if len(ids) <= max_len:
        return ids
    keep_head = max_len // 2
    keep_tail = max_len - keep_head
    return ids[:keep_head] + ids[-keep_tail:]


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
    task: str = "causal_discovery",
    response_format: str = "think_answer",
    wrap_system_prompt: bool,
    append_format_hint: bool = False,
    format_hint_text: str = "",
    prefill_answer: bool = False,
    think_text: str = "",
) -> list[dict]:
    rows: list[dict] = []
    _set_csv_field_limit()
    is_jsonl = csv_path.suffix.lower() in (".jsonl", ".jsonlines")
    if is_jsonl:
        raw_rows = []
        with csv_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_rows.append(json.loads(line))
        iter_rows = enumerate(raw_rows)
    else:
        _f = csv_path.open("r", encoding="utf-8", newline="")
        iter_rows = enumerate(csv.DictReader(_f))
    try:
        for i, row in iter_rows:
            prompt_raw = (row.get("prompt_text") or row.get("prompt") or "").strip()
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
                task=task,
                response_format=response_format,
                wrap_system_prompt=bool(wrap_system_prompt),
                append_format_hint=bool(append_format_hint),
                format_hint_text=str(format_hint_text),
                prefill_think=not bool(prefill_answer),
                prefill_answer=bool(prefill_answer),
                think_text=str(think_text),
                strip_output_instructions=bool(wrap_system_prompt),
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
                (
                    {
                        "prompt_raw": prompt_raw,
                        "prompt": prompt_text,
                        "answer": answer_raw,
                    }
                    | (
                        build_cd_stage_targets(prompt_raw, answer_raw) or {}
                        if task == "causal_discovery"
                        else {}
                    )
                )
            )
    finally:
        if not is_jsonl:
            _f.close()
    return rows


def _dataset_from_cd_csvs(
    csv_paths: list[str],
    *,
    task: str = "causal_discovery",
    response_format: str = "think_answer",
    wrap_system_prompt: bool,
    append_format_hint: bool = False,
    format_hint_text: str = "",
    prefill_answer: bool = False,
    think_text: str = "",
) -> Dataset:
    all_rows: list = []
    for path_str in csv_paths:
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"--cd-train/test-csv file not found: {p}")
        rows = _load_cd_rows_from_prompt_csv(
            p,
            task=task,
            response_format=response_format,
            wrap_system_prompt=wrap_system_prompt,
            append_format_hint=append_format_hint,
            format_hint_text=format_hint_text,
            prefill_answer=prefill_answer,
            think_text=think_text,
        )
        if not rows:
            raise ValueError(f"No usable rows found in {p}")
        all_rows.extend(rows)

    if not all_rows:
        raise ValueError("No causal discovery CSV inputs were provided.")
    return Dataset.from_list(all_rows)


def _dataset_from_cd_config_file(
    *,
    config_file: str,
    bif_file: str,
    num_prompts: int,
    seed: int,
    task: str,
    response_format: str,
    wrap_system_prompt: bool,
    append_format_hint: bool = False,
    format_hint_text: str = "",
    prefill_answer: bool = False,
    think_text: str = "",
    causal_rules: bool = False,
    give_steps: bool = False,
    def_int: bool = False,
    intervene_vars: str = "all",
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

    style_aliases = {"summary_join": "summary", "summary_joint": "summary"}
    all_styles = ["cases", "matrix", "summary", "payload", "payload_topk"]
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
    for style, anon, obs_n, int_n, row_ord, col_ord, shuf_n, legacy_wrapper_mode, config_append_format_hint in configs:
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
            wrapper_mode=legacy_wrapper_mode,
            append_format_hint=bool(config_append_format_hint),
        )
        adj = answer_obj.get("adjacency_matrix") if isinstance(answer_obj, dict) else None
        variables_out = answer_obj.get("variables") if isinstance(answer_obj, dict) else None
        # Export only adjacency_matrix — variables is internal metadata not used by reward fns.
        if isinstance(answer_obj, dict) and "adjacency_matrix" in answer_obj:
            answer_raw = json.dumps({"adjacency_matrix": answer_obj["adjacency_matrix"]}, ensure_ascii=False)
        else:
            answer_raw = json.dumps(answer_obj, ensure_ascii=False)
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
                        task=task,
                        response_format=response_format,
                        wrap_system_prompt=bool(wrap_system_prompt),
                        append_format_hint=bool(append_format_hint),
                        format_hint_text=str(format_hint_text),
                        prefill_think=not bool(prefill_answer),
                        prefill_answer=bool(prefill_answer),
                        think_text=str(think_text),
                        strip_output_instructions=bool(wrap_system_prompt),
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
                task=task,
                response_format=response_format,
                wrap_system_prompt=bool(wrap_system_prompt),
                append_format_hint=bool(append_format_hint),
                format_hint_text=str(format_hint_text),
                prefill_think=not bool(prefill_answer),
                prefill_answer=bool(prefill_answer),
                think_text=str(think_text),
                strip_output_instructions=bool(wrap_system_prompt),
            )
            rows.append(
                (
                    {
                        "prompt_raw": prompt_raw,
                        "prompt": prompt_text,
                        "answer": answer_raw,
                    }
                    | (
                        build_cd_stage_targets(prompt_raw, answer_raw) or {}
                        if task == "causal_discovery"
                        else {}
                    )
                )
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


class _TokenSequenceStoppingCriteria:
    """Stop generation when a specific token-ID sequence appears at the end of any sequence."""

    def __init__(self, stop_token_ids: list[int], prompt_length: int):
        self.stop_ids = stop_token_ids
        self.n = len(stop_token_ids)
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        # Only inspect newly generated tokens (after prompt)
        generated = input_ids[:, self.prompt_length:]
        if generated.shape[1] < self.n:
            return False
        tail = generated[:, -self.n:].tolist()
        return all(seq == self.stop_ids for seq in tail)


def _wrap_generate_with_eval_mode(model, stop_strings: list[str] | None = None, tokenizer=None):
    if getattr(model, "_grpo_eval_generate_wrapped", False):
        return model

    original_generate = getattr(model, "generate", None)
    if original_generate is None:
        return model

    # Pre-tokenize each stop string into its token-ID sequence.
    stop_token_id_seqs: list[list[int]] = []
    if stop_strings and tokenizer is not None:
        for s in stop_strings:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if ids:
                stop_token_id_seqs.append(ids)
                print(f"[generate] stop string {s!r} -> token ids {ids}")

    def _generate_with_eval_mode(*args, **kwargs):
        was_training = bool(getattr(model, "training", False))
        if was_training:
            model.eval()
        try:
            # Inject StoppingCriteria for non-vLLM generation when stop strings are set.
            if stop_token_id_seqs and "stopping_criteria" not in kwargs:
                from transformers import StoppingCriteriaList

                _kw_ids = kwargs.get("input_ids")
                input_ids = _kw_ids if _kw_ids is not None else (args[0] if args else None)
                prompt_len = int(input_ids.shape[1]) if input_ids is not None else 0
                criteria = StoppingCriteriaList([
                    _TokenSequenceStoppingCriteria(ids, prompt_len)
                    for ids in stop_token_id_seqs
                ])
                kwargs["stopping_criteria"] = criteria

            # Transformers requires a tokenizer when stop_strings are present in kwargs
            # or the generation config. Inject it if not already provided.
            if "tokenizer" not in kwargs and tokenizer is not None:
                kwargs["tokenizer"] = tokenizer

            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return original_generate(*args, **kwargs)
            return original_generate(*args, **kwargs)
        finally:
            if was_training:
                model.train()

    setattr(model, "generate", _generate_with_eval_mode)
    setattr(model, "_grpo_eval_generate_wrapped", True)
    return model


def _coerce_model_floating_dtype(model, target_dtype: torch.dtype) -> tuple[int, int]:
    """Best-effort cast for floating params/buffers to avoid mixed-dtype matmul crashes."""
    converted = 0
    skipped = 0

    for param in model.parameters():
        try:
            if not getattr(param, "is_floating_point", lambda: False)():
                continue
            if param.dtype == target_dtype:
                continue
            param.data = param.data.to(dtype=target_dtype)
            converted += 1
        except Exception:
            skipped += 1

    for buf in model.buffers():
        try:
            if not getattr(buf, "is_floating_point", lambda: False)():
                continue
            if buf.dtype == target_dtype:
                continue
            buf.data = buf.data.to(dtype=target_dtype)
            converted += 1
        except Exception:
            skipped += 1

    return converted, skipped


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
        if max_prompt_tokens and int(max_prompt_tokens) > 0:
            truncated_ids = [
                _middle_truncate_ids(tokenizer(p, truncation=False)["input_ids"], int(max_prompt_tokens))
                for p in prompts
            ]
            inputs = tokenizer.pad(
                [{"input_ids": ids} for ids in truncated_ids],
                return_tensors="pt",
                padding=True,
            ).to(input_device)
        else:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(input_device)
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
            response_format=str(args.cd_response_format),
            wrap_system_prompt=bool(args.cd_wrap_system_prompt),
            append_format_hint=bool(args.cd_append_format_hint),
            format_hint_text=str(args.cd_format_hint_text),
            causal_rules=bool(args.cd_config_causal_rules),
            give_steps=bool(args.cd_config_give_steps),
            def_int=bool(args.cd_config_def_int),
            intervene_vars=str(args.cd_config_intervene_vars),
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
            task=str(args.task),
            response_format=str(args.cd_response_format),
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
            response_format=str(args.cd_response_format),
            wrap_system_prompt=bool(args.cd_wrap_system_prompt),
            append_format_hint=bool(args.cd_append_format_hint),
            format_hint_text=str(args.cd_format_hint_text),
            causal_rules=bool(args.cd_config_causal_rules),
            give_steps=bool(args.cd_config_give_steps),
            def_int=bool(args.cd_config_def_int),
            intervene_vars=str(args.cd_config_intervene_vars),
        )
    else:
        if not args.cd_train_csv:
            raise ValueError(
                f"For --task {args.task}, pass --cd-config-file + --cd-bif-file "
                "or --cd-train-csv."
            )
        dataset = _dataset_from_cd_csvs(
            args.cd_train_csv,
            task=str(args.task),
            response_format=str(args.cd_response_format),
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
    if args.vllm_server_max_model_len is not None and args.vllm_server_max_model_len > 0:
        cmd.extend(["--max-model-len", str(args.vllm_server_max_model_len)])
    return cmd


def _maybe_launch_local_vllm_and_reexec_train(args):
    if args.mode != "train" or not args.use_vllm or not args.auto_launch_vllm_server:
        return
    if args.vllm_mode != "server":
        raise RuntimeError("--auto-launch-vllm-server only works with --vllm-mode server.")
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
    args.vllm_server_base_url = args.vllm_server_base_url.rstrip("/")
    explicit_use_vllm = _argv_has_flag(argv, "--use-vllm")
    explicit_no_use_vllm = _argv_has_flag(argv, "--no-use-vllm")
    startup_t0 = time.perf_counter()
    startup_last_t = startup_t0

    def _log_startup_timing(stage: str) -> None:
        nonlocal startup_last_t
        now = time.perf_counter()
        delta_s = now - startup_last_t
        total_s = now - startup_t0
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        print(
            f"[startup][rank={rank} local_rank={local_rank}] {stage}: "
            f"+{delta_s:.2f}s (total {total_s:.2f}s)",
            flush=True,
        )
        startup_last_t = now

    # if args.use_vllm and args.vllm_mode == "server":
    #     timeout_s = min(max(float(args.vllm_server_timeout), 1.0), 10.0)
    #     server_ok, server_error = _probe_trl_vllm_server(args.vllm_server_base_url, timeout=timeout_s)
    #     if not server_ok:
    #         if not explicit_use_vllm and not explicit_no_use_vllm:
    #             args.use_vllm = False
    #             if int(os.environ.get("RANK", "0")) == 0:
    #                 print(
    #                     "[warn] no reachable TRL vLLM server at "
    #                     f"{args.vllm_server_base_url}; defaulting to Transformers generation "
    #                     "because neither --use-vllm nor --no-use-vllm was passed. "
    #                     f"Probe error: {server_error}"
    #                 )
    #         else:
    #             raise RuntimeError(
    #                 f"Could not reach a TRL vLLM server at {args.vllm_server_base_url}. "
    #                 "Start TRL's server (`trl vllm-serve`) or rerun with --no-use-vllm "
    #                 "to use Transformers generation directly. "
    #                 f"Probe error: {server_error}"
    #             )

    GRPOConfig, GRPOTrainer = _import_trl_grpo(use_vllm=bool(args.use_vllm))
    _patch_ddp_config_attr()
    _suppress_transformers_attn_mask_warning_bug()
    _log_startup_timing("TRL import and startup checks")

    if args.max_prompt_tokens <= 0:
        args.max_prompt_tokens = 2048
        print("[warn] max_prompt_tokens <= 0; using safer default 2048.")
    if args.max_completion_length <= 0:
        args.max_completion_length = 512
        print("[warn] max_completion_length <= 0; using safer default 512.")
    if args.use_vllm and args.vllm_mode == "colocate":
        if not (0 < float(args.vllm_gpu_memory_utilization) <= 1.0):
            raise ValueError("--vllm_gpu_memory_utilization must be in (0, 1].")
        target_vllm_max_model_length = int(args.max_prompt_tokens) + int(args.max_completion_length)
        if args.vllm_max_model_length is None or args.vllm_max_model_length <= 0:
            args.vllm_max_model_length = target_vllm_max_model_length
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    "[train] auto-set colocated vLLM max model length to "
                    f"{args.vllm_max_model_length} "
                    "(max_prompt_tokens + max_completion_length)."
                )
        elif args.vllm_max_model_length < target_vllm_max_model_length and int(os.environ.get("RANK", "0")) == 0:
            print(
                "[warn] colocated vLLM max model length is smaller than "
                "max_prompt_tokens + max_completion_length; long prompts or completions may be truncated."
            )
    if args.train_temperature < 0:
        raise ValueError("--train_temperature must be >= 0.")
    if not (0 < args.train_top_p <= 1.0):
        raise ValueError("--train_top_p must be in (0, 1].")
    if args.grpo_beta < 0:
        raise ValueError("--grpo-beta must be >= 0.")
    if not args.stop_sequence:
        args.stop_sequence = ["</answer>"]
    args.stop_sequence = [s for s in args.stop_sequence if isinstance(s, str) and s]

    if args.task == "causal_discovery" and args.cd_grpo_prefill_answer:
        staged_reward_scales = {
            "cd-cot-structure": float(args.cd_cot_structure_reward_scale),
            "cd-skeleton-f1": float(args.cd_skeleton_f1_reward_scale),
            "cd-vstruct-f1": float(args.cd_vstruct_f1_reward_scale),
            "cd-orientation-f1": float(args.cd_orientation_f1_reward_scale),
        }
        enabled_staged_rewards = [
            name for name, scale in staged_reward_scales.items() if scale > 0.0
        ]
        if enabled_staged_rewards:
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    "[warn] --cd-grpo-prefill-answer is incompatible with staged "
                    "causal-discovery rewards because the model does not generate "
                    "Stage 1/2/3 text in that mode. Falling back to assistant "
                    f"<think> prefill. Enabled staged rewards: {', '.join(enabled_staged_rewards)}"
                )
            args.cd_grpo_prefill_answer = False

    # Throughput-oriented defaults for Ampere/Hopper GPUs.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

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

    if args.use_vllm and args.vllm_mode == "server" and args.enable_vllm_preflight:
        _preflight_vllm_communicator()

    # Default run_name to the output_dir basename so W&B runs are self-labeling.
    if not args.run_name:
        args.run_name = Path(args.output_dir).name

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
    _log_startup_timing("W&B setup")

    # ---- Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    _log_startup_timing("Tokenizer load")

    # ---- Dataset
    if args.task == "math":
        train_dataset, test_dataset = load_dataset(
            args.dataset_id,
            split=[args.train_split, args.test_split],
        )

        def add_prompt(example, max_prompt_tokens=args.max_prompt_tokens):
            text = build_prompt(example["problem"])
            orig_ids = tokenizer(text, truncation=False)["input_ids"]
            ids = _middle_truncate_ids(orig_ids, int(max_prompt_tokens))
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
        # Honor the rollout answer-prefill flag so GRPO can generate only the answer tail.
        rollout_prefill_answer = bool(args.cd_grpo_prefill_answer)
        rollout_think_text = ""
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
                response_format=str(args.cd_response_format),
                wrap_system_prompt=bool(args.cd_wrap_system_prompt),
                append_format_hint=bool(args.cd_append_format_hint),
                format_hint_text=str(args.cd_format_hint_text),
                prefill_answer=rollout_prefill_answer,
                think_text=rollout_think_text,
                causal_rules=bool(args.cd_config_causal_rules),
                give_steps=bool(args.cd_config_give_steps),
                def_int=bool(args.cd_config_def_int),
                intervene_vars=str(args.cd_config_intervene_vars),
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
                task=str(args.task),
                response_format=str(args.cd_response_format),
                wrap_system_prompt=bool(args.cd_wrap_system_prompt),
                append_format_hint=bool(args.cd_append_format_hint),
                format_hint_text=str(args.cd_format_hint_text),
                prefill_answer=rollout_prefill_answer,
                think_text=rollout_think_text,
            )
            if args.cd_test_csv:
                raw_test = _dataset_from_cd_csvs(
                    args.cd_test_csv,
                    task=str(args.task),
                    response_format=str(args.cd_response_format),
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
            ids = _middle_truncate_ids(orig_ids, int(max_prompt_tokens))
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
        keep_cols = [
            "prompt",
            "prompt_raw",
            "answer",
            "target_stage1_skeleton_edges",
            "target_stage2_vstructures",
            "target_stage3_directed_edges",
        ]
        train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in keep_cols])
        test_dataset = test_dataset.remove_columns([c for c in test_dataset.column_names if c not in keep_cols])
    _log_startup_timing("Dataset load and preprocessing")

    # Debug: verify tokenized prompt before model load (fast, no GPU needed)
    if int(os.environ.get("RANK", "0")) == 0:
        _dbg_prompt = train_dataset[0]["prompt"]
        _dbg_ids = tokenizer(_dbg_prompt, return_tensors="pt")["input_ids"][0]
        _dbg_decoded = tokenizer.decode(_dbg_ids)
        print(f"\n[DEBUG] First prompt token count: {len(_dbg_ids)}")
        print(f"[DEBUG] First 300 chars:\n{_dbg_decoded[:300]}")
        print(f"[DEBUG] Last 300 chars:\n{_dbg_decoded[-300:]}\n")
        import sys; sys.stdout.flush()

    # ---- Model + LoRA
    distributed_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    resolved_attn_implementation = _resolve_attn_implementation(args)
    if int(os.environ.get("RANK", "0")) == 0:
        model_ref = _resolve_model_reference_for_attention(args)
        print(
            f"[attn] using attn_implementation={resolved_attn_implementation} "
            f"for model reference {model_ref}"
        )
    model_load_kwargs = {
        "dtype": "auto",
        "attn_implementation": resolved_attn_implementation,
    }
    # `device_map="auto"` is incompatible with distributed Accelerate launches.
    if distributed_world_size == 1:
        model_load_kwargs["device_map"] = "auto"

    is_adapter = (Path(args.model_id) / "adapter_config.json").exists()
    if is_adapter:
        from peft import AutoPeftModelForCausalLM, PeftModel

        if getattr(args, "base_model_override", None):
            # Load full-precision base then apply adapter on top (bypasses quantized base).
            print(f"[train] Loading full-precision base from {args.base_model_override}, "
                  f"then applying adapter from {args.model_id}")
            base = AutoModelForCausalLM.from_pretrained(
                args.base_model_override,
                **model_load_kwargs,
            )
            model = PeftModel.from_pretrained(base, args.model_id, is_trainable=True)
        else:
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
        # Required for gradient checkpointing + LoRA: base weights are frozen,
        # so we need embedding outputs to require grad for recomputation to work.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            **model_load_kwargs,
        )
    _log_startup_timing("Model load")
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
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    _ensure_model_warnings_issued(model)

    # Keep all floating modules in one dtype. Some adapter/base combinations can
    # otherwise mix fp32 and bf16 and fail in Linear matmul.
    if torch.cuda.is_available():
        cast_dtype = torch.bfloat16
        casted, skipped_cast = _coerce_model_floating_dtype(model, cast_dtype)
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[train] dtype normalization -> {cast_dtype}: converted={casted}, skipped={skipped_cast}"
            )
    _log_startup_timing("LoRA/adapters and dtype normalization")

    _wrap_generate_with_eval_mode(
        model,
        stop_strings=args.stop_sequence if not args.use_vllm else None,
        tokenizer=tokenizer,
    )

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
        if float(args.cd_cot_structure_reward_scale) > 0.0:
            reward_funcs.append(build_cd_cot_structure_reward(scale=float(args.cd_cot_structure_reward_scale)))
        if float(args.cd_skeleton_f1_reward_scale) > 0.0:
            reward_funcs.append(build_cd_skeleton_f1_reward(scale=float(args.cd_skeleton_f1_reward_scale)))
        if float(args.cd_vstruct_f1_reward_scale) > 0.0:
            reward_funcs.append(build_cd_vstruct_f1_reward(scale=float(args.cd_vstruct_f1_reward_scale)))
        if float(args.cd_orientation_f1_reward_scale) > 0.0:
            reward_funcs.append(build_cd_orientation_f1_reward(scale=float(args.cd_orientation_f1_reward_scale)))
        reward_funcs.append(cd_graph_reward)
    elif args.task == "cd_descendants":
        if float(args.cd_format_reward_scale) > 0.0:
            reward_funcs.append(build_cd_format_reward(scale=float(args.cd_format_reward_scale)))
        if float(args.cd_partial_format_reward_scale) > 0.0:
            # Rollout always prefills through <think> only, so use the full-format reward.
            reward_funcs.append(
                build_cd_descendant_partial_format_reward(scale=float(args.cd_partial_format_reward_scale))
            )
        if float(args.cd_descendant_cot_structure_reward_scale) > 0.0:
            reward_funcs.append(
                build_cd_descendant_cot_structure_reward(scale=float(args.cd_descendant_cot_structure_reward_scale))
            )
        if float(args.cd_descendant_shift_ranking_reward_scale) > 0.0:
            reward_funcs.append(
                build_cd_descendant_shift_ranking_reward(scale=float(args.cd_descendant_shift_ranking_reward_scale))
            )
        if float(args.cd_descendant_variable_classification_reward_scale) > 0.0:
            reward_funcs.append(
                build_cd_descendant_variable_classification_reward(
                    scale=float(args.cd_descendant_variable_classification_reward_scale)
                )
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
    reward_extrema_snapshot = {"latest": {}}

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

    def _rollout_snippet(text: str, head: int = 120, tail: int = 120) -> str:
        """Return '{first head chars}....{last tail chars}' for a rollout completion."""
        s = str(text or "").replace("\n", " ")
        if len(s) <= head + tail + 4:
            return s
        return f"{s[:head]}....{s[-tail:]}"

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

    def _prompt_as_model_sees(text: str) -> str:
        """Tokenize the prompt then decode with special tokens preserved.

        This reflects what the model actually receives: applies the same
        max_prompt_tokens truncation used at generation time, so if the
        tokenizer adds BOS tokens or the text round-trips differently it will
        show up here. Also truncates at the end of the assistant-prefill
        <think> tag so the logged prompt does not include any prefilled
        chain-of-thought content.
        """
        if not text:
            return text
        try:
            max_len = int(args.max_prompt_tokens) if args.max_prompt_tokens and int(args.max_prompt_tokens) > 0 else None
            raw_ids = tokenizer(text, truncation=False)["input_ids"]
            ids = _middle_truncate_ids(raw_ids, max_len) if max_len else raw_ids
            decoded = tokenizer.decode(ids)  # skip_special_tokens=False by default
        except Exception:
            decoded = text
        # Prefer the final assistant-prefill boundary. Descendant prompts also
        # mention literal "<think>" tags inside the instructions, so trimming at
        # the first occurrence can cut the prompt before the assistant turn.
        assistant_prefill_matches = list(re.finditer(r"assistant\s*<think>", decoded))
        if assistant_prefill_matches:
            decoded = decoded[:assistant_prefill_matches[-1].end()]
            return decoded
        think_tag = "<think>"
        idx = decoded.rfind(think_tag)
        if idx != -1:
            decoded = decoded[:idx + len(think_tag)]
        return decoded

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
                completion_log_text = _truncate_text_for_log(completion_text)
                answer_log_text = _truncate_text_for_log(answer_text)
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
                    "prompt_model_input": _truncate_text_for_log(
                        _prompt_as_model_sees(prompt_texts[i] if i < len(prompt_texts) else "")
                    ),
                    "completion": completion_log_text,
                    "answer_text": answer_log_text,
                    "target_answer": _truncate_text_for_log(_value_at(answers, i) or _value_at(solutions, i) or ""),
                    "completion_chars": len(str(completion_text or "")),
                    "answer_text_chars": len(str(answer_text or "")),
                    "completion_truncated_for_eval_log": int(completion_log_text != str(completion_text or "")),
                    "answer_text_truncated_for_eval_log": int(answer_log_text != str(answer_text or "")),
                    "format_ok_scored_on_full_completion": 1,
                    "rewards": {},
                    "_sampled_for_log": i in sampled_indices,
                    "_sample_completion_full": str(completion_text or "") if i in sampled_indices else "",
                    "_sample_answer_text_full": str(answer_text or "") if i in sampled_indices else "",
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
            step_totals = []
            step_reward_values = {}
            try:
                for key in emit_keys:
                    row = pending_reward_rows.pop(key)
                    rewards_map = row.get("rewards", {})
                    row["reward_total"] = float(sum(float(v) for v in rewards_map.values()))
                    step_totals.append(float(row["reward_total"]))
                    for reward_key, reward_value in rewards_map.items():
                        step_reward_values.setdefault(str(reward_key), []).append(float(reward_value))
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
                            "format_ok_scored_on_full_completion": row["format_ok_scored_on_full_completion"],
                            "prompt": _truncate_sample_text(row["prompt"]),
                            "prompt_model_input": _truncate_sample_text(row["prompt_model_input"]),
                            "completion": _sanitize_for_log(row.get("_sample_completion_full", row["completion"]), 0),
                            "answer_text": _sanitize_for_log(row.get("_sample_answer_text_full", row["answer_text"]), 0),
                            "target_answer": _truncate_sample_text(row["target_answer"]),
                            "completion_chars": row["completion_chars"],
                            "answer_text_chars": row["answer_text_chars"],
                            "completion_truncated_for_eval_log": row["completion_truncated_for_eval_log"],
                            "answer_text_truncated_for_eval_log": row["answer_text_truncated_for_eval_log"],
                            "rewards": rewards_map,
                            "reward_total": row["reward_total"],
                        }
                        sample_file.write(json.dumps(sample_row, ensure_ascii=False) + "\n")
                payload = {}
                if step_totals:
                    payload["reward_min"] = min(step_totals)
                    payload["reward_max"] = max(step_totals)
                for reward_key, values in step_reward_values.items():
                    if not values:
                        continue
                    payload[f"rewards/{reward_key}/min"] = min(values)
                    payload[f"rewards/{reward_key}/max"] = max(values)
                reward_extrema_snapshot["latest"] = payload
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
                        if completions and int(os.environ.get("LOCAL_RANK", "0")) == 0:
                            step = int(train_state_snapshot.get("global_step") or 0)
                            text = _completion_to_text(completions[0])
                            print(f"[rollout step={step}] {_rollout_snippet(text)}", flush=True)
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
        grpo_kwargs["vllm_mode"] = args.vllm_mode
        if args.vllm_mode == "server":
            grpo_kwargs.update(
                {
                    "vllm_server_base_url": args.vllm_server_base_url,
                    "vllm_server_timeout": args.vllm_server_timeout,
                }
            )
        elif args.vllm_mode == "colocate":
            grpo_kwargs.update(
                {
                    "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                    "vllm_max_model_length": args.vllm_max_model_length,
                }
            )

    train_generation_kwargs = {}
    if args.stop_sequence:
        # "stop" is the vLLM kwarg; "stop_strings" is the HuggingFace generate kwarg
        # (transformers >= 4.46). Pass both so the right one is used regardless of backend.
        train_generation_kwargs["stop"] = args.stop_sequence
        if not args.use_vllm:
            train_generation_kwargs["stop_strings"] = args.stop_sequence
    if args.train_temperature > 0:
        train_generation_kwargs["temperature"] = args.train_temperature
        train_generation_kwargs["top_p"] = args.train_top_p
    training_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "remove_unused_columns": False,
        "num_train_epochs": args.num_train_epochs,
        "beta": float(args.grpo_beta),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_grad_norm": args.max_grad_norm,
        "ddp_find_unused_parameters": False,
        "bf16": torch.cuda.is_available(),
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "torch_compile": False,  # set True for ~20% speedup if your torch version supports it
        "max_prompt_length": args.max_prompt_tokens,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "generation_kwargs": train_generation_kwargs or None,
        "report_to": report_to,
        "run_name": args.run_name,
        "disable_tqdm": False,
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
    supported_training_kwargs = _filter_supported_kwargs(GRPOConfig.__init__, training_kwargs)
    dropped_training_kwargs = sorted(set(training_kwargs) - set(supported_training_kwargs))
    if dropped_training_kwargs and int(os.environ.get("RANK", "0")) == 0:
        print(f"[warn] dropping unsupported GRPOConfig kwargs: {', '.join(dropped_training_kwargs)}")
    training_args = GRPOConfig(**supported_training_kwargs)
    print(f"[train] generation stop sequences: {args.stop_sequence if args.stop_sequence else 'none'}")
    print(
        f"[train] generation sampling: temperature={args.train_temperature}, "
        f"top_p={args.train_top_p}"
    )

    # Resolve resume checkpoint before building the trainer so we can log it.
    resume_from_checkpoint = None
    if args.resume:
        resume_from_checkpoint = get_last_checkpoint(args.output_dir)
        if resume_from_checkpoint is None:
            raise ValueError(
                f"--resume was set but no checkpoint was found in {args.output_dir!r}. "
                "Run without --resume to start a fresh training run."
            )
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[train] resuming from checkpoint: {resume_from_checkpoint}")

    # Write launch command to output_dir immediately so it's saved even if training crashes.
    if int(os.environ.get("RANK", "0")) == 0:
        launch_cmd_path = Path(args.output_dir) / "launch_command.sh"
        launch_cmd_path.parent.mkdir(parents=True, exist_ok=True)
        launch_cmd_path.write_text(
            "#!/usr/bin/env bash\n" + _build_launch_command(argv) + "\n",
            encoding="utf-8",
        )

    try:
        callbacks = [
            CompactMetricsCallback(),
            RawMetricsCallback(
                edge_f1_scale=float(args.cd_edge_f1_reward_scale),
                shd_scale=float(args.cd_low_shd_reward_scale),
            ),
            RewardExtremaCallback(reward_extrema_snapshot),
            TrainStateSnapshotCallback(train_state_snapshot),
            SaveLaunchCommandCallback(argv),
        ]
        metrics_jsonl_path = (
            Path(args.train_log_jsonl)
            if args.train_log_jsonl
            else (logs_dir / "train_metrics.jsonl")
        )
        metrics_json_path = metrics_jsonl_path.with_suffix(".json")
        metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(JsonlMetricsCallback(str(metrics_jsonl_path)))
        callbacks.append(JsonMetricsCallback(str(metrics_json_path)))
        print(f"[train] JSONL logging enabled: {metrics_jsonl_path}")
        print(f"[train] JSON logging enabled: {metrics_json_path}")

        # Debug: verify the prompt the model actually sees (with special tokens)
        if int(os.environ.get("RANK", "0")) == 0:
            sample = train_dataset[0]["prompt"]
            ids = tokenizer(sample, return_tensors="pt")["input_ids"][0]
            decoded = tokenizer.decode(ids)  # keep special tokens to see chat template markers
            print(f"\n[DEBUG] First prompt token count: {len(ids)}")
            print(f"[DEBUG] Decoded (with special tokens):\n{decoded}\n")

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=callbacks,
        )
        _log_startup_timing("GRPOTrainer construction")
        # breakpoint()
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

    # Sanity-check: ensure at least one parameter requires grad before launching training.
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(
            "No trainable parameters found before trainer.train(). "
            "All parameters have requires_grad=False. "
            "Check that the LoRA adapter is loaded correctly and enable_input_require_grads() was called."
        )
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[train] trainable params: {len(trainable_params)} (first few: {trainable_params[:3]})")

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
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
    if args.cd_wrapper_mode is not None:
        args.cd_wrap_system_prompt = (args.cd_wrapper_mode == "chat")
    if args.task in {"causal_discovery", "cd_descendants"} and args.cd_response_format != "think_answer":
        raise ValueError(
            "GRPO causal-discovery tasks currently support only --cd-response-format think_answer, "
            "because the reward stack expects think/answer completions."
        )
    _maybe_enable_small_graph_logging(args, argv)
    _maybe_prepare_vllm_model(args)
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
