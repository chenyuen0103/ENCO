#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import inspect
import importlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any

try:
    from cd_generation.format import validate_sft_example
except ModuleNotFoundError:
    from experiments.cd_generation.format import validate_sft_example


def _set_csv_field_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10_000_000)


def _read_text_from_row(row: dict[str, Any], key_text: str, key_path: str) -> str:
    txt = (row.get(key_text) or "").strip()
    if txt:
        return txt
    p = (row.get(key_path) or "").strip()
    if not p:
        return ""
    return Path(p).read_text(encoding="utf-8")


def _load_answer_obj(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    s = str(raw or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        p = Path(s)
        if p.exists() and p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except OSError:
        pass
    return json.loads(s)


def _extract_answer_payload(answer_obj: Any) -> Any:
    if isinstance(answer_obj, dict) and "answer" in answer_obj:
        return answer_obj["answer"]
    return answer_obj


def _extract_adjacency_matrix(answer_payload: Any) -> list[list[int]]:
    if isinstance(answer_payload, dict) and "adjacency_matrix" in answer_payload:
        mat = answer_payload["adjacency_matrix"]
    else:
        mat = answer_payload
    if not isinstance(mat, list) or not mat:
        raise ValueError("missing adjacency_matrix")
    rows = [[int(x) for x in r] for r in mat]
    n = len(rows)
    if any(len(r) != n for r in rows):
        raise ValueError("adjacency_matrix must be square")
    for r in rows:
        for x in r:
            if x not in (0, 1):
                raise ValueError("adjacency_matrix must be binary")
    return rows


def _looks_quantized_model_id(model_name: str) -> bool:
    name = str(model_name or "").lower()
    quant_markers = ("4bit", "8bit", "bnb", "gptq", "awq", "gguf", "nf4")
    return any(tok in name for tok in quant_markers)


def _derive_non_quantized_model_id(model_name: str) -> str:
    name = str(model_name or "")
    patterns = [
        r"(?i)[-_]?bnb[-_]?4bit$",
        r"(?i)[-_]?bnb[-_]?8bit$",
        r"(?i)[-_]?4bit$",
        r"(?i)[-_]?8bit$",
        r"(?i)[-_]?gptq$",
        r"(?i)[-_]?awq$",
    ]
    candidate = name
    for pat in patterns:
        candidate = re.sub(pat, "", candidate)
    candidate = re.sub(r"[-_]{2,}", "-", candidate).rstrip("-_")
    return candidate or name


def _import_trl_sft() -> tuple[Any, Any]:
    """
    Import TRL SFT classes without letting TRL's vLLM compatibility shim fire.

    SFT in this script does not use vLLM, but recent TRL builds still probe the
    installed vLLM package during import and emit noisy compatibility warnings.
    Hiding vLLM for this import keeps the SFT path decoupled from local vLLM
    versions and avoids repeated per-rank warnings.
    """
    from transformers.utils import import_utils as tf_import_utils

    orig_is_pkg_available = tf_import_utils._is_package_available

    def _patched_is_package_available(package_name: str, *args: Any, **kwargs: Any) -> Any:
        return_version = kwargs.get("return_version", args[0] if args else False)
        if package_name in {"vllm", "vllm_ascend"}:
            return (False, "0.0.0") if return_version else False
        return orig_is_pkg_available(package_name, *args, **kwargs)

    tf_import_utils._is_package_available = _patched_is_package_available
    try:
        if "trl" in sys.modules:
            stale = [name for name in sys.modules if name == "trl" or name.startswith("trl.")]
            for name in stale:
                sys.modules.pop(name, None)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"TRL currently only supports vLLM version `0\.10\.2`.*",
                category=UserWarning,
            )
            trl = importlib.import_module("trl")
        return trl.SFTConfig, trl.SFTTrainer
    finally:
        tf_import_utils._is_package_available = orig_is_pkg_available


def _build_format_check_prompt(
    prompt_text: str,
    *,
    tokenizer: Any,
    enable_thinking: bool,
) -> str:
    if "<think>" in str(prompt_text or ""):
        return prompt_text
    if not enable_thinking:
        return prompt_text
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return prompt_text

    messages = [{"role": "user", "content": str(prompt_text)}]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        params = inspect.signature(tokenizer.apply_chat_template).parameters
        if "enable_thinking" in params:
            kwargs["enable_thinking"] = True
    except Exception:
        pass

    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(rendered, str):
            return rendered
    except TypeError:
        kwargs.pop("enable_thinking", None)
        try:
            rendered = tokenizer.apply_chat_template(messages, **kwargs)
            if isinstance(rendered, str):
                return rendered
        except Exception:
            pass
    except Exception:
        pass
    return prompt_text


def torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_train_log_path(sft_output_dir: Path, train_log_jsonl: Path | None) -> Path:
    if train_log_jsonl is not None:
        out = Path(train_log_jsonl)
    else:
        out = Path(sft_output_dir) / "train_metrics.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_eval_log_path(sft_output_dir: Path, eval_log_jsonl: Path | None) -> Path:
    if eval_log_jsonl is not None:
        out = Path(eval_log_jsonl)
    else:
        out = Path(sft_output_dir) / "eval_metrics.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_resume_checkpoint(
    sft_output_dir: Path,
    resume_from_checkpoint: str | None,
) -> str | None:
    value = str(resume_from_checkpoint or "").strip()
    if not value:
        return None

    if value.lower() != "latest":
        ckpt = Path(value).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt}")
        return str(ckpt)

    checkpoint_dirs: list[tuple[int, Path]] = []
    for child in Path(sft_output_dir).glob("checkpoint-*"):
        if not child.is_dir():
            continue
        try:
            step = int(child.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        checkpoint_dirs.append((step, child.resolve()))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint-* directories found under {Path(sft_output_dir).resolve()}")
    checkpoint_dirs.sort(key=lambda item: item[0])
    return str(checkpoint_dirs[-1][1])


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return dict(kwargs)


def _load_json_dataset(
    jsonl_path: Path,
    *,
    keep_in_memory: bool,
):
    from datasets import load_dataset

    return load_dataset(
        "json",
        **_filter_supported_kwargs(
            load_dataset,
            {
                "data_files": str(Path(jsonl_path).resolve()),
                "split": "train",
                "keep_in_memory": keep_in_memory,
            },
        ),
    )


def _prepare_train_eval_datasets(
    *,
    train_jsonl: Path,
    eval_jsonl: Path | None,
    validation_split_ratio: float,
    keep_in_memory: bool,
):
    if validation_split_ratio < 0.0 or validation_split_ratio >= 1.0:
        raise ValueError("--validation-split-ratio must be in [0.0, 1.0)")

    train_ds = _load_json_dataset(train_jsonl, keep_in_memory=keep_in_memory)
    eval_ds = None

    if eval_jsonl is not None:
        eval_ds = _load_json_dataset(eval_jsonl, keep_in_memory=keep_in_memory)
        return train_ds, eval_ds

    if validation_split_ratio > 0.0:
        split = train_ds.train_test_split(
            **_filter_supported_kwargs(
                train_ds.train_test_split,
                {
                    "test_size": float(validation_split_ratio),
                    "seed": 42,
                    "shuffle": True,
                },
            ),
        )
        train_ds = split["train"]
        eval_ds = split["test"]

    return train_ds, eval_ds


def _normalize_prompt_answer_dataset(ds: Any, *, dataset_label: str) -> Any:
    if "prompt" not in ds.column_names or "answer" not in ds.column_names:
        raise ValueError(f"{dataset_label} JSONL must have 'prompt' and 'answer' columns; got {ds.column_names}")
    ds = ds.remove_columns([c for c in ds.column_names if c not in {"prompt", "answer"}])
    return ds.rename_column("answer", "completion")


def _build_eval_and_early_stopping_kwargs(
    *,
    eval_dataset_present: bool,
    eval_every: int | None,
    save_steps: int,
    early_stopping_patience: int | None,
    early_stopping_threshold: float,
    load_best_model_at_end: bool,
    metric_for_best_model: str,
    greater_is_better: bool,
) -> tuple[dict[str, Any], list[Any]]:
    callbacks: list[Any] = []
    if not eval_dataset_present:
        return {}, callbacks

    actual_eval_steps = int(eval_every or save_steps)
    kwargs: dict[str, Any] = {
        "do_eval": True,
        "eval_strategy": "steps",
        "eval_steps": actual_eval_steps,
    }
    use_best_model = True
    if use_best_model:
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = metric_for_best_model
        kwargs["greater_is_better"] = greater_is_better
        kwargs["save_strategy"] = "steps"
        kwargs["save_steps"] = actual_eval_steps
    if early_stopping_patience is not None:
        from transformers import EarlyStoppingCallback

        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(early_stopping_patience),
                early_stopping_threshold=float(early_stopping_threshold),
            )
        )
    return kwargs, callbacks


def _validate_sft_jsonl(path: Path, *, allow_legacy_text: bool = False, sample_limit: int = 8) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"SFT JSONL not found: {path}")
    if not path.is_file():
        raise ValueError(f"SFT JSONL is not a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(
            f"SFT JSONL is empty: {path}. "
            "Regenerate it before training; for staged CD data use cd_sft/staged_targets.py "
            "or collect_format_sft_data.py."
        )

    total = 0
    prompt_answer_rows = 0
    text_rows = 0
    sample_keys: set[str] = set()
    sample_issues: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            total += 1
            try:
                row = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"SFT JSONL has invalid JSON on line {lineno}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"SFT JSONL line {lineno} must decode to an object, got {type(row).__name__}")

            sample_keys.update(str(k) for k in row.keys())
            has_prompt_answer = "prompt" in row and "answer" in row
            has_text = isinstance(row.get("text"), str) and bool(str(row.get("text") or "").strip())
            if has_prompt_answer:
                prompt_answer_rows += 1
                if len(sample_issues) < sample_limit:
                    issues = validate_sft_example(str(row.get("prompt") or ""), str(row.get("answer") or ""))
                    if issues:
                        sample_issues.append(f"line {lineno}: {'; '.join(issues)}")
            if has_text:
                text_rows += 1

    if total == 0:
        raise ValueError(
            f"SFT JSONL has no non-empty records: {path}. "
            "Regenerate it before training."
        )
    if sample_issues:
        raise ValueError(
            f"SFT JSONL sample validation failed for {path}:\n- " + "\n- ".join(sample_issues)
        )
    if allow_legacy_text:
        if prompt_answer_rows == 0 and text_rows == 0:
            raise ValueError(
                f"SFT JSONL must contain either 'prompt'+'answer' or legacy 'text' rows; "
                f"observed keys: {sorted(sample_keys)}"
            )
    elif prompt_answer_rows == 0:
        raise ValueError(
            f"SFT JSONL must contain 'prompt' and 'answer' columns; observed keys: {sorted(sample_keys)}. "
            "Use cd_sft/staged_targets.py to produce a compatible JSONL."
        )

    return {
        "rows": total,
        "prompt_answer_rows": prompt_answer_rows,
        "text_rows": text_rows,
        "keys": sorted(sample_keys),
    }


def _build_jsonl_metrics_callback(
    output_path: Path,
    *,
    include_prefixes: tuple[str, ...] | None = None,
    exclude_prefixes: tuple[str, ...] | None = None,
):
    from transformers import TrainerCallback

    class _JsonlMetricsCallback(TrainerCallback):
        def __init__(self, p: Path):
            self.output_path = Path(p)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

        @staticmethod
        def _to_jsonable(value: Any) -> Any:
            if value is None or isinstance(value, (bool, int, float, str)):
                return value
            if hasattr(value, "item"):
                try:
                    return _JsonlMetricsCallback._to_jsonable(value.item())
                except Exception:
                    pass
            return str(value)

        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
            if not state.is_local_process_zero or not logs:
                return
            filtered_logs = {
                k: v
                for k, v in logs.items()
                if (not include_prefixes or any(str(k).startswith(prefix) for prefix in include_prefixes))
                and (not exclude_prefixes or not any(str(k).startswith(prefix) for prefix in exclude_prefixes))
            }
            if not filtered_logs:
                return
            payload = {
                "global_step": int(state.global_step),
                "epoch": float(state.epoch) if state.epoch is not None else None,
                "time_unix": time.time(),
            }
            payload.update({k: self._to_jsonable(v) for k, v in filtered_logs.items()})
            with self.output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")

    return _JsonlMetricsCallback(output_path)


def _run_sft_unsloth(
    *,
    base_model: str,
    sft_jsonl: Path,
    sft_output_dir: Path,
    sft_epochs: float,
    sft_lr: float,
    sft_batch_size: int,
    sft_eval_batch_size: int,
    sft_grad_accum: int,
    sft_max_seq_length: int,
    sft_save_steps: int,
    sft_logging_steps: int,
    sft_save_total_limit: int | None = None,
    lora_r: int,
    lora_alpha: int,
    train_log_jsonl: Path | None = None,
    eval_jsonl: Path | None = None,
    eval_log_jsonl: Path | None = None,
    validation_split_ratio: float = 0.0,
    eval_every: int | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_threshold: float = 0.0,
    load_best_model_at_end: bool = False,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    grpo_base_model: str | None = None,
    resume_from_checkpoint: str | None = None,
) -> None:
    """Single-GPU SFT via Unsloth — ~2× faster, supports 4-bit to reduce VRAM."""
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    import torch
    distributed_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    keep_dataset_in_memory = distributed_world_size > 1

    is_adapter = (Path(base_model) / "adapter_config.json").exists()
    print(f"[sft/unsloth] loading {'adapter' if is_adapter else 'base'} model: {base_model!r}")

    if is_adapter:
        # Read the base model name from adapter config and load via Unsloth
        adapter_cfg = json.loads((Path(base_model) / "adapter_config.json").read_text())
        base_id = adapter_cfg.get("base_model_name_or_path", base_model)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_id,
            max_seq_length=sft_max_seq_length,
            load_in_4bit=True,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, base_model, is_trainable=True)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=sft_max_seq_length,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[sft/unsloth] trainable={trainable} total={total} ({100.0 * trainable / max(total, 1):.4f}%)")

    # --- dataset ---
    if keep_dataset_in_memory:
        print("[sft/unsloth] distributed mode: keeping dataset in memory and bypassing Arrow cache files")
    train_ds, eval_ds = _prepare_train_eval_datasets(
        train_jsonl=sft_jsonl,
        eval_jsonl=eval_jsonl,
        validation_split_ratio=validation_split_ratio,
        keep_in_memory=keep_dataset_in_memory,
    )
    train_ds = _normalize_prompt_answer_dataset(train_ds, dataset_label="train")
    if eval_ds is not None:
        eval_ds = _normalize_prompt_answer_dataset(eval_ds, dataset_label="eval")

    eos_token_id = int(tokenizer.eos_token_id)
    max_len = int(sft_max_seq_length)

    def _tokenize(example: dict[str, Any]) -> dict[str, Any]:
        prompt = str(example.get("prompt") or "")
        completion = str(example.get("completion") or "")
        issues = validate_sft_example(prompt, completion)
        if issues:
            raise ValueError("; ".join(issues))
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if not completion_ids:
            raise ValueError("empty completion")
        if completion_ids[-1] != eos_token_id:
            completion_ids = [*completion_ids, eos_token_id]
        if len(completion_ids) >= max_len:
            completion_ids = completion_ids[:max_len]
            prompt_ids = []
        else:
            keep = max_len - len(completion_ids)
            if len(prompt_ids) > keep:
                prompt_ids = prompt_ids[-keep:]
        input_ids = [*prompt_ids, *completion_ids]
        labels = ([-100] * len(prompt_ids)) + list(completion_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    tokenized_ds = train_ds.map(
        _tokenize,
        remove_columns=train_ds.column_names,
        desc="Tokenizing",
        **_filter_supported_kwargs(
            train_ds.map,
            {
                "keep_in_memory": keep_dataset_in_memory,
                "load_from_cache_file": not keep_dataset_in_memory,
            },
        ),
    )
    tokenized_eval_ds = None
    if eval_ds is not None:
        tokenized_eval_ds = eval_ds.map(
            _tokenize,
            remove_columns=eval_ds.column_names,
            desc="Tokenizing eval",
            **_filter_supported_kwargs(
                eval_ds.map,
                {
                    "keep_in_memory": keep_dataset_in_memory,
                    "load_from_cache_file": not keep_dataset_in_memory,
                },
            ),
        )
    supervised_total = sum(
        sum(1 for x in row["labels"] if x != -100)
        for row in tokenized_ds
    )
    print(f"[sft/unsloth] rows={len(tokenized_ds)} supervised_tokens_total={supervised_total}")
    if tokenized_eval_ds is not None:
        eval_supervised_total = sum(
            sum(1 for x in row["labels"] if x != -100)
            for row in tokenized_eval_ds
        )
        print(
            f"[sft/unsloth] eval_rows={len(tokenized_eval_ds)} "
            f"eval_supervised_tokens_total={eval_supervised_total}"
        )

    SFTConfig, SFTTrainer = _import_trl_sft()
    metrics_jsonl_path = _resolve_train_log_path(sft_output_dir, train_log_jsonl)
    print(f"[sft/unsloth] JSONL logging enabled: {metrics_jsonl_path}")
    eval_kwargs, extra_callbacks = _build_eval_and_early_stopping_kwargs(
        eval_dataset_present=tokenized_eval_ds is not None,
        eval_every=eval_every,
        save_steps=int(sft_save_steps),
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
    )
    callbacks = [_build_jsonl_metrics_callback(metrics_jsonl_path)]
    if tokenized_eval_ds is not None:
        eval_metrics_jsonl_path = _resolve_eval_log_path(sft_output_dir, eval_log_jsonl)
        print(f"[sft/unsloth] eval metrics path: {eval_metrics_jsonl_path}")
        callbacks.append(_build_jsonl_metrics_callback(eval_metrics_jsonl_path, include_prefixes=("eval_",)))
    callbacks.extend(extra_callbacks)
    cfg_kwargs = {
        "output_dir": str(sft_output_dir),
        "num_train_epochs": float(sft_epochs),
        "learning_rate": float(sft_lr),
        "per_device_train_batch_size": int(sft_batch_size),
        "per_device_eval_batch_size": int(sft_eval_batch_size),
        "gradient_accumulation_steps": int(sft_grad_accum),
        "logging_steps": int(sft_logging_steps),
        "save_steps": int(sft_save_steps),
        "bf16": is_bfloat16_supported(),
        "fp16": not is_bfloat16_supported(),
        "report_to": "none",
        "remove_unused_columns": False,
        "save_total_limit": sft_save_total_limit,
        "max_seq_length": max_len,
        "dataset_text_field": None,
        "packing": False,
    }
    cfg_kwargs.update(eval_kwargs)
    cfg = SFTConfig(**_filter_supported_kwargs(SFTConfig.__init__, cfg_kwargs))
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_ds,
        eval_dataset=tokenized_eval_ds,
        args=cfg,
        callbacks=callbacks,
    )
    resume_ckpt = _resolve_resume_checkpoint(sft_output_dir, resume_from_checkpoint)
    if resume_ckpt:
        print(f"[sft/unsloth] resuming from checkpoint: {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    model.save_pretrained(str(sft_output_dir))
    tokenizer.save_pretrained(str(sft_output_dir))

    # Patch adapter_config.json so GRPO can load it without a quantized base.
    if grpo_base_model:
        adapter_cfg_path = sft_output_dir / "adapter_config.json"
        if adapter_cfg_path.exists():
            cfg_data = json.loads(adapter_cfg_path.read_text())
            old_base = cfg_data.get("base_model_name_or_path", "")
            cfg_data["base_model_name_or_path"] = grpo_base_model
            adapter_cfg_path.write_text(json.dumps(cfg_data, indent=2))
            print(f"[sft/unsloth] patched adapter_config.json: {old_base!r} -> {grpo_base_model!r}")

    print(f"[sft/unsloth] saved -> {sft_output_dir}")


def run_sft(
    *,
    base_model: str,
    distributed_base_model: str | None,
    sft_jsonl: Path,
    sft_output_dir: Path,
    sft_epochs: float,
    sft_lr: float,
    sft_batch_size: int,
    sft_eval_batch_size: int,
    sft_grad_accum: int,
    sft_max_seq_length: int,
    sft_save_steps: int,
    sft_logging_steps: int,
    sft_save_total_limit: int | None,
    train_log_jsonl: Path | None,
    eval_jsonl: Path | None = None,
    eval_log_jsonl: Path | None = None,
    validation_split_ratio: float = 0.0,
    eval_every: int | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_threshold: float = 0.0,
    load_best_model_at_end: bool = False,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    use_unsloth: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 16,
    grpo_base_model: str | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    resume_from_checkpoint: str | None = None,
) -> None:
    sft_summary = _validate_sft_jsonl(sft_jsonl)
    print(
        f"[sft] validated dataset: rows={sft_summary['rows']} "
        f"prompt_answer_rows={sft_summary['prompt_answer_rows']} path={Path(sft_jsonl).resolve()}"
    )
    if eval_jsonl is not None:
        eval_summary = _validate_sft_jsonl(eval_jsonl)
        print(
            f"[sft] validated eval dataset: rows={eval_summary['rows']} "
            f"prompt_answer_rows={eval_summary['prompt_answer_rows']} path={Path(eval_jsonl).resolve()}"
        )

    if use_unsloth:
        _run_sft_unsloth(
            base_model=base_model,
            sft_jsonl=sft_jsonl,
            sft_output_dir=sft_output_dir,
            sft_epochs=sft_epochs,
            sft_lr=sft_lr,
            sft_batch_size=sft_batch_size,
            sft_eval_batch_size=sft_eval_batch_size,
            sft_grad_accum=sft_grad_accum,
            sft_max_seq_length=sft_max_seq_length,
            sft_save_steps=sft_save_steps,
            sft_logging_steps=sft_logging_steps,
            sft_save_total_limit=sft_save_total_limit,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            train_log_jsonl=train_log_jsonl,
            eval_jsonl=eval_jsonl,
            eval_log_jsonl=eval_log_jsonl,
            validation_split_ratio=validation_split_ratio,
            eval_every=eval_every,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            grpo_base_model=grpo_base_model,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        return

    import torch
    from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    is_adapter = (Path(base_model) / "adapter_config.json").exists()
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    distributed_world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # DDP (torchrun, WORLD_SIZE>1) cannot handle 4-bit quantized models.
    # Force bfloat16 so each rank loads a full-precision copy on its own GPU.
    if distributed_world_size > 1:
        load_dtype = torch.bfloat16
        print(
            f"[sft] distributed (WORLD_SIZE={distributed_world_size}): "
            "forcing dtype=bfloat16 (4-bit incompatible with DDP)"
        )
    else:
        load_dtype = "auto"

    model_load_kwargs: dict[str, Any] = {
        "dtype": load_dtype,
        "attn_implementation": "sdpa",
    }

    if is_adapter:
        if distributed_world_size > 1:
            adapter_cfg_path = Path(base_model) / "adapter_config.json"
            adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
            base_model_name = adapter_cfg.get("base_model_name_or_path", "")
            resolved = str(base_model_name or "")
            if _looks_quantized_model_id(resolved):
                resolved = distributed_base_model or _derive_non_quantized_model_id(resolved)
                print(f"[sft] distributed: adapter points to quantized base; switching to {resolved!r}")
            else:
                print(f"[sft] distributed: loading base {resolved!r} in bfloat16, then applying adapter")
            base = AutoModelForCausalLM.from_pretrained(resolved, **model_load_kwargs)
            model = PeftModel.from_pretrained(base, base_model, is_trainable=True)
        else:
            adapter_load_kwargs = dict(model_load_kwargs)
            try:
                if "is_trainable" in inspect.signature(AutoPeftModelForCausalLM.from_pretrained).parameters:
                    adapter_load_kwargs["is_trainable"] = True
            except Exception:
                pass
            model = AutoPeftModelForCausalLM.from_pretrained(base_model, **adapter_load_kwargs)
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
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_load_kwargs)

    # Resolve EOS token to an in-vocab token.
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    for tok in [getattr(tokenizer, "eos_token", None), "<|im_end|>", "<|endoftext|>"]:
        if tok and isinstance(vocab, dict) and tok in vocab:
            tokenizer.eos_token = tok
            try:
                tokenizer.eos_token_id = int(vocab[tok])
            except Exception:
                pass
            break
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    if not is_adapter:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)

    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
    else:
        try:
            emb = model.get_input_embeddings()
            if emb is not None:
                def _require_grad_hook(module, inputs, output):
                    if hasattr(output, "requires_grad_"):
                        output.requires_grad_(True)
                    return output
                emb.register_forward_hook(_require_grad_hook)
        except Exception:
            pass

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[sft] trainable={trainable} total={total} ({100.0 * trainable / max(total, 1):.4f}%)")

    keep_dataset_in_memory = distributed_world_size > 1
    if keep_dataset_in_memory:
        print("[sft] distributed mode: keeping dataset in memory and bypassing Arrow cache files")
    train_ds, eval_ds = _prepare_train_eval_datasets(
        train_jsonl=sft_jsonl,
        eval_jsonl=eval_jsonl,
        validation_split_ratio=validation_split_ratio,
        keep_in_memory=keep_dataset_in_memory,
    )
    train_ds = _normalize_prompt_answer_dataset(train_ds, dataset_label="train")
    if eval_ds is not None:
        eval_ds = _normalize_prompt_answer_dataset(eval_ds, dataset_label="eval")

    eos_token_id = int(tokenizer.eos_token_id)
    eos_id_check = tokenizer.convert_tokens_to_ids(tokenizer.eos_token) if getattr(tokenizer, "eos_token", None) else None
    if eos_id_check is None:
        raise ValueError(
            f"eos_token not in vocab: {getattr(tokenizer, 'eos_token', None)!r}"
        )
    print(f"[sft] eos_token={tokenizer.eos_token!r} eos_token_id={eos_id_check}")

    max_len = int(sft_max_seq_length)

    def _tokenize(example: dict[str, Any]) -> dict[str, Any]:
        prompt = str(example.get("prompt") or "")
        completion = str(example.get("completion") or "")
        issues = validate_sft_example(prompt, completion)
        if issues:
            raise ValueError("; ".join(issues))

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if not completion_ids:
            raise ValueError("completion tokenization produced zero tokens")
        if completion_ids[-1] != eos_token_id:
            completion_ids = [*completion_ids, eos_token_id]

        if len(completion_ids) >= max_len:
            completion_ids = completion_ids[:max_len]
            prompt_ids = []
        else:
            keep = max_len - len(completion_ids)
            if len(prompt_ids) > keep:
                prompt_ids = prompt_ids[-keep:]

        input_ids = [*prompt_ids, *completion_ids]
        labels = ([-100] * len(prompt_ids)) + list(completion_ids)
        supervised = sum(1 for x in labels if x != -100)
        if supervised <= 0:
            raise ValueError("no supervised completion tokens")
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "__prompt_tokens": len(prompt_ids),
            "__completion_tokens": len(completion_ids),
            "__supervised_tokens": supervised,
        }

    tokenized_ds = train_ds.map(
        _tokenize,
        remove_columns=train_ds.column_names,
        desc="Tokenizing",
        **_filter_supported_kwargs(
            train_ds.map,
            {
                "keep_in_memory": keep_dataset_in_memory,
                "load_from_cache_file": not keep_dataset_in_memory,
            },
        ),
    )
    tokenized_eval_ds = None
    if eval_ds is not None:
        tokenized_eval_ds = eval_ds.map(
            _tokenize,
            remove_columns=eval_ds.column_names,
            desc="Tokenizing eval",
            **_filter_supported_kwargs(
                eval_ds.map,
                {
                    "keep_in_memory": keep_dataset_in_memory,
                    "load_from_cache_file": not keep_dataset_in_memory,
                },
            ),
        )
    supervised_total = sum(int(x) for x in tokenized_ds["__supervised_tokens"])
    if supervised_total <= 0:
        raise ValueError("dataset has zero supervised tokens")
    prompt_mean = sum(int(x) for x in tokenized_ds["__prompt_tokens"]) / max(len(tokenized_ds), 1)
    comp_mean = sum(int(x) for x in tokenized_ds["__completion_tokens"]) / max(len(tokenized_ds), 1)
    print(
        f"[sft] rows={len(tokenized_ds)} "
        f"prompt_tokens_mean={prompt_mean:.1f} "
        f"completion_tokens_mean={comp_mean:.1f} "
        f"supervised_tokens_total={supervised_total}"
    )
    if tokenized_eval_ds is not None:
        eval_supervised_total = sum(int(x) for x in tokenized_eval_ds["__supervised_tokens"])
        eval_prompt_mean = sum(int(x) for x in tokenized_eval_ds["__prompt_tokens"]) / max(len(tokenized_eval_ds), 1)
        eval_comp_mean = sum(int(x) for x in tokenized_eval_ds["__completion_tokens"]) / max(len(tokenized_eval_ds), 1)
        print(
            f"[sft] eval_rows={len(tokenized_eval_ds)} "
            f"eval_prompt_tokens_mean={eval_prompt_mean:.1f} "
            f"eval_completion_tokens_mean={eval_comp_mean:.1f} "
            f"eval_supervised_tokens_total={eval_supervised_total}"
        )

    class _Collator:
        def __init__(self, pad_token_id: int):
            self.pad_token_id = int(pad_token_id)

        def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
            max_l = max(len(f["input_ids"]) for f in features)
            batch_input_ids, batch_attn, batch_labels = [], [], []
            for feat in features:
                pad = max_l - len(feat["input_ids"])
                batch_input_ids.append(list(feat["input_ids"]) + [self.pad_token_id] * pad)
                batch_attn.append(list(feat["attention_mask"]) + [0] * pad)
                batch_labels.append(list(feat["labels"]) + [-100] * pad)
            return {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
                "labels": torch.tensor(batch_labels, dtype=torch.long),
            }

    eval_kwargs, extra_callbacks = _build_eval_and_early_stopping_kwargs(
        eval_dataset_present=tokenized_eval_ds is not None,
        eval_every=eval_every,
        save_steps=int(sft_save_steps),
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
    )
    train_args_kwargs = {
        "output_dir": str(sft_output_dir),
        "num_train_epochs": float(sft_epochs),
        "learning_rate": float(sft_lr),
        "per_device_train_batch_size": int(sft_batch_size),
        "per_device_eval_batch_size": int(sft_eval_batch_size),
        "gradient_accumulation_steps": int(sft_grad_accum),
        "logging_steps": int(sft_logging_steps),
        "save_steps": int(sft_save_steps),
        "bf16": torch_cuda_available(),
        "report_to": "wandb" if wandb_project else "none",
        "run_name": wandb_run_name,
        "remove_unused_columns": False,
        "save_total_limit": sft_save_total_limit,
        # Non-reentrant gradient checkpointing avoids the DDP + LoRA
        # "marked ready twice" error caused by reentrant recomputation
        # firing DDP all-reduce hooks multiple times per parameter.
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "ddp_find_unused_parameters": False,
    }
    train_args_kwargs.update(eval_kwargs)
    train_args = TrainingArguments(**_filter_supported_kwargs(TrainingArguments.__init__, train_args_kwargs))
    metrics_jsonl_path = _resolve_train_log_path(sft_output_dir, train_log_jsonl)
    print(f"[sft] JSONL logging enabled: {metrics_jsonl_path}")
    callbacks = [_build_jsonl_metrics_callback(metrics_jsonl_path)]
    if tokenized_eval_ds is not None:
        eval_metrics_jsonl_path = _resolve_eval_log_path(sft_output_dir, eval_log_jsonl)
        print(f"[sft] eval metrics path: {eval_metrics_jsonl_path}")
        callbacks.append(_build_jsonl_metrics_callback(eval_metrics_jsonl_path, include_prefixes=("eval_",)))
    callbacks.extend(extra_callbacks)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_ds,
        eval_dataset=tokenized_eval_ds,
        data_collator=_Collator(tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    resume_ckpt = _resolve_resume_checkpoint(sft_output_dir, resume_from_checkpoint)
    if resume_ckpt:
        print(f"[sft] resuming from checkpoint: {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(sft_output_dir))
    tokenizer.save_pretrained(str(sft_output_dir))


def run_grpo(
    *,
    python_exe: str,
    nproc_per_node: int,
    grpo_script: Path,
    sft_output_dir: Path,
    grpo_output_dir: Path,
    grpo_train_csv: Path,
    extra_args: list[str],
    env: dict[str, str] | None = None,
) -> None:
    cmd = [
        "torchrun",
        "--nproc_per_node", str(int(nproc_per_node)),
        str(grpo_script),
        "--task", "causal_discovery",
        "--model_id", str(sft_output_dir),
        "--cd-train-csv", str(grpo_train_csv),
        "--output_dir", str(grpo_output_dir),
    ]
    cmd.extend(extra_args)
    print("[run]", " ".join(cmd))
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    subprocess.run(cmd, env=child_env, check=True)


def merge_sft_adapter(
    *,
    sft_model_dir: Path,
    merged_output_dir: Path,
) -> Path:
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    sft_model_dir = Path(sft_model_dir).resolve()
    merged_output_dir = Path(merged_output_dir).resolve()

    if not (sft_model_dir / "adapter_config.json").exists():
        raise ValueError(f"{sft_model_dir} is not a PEFT adapter directory (missing adapter_config.json)")

    adapter_cfg = json.loads((sft_model_dir / "adapter_config.json").read_text(encoding="utf-8"))
    tokenizer_source = sft_model_dir
    if not (sft_model_dir / "tokenizer.json").exists() and not (sft_model_dir / "vocab.json").exists():
        base_model_name = str(adapter_cfg.get("base_model_name_or_path") or "").strip()
        if not base_model_name:
            raise ValueError(
                f"{sft_model_dir} is missing tokenizer files and adapter_config.json does not declare "
                "base_model_name_or_path"
            )
        tokenizer_source = Path(base_model_name) if Path(base_model_name).exists() else Path(str(base_model_name))

    print(f"[merge] loading adapter from {sft_model_dir}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(sft_model_dir),
        torch_dtype="auto",
    )
    merged = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), use_fast=True)

    merged_output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_output_dir))
    tokenizer.save_pretrained(str(merged_output_dir))
    print(f"[merge] saved merged model -> {merged_output_dir}")
    return merged_output_dir


FORMAT_RE = re.compile(r"(?s)^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$")


def run_format_check(
    *,
    sft_model_dir: Path,
    train_csv: Path,
    prompt_text_col: str,
    prompt_path_col: str,
    answer_col: str,
    answer_path_col: str,
    max_rows: int,
    max_new_tokens: int,
    output_path: Path,
    sft_max_seq_length: int,
    format_check_enable_thinking: bool,
) -> None:
    import torch
    from tqdm.auto import tqdm
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
    from verifier_cd import extract_adjacency_matrix

    _set_csv_field_limit()
    rows = list(csv.DictReader(train_csv.open("r", encoding="utf-8", newline="")))
    if max_rows > 0:
        rows = rows[:max_rows]
    if not rows:
        raise RuntimeError(f"No rows found in {train_csv}")

    is_adapter = (Path(sft_model_dir) / "adapter_config.json").exists()
    load_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() else "auto",
        "attn_implementation": "sdpa",
        "device_map": "auto",
    }
    if is_adapter:
        model = AutoPeftModelForCausalLM.from_pretrained(str(sft_model_dir), **load_kwargs).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(str(sft_model_dir), **load_kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(sft_model_dir), use_fast=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stop_phrases = ["</answer>", "\n</answer>", "\n\n</answer>", " </answer>"]
    stop_token_suffixes: list[list[int]] = [
        ids for phrase in stop_phrases
        if (ids := tokenizer.encode(phrase, add_special_tokens=False))
    ]

    class _StopOnSuffix(StoppingCriteria):
        def __init__(self, suffixes: list[list[int]]):
            super().__init__()
            self.suffixes = [s for s in suffixes if s]

        def __call__(self, input_ids, scores, **kwargs):
            if input_ids is None or input_ids.numel() == 0:
                return False
            seq = input_ids[0].tolist()
            return any(
                len(seq) >= len(suf) and seq[-len(suf):] == suf
                for suf in self.suffixes
            )

    stopping_criteria = (
        StoppingCriteriaList([_StopOnSuffix(stop_token_suffixes)])
        if stop_token_suffixes else None
    )

    n = fmt_ok = parse_ok = 0
    with output_path.open("w", encoding="utf-8") as fout:
        progress = tqdm(rows, total=len(rows), desc="check", unit="row", dynamic_ncols=True)
        for i, row in enumerate(progress):
            prompt = _read_text_from_row(row, prompt_text_col, prompt_path_col).strip()
            prompt_for_model = _build_format_check_prompt(
                prompt, tokenizer=tokenizer, enable_thinking=bool(format_check_enable_thinking)
            )
            answer_raw = row.get(answer_col) or row.get(answer_path_col)
            answer_obj = _load_answer_obj(answer_raw)
            answer_payload = _extract_answer_payload(answer_obj)
            try:
                expected_n = len(_extract_adjacency_matrix(answer_payload))
            except Exception:
                expected_n = None

            inputs = tokenizer(prompt_for_model, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": False,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if stopping_criteria is not None:
                gen_kwargs["stopping_criteria"] = stopping_criteria
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)

            prompt_len = int(inputs["input_ids"].shape[-1])
            gen_ids = out[0][prompt_len:]
            resp = tokenizer.decode(gen_ids, skip_special_tokens=True)
            generated_tokens = int(gen_ids.shape[-1])
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            eos_seen = eos_token_id is not None and any(
                int(tok) == int(eos_token_id) for tok in gen_ids.tolist()
            )

            is_fmt_ok = int(bool(FORMAT_RE.match(resp or "")))
            mat = extract_adjacency_matrix(resp, expected_n=expected_n)
            is_parse_ok = int(mat is not None)

            n += 1
            fmt_ok += is_fmt_ok
            parse_ok += is_parse_ok

            rec = {
                "row_idx": i,
                "data_idx": row.get("data_idx"),
                "shuffle_idx": row.get("shuffle_idx"),
                "prompt_tokens": prompt_len,
                "generated_tokens": generated_tokens,
                "eos_seen": int(eos_seen),
                "answer_closed": int((resp or "").rstrip().endswith("</answer>")),
                "format_ok": is_fmt_ok,
                "parse_ok": is_parse_ok,
                "response_chars": len(resp),
                "response": resp,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            progress.set_postfix(
                format_ok=f"{fmt_ok}/{n}",
                parse_ok=f"{parse_ok}/{n}",
                prompt_tokens=prompt_len,
                gen_tokens=generated_tokens,
            )

    fmt_rate = (fmt_ok / n) if n > 0 else 0.0
    parse_rate = (parse_ok / n) if n > 0 else 0.0
    print(
        f"[check] rows={n} format_ok={fmt_ok} ({fmt_rate:.3f}) "
        f"parse_ok={parse_ok} ({parse_rate:.3f}) -> {output_path}"
    )


def _mean_or_zero(values: list[float]) -> float:
    return sum(float(v) for v in values) / len(values) if values else 0.0


def _list_grpo_readiness_candidates(sft_output_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for child in Path(sft_output_dir).glob("checkpoint-*"):
        if child.is_dir():
            candidates.append(child.resolve())
    candidates.sort(key=lambda p: int(p.name.rsplit("-", 1)[-1]) if p.name.rsplit("-", 1)[-1].isdigit() else 10**18)

    root = Path(sft_output_dir).resolve()
    if (root / "adapter_config.json").exists() or (root / "config.json").exists():
        if root not in candidates:
            candidates.append(root)
    return candidates


def _parse_cuda_visible_devices(value: str | None) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _summarize_grpo_readiness_eval(eval_jsonl_path: Path, checkpoint_dir: Path, sft_output_dir: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in Path(eval_jsonl_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"No rows found in readiness eval output: {eval_jsonl_path}")

    prompt_summaries = [dict(row.get("rollout_summary") or {}) for row in rows]
    rollouts = [rollout for row in rows for rollout in list(row.get("rollouts") or [])]
    if not rollouts:
        rollouts = [row for row in rows]

    rollout_tags_correct_rate = _mean_or_zero([float(item.get("tags_correct", 0.0)) for item in rollouts])
    rollout_payload_present_rate = _mean_or_zero([float(item.get("payload_present", 0.0)) for item in rollouts])
    rollout_exact_match_rate = _mean_or_zero([float(item.get("exact_match", 0.0)) for item in rollouts])
    rollout_primary_f1_mean = _mean_or_zero([float(item.get("primary_f1", 0.0)) for item in rollouts])
    rollout_prompt_copy_rate = _mean_or_zero([float(item.get("prompt_copy", 0.0)) for item in rollouts])
    prompt_all_rollouts_scoreable_rate = _mean_or_zero(
        [float(item.get("all_rollouts_scoreable", 0.0)) for item in prompt_summaries]
    )
    prompt_any_rollout_scoreable_rate = _mean_or_zero(
        [float(item.get("any_rollout_scoreable", 0.0)) for item in prompt_summaries]
    )
    prompt_completion_collapsed_rate = _mean_or_zero(
        [float(item.get("completion_collapse", 0.0)) for item in prompt_summaries]
    )
    prompt_reward_nonzero_variance_rate = _mean_or_zero(
        [float(item.get("reward_total_nonzero_variance", 0.0)) for item in prompt_summaries]
    )
    prompt_all_zero_reward_rate = _mean_or_zero(
        [float(item.get("all_zero_reward", 0.0)) for item in prompt_summaries]
    )
    reward_total_std_per_prompt = _mean_or_zero(
        [float(item.get("reward_total_std", 0.0)) for item in prompt_summaries]
    )
    unique_completions_per_prompt = _mean_or_zero(
        [float(item.get("unique_completion_count", 0.0)) for item in prompt_summaries]
    )

    readiness_score = 100.0 * (
        0.30 * prompt_all_rollouts_scoreable_rate +
        0.20 * rollout_payload_present_rate +
        0.15 * rollout_tags_correct_rate +
        0.20 * rollout_primary_f1_mean +
        0.10 * prompt_reward_nonzero_variance_rate +
        0.05 * (1.0 - rollout_prompt_copy_rate)
    )

    checkpoint_dir = Path(checkpoint_dir).resolve()
    sft_output_dir = Path(sft_output_dir).resolve()
    checkpoint_label = "output_dir" if checkpoint_dir == sft_output_dir else checkpoint_dir.name
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_label": checkpoint_label,
        "eval_output_jsonl": str(Path(eval_jsonl_path).resolve()),
        "prompt_count": len(rows),
        "rollout_count": len(rollouts),
        "rollout_tags_correct_rate": rollout_tags_correct_rate,
        "rollout_payload_present_rate": rollout_payload_present_rate,
        "rollout_exact_match_rate": rollout_exact_match_rate,
        "rollout_primary_f1_mean": rollout_primary_f1_mean,
        "rollout_prompt_copy_rate": rollout_prompt_copy_rate,
        "prompt_all_rollouts_scoreable_rate": prompt_all_rollouts_scoreable_rate,
        "prompt_any_rollout_scoreable_rate": prompt_any_rollout_scoreable_rate,
        "prompt_completion_collapsed_rate": prompt_completion_collapsed_rate,
        "prompt_reward_nonzero_variance_rate": prompt_reward_nonzero_variance_rate,
        "prompt_all_zero_reward_rate": prompt_all_zero_reward_rate,
        "reward_total_std_per_prompt": reward_total_std_per_prompt,
        "unique_completions_per_prompt": unique_completions_per_prompt,
        "readiness_score": readiness_score,
        "criterion_version": "grpo_readiness_v1",
    }


def run_grpo_readiness_checkpoint_sweep(
    *,
    python_exe: str,
    sft_output_dir: Path,
    eval_jsonl: Path,
    eval_n: int,
    num_rollouts: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    base_model_override: str | None,
    summary_out: Path,
    graph_filter: list[str] | None = None,
    dtype: str = "auto",
    device_map: str = "auto",
    parallel_workers: int = 1,
    env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    eval_script = (Path(__file__).resolve().parent / "eval_sft_on_jsonl.py").resolve()
    candidates = _list_grpo_readiness_candidates(sft_output_dir)
    if not candidates:
        raise RuntimeError(f"No checkpoint candidates found under {Path(sft_output_dir).resolve()}")
    summary_out = Path(summary_out).resolve()
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    def _write_partial_summary(rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        ordered = sorted(rows, key=lambda item: (-float(item["readiness_score"]), str(item["checkpoint_dir"])))
        tmp_path = summary_out.with_suffix(summary_out.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for row in ordered:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp_path.replace(summary_out)

        best = ordered[0]
        best_out = summary_out.with_name(summary_out.stem + "_best.json")
        best_tmp = best_out.with_suffix(best_out.suffix + ".tmp")
        best_tmp.write_text(json.dumps(best, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        best_tmp.replace(best_out)

    existing_results: dict[str, dict[str, Any]] = {}
    if summary_out.exists():
        try:
            with summary_out.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    checkpoint_dir = str(Path(row["checkpoint_dir"]).resolve())
                    existing_results[checkpoint_dir] = row
            if existing_results:
                print(
                    f"[grpo-readiness] resuming from existing summary with "
                    f"{len(existing_results)} completed checkpoint(s): {summary_out}"
                )
        except Exception as exc:
            print(f"[grpo-readiness] could not read existing summary {summary_out}: {exc}")

    def _build_eval_cmd(checkpoint_dir: Path) -> tuple[str, Path, list[str]]:
        checkpoint_label = "output_dir" if checkpoint_dir == Path(sft_output_dir).resolve() else checkpoint_dir.name
        output_jsonl = checkpoint_dir / "grpo_readiness_eval.jsonl"
        cmd = [
            str(python_exe),
            str(eval_script),
            "--model", str(checkpoint_dir),
            "--jsonl", str(Path(eval_jsonl).resolve()),
            "--n", str(int(eval_n)),
            "--num-rollouts", str(int(num_rollouts)),
            "--temperature", str(float(temperature)),
            "--top-p", str(float(top_p)),
            "--max-new-tokens", str(int(max_new_tokens)),
            "--dtype", str(dtype),
            "--device-map", str(device_map),
            "--output-jsonl", str(output_jsonl),
        ]
        if base_model_override:
            cmd.extend(["--base-model-override", str(base_model_override)])
        if graph_filter:
            cmd.append("--graph-filter")
            cmd.extend([str(x) for x in graph_filter])
        return checkpoint_label, output_jsonl, cmd

    for checkpoint_dir in candidates:
        checkpoint_key = str(Path(checkpoint_dir).resolve())
        if checkpoint_key in existing_results:
            continue
        _, output_jsonl, _ = _build_eval_cmd(checkpoint_dir)
        if output_jsonl.exists():
            try:
                summary = _summarize_grpo_readiness_eval(output_jsonl, checkpoint_dir, sft_output_dir)
                existing_results[checkpoint_key] = summary
                print(
                    f"[grpo-readiness] recovered existing eval for "
                    f"{summary['checkpoint_label']} from {output_jsonl}"
                )
            except Exception as exc:
                print(f"[grpo-readiness] ignoring unreadable existing eval {output_jsonl}: {exc}")

    if existing_results:
        _write_partial_summary(list(existing_results.values()))

    def _evaluate_one(checkpoint_dir: Path, *, gpu_id: str | None) -> dict[str, Any]:
        checkpoint_label, output_jsonl, cmd = _build_eval_cmd(checkpoint_dir)
        gpu_note = f" on GPU {gpu_id}" if gpu_id is not None else ""
        print(f"[grpo-readiness] evaluating {checkpoint_label}{gpu_note} on {Path(eval_jsonl).resolve()}")
        print("[run]", " ".join(cmd))
        child_env = os.environ.copy()
        if env:
            child_env.update(env)
        if gpu_id is not None:
            child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        subprocess.run(cmd, env=child_env, check=True)
        summary = _summarize_grpo_readiness_eval(output_jsonl, checkpoint_dir, sft_output_dir)
        print(
            f"[grpo-readiness] {checkpoint_label}: "
            f"score={summary['readiness_score']:.2f} "
            f"scoreable={summary['prompt_all_rollouts_scoreable_rate']:.2%} "
            f"edge_f1={summary['rollout_primary_f1_mean']:.4f} "
            f"reward_var={summary['prompt_reward_nonzero_variance_rate']:.2%}"
        )
        return summary

    pending_candidates = [
        checkpoint_dir
        for checkpoint_dir in candidates
        if str(Path(checkpoint_dir).resolve()) not in existing_results
    ]
    if not pending_candidates:
        results = list(existing_results.values())
        results.sort(key=lambda item: (-float(item["readiness_score"]), str(item["checkpoint_dir"])))
        print(f"[grpo-readiness] all checkpoint results already available in {summary_out}")
        return results

    requested_workers = max(1, int(parallel_workers))
    visible_gpus = _parse_cuda_visible_devices((env or {}).get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES"))
    if requested_workers > 1 and not visible_gpus:
        print("[grpo-readiness] no CUDA_VISIBLE_DEVICES list found; falling back to sequential readiness sweep")
        requested_workers = 1

    results: list[dict[str, Any]] = list(existing_results.values())
    results_lock = threading.Lock()
    worker_errors: list[Exception] = []

    def _record_result(summary: dict[str, Any]) -> None:
        with results_lock:
            checkpoint_key = str(Path(summary["checkpoint_dir"]).resolve())
            deduped = {
                str(Path(item["checkpoint_dir"]).resolve()): item
                for item in results
            }
            deduped[checkpoint_key] = summary
            results.clear()
            results.extend(deduped.values())
            _write_partial_summary(results)

    if requested_workers <= 1:
        for checkpoint_dir in pending_candidates:
            summary = _evaluate_one(checkpoint_dir, gpu_id=None)
            _record_result(summary)
    else:
        worker_gpus = visible_gpus[:requested_workers]
        if len(worker_gpus) < requested_workers:
            print(
                f"[grpo-readiness] requested {requested_workers} workers but only "
                f"{len(worker_gpus)} visible GPU(s); using {len(worker_gpus)} worker(s)"
            )
        requested_workers = max(1, len(worker_gpus))
        shard_count = min(requested_workers, len(pending_candidates))
        candidate_shards: list[list[Path]] = [[] for _ in range(shard_count)]
        for idx, checkpoint_dir in enumerate(pending_candidates):
            candidate_shards[idx % shard_count].append(checkpoint_dir)
        worker_specs = [
            (worker_gpus[idx], candidate_shards[idx])
            for idx in range(shard_count)
            if candidate_shards[idx]
        ]
        print(
            f"[grpo-readiness] parallel sweep across {len(worker_specs)} worker(s): "
            + ", ".join(f"GPU {gpu_id} -> {len(shard)} checkpoint(s)" for gpu_id, shard in worker_specs)
        )

        def _run_worker(gpu_id: str, shard: list[Path]) -> None:
            for checkpoint_dir in shard:
                summary = _evaluate_one(checkpoint_dir, gpu_id=gpu_id)
                _record_result(summary)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_specs)) as executor:
            futures = [
                executor.submit(_run_worker, gpu_id, shard)
                for gpu_id, shard in worker_specs
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    worker_errors.append(exc)

        if worker_errors:
            raise worker_errors[0]

    results.sort(key=lambda item: (-float(item["readiness_score"]), str(item["checkpoint_dir"])))
    _write_partial_summary(results)

    print(f"[grpo-readiness] wrote ranking -> {summary_out}")
    best = results[0]
    best_out = summary_out.with_name(summary_out.stem + "_best.json")
    print(f"[grpo-readiness] wrote best checkpoint summary -> {best_out}")
    print(
        f"[grpo-readiness] best checkpoint: {best['checkpoint_label']} "
        f"({best['checkpoint_dir']}) score={best['readiness_score']:.2f}"
    )
    return results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SFT warmup -> optional format check -> optional GRPO.")
    ap.add_argument(
        "--sft-jsonl",
        type=Path,
        required=True,
        help="Pre-built SFT JSONL (from cd_sft/staged_targets.py) with 'prompt' and 'answer' columns.",
    )
    ap.add_argument(
        "--sft-output-dir",
        type=Path,
        default=Path("experiments/checkpoints/sft_staged"),
    )
    ap.add_argument(
        "--grpo-output-dir",
        type=Path,
        default=Path("experiments/checkpoints/grpo_from_sft"),
    )
    ap.add_argument(
        "--base-model",
        default="Qwen/Qwen3-4B-Thinking-2507",
    )
    ap.add_argument(
        "--distributed-base-model",
        default=None,
        help=(
            "Non-quantized base model id to use when WORLD_SIZE>1. "
            "Auto-derived from adapter config if omitted."
        ),
    )
    ap.add_argument(
        "--grpo-script",
        type=Path,
        default=Path("experiments/grpo.py"),
    )
    ap.add_argument("--nproc-per-node", type=int, default=2)
    ap.add_argument("--python-exe", default=sys.executable)
    ap.add_argument(
        "--cuda-visible-devices",
        default="0",
        help="CUDA_VISIBLE_DEVICES for GPU stages (default: 0,1).",
    )

    ap.add_argument("--sft-epochs", type=float, default=2.0)
    ap.add_argument("--sft-lr", type=float, default=2e-5)
    ap.add_argument("--sft-batch-size", type=int, default=1)
    ap.add_argument("--sft-eval-batch-size", type=int, default=1)
    ap.add_argument("--sft-grad-accum", type=int, default=4)
    ap.add_argument("--sft-max-seq-length", type=int, default=16384)
    ap.add_argument("--sft-save-steps", type=int, default=100)
    ap.add_argument("--sft-logging-steps", type=int, default=10)
    ap.add_argument(
        "--sft-save-total-limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep. Default: keep all intermediate checkpoints.",
    )
    ap.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume SFT trainer state from a checkpoint path, or use 'latest' to pick the highest checkpoint-* under --sft-output-dir.",
    )
    ap.add_argument(
        "--train-log-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL metrics path for SFT logs. Default: <sft_output_dir>/train_metrics.jsonl",
    )
    ap.add_argument(
        "--eval-jsonl",
        type=Path,
        default=None,
        help="Optional held-out eval JSONL with the same prompt/answer schema as --sft-jsonl.",
    )
    ap.add_argument(
        "--eval-log-jsonl",
        type=Path,
        default=None,
        help="Optional eval-only JSONL metrics path. Default: <sft_output_dir>/eval_metrics.jsonl",
    )
    ap.add_argument(
        "--validation-split-ratio",
        type=float,
        default=0.0,
        help="If --eval-jsonl is not set, reserve this fraction of --sft-jsonl for eval.",
    )
    ap.add_argument(
        "--eval-every",
        "--eval-steps",
        dest="eval_every",
        type=int,
        default=None,
        help="Run eval every N optimizer steps. Defaults to --sft-save-steps when eval is enabled.",
    )
    ap.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop after this many evals without improvement on --metric-for-best-model.",
    )
    ap.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0,
        help="Minimum improvement required to reset early stopping patience.",
    )
    ap.add_argument(
        "--load-best-model-at-end",
        action="store_true",
        help="Reload the checkpoint with the best eval metric at the end of training.",
    )
    ap.add_argument(
        "--metric-for-best-model",
        type=str,
        default="eval_loss",
        help="Metric name to monitor when loading the best model or using early stopping.",
    )
    ap.add_argument(
        "--greater-is-better",
        action="store_true",
        help="Treat --metric-for-best-model as a metric to maximize instead of minimize.",
    )
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument(
        "--use-unsloth",
        action="store_true",
        help="Use Unsloth for single-GPU SFT (faster, 4-bit, all 7 projection modules).",
    )
    ap.add_argument(
        "--grpo-base-model",
        type=str,
        default=None,
        help=(
            "Full-precision base model id/path to write into adapter_config.json after Unsloth SFT. "
            "Required when the Unsloth base is a BNB-quantized model (e.g. *-bnb-4bit) so that "
            "GRPO can load the adapter without quantization on multi-GPU setups."
        ),
    )

    ap.add_argument(
        "--stage",
        choices=["all", "sft", "check", "grpo", "readiness"],
        default="sft",
        help="Which stage(s) to run (default: sft).",
    )

    # Format check options (--stage check or --stage all)
    ap.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="CSV with prompt/answer columns, required for --stage check.",
    )
    ap.add_argument("--prompt-text-col", default="prompt_text")
    ap.add_argument("--prompt-path-col", default="prompt_path")
    ap.add_argument("--answer-col", default="answer")
    ap.add_argument("--answer-path-col", default="answer_path")
    ap.add_argument("--skip-format-check", action="store_true")
    ap.add_argument("--format-check-rows", type=int, default=32)
    ap.add_argument("--format-check-max-new-tokens", type=int, default=1024)
    ap.add_argument(
        "--format-check-enable-thinking",
        dest="format_check_enable_thinking",
        action="store_true",
    )
    ap.add_argument(
        "--no-format-check-enable-thinking",
        dest="format_check_enable_thinking",
        action="store_false",
    )
    ap.set_defaults(format_check_enable_thinking=False)
    ap.add_argument("--format-check-out", type=Path, default=None)

    # Logging options
    ap.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name for logging. If set, run name is derived from --sft-output-dir basename.",
    )
    ap.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Custom W&B run name (defaults to --sft-output-dir basename if --wandb-project is set).",
    )

    ap.add_argument(
        "--grpo-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed through to GRPO script.",
    )
    ap.add_argument(
        "--merge-after-sft",
        action="store_true",
        help="After SFT finishes, merge the LoRA adapter into a full model checkpoint.",
    )
    ap.add_argument(
        "--no-merge-after-sft",
        dest="merge_after_sft",
        action="store_false",
        help="Do not merge the LoRA adapter into a full model checkpoint after SFT.",
    )
    ap.add_argument(
        "--merged-output-dir",
        type=Path,
        default=None,
        help=(
            "Output dir for the merged full model checkpoint. "
            "Defaults to <sft_output_dir>_merged when --merge-after-sft is set."
        ),
    )
    ap.set_defaults(merge_after_sft=True)
    ap.add_argument(
        "--run-grpo-readiness-eval",
        action="store_true",
        help="After SFT, run a small generation-based checkpoint sweep and rank checkpoints for GRPO readiness.",
    )
    ap.add_argument(
        "--grpo-readiness-eval-jsonl",
        type=Path,
        default=None,
        help="Held-out JSONL/CSV used for GRPO-readiness checkpoint ranking. Defaults to --eval-jsonl when set.",
    )
    ap.add_argument(
        "--grpo-readiness-summary-out",
        type=Path,
        default=None,
        help="Output JSONL path for checkpoint readiness summaries. Default: <sft_output_dir>/grpo_readiness_summary.jsonl",
    )
    ap.add_argument("--grpo-readiness-n", type=int, default=32)
    ap.add_argument("--grpo-readiness-num-rollouts", type=int, default=2)
    ap.add_argument("--grpo-readiness-temperature", type=float, default=0.7)
    ap.add_argument("--grpo-readiness-top-p", type=float, default=1.0)
    ap.add_argument("--grpo-readiness-max-new-tokens", type=int, default=1024)
    ap.add_argument(
        "--grpo-readiness-parallel-workers",
        type=int,
        default=1,
        help=(
            "Number of checkpoint-eval workers to run in parallel during readiness ranking. "
            "When >1, each worker is pinned to one visible GPU from CUDA_VISIBLE_DEVICES."
        ),
    )
    ap.add_argument(
        "--grpo-readiness-base-model-override",
        type=str,
        default=None,
        help="Optional local base-model path/id forwarded to eval_sft_on_jsonl.py during readiness ranking.",
    )
    ap.add_argument(
        "--grpo-readiness-graph-filter",
        nargs="*",
        default=None,
        help="Optional graph subset for readiness ranking, forwarded to eval_sft_on_jsonl.py.",
    )
    ap.add_argument(
        "--grpo-readiness-dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
    )
    ap.add_argument("--grpo-readiness-device-map", default="auto")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Derive wandb run name from sft-output-dir if not provided
    wandb_run_name = args.wandb_run_name
    if args.wandb_project and not wandb_run_name:
        wandb_run_name = Path(args.sft_output_dir).name

    # Make wandb project/name explicit for Trainer integrations.
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = str(args.wandb_project)
        if wandb_run_name:
            os.environ["WANDB_NAME"] = str(wandb_run_name)

    if args.stage == "readiness":
        readiness_eval_jsonl = (
            Path(args.grpo_readiness_eval_jsonl).resolve()
            if args.grpo_readiness_eval_jsonl is not None
            else (Path(args.eval_jsonl).resolve() if args.eval_jsonl is not None else None)
        )
        if readiness_eval_jsonl is None:
            raise ValueError(
                "--stage readiness requires --grpo-readiness-eval-jsonl "
                "(or --eval-jsonl so it can default to that)."
            )
        readiness_summary_out = (
            Path(args.grpo_readiness_summary_out).resolve()
            if args.grpo_readiness_summary_out is not None
            else Path(args.sft_output_dir).resolve() / "grpo_readiness_summary.jsonl"
        )
        readiness_base_model = (
            args.grpo_readiness_base_model_override
            or getattr(args, "grpo_base_model", None)
            or args.distributed_base_model
        )
        run_grpo_readiness_checkpoint_sweep(
            python_exe=args.python_exe,
            sft_output_dir=Path(args.sft_output_dir).resolve(),
            eval_jsonl=readiness_eval_jsonl,
            eval_n=args.grpo_readiness_n,
            num_rollouts=args.grpo_readiness_num_rollouts,
            temperature=args.grpo_readiness_temperature,
            top_p=args.grpo_readiness_top_p,
            max_new_tokens=args.grpo_readiness_max_new_tokens,
            base_model_override=readiness_base_model,
            summary_out=readiness_summary_out,
            graph_filter=args.grpo_readiness_graph_filter,
            dtype=args.grpo_readiness_dtype,
            device_map=args.grpo_readiness_device_map,
            parallel_workers=args.grpo_readiness_parallel_workers,
            env={"CUDA_VISIBLE_DEVICES": str(args.cuda_visible_devices)},
        )
        return

    if args.stage in ("all", "sft"):
        already_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        if not already_distributed:
            prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
            requested_cvd = str(args.cuda_visible_devices)
            if prev_cvd:
                if prev_cvd != requested_cvd:
                    print(
                        f"[sft] respecting existing CUDA_VISIBLE_DEVICES={prev_cvd!r}; "
                        f"ignoring script default {requested_cvd!r}"
                    )
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = requested_cvd
        try:
            run_sft(
                base_model=args.base_model,
                distributed_base_model=args.distributed_base_model,
                sft_jsonl=Path(args.sft_jsonl).resolve(),
                sft_output_dir=Path(args.sft_output_dir).resolve(),
                sft_epochs=args.sft_epochs,
                sft_lr=args.sft_lr,
                sft_batch_size=args.sft_batch_size,
                sft_eval_batch_size=args.sft_eval_batch_size,
                sft_grad_accum=args.sft_grad_accum,
                sft_max_seq_length=args.sft_max_seq_length,
                sft_save_steps=args.sft_save_steps,
                sft_logging_steps=args.sft_logging_steps,
                sft_save_total_limit=args.sft_save_total_limit,
                train_log_jsonl=args.train_log_jsonl,
                eval_jsonl=args.eval_jsonl,
                eval_log_jsonl=args.eval_log_jsonl,
                validation_split_ratio=args.validation_split_ratio,
                eval_every=args.eval_every,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
                load_best_model_at_end=args.load_best_model_at_end,
                metric_for_best_model=args.metric_for_best_model,
                greater_is_better=args.greater_is_better,
                use_unsloth=args.use_unsloth,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                grpo_base_model=getattr(args, "grpo_base_model", None),
                wandb_project=args.wandb_project,
                wandb_run_name=wandb_run_name,
                resume_from_checkpoint=args.resume_from_checkpoint,
            )
        finally:
            if not already_distributed:
                if prev_cvd is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
        print(f"[sft] saved -> {args.sft_output_dir}")
        if args.run_grpo_readiness_eval:
            readiness_eval_jsonl = (
                Path(args.grpo_readiness_eval_jsonl).resolve()
                if args.grpo_readiness_eval_jsonl is not None
                else (Path(args.eval_jsonl).resolve() if args.eval_jsonl is not None else None)
            )
            if readiness_eval_jsonl is None:
                raise ValueError(
                    "--run-grpo-readiness-eval requires --grpo-readiness-eval-jsonl "
                    "(or --eval-jsonl so it can default to that)."
                )
            readiness_summary_out = (
                Path(args.grpo_readiness_summary_out).resolve()
                if args.grpo_readiness_summary_out is not None
                else Path(args.sft_output_dir).resolve() / "grpo_readiness_summary.jsonl"
            )
            readiness_base_model = (
                args.grpo_readiness_base_model_override
                or getattr(args, "grpo_base_model", None)
                or args.distributed_base_model
            )
            run_grpo_readiness_checkpoint_sweep(
                python_exe=args.python_exe,
                sft_output_dir=Path(args.sft_output_dir).resolve(),
                eval_jsonl=readiness_eval_jsonl,
                eval_n=args.grpo_readiness_n,
                num_rollouts=args.grpo_readiness_num_rollouts,
                temperature=args.grpo_readiness_temperature,
                top_p=args.grpo_readiness_top_p,
                max_new_tokens=args.grpo_readiness_max_new_tokens,
                base_model_override=readiness_base_model,
                summary_out=readiness_summary_out,
                graph_filter=args.grpo_readiness_graph_filter,
                dtype=args.grpo_readiness_dtype,
                device_map=args.grpo_readiness_device_map,
                parallel_workers=args.grpo_readiness_parallel_workers,
                env={"CUDA_VISIBLE_DEVICES": str(args.cuda_visible_devices)},
            )
        if args.merge_after_sft:
            merged_output_dir = (
                Path(args.merged_output_dir).resolve()
                if args.merged_output_dir is not None
                else Path(str(args.sft_output_dir) + "_merged").resolve()
            )
            merge_sft_adapter(
                sft_model_dir=Path(args.sft_output_dir).resolve(),
                merged_output_dir=merged_output_dir,
            )
        if args.stage == "sft":
            return

    if args.stage in ("all", "check") and not args.skip_format_check:
        if args.train_csv is None:
            print("[check] --train-csv not provided; skipping format check.", file=sys.stderr)
        else:
            format_out = args.format_check_out or (args.sft_output_dir / "format_check.jsonl")
            run_format_check(
                sft_model_dir=args.sft_output_dir,
                train_csv=args.train_csv,
                prompt_text_col=args.prompt_text_col,
                prompt_path_col=args.prompt_path_col,
                answer_col=args.answer_col,
                answer_path_col=args.answer_path_col,
                max_rows=args.format_check_rows,
                max_new_tokens=args.format_check_max_new_tokens,
                output_path=format_out,
                sft_max_seq_length=args.sft_max_seq_length,
                format_check_enable_thinking=bool(args.format_check_enable_thinking),
            )
        if args.stage == "check":
            return

    if args.stage in ("all", "grpo"):
        if args.train_csv is None:
            raise ValueError("--train-csv is required for --stage grpo")
        extra = list(args.grpo_args or [])
        if extra and extra[0] == "--":
            extra = extra[1:]
        run_grpo(
            python_exe=args.python_exe,
            nproc_per_node=args.nproc_per_node,
            grpo_script=args.grpo_script,
            sft_output_dir=args.sft_output_dir,
            grpo_output_dir=args.grpo_output_dir,
            grpo_train_csv=args.train_csv,
            extra_args=extra,
            env={"CUDA_VISIBLE_DEVICES": str(args.cuda_visible_devices)},
        )
        print(f"[grpo] saved -> {args.grpo_output_dir}")


if __name__ == "__main__":
    main()
