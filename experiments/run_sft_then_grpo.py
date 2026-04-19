#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import inspect
import importlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from cd_training_format import (
        default_short_think_text,
        ensure_assistant_think_prefill,
        validate_sft_example,
    )
except ModuleNotFoundError:
    from experiments.cd_training_format import (
        default_short_think_text,
        ensure_assistant_think_prefill,
        validate_sft_example,
    )


def _set_csv_field_limit() -> None:
    # Large inlined prompts/answers can exceed Python csv's default field cap.
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
    # Prefer inline JSON first; large JSON strings can exceed filename limits
    # if interpreted as a filesystem path.
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        p = Path(s)
        if p.exists() and p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except OSError:
        # Not a usable path; keep the original JSON error semantics.
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


def _import_trl_sft() -> tuple[Any, Any]:
    """
    Import TRL SFT classes without letting TRL's vLLM compatibility shim probe NVML.

    Some TRL builds import `trl._compat` eagerly, which in turn patches vLLM by
    importing it at module import time. On machines with a broken/unavailable NVML
    stack, that crashes even though SFT does not use vLLM at all.
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
            # Clear partially imported TRL modules so the patched availability check
            # is used consistently on retry after an earlier failed import.
            stale = [name for name in sys.modules if name == "trl" or name.startswith("trl.")]
            for name in stale:
                sys.modules.pop(name, None)
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
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
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


def build_sft_jsonl(
    *,
    in_csv: Path,
    out_jsonl: Path,
    think_text: str,
    prompt_text_col: str,
    prompt_path_col: str,
    answer_col: str,
    answer_path_col: str,
) -> tuple[int, int]:
    _set_csv_field_limit()
    rows = list(csv.DictReader(in_csv.open("r", encoding="utf-8", newline="")))
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    skipped = 0
    with out_jsonl.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            try:
                prompt = _read_text_from_row(row, prompt_text_col, prompt_path_col).strip()
                if not prompt:
                    raise ValueError("empty prompt")
                prompt = ensure_assistant_think_prefill(prompt)

                answer_raw = row.get(answer_col)
                if not answer_raw:
                    answer_raw = row.get(answer_path_col)
                answer_obj = _load_answer_obj(answer_raw)
                answer_payload = _extract_answer_payload(answer_obj)
                matrix = _extract_adjacency_matrix(answer_payload)

                target = {"adjacency_matrix": matrix}
                completion = (
                    f"{str(think_text or default_short_think_text('causal_discovery')).strip()}</think>"
                    f"<answer>{json.dumps(target, ensure_ascii=False)}</answer>"
                )
                issues = validate_sft_example(prompt, completion)
                if issues:
                    raise ValueError("; ".join(issues))
                rec = {
                    "prompt": prompt,
                    "answer": completion,
                    "text": prompt + "\n\n" + completion,  # backward-compatible fallback
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wrote += 1
            except Exception as exc:
                skipped += 1
                print(f"[warn] skip row {i}: {exc}", file=sys.stderr)
    return wrote, skipped


def _looks_quantized_model_id(model_name: str) -> bool:
    name = str(model_name or "").lower()
    quant_markers = ("4bit", "8bit", "bnb", "gptq", "awq", "gguf", "nf4")
    return any(tok in name for tok in quant_markers)


def _derive_non_quantized_model_id(model_name: str) -> str:
    # Heuristic: strip common quantization suffixes in Hub ids.
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


def _validate_sft_jsonl(path: Path, *, allow_legacy_text: bool = False, sample_limit: int = 8) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"SFT JSONL not found: {path}")
    if not path.is_file():
        raise ValueError(f"SFT JSONL is not a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(
            f"SFT JSONL is empty: {path}. "
            "Regenerate it before training; for staged CD data use generate_staged_sft_data.py."
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
        raise ValueError(f"SFT JSONL has no non-empty records: {path}")
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
            f"SFT JSONL must contain 'prompt' and 'answer' columns; observed keys: {sorted(sample_keys)}"
        )

    return {
        "rows": total,
        "prompt_answer_rows": prompt_answer_rows,
        "text_rows": text_rows,
        "keys": sorted(sample_keys),
    }


def run_sft(
    *,
    base_model: str,
    distributed_base_model: str | None,
    sft_jsonl: Path,
    sft_output_dir: Path,
    sft_epochs: float,
    sft_lr: float,
    sft_batch_size: int,
    sft_grad_accum: int,
    sft_max_seq_length: int,
    sft_save_steps: int,
    sft_logging_steps: int,
    sft_load_in_4bit: bool,
    sft_use_unsloth_gc: bool,
) -> None:
    sft_summary = _validate_sft_jsonl(sft_jsonl, allow_legacy_text=True)
    print(
        f"[sft] validated dataset: rows={sft_summary['rows']} "
        f"prompt_answer_rows={sft_summary['prompt_answer_rows']} "
        f"text_rows={sft_summary['text_rows']} path={Path(sft_jsonl).resolve()}"
    )

    import torch
    from datasets import load_dataset
    from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    SFTConfig, SFTTrainer = _import_trl_sft()

    def _filter_supported(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            sig = inspect.signature(callable_obj)
            allowed = set(sig.parameters.keys())
            return {k: v for k, v in kwargs.items() if k in allowed}
        except Exception:
            return dict(kwargs)

    is_adapter = (Path(base_model) / "adapter_config.json").exists()
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    distributed_world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # DDP (torchrun, WORLD_SIZE>1) cannot handle 4-bit quantized models because
    # they cannot be moved across devices. Force bfloat16 in distributed mode so
    # that each rank loads a full-precision copy on its own GPU.
    if distributed_world_size > 1:
        load_dtype = torch.bfloat16
        print(f"[sft] distributed mode (WORLD_SIZE={distributed_world_size}): forcing dtype=bfloat16 (4-bit incompatible with DDP)")
    else:
        load_dtype = "auto"

    model_load_kwargs: dict[str, Any] = {
        "torch_dtype": load_dtype,
        "attn_implementation": "flash_attention_2",
    }
    # Match grpo.py: avoid device_map='auto' when Accelerate sees distributed mode.
    if distributed_world_size == 1:
        model_load_kwargs["device_map"] = "auto"
    if bool(sft_load_in_4bit):
        print(
            "[sft] note: ignoring sft_load_in_4bit and loading with torch_dtype setting "
            "to match grpo.py model setup and avoid device_map/distributed issues."
        )
    if is_adapter:
        # In distributed mode: load the base model in bfloat16 directly, then
        # apply the LoRA adapter weights. This avoids inheriting the 4-bit
        # quantization config from the checkpoint's base_model_name_or_path.
        if distributed_world_size > 1:
            adapter_cfg_path = Path(base_model) / "adapter_config.json"
            adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
            base_model_name = adapter_cfg.get("base_model_name_or_path", "")
            resolved_base_model_name = str(base_model_name or "")
            if _looks_quantized_model_id(resolved_base_model_name):
                if distributed_base_model:
                    resolved_base_model_name = str(distributed_base_model)
                else:
                    resolved_base_model_name = _derive_non_quantized_model_id(resolved_base_model_name)
                print(
                    "[sft] distributed: adapter points to a quantized base; "
                    f"switching to non-quantized base {resolved_base_model_name!r}"
                )
            else:
                print(
                    f"[sft] distributed: loading base model {resolved_base_model_name!r} in bfloat16, "
                    "then applying adapter"
                )
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(resolved_base_model_name, **model_load_kwargs)
            model = PeftModel.from_pretrained(base, base_model, is_trainable=True)
        else:
            adapter_load_kwargs = dict(model_load_kwargs)
            try:
                adapter_params = inspect.signature(AutoPeftModelForCausalLM.from_pretrained).parameters
                if "is_trainable" in adapter_params:
                    adapter_load_kwargs["is_trainable"] = True
            except Exception:
                pass
            model = AutoPeftModelForCausalLM.from_pretrained(base_model, **adapter_load_kwargs)
        # Older PEFT paths may still leave the adapter frozen; make the adapter
        # trainable explicitly so stage-to-stage SFT continuation works.
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
            base_model,
            **model_load_kwargs,
        )
    # Some tokenizer variants expose placeholder EOS tokens in config.
    # Resolve to an actual in-vocab token before building SFTConfig.
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    eos_candidates = [
        getattr(tokenizer, "eos_token", None),
        "<|im_end|>",
        "<|endoftext|>",
    ]
    resolved_eos = None
    for tok in eos_candidates:
        if tok and isinstance(vocab, dict) and tok in vocab:
            resolved_eos = tok
            break
    if resolved_eos is not None:
        tokenizer.eos_token = resolved_eos
        try:
            tokenizer.eos_token_id = int(vocab[resolved_eos])
        except Exception:
            pass
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token

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
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
    else:
        try:
            input_embeddings = model.get_input_embeddings()
            if input_embeddings is not None:
                def _require_grad_hook(module, inputs, output):
                    if hasattr(output, "requires_grad_"):
                        output.requires_grad_(True)
                    return output

                input_embeddings.register_forward_hook(_require_grad_hook)
        except Exception:
            pass
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    trainable_params = 0
    total_params = 0
    for param in model.parameters():
        count = int(param.numel())
        total_params += count
        if bool(getattr(param, "requires_grad", False)):
            trainable_params += count
    print(
        f"[sft] trainable_params={trainable_params} total_params={total_params} "
        f"trainable_pct={(100.0 * trainable_params / max(total_params, 1)):.4f}"
    )

    ds = load_dataset("json", data_files=str(Path(sft_jsonl).resolve()), split="train")
    dataset_uses_prompt_completion = "prompt" in ds.column_names and "answer" in ds.column_names
    if dataset_uses_prompt_completion:
        keep = {"prompt", "answer"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        ds = ds.rename_column("answer", "completion")
    else:
        # Backward compatibility for older JSONL files that only contain "text".
        def _to_sft_text(ex: dict[str, Any]) -> dict[str, str]:
            text0 = ex.get("text")
            return {"text": text0 if isinstance(text0, str) else ""}

        ds = ds.map(
            _to_sft_text,
            desc="Formatting legacy SFT data",
            remove_columns=ds.column_names,
        )
    cfg_kwargs = {
        "output_dir": str(sft_output_dir),
        "num_train_epochs": float(sft_epochs),
        "learning_rate": float(sft_lr),
        "per_device_train_batch_size": int(sft_batch_size),
        "gradient_accumulation_steps": int(sft_grad_accum),
        "disable_tqdm": False,
        "logging_steps": int(sft_logging_steps),
        "save_steps": int(sft_save_steps),
        "bf16": torch_cuda_available(),
        "max_seq_length": int(sft_max_seq_length),
        "report_to": "none",
    }
    # Older/newer TRL versions may default eos_token to a placeholder like "<EOS_TOKEN>".
    # Force EOS to the tokenizer's actual EOS symbol when available.
    eos_id_check = tokenizer.convert_tokens_to_ids(tokenizer.eos_token) if getattr(tokenizer, "eos_token", None) else None
    if eos_id_check is None:
        raise ValueError(
            f"SFT eos_token is not in tokenizer vocab: eos_token={getattr(tokenizer, 'eos_token', None)!r}. "
            "Set a valid eos token before trainer init."
        )
    print(f"[sft] eos_token={tokenizer.eos_token!r} eos_token_id={eos_id_check}")

    if dataset_uses_prompt_completion:
        eos_token_id = int(tokenizer.eos_token_id)

        def _tokenize_prompt_completion(example: dict[str, Any]) -> dict[str, Any]:
            prompt = str(example.get("prompt") or "")
            completion = str(example.get("completion") or "")
            issues = validate_sft_example(prompt, completion)
            if issues:
                raise ValueError("; ".join(issues))

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            if not completion_ids:
                raise ValueError("completion tokenization produced zero tokens")
            if not completion_ids or completion_ids[-1] != eos_token_id:
                completion_ids = [*completion_ids, eos_token_id]

            max_len = int(sft_max_seq_length)
            if len(completion_ids) >= max_len:
                completion_ids = completion_ids[:max_len]
                prompt_ids = []
            else:
                keep_prompt = max_len - len(completion_ids)
                if len(prompt_ids) > keep_prompt:
                    prompt_ids = prompt_ids[-keep_prompt:]

            input_ids = [*prompt_ids, *completion_ids]
            labels = ([-100] * len(prompt_ids)) + list(completion_ids)
            if len(input_ids) != len(labels):
                raise ValueError("input_ids/labels length mismatch")
            supervised_tokens = sum(1 for x in labels if x != -100)
            if supervised_tokens <= 0:
                raise ValueError("no supervised completion tokens")
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                "__prompt_tokens": len(prompt_ids),
                "__completion_tokens": len(completion_ids),
                "__supervised_tokens": supervised_tokens,
            }

        tokenized_ds = ds.map(
            _tokenize_prompt_completion,
            remove_columns=ds.column_names,
            desc="Tokenizing prompt/completion SFT data",
        )
        supervised_total = int(sum(int(x) for x in tokenized_ds["__supervised_tokens"]))
        if supervised_total <= 0:
            raise ValueError("prompt/completion SFT dataset has zero supervised tokens")
        prompt_tok_mean = sum(int(x) for x in tokenized_ds["__prompt_tokens"]) / max(len(tokenized_ds), 1)
        completion_tok_mean = sum(int(x) for x in tokenized_ds["__completion_tokens"]) / max(len(tokenized_ds), 1)
        print(
            "[sft] prompt/completion dataset "
            f"rows={len(tokenized_ds)} "
            f"prompt_tokens_mean={prompt_tok_mean:.1f} "
            f"completion_tokens_mean={completion_tok_mean:.1f} "
            f"supervised_tokens_total={supervised_total}"
        )

        class _PromptCompletionCollator:
            def __init__(self, pad_token_id: int):
                self.pad_token_id = int(pad_token_id)

            def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
                max_len = max(len(f["input_ids"]) for f in features)
                batch_input_ids = []
                batch_attention = []
                batch_labels = []
                for feat in features:
                    pad = max_len - len(feat["input_ids"])
                    batch_input_ids.append(list(feat["input_ids"]) + ([self.pad_token_id] * pad))
                    batch_attention.append(list(feat["attention_mask"]) + ([0] * pad))
                    batch_labels.append(list(feat["labels"]) + ([-100] * pad))
                return {
                    "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
                    "labels": torch.tensor(batch_labels, dtype=torch.long),
                }

        train_args = TrainingArguments(
            output_dir=str(sft_output_dir),
            num_train_epochs=float(sft_epochs),
            learning_rate=float(sft_lr),
            per_device_train_batch_size=int(sft_batch_size),
            gradient_accumulation_steps=int(sft_grad_accum),
            logging_steps=int(sft_logging_steps),
            save_steps=int(sft_save_steps),
            bf16=torch_cuda_available(),
            report_to="none",
            remove_unused_columns=False,
            save_total_limit=1,
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=tokenized_ds,
            data_collator=_PromptCompletionCollator(tokenizer.pad_token_id),
        )
        trainer.train()
        trainer.save_model(str(sft_output_dir))
        tokenizer.save_pretrained(str(sft_output_dir))
        return

    cfg_kwargs["dataset_text_field"] = "text"
    if getattr(tokenizer, "eos_token", None):
        cfg_kwargs["eos_token"] = tokenizer.eos_token
    cfg = SFTConfig(**_filter_supported(SFTConfig.__init__, cfg_kwargs))
    if getattr(tokenizer, "eos_token", None):
        cfg.eos_token = tokenizer.eos_token

    trainer_kwargs = {
        "model": model,
        "train_dataset": ds,
        "args": cfg,
    }
    trainer_init_sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in trainer_init_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "max_seq_length" in trainer_init_sig.parameters:
        trainer_kwargs["max_seq_length"] = int(sft_max_seq_length)
    if "dataset_text_field" in trainer_init_sig.parameters:
        trainer_kwargs["dataset_text_field"] = "text"

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(sft_output_dir))
    tokenizer.save_pretrained(str(sft_output_dir))


def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


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
        "--nproc_per_node",
        str(int(nproc_per_node)),
        str(grpo_script),
        "--task",
        "causal_discovery",
        "--model_id",
        str(sft_output_dir),
        "--cd-train-csv",
        str(grpo_train_csv),
        "--output_dir",
        str(grpo_output_dir),
    ]
    cmd.extend(extra_args)
    print("[run]", " ".join(cmd))
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    subprocess.run(cmd, env=child_env, check=True)


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
    sft_load_in_4bit: bool,
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
        rows = rows[: max_rows]
    if not rows:
        raise RuntimeError(f"No rows found in {train_csv}")

    is_adapter = (Path(sft_model_dir) / "adapter_config.json").exists()
    check_load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else "auto",
        "attn_implementation": "flash_attention_2",
        "device_map": "auto",
    }
    if is_adapter:
        model = AutoPeftModelForCausalLM.from_pretrained(
            str(sft_model_dir), **check_load_kwargs
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(str(sft_model_dir), use_fast=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(sft_model_dir), **check_load_kwargs
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(str(sft_model_dir), use_fast=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    fmt_ok = 0
    parse_ok = 0
    with output_path.open("w", encoding="utf-8") as fout:
        class _StopOnTokenSuffix(StoppingCriteria):
            def __init__(self, suffixes: list[list[int]]):
                super().__init__()
                self.suffixes = [s for s in suffixes if s]

            def __call__(self, input_ids, scores, **kwargs):  # type: ignore[override]
                if input_ids is None or input_ids.numel() == 0:
                    return False
                # run_format_check generates one sample at a time
                seq = input_ids[0].tolist()
                for suf in self.suffixes:
                    if len(seq) >= len(suf) and seq[-len(suf):] == suf:
                        return True
                return False

        stop_phrases = ["</answer>", "\n</answer>", "\n\n</answer>", " </answer>"]
        stop_token_suffixes: list[list[int]] = []
        for phrase in stop_phrases:
            ids = tokenizer.encode(phrase, add_special_tokens=False)
            if ids:
                stop_token_suffixes.append(ids)
        stopping_criteria = (
            StoppingCriteriaList([_StopOnTokenSuffix(stop_token_suffixes)])
            if stop_token_suffixes
            else None
        )

        progress = tqdm(
            rows,
            total=len(rows),
            desc="check",
            unit="row",
            dynamic_ncols=True,
        )
        for i, row in enumerate(progress):
            prompt = _read_text_from_row(row, prompt_text_col, prompt_path_col).strip()
            prompt_for_model = _build_format_check_prompt(
                prompt,
                tokenizer=tokenizer,
                enable_thinking=bool(format_check_enable_thinking),
            )
            answer_raw = row.get(answer_col)
            if not answer_raw:
                answer_raw = row.get(answer_path_col)
            answer_obj = _load_answer_obj(answer_raw)
            answer_payload = _extract_answer_payload(answer_obj)
            try:
                target_matrix = _extract_adjacency_matrix(answer_payload)
                expected_n = len(target_matrix)
            except Exception:
                expected_n = None

            inputs = tokenizer(prompt_for_model, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": int(max_new_tokens),
                    "do_sample": False,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                if stopping_criteria is not None:
                    gen_kwargs["stopping_criteria"] = stopping_criteria
                out = model.generate(**inputs, **gen_kwargs)

            prompt_len = int(inputs["input_ids"].shape[-1])
            gen_ids = out[0][prompt_len:]
            resp = tokenizer.decode(gen_ids, skip_special_tokens=True)
            generated_tokens = int(gen_ids.shape[-1])
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            eos_seen = False
            if eos_token_id is not None:
                try:
                    eos_seen = any(int(tok) == int(eos_token_id) for tok in gen_ids.tolist())
                except Exception:
                    eos_seen = False
            answer_closed = bool((resp or "").rstrip().endswith("</answer>"))

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
                "answer_closed": int(answer_closed),
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pipeline: build SFT data -> SFT warmup -> GRPO.")
    ap.add_argument(
        "--train-csv",
        type=Path,
        required=True,
        help="CSV with prompt/answer columns (e.g., prompts_cd_mix_p120_thinktags_summary_joint.csv).",
    )
    ap.add_argument(
        "--sft-jsonl",
        type=Path,
        default=Path("experiments/prompts/grpo_cd_mix_train/sft_train.jsonl"),
    )
    ap.add_argument(
        "--sft-output-dir",
        type=Path,
        default=Path("experiments/checkpoints/sft_cd_format_4b"),
    )
    ap.add_argument(
        "--grpo-output-dir",
        type=Path,
        default=Path("experiments/checkpoints/grpo_from_sft_4b"),
    )
    ap.add_argument(
        "--base-model",
        default="unsloth/qwen3-4b-thinking-2507-unsloth-bnb-4bit",
    )
    ap.add_argument(
        "--distributed-base-model",
        default=None,
        help=(
            "Optional non-quantized base model id to force when WORLD_SIZE>1. "
            "If omitted, script auto-derives one from adapter/base model id."
        ),
    )
    ap.add_argument(
        "--grpo-script",
        type=Path,
        default=Path("experiments/grpo_unsloth.py"),
    )
    ap.add_argument("--nproc-per-node", type=int, default=4)
    ap.add_argument("--python-exe", default=sys.executable)
    ap.add_argument(
        "--cuda-visible-devices",
        default="0",
        help="CUDA_VISIBLE_DEVICES for GPU stages (default: 0).",
    )

    ap.add_argument("--prompt-text-col", default="prompt_text")
    ap.add_argument("--prompt-path-col", default="prompt_path")
    ap.add_argument("--answer-col", default="answer")
    ap.add_argument("--answer-path-col", default="answer_path")
    ap.add_argument(
        "--think-text",
        default="I will output only the required JSON adjacency matrix.",
    )

    ap.add_argument("--sft-epochs", type=float, default=1.0)
    ap.add_argument("--sft-lr", type=float, default=2e-5)
    ap.add_argument("--sft-batch-size", type=int, default=1)
    ap.add_argument("--sft-grad-accum", type=int, default=4)
    ap.add_argument("--sft-max-seq-length", type=int, default=262144)
    ap.add_argument("--sft-save-steps", type=int, default=50)
    ap.add_argument("--sft-logging-steps", type=int, default=5)
    ap.add_argument("--sft-load-in-4bit", action="store_true")
    ap.add_argument("--no-sft-unsloth-gc", dest="sft_use_unsloth_gc", action="store_false")
    ap.set_defaults(sft_use_unsloth_gc=True)

    ap.add_argument(
        "--stage",
        choices=["all", "build", "sft", "check", "grpo"],
        default="all",
        help="Run all stages or a single stage.",
    )
    ap.add_argument("--skip-format-check", action="store_true", help="Skip post-SFT format check.")
    ap.add_argument("--format-check-rows", type=int, default=32, help="Rows to probe after SFT (0=all).")
    ap.add_argument("--format-check-max-new-tokens", type=int, default=1024)
    ap.add_argument(
        "--format-check-enable-thinking",
        dest="format_check_enable_thinking",
        action="store_true",
        help="For post-SFT format checks, render prompts via chat template with enable_thinking=True when supported.",
    )
    ap.add_argument(
        "--no-format-check-enable-thinking",
        dest="format_check_enable_thinking",
        action="store_false",
    )
    ap.set_defaults(format_check_enable_thinking=False)
    ap.add_argument(
        "--format-check-out",
        type=Path,
        default=None,
        help="JSONL output path for post-SFT probe. Default: <sft_output_dir>/format_check.jsonl",
    )
    ap.add_argument(
        "--grpo-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed through to GRPO script (prefix with --grpo-args).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.stage in ("all", "build"):
        wrote, skipped = build_sft_jsonl(
            in_csv=args.train_csv,
            out_jsonl=args.sft_jsonl,
            think_text=args.think_text,
            prompt_text_col=args.prompt_text_col,
            prompt_path_col=args.prompt_path_col,
            answer_col=args.answer_col,
            answer_path_col=args.answer_path_col,
        )
        print(f"[build] wrote={wrote} skipped={skipped} -> {args.sft_jsonl}")
        if args.stage == "build":
            return

    if args.stage in ("all", "sft"):
        # Under torchrun, WORLD_SIZE > 1 and each rank already has LOCAL_RANK set.
        # Do NOT override CUDA_VISIBLE_DEVICES in that case — torchrun manages GPU
        # assignment and overriding to a single device causes both ranks to collide
        # on the same GPU (NCCL duplicate GPU error).
        already_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        if already_distributed:
            prev_cvd = None
        else:
            prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        try:
            run_sft(
                base_model=args.base_model,
                distributed_base_model=args.distributed_base_model,
                sft_jsonl=Path(args.sft_jsonl).resolve(),
                sft_output_dir=Path(args.sft_output_dir).resolve(),
                sft_epochs=args.sft_epochs,
                sft_lr=args.sft_lr,
                sft_batch_size=args.sft_batch_size,
                sft_grad_accum=args.sft_grad_accum,
                sft_max_seq_length=args.sft_max_seq_length,
                sft_save_steps=args.sft_save_steps,
                sft_logging_steps=args.sft_logging_steps,
                sft_load_in_4bit=args.sft_load_in_4bit,
                sft_use_unsloth_gc=args.sft_use_unsloth_gc,
            )
        finally:
            if not already_distributed:
                if prev_cvd is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
        print(f"[sft] saved -> {args.sft_output_dir}")
        if args.stage == "sft":
            return

    if args.stage in ("all", "check") and not args.skip_format_check:
        format_out = (
            args.format_check_out
            if args.format_check_out is not None
            else (args.sft_output_dir / "format_check.jsonl")
        )
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
            sft_load_in_4bit=args.sft_load_in_4bit,
            format_check_enable_thinking=bool(args.format_check_enable_thinking),
        )
        if args.stage == "check":
            return

    if args.stage in ("all", "grpo"):
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
