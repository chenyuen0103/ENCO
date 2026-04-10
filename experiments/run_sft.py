#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from cd_training_format import validate_sft_example
except ModuleNotFoundError:
    from experiments.cd_training_format import validate_sft_example


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


def _run_sft_unsloth(
    *,
    base_model: str,
    sft_jsonl: Path,
    sft_output_dir: Path,
    sft_epochs: float,
    sft_lr: float,
    sft_batch_size: int,
    sft_grad_accum: int,
    sft_max_seq_length: int,
    sft_save_steps: int,
    sft_logging_steps: int,
    lora_r: int,
    lora_alpha: int,
    grpo_base_model: str | None = None,
) -> None:
    """Single-GPU SFT via Unsloth — ~2× faster, supports 4-bit to reduce VRAM."""
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    from datasets import load_dataset
    import torch

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
    ds = load_dataset("json", data_files=str(Path(sft_jsonl).resolve()), split="train")
    if "prompt" not in ds.column_names or "answer" not in ds.column_names:
        raise ValueError(f"SFT JSONL must have 'prompt' and 'answer' columns; got {ds.column_names}")
    ds = ds.remove_columns([c for c in ds.column_names if c not in {"prompt", "answer"}])
    ds = ds.rename_column("answer", "completion")

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

    tokenized_ds = ds.map(_tokenize, remove_columns=ds.column_names, desc="Tokenizing")
    supervised_total = sum(
        sum(1 for x in row["labels"] if x != -100)
        for row in tokenized_ds
    )
    print(f"[sft/unsloth] rows={len(tokenized_ds)} supervised_tokens_total={supervised_total}")

    from trl import SFTTrainer, SFTConfig
    cfg = SFTConfig(
        output_dir=str(sft_output_dir),
        num_train_epochs=float(sft_epochs),
        learning_rate=float(sft_lr),
        per_device_train_batch_size=int(sft_batch_size),
        gradient_accumulation_steps=int(sft_grad_accum),
        logging_steps=int(sft_logging_steps),
        save_steps=int(sft_save_steps),
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        report_to="none",
        remove_unused_columns=False,
        save_total_limit=1,
        max_seq_length=max_len,
        dataset_text_field=None,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_ds,
        args=cfg,
    )
    trainer.train()
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
    sft_grad_accum: int,
    sft_max_seq_length: int,
    sft_save_steps: int,
    sft_logging_steps: int,
    use_unsloth: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 16,
    grpo_base_model: str | None = None,
) -> None:
    if use_unsloth:
        _run_sft_unsloth(
            base_model=base_model,
            sft_jsonl=sft_jsonl,
            sft_output_dir=sft_output_dir,
            sft_epochs=sft_epochs,
            sft_lr=sft_lr,
            sft_batch_size=sft_batch_size,
            sft_grad_accum=sft_grad_accum,
            sft_max_seq_length=sft_max_seq_length,
            sft_save_steps=sft_save_steps,
            sft_logging_steps=sft_logging_steps,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            grpo_base_model=grpo_base_model,
        )
        return

    import torch
    from datasets import load_dataset
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
    if distributed_world_size == 1:
        model_load_kwargs["device_map"] = "auto"

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

    ds = load_dataset("json", data_files=str(Path(sft_jsonl).resolve()), split="train")
    if "prompt" not in ds.column_names or "answer" not in ds.column_names:
        raise ValueError(
            f"SFT JSONL must have 'prompt' and 'answer' columns; got {ds.column_names}. "
            "Use generate_staged_sft_data.py to produce a compatible JSONL."
        )
    ds = ds.remove_columns([c for c in ds.column_names if c not in {"prompt", "answer"}])
    ds = ds.rename_column("answer", "completion")

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

    tokenized_ds = ds.map(_tokenize, remove_columns=ds.column_names, desc="Tokenizing")
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
        # Non-reentrant gradient checkpointing avoids the DDP + LoRA
        # "marked ready twice" error caused by reentrant recomputation
        # firing DDP all-reduce hooks multiple times per parameter.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_ds,
        data_collator=_Collator(tokenizer.pad_token_id),
    )
    trainer.train()
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SFT warmup -> optional format check -> optional GRPO.")
    ap.add_argument(
        "--sft-jsonl",
        type=Path,
        required=True,
        help="Pre-built SFT JSONL (from generate_staged_sft_data.py) with 'prompt' and 'answer' columns.",
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
        default="0,1",
        help="CUDA_VISIBLE_DEVICES for GPU stages (default: 0,1).",
    )

    ap.add_argument("--sft-epochs", type=float, default=2.0)
    ap.add_argument("--sft-lr", type=float, default=2e-5)
    ap.add_argument("--sft-batch-size", type=int, default=1)
    ap.add_argument("--sft-grad-accum", type=int, default=4)
    ap.add_argument("--sft-max-seq-length", type=int, default=3000)
    ap.add_argument("--sft-save-steps", type=int, default=100)
    ap.add_argument("--sft-logging-steps", type=int, default=10)
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
        choices=["all", "sft", "check", "grpo"],
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

    ap.add_argument(
        "--grpo-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed through to GRPO script.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.stage in ("all", "sft"):
        already_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        if not already_distributed:
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
                use_unsloth=args.use_unsloth,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                grpo_base_model=getattr(args, "grpo_base_model", None),
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
