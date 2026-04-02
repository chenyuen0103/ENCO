#!/usr/bin/env python3
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

                answer_raw = row.get(answer_col)
                if not answer_raw:
                    answer_raw = row.get(answer_path_col)
                answer_obj = _load_answer_obj(answer_raw)
                answer_payload = _extract_answer_payload(answer_obj)
                matrix = _extract_adjacency_matrix(answer_payload)

                target = {"adjacency_matrix": matrix}
                completion = (
                    f"<think>{think_text}</think>"
                    f"<answer>{json.dumps(target, ensure_ascii=False)}</answer>"
                )
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


def run_sft(
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
    sft_load_in_4bit: bool,
    sft_use_unsloth_gc: bool,
) -> None:
    from datasets import load_dataset
    from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    model_load_kwargs: dict[str, Any] = {
        "dtype": "auto",
    }
    # Match grpo.py: avoid device_map='auto' when Accelerate sees distributed mode.
    if distributed_world_size == 1:
        model_load_kwargs["device_map"] = "auto"
    if bool(sft_load_in_4bit):
        print(
            "[sft] note: ignoring sft_load_in_4bit and loading with dtype='auto' "
            "to match grpo.py model setup and avoid device_map/distributed issues."
        )
    if is_adapter:
        model = AutoPeftModelForCausalLM.from_pretrained(
            base_model,
            **model_load_kwargs,
        )
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
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass

    ds = load_dataset("json", data_files=str(sft_jsonl), split="train")
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
    if dataset_uses_prompt_completion:
        cfg_kwargs["completion_only_loss"] = True
    else:
        cfg_kwargs["dataset_text_field"] = "text"
    # Older/newer TRL versions may default eos_token to a placeholder like "<EOS_TOKEN>".
    # Force EOS to the tokenizer's actual EOS symbol when available.
    if getattr(tokenizer, "eos_token", None):
        cfg_kwargs["eos_token"] = tokenizer.eos_token
    cfg = SFTConfig(**_filter_supported(SFTConfig.__init__, cfg_kwargs))
    # Some TRL builds may still carry a placeholder EOS token from defaults.
    # Force args.eos_token to a tokenizer token that resolves in vocab.
    if getattr(tokenizer, "eos_token", None):
        cfg.eos_token = tokenizer.eos_token
    eos_id_check = tokenizer.convert_tokens_to_ids(cfg.eos_token) if getattr(cfg, "eos_token", None) else None
    if eos_id_check is None:
        raise ValueError(
            f"SFT eos_token is not in tokenizer vocab: eos_token={cfg.eos_token!r}. "
            "Set a valid eos token before SFTTrainer init."
        )
    print(f"[sft] eos_token={cfg.eos_token!r} eos_token_id={eos_id_check}")

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


# For format-check inference, allow free-form reasoning before the final answer
# and require only that the completion ends with <answer>...</answer>.
FORMAT_RE = re.compile(r"(?s)^.*<answer>.*?</answer>\s*$")


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
    from unsloth import FastLanguageModel
    from transformers import StoppingCriteria, StoppingCriteriaList
    from verifier_cd import extract_adjacency_matrix

    # Transformers 5.2 + Unsloth can emit a malformed warning log call from
    # modeling_attn_mask_utils; suppress that module's warning-level logs here.
    logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

    _set_csv_field_limit()
    rows = list(csv.DictReader(train_csv.open("r", encoding="utf-8", newline="")))
    if max_rows > 0:
        rows = rows[: max_rows]
    if not rows:
        raise RuntimeError(f"No rows found in {train_csv}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(sft_model_dir),
        max_seq_length=int(sft_max_seq_length),
        load_in_4bit=bool(sft_load_in_4bit),
    )
    FastLanguageModel.for_inference(model)

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

        for i, row in enumerate(rows):
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
                "format_ok": is_fmt_ok,
                "parse_ok": is_parse_ok,
                "response_chars": len(resp),
                "response": resp,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

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
    ap.add_argument("--sft-max-seq-length", type=int, default=8192)
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
        prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        try:
            run_sft(
                base_model=args.base_model,
                sft_jsonl=args.sft_jsonl,
                sft_output_dir=args.sft_output_dir,
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
