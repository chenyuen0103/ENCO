#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Optional


def _read_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> str:
    if prompt and prompt_file:
        raise SystemExit("Pass exactly one of --prompt or --prompt-file.")
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8")
    if prompt:
        return prompt
    raise SystemExit("You must pass --prompt or --prompt-file.")


def _resolve_dtype(torch, name: str):
    key = (name or "auto").lower()
    if key == "auto":
        return "auto"
    if key in {"float16", "fp16", "half"}:
        return torch.float16
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if key in {"float32", "fp32"}:
        return torch.float32
    raise SystemExit(f"Unsupported --dtype: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prompt a Hugging Face model from CLI.")
    ap.add_argument("--model", required=True, help="HF model ID, e.g. Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--prompt", default=None, help="Prompt text.")
    ap.add_argument("--prompt-file", default=None, help="Path to a UTF-8 prompt file.")
    ap.add_argument("--system", default=None, help="Optional system message for chat-template models.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--device-map", default="auto", help="Transformers device_map (default: auto).")
    ap.add_argument("--dtype", default="auto", help="auto|bfloat16|float16|float32")
    ap.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code.")
    ap.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template and send raw text directly.",
    )
    args = ap.parse_args()

    prompt = _read_prompt(args.prompt, args.prompt_file)

    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing dependency. Install with:\n"
            "  pip install -U transformers accelerate torch"
        ) from e

    dtype = _resolve_dtype(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=args.device_map if args.device_map else None,
        torch_dtype=dtype,
    )
    model.eval()

    use_chat_template = (not args.no_chat_template) and hasattr(tokenizer, "apply_chat_template")
    if use_chat_template:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": prompt})
        model_input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        model_input_text = prompt

    inputs = tokenizer(model_input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = args.temperature > 0.0
    gen_kwargs = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": do_sample,
        "repetition_penalty": float(args.repetition_penalty),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(args.temperature)
        gen_kwargs["top_p"] = float(args.top_p)

    gen = model.generate(**inputs, **gen_kwargs)

    output_ids = gen[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    sys.stdout.write(text + ("\n" if not text.endswith("\n") else ""))


if __name__ == "__main__":
    main()
