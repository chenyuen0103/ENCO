#!/usr/bin/env python3
import os, sys, csv, time, json, argparse, tempfile, traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import random

from multiprocessing import get_context
from queue import Empty as QueueEmpty  # <-- correct exception

# OPTIONAL: set the start method once, at module import time.
try:
    import multiprocessing as mp
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


# ------------------- Gemini -------------------


def call_gemini(
    model_name: str,
    prompt: str,
    *,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
) -> str:
    """
    Call Gemini exactly once, without retries or multiprocessing.
    Returns the model text, or a '[ERROR] ...' string containing the real exception.
    Also includes finish_reason if the SDK returns a blocked/filtered response.
    """
    try:
        # Prefer GOOGLE_API_KEY if both are set; otherwise fall back to GEMINI_API_KEY
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        if not api_key:
            return "[ERROR] Missing API key (set GOOGLE_API_KEY or GEMINI_API_KEY)."

        # Lazy import so HF-only/vLLM-only use doesn't require google sdk installed
        try:
            from google import genai  # type: ignore
        except Exception as ie:
            return f"[ERROR] Google SDK not available: {type(ie).__name__}: {ie}"

        client = genai.Client(api_key=api_key)

        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"temperature": float(temperature)},
        )

        if getattr(resp, "text", None):
            return resp.text

        cand = (resp.candidates or [None])[0]
        if cand is not None:
            fr = getattr(cand, "finish_reason", None)
            return f"[ERROR] Empty text (finish_reason={fr})"
        return "[ERROR] Empty response with no candidates."

    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return f"[ERROR] {type(e).__name__}: {e}"


def is_gemini_model(model_name: str) -> bool:
    name = (model_name or "").lower()
    return "gemini" in name


# ------------------- Hugging Face (transformers) -------------------


def build_hf_pipeline(
    model_name: str,
    *,
    trust_remote_code: bool = True,
    device_map: Optional[str] = None,
    torch_dtype: Optional[str] = None,
):
    """
    Build a Hugging Face text-generation pipeline for the given model name.
    Requires: transformers (and torch). Returns a callable pipeline.
    """
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            TextGenerationPipeline,
        )  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Hugging Face 'transformers' is required for non-Gemini / non-vLLM HF models.\n"
            "Install with: pip install transformers accelerate torch --upgrade"
        ) from e

    dtype_val = None
    try:
        import torch  # type: ignore
        if torch_dtype:
            td = (torch_dtype or "auto").lower()
            if td == "auto":
                dtype_val = "auto"
            elif td in {"fp16", "float16", "half"}:
                dtype_val = torch.float16
            elif td in {"bf16", "bfloat16"}:
                dtype_val = torch.bfloat16
            elif td in {"fp32", "float32"}:
                dtype_val = torch.float32
            else:
                dtype_val = None
    except Exception:
        dtype_val = None

    try:
        tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=bool(trust_remote_code),
        )
        try:
            tok.padding_side = "left"
            if getattr(tok, "pad_token", None) is None:
                if getattr(tok, "eos_token", None) is not None:
                    tok.pad_token = tok.eos_token
        except Exception:
            pass

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=bool(trust_remote_code),
            device_map=device_map if device_map else None,
            dtype=dtype_val if dtype_val is not None else None,
        )
        try:
            if getattr(model.config, "pad_token_id", None) is None and getattr(tok, "pad_token_id", None) is not None:
                model.config.pad_token_id = tok.pad_token_id
        except Exception:
            pass

        pipe = TextGenerationPipeline(model=model, tokenizer=tok)
        return pipe
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Hugging Face model '{model_name}': {type(e).__name__}: {e}"
        ) from e


def call_hf_textgen(pipe, prompt: str, *, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
    """Generate text using a HF text-generation pipeline."""
    try:
        do_sample = bool(temperature and temperature > 0.0)
        outputs = pipe(
            prompt,
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=float(temperature),
            return_full_text=False,
        )
        if isinstance(outputs, list) and outputs:
            text = outputs[0].get("generated_text", "")
            return text if isinstance(text, str) else str(text)
        return "[ERROR] Empty output from HF pipeline."
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return f"[ERROR] {type(e).__name__}: {e}"


def call_hf_textgen_batch(
    pipe,
    prompts: List[str],
    *,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    batch_size: int = 1,
) -> List[str]:
    """
    Generate text for a batch of prompts using a HF text-generation pipeline.
    Returns a list of strings (one per prompt).
    """
    try:
        do_sample = bool(temperature and temperature > 0.0)
        outputs = pipe(
            prompts,
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=float(temperature),
            return_full_text=False,
            batch_size=int(batch_size),
        )
        result: List[str] = []
        if isinstance(outputs, list):
            for out in outputs:
                text = out.get("generated_text", "")
                result.append(text if isinstance(text, str) else str(text))
        else:
            for out in outputs:
                text = out.get("generated_text", "")
                result.append(text if isinstance(text, str) else str(text))
        return result
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return [f"[ERROR] {type(e).__name__}: {e}"] * len(prompts)


# ------------------- vLLM backend -------------------


def build_vllm_engine(
    model_name: str,
    *,
    tensor_parallel_size: int = 1,
    vllm_dtype: str = "auto",
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = False,
    trust_remote_code: bool = True,
):
    """
    Build a vLLM LLM engine for the given model name.
    Assumes vllm is installed and CUDA / torch are configured.
    """
    try:
        from vllm import LLM  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "vLLM is required for provider 'vllm'. Install with: pip install vllm"
        ) from e

    kwargs: Dict[str, Any] = {
        "model": model_name,
        "tensor_parallel_size": int(tensor_parallel_size),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "trust_remote_code": bool(trust_remote_code),
        "enforce_eager": bool(enforce_eager),
    }

    dt = (vllm_dtype or "auto").lower()
    if dt not in {"", "auto"}:
        # vLLM expects strings like "float16", "bfloat16", "float32"
        # Map a couple of shorthand names.
        if dt in {"fp16", "half"}:
            dt = "float16"
        elif dt in {"bf16"}:
            dt = "bfloat16"
        elif dt in {"fp32"}:
            dt = "float32"
        kwargs["dtype"] = dt

    if max_model_len is not None:
        kwargs["max_model_len"] = int(max_model_len)

    llm = LLM(**kwargs)
    return llm


def call_vllm_textgen(
    llm,
    prompt: str,
    *,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
) -> str:
    """
    Generate text using a vLLM LLM engine for a single prompt.
    """
    try:
        from vllm import SamplingParams  # type: ignore
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return f"[ERROR] vLLM SamplingParams unavailable: {type(e).__name__}: {e}"

    try:
        sp = SamplingParams(
            temperature=float(temperature),
            max_tokens=int(max_new_tokens),
        )
        # vLLM expects a list of prompts
        outputs = llm.generate([prompt], sp)
        if not outputs:
            return "[ERROR] Empty output list from vLLM."
        out0 = outputs[0]
        outs = getattr(out0, "outputs", None)
        if not outs:
            return "[ERROR] Empty outputs field in vLLM result."
        text = outs[0].text
        return text if isinstance(text, str) else str(text)
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return f"[ERROR] {type(e).__name__}: {e}"


def call_vllm_textgen_batch(
    llm,
    prompts: List[str],
    *,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
) -> List[str]:
    """
    Generate text using vLLM for a batch of prompts.
    """
    try:
        from vllm import SamplingParams  # type: ignore
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return [f"[ERROR] vLLM SamplingParams unavailable: {type(e).__name__}: {e}"] * len(prompts)

    try:
        sp = SamplingParams(
            temperature=float(temperature),
            max_tokens=int(max_new_tokens),
        )
        outputs = llm.generate(prompts, sp)
        results: List[str] = []
        for out in outputs:
            outs = getattr(out, "outputs", None)
            if not outs:
                results.append("[ERROR] Empty outputs field in vLLM result.")
                continue
            text = outs[0].text
            results.append(text if isinstance(text, str) else str(text))
        return results
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return [f"[ERROR] {type(e).__name__}: {e}"] * len(prompts)


# ------------------- JSON parsing helper -------------------


def extract_adjacency_matrix(text: str):
    """
    Try to extract the 'adjacency_matrix' field from a JSON object in `text`.
    Returns the adjacency_matrix (list of lists) or None.
    """
    text = text.strip()
    try:
        obj = json.loads(text)
    except Exception:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            obj = json.loads(text[start:end])
        except Exception:
            return None

    if isinstance(obj, dict) and "adjacency_matrix" in obj:
        return obj["adjacency_matrix"]
    return None


# ------------------- Main -------------------


def main():
    ap = argparse.ArgumentParser(
        description="Append LLM responses to CSV with raw_response and prediction columns (Gemini, HF, or vLLM)."
    )
    ap.add_argument(
        "--csv",
        default="./out/cancer/prompts_obs200_int3_shuf5_anon.csv",
        help="Path to the input CSV produced by your generator."
    )
    ap.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Model name. For vLLM/HF, this is a HF model ID or local path."
    )
    ap.add_argument(
        "--provider",
        choices=["auto", "gemini", "hf", "vllm"],
        default="auto",
        help=(
            "Provider backend: 'gemini', 'hf' (transformers), or 'vllm'. "
            "'auto' picks 'gemini' if model name contains 'gemini', else 'hf'."
        ),
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model (all providers)."
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum new tokens for generation (HF/vLLM)."
    )

    # HF specific
    ap.add_argument(
        "--hf-device-map",
        default="auto",
        help="Device map for HF model loading (e.g., 'auto', 'cuda', 'cpu')."
    )
    ap.add_argument(
        "--hf-dtype",
        default="auto",
        help="Torch dtype for HF model (auto, float16, bfloat16, float32)."
    )
    ap.add_argument(
        "--hf-trust-remote-code",
        dest="hf_trust_remote_code",
        action="store_true",
        help="Allow loading custom modeling code from the repo (recommended for Qwen)."
    )
    ap.set_defaults(hf_trust_remote_code=True)

    # vLLM specific
    ap.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (number of GPUs)."
    )
    ap.add_argument(
        "--vllm-dtype",
        default="auto",
        help="vLLM dtype string (auto, float16/fp16, bfloat16/bf16, float32/fp32)."
    )
    ap.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Optional max_model_len for vLLM."
    )
    ap.add_argument(
        "--vllm-gpu-mem-util",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization (0.0–1.0)."
    )
    ap.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="If set, pass enforce_eager=True to vLLM (debugging)."
    )

    ap.add_argument(
        "--prompt-col",
        default="prompt",
        help="Name of the column containing prompts."
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-query and overwrite existing responses in raw_response."
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of NEW rows to process (ignoring rows already saved to out CSV)."
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't call the model; just parse existing raw_response into prediction for new rows."
    )
    ap.add_argument(
        "--timeout-s",
        type=float,
        default=40.0,
        help="(Unused) Per-row hard timeout (seconds)."
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=5,
        help="(Unused) Max retries per row after timeout/error."
    )
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Optional output CSV path. If not given, appends model name to input filename."
    )
    args = ap.parse_args()

    # Decide provider
    provider = args.provider
    if provider == "auto":
        provider = "gemini" if is_gemini_model(args.model) else "hf"

    # Require API key only for Gemini provider and not dry-run
    if provider == "gemini" and not args.dry_run:
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            sys.exit("Missing API key: set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini provider.")

    in_path = Path(args.csv)
    if not in_path.exists():
        sys.exit(f"CSV not found: {in_path}")

    # If user did not explicitly set --max-new-tokens, choose based on filename
    if args.max_new_tokens is None:
        if "_steps" in in_path.stem:
            args.max_new_tokens = 4096
        else:
            args.max_new_tokens = 128
        print(f"[info] Using max_new_tokens={args.max_new_tokens} for {in_path.name}")

    # --------- 1. Read input CSV once ---------
    with in_path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        orig_fieldnames = list(reader.fieldnames or [])
        rows_in = [row for row in reader if any((v or "").strip() for v in row.values())]

    # Ensure output columns
    fieldnames = orig_fieldnames[:]
    for extra in ["raw_response", "prediction"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    # Decide output path
    if args.out_csv is not None:
        out_path = Path(args.out_csv)
    else:
        safe_model_tag = args.model.split("/")[-1]
        for ch in [":", " "]:
            safe_model_tag = safe_model_tag.replace(ch, "_")
        if safe_model_tag not in in_path.stem:
            out_path = in_path.with_name(f"{in_path.stem}_{safe_model_tag}{in_path.suffix}")
        else:
            out_path = in_path

    json_out_path = out_path.with_suffix(".jsonl")

    # --------- 2. Determine resume vs overwrite ---------
    resume = out_path.exists() and not args.overwrite

    if resume:
        already_done = 0
        with out_path.open("r", encoding="utf-8", newline="") as f_existing:
            r_existing = csv.reader(f_existing)
            next(r_existing, None)
            for _ in r_existing:
                already_done += 1
        print(f"[info] Resuming: {already_done} rows already present in {out_path.name}")
    else:
        already_done = 0
        if out_path.exists():
            print(f"[info] Overwrite requested; ignoring existing {out_path.name}")
        else:
            print(f"[info] No existing output file. Starting from scratch: {out_path.name}")

    total_rows = len(rows_in)
    max_new = args.max_rows if args.max_rows is not None else float("inf")

    # --------- 3. Build backend (HF or vLLM) if needed ---------
    hf_pipe = None
    vllm_engine = None

    if provider == "hf" and not args.dry_run:
        try:
            dm = None if not args.hf_device_map or args.hf_device_map == "none" else args.hf_device_map
            hf_pipe = build_hf_pipeline(
                args.model,
                trust_remote_code=bool(args.hf_trust_remote_code),
                device_map=dm,
                torch_dtype=args.hf_dtype,
            )
        except Exception as e:
            sys.exit(str(e))

        try:
            gen_cfg = hf_pipe.model.generation_config
            gen_cfg.temperature = float(args.temperature)
        except Exception:
            pass

    if provider == "vllm" and not args.dry_run:
        try:
            vllm_engine = build_vllm_engine(
                args.model,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                vllm_dtype=args.vllm_dtype,
                max_model_len=args.vllm_max_model_len,
                gpu_memory_utilization=args.vllm_gpu_mem_util,
                enforce_eager=bool(args.vllm_enforce_eager),
                trust_remote_code=True,
            )
        except Exception as e:
            sys.exit(str(e))

    # --------- 4. Open output files in append-or-write mode ---------
    if resume:
        fout = out_path.open("a", encoding="utf-8", newline="")
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
    else:
        fout = out_path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

    if resume and json_out_path.exists():
        jf = json_out_path.open("a", encoding="utf-8")
    else:
        jf = json_out_path.open("w", encoding="utf-8")

    processed_new = 0
    skipped_new = 0

    try:
        with fout, jf, tqdm(total=total_rows, desc="Rows", unit="row") as pbar:
            for idx, row in enumerate(rows_in):
                pbar.update(1)

                if resume and idx < already_done:
                    continue

                prompt = row.get(args.prompt_col, "") or ""
                raw = row.get("raw_response", "") or ""
                pred = row.get("prediction", "") or ""

                if processed_new >= max_new:
                    writer.writerow(row)
                    jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    skipped_new += 1
                    continue

                if raw.strip() and not pred.strip():
                    adj = extract_adjacency_matrix(raw)
                    if adj is not None:
                        row["prediction"] = json.dumps(adj, ensure_ascii=False)

                need_call = (
                    not args.dry_run
                    and bool(prompt)
                    and (args.overwrite or not raw.strip())
                )

                if need_call:
                    if provider == "gemini":
                        resp = call_gemini(
                            args.model,
                            prompt,
                            temperature=args.temperature,
                        )
                    elif provider == "hf":
                        resp = call_hf_textgen(
                            hf_pipe,
                            prompt,
                            temperature=args.temperature,
                            max_new_tokens=args.max_new_tokens,
                        )
                    elif provider == "vllm":
                        resp = call_vllm_textgen(
                            vllm_engine,
                            prompt,
                            temperature=args.temperature,
                            max_new_tokens=args.max_new_tokens,
                        )
                    else:
                        resp = "[ERROR] Unknown provider"

                    row["raw_response"] = resp

                    adj = extract_adjacency_matrix(resp)
                    if adj is not None:
                        row["prediction"] = json.dumps(adj, ensure_ascii=False)
                    else:
                        row["prediction"] = ""

                    processed_new += 1
                else:
                    skipped_new += 1

                writer.writerow(row)
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")

                if (idx + 1) % 10 == 0:
                    fout.flush()
                    jf.flush()

                pbar.set_postfix(processed_new=processed_new, skipped_new=skipped_new)

    finally:
        try:
            fout.flush()
        except Exception:
            pass
        try:
            jf.flush()
        except Exception:
            pass

    print(
        f"\nDone.\n"
        f"- Input rows: {total_rows}\n"
        f"- Previously done (resumed): {already_done}\n"
        f"- Newly processed: {processed_new}\n"
        f"- Newly skipped (no call / max_rows): {skipped_new}\n"
        f"- CSV: {out_path}\n"
        f"- JSONL: {json_out_path}"
    )


if __name__ == "__main__":
    main()
