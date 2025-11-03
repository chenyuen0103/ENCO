#!/usr/bin/env python3
import os, sys, csv, time, json, argparse, tempfile, traceback
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import random
# ---------- Subprocess wrapper to enforce hard timeouts ----------
# --- imports you need at module top ---
from multiprocessing import get_context
from queue import Empty as QueueEmpty  # <-- correct exception
# --- imports near top ---


# OPTIONAL: set the start method once, at module import time.
# On some clusters, calling set_start_method multiple times raises.
try:
    import multiprocessing as mp
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

def _worker_call(model_name: str, prompt: str, temperature: float, api_key: str, out_q) -> None:
    """
    Run inside a child process: call the SDK and put a result dict on the queue.
    Must NEVER raise—always put a dict into out_q.
    """
    try:
        # Ensure the child has the key even if parent set it only in-process.
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        # New SDK: github.com/googleapis/python-genai
        from google import genai
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))

        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"temperature": temperature},
        )

        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):
            # very defensive fallback
            cand = resp.candidates[0]
            parts = getattr(cand, "content", getattr(cand, "contents", None))
            text = getattr(parts[0], "text", "") if parts else ""
        out_q.put({"ok": True, "text": text or ""})
    except Exception as e:
        out_q.put({"ok": False, "err": f"{type(e).__name__}: {e}"})


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

        # Lazy import so HF-only use doesn't require google sdk installed
        try:
            from google import genai  # type: ignore
        except Exception as ie:
            return f"[ERROR] Google SDK not available: {type(ie).__name__}: {ie}"

        client = genai.Client(api_key=api_key)

        # NOTE: temperature→0 for deterministic evaluation runs
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"temperature": temperature},
        )

        # If text exists, return it
        if getattr(resp, "text", None):
            return resp.text

        # Otherwise try to explain what happened (finish reason / safety)
        # SDK schema: resp.candidates[i].finish_reason and .safety_ratings may be present.
        cand = (resp.candidates or [None])[0]
        if cand is not None:
            fr = getattr(cand, "finish_reason", None)
            return f"[ERROR] Empty text (finish_reason={fr})"
        return "[ERROR] Empty response with no candidates."

    except Exception as e:
        # Print the full traceback once to stderr (so you see the exact error)
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        # Return a short string into the CSV cell
        return f"[ERROR] {type(e).__name__}: {e}"


def is_gemini_model(model_name: str) -> bool:
    name = (model_name or "").lower()
    return "gemini" in name


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
        # Prefer explicit model/tokenizer construction to pass trust_remote_code and device_map
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            TextGenerationPipeline,
        )  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Hugging Face 'transformers' is required for non-Gemini models.\n"
            "Install with: pip install transformers accelerate torch --upgrade"
        ) from e

    # Optional torch dtype handling
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
        # If no dtype given, leave None so HF chooses defaults
    except Exception:
        dtype_val = None

    try:
        tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=bool(trust_remote_code),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=bool(trust_remote_code),
            device_map=device_map if device_map else None,
            torch_dtype=dtype_val if dtype_val is not None else None,
        )

        pipe = TextGenerationPipeline(model=model, tokenizer=tok)
        return pipe
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Hugging Face model '{model_name}': {type(e).__name__}: {e}"
        ) from e


def call_hf_textgen(pipe, prompt: str, *, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
    """Generate text using a HF text-generation pipeline."""
    try:
        do_sample = temperature and temperature > 0.0
        outputs = pipe(
            prompt,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
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


def main():
    ap = argparse.ArgumentParser(
        description="Append LLM responses to CSV as a new column named by the model (Gemini or Hugging Face)."
    )
    ap.add_argument(
        "--csv",
        default="/home/yuen_chen/ENCO/experiments/out/cancer/prompts_obs200_int3_shuf5_anon.csv",
        help="Path to the input CSV produced by your generator."
    )
    ap.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Model name. If it contains 'gemini', uses Google Gemini; otherwise loads from Hugging Face."
    )
    ap.add_argument(
        "--provider",
        choices=["auto", "gemini", "hf"],
        default="auto",
        help="Force provider. Default 'auto' picks 'gemini' if model name contains 'gemini', else 'hf'."
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the model (both providers)."
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens for Hugging Face text generation."
    )
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
    ap.add_argument(
        "--prompt-col",
        default="prompt",
        help="Name of the column containing prompts."
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-query and overwrite existing responses in the model column."
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of rows to process."
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't call the model; just copy and add column if missing."
    )
    ap.add_argument(
        "--timeout-s",
        type=float,
        default=40.0,
        help="Per-row hard timeout (seconds)."
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Max retries per row after timeout/error."
    )
    args = ap.parse_args()

    # Decide provider
    provider = args.provider
    if provider == "auto":
        provider = "gemini" if is_gemini_model(args.model) else "hf"

    # Require API key only for Gemini provider and not dry-run
    if provider == "gemini" and not args.dry_run:
        # Prefer GOOGLE_API_KEY; fall back to GEMINI_API_KEY
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            sys.exit("Missing API key: set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini provider.")

    in_path = Path(args.csv)
    if not in_path.exists():
        sys.exit(f"CSV not found: {in_path}")

    # --------- 1. Read once, fully, via DictReader ---------
    with in_path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        orig_fieldnames = list(reader.fieldnames or [])
        rows_in = [row for row in reader if any(v.strip() for v in row.values())]
        # ^ filter out rows that are "all-empty" (only commas / blanks),
        #   so tqdm reflects real usable rows.

    # Add/ensure model column
    model_col = args.model
    fieldnames = orig_fieldnames[:]
    if model_col not in fieldnames:
        fieldnames.append(model_col)

    # Respect max_rows limit for how many we'll *call* the API on
    # (We'll still output all rows.)
    max_to_process = args.max_rows if args.max_rows is not None else float("inf")

    # --------- 2. Prepare temp output for atomic replace ---------
    tmp_fh = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="",
        dir=str(in_path.parent),
        delete=False,
        prefix=in_path.stem + ".",
        suffix=".tmp"
    )
    tmp_path = Path(tmp_fh.name)

    processed = 0
    skipped = 0

    # Build HF pipeline once if needed
    hf_pipe = None
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
        with tmp_fh as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            # tqdm total is actual # of data rows after filtering empties
            with tqdm(total=len(rows_in),
                      desc="Rows",
                      unit="row") as pbar:

                for i, row in enumerate(rows_in):

                    prompt = row.get(args.prompt_col, "")
                    # If we've already filled this cell and !overwrite -> skip
                    already = row.get(model_col, "")
                    existing = already.strip()
                    need_call = (
                        (processed < max_to_process) and
                        (prompt != "") and
                        (args.overwrite or not already) and 
                        (existing == "")
                    )

                    if args.dry_run:
                        # dry run never calls model
                        need_call = False

                    if need_call:
                        # Call the selected provider
                        if provider == "gemini":
                            response_text = call_gemini(
                                args.model,
                                prompt,
                                temperature=args.temperature,
                            )
                        else:
                            # Hugging Face
                            response_text = call_hf_textgen(
                                hf_pipe,
                                prompt,
                                temperature=args.temperature,
                                max_new_tokens=args.max_new_tokens,
                            )
                        row[model_col] = response_text
                        processed += 1
                    else:
                        # We didn't generate a new answer
                        if not prompt:
                            skipped += 1
                        elif already and not args.overwrite:
                            skipped += 1
                        else:
                            # hit max_rows limit
                            skipped += 1

                    writer.writerow(row)

                    # Progress bar housekeeping
                    pbar.update(1)
                    pbar.set_postfix(processed=processed, skipped=skipped)

                    # Periodic flush
                    if (i + 1) % 50 == 0:
                        fout.flush()

        # --------- 3. Atomic replace original file ---------
        os.replace(tmp_path, in_path)

        print(
            f"Updated '{in_path}'. "
            f"Column='{model_col}'. "
            f"Processed={processed}, Skipped={skipped}"
        )

    except Exception:
        # On any error, best-effort cleanup the temp
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        finally:
            raise


# def main():
#     os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")
#     assert os.environ["GOOGLE_API_KEY"], "No key found in GEMINI_API_KEY"
#     client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

#     resp = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents="Hello!"
#     )
#     print(resp.text)


if __name__ == "__main__":
    main()
