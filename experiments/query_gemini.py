#!/usr/bin/env python3
import os, sys, csv, time, json, argparse, tempfile, traceback
import inspect
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import random
import re
# ---------- Subprocess wrapper to enforce hard timeouts ----------
# --- imports you need at module top ---
from multiprocessing import get_context
from queue import Empty as QueueEmpty  # <-- correct exception
import numpy as np

FORMAT_RE = re.compile(r"(?s)^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$")
ANSWER_RE = re.compile(r"(?s)<answer>\s*(.*?)\s*</answer>")

# ---- OpenAI token counting helper ----
try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None

# OpenAI Responses API per-string input length cap.
OPENAI_MAX_INPUT_CHARS = 10_485_760


def count_openai_tokens(model_name: str, text: str) -> int:
    """
    Return the number of tokens this text would use for a given OpenAI model.
    If tiktoken is not available, returns -1.
    """
    if tiktoken is None:
        return -1
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to a reasonable default for modern GPT models
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def classify_error_type(raw_response: str) -> str:
    """
    Map standardized '[ERROR] ...' responses to a compact error_type label.
    Returns empty string for non-error responses.
    """
    text = (raw_response or "").strip()
    if not text.startswith("[ERROR]"):
        return ""
    low = text.lower()
    if "input too long" in low or "string_above_max_length" in low:
        return "input_too_long"
    if "rate limit" in low or "ratelimit" in low or "too many requests" in low:
        return "rate_limit"
    if "timed out" in low or "timeout" in low:
        return "timeout"
    if "missing api key" in low:
        return "missing_api_key"
    if "device-side assert triggered" in low or "acceleratorerror" in low:
        return "cuda_device_assert"
    if "out of memory" in low or "cuda out of memory" in low:
        return "oom"
    return "error"


def edges_to_adjacency(edges: Any, variables: Any) -> Optional[np.ndarray]:
    """
    Convert an edge list [[src, dst], ...] into a square adjacency matrix.
    If variables is not a non-empty list, infer variable order from first appearance in edges.
    """
    if not isinstance(edges, list):
        return None
    try:
        if not isinstance(variables, list) or not variables:
            seen = set()
            inferred = []
            for e in edges:
                if not isinstance(e, (list, tuple)) or len(e) != 2:
                    continue
                s, d = str(e[0]), str(e[1])
                if s not in seen:
                    inferred.append(s)
                    seen.add(s)
                if d not in seen:
                    inferred.append(d)
                    seen.add(d)
            variables = inferred
        if not variables:
            return None

        var_to_idx = {v: i for i, v in enumerate(variables)}
        n = len(variables)
        A = np.zeros((n, n), dtype=int)
        for e in edges:
            if not isinstance(e, (list, tuple)) or len(e) != 2:
                continue
            s, d = str(e[0]), str(e[1])
            if s in var_to_idx and d in var_to_idx:
                A[var_to_idx[s], var_to_idx[d]] = 1
        return A
    except Exception:
        return None

# --- imports near top ---


# OPTIONAL: set the start method once, at module import time.
# On some clusters, calling set_start_method multiple times raises.
try:
    import multiprocessing as mp
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass



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
    

# def call_openai(
#     model_name: str,
#     prompt: str,
#     *,
#     temperature: float = 0.0,
#     api_key: Optional[str] = None,
#     max_retries: int = 0,
#     request_timeout: float = 20.0,
# ) -> str:
#     """
#     Call an OpenAI chat model once (or with a small, explicit number of retries).
#     Returns model text, or a '[ERROR] ...' string.
#     """
#     try:
#         if api_key is None:
#             api_key = os.getenv("OPENAI_API_KEY") or ""
#         if not api_key:
#             return "[ERROR] Missing API key (set OPENAI_API_KEY)."

#         try:
#             from openai import OpenAI  # type: ignore
#         except Exception as ie:
#             return f"[ERROR] OpenAI SDK not available: {type(ie).__name__}: {ie}"

#         # IMPORTANT: disable/limit automatic retries and set a per-request timeout
#         client = OpenAI(
#             api_key=api_key,
#             max_retries=max_retries,       # 0 = no automatic retries
#             timeout=request_timeout,       # seconds for network / HTTP timeouts
#         )

#         # resp = client.chat.completions.create(
#         #     model=model_name,
#         #     messages=[{"role": "user", "content": prompt}],
#         #     temperature=float(temperature),
#         # )
#         resp = client.responses.create(
#             model=model_name,
#             input=prompt,
#             )

#         if not resp.choices:
#             return "[ERROR] Empty response (no choices)."

#         # msg = resp.choices[0].message
#         # text = getattr(msg, "content", "") or ""
#         msg = resp.output[1].content[0]
#         text = msg['text']
#         return text

#     except KeyboardInterrupt:
#         # Let Ctrl-C bubble out so you can stop the whole run cleanly
#         raise

#     except Exception as e:
#         tb = traceback.format_exc()
#         print(tb, file=sys.stderr)
#         return f"[ERROR] {type(e).__name__}: {e}"


# Heuristic: some OpenAI models enforce default decoding and reject any temperature value.
# Adjust this list/pattern as you observe behavior.
_OPENAI_FIXED_TEMP_PAT = re.compile(r"^(o\d|o-|o3|o4)\b", re.IGNORECASE)
_OPENAI_FIXED_TEMP_SET = {
    "gpt-4.1", "gpt-4.1-mini",
    # add other exact names if you encounter them
}

def _model_requires_default_temperature(model_name: str) -> bool:
    mn = (model_name or "").strip()
    return bool(_OPENAI_FIXED_TEMP_PAT.match(mn)) or (mn in _OPENAI_FIXED_TEMP_SET)

def _infer_dataset_name_from_csv_path(csv_path: Path) -> Optional[str]:
    """
    Infer the dataset name from a prompt CSV path.

    Expected layouts include:
      - prompts/experiment1/<dataset>/.../prompts_*.csv
      - prompts/<dataset>/.../prompts_*.csv
      - out/<dataset>/.../prompts_*.csv
    Returns None if it can't be inferred safely.
    """
    parts = list(csv_path.parts)
    for anchor in ("experiment1", "prompts", "out"):
        if anchor not in parts:
            continue
        i = parts.index(anchor)
        j = i + 1
        if anchor == "prompts" and j < len(parts) - 1 and parts[j] == "experiment1":
            j += 1
        if j < len(parts) - 1:
            return parts[j]
    return None



def call_openai(
    model_name: str,
    prompt: str,
    *,
    temperature: float = 0.0,   # caller can still pass 0.0, we'll omit it on the wire
    api_key: Optional[str] = None,
    max_retries: int = 0,
    request_timeout: float = 6000.0,
) -> str:
    """
    Call an OpenAI model once via the Responses API.
    - Omits 'temperature' if it is None, 0.0, or 1.0 (default-only models like 'gpt-5-mini').
    - Returns '[ERROR] ...' on failure.
    """
    try:
        prompt_len = len(prompt)
        if prompt_len > OPENAI_MAX_INPUT_CHARS:
            return (
                "[ERROR][input_too_long] OpenAI input too long: "
                f"{prompt_len} chars > {OPENAI_MAX_INPUT_CHARS} max."
            )

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY") or ""
        if not api_key:
            return "[ERROR] Missing API key (set OPENAI_API_KEY)."

        try:
            from openai import OpenAI
        except Exception as ie:
            return f"[ERROR] OpenAI SDK not available: {type(ie).__name__}: {ie}"

        client = OpenAI(
            api_key=api_key,
            max_retries=max_retries,
            timeout=request_timeout,
        )

        def _is_rate_limitish(err: Exception) -> bool:
            msg = str(err).lower()
            return any(
                token in msg
                for token in (
                    "rate limit",
                    "ratelimit",
                    "too many requests",
                    "tpm",
                    "rpm",
                    "429",
                )
            )

        def _is_transientish(err: Exception) -> bool:
            """
            Best-effort detection of transient OpenAI/API transport errors worth retrying.
            We use string/type checks to avoid depending on exact SDK exception classes.
            """
            name = type(err).__name__.lower()
            msg = str(err).lower()
            if any(k in name for k in ("internalservererror", "apiconnectionerror", "apitimeouterror")):
                return True
            return any(
                token in msg
                for token in (
                    "502",
                    "503",
                    "504",
                    "bad gateway",
                    "service unavailable",
                    "gateway timeout",
                    "cloudflare",
                    "temporarily unavailable",
                    "connection reset",
                    "connection aborted",
                    "connection error",
                    "timed out",
                )
            )

        # Build request kwargs. Do NOT include temperature if it's default-like.
        req: Dict[str, Any] = {"model": model_name, "input": prompt}
        try:
            t = float(temperature) if temperature is not None else None
        except Exception:
            t = None

        # Only include temperature if it is a non-default value not equal to 1.0.
        if t is not None and t not in (0.0, 1.0):
            req["temperature"] = t  # include only when explicitly non-default

        # Retry on transient-ish failures (rate limit + 5xx/transport).
        last_exc: Optional[Exception] = None
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                resp = client.responses.create(**req)
                break
            except Exception as e:
                last_exc = e
                if _is_rate_limitish(e):
                    # Conservative backoff for TPM/RPM bursts.
                    sleep_s = 60.0 if attempt == 0 else min(120.0, 30.0 * (attempt + 1))
                    print(
                        f"[warn] OpenAI rate limit (attempt {attempt+1}/{max_attempts}). "
                        f"Sleeping {sleep_s:.0f}s then retrying. Error: {e}",
                        file=sys.stderr,
                    )
                    time.sleep(sleep_s)
                    continue
                if _is_transientish(e) and attempt < (max_attempts - 1):
                    # Exponential backoff with jitter, capped.
                    import random

                    base = min(60.0, 2.0 * (2 ** attempt))
                    sleep_s = base + random.random() * 0.5 * base
                    print(
                        f"[warn] OpenAI transient error (attempt {attempt+1}/{max_attempts}). "
                        f"Sleeping {sleep_s:.1f}s then retrying. Error: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    time.sleep(sleep_s)
                    continue
                raise
        else:
            # Should be unreachable, but keep mypy happy.
            if last_exc is not None:
                raise last_exc

        # Preferred extraction path
        text = getattr(resp, "output_text", None)
        if text:
            return text

        # Fallbacks
        try:
            if hasattr(resp, "output") and resp.output:
                for blk in resp.output:
                    parts = getattr(blk, "content", None)
                    if isinstance(parts, list):
                        for part in parts:
                            # dict-like
                            if isinstance(part, dict) and part.get("type") == "output_text":
                                return part.get("text", "")
                            # object-like
                            if getattr(part, "type", None) == "output_text":
                                t = getattr(part, "text", "") or ""
                                if t:
                                    return t
                    t = getattr(blk, "text", "") or ""
                    if t:
                        return t
            if hasattr(resp, "choices") and resp.choices:
                msg = getattr(resp.choices[0], "message", None)
                if msg is not None:
                    t = getattr(msg, "content", "") or ""
                    if t:
                        return t
        except Exception:
            pass

        return "[ERROR] Empty response (no text found)."

    except KeyboardInterrupt:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return f"[ERROR] {type(e).__name__}: {e}"


def is_gemini_model(model_name: str) -> bool:
    name = (model_name or "").lower()
    return "gemini" in name

def is_openai_model(model_name: str) -> bool:
    """
    Heuristic: treat GPT / o* models as OpenAI API models.
    Adjust this if you have custom naming.
    """
    name = (model_name or "").lower()
    return any(tag in name for tag in ("gpt", "o1-", "o3-", "omni"))

def build_hf_pipeline(
    model_name: str,
    *,
    trust_remote_code: bool = True,
    device_map: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    clear_length_cap: bool = True,
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
        # For decoder-only models, use left padding to avoid generation issues
        try:
            tok.padding_side = "left"
            # Ensure a pad token exists; many decoder-only models reuse EOS
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
        # Align model pad_token_id with tokenizer if missing
        try:
            if getattr(model.config, "pad_token_id", None) is None and getattr(tok, "pad_token_id", None) is not None:
                model.config.pad_token_id = tok.pad_token_id
        except Exception:
            pass

        # Optional: remove model-default generation caps so "no explicit cap"
        # means generation is not length-limited by max_new_tokens/max_length.
        if clear_length_cap:
            try:
                gen_cfg = model.generation_config
                if hasattr(gen_cfg, "max_new_tokens"):
                    gen_cfg.max_new_tokens = None
                if hasattr(gen_cfg, "max_length"):
                    gen_cfg.max_length = None
            except Exception:
                pass

        pipe = TextGenerationPipeline(model=model, tokenizer=tok)
        return pipe
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Hugging Face model '{model_name}': {type(e).__name__}: {e}"
        ) from e


def _prepare_hf_prompt(tokenizer: Any, prompt: str) -> str:
    """
    Apply model-native chat template when available.
    For Qwen3 tokenizers, enable_thinking=True if supported.
    Falls back to the raw prompt string for non-chat tokenizers.
    """
    text = str(prompt)
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return text

    messages = [{"role": "user", "content": text}]
    kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        params = inspect.signature(tokenizer.apply_chat_template).parameters
        if "enable_thinking" in params:
            kwargs["enable_thinking"] = True
    except (TypeError, ValueError):
        pass

    try:
        return str(tokenizer.apply_chat_template(messages, **kwargs))
    except TypeError:
        kwargs.pop("enable_thinking", None)
        try:
            return str(tokenizer.apply_chat_template(messages, **kwargs))
        except Exception:
            return text
    except Exception:
        return text


def call_hf_textgen(pipe, prompt: str, *, temperature: float = 0.0, max_new_tokens: Optional[int] = None) -> str:
    """Generate text using a HF text-generation pipeline."""
    try:
        prompt = _prepare_hf_prompt(getattr(pipe, "tokenizer", None), prompt)
        do_sample = temperature and temperature > 0.0
        gen_kwargs: Dict[str, Any] = {
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "return_full_text": False,
        }
        if max_new_tokens is not None and int(max_new_tokens) > 0:
            gen_kwargs["max_new_tokens"] = int(max_new_tokens)
        outputs = pipe(prompt, **gen_kwargs)
        if isinstance(outputs, list) and outputs:
            text = outputs[0].get("generated_text", "")
            return text if isinstance(text, str) else str(text)
        return "[ERROR] Empty output from HF pipeline."
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return f"[ERROR] {type(e).__name__}: {e}"

def call_hf_textgen_batch(pipe, prompts, *, temperature: float = 0.0,
                          max_new_tokens: Optional[int] = None, batch_size: int = 1) -> list[str]:
    """
    Generate text for a batch of prompts using a HF text-generation pipeline.

    Returns a list of strings (one per prompt).
    """
    try:
        tok = getattr(pipe, "tokenizer", None)
        prompts = [_prepare_hf_prompt(tok, p) for p in prompts]
        do_sample = temperature and temperature > 0.0

        gen_kwargs: Dict[str, Any] = {
            "return_full_text": False,
            "batch_size": int(batch_size),
        }
        if max_new_tokens is not None and int(max_new_tokens) > 0:
            gen_kwargs["max_new_tokens"] = int(max_new_tokens)
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)

        outputs = pipe(prompts, **gen_kwargs)

        # Normalize outputs:
        # - HF often returns [[{...}], [{...}], ...]
        # - Sometimes returns [{...}, {...}, ...]
        norm_outputs = []
        if isinstance(outputs, list):
            for item in outputs:
                if isinstance(item, list):
                    # e.g. [{'generated_text': '...'}]
                    norm_outputs.append(item[0] if item else {})
                else:
                    norm_outputs.append(item)
        else:
            norm_outputs = [outputs]

        result: list[str] = []
        for out in norm_outputs:
            if isinstance(out, dict):
                text = out.get("generated_text", "")
            else:
                text = str(out)
            result.append(text if isinstance(text, str) else str(text))

        return result

    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        # propagate an explicit error string for each prompt
        return [f"[ERROR] {type(e).__name__}: {e}"] * len(prompts)


def _extract_answer_text(text: str) -> str:
    m = ANSWER_RE.search(text or "")
    return m.group(1) if m else (text or "")


def _extract_adjacency_from_response(
    text: str,
    *,
    fallback_variables: Optional[list[str]] = None,
) -> Optional[np.ndarray]:
    answer_text = _extract_answer_text(text)
    mat = extract_adjacency_matrix(answer_text, fallback_variables=fallback_variables)
    if mat is not None:
        return mat
    if answer_text != (text or ""):
        return extract_adjacency_matrix(text, fallback_variables=fallback_variables)
    return None


def _format_ok(text: str) -> int:
    return int(bool(FORMAT_RE.match(text or "")))


def _truncation_suspected(text: str, *, output_tokens: int, max_new_tokens_hint: Optional[int]) -> int:
    t = text or ""
    missing_close_tags = (
        ("<think>" in t and "</think>" not in t)
        or ("<answer>" in t and "</answer>" not in t)
    )
    near_limit = False
    if max_new_tokens_hint is not None and max_new_tokens_hint > 0 and output_tokens >= 0:
        near_limit = output_tokens >= int(0.98 * max_new_tokens_hint)
    return int(missing_close_tags or near_limit)


def extract_adjacency_matrix(text: str, *, fallback_variables: Optional[list[str]] = None) -> Optional[np.ndarray]:
    """
    Robustly extract a square adjacency matrix from a messy LLM response.
    Returns a (N, N) numpy array of ints, or None.
    """

    if not text:
        return None

    # Helper: convert list-of-lists to square numpy array
    def _normalize_matrix(mat: Any) -> Optional[np.ndarray]:
        if not isinstance(mat, list) or not mat:
            return None
        try:
            rows: List[List[int]] = [[int(x) for x in row] for row in mat]
        except Exception:
            return None
        n = len(rows)
        if any(len(r) != n for r in rows):
            return None
        try:
            arr = np.asarray(rows, dtype=int)
        except Exception:
            return None
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            return None
        return arr

    def _from_obj(obj: Any) -> Optional[np.ndarray]:
        if isinstance(obj, dict):
            if "adjacency_matrix" in obj:
                mat = _normalize_matrix(obj["adjacency_matrix"])
                if mat is not None:
                    return mat
            if "edges" in obj:
                vars_for_edges = obj.get("variables")
                if not isinstance(vars_for_edges, list):
                    vars_for_edges = fallback_variables
                return edges_to_adjacency(obj.get("edges"), vars_for_edges)
            ans = obj.get("answer")
            if isinstance(ans, dict):
                if "adjacency_matrix" in ans:
                    mat = _normalize_matrix(ans["adjacency_matrix"])
                    if mat is not None:
                        return mat
                if "edges" in ans:
                    vars_for_edges = ans.get("variables")
                    if not isinstance(vars_for_edges, list):
                        vars_for_edges = fallback_variables
                    return edges_to_adjacency(ans.get("edges"), vars_for_edges)
        return _normalize_matrix(obj)

    def _try_one_variant(txt: str) -> Optional[np.ndarray]:
        txt = txt.strip()

        # 1) Whole-string JSON
        try:
            obj = json.loads(txt)
            m = _from_obj(obj)
            if m is not None:
                return m
        except Exception:
            pass

        # 2) Objects that start with "variables"
        for m in re.finditer(r'\{\s*"variables"\s*:[\s\S]*?\}', txt):
            frag = m.group(0)
            try:
                obj = json.loads(frag)
                mat = _from_obj(obj)
                if mat is not None:
                    return mat
            except Exception:
                continue

        # 3) Generic { ... } blocks
        for m in re.finditer(r"\{[\s\S]*?\}", txt):
            frag = m.group(0)
            try:
                obj = json.loads(frag)
                mat = _from_obj(obj)
                if mat is not None:
                    return mat
            except Exception:
                continue

        # 4) `"adjacency_matrix": [ ... ]` via bracket-balancing
        for m in re.finditer(r'"adjacency_matrix"\s*:', txt):
            start = m.end()
            lb = txt.find("[", start)
            if lb == -1:
                continue
            depth = 0
            for i in range(lb, len(txt)):
                ch = txt[i]
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        candidate = txt[lb:i+1]
                        try:
                            obj = json.loads(candidate)
                            mat = _normalize_matrix(obj)
                            if mat is not None:
                                return mat
                        except Exception:
                            pass
                        break  # done with this "adjacency_matrix" block

        # 5) Any [[...]]-style matrix
        for m in re.finditer(r"\[\s*\[", txt):
            lb = m.start()
            depth = 0
            seen_inner = False
            for i in range(lb, len(txt)):
                ch = txt[i]
                if ch == "[":
                    depth += 1
                    if depth > 1:
                        seen_inner = True
                elif ch == "]":
                    depth -= 1
                    if depth == 0 and seen_inner:
                        candidate = txt[lb:i+1]
                        try:
                            obj = json.loads(candidate)
                            mat = _normalize_matrix(obj)
                            if mat is not None:
                                return mat
                        except Exception:
                            pass
                        break

        return None

    # First try the text as-is
    mat = _try_one_variant(text)
    if mat is not None:
        return mat

    # If we have literal "\n" sequences (backslash+n) but few real newlines,
    # try again with "\n" converted to actual newlines.
    if "\\n" in text:
        alt = text.replace("\\n", "\n")
        if alt != text:
            mat = _try_one_variant(alt)
            if mat is not None:
                return mat

    return None

# def main():
#     ap = argparse.ArgumentParser(
#         description="Append LLM responses to CSV with raw_response and prediction columns (Gemini or Hugging Face)."
#     )
#     ap.add_argument(
#         "--csv",
#         default="./experiments/out/cancer/prompts_obs200_int3_shuf5_anon.csv",
#         help="Path to the input CSV produced by your generator."
#     )
#     ap.add_argument(
#         "--model",
#         default="gemini-2.5-flash",
#         help="Model name. If it contains 'gemini', uses Google Gemini; otherwise loads from Hugging Face."
#     )
#     ap.add_argument(
#         "--provider",
#         choices=["auto", "gemini", "hf"],
#         default="auto",
#         help="Force provider. Default 'auto' picks 'gemini' if model name contains 'gemini', else 'hf'."
#     )
#     ap.add_argument(
#         "--temperature",
#         type=float,
#         default=0.0,
#         help="Sampling temperature for the model (both providers)."
#     )
#     ap.add_argument(
#         "--max-new-tokens",
#         type=int,
#         default=None,
#         help="Maximum new tokens for Hugging Face text generation."
#     )
#     ap.add_argument(
#         "--hf-device-map",
#         default="auto",
#         help="Device map for HF model loading (e.g., 'auto', 'cuda', 'cpu')."
#     )
#     ap.add_argument(
#         "--hf-dtype",
#         default="auto",
#         help="Torch dtype for HF model (auto, float16, bfloat16, float32)."
#     )
#     ap.add_argument(
#         "--hf-trust-remote-code",
#         dest="hf_trust_remote_code",
#         action="store_true",
#         help="Allow loading custom modeling code from the repo (recommended for Qwen)."
#     )
#     ap.add_argument(
#         "--hf-batch-size",
#         type=int,
#         default=4,
#         help="Batch size for Hugging Face generation (number of prompts per forward pass).",
#     )

#     ap.set_defaults(hf_trust_remote_code=True)
#     ap.add_argument(
#         "--prompt-col",
#         default="prompt",
#         help="Name of the column containing prompts."
#     )
#     ap.add_argument(
#         "--overwrite",
#         action="store_true",
#         help="Re-query and overwrite existing responses in raw_response."
#     )
#     ap.add_argument(
#         "--max-rows",
#         type=int,
#         default=None,
#         help="Optional cap on number of rows to process."
#     )
#     ap.add_argument(
#         "--dry-run",
#         action="store_true",
#         help="Don't call the model; just parse existing raw_response into prediction."
#     )
#     ap.add_argument(
#         "--timeout-s",
#         type=float,
#         default=40.0,
#         help="(Unused) Per-row hard timeout (seconds)."
#     )
#     ap.add_argument(
#         "--retries",
#         type=int,
#         default=5,
#         help="(Unused) Max retries per row after timeout/error."
#     )
#     ap.add_argument(
#         "--out-csv",
#         default=None,
#         help="Optional output CSV path. If not given, appends model name to input filename."
#     )
#     args = ap.parse_args()

#     # Decide provider
#     provider = args.provider
#     if provider == "auto":
#         provider = "gemini" if is_gemini_model(args.model) else "hf"

#     # Require API key only for Gemini provider and not dry-run
#     if provider == "gemini" and not args.dry_run:
#         if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
#             sys.exit("Missing API key: set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini provider.")

#     in_path = Path(args.csv)
#     if not in_path.exists():
#         sys.exit(f"CSV not found: {in_path}")
#     # If user did not explicitly set --max-new-tokens, choose based on filename
#     if args.max_new_tokens is None:
#         if "_steps" in in_path.stem:
#             args.max_new_tokens = 4096
#         else:
#             args.max_new_tokens = 128
#         print(f"[info] Using max_new_tokens={args.max_new_tokens} for {in_path.name}")

#     # --------- 1. Read once, fully, via DictReader ---------
#     with in_path.open("r", encoding="utf-8", newline="") as fin:
#         reader = csv.DictReader(fin)
#         orig_fieldnames = list(reader.fieldnames or [])
#         # Filter out completely empty rows
#         rows_in = [row for row in reader if any((v or "").strip() for v in row.values())]

#     # Ensure output columns
#     fieldnames = orig_fieldnames[:]
#     for extra in ["raw_response", "prediction"]:
#         if extra not in fieldnames:
#             fieldnames.append(extra)

#     # Decide output path: either user-specified or input + model suffix
#     if args.out_csv is not None:
#         out_path = Path(args.out_csv)
#     else:
#         # Derive a short tag for filenames (e.g. "Qwen3-4B-Thinking-2507" from "Qwen/Qwen3-4B-Thinking-2507")
#         safe_model_tag = args.model.split("/")[-1]
#         # Sanitize a couple of characters
#         for ch in [":", " "]:
#             safe_model_tag = safe_model_tag.replace(ch, "_")
#         if safe_model_tag not in in_path.stem:
#             out_path = in_path.with_name(f"{in_path.stem}_{safe_model_tag}{in_path.suffix}") 
#         else:
#             out_path = in_path

#     if out_path != in_path:
#         fieldnames = [fn for fn in fieldnames if fn != args.prompt_col]
#     else:
#         fieldnames = fieldnames

#     json_out_path = out_path.with_suffix(".jsonl")

#     max_to_process = args.max_rows if args.max_rows is not None else float("inf")

#     # --------- 2. Pre-pass: parse prediction from existing raw_response ---------
#     for row in rows_in:
#         raw = row.get("raw_response", "") or ""
#         pred = row.get("prediction", "") or ""
#         if raw.strip() and not pred.strip():
#             adj = extract_adjacency_matrix(raw)
#             if adj is not None:
#                 row["prediction"] = json.dumps(adj, ensure_ascii=False)

#     processed = 0
#     skipped = 0

#     # --------- 3. Provider-specific generation ---------

#     # HF provider: batch prompts to avoid "sequential" GPU warning
#     if provider == "hf" and not args.dry_run:
#         # Build HF pipeline once
#         try:
#             dm = None if not args.hf_device_map or args.hf_device_map == "none" else args.hf_device_map
#             hf_pipe = build_hf_pipeline(
#                 args.model,
#                 trust_remote_code=bool(args.hf_trust_remote_code),
#                 device_map=dm,
#                 torch_dtype=args.hf_dtype,
#             )
#         except Exception as e:
#             sys.exit(str(e))

#         # Optional: set generation_config.temperature instead of passing as flag
#         try:
#             gen_cfg = hf_pipe.model.generation_config
#             gen_cfg.temperature = float(args.temperature)
#         except Exception:
#             pass  # not all models expose this cleanly
#         hf_pipe.batch_size = max(1, int(args.hf_batch_size))
#         # Collect rows that actually need a model call
#         to_call_indices = []
#         to_call_prompts = []
#         for idx, row in enumerate(rows_in):
#             prompt = row.get(args.prompt_col, "") or ""
#             raw = row.get("raw_response", "") or ""
#             if not prompt:
#                 continue
#             need_call = (args.overwrite or not raw.strip()) and (len(to_call_indices) < max_to_process)
#             if need_call:
#                 to_call_indices.append(idx)
#                 to_call_prompts.append(prompt)

#         batch_size = max(1, int(args.hf_batch_size))
#         total_calls = len(to_call_indices)

#         with tqdm(total=total_calls, desc="HF batched generation", unit="prompt") as pbar:
#             for b_start in range(0, total_calls, batch_size):
#                 b_end = min(total_calls, b_start + batch_size)
#                 batch_prompts = to_call_prompts[b_start:b_end]
#                 batch_indices = to_call_indices[b_start:b_end]

#                 # Build kwargs without unsupported flags
#                 gen_kwargs = {
#                     "max_new_tokens": args.max_new_tokens,
#                     "return_full_text": False,
#                     "batch_size": batch_size,  
#                 }
#                 if args.temperature > 0.0:
#                     gen_kwargs["do_sample"] = True
#                     # (top_p/top_k can be added here if the model supports them)

#                 outputs = hf_pipe(batch_prompts, **gen_kwargs)

#                 # Normalize outputs to a list of dicts
#                 norm_outputs = []
#                 if isinstance(outputs, list):
#                     for item in outputs:
#                         if isinstance(item, list):
#                             norm_outputs.append(item[0] if item else {})
#                         else:
#                             norm_outputs.append(item)
#                 else:
#                     norm_outputs = [outputs]

#                 for j, out in enumerate(norm_outputs):
#                     row_idx = batch_indices[j]
#                     row = rows_in[row_idx]

#                     if isinstance(out, dict):
#                         text = out.get("generated_text", "")
#                     else:
#                         text = str(out)

#                     row["raw_response"] = text

#                     adj = extract_adjacency_matrix(text)
#                     if adj is not None:
#                         row["prediction"] = json.dumps(adj, ensure_ascii=False)
#                     else:
#                         row["prediction"] = ""

#                     processed += 1

#                 pbar.update(len(batch_prompts))

#         skipped = len(rows_in) - processed

#     # Gemini provider: row-by-row (API-based, no GPU efficiency warning)
#     elif provider == "gemini" and not args.dry_run:
#         with tqdm(total=len(rows_in), desc="Gemini rows", unit="row") as pbar:
#             for idx, row in enumerate(rows_in):
#                 prompt = row.get(args.prompt_col, "") or ""
#                 raw = row.get("raw_response", "") or ""

#                 if not prompt:
#                     skipped += 1
#                     pbar.update(1)
#                     continue

#                 need_call = (processed < max_to_process) and (args.overwrite or not raw.strip())
#                 if need_call:
#                     resp = call_gemini(
#                         args.model,
#                         prompt,
#                         temperature=args.temperature,
#                     )
#                     row["raw_response"] = resp
#                     adj = extract_adjacency_matrix(resp)
#                     if adj is not None:
#                         row["prediction"] = json.dumps(adj, ensure_ascii=False)
#                     else:
#                         row["prediction"] = ""
#                     processed += 1
#                 else:
#                     skipped += 1

#                 pbar.update(1)

#     # dry-run or no provider => we only parsed existing raw_response above

#     # --------- 4. Write all rows to output CSV ---------
#     with out_path.open("w", encoding="utf-8", newline="") as fout:
#         writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
#         writer.writeheader()
#         for row in rows_in:
#             writer.writerow(row)

#     # --------- 5. Also write JSONL next to the CSV ---------

#     with json_out_path.open("w", encoding="utf-8") as jf:
#         for row in rows_in:
#             jf.write(json.dumps(row, ensure_ascii=False) + "\n")


#     print(
#         f"Wrote CSV: '{out_path}'\n"
#         f"Wrote JSONL: '{json_out_path}'\n"
#         f"Columns='raw_response', 'prediction'. "
#         f"Processed={processed}, Skipped={skipped}"
#     )

#     print(
#         f"Wrote '{out_path}'. "
#         f"Columns='raw_response', 'prediction'. "
#         f"Processed={processed}, Skipped={skipped}"
#     )

def main():
    ap = argparse.ArgumentParser(
        description="Append LLM responses to CSV with raw_response and prediction columns (Gemini or Hugging Face)."
    )
    ap.add_argument(
        "--csv",
        default="./out/cancer/prompts_obs200_int3_shuf5_anon.csv",
        help="Path to the input CSV produced by your generator."
    )
    ap.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Model name. If it contains 'gemini', uses Google Gemini; otherwise loads from Hugging Face."
    )
    ap.add_argument(
        "--provider",
        choices=["auto", "gemini", "hf","openai"],
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
        default=None,
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
        default="prompt_path",
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
    ap.add_argument(
        "--hf-batch-size",
        type=int,
        default=4,
        help="Batch size for Hugging Face generation (number of prompts per forward pass)."
    )

    args = ap.parse_args()

    # Decide provider
    provider = args.provider
    if provider == "auto":
        if is_gemini_model(args.model):
            provider = "gemini"
        elif is_openai_model(args.model):
            provider = "openai"
        else:
            provider = "hf"

    in_path = Path(args.csv)
    if not in_path.exists():
        sys.exit(f"CSV not found: {in_path}")

    # Keep max_new_tokens optional: None/<=0 means no explicit cap passed to HF generate.
    if args.max_new_tokens is None:
        print(f"[info] Using model/default generation length (no explicit --max-new-tokens) for {in_path.name}")

    # --------- 1. Read input CSV once ---------
    with in_path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        orig_fieldnames = list(reader.fieldnames or [])
        # breakpoint()
        rows_in = [row for row in reader if any((v or "").strip() for v in row.values())]

    # Ensure output columns
    fieldnames = orig_fieldnames[:]
    for extra in ["raw_response", "prediction", "valid", "format_ok", "truncation_suspected", "prompt_tokens", "error_type"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    # ---- Reorder columns for writing ----
    # "indices" = any of these if present
    index_cols = [c for c in ("data_idx", "shuffle_idx") if c in fieldnames]

    # Priority block in this exact order, but only if present
    priority_cols = [c for c in ["prediction", "valid", "raw_response"]
                     if c in fieldnames]

    # Everything else that wasn't already included
    remaining_cols = [
        c for c in fieldnames
        if c not in index_cols and c not in priority_cols
    ]

    write_fieldnames = index_cols + priority_cols + remaining_cols


    # Decide output path: either user-specified or input + model suffix
    # Decide output path: either user-specified, or under responses/cancer
    if args.out_csv is not None:
        # If user passes an explicit path, respect it as-is
        out_path = Path(args.out_csv)
    else:
        # inputs:    prompts/experiment1/<dataset>/.../prompts_....csv
        # outputs:   responses/<dataset>/responses_...._MODEL.csv
        safe_model_tag = args.model.split("/")[-1]
        for ch in [":", " "]:
            safe_model_tag = safe_model_tag.replace(ch, "_")

        # responses_root: "responses"
        # subdir: dataset inferred from the input path (falls back to parent dir name)
        responses_root = Path("responses")
        subdir_name = _infer_dataset_name_from_csv_path(in_path) or in_path.parent.name
        responses_dir = responses_root / subdir_name
        responses_dir.mkdir(parents=True, exist_ok=True)

        # start from input filename, swap 'prompts' -> 'responses'
        base_name = in_path.name.replace("prompts", "responses")
        base_stem = Path(base_name).stem
        base_suffix = Path(base_name).suffix

        # Ensure response filenames are unique across prompt styles.
        # Some styles (e.g., legacy payload and cases) may share the same prompt CSV filename
        # like "prompts_obs100_int0_shuf1.csv" but live in different config folders.
        # If the config folder indicates payload but the filename doesn't, tag it to avoid
        # overwriting other styles' response files.
        cfg_name = in_path.parent.name.lower()
        stem_l = base_stem.lower()
        if cfg_name.startswith("payload_topk_") and "payload_topk" not in stem_l:
            base_stem = f"{base_stem}_payload_topk"
        elif cfg_name.startswith("payload_") and "payload" not in stem_l:
            base_stem = f"{base_stem}_payload"

        if safe_model_tag not in base_stem:
            base_stem = f"{base_stem}_{safe_model_tag}"

        out_path = responses_dir / f"{base_stem}{base_suffix}"


    # JSONL path next to CSV
    # json_out_path = out_path.with_suffix(".jsonl")

    # --------- 2. Determine resume vs overwrite ---------
    resume = out_path.exists() and not args.overwrite

    def _is_row_complete_and_valid(r: dict) -> bool:
        raw = (r.get("raw_response") or "").lstrip()
        pred = (r.get("prediction") or "").strip()
        if raw.startswith("[ERROR]"):
            return False
        valid_str = str(r.get("valid", "")).strip()
        try:
            valid = int(valid_str) == 1
        except Exception:
            # Backward-compatible fallback for older files without valid column.
            valid = bool(pred)
        return bool(raw.strip()) and bool(pred) and valid

    existing_rows = []
    if resume:
        with out_path.open("r", encoding="utf-8", newline="") as f_existing:
            r_existing = csv.DictReader(f_existing)
            for r in r_existing:
                existing_rows.append(r)

        # Find first row that should be retried:
        # [ERROR], invalid parse (valid!=1), or missing prediction/response.
        first_retry_idx = None
        for i, r in enumerate(existing_rows):
            if not _is_row_complete_and_valid(r):
                first_retry_idx = i
                break

        if first_retry_idx is None:
            already_done = len(existing_rows)
            print(f"[info] Resuming: {already_done} rows already present in {out_path.name}, no retry-needed rows.")
        else:
            already_done = first_retry_idx
            print(
                f"[info] Resuming from row {already_done} (first retry-needed row at index {first_retry_idx}) "
                f"in {out_path.name}"
            )
    else:
        already_done = 0
        existing_rows = []

    total_rows = len(rows_in)

    # Respect max_rows only for **new** rows
    max_new = args.max_rows if args.max_rows is not None else float("inf")

    # Check if there is *any* work to do (any new rows)
    all_rows_done = resume and (already_done >= total_rows)
    need_model_calls = (not args.dry_run) and (not all_rows_done)

    if all_rows_done:
        print("[info] All rows already processed in existing output. No new model calls will be made.")

    # Require API key only if we actually need to call Gemini
    if provider == "gemini" and need_model_calls:
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            sys.exit("Missing API key: set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini provider.")
    # Require API key only if we actually need to call OpenAI
    if provider == "openai" and need_model_calls:
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("Missing API key: set OPENAI_API_KEY for OpenAI provider.")

    # --------- 3. Build HF pipeline (if needed) ---------
    hf_pipe = None
    if provider == "hf" and need_model_calls:
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

        # Optional: set generation_config.temperature
        try:
            gen_cfg = hf_pipe.model.generation_config
            gen_cfg.temperature = float(args.temperature)
        except Exception:
            pass

    # --------- 4. Open output files in append-or-write mode ---------
    # CSV

    if resume:
        fout = out_path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(fout, fieldnames=write_fieldnames, extrasaction="ignore")
        writer.writeheader()

        # jf = json_out_path.open("w", encoding="utf-8")

        # If resuming, first write all good rows (before first [ERROR]) back out
        if resume and already_done > 0:
            for i in range(already_done):
                good_row = existing_rows[i]
                writer.writerow(good_row)
                # jf.write(json.dumps(good_row, ensure_ascii=False) + "\n")
        # header already there
    else:
        fout = out_path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(fout, fieldnames=write_fieldnames, extrasaction="ignore")
        writer.writeheader()


    # JSONL: append only when resuming; otherwise overwrite
    # if resume and json_out_path.exists():
    #     jf = json_out_path.open("a", encoding="utf-8")
    # else:
    #     jf = json_out_path.open("w", encoding="utf-8")

    processed_new = 0
    skipped_new = 0

    # HF batching buffers
    hf_batch_prompts: list[str] = []
    hf_batch_rows: list[dict] = []

    def flush_hf_batch(writer_obj):
        nonlocal processed_new
        if not hf_batch_prompts:
            return
        batch_size = len(hf_batch_prompts)
        outputs = call_hf_textgen_batch(
            hf_pipe,
            hf_batch_prompts,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            batch_size=batch_size,
        )
        for row_obj, resp in zip(hf_batch_rows, outputs):
            row_obj["raw_response"] = resp
            row_obj["format_ok"] = _format_ok(resp)
            out_tok = count_openai_tokens(args.model, resp)
            row_obj["truncation_suspected"] = _truncation_suspected(
                resp,
                output_tokens=out_tok,
                max_new_tokens_hint=int(args.max_new_tokens) if args.max_new_tokens is not None else None,
            )
            adj = _extract_adjacency_from_response(resp)
            if adj is not None:
                row_obj["prediction"] = json.dumps(adj.tolist(), ensure_ascii=False)
                row_obj["valid"] = 1
                row_obj["error_type"] = ""
            else:
                row_obj["prediction"] = ""
                row_obj["valid"] = 0
                row_obj["error_type"] = classify_error_type(resp)
            processed_new += 1
            writer_obj.writerow(row_obj)
            # jf_obj.write(json.dumps(row_obj, ensure_ascii=False) + "\n")
        hf_batch_prompts.clear()
        hf_batch_rows.clear()

    try:
        with fout, tqdm(total=total_rows, desc="Rows", unit="row") as pbar:
            for idx, row in enumerate(rows_in):
                pbar.update(1)

                # 1) Skip rows we've already fully written (before first [ERROR])
                #    Those rows were re-emitted from existing_rows[:already_done]
                if resume and idx < already_done:
                    continue

                # For rows >= already_done, start from the input CSV row
                current = row

                # prompt = current.get(args.prompt_col, "") or "
                prompt_path_str = current.get(args.prompt_col, "") or ""
                prompt = Path(prompt_path_str).read_text(encoding="utf-8")
                raw = current.get("raw_response", "") or ""
                pred = current.get("prediction", "") or ""
                valid_str = str(current.get("valid", "")).strip()
                try:
                    is_valid = int(valid_str) == 1
                except Exception:
                    is_valid = bool(pred.strip())

                # Treat [ERROR] responses as invalid and eligible for retry
                is_error_resp = raw.lstrip().startswith("[ERROR]")
                is_invalid_resp = (not pred.strip()) or (not is_valid)

                # 2) Respect max_rows: don't make new model calls past this,
                #    but still write the row as-is so the file stays consistent.
                if processed_new >= max_new:
                    writer.writerow(current)
                    # jf.write(json.dumps(current, ensure_ascii=False) + "\n")
                    skipped_new += 1
                    continue

                # 3) If we already have a non-error, valid response, reuse it.
                if raw.strip() and not is_error_resp and pred.strip() and is_valid:
                    writer.writerow(current)
                    # jf.write(json.dumps(current, ensure_ascii=False) + "\n")
                    skipped_new += 1
                    continue

                # 4) Decide if we need to call the model
                need_call = (
                    need_model_calls
                    and bool(prompt)
                    and (args.overwrite or not raw.strip() or is_error_resp or is_invalid_resp)
                )

                if need_call:
                    if provider == "gemini":
                        resp = call_gemini(
                            args.model,
                            prompt,
                            temperature=args.temperature,
                        )
                        current["raw_response"] = resp
                        current["format_ok"] = _format_ok(resp)
                        out_tok = count_openai_tokens(args.model, resp)
                        current["truncation_suspected"] = _truncation_suspected(
                            resp,
                            output_tokens=out_tok,
                            max_new_tokens_hint=int(args.max_new_tokens) if args.max_new_tokens is not None else None,
                        )
                        adj = _extract_adjacency_from_response(resp)
                        if adj is not None:
                            current["prediction"] = json.dumps(adj.tolist(), ensure_ascii=False)
                            current["valid"] = 1
                            current["error_type"] = ""
                        else:
                            current["prediction"] = ""
                            current["valid"] = 0
                            current["error_type"] = classify_error_type(resp)
                        processed_new += 1

                        writer.writerow(current)
                        # jf.write(json.dumps(current, ensure_ascii=False) + "\n")
                    elif provider == "openai":
                        n_tok = count_openai_tokens(args.model, prompt)
                        row["prompt_tokens"] = n_tok

                        # Optional: print to stderr for live debugging
                        if n_tok >= 0:
                            print(f"[debug] row {idx}: prompt_tokens={n_tok}", file=sys.stderr)

                        resp = call_openai(
                            model_name=args.model,
                            prompt=prompt,
                            temperature=args.temperature,
                            max_retries=0,          # or 1 if you want *one* retry
                            request_timeout=6000.0,   # tune as you like
                        )
                        row["raw_response"] = resp
                        row["format_ok"] = _format_ok(resp)
                        out_tok = count_openai_tokens(args.model, resp)
                        row["truncation_suspected"] = _truncation_suspected(
                            resp,
                            output_tokens=out_tok,
                            max_new_tokens_hint=int(args.max_new_tokens) if args.max_new_tokens is not None else None,
                        )
                        adj = _extract_adjacency_from_response(resp)
                        if adj is not None:
                            row["prediction"] = json.dumps(adj.tolist(), ensure_ascii=False)
                            row["valid"] = 1
                            row["error_type"] = ""
                        else:
                            row["prediction"] = ""
                            row["valid"] = 0
                            row["error_type"] = classify_error_type(resp)
                        processed_new += 1

                        writer.writerow(row)
                        # jf.write(json.dumps(row, ensure_ascii=False) + "\n")

                    elif provider == "hf":
                        # Buffer for batched HF generation
                        hf_batch_prompts.append(prompt)
                        hf_batch_rows.append(current)
                        if len(hf_batch_prompts) >= int(args.hf_batch_size):
                            flush_hf_batch(writer)

                    else:
                        current["raw_response"] = "[ERROR] Unknown provider"
                        current["prediction"] = ""
                        current["valid"] = 0
                        current["error_type"] = "unknown_provider"
                        processed_new += 1
                        writer.writerow(current)
                        # jf.write(json.dumps(current, ensure_ascii=False) + "\n")

                else:
                    # No new call; just write the row as-is
                    writer.writerow(current)
                    # jf.write(json.dumps(current, ensure_ascii=False) + "\n")
                    skipped_new += 1

                # Flush occasionally so crashes don't lose much
                if (idx + 1) % 10 == 0:
                    fout.flush()
                    # jf.flush()

                pbar.set_postfix(processed_new=processed_new, skipped_new=skipped_new)

            # After the loop, flush any remaining HF batch
            if provider == "hf" and need_model_calls:
                flush_hf_batch(writer)

    finally:
        try:
            fout.flush()
        except Exception:
            pass
        # try:
        #     jf.flush()
        # except Exception:
        #     pass


    print(
        f"\nDone.\n"
        f"- Input rows: {total_rows}\n"
        f"- Previously done (resumed): {already_done}\n"
        f"- Newly processed: {processed_new}\n"
        f"- Newly skipped (no call / max_rows): {skipped_new}\n"
        f"- CSV: {out_path}\n"
        # f"- JSONL: {json_out_path}"
    )


if __name__ == "__main__":
    main()
