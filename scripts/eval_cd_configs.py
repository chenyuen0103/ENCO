#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
for _path in (EXPERIMENTS_DIR, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    import torch
except Exception:
    torch = None

try:
    from cd_generation.names_only import iter_names_only_prompts_in_memory
except ImportError:
    from experiments.cd_generation.names_only import iter_names_only_prompts_in_memory
from query_api import (
    call_gemini,
    call_openai,
    call_hf_textgen_batch,
    build_hf_pipeline,
    is_gemini_model,
    is_openai_model,
    extract_adjacency_matrix,
    count_openai_tokens,
)
try:
    from query_vllm import build_vllm_engine, build_yarn_hf_overrides, call_vllm_textgen_batch
except ImportError:
    from experiments.query_vllm import build_vllm_engine, build_yarn_hf_overrides, call_vllm_textgen_batch

FORMAT_RE = re.compile(r"(?s)^\s*(?:<think>)?.*?</think>\s*<answer>.*?</answer>\s*$")
ANSWER_RE = re.compile(r"(?s)<answer>\s*(.*?)\s*</answer>")


def _call_openai_compatible_batch(
    *,
    base_url: str,
    model: str,
    prompts: list[str],
    temperature: float,
    max_new_tokens: int,
    timeout: float,
) -> list[str]:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        return [f"[ERROR] OpenAI SDK not available: {type(e).__name__}: {e}"] * len(prompts)

    try:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
            base_url=base_url.rstrip("/"),
            timeout=float(timeout),
            max_retries=0,
        )
        out: list[str] = []
        for prompt in prompts:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=float(temperature),
                    max_tokens=int(max_new_tokens),
                )
                out.append(resp.choices[0].message.content or "")
            except Exception as e:
                out.append(f"[ERROR] {type(e).__name__}: {e}")
        return out
    except Exception as e:
        return [f"[ERROR] {type(e).__name__}: {e}"] * len(prompts)


def _extract_answer_text(text: str) -> str:
    m = ANSWER_RE.search(text or "")
    return m.group(1) if m else (text or "")


def _extract_adjacency_from_response(text: str, *, fallback_variables: list[str] | None = None):
    answer_text = _extract_answer_text(text)
    mat = extract_adjacency_matrix(answer_text, fallback_variables=fallback_variables)
    if mat is not None:
        return mat
    if answer_text != (text or ""):
        return extract_adjacency_matrix(text, fallback_variables=fallback_variables)
    return None


def _expected_matrix_size(answer_obj: dict[str, Any]) -> int | None:
    variables = answer_obj.get("variables")
    if isinstance(variables, list) and variables:
        return len(variables)

    adj = answer_obj.get("adjacency_matrix")
    if isinstance(adj, list) and adj:
        return len(adj)
    return None


def _right_shape(adj: Any, expected_n: int | None) -> int:
    if adj is None or expected_n is None:
        return 0
    shape = getattr(adj, "shape", None)
    if shape is not None:
        return int(tuple(shape) == (expected_n, expected_n))
    if not isinstance(adj, list) or len(adj) != expected_n:
        return 0
    return int(all(isinstance(row, list) and len(row) == expected_n for row in adj))


def _format_ok(text: str) -> int:
    t = text or ""
    if FORMAT_RE.match(t):
        return 1
    # Also accept JSON-only contract used by prompt generators.
    s = t.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return 0
    try:
        obj = json.loads(s)
    except Exception:
        return 0
    return int(isinstance(obj, dict) and isinstance(obj.get("adjacency_matrix"), list))


def _classify_error_type(raw_response: str) -> str:
    text = (raw_response or "").strip()
    if not text.startswith("[ERROR]"):
        return ""
    low = text.lower()
    if "input too long" in low or "string_above_max_length" in low or "prompt too long" in low:
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


def _clear_cuda_cache() -> None:
    if torch is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _compose_prompt(prompt_text: str, extra_output_instruction: str) -> str:
    if extra_output_instruction and extra_output_instruction.strip():
        return (
            (prompt_text or "").rstrip()
            + "\n\n"
            + extra_output_instruction.strip()
            + "\n"
        )
    return prompt_text or ""


def _truncation_suspected(text: str, *, output_tokens: int, max_new_tokens_hint: int | None) -> int:
    t = text or ""
    missing_close_tags = (
        ("<think>" in t and "</think>" not in t)
        or ("<answer>" in t and "</answer>" not in t)
    )
    near_limit = False
    if max_new_tokens_hint is not None and max_new_tokens_hint > 0 and output_tokens >= 0:
        near_limit = output_tokens >= int(0.98 * max_new_tokens_hint)
    return int(missing_close_tags or near_limit)


def _safe_model_tag(model: str) -> str:
    model_path = Path(str(model))
    tag = model_path.name or str(model)
    if tag.startswith("checkpoint-") and model_path.parent.name:
        tag = f"{model_path.parent.name}_{tag}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_") or "model"


def _default_response_path(responses_root: Path, dataset: str, base_name: str, model: str) -> Path:
    responses_dir = Path(responses_root) / dataset
    responses_dir.mkdir(parents=True, exist_ok=True)

    base_name = base_name.replace("prompts", "responses", 1)
    base_stem = Path(base_name).stem
    base_suffix = Path(base_name).suffix or ".csv"

    safe_model_tag = _safe_model_tag(model)
    if safe_model_tag not in base_stem:
        base_stem = f"{base_stem}_{safe_model_tag}"
    return responses_dir / f"{base_stem}{base_suffix}"


def _default_example_prompt_path(dataset: str, base_name: str, example_dir: Path | None) -> Path:
    """
    One prompt per configuration, saved for debugging.
    Example-prompt filenames are config-based rather than model-based because the prompt text
    itself does not depend on the queried model.
    """
    # Default to experiments/prompts/<dataset>/example_prompts so prompts live next to the graph's prompt assets.
    out_dir = example_dir or (Path(__file__).parent / "prompts" / dataset / "example_prompts")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(base_name).stem
    return out_dir / f"{stem}_example_prompt.txt"


def _normalize_hist_mass_keep_frac(
    value: Any,
    *,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        frac = float(value)
    except Exception as e:
        raise SystemExit(f"{field_name} must be a float in (0, 1] or a percent in (0, 100].") from e
    if frac <= 0:
        raise SystemExit(f"{field_name} must be > 0.")
    if frac <= 1:
        return frac
    if frac <= 100:
        return frac / 100.0
    raise SystemExit(f"{field_name} must be <= 1.0 (fraction) or <= 100 (percent).")


def _default_vllm_error_log_path(dataset: str, base_name: str, model: str) -> Path:
    out_dir = Path(__file__).parent / "logs" / "vllm_errors" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model_tag = _safe_model_tag(model)
    stem = Path(base_name).stem
    return out_dir / f"{stem}_{safe_model_tag}_vllm_error.log"


def _append_vllm_error_log(
    *,
    log_path: Path,
    dataset: str,
    base_name: str,
    model: str,
    message: str,
    detail: str | None = None,
) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    lines = [
        f"[{timestamp}] dataset={dataset} base={base_name} model={model}",
        message.rstrip(),
    ]
    if detail and detail.strip():
        lines.extend(["", detail.rstrip()])
    lines.append("\n" + ("=" * 80) + "\n")
    with log_path.open("a", encoding="utf-8") as fout:
        fout.write("\n".join(lines))
    print(f"[vllm:error] saved details to {log_path}", file=sys.stderr, flush=True)


class _FdTee:
    """
    Tee process stdout/stderr to a log file while preserving terminal output.

    This is started once before vLLM engine initialization so worker-process logs
    inherited from this process are also captured.
    """

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self._saved_fds: dict[int, int] = {}
        self._threads: list[threading.Thread] = []
        self._log_lock = threading.Lock()
        self._log_file = None
        self._active = False

    def _pump(self, read_fd: int, target_fd: int) -> None:
        try:
            while True:
                chunk = os.read(read_fd, 65536)
                if not chunk:
                    break
                try:
                    os.write(target_fd, chunk)
                except OSError:
                    pass
                try:
                    with self._log_lock:
                        if self._log_file is not None:
                            self._log_file.write(chunk)
                            self._log_file.flush()
                except Exception:
                    pass
        finally:
            try:
                os.close(read_fd)
            except OSError:
                pass

    def start(self) -> None:
        if self._active:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Start each process invocation with a fresh vLLM log file.
        self._log_file = self.log_path.open("wb", buffering=0)
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        for fd in (1, 2):
            saved_fd = os.dup(fd)
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, fd)
            os.close(write_fd)
            thread = threading.Thread(
                target=self._pump,
                args=(read_fd, saved_fd),
                daemon=True,
            )
            thread.start()
            self._saved_fds[fd] = saved_fd
            self._threads.append(thread)
        self._active = True
        print(f"[vllm:log] teeing stdout/stderr to {self.log_path}", file=sys.stderr, flush=True)


def _maybe_write_example_prompt(
    *,
    dataset: str,
    base_name: str,
    prompt_row: dict[str, Any],
    example_dir: Path | None,
    overwrite: bool,
) -> Path:
    out_path = _default_example_prompt_path(dataset, base_name, example_dir)
    if out_path.exists() and not overwrite:
        return out_path

    payload = {
        "dataset": dataset,
        "base_name": base_name,
        "data_idx": prompt_row.get("data_idx"),
        "shuffle_idx": prompt_row.get("shuffle_idx"),
    }
    header = [
        "=== META ===",
        json.dumps(payload, ensure_ascii=False, indent=2),
        "",
        "=== PROMPT ===",
    ]
    out_path.write_text("\n".join(header) + "\n" + str(prompt_row.get("prompt_text", "")) + "\n", encoding="utf-8")
    return out_path


def _select_provider(model: str, provider: str) -> str:
    if provider and provider != "auto":
        return provider
    if is_gemini_model(model):
        return "gemini"
    if is_openai_model(model):
        return "openai"
    return "hf"


def _resolve_hf_merge_lora(model: str, cli_value: bool | None) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    model_path = Path(model)
    return model_path.exists() and (model_path / "adapter_config.json").exists()


def _dtype_nbytes(dtype_obj: Any, fallback_hf_dtype: str = "auto") -> int:
    if dtype_obj is not None:
        s = str(dtype_obj).lower()
    else:
        s = str(fallback_hf_dtype).lower()
    if "float64" in s or "fp64" in s:
        return 8
    if "float32" in s or "fp32" in s:
        return 4
    if "bfloat16" in s or "bf16" in s or "float16" in s or "fp16" in s or "half" in s:
        return 2
    if "int8" in s or "uint8" in s:
        return 1
    # Conservative default for inference.
    return 2


def _estimate_hf_kv_cache_bytes(
    hf_pipe: Any,
    *,
    total_tokens: int,
    fallback_hf_dtype: str = "auto",
) -> int | None:
    """
    Rough KV-cache estimate (bytes):
      2 (K,V) * layers * kv_heads * head_dim * total_tokens * bytes_per_elem
    """
    try:
        if hf_pipe is None:
            return None
        model = getattr(hf_pipe, "model", None)
        cfg = getattr(model, "config", None)
        if cfg is None:
            return None
        layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
        num_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
        hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
        if not layers or not num_heads or not hidden_size:
            return None
        kv_heads = getattr(cfg, "num_key_value_heads", None) or num_heads
        head_dim = int(hidden_size) // int(num_heads)
        if head_dim <= 0:
            return None
        bytes_per_elem = _dtype_nbytes(getattr(model, "dtype", None), fallback_hf_dtype=fallback_hf_dtype)
        elems = int(2 * int(layers) * int(kv_heads) * int(head_dim) * max(int(total_tokens), 0))
        return int(elems * bytes_per_elem)
    except Exception:
        return None


def _visible_vram_bytes() -> int | None:
    if torch is None or not torch.cuda.is_available():
        return None
    total = 0
    for i in range(torch.cuda.device_count()):
        try:
            total += int(torch.cuda.get_device_properties(i).total_memory)
        except Exception:
            continue
    return total if total > 0 else None


def _hf_prompt_context_limit_tokens(hf_pipe: Any) -> int | None:
    """
    Best-effort HF context length (tokens) from tokenizer/model config.
    Returns None when unavailable.
    """
    if hf_pipe is None:
        return None

    def _normalize_limit(value: Any) -> int | None:
        try:
            n = int(value)
        except Exception:
            return None
        # Ignore unset/sentinel values often used by tokenizers.
        if n <= 0 or n >= 1_000_000_000:
            return None
        return n

    tok = getattr(hf_pipe, "tokenizer", None)
    if tok is not None:
        for attr in ("model_max_length", "max_len_single_sentence"):
            lim = _normalize_limit(getattr(tok, attr, None))
            if lim is not None:
                return lim

    model = getattr(hf_pipe, "model", None)
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("max_position_embeddings", "n_positions", "seq_length", "max_seq_len"):
            lim = _normalize_limit(getattr(cfg, attr, None))
            if lim is not None:
                return lim

    return None


def _load_configs_from_file(
    *,
    config_file: Path,
    style_aliases: dict[str, str],
    allowed_styles: set[str],
    allowed_row_orders: set[str],
    allowed_col_orders: set[str],
) -> list[tuple[str, bool, int, int, str, str, int, str | None, bool, str, float | None]]:
    def _expand_config_product(raw_product: Any) -> list[dict[str, Any]]:
        if raw_product is None:
            return []
        if not isinstance(raw_product, dict) or not raw_product:
            raise SystemExit("'config_product' must be a non-empty object when present.")
        keys = list(raw_product.keys())
        value_lists: list[list[Any]] = []
        for key in keys:
            raw_vals = raw_product.get(key)
            if isinstance(raw_vals, list):
                vals = list(raw_vals)
            else:
                vals = [raw_vals]
            vals = [v for v in vals if v is not None]
            if not vals:
                raise SystemExit(f"'config_product.{key}' must contain at least one value.")
            value_lists.append(vals)
        return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]

    try:
        payload = json.loads(config_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to read --config-file {config_file}: {e}") from e

    if isinstance(payload, dict):
        raw_configs = payload.get("configs")
        raw_product = payload.get("config_product", payload.get("config_grid"))
        raw_defaults = payload.get("config_defaults", payload.get("defaults", {}))
    elif isinstance(payload, list):
        raw_configs = payload
        raw_product = None
        raw_defaults = {}
    else:
        raise SystemExit("--config-file must contain either a JSON list or an object with key 'configs'.")
    configs_from_product = _expand_config_product(raw_product)
    if raw_configs is None:
        raw_configs = []
    if not isinstance(raw_configs, list):
        raise SystemExit("'configs' must be a list when present.")
    raw_configs = list(raw_configs) + configs_from_product
    if not raw_configs:
        raise SystemExit("--config-file contains no configs.")
    if raw_defaults is None:
        raw_defaults = {}
    if not isinstance(raw_defaults, dict):
        raise SystemExit("'config_defaults' must be an object when present.")

    out: list[tuple[str, bool, int, int, str, str, int, str | None, bool, str, float | None]] = []
    for i, item in enumerate(raw_configs):
        if not isinstance(item, dict):
            raise SystemExit(f"Config #{i} must be an object, got: {type(item).__name__}")
        merged = dict(raw_defaults)
        merged.update(item)

        style_raw = str(merged.get("prompt_style", merged.get("style", "cases"))).strip().lower()
        style = style_aliases.get(style_raw, style_raw)
        if style not in allowed_styles:
            raise SystemExit(
                f"Config #{i}: unknown prompt_style '{style_raw}' (normalized '{style}'). "
                f"Allowed: {sorted(allowed_styles)}"
            )

        row_ord = str(merged.get("row_order", "random")).strip().lower()
        col_ord = str(merged.get("col_order", "original")).strip().lower()
        if row_ord not in allowed_row_orders:
            raise SystemExit(f"Config #{i}: invalid row_order '{row_ord}'. Allowed: {sorted(allowed_row_orders)}")
        if col_ord not in allowed_col_orders:
            raise SystemExit(f"Config #{i}: invalid col_order '{col_ord}'. Allowed: {sorted(allowed_col_orders)}")

        try:
            obs_n = int(merged.get("obs_per_prompt", merged.get("obs", 0)))
            int_n = int(merged.get("int_per_combo", merged.get("int", 0)))
            shuf_n = int(merged.get("shuffles_per_graph", merged.get("shuffle", merged.get("shuf", 1))))
        except Exception as e:
            raise SystemExit(f"Config #{i}: obs/int/shuf must be integers: {e}") from e

        if shuf_n <= 0:
            raise SystemExit(f"Config #{i}: shuffles_per_graph must be > 0.")

        anon_raw = merged.get("anonymize", False)
        if isinstance(anon_raw, bool):
            anon = anon_raw
        elif isinstance(anon_raw, str):
            anon = anon_raw.strip().lower() in {"1", "true", "yes", "y", "on"}
        elif isinstance(anon_raw, (int, float)):
            anon = bool(int(anon_raw))
        else:
            anon = False

        wrapper_mode_raw = merged.get("wrapper_mode", None)
        if wrapper_mode_raw is None:
            # Legacy config compatibility only: historical `cot_hint` files used this
            # field as a proxy for chat wrapping. It does not control staged reasoning.
            cot_hint_raw = merged.get("cot_hint", False)
            if isinstance(cot_hint_raw, bool):
                wrapper_mode = "chat" if cot_hint_raw else None
            elif isinstance(cot_hint_raw, str):
                wrapper_mode = "chat" if cot_hint_raw.strip().lower() in {"1", "true", "yes", "y", "on"} else None
            elif isinstance(cot_hint_raw, (int, float)):
                wrapper_mode = "chat" if bool(int(cot_hint_raw)) else None
            else:
                wrapper_mode = None
        else:
            wrapper_mode = str(wrapper_mode_raw).strip().lower() or None
            if wrapper_mode == "none":
                wrapper_mode = None
            if wrapper_mode not in {None, "plain", "chat"}:
                raise SystemExit(
                    f"Config #{i}: invalid wrapper_mode '{wrapper_mode_raw}'. Allowed: ['plain', 'chat']."
                )

        append_format_hint_raw = merged.get("append_format_hint", False)
        if isinstance(append_format_hint_raw, bool):
            append_format_hint = append_format_hint_raw
        elif isinstance(append_format_hint_raw, str):
            append_format_hint = append_format_hint_raw.strip().lower() in {"1", "true", "yes", "y", "on"}
        elif isinstance(append_format_hint_raw, (int, float)):
            append_format_hint = bool(int(append_format_hint_raw))
        else:
            append_format_hint = False

        reasoning_guidance = str(merged.get("reasoning_guidance", "staged") or "staged").strip().lower()
        if reasoning_guidance not in {"staged", "concise", "none"}:
            raise SystemExit(
                f"Config #{i}: invalid reasoning_guidance '{merged.get('reasoning_guidance')}'. "
                "Allowed: ['staged', 'concise', 'none']."
            )
        hist_mass_keep_frac = _normalize_hist_mass_keep_frac(
            merged.get("hist_mass_keep_frac"),
            field_name=f"Config #{i}: hist_mass_keep_frac",
        )

        out.append(
            (
                style,
                anon,
                obs_n,
                int_n,
                row_ord,
                col_ord,
                shuf_n,
                wrapper_mode,
                append_format_hint,
                reasoning_guidance,
                hist_mass_keep_frac,
            )
        )

    return out


def _load_completed(out_path: Path, overwrite: bool) -> tuple[set[tuple[int, int]], list[dict[str, Any]]]:
    if overwrite or not out_path.exists():
        return set(), []
    completed: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    try:
        csv.field_size_limit(10_000_000)
    except OverflowError:
        csv.field_size_limit(1_000_000)
    with out_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("raw_response") or "").lstrip()
            is_error = raw.startswith("[ERROR]")
            if is_error:
                # Rerun only explicit error rows.
                continue
            try:
                key = (int(row.get("data_idx", -1)), int(row.get("shuffle_idx", -1)))
                completed.add(key)
                rows.append(row)
            except Exception:
                # Keep malformed-key rows in the output file, but do not mark as completed.
                rows.append(row)
    return completed, rows


def _import_iter_prompts_in_memory():
    try:
        from generate_prompts import iter_prompts_in_memory
        return iter_prompts_in_memory
    except Exception:
        try:
            from experiments.generate_prompts import iter_prompts_in_memory
            return iter_prompts_in_memory
        except Exception as e:
            raise SystemExit(
                "Failed to import generate_prompts.iter_prompts_in_memory. "
                "Install the prompt-generation dependencies or run with --only-names-only."
            ) from e


def _iter_prompts_for_config(
    *,
    bif_file: str,
    num_prompts: int,
    shuffles_per_graph: int,
    seed: int,
    prompt_style: str,
    obs_per_prompt: int,
    int_per_combo: int,
    row_order: str,
    col_order: str,
    anonymize: bool,
    causal_rules: bool,
    give_steps: bool,
    def_int: bool,
    intervene_vars: str,
    wrapper_mode: str | None,
    append_format_hint: bool,
    reasoning_guidance: str,
    hist_mass_keep_frac: float | None,
) -> tuple[str, dict[str, Any], Iterator[dict[str, Any]]]:
    is_names_only = (obs_per_prompt == 0 and int_per_combo == 0)
    if is_names_only:
        return iter_names_only_prompts_in_memory(
            bif_file=bif_file,
            num_prompts=num_prompts,
            seed=seed,
            col_order=col_order,
            anonymize=anonymize,
            causal_rules=causal_rules,
            wrapper_mode=wrapper_mode,
            append_format_hint=append_format_hint,
            reasoning_guidance=reasoning_guidance,
        )
    iter_prompts_in_memory = _import_iter_prompts_in_memory()
    return iter_prompts_in_memory(
        bif_file=bif_file,
        num_prompts=num_prompts,
        shuffles_per_graph=shuffles_per_graph,
        seed=seed,
        prompt_style=prompt_style,
        obs_per_prompt=obs_per_prompt,
        int_per_combo=int_per_combo,
        row_order=row_order,
        col_order=col_order,
        anonymize=anonymize,
        causal_rules=causal_rules,
        give_steps=give_steps,
        def_int=def_int,
        intervene_vars=intervene_vars,
        wrapper_mode=wrapper_mode,
        append_format_hint=append_format_hint,
        reasoning_guidance=reasoning_guidance,
        hist_mass_keep_frac=hist_mass_keep_frac,
    )


def _run_model_for_config(
    *,
    dataset: str,
    base_name: str,
    answer_obj: dict[str, Any],
    prompt_iter: Iterator[dict[str, Any]],
    responses_root: Path,
    model: str,
    provider: str,
    temperature: float,
    overwrite: bool,
    request_timeout_s: float,
    progress_every: int,
    log_calls: bool,
    hf_pipe: Any = None,
    hf_max_new_tokens: int = 0,
    hf_batch_size: int = 1,
    hf_context_limit: int = 0,
    hf_skip_if_prompt_tokens_over: int = 0,
    hf_skip_if_est_kv_exceeds_vram: bool = False,
    hf_kv_vram_fraction: float = 0.85,
    hf_dtype_for_estimate: str = "auto",
    vllm_engine: Any = None,
    vllm_max_new_tokens: int = 0,
    vllm_batch_size: int = 1,
    vllm_max_model_len: int = 0,
    vllm_server_base_url: str = "http://127.0.0.1:8000/v1",
    extra_output_instruction: str = "",
) -> None:
    out_path = _default_response_path(responses_root, dataset, base_name + ".csv", model)
    completed, existing_rows = _load_completed(out_path, overwrite)

    fieldnames = [
        "data_idx",
        "shuffle_idx",
        "answer",
        "given_edges",
        "raw_response",
        "prediction",
        "valid",
        "format_ok",
        "right_shape",
        "truncation_suspected",
        "prompt_tokens",
        "output_tokens",
        "total_tokens",
        "error_type",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)

        wrote = 0
        skipped = 0
        print(
            f"[run] dataset={dataset} base={base_name} model={model} provider={provider} "
            f"resume_completed={len(completed)} out={out_path}",
            file=sys.stderr,
            flush=True,
        )
        variables_for_parse = answer_obj.get("variables")
        if not isinstance(variables_for_parse, list):
            variables_for_parse = None
        expected_n = _expected_matrix_size(answer_obj)

        local_pending: list[dict[str, Any]] = []
        visible_vram_bytes = (
            _visible_vram_bytes()
            if provider == "hf" and bool(hf_skip_if_est_kv_exceeds_vram)
            else None
        )
        if provider == "hf":
            hf_prompt_context_limit = int(hf_context_limit) if int(hf_context_limit) > 0 else _hf_prompt_context_limit_tokens(hf_pipe)
        else:
            hf_prompt_context_limit = None
        vllm_prompt_context_limit = (
            int(vllm_max_model_len)
            if provider in {"vllm", "vllm_server"} and int(vllm_max_model_len) > 0
            else None
        )
        if provider == "hf" and bool(hf_skip_if_est_kv_exceeds_vram):
            print(
                f"[preflight] hf_visible_vram_bytes={visible_vram_bytes} "
                f"hf_kv_vram_fraction={float(hf_kv_vram_fraction):.2f}",
                file=sys.stderr,
                flush=True,
            )

        def _write_out_row(row: dict[str, Any], key: tuple[int, int], prompt_tokens: int, resp: str) -> None:
            nonlocal wrote
            output_tokens = count_openai_tokens(model, resp)
            total_tokens = (
                (prompt_tokens + output_tokens)
                if (prompt_tokens is not None and output_tokens is not None and prompt_tokens >= 0 and output_tokens >= 0)
                else -1
            )
            if log_calls:
                print(
                    f"[tokens] key={key} prompt={prompt_tokens} output={output_tokens} total={total_tokens}",
                    file=sys.stderr,
                    flush=True,
                )

            answer_text = _extract_answer_text(resp)
            adj = _extract_adjacency_from_response(resp, fallback_variables=variables_for_parse)
            pred = json.dumps(adj.tolist(), ensure_ascii=False) if adj is not None else ""
            valid = 1 if adj is not None else 0
            format_ok = _format_ok(resp)
            right_shape = _right_shape(adj, expected_n)
            error_type = "" if valid else _classify_error_type(resp)
            truncation_suspected = _truncation_suspected(
                resp,
                output_tokens=output_tokens,
                max_new_tokens_hint=(
                    int(hf_max_new_tokens)
                    if provider == "hf" and int(hf_max_new_tokens) > 0
                    else int(vllm_max_new_tokens)
                    if provider in {"vllm", "vllm_server"} and int(vllm_max_new_tokens) > 0
                    else None
                ),
            )

            out_row = {
                "data_idx": row["data_idx"],
                "shuffle_idx": row["shuffle_idx"],
                "answer": json.dumps(answer_obj, ensure_ascii=False),
                "given_edges": row.get("given_edges"),
                "raw_response": resp,
                "prediction": pred,
                "valid": valid,
                "format_ok": format_ok,
                "right_shape": right_shape,
                "truncation_suspected": truncation_suspected,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "error_type": error_type,
            }
            writer.writerow(out_row)
            wrote += 1
            if progress_every > 0 and (wrote % progress_every == 0):
                print(
                    f"[progress] wrote={wrote} skipped={skipped} last_key={key} valid={valid} out={out_path.name}",
                    file=sys.stderr,
                    flush=True,
                )

        def _flush_local_batch() -> None:
            nonlocal local_pending
            if not local_pending:
                return
            prompts = [x["prompt"] for x in local_pending]
            keys = [x["key"] for x in local_pending]
            t0 = time.monotonic()
            if provider == "hf":
                if hf_pipe is None:
                    raise SystemExit("HF provider requested but no pipeline is initialized.")
                if log_calls:
                    print(
                        f"[call:start] provider=hf model={model} batch_size={len(prompts)} keys={keys}",
                        file=sys.stderr,
                        flush=True,
                    )
                responses = call_hf_textgen_batch(
                    hf_pipe,
                    prompts,
                    temperature=temperature,
                    max_new_tokens=int(hf_max_new_tokens),
                    batch_size=max(1, int(hf_batch_size)),
                )
            elif provider == "vllm":
                if vllm_engine is None:
                    raise SystemExit("vLLM provider requested but no engine is initialized.")
                if log_calls:
                    print(
                        f"[call:start] provider=vllm model={model} batch_size={len(prompts)} keys={keys}",
                        file=sys.stderr,
                        flush=True,
                    )
                responses = call_vllm_textgen_batch(
                    vllm_engine,
                    prompts,
                    temperature=temperature,
                    max_new_tokens=int(vllm_max_new_tokens),
                )
            elif provider == "vllm_server":
                if log_calls:
                    print(
                        f"[call:start] provider=vllm_server model={model} "
                        f"base_url={vllm_server_base_url} batch_size={len(prompts)} keys={keys}",
                        file=sys.stderr,
                        flush=True,
                    )
                responses = _call_openai_compatible_batch(
                    base_url=vllm_server_base_url,
                    model=model,
                    prompts=prompts,
                    temperature=temperature,
                    max_new_tokens=int(vllm_max_new_tokens),
                    timeout=float(request_timeout_s),
                )
            else:
                raise SystemExit(f"Batch flushing is not supported for provider: {provider}")

            if log_calls:
                dt = time.monotonic() - t0
                print(
                    f"[call:done] provider={provider} model={model} batch_size={len(prompts)} seconds={dt:.1f}",
                    file=sys.stderr,
                    flush=True,
                )

            if not isinstance(responses, list):
                responses = [f"[ERROR] Invalid {provider} batch response type: {type(responses).__name__}"] * len(local_pending)
            if len(responses) != len(local_pending):
                fixed = list(responses[: len(local_pending)])
                while len(fixed) < len(local_pending):
                    fixed.append(f"[ERROR] Missing {provider} batch response.")
                responses = fixed

            for item, resp in zip(local_pending, responses):
                _write_out_row(
                    row=item["row"],
                    key=item["key"],
                    prompt_tokens=item["prompt_tokens"],
                    resp=str(resp),
                )
            local_pending = []

        for row in prompt_iter:
            key = (int(row["data_idx"]), int(row["shuffle_idx"]))
            if key in completed:
                skipped += 1
                continue

            prompt = _compose_prompt(row["prompt_text"], extra_output_instruction)
            prompt_tokens = count_openai_tokens(model, prompt)
            print(
                f"[prompt_tokens] key={key} prompt_tokens={prompt_tokens} model={model}",
                file=sys.stderr,
                flush=True,
            )
            if (
                provider == "hf"
                and hf_prompt_context_limit is not None
                and prompt_tokens >= 0
                and prompt_tokens > int(hf_prompt_context_limit)
            ):
                resp = f"[ERROR] Prompt too long (num tokens:{prompt_tokens})"
                if log_calls:
                    print(
                        f"[skip] key={key} provider=hf reason=prompt_tokens_exceed_context "
                        f"prompt_tokens={prompt_tokens} context_limit={hf_prompt_context_limit}",
                        file=sys.stderr,
                        flush=True,
                    )
                _write_out_row(row=row, key=key, prompt_tokens=prompt_tokens, resp=resp)
                continue

            if (
                provider in {"vllm", "vllm_server"}
                and vllm_prompt_context_limit is not None
                and prompt_tokens >= 0
                and (prompt_tokens + max(int(vllm_max_new_tokens), 0)) > int(vllm_prompt_context_limit)
            ):
                resp = (
                    "[ERROR] Prompt plus generation budget exceeds vLLM max_model_len "
                    f"(prompt_tokens={prompt_tokens}, max_new_tokens={int(vllm_max_new_tokens)}, "
                    f"max_model_len={vllm_prompt_context_limit})"
                )
                if log_calls:
                    print(
                        f"[skip] key={key} provider=vllm reason=prompt_plus_output_exceeds_context "
                        f"prompt_tokens={prompt_tokens} max_new_tokens={int(vllm_max_new_tokens)} "
                        f"max_model_len={vllm_prompt_context_limit}",
                        file=sys.stderr,
                        flush=True,
                    )
                _write_out_row(row=row, key=key, prompt_tokens=prompt_tokens, resp=resp)
                continue

            if provider == "hf":
                skip_reason = None
                if int(hf_skip_if_prompt_tokens_over) > 0 and prompt_tokens >= 0 and prompt_tokens > int(hf_skip_if_prompt_tokens_over):
                    skip_reason = (
                        f"prompt_tokens={prompt_tokens} exceeds "
                        f"hf_skip_if_prompt_tokens_over={int(hf_skip_if_prompt_tokens_over)}"
                    )
                elif bool(hf_skip_if_est_kv_exceeds_vram) and prompt_tokens >= 0:
                    est_total_tokens = int(prompt_tokens) + max(int(hf_max_new_tokens), 0)
                    est_kv_bytes = _estimate_hf_kv_cache_bytes(
                        hf_pipe,
                        total_tokens=est_total_tokens,
                        fallback_hf_dtype=str(hf_dtype_for_estimate),
                    )
                    if est_kv_bytes is not None and visible_vram_bytes is not None:
                        frac = max(0.05, min(float(hf_kv_vram_fraction), 1.0))
                        budget_bytes = int(visible_vram_bytes * frac)
                        if est_kv_bytes > budget_bytes:
                            skip_reason = (
                                f"estimated_kv_cache_bytes={est_kv_bytes} exceeds "
                                f"budget_bytes={budget_bytes} (visible_vram_bytes={visible_vram_bytes}, "
                                f"hf_kv_vram_fraction={frac})"
                            )

                if skip_reason is not None:
                    resp = f"[ERROR] Skipped HF preflight: {skip_reason}"
                    if log_calls:
                        print(
                            f"[skip] key={key} provider=hf reason={skip_reason}",
                            file=sys.stderr,
                            flush=True,
                        )
                    _write_out_row(row=row, key=key, prompt_tokens=prompt_tokens, resp=resp)
                    continue

            if provider == "openai":
                t0 = time.monotonic()
                if log_calls:
                    print(
                        f"[call:start] key={key} prompt_tokens={prompt_tokens} model={model}",
                        file=sys.stderr,
                        flush=True,
                    )
                resp = call_openai(
                    model_name=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_retries=0,
                    request_timeout=float(request_timeout_s),
                )
                if log_calls:
                    dt = time.monotonic() - t0
                    print(f"[call:done] key={key} seconds={dt:.1f}", file=sys.stderr, flush=True)
            elif provider == "gemini":
                t0 = time.monotonic()
                if log_calls:
                    print(f"[call:start] key={key} model={model}", file=sys.stderr, flush=True)
                resp = call_gemini(
                    model_name=model,
                    prompt=prompt,
                    temperature=temperature,
                )
                if log_calls:
                    dt = time.monotonic() - t0
                    print(f"[call:done] key={key} seconds={dt:.1f}", file=sys.stderr, flush=True)
            elif provider == "hf":
                local_pending.append(
                    {
                        "row": row,
                        "key": key,
                        "prompt": prompt,
                        "prompt_tokens": prompt_tokens,
                    }
                )
                if len(local_pending) >= max(1, int(hf_batch_size)):
                    _flush_local_batch()
                continue
            elif provider == "vllm":
                local_pending.append(
                    {
                        "row": row,
                        "key": key,
                        "prompt": prompt,
                        "prompt_tokens": prompt_tokens,
                    }
                )
                if len(local_pending) >= max(1, int(vllm_batch_size)):
                    _flush_local_batch()
                continue
            elif provider == "vllm_server":
                local_pending.append(
                    {
                        "row": row,
                        "key": key,
                        "prompt": prompt,
                        "prompt_tokens": prompt_tokens,
                    }
                )
                if len(local_pending) >= max(1, int(vllm_batch_size)):
                    _flush_local_batch()
                continue
            else:
                raise SystemExit(f"Unknown provider: {provider}")

            _write_out_row(row=row, key=key, prompt_tokens=prompt_tokens, resp=resp)

        if provider in {"hf", "vllm", "vllm_server"}:
            _flush_local_batch()

    print(
        f"[summary] wrote={wrote} skipped={skipped} kept_existing={len(existing_rows)} out={out_path.name}",
        file=sys.stderr,
        flush=True,
    )
    print(f"[info] Wrote responses: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Experiment 1 prompts in-memory (no prompt files)."
    )
    ap.add_argument("--bif-file", required=True)
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    # default=[] to avoid duplication when this script is called by run_cd_eval_pipeline.py
    ap.add_argument("--shuffles-per-graph", type=int, action="append", default=[])
    ap.add_argument(
        "--responses-root",
        type=Path,
        default=(Path(__file__).resolve().parent / "responses"),
        help=(
            "Root directory to write response CSVs under. "
            "Default: experiments/responses (relative to repo root)."
        ),
    )

    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--provider", default="auto", choices=["auto", "gemini", "openai", "hf", "vllm", "vllm_server"])
    ap.add_argument(
        "--hf-max-new-tokens",
        type=int,
        default=8192,
        help="HF generation max_new_tokens. Set <=0 for no explicit cap (default: 0).",
    )
    ap.add_argument(
        "--hf-dtype",
        default="auto",
        help="HF torch dtype for model load (auto, bf16, fp16, fp32).",
    )
    ap.add_argument(
        "--hf-device-map",
        default="auto",
        help='HF device_map for model load (e.g. "auto", "cuda:0", "none").',
    )
    ap.add_argument(
        "--hf-merge-lora",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "For PEFT/LoRA adapter checkpoints, merge the adapter into the base model once at load time. "
            "Default: auto-enable for local adapter checkpoints with adapter_config.json; use "
            "--no-hf-merge-lora to keep runtime LoRA layers."
        ),
    )
    ap.add_argument(
        "--hf-batch-size",
        type=int,
        default=1,
        help="Batch size for HF text generation calls (default: 1).",
    )
    ap.add_argument(
        "--hf-context-limit",
        type=int,
        default=0,
        help=(
            "Override the auto-detected HF context window (tokens). "
            "If >0, use this value instead of reading model_max_length / max_position_embeddings from the model config."
        ),
    )
    ap.add_argument(
        "--hf-skip-if-prompt-tokens-over",
        type=int,
        default=0,
        help="If >0, skip HF rows whose prompt token count exceeds this threshold.",
    )
    ap.add_argument(
        "--hf-skip-if-est-kv-exceeds-vram",
        action="store_true",
        help="Skip HF rows when estimated KV-cache memory exceeds visible VRAM budget.",
    )
    ap.add_argument(
        "--hf-kv-vram-fraction",
        type=float,
        default=0.85,
        help="VRAM budget fraction for KV estimate when --hf-skip-if-est-kv-exceeds-vram is enabled.",
    )
    ap.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (number of GPUs).",
    )
    ap.add_argument(
        "--vllm-dtype",
        default="auto",
        help="vLLM dtype string (auto, float16/fp16, bfloat16/bf16, float32/fp32).",
    )
    ap.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=0,
        help="Optional vLLM max_model_len. Set >0 to enforce a total context budget.",
    )
    ap.add_argument(
        "--vllm-gpu-mem-util",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization (0.0-1.0).",
    )
    ap.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="Pass enforce_eager=True to vLLM (useful for debugging).",
    )
    ap.add_argument(
        "--vllm-enable-yarn",
        action="store_true",
        help="Enable YaRN RoPE scaling through vLLM hf_overrides for long-context Qwen models.",
    )
    ap.add_argument(
        "--vllm-yarn-factor",
        type=float,
        default=4.0,
        help="YaRN scaling factor for --vllm-enable-yarn. Qwen2.5 long-context docs use 4.0.",
    )
    ap.add_argument(
        "--vllm-yarn-original-max-position-embeddings",
        type=int,
        default=32768,
        help="Original context length for YaRN. Qwen2.5 long-context docs use 32768.",
    )
    ap.add_argument(
        "--vllm-max-new-tokens",
        type=int,
        default=0,
        help="vLLM generation max_new_tokens. If <=0, reuse --hf-max-new-tokens.",
    )
    ap.add_argument(
        "--vllm-batch-size",
        type=int,
        default=0,
        help="Batch size for vLLM generation calls. If <=0, reuse --hf-batch-size.",
    )
    ap.add_argument(
        "--vllm-error-log-file",
        type=Path,
        default=None,
        help=(
            "Optional path to append vLLM startup/runtime errors. "
            "Default: experiments/logs/vllm_errors/<dataset>/<base>_<model>_vllm_error.log"
        ),
    )
    ap.add_argument(
        "--vllm-server-base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible vLLM server base URL for --provider vllm_server.",
    )
    ap.add_argument(
        "--extra-output-instruction",
        default="",
        help=(
            "Extra instruction appended to every prompt before querying. "
            "Use empty string to disable."
        ),
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--dry-run-tokens",
        action="store_true",
        help="Compute and print prompt token lengths for each row/model without querying any model API.",
    )
    ap.add_argument("--only-names-only", action="store_true")
    ap.add_argument(
        "--request-timeout-s",
        type=float,
        default=6000.0,
        help="Per-request timeout for OpenAI calls (seconds).",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print a progress line every N new rows written (0 disables).",
    )
    ap.add_argument(
        "--log-calls",
        action="store_true",
        help="Print a line before/after each model API call (useful to see if you are stuck on a request).",
    )
    ap.add_argument(
        "--styles",
        nargs="*",
        default=None,
        help=(
            'Optional subset of prompt styles to run (any of: "cases", "matrix", "summary", '
            '"payload", "payload_topk").'
        ),
    )
    ap.add_argument(
        "--save-example-prompt",
        action="store_true",
        help="For each configuration, write ONE example prompt (first row) to disk for debugging.",
    )
    ap.add_argument(
        "--no-save-example-prompt",
        dest="save_example_prompt",
        action="store_false",
        help="Disable writing one example prompt per configuration.",
    )
    ap.set_defaults(save_example_prompt=True)
    ap.add_argument(
        "--example-prompt-dir",
        type=Path,
        default=None,
        help='Output directory for example prompts (default: experiments/prompts/<dataset>/example_prompts).',
    )
    ap.add_argument(
        "--overwrite-example-prompt",
        action="store_true",
        help="Overwrite existing example prompt files (if --save-example-prompt).",
    )

    ap.add_argument("--def-int", action="store_true")
    ap.add_argument("--intervene-vars", default="all")
    ap.add_argument("--causal-rules", action="store_true")
    ap.add_argument("--give-steps", action="store_true")
    ap.add_argument(
        "--wrapper-mode",
        choices=["plain", "chat"],
        default=None,
        help="Prompt transport: plain text or system/user/assistant chat wrapper.",
    )
    ap.add_argument(
        "--append-format-hint",
        action="store_true",
        help=(
            "Append the canonical Formatting requirement line. For causal discovery this "
            "adds the optional stage-by-stage reasoning instructions."
        ),
    )
    ap.add_argument(
        "--cot-hint",
        action="store_true",
        help=(
            "Legacy alias for chat-style prompt wrapping. This maps to wrapper_mode=chat "
            "and does not change the staged reasoning instructions."
        ),
    )
    ap.add_argument("--single-config", action="store_true")
    ap.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help=(
            "JSON file listing explicit configs to run. "
            "Format: either a list of config objects or {\"configs\": [...]}."
        ),
    )
    ap.add_argument(
        "--print-one-prompt",
        action="store_true",
        help="Generate and print a single prompt (first row) for debugging, then exit.",
    )
    ap.add_argument(
        "--prompt-style",
        choices=["cases", "matrix", "summary", "payload", "payload_topk"],
        default="cases",
    )
    ap.add_argument("--obs-per-prompt", type=int, default=0)
    ap.add_argument("--int-per-combo", type=int, default=0)
    ap.add_argument(
        "--hist-mass-keep-frac",
        type=float,
        default=None,
        help=(
            "For summary prompts, keep only the most frequent histogram assignments per regime until the listed "
            "entries cover this much empirical mass. Accepts a fraction in (0,1] or a percent in (0,100]. "
            "Default: disabled (no cutoff)."
        ),
    )
    ap.add_argument("--row-order", choices=["random", "sorted", "reverse"], default="random")
    ap.add_argument("--col-order", choices=["original", "reverse", "random", "topo", "reverse_topo"], default="original")
    ap.add_argument("--anonymize", action="store_true")

    args = ap.parse_args()
    if not args.model:
        args.model = ["gpt-5-mini"]
    args.hist_mass_keep_frac = _normalize_hist_mass_keep_frac(
        args.hist_mass_keep_frac,
        field_name="--hist-mass-keep-frac",
    )

    dataset = Path(args.bif_file).stem
    responses_root = Path(args.responses_root)
    shuf_values = [int(x) for x in (args.shuffles_per_graph or [1])]

    style_aliases = {
        "summary_join": "summary",
        "summary_joint": "summary",
    }

    all_styles = ["cases", "matrix", "summary", "payload", "payload_topk"]
    styles = list(all_styles)
    allowed_row_orders = {"random", "sorted", "reverse"}
    allowed_col_orders = {"original", "reverse", "random", "topo", "reverse_topo"}
    if args.styles:
        requested_raw = [s.strip().lower() for s in args.styles if s.strip()]
        requested = [style_aliases.get(s, s) for s in requested_raw]
        unknown = [s for s in requested if s not in set(styles)]
        if unknown:
            raise SystemExit(f"Unknown --styles: {unknown}. Allowed: {styles}")
        styles = requested

    if args.prompt_style in {"summary_join", "summary_joint"}:
        args.prompt_style = "summary"
    cli_wrapper_mode = args.wrapper_mode or ("chat" if args.cot_hint else None)
    anonymize_opts = [False, True]
    obs_sizes = [0, 100, 1000, 5000, 8000]
    int_sizes = [0, 50, 100, 200]
    row_order_opts = ["random", "sorted", "reverse"]
    # For now, only use the default column order.
    col_order_opts = ["original"]

    count = 0
    print(f"--- Starting Experiment 1 In-Memory Run ---")
    print(f"BIF File: {args.bif_file}")

    if args.single_config and args.config_file is not None:
        raise SystemExit("Use either --single-config or --config-file, not both.")

    if args.single_config:
        configs = [(
            args.prompt_style,
            bool(args.anonymize),
            int(args.obs_per_prompt),
            int(args.int_per_combo),
            args.row_order,
            args.col_order,
            int(shuf_values[0] if shuf_values else 1),
            cli_wrapper_mode,
            bool(args.append_format_hint),
            "staged",
            args.hist_mass_keep_frac,
        )]
    elif args.config_file is not None:
        configs = _load_configs_from_file(
            config_file=args.config_file,
            style_aliases=style_aliases,
            allowed_styles=set(all_styles),
            allowed_row_orders=allowed_row_orders,
            allowed_col_orders=allowed_col_orders,
        )
    else:
        configs = [
            (
                style,
                anon,
                obs_n,
                int_n,
                row_ord,
                col_ord,
                shuf_n,
                cli_wrapper_mode,
                bool(args.append_format_hint),
                "staged",
                args.hist_mass_keep_frac,
            )
            for style in styles
            for anon in anonymize_opts
            for obs_n in obs_sizes
            for int_n in int_sizes
            for row_ord in row_order_opts
            for col_ord in col_order_opts
            for shuf_n in shuf_values
        ]

    hf_pipe_cache: dict[tuple[str, str | None, str, bool], Any] = {}
    vllm_engine_cache: dict[tuple[str, int, str, int, float, bool, str], Any] = {}
    vllm_log_tee: _FdTee | None = None
    using_explicit_config_file = args.config_file is not None

    for (
        style,
        anon,
        obs_n,
        int_n,
        row_ord,
        col_ord,
        shuf_n,
        wrapper_mode,
        append_format_hint,
        reasoning_guidance,
        hist_mass_keep_frac,
    ) in configs:
                                is_names_only = (obs_n == 0 and int_n == 0)
                                is_payload_without_obs = (style in {"payload", "payload_topk"} and obs_n == 0 and int_n > 0)
                                if is_payload_without_obs:
                                    continue
                                if is_names_only and shuf_n != 1:
                                    continue
                                if args.only_names_only and not is_names_only:
                                    continue

                                is_robustness_baseline = (
                                    obs_n == 5000 and
                                    int_n == 200 and
                                    style == "cases" and
                                    anon is False
                                )

                                if is_names_only:
                                    if not using_explicit_config_file:
                                        if row_ord != "random":
                                            continue
                                        if anon is True:
                                            continue
                                        # Names-only is independent of prompt style. To avoid duplicate work in the
                                        # full grid, we run it exactly once. If the user requests styles explicitly
                                        # and excludes "cases" (e.g., --styles payload), we still run names-only once.
                                        if not args.single_config:
                                            if args.styles:
                                                if "cases" in styles:
                                                    if style != "summary":
                                                        continue
                                                else:
                                                    if style != styles[0]:
                                                        continue
                                            else:
                                                if style != "cases":
                                                    continue
                                else:
                                    if obs_n == 0 and int_n == 0:
                                        continue
                                    if not using_explicit_config_file:
                                        is_non_default_ordering = (row_ord != "random" or col_ord != "original")
                                        if is_non_default_ordering and not is_robustness_baseline:
                                            continue
                                        if obs_n >= 5000 and style == "cases" and not is_robustness_baseline:
                                            continue

                                base_name, answer_obj, prompt_iter = _iter_prompts_for_config(
                                    bif_file=args.bif_file,
                                    num_prompts=args.num_prompts,
                                    shuffles_per_graph=shuf_n,
                                    seed=args.seed,
                                    prompt_style=style,
                                    obs_per_prompt=obs_n,
                                    int_per_combo=int_n,
                                    row_order=row_ord,
                                    col_order=col_ord,
                                    anonymize=anon,
                                    causal_rules=args.causal_rules,
                                    give_steps=args.give_steps,
                                    def_int=args.def_int,
                                    intervene_vars=args.intervene_vars,
                                    wrapper_mode=wrapper_mode,
                                    append_format_hint=bool(append_format_hint),
                                    reasoning_guidance=reasoning_guidance,
                                    hist_mass_keep_frac=hist_mass_keep_frac,
                                )

                                if args.save_example_prompt:
                                    try:
                                        first = next(prompt_iter)
                                    except StopIteration:
                                        raise SystemExit("No prompts produced for this configuration.")
                                    # Save once per config. Use the first model tag for stable filenames.
                                    out_p = _maybe_write_example_prompt(
                                        dataset=dataset,
                                        base_name=base_name,
                                        prompt_row=first,
                                        example_dir=args.example_prompt_dir,
                                        overwrite=bool(args.overwrite_example_prompt),
                                    )
                                    print(f"[info] Wrote example prompt: {out_p}", file=sys.stderr, flush=True)
                                    # Preserve all rows for actual querying after sampling one debug prompt.
                                    prompt_iter = itertools.chain([first], prompt_iter)

                                if args.print_one_prompt:
                                    try:
                                        first = next(prompt_iter)
                                    except StopIteration:
                                        raise SystemExit("No prompts produced for this configuration.")
                                    print("=== CONFIG ===")
                                    print(
                                        json.dumps(
                                            {
                                                "style": style,
                                                "anonymize": anon,
                                                "obs_per_prompt": obs_n,
                                                "int_per_combo": int_n,
                                                "row_order": row_ord,
                                                "col_order": col_ord,
                                                "shuffles_per_graph": shuf_n,
                                                "base_name": base_name,
                                                "wrapper_mode": wrapper_mode or "plain",
                                                "append_format_hint": bool(append_format_hint),
                                                "reasoning_guidance": reasoning_guidance,
                                                "hist_mass_keep_frac": hist_mass_keep_frac,
                                            },
                                            ensure_ascii=False,
                                            indent=2,
                                        )
                                    )
                                    final_prompt = _compose_prompt(
                                        str(first.get("prompt_text", "")),
                                        str(args.extra_output_instruction),
                                    )
                                    print("\n=== PROMPT (first row; exact text sent) ===")
                                    print(final_prompt)
                                    return

                                for model in args.model:
                                    if args.dry_run_tokens:
                                        print(
                                            f"[dry-run-tokens] base={base_name} model={model}",
                                            file=sys.stderr,
                                            flush=True,
                                        )
                                        _, _, token_iter = _iter_prompts_for_config(
                                            bif_file=args.bif_file,
                                            num_prompts=args.num_prompts,
                                            shuffles_per_graph=shuf_n,
                                            seed=args.seed,
                                            prompt_style=style,
                                            obs_per_prompt=obs_n,
                                            int_per_combo=int_n,
                                            row_order=row_ord,
                                            col_order=col_ord,
                                            anonymize=anon,
                                            causal_rules=args.causal_rules,
                                            give_steps=args.give_steps,
                                            def_int=args.def_int,
                                            intervene_vars=args.intervene_vars,
                                            wrapper_mode=wrapper_mode,
                                            append_format_hint=bool(append_format_hint),
                                            reasoning_guidance=reasoning_guidance,
                                            hist_mass_keep_frac=hist_mass_keep_frac,
                                        )
                                        n_rows = 0
                                        for tok_row in token_iter:
                                            tok_key = (int(tok_row["data_idx"]), int(tok_row["shuffle_idx"]))
                                            tok_prompt = tok_row["prompt_text"]
                                            tok_count = count_openai_tokens(model, tok_prompt)
                                            print(
                                                f"[prompt_tokens] key={tok_key} prompt_tokens={tok_count} model={model}",
                                                file=sys.stderr,
                                                flush=True,
                                            )
                                            n_rows += 1
                                        print(
                                            f"[dry-run-tokens] done base={base_name} model={model} rows={n_rows}",
                                            file=sys.stderr,
                                            flush=True,
                                        )
                                        continue
                                    if args.dry_run:
                                        print(f"[dry-run] Would run {base_name} with model={model}")
                                        continue
                                    print(
                                        f"[config] style={style} anon={anon} obs={obs_n} int={int_n} "
                                        f"row={row_ord} col={col_ord} shuf={shuf_n} "
                                        f"reasoning={reasoning_guidance} hist_mass_keep_frac={hist_mass_keep_frac} model={model}",
                                        file=sys.stderr,
                                        flush=True,
                                    )
                                    provider = _select_provider(model, args.provider)
                                    hf_pipe = None
                                    vllm_engine = None
                                    vllm_max_new_tokens = int(args.vllm_max_new_tokens)
                                    if vllm_max_new_tokens <= 0:
                                        vllm_max_new_tokens = int(args.hf_max_new_tokens)
                                    vllm_batch_size = int(args.vllm_batch_size)
                                    if vllm_batch_size <= 0:
                                        vllm_batch_size = max(1, int(args.hf_batch_size))
                                    if provider == "hf":
                                        dm = None if not args.hf_device_map or args.hf_device_map == "none" else args.hf_device_map
                                        hf_merge_lora = _resolve_hf_merge_lora(model, args.hf_merge_lora)
                                        hf_key = (model, dm, str(args.hf_dtype), hf_merge_lora)
                                        hf_pipe = hf_pipe_cache.get(hf_key)
                                        if hf_pipe is None:
                                            print(
                                                f"[hf:init] loading HF pipeline once for model={model} device_map={dm} "
                                                f"dtype={args.hf_dtype} merge_lora={hf_merge_lora}",
                                                file=sys.stderr,
                                                flush=True,
                                            )
                                            hf_pipe = build_hf_pipeline(
                                                model,
                                                device_map=dm,
                                                torch_dtype=args.hf_dtype,
                                                merge_lora=hf_merge_lora,
                                            )
                                            hf_pipe_cache[hf_key] = hf_pipe
                                        else:
                                            print(
                                                f"[hf:init] reusing cached HF pipeline for model={model}",
                                                file=sys.stderr,
                                                flush=True,
                                            )
                                    elif provider == "vllm":
                                        if vllm_max_new_tokens <= 0:
                                            raise SystemExit(
                                                "vLLM provider requires a positive generation cap. "
                                                "Set --vllm-max-new-tokens or --hf-max-new-tokens."
                                            )
                                        vllm_max_model_len = max(0, int(args.vllm_max_model_len))
                                        vllm_hf_overrides = build_yarn_hf_overrides(
                                            enabled=bool(args.vllm_enable_yarn),
                                            factor=float(args.vllm_yarn_factor),
                                            original_max_position_embeddings=int(
                                                args.vllm_yarn_original_max_position_embeddings
                                            ),
                                        )
                                        vllm_hf_overrides_key = json.dumps(
                                            vllm_hf_overrides or {},
                                            sort_keys=True,
                                            separators=(",", ":"),
                                        )
                                        vllm_key = (
                                            model,
                                            int(args.vllm_tensor_parallel_size),
                                            str(args.vllm_dtype),
                                            vllm_max_model_len,
                                            float(args.vllm_gpu_mem_util),
                                            bool(args.vllm_enforce_eager),
                                            vllm_hf_overrides_key,
                                        )
                                        vllm_engine = vllm_engine_cache.get(vllm_key)
                                        if vllm_engine is None:
                                            print(
                                                "[vllm:init] loading vLLM engine once for "
                                                f"model={model} tp={int(args.vllm_tensor_parallel_size)} "
                                                f"dtype={args.vllm_dtype} max_model_len={vllm_max_model_len or 'auto'} "
                                                f"gpu_mem_util={float(args.vllm_gpu_mem_util):.2f} "
                                                f"yarn={bool(args.vllm_enable_yarn)}",
                                                file=sys.stderr,
                                                flush=True,
                                            )
                                            vllm_error_log_path = (
                                                Path(args.vllm_error_log_file)
                                                if args.vllm_error_log_file is not None
                                                else _default_vllm_error_log_path(dataset, base_name, model)
                                            )
                                            if vllm_log_tee is None:
                                                vllm_log_tee = _FdTee(vllm_error_log_path)
                                                vllm_log_tee.start()
                                            try:
                                                vllm_engine = build_vllm_engine(
                                                    model,
                                                    tensor_parallel_size=int(args.vllm_tensor_parallel_size),
                                                    vllm_dtype=str(args.vllm_dtype),
                                                    max_model_len=(vllm_max_model_len if vllm_max_model_len > 0 else None),
                                                    gpu_memory_utilization=float(args.vllm_gpu_mem_util),
                                                    enforce_eager=bool(args.vllm_enforce_eager),
                                                    hf_overrides=vllm_hf_overrides,
                                                )
                                            except Exception:
                                                _append_vllm_error_log(
                                                    log_path=vllm_error_log_path,
                                                    dataset=dataset,
                                                    base_name=base_name,
                                                    model=model,
                                                    message=(
                                                        "vLLM engine initialization failed "
                                                        f"(tp={int(args.vllm_tensor_parallel_size)}, "
                                                        f"dtype={args.vllm_dtype}, "
                                                        f"max_model_len={vllm_max_model_len or 'auto'}, "
                                                        f"gpu_mem_util={float(args.vllm_gpu_mem_util):.2f}, "
                                                        f"enforce_eager={bool(args.vllm_enforce_eager)}, "
                                                        f"hf_overrides={vllm_hf_overrides_key})"
                                                    ),
                                                    detail=traceback.format_exc(),
                                                )
                                                raise
                                            vllm_engine_cache[vllm_key] = vllm_engine
                                        else:
                                            print(
                                                f"[vllm:init] reusing cached vLLM engine for model={model}",
                                                file=sys.stderr,
                                                flush=True,
                                            )
                                    if provider == "gemini":
                                        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
                                            raise SystemExit("Missing API key: set GOOGLE_API_KEY or GEMINI_API_KEY.")
                                    if provider == "openai":
                                        if not os.getenv("OPENAI_API_KEY"):
                                            raise SystemExit("Missing API key: set OPENAI_API_KEY.")

                                    _, _, prompt_iter = _iter_prompts_for_config(
                                        bif_file=args.bif_file,
                                        num_prompts=args.num_prompts,
                                        shuffles_per_graph=shuf_n,
                                        seed=args.seed,
                                        prompt_style=style,
                                        obs_per_prompt=obs_n,
                                        int_per_combo=int_n,
                                        row_order=row_ord,
                                        col_order=col_ord,
                                        anonymize=anon,
                                        causal_rules=args.causal_rules,
                                        give_steps=args.give_steps,
                                        def_int=args.def_int,
                                        intervene_vars=args.intervene_vars,
                                        wrapper_mode=wrapper_mode,
                                        append_format_hint=bool(append_format_hint),
                                        reasoning_guidance=reasoning_guidance,
                                        hist_mass_keep_frac=hist_mass_keep_frac,
                                    )
                                    _run_model_for_config(
                                        dataset=dataset,
                                        base_name=base_name,
                                        answer_obj=answer_obj,
                                        prompt_iter=prompt_iter,
                                        responses_root=responses_root,
                                        model=model,
                                        provider=provider,
                                        temperature=args.temperature,
                                        overwrite=args.overwrite,
                                        request_timeout_s=float(args.request_timeout_s),
                                        progress_every=int(args.progress_every),
                                        log_calls=bool(args.log_calls),
                                        hf_pipe=hf_pipe,
                                        hf_max_new_tokens=int(args.hf_max_new_tokens),
                                        hf_batch_size=max(1, int(args.hf_batch_size)),
                                        hf_context_limit=max(0, int(args.hf_context_limit)),
                                        hf_skip_if_prompt_tokens_over=max(0, int(args.hf_skip_if_prompt_tokens_over)),
                                        hf_skip_if_est_kv_exceeds_vram=bool(args.hf_skip_if_est_kv_exceeds_vram),
                                        hf_kv_vram_fraction=float(args.hf_kv_vram_fraction),
                                        hf_dtype_for_estimate=str(args.hf_dtype),
                                        vllm_engine=vllm_engine,
                                        vllm_max_new_tokens=vllm_max_new_tokens,
                                        vllm_batch_size=vllm_batch_size,
                                        vllm_max_model_len=max(0, int(args.vllm_max_model_len)),
                                        vllm_server_base_url=str(args.vllm_server_base_url),
                                        extra_output_instruction=str(args.extra_output_instruction),
                                    )

                                count += 1

    print(f"\n=== In-memory run complete. {count} configurations executed. ===")


if __name__ == "__main__":
    main()
