#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from generate_prompts_names_only import iter_names_only_prompts_in_memory
from query_gemini import (
    call_gemini,
    call_openai,
    call_hf_textgen,
    build_hf_pipeline,
    is_gemini_model,
    is_openai_model,
    extract_adjacency_matrix,
    count_openai_tokens,
)

FORMAT_RE = re.compile(r"(?s)^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$")
ANSWER_RE = re.compile(r"(?s)<answer>\s*(.*?)\s*</answer>")


def _extract_answer_text(text: str) -> str:
    m = ANSWER_RE.search(text or "")
    return m.group(1) if m else (text or "")


def _format_ok(text: str) -> int:
    return int(bool(FORMAT_RE.match(text or "")))


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
    tag = model.split("/")[-1]
    for ch in [":", " "]:
        tag = tag.replace(ch, "_")
    return tag


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


def _default_example_prompt_path(dataset: str, base_name: str, model: str, example_dir: Path | None) -> Path:
    """
    One prompt per configuration, saved for debugging. Prompt text does not depend on the model,
    but we include it in the filename to avoid collisions when reusing base names across runs.
    """
    # Default to experiments/prompts/<dataset>/example_prompts so prompts live next to the graph's prompt assets.
    out_dir = example_dir or (Path(__file__).parent / "prompts" / dataset / "example_prompts")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model_tag = _safe_model_tag(model)
    stem = Path(base_name).stem
    return out_dir / f"{stem}_{safe_model_tag}_example_prompt.txt"


def _maybe_write_example_prompt(
    *,
    dataset: str,
    base_name: str,
    model: str,
    prompt_row: dict[str, Any],
    example_dir: Path | None,
    overwrite: bool,
) -> Path:
    out_path = _default_example_prompt_path(dataset, base_name, model, example_dir)
    if out_path.exists() and not overwrite:
        return out_path

    payload = {
        "dataset": dataset,
        "base_name": base_name,
        "model": model,
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


def _load_configs_from_file(
    *,
    config_file: Path,
    style_aliases: dict[str, str],
    allowed_styles: set[str],
    allowed_row_orders: set[str],
    allowed_col_orders: set[str],
) -> list[tuple[str, bool, int, int, str, str, int]]:
    try:
        payload = json.loads(config_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to read --config-file {config_file}: {e}") from e

    if isinstance(payload, dict):
        raw_configs = payload.get("configs")
    elif isinstance(payload, list):
        raw_configs = payload
    else:
        raise SystemExit("--config-file must contain either a JSON list or an object with key 'configs'.")

    if not isinstance(raw_configs, list) or not raw_configs:
        raise SystemExit("--config-file contains no configs.")

    out: list[tuple[str, bool, int, int, str, str, int]] = []
    for i, item in enumerate(raw_configs):
        if not isinstance(item, dict):
            raise SystemExit(f"Config #{i} must be an object, got: {type(item).__name__}")

        style_raw = str(item.get("prompt_style", item.get("style", "cases"))).strip().lower()
        style = style_aliases.get(style_raw, style_raw)
        if style not in allowed_styles:
            raise SystemExit(
                f"Config #{i}: unknown prompt_style '{style_raw}' (normalized '{style}'). "
                f"Allowed: {sorted(allowed_styles)}"
            )

        row_ord = str(item.get("row_order", "random")).strip().lower()
        col_ord = str(item.get("col_order", "original")).strip().lower()
        if row_ord not in allowed_row_orders:
            raise SystemExit(f"Config #{i}: invalid row_order '{row_ord}'. Allowed: {sorted(allowed_row_orders)}")
        if col_ord not in allowed_col_orders:
            raise SystemExit(f"Config #{i}: invalid col_order '{col_ord}'. Allowed: {sorted(allowed_col_orders)}")

        try:
            obs_n = int(item.get("obs_per_prompt", item.get("obs", 0)))
            int_n = int(item.get("int_per_combo", item.get("int", 0)))
            shuf_n = int(item.get("shuffles_per_graph", item.get("shuffle", item.get("shuf", 1))))
        except Exception as e:
            raise SystemExit(f"Config #{i}: obs/int/shuf must be integers: {e}") from e

        if shuf_n <= 0:
            raise SystemExit(f"Config #{i}: shuffles_per_graph must be > 0.")

        anon_raw = item.get("anonymize", False)
        if isinstance(anon_raw, bool):
            anon = anon_raw
        elif isinstance(anon_raw, str):
            anon = anon_raw.strip().lower() in {"1", "true", "yes", "y", "on"}
        elif isinstance(anon_raw, (int, float)):
            anon = bool(int(anon_raw))
        else:
            anon = False

        out.append((style, anon, obs_n, int_n, row_ord, col_ord, shuf_n))

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
            pred = (row.get("prediction") or "").strip()
            valid_str = str(row.get("valid", "")).strip()
            try:
                valid = int(valid_str) == 1
            except Exception:
                valid = bool(pred)
            is_error = raw.startswith("[ERROR]")
            if raw and not is_error and pred and valid:
                try:
                    key = (int(row.get("data_idx", -1)), int(row.get("shuffle_idx", -1)))
                    completed.add(key)
                    rows.append(row)
                except Exception:
                    continue
            elif not is_error and pred and valid:
                # keep non-error, valid rows even if raw is empty (should be rare)
                rows.append(row)
    return completed, rows


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
        )
    try:
        from generate_prompts import iter_prompts_in_memory
    except Exception as e:
        raise SystemExit(
            "Failed to import generate_prompts (likely missing torch). "
            "Install torch or run with --only-names-only."
        ) from e
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
        "truncation_suspected",
        "prompt_tokens",
        "output_tokens",
        "total_tokens",
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
            else:
                if hf_pipe is None:
                    raise SystemExit("HF provider requested but no pipeline is initialized.")
                t0 = time.monotonic()
                if log_calls:
                    print(f"[call:start] key={key} provider=hf model={model}", file=sys.stderr, flush=True)
                resp = call_hf_textgen(
                    hf_pipe,
                    prompt,
                    temperature=temperature,
                    max_new_tokens=(int(hf_max_new_tokens) if int(hf_max_new_tokens) > 0 else None),
                )
                if log_calls:
                    dt = time.monotonic() - t0
                    print(f"[call:done] key={key} seconds={dt:.1f}", file=sys.stderr, flush=True)

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
            adj = extract_adjacency_matrix(answer_text, fallback_variables=variables_for_parse)
            pred = json.dumps(adj.tolist(), ensure_ascii=False) if adj is not None else ""
            valid = 1 if adj is not None else 0
            format_ok = _format_ok(resp)
            truncation_suspected = _truncation_suspected(
                resp,
                output_tokens=output_tokens,
                max_new_tokens_hint=(
                    int(hf_max_new_tokens)
                    if provider == "hf" and int(hf_max_new_tokens) > 0
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
                "truncation_suspected": truncation_suspected,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
            writer.writerow(out_row)
            wrote += 1
            if progress_every > 0 and (wrote % progress_every == 0):
                print(
                    f"[progress] wrote={wrote} skipped={skipped} last_key={key} valid={valid} out={out_path.name}",
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
    # default=[] to avoid duplication when this script is called by run_experiment1_pipeline.py
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
    ap.add_argument("--provider", default="auto", choices=["auto", "gemini", "openai", "hf"])
    ap.add_argument(
        "--hf-max-new-tokens",
        type=int,
        default=0,
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
        "--extra-output-instruction",
        default=(
            "First write your reasoning inside <think>...</think>, then provide only the final "
            "JSON adjacency matrix inside <answer>...</answer>. Do not put JSON outside <answer>."
        ),
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
            '"summary_joint" (alias: "summary_join"), "summary_probs", "payload", "payload_topk").'
        ),
    )
    ap.add_argument(
        "--save-example-prompt",
        action="store_true",
        help="For each configuration, write ONE example prompt (first row) to disk for debugging.",
    )
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
        choices=["cases", "matrix", "summary", "summary_joint", "summary_join", "summary_probs", "payload", "payload_topk"],
        default="cases",
    )
    ap.add_argument("--obs-per-prompt", type=int, default=0)
    ap.add_argument("--int-per-combo", type=int, default=0)
    ap.add_argument("--row-order", choices=["random", "sorted", "reverse"], default="random")
    ap.add_argument("--col-order", choices=["original", "reverse", "random", "topo", "reverse_topo"], default="original")
    ap.add_argument("--anonymize", action="store_true")

    args = ap.parse_args()
    if not args.model:
        args.model = ["gpt-5-mini"]

    dataset = Path(args.bif_file).stem
    responses_root = Path(args.responses_root)
    shuf_values = [int(x) for x in (args.shuffles_per_graph or [1])]

    style_aliases = {
        "summary_join": "summary_joint",
    }

    all_styles = ["cases", "matrix", "summary", "summary_joint", "summary_probs", "payload", "payload_topk"]
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

    if args.prompt_style == "summary_join":
        args.prompt_style = "summary_joint"
    anonymize_opts = [False, True]
    obs_sizes = [0, 100, 1000, 5000, 8000]
    int_sizes = [0, 50, 100, 200, 500]
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
            (style, anon, obs_n, int_n, row_ord, col_ord, shuf_n)
            for style in styles
            for anon in anonymize_opts
            for obs_n in obs_sizes
            for int_n in int_sizes
            for row_ord in row_order_opts
            for col_ord in col_order_opts
            for shuf_n in shuf_values
        ]

    hf_pipe_cache: dict[tuple[str, str | None, str], Any] = {}

    for style, anon, obs_n, int_n, row_ord, col_ord, shuf_n in configs:
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
                                                if style != "cases":
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
                                        model=(args.model[0] if args.model else "unknown"),
                                        prompt_row=first,
                                        example_dir=args.example_prompt_dir,
                                        overwrite=bool(args.overwrite_example_prompt),
                                    )
                                    print(f"[info] Wrote example prompt: {out_p}", file=sys.stderr, flush=True)

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
                                        f"row={row_ord} col={col_ord} shuf={shuf_n} model={model}",
                                        file=sys.stderr,
                                        flush=True,
                                    )
                                    provider = _select_provider(model, args.provider)
                                    hf_pipe = None
                                    if provider == "hf":
                                        dm = None if not args.hf_device_map or args.hf_device_map == "none" else args.hf_device_map
                                        hf_key = (model, dm, str(args.hf_dtype))
                                        hf_pipe = hf_pipe_cache.get(hf_key)
                                        if hf_pipe is None:
                                            print(
                                                f"[hf:init] loading HF pipeline once for model={model} device_map={dm} dtype={args.hf_dtype}",
                                                file=sys.stderr,
                                                flush=True,
                                            )
                                            hf_pipe = build_hf_pipeline(
                                                model,
                                                device_map=dm,
                                                torch_dtype=args.hf_dtype,
                                            )
                                            hf_pipe_cache[hf_key] = hf_pipe
                                        else:
                                            print(
                                                f"[hf:init] reusing cached HF pipeline for model={model}",
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
                                        extra_output_instruction=str(args.extra_output_instruction),
                                    )

                                count += 1

    print(f"\n=== In-memory run complete. {count} configurations executed. ===")


if __name__ == "__main__":
    main()
