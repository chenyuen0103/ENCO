#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
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


def _safe_model_tag(model: str) -> str:
    tag = model.split("/")[-1]
    for ch in [":", " "]:
        tag = tag.replace(ch, "_")
    return tag


def _default_response_path(dataset: str, base_name: str, model: str) -> Path:
    responses_root = Path("responses")
    responses_dir = responses_root / dataset
    responses_dir.mkdir(parents=True, exist_ok=True)

    base_name = base_name.replace("prompts", "responses", 1)
    base_stem = Path(base_name).stem
    base_suffix = Path(base_name).suffix or ".csv"

    safe_model_tag = _safe_model_tag(model)
    if safe_model_tag not in base_stem:
        base_stem = f"{base_stem}_{safe_model_tag}"
    return responses_dir / f"{base_stem}{base_suffix}"


def _select_provider(model: str, provider: str) -> str:
    if provider and provider != "auto":
        return provider
    if is_gemini_model(model):
        return "gemini"
    if is_openai_model(model):
        return "openai"
    return "hf"


def _load_completed(out_path: Path, overwrite: bool) -> tuple[set[tuple[int, int]], list[dict[str, Any]]]:
    if overwrite or not out_path.exists():
        return set(), []
    completed: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    with out_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("raw_response") or "").lstrip()
            pred = (row.get("prediction") or "").strip()
            is_error = raw.startswith("[ERROR]")
            if raw and not is_error and pred:
                try:
                    key = (int(row.get("data_idx", -1)), int(row.get("shuffle_idx", -1)))
                    completed.add(key)
                    rows.append(row)
                except Exception:
                    continue
            elif not is_error and pred:
                # keep non-error rows even if raw is empty (should be rare)
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
    model: str,
    provider: str,
    temperature: float,
    overwrite: bool,
    hf_pipe: Any = None,
) -> None:
    out_path = _default_response_path(dataset, base_name + ".csv", model)
    completed, existing_rows = _load_completed(out_path, overwrite)

    fieldnames = [
        "data_idx",
        "shuffle_idx",
        "answer",
        "given_edges",
        "raw_response",
        "prediction",
        "valid",
        "prompt_tokens",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)

        total = 0
        for row in prompt_iter:
            key = (int(row["data_idx"]), int(row["shuffle_idx"]))
            if key in completed:
                continue

            prompt = row["prompt_text"]
            prompt_tokens = -1
            if provider == "openai":
                prompt_tokens = count_openai_tokens(model, prompt)
                resp = call_openai(
                    model_name=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_retries=0,
                    request_timeout=6000.0,
                )
            elif provider == "gemini":
                resp = call_gemini(
                    model_name=model,
                    prompt=prompt,
                    temperature=temperature,
                )
            else:
                if hf_pipe is None:
                    raise SystemExit("HF provider requested but no pipeline is initialized.")
                resp = call_hf_textgen(hf_pipe, prompt, temperature=temperature, max_new_tokens=1024)

            adj = extract_adjacency_matrix(resp)
            pred = json.dumps(adj.tolist(), ensure_ascii=False) if adj is not None else ""
            valid = 1 if adj is not None else 0

            out_row = {
                "data_idx": row["data_idx"],
                "shuffle_idx": row["shuffle_idx"],
                "answer": json.dumps(answer_obj, ensure_ascii=False),
                "given_edges": row.get("given_edges"),
                "raw_response": resp,
                "prediction": pred,
                "valid": valid,
                "prompt_tokens": prompt_tokens,
            }
            writer.writerow(out_row)
            total += 1

    print(f"[info] Wrote responses: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Experiment 1 prompts in-memory (no prompt files)."
    )
    ap.add_argument("--bif-file", required=True)
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffles-per-graph", type=int, action="append", default=[1])

    ap.add_argument("--model", action="append", default=["gpt-5-mini"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--provider", default="auto", choices=["auto", "gemini", "openai", "hf"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-names-only", action="store_true")

    ap.add_argument("--def-int", action="store_true")
    ap.add_argument("--intervene-vars", default="all")
    ap.add_argument("--causal-rules", action="store_true")
    ap.add_argument("--give-steps", action="store_true")
    ap.add_argument("--single-config", action="store_true")
    ap.add_argument("--prompt-style", choices=["cases", "matrix"], default="cases")
    ap.add_argument("--obs-per-prompt", type=int, default=0)
    ap.add_argument("--int-per-combo", type=int, default=0)
    ap.add_argument("--row-order", choices=["random", "sorted", "reverse"], default="random")
    ap.add_argument("--col-order", choices=["original", "reverse", "random", "topo", "reverse_topo"], default="original")
    ap.add_argument("--anonymize", action="store_true")

    args = ap.parse_args()

    dataset = Path(args.bif_file).stem
    shuf_values = [int(x) for x in (args.shuffles_per_graph or [1])]

    styles = ["cases", "matrix"]
    anonymize_opts = [False, True]
    obs_sizes = [0, 100, 1000, 5000, 8000]
    int_sizes = [0, 50, 100, 200, 500]
    row_order_opts = ["random", "sorted", "reverse"]
    col_order_opts = ["original", "reverse", "random", "topo", "reverse_topo"]

    count = 0
    print(f"--- Starting Experiment 1 In-Memory Run ---")
    print(f"BIF File: {args.bif_file}")

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

    for style, anon, obs_n, int_n, row_ord, col_ord, shuf_n in configs:
                                is_names_only = (obs_n == 0 and int_n == 0)
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

                                for model in args.model:
                                    if args.dry_run:
                                        print(f"[dry-run] Would run {base_name} with model={model}")
                                        continue
                                    provider = _select_provider(model, args.provider)
                                    hf_pipe = None
                                    if provider == "hf":
                                        hf_pipe = build_hf_pipeline(model)
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
                                        model=model,
                                        provider=provider,
                                        temperature=args.temperature,
                                        overwrite=args.overwrite,
                                        hf_pipe=hf_pipe,
                                    )

                                count += 1

    print(f"\n=== In-memory run complete. {count} configurations executed. ===")


if __name__ == "__main__":
    main()
