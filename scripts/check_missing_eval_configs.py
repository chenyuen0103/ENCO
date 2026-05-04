#!/usr/bin/env python3
"""Audit missing/incomplete ENCO eval response files.

The script expands an eval config grid the same way scripts/eval_cd_configs.py
does, checks the expected response CSV for each (graph, config, model), and
filters out combinations whose largest prompt plus generation budget exceeds
the model's configured context window.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "scripts", REPO_ROOT / "experiments", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import eval_cd_configs as eval_cfg  # noqa: E402


DEFAULT_GRAPHS = ["cancer", "earthquake", "asia", "sachs"]
ORIGINAL_VLLM_MODELS = [
    "Qwen/Qwen3-4B-Thinking-2507",
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct-1M",
    "Qwen/Qwen2.5-7B-Instruct-1M",
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Qwen/Qwen2.5-72B-Instruct-AWQ",
]


def _load_context(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_model = {
        str(item.get("model")): item
        for item in data.get("models", [])
        if item.get("model")
    }
    return data, by_model


def _resolve_models(args: argparse.Namespace, context_by_model: dict[str, dict[str, Any]]) -> list[str]:
    if args.model:
        return list(dict.fromkeys(args.model))
    if args.model_set == "original-vllm":
        return ORIGINAL_VLLM_MODELS
    if args.model_set == "vllm":
        return [
            model
            for model, meta in context_by_model.items()
            if str(meta.get("provider", "")).lower() == "vllm"
        ]
    return list(context_by_model.keys())


def _response_path(responses_root: Path, dataset: str, base_name: str, model: str) -> Path:
    base_name = base_name.replace("prompts", "responses", 1)
    base_stem = Path(base_name).stem
    base_suffix = Path(base_name).suffix or ".csv"
    model_tag = eval_cfg._safe_model_tag(model)
    if model_tag not in base_stem:
        base_stem = f"{base_stem}_{model_tag}"
    return responses_root / dataset / f"{base_stem}{base_suffix}"


def _matrix_has_right_shape(prediction: str, expected_n: int | None) -> bool:
    if expected_n is None:
        return False
    try:
        mat = json.loads(prediction)
    except Exception:
        return False
    return (
        isinstance(mat, list)
        and len(mat) == expected_n
        and all(isinstance(row, list) and len(row) == expected_n for row in mat)
    )


def _csv_status(path: Path, *, expected_rows: int, expected_n: int | None, require_right_shape: bool) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "physical_rows": "",
            "complete_rows": 0,
            "error_type": "",
        }

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    complete_rows = 0
    first_error = ""
    for row in rows:
        raw = (row.get("raw_response") or "").lstrip()
        if not first_error and raw.startswith("[ERROR]"):
            first_error = row.get("error_type") or raw[:160]

        valid = str(row.get("valid", "")).strip().lower() in {"1", "true", "yes"}
        if not valid:
            continue
        if require_right_shape:
            right_shape_raw = str(row.get("right_shape", "")).strip().lower()
            if right_shape_raw in {"1", "true", "yes"}:
                complete_rows += 1
            elif right_shape_raw == "":
                complete_rows += int(_matrix_has_right_shape(row.get("prediction", ""), expected_n))
        else:
            complete_rows += 1

    status = "ok" if complete_rows >= expected_rows else "incomplete"
    return {
        "status": status,
        "physical_rows": len(rows),
        "complete_rows": complete_rows,
        "error_type": first_error,
    }


def _model_context_limit(
    model: str,
    context_by_model: dict[str, dict[str, Any]],
    default_context_limit: int,
) -> int:
    meta = context_by_model.get(model, {})
    raw = (
        meta.get("recommended_max_model_len")
        or meta.get("advertised_context_window_tokens")
        or default_context_limit
        or 0
    )
    return int(raw or 0)


def _load_configs(config_file: Path):
    return eval_cfg._load_configs_from_file(
        config_file=config_file,
        style_aliases={"summary_join": "summary", "summary_joint": "summary"},
        allowed_styles={"cases", "matrix", "summary", "payload", "payload_topk"},
        allowed_row_orders={"random", "sorted", "reverse"},
        allowed_col_orders={"original", "reverse", "random", "topo", "reverse_topo"},
    )


def _iter_expected_items(args: argparse.Namespace, models: list[str], context: dict[str, Any], context_by_model: dict[str, dict[str, Any]]):
    configs = _load_configs(args.config_file)
    max_new_tokens = int(args.max_new_tokens or context.get("defaults", {}).get("max_new_tokens") or 4096)

    for graph in args.graph:
        bif_file = args.graphs_dir / f"{graph}.bif"
        for config in configs:
            (
                style,
                anonymize,
                obs_per_prompt,
                int_per_combo,
                row_order,
                col_order,
                shuffles_per_graph,
                wrapper_mode,
                append_format_hint,
                reasoning_guidance,
                hist_mass_keep_frac,
            ) = config
            base_name, answer_obj, prompt_iter = eval_cfg._iter_prompts_for_config(
                bif_file=str(bif_file),
                num_prompts=args.num_prompts,
                shuffles_per_graph=shuffles_per_graph,
                seed=args.seed,
                prompt_style=style,
                obs_per_prompt=obs_per_prompt,
                int_per_combo=int_per_combo,
                row_order=row_order,
                col_order=col_order,
                anonymize=anonymize,
                causal_rules=False,
                give_steps=False,
                def_int=False,
                intervene_vars="",
                wrapper_mode=wrapper_mode,
                append_format_hint=append_format_hint,
                reasoning_guidance=reasoning_guidance,
                hist_mass_keep_frac=hist_mass_keep_frac,
            )
            prompts = [str(row.get("prompt_text", "")) for row in prompt_iter]
            expected_rows = len(prompts)
            variables = answer_obj.get("variables")
            expected_n = len(variables) if isinstance(variables, list) else None
            config_label = base_name.replace("prompts_", "", 1)

            for model in models:
                response_path = _response_path(args.responses_root, graph, base_name + ".csv", model)
                max_prompt_tokens = max(
                    eval_cfg.count_openai_tokens(model, eval_cfg._compose_prompt(prompt, ""))
                    for prompt in prompts
                )
                required_tokens = max_prompt_tokens + max_new_tokens
                context_limit = _model_context_limit(model, context_by_model, args.default_context_limit)
                context_blocked = bool(context_limit and required_tokens > context_limit)
                status = _csv_status(
                    response_path,
                    expected_rows=expected_rows,
                    expected_n=expected_n,
                    require_right_shape=args.require_right_shape,
                )
                yield {
                    "graph": graph,
                    "config": config_label,
                    "model": model,
                    "status": status["status"],
                    "physical_rows": status["physical_rows"],
                    "complete_rows": status["complete_rows"],
                    "expected_rows": expected_rows,
                    "error_type": status["error_type"],
                    "required_tokens": required_tokens,
                    "context_limit": context_limit,
                    "context_blocked": int(context_blocked),
                    "response_path": str(response_path),
                }


def _print_summary(rows: list[dict[str, Any]], filtered_rows: list[dict[str, Any]]) -> None:
    print(
        "SUMMARY "
        f"actionable={len(rows)} "
        f"context_filtered={len(filtered_rows)}"
    )
    for key in ("graph", "model", "status"):
        counts = Counter(str(row[key]) for row in rows)
        if not counts:
            continue
        print(f"\nBY_{key.upper()}")
        for name, count in sorted(counts.items()):
            print(f"{name}\t{count}")


def _write_rows_tsv(rows: list[dict[str, Any]], *, out_file: Any) -> None:
    fields = [
        "graph",
        "config",
        "model",
        "status",
        "physical_rows",
        "complete_rows",
        "expected_rows",
        "required_tokens",
        "context_limit",
        "error_type",
        "response_path",
    ]
    writer = csv.DictWriter(out_file, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check which (graph, config, model) eval response files are missing "
            "or incomplete, excluding context-window-blocked combinations."
        )
    )
    parser.add_argument("--config-file", type=Path, default=REPO_ROOT / "experiments/configs/eval_configs.json")
    parser.add_argument("--model-context-file", type=Path, default=REPO_ROOT / "experiments/configs/model_context_windows.json")
    parser.add_argument("--responses-root", type=Path, default=REPO_ROOT / "scripts/responses")
    parser.add_argument("--graphs-dir", type=Path, default=REPO_ROOT / "causal_graphs/real_data/small_graphs")
    parser.add_argument("--graph", action="append", choices=DEFAULT_GRAPHS, default=None)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--model-set", choices=["all", "vllm", "original-vllm"], default="original-vllm")
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--default-context-limit", type=int, default=0)
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Also report existing CSVs with fewer than the expected number of complete rows.",
    )
    parser.add_argument(
        "--require-right-shape",
        action="store_true",
        help="When checking completeness, require valid rows to have the correct N x N shape.",
    )
    parser.add_argument(
        "--show-context-filtered",
        action="store_true",
        help="Print the combinations omitted because required_tokens > context_limit.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    if args.graph is None:
        args.graph = list(DEFAULT_GRAPHS)

    try:
        csv.field_size_limit(100_000_000)
    except OverflowError:
        csv.field_size_limit(10_000_000)

    context, context_by_model = _load_context(args.model_context_file)
    models = _resolve_models(args, context_by_model)
    unknown = [model for model in models if model not in context_by_model]
    if unknown:
        raise SystemExit(
            "Unknown model(s) in context file:\n"
            + "\n".join(f"  - {model}" for model in unknown)
        )

    actionable: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for row in _iter_expected_items(args, models, context, context_by_model):
        if row["status"] == "ok":
            continue
        if row["status"] == "incomplete" and not args.include_incomplete:
            continue
        if row["context_blocked"]:
            filtered.append(row)
        else:
            actionable.append(row)

    _print_summary(actionable, filtered)
    print("\nACTIONABLE")
    _write_rows_tsv(actionable, out_file=sys.stdout)
    if args.show_context_filtered:
        print("\nCONTEXT_FILTERED")
        _write_rows_tsv(filtered, out_file=sys.stdout)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(actionable[0].keys()) if actionable else [
                "graph",
                "config",
                "model",
                "status",
                "physical_rows",
                "complete_rows",
                "expected_rows",
                "error_type",
                "required_tokens",
                "context_limit",
                "context_blocked",
                "response_path",
            ])
            writer.writeheader()
            writer.writerows(actionable)


if __name__ == "__main__":
    main()
