#!/usr/bin/env python3
"""Build data-only causal-discovery benchmark artifacts.

This script is intentionally compatibility-first for Sachs and other overlapping
configs that are also evaluated through `scripts/eval_cd_configs.py`.

It reuses that script's config parsing and prompt/data generation helpers so
that, for the same:
  - `--bif-file`
  - `--config-file`
  - `--num-prompts`
  - `--seed`
the emitted prompts and gold answers are identical for overlapping configs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_cd_configs.py"


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("_enco_eval_cd_configs", EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {EVAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _load_eval_module()


STYLE_ALIASES = {
    "summary_join": "summary",
    "summary_joint": "summary",
}
ALLOWED_STYLES = {"cases", "matrix", "summary", "payload", "payload_topk"}
ALLOWED_ROW_ORDERS = {"random", "sorted", "reverse"}
ALLOWED_COL_ORDERS = {"original", "reverse", "random", "topo", "reverse_topo"}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _count_reference_tokens(text: str) -> int:
    """
    Stable benchmark token count for prompt_text.

    This intentionally uses a fixed tokenizer (`cl100k_base`) rather than a
    model-specific tokenizer so benchmark metadata stays comparable across
    downstream model runs.
    """
    if tiktoken is None:
        return -1
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _parse_index_list(raw: str | None) -> set[int] | None:
    if raw is None or not raw.strip():
        return None
    out: set[int] = set()
    for tok in raw.split(","):
        s = tok.strip()
        if not s:
            continue
        out.add(int(s))
    return out


def _config_name(
    *,
    prompt_style: str,
    anonymize: bool,
    obs_per_prompt: int,
    int_per_combo: int,
    row_order: str,
    col_order: str,
    reasoning_guidance: str,
    wrapper_mode: str | None,
    append_format_hint: bool,
    hist_mass_keep_frac: float | None,
) -> str:
    parts: list[str] = []
    is_names_only = obs_per_prompt == 0 and int_per_combo == 0
    if is_names_only:
        parts.append("names_only")
    else:
        parts.append(prompt_style)
    parts.append("anon" if anonymize else "real")
    parts.append(f"obs{obs_per_prompt}")
    parts.append(f"int{int_per_combo}")
    if reasoning_guidance != "staged":
        parts.append(f"reason{reasoning_guidance}")
    if row_order != "random":
        parts.append(f"row{row_order}")
    if col_order != "original":
        parts.append(f"col{col_order}")
    if wrapper_mode:
        parts.append(f"wrap{wrapper_mode}")
    if append_format_hint:
        parts.append("formathint")
    if hist_mass_keep_frac is not None:
        pct = int(round(100.0 * float(hist_mass_keep_frac)))
        parts.append(f"histmass{pct}")
    return "_".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build data-only benchmark CSVs from eval-compatible prompt configs.")
    parser.add_argument("--bif-file", required=True, help="Path to the source BIF/PT/NPZ graph file.")
    parser.add_argument("--config-file", required=True, help="JSON config file in eval_cd_configs-compatible format.")
    parser.add_argument("--output-csv", required=True, help="Merged output CSV path.")
    parser.add_argument("--manifest-json", default=None, help="Optional manifest JSON path. Defaults to <output-csv>.manifest.json.")
    parser.add_argument("--dataset", default=None, help="Dataset name override. Defaults to stem of --bif-file.")
    parser.add_argument("--num-prompts", type=int, default=5, help="Number of prompts per config.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for prompt/data generation.")
    parser.add_argument("--config-indexes", default=None, help="Optional comma-separated subset of config indices to export.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    bif_file = Path(args.bif_file).resolve(strict=True)
    config_file = Path(args.config_file).resolve(strict=True)
    output_csv = Path(args.output_csv).resolve()
    manifest_json = Path(args.manifest_json).resolve() if args.manifest_json else output_csv.with_suffix(output_csv.suffix + ".manifest.json")
    dataset = args.dataset or bif_file.stem
    selected_indexes = _parse_index_list(args.config_indexes)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)

    configs = EVAL._load_configs_from_file(
        config_file=config_file,
        style_aliases=STYLE_ALIASES,
        allowed_styles=ALLOWED_STYLES,
        allowed_row_orders=ALLOWED_ROW_ORDERS,
        allowed_col_orders=ALLOWED_COL_ORDERS,
    )

    fieldnames = [
        "dataset",
        "bif_file",
        "config_index",
        "config_name",
        "prompt_basename",
        "prompt_style",
        "benchmark_view",
        "anonymize",
        "obs_per_prompt",
        "int_per_combo",
        "row_order",
        "col_order",
        "shuffles_per_graph",
        "wrapper_mode",
        "append_format_hint",
        "reasoning_guidance",
        "hist_mass_keep_frac",
        "data_idx",
        "shuffle_idx",
        "given_edges",
        "prompt_text",
        "prompt_tokens_ref",
        "prompt_sha256",
        "answer",
        "answer_sha256",
    ]

    config_summaries: list[dict[str, Any]] = []
    total_rows = 0
    prompt_tokens_ref_values: list[int] = []

    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for config_index, config in enumerate(configs):
            if selected_indexes is not None and config_index not in selected_indexes:
                continue

            (
                prompt_style,
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

            config_name = _config_name(
                prompt_style=prompt_style,
                anonymize=anonymize,
                obs_per_prompt=obs_per_prompt,
                int_per_combo=int_per_combo,
                row_order=row_order,
                col_order=col_order,
                reasoning_guidance=reasoning_guidance,
                wrapper_mode=wrapper_mode,
                append_format_hint=append_format_hint,
                hist_mass_keep_frac=hist_mass_keep_frac,
            )
            benchmark_view = "names_only" if (obs_per_prompt == 0 and int_per_combo == 0) else "evidence"

            base_name, answer_obj, prompt_iter = EVAL._iter_prompts_for_config(
                bif_file=str(bif_file),
                num_prompts=int(args.num_prompts),
                shuffles_per_graph=int(shuffles_per_graph),
                seed=int(args.seed),
                prompt_style=prompt_style,
                obs_per_prompt=obs_per_prompt,
                int_per_combo=int_per_combo,
                row_order=row_order,
                col_order=col_order,
                anonymize=anonymize,
                causal_rules=False,
                give_steps=False,
                def_int=False,
                intervene_vars="all",
                wrapper_mode=wrapper_mode,
                append_format_hint=append_format_hint,
                reasoning_guidance=reasoning_guidance,
                hist_mass_keep_frac=hist_mass_keep_frac,
            )

            answer_json = json.dumps(answer_obj, ensure_ascii=False)
            answer_sha256 = _sha256_text(answer_json)
            row_count = 0

            for row in prompt_iter:
                prompt_text = str(row["prompt_text"])
                prompt_tokens_ref = _count_reference_tokens(prompt_text)
                if prompt_tokens_ref >= 0:
                    prompt_tokens_ref_values.append(prompt_tokens_ref)
                writer.writerow(
                    {
                        "dataset": dataset,
                        "bif_file": str(bif_file),
                        "config_index": config_index,
                        "config_name": config_name,
                        "prompt_basename": base_name,
                        "prompt_style": prompt_style,
                        "benchmark_view": benchmark_view,
                        "anonymize": int(bool(anonymize)),
                        "obs_per_prompt": obs_per_prompt,
                        "int_per_combo": int_per_combo,
                        "row_order": row_order,
                        "col_order": col_order,
                        "shuffles_per_graph": shuffles_per_graph,
                        "wrapper_mode": wrapper_mode or "",
                        "append_format_hint": int(bool(append_format_hint)),
                        "reasoning_guidance": reasoning_guidance,
                        "hist_mass_keep_frac": "" if hist_mass_keep_frac is None else hist_mass_keep_frac,
                        "data_idx": int(row["data_idx"]),
                        "shuffle_idx": int(row["shuffle_idx"]),
                        "given_edges": row.get("given_edges", ""),
                        "prompt_text": prompt_text,
                        "prompt_tokens_ref": prompt_tokens_ref,
                        "prompt_sha256": _sha256_text(prompt_text),
                        "answer": answer_json,
                        "answer_sha256": answer_sha256,
                    }
                )
                row_count += 1
                total_rows += 1

            config_summaries.append(
                {
                    "config_index": config_index,
                    "config_name": config_name,
                    "prompt_basename": base_name,
                    "prompt_style": prompt_style,
                    "anonymize": bool(anonymize),
                    "obs_per_prompt": obs_per_prompt,
                    "int_per_combo": int_per_combo,
                    "row_order": row_order,
                    "col_order": col_order,
                    "shuffles_per_graph": shuffles_per_graph,
                    "wrapper_mode": wrapper_mode,
                    "append_format_hint": bool(append_format_hint),
                    "reasoning_guidance": reasoning_guidance,
                    "hist_mass_keep_frac": hist_mass_keep_frac,
                    "rows_written": row_count,
                }
            )

    manifest = {
        "schema_version": "benchmark_data/v1",
        "generator": "scripts/build_benchmark_data.py",
        "compatibility_source": str(EVAL_SCRIPT),
        "dataset": dataset,
        "bif_file": str(bif_file),
        "bif_file_sha256": _sha256_file(bif_file),
        "config_file": str(config_file),
        "config_file_sha256": _sha256_file(config_file),
        "num_prompts": int(args.num_prompts),
        "seed": int(args.seed),
        "selected_config_indexes": sorted(selected_indexes) if selected_indexes is not None else None,
        "output_csv": str(output_csv),
        "output_csv_sha256": _sha256_file(output_csv),
        "total_rows": total_rows,
        "prompt_token_reference": {
            "tokenizer": "cl100k_base",
            "available": tiktoken is not None,
            "prompt_tokens_ref_max": max(prompt_tokens_ref_values) if prompt_tokens_ref_values else None,
            "prompt_tokens_ref_mean": (
                sum(prompt_tokens_ref_values) / float(len(prompt_tokens_ref_values))
                if prompt_tokens_ref_values
                else None
            ),
        },
        "configs": config_summaries,
        "guarantee": (
            "For overlapping configs, rows in this CSV are generated via the same "
            "config parser and prompt/data iterator used by scripts/eval_cd_configs.py."
        ),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[done] dataset={dataset} rows={total_rows} output={output_csv}")
    print(f"[done] manifest={manifest_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
