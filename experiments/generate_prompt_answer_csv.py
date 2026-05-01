#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

from generate_prompts import iter_prompts_in_memory


def _parse_int_list(raw: str) -> list[int]:
    vals = []
    for tok in (raw or "").split(","):
        s = tok.strip()
        if not s:
            continue
        vals.append(int(s))
    if not vals:
        raise ValueError("expected at least one integer value")
    return vals


def _parse_name_list(raw: str) -> list[str]:
    out = [x.strip() for x in (raw or "").split(",") if x.strip()]
    if not out:
        raise ValueError("expected at least one graph name")
    return out


def _parse_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(int(raw))
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _parse_wrapper_mode(raw: Any) -> str | None:
    if raw is None:
        return None
    val = str(raw).strip().lower()
    if not val or val == "none":
        return None
    if val not in {"plain", "chat"}:
        raise ValueError(f"invalid wrapper_mode={raw!r}")
    return val


def _parse_reasoning_guidance_values(raw: Any, default: list[str] | None = None) -> list[str]:
    vals: list[str] = []
    if raw is None:
        vals = list(default or [])
    elif isinstance(raw, list):
        vals = [str(x).strip().lower() for x in raw if str(x).strip()]
    else:
        vals = [str(raw).strip().lower()]
    if not vals:
        vals = list(default or ["staged"])
    allowed = {"staged", "concise", "none"}
    for v in vals:
        if v not in allowed:
            raise ValueError(f"invalid reasoning_guidance={v!r}; allowed={sorted(allowed)}")
    return vals


def _load_config_file(config_file: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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

    if not isinstance(payload, dict):
        raise SystemExit("--config-file must contain a JSON object.")

    raw_defaults = payload.get("config_defaults", payload.get("defaults", {}))
    if raw_defaults is None:
        raw_defaults = {}
    if not isinstance(raw_defaults, dict):
        raise SystemExit("'config_defaults' must be an object when present.")

    raw_configs = payload.get("configs")
    raw_product = payload.get("config_product", payload.get("config_grid"))
    configs_from_product = _expand_config_product(raw_product)
    if raw_configs is None:
        base_configs: list[dict[str, Any]] = []
    elif isinstance(raw_configs, list):
        base_configs = list(raw_configs)
    else:
        raise SystemExit("'configs' must be a list when present.")
    raw_configs = base_configs + configs_from_product
    if not raw_configs:
        raise SystemExit("--config-file must contain a non-empty 'configs' list or 'config_product'.")

    merged_configs: list[dict[str, Any]] = []
    for cfg in raw_configs:
        if not isinstance(cfg, dict):
            raise SystemExit("Each config row must be an object.")
        merged = dict(raw_defaults)
        merged.update(cfg)
        merged_configs.append(merged)

    generation = payload.get("dataset_generation") or payload.get("generation") or {}
    if generation is None:
        generation = {}
    if not isinstance(generation, dict):
        raise SystemExit("'dataset_generation' must be an object when present.")
    return generation, merged_configs


def _iter_rows_for_config(
    *,
    bif_file: Path,
    prompt_style: str,
    obs_per_prompt: int,
    int_per_combo: int,
    num_prompts_per_config: int,
    shuffles_per_graph: int,
    seed: int,
    anonymize: bool,
    wrapper_mode: str | None,
    append_format_hint: bool,
    reasoning_guidance: str = "staged",
    col_order: str = "original",
    col_perms: int = 1,
):
    """Yield (answer_obj, prompt_iter) once per column permutation.

    When col_perms > 1 and col_order == "random", each permutation uses a
    distinct seed offset (seed, seed+1, seed+2, ...) so the variable ordering
    differs across repetitions while the graph structure stays fixed.
    """
    for perm_idx in range(max(1, int(col_perms))):
        perm_seed = int(seed) + perm_idx
        _base_name, answer_obj, prompt_iter = iter_prompts_in_memory(
            bif_file=str(bif_file),
            num_prompts=int(num_prompts_per_config),
            shuffles_per_graph=int(shuffles_per_graph),
            seed=perm_seed,
            prompt_style=str(prompt_style),
            obs_per_prompt=int(obs_per_prompt),
            int_per_combo=int(int_per_combo),
            row_order="random",
            col_order=col_order,
            anonymize=bool(anonymize),
            causal_rules=False,
            give_steps=False,
            reasoning_guidance=str(reasoning_guidance),
            def_int=False,
            intervene_vars="all",
            wrapper_mode=wrapper_mode,
            append_format_hint=bool(append_format_hint),
        )
        yield answer_obj, prompt_iter


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a mixed causal-discovery CSV (prompt_text + answer) for SFT/GRPO "
            "from multiple small BIF graphs across varying obs/int sizes."
        )
    )
    ap.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help=(
            "JSON config file describing the prompt pool and shared generation settings. "
            "When set, per-row configs are read from the file instead of the CLI grid."
        ),
    )
    ap.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path("causal_graphs/real_data/small_graphs"),
        help="Directory containing *.bif graph files.",
    )
    ap.add_argument(
        "--graph-names",
        default="cancer,earthquake,asia,sachs",
        help="Comma-separated graph basenames (without .bif).",
    )
    ap.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    ap.add_argument(
        "--prompt-style",
        choices=["cases", "matrix", "summary", "summary_joint", "summary_join", "payload", "payload_topk"],
        default="summary",
        help="Prompt style. summary_joint/summary_join are legacy aliases for summary.",
    )
    ap.add_argument(
        "--obs-values",
        default="0,100,1000,5000,8000",
        help="Comma-separated observation counts per prompt.",
    )
    ap.add_argument(
        "--int-values",
        default="0,50,100,200",
        help="Comma-separated intervention samples per (variable,value).",
    )
    ap.add_argument(
        "--num-prompts-per-config",
        type=int,
        default=1,
        help="Number of prompts generated for each (graph,obs,int) config.",
    )
    ap.add_argument("--shuffles-per-graph", type=int, default=1)
    ap.add_argument(
        "--col-perms",
        type=int,
        default=1,
        help=(
            "Number of column (variable) orderings to generate per (graph,obs,int) config. "
            "Each permutation uses seed+perm_idx, so variable order varies across repetitions. "
            "Only meaningful when --col-order random."
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--col-order",
        choices=["original", "reverse", "random", "topo", "reverse_topo"],
        default="original",
        help="Column (variable) ordering passed to iter_prompts_in_memory.",
    )
    ap.add_argument("--anonymize", action="store_true", help="Use anonymized variable names (X1, X2, ...).")
    ap.add_argument(
        "--include-names-only",
        action="store_true",
        help="Include obs=0,int=0 names-only rows (off by default).",
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
        "--reasoning-guidance",
        choices=["staged", "concise", "none"],
        default="staged",
        help=(
            "How much guidance to give for the <think> block. "
            "'staged' keeps the current 3-stage scaffold, "
            "'concise' says to reason however you want but keep it concise, "
            "and 'none' uses only the output contract."
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
    args = ap.parse_args()

    config_generation: dict[str, Any] = {}
    config_rows: list[dict[str, Any]] | None = None
    if args.config_file is not None:
        config_generation, config_rows = _load_config_file(args.config_file)

    graphs_dir = Path(config_generation.get("graphs_dir", args.graphs_dir))
    graph_names = (
        list(config_generation.get("graph_names"))
        if isinstance(config_generation.get("graph_names"), list)
        else _parse_name_list(str(config_generation.get("graph_names", args.graph_names)))
    )
    num_prompts_per_config = int(config_generation.get("num_prompts_per_config", args.num_prompts_per_config))
    default_seed = int(config_generation.get("seed", args.seed))
    default_include_names_only = _parse_bool(config_generation.get("include_names_only"), bool(args.include_names_only))
    default_reasoning_guidances = _parse_reasoning_guidance_values(
        config_generation.get("reasoning_guidances", config_generation.get("reasoning_guidance")),
        [str(args.reasoning_guidance)],
    )

    out_path = args.output_csv.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "bif_file",
        "prompt_style",
        "reasoning_guidance",
        "anonymize",
        "obs_per_prompt",
        "int_per_combo",
        "data_idx",
        "shuffle_idx",
        "given_edges",
        "prompt_text",
        "answer",
    ]

    wrote = 0
    with out_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for graph_name in graph_names:
            bif_file = (graphs_dir / f"{graph_name}.bif").resolve()
            if not bif_file.exists():
                raise SystemExit(f"Missing BIF file: {bif_file}")

            if config_rows is None:
                prompt_styles = [args.prompt_style]
                obs_values = _parse_int_list(args.obs_values)
                int_values = _parse_int_list(args.int_values)
                rows_to_run: list[dict[str, Any]] = []
                for prompt_style in prompt_styles:
                    for obs_n in obs_values:
                        for int_n in int_values:
                            rows_to_run.append(
                                {
                                    "prompt_style": prompt_style,
                                    "obs_per_prompt": obs_n,
                                    "int_per_combo": int_n,
                                    "anonymize": bool(args.anonymize),
                                    "wrapper_mode": ("chat" if args.cot_hint else None),
                                    "append_format_hint": bool(args.append_format_hint),
                                    "reasoning_guidance": str(args.reasoning_guidance),
                                    "row_order": "random",
                                    "col_order": args.col_order,
                                    "col_perms": int(args.col_perms),
                                    "shuffles_per_graph": int(args.shuffles_per_graph),
                                    "seed": default_seed,
                                    "include_names_only": bool(args.include_names_only),
                                }
                            )
            else:
                rows_to_run = list(config_rows)

            for cfg in rows_to_run:
                prompt_style = str(cfg.get("prompt_style", cfg.get("style", args.prompt_style)))
                obs_n = int(cfg.get("obs_per_prompt", cfg.get("obs", 0)))
                int_n = int(cfg.get("int_per_combo", cfg.get("int", 0)))
                anonymize = _parse_bool(cfg.get("anonymize"), False)
                wrapper_mode = _parse_wrapper_mode(cfg.get("wrapper_mode", config_generation.get("wrapper_mode", None)))
                append_format_hint = _parse_bool(
                    cfg.get("append_format_hint", config_generation.get("append_format_hint")),
                    bool(args.append_format_hint),
                )
                row_order = str(cfg.get("row_order", config_generation.get("row_order", "random")) or "random")
                col_order = str(cfg.get("col_order", config_generation.get("col_order", args.col_order)) or args.col_order)
                col_perms = int(cfg.get("col_perms", config_generation.get("col_perms", args.col_perms)))
                shuffles_per_graph = int(
                    cfg.get("shuffles_per_graph", config_generation.get("shuffles_per_graph", args.shuffles_per_graph))
                )
                seed = int(cfg.get("seed", config_generation.get("seed", default_seed)))
                include_names_only = _parse_bool(
                    cfg.get("include_names_only", config_generation.get("include_names_only")),
                    default_include_names_only,
                )

                if row_order != "random":
                    raise SystemExit(
                        "generate_prompt_answer_csv.py currently supports only row_order='random' in config-file mode."
                    )
                if obs_n == 0 and int_n == 0 and not include_names_only:
                    continue

                guidance_values = _parse_reasoning_guidance_values(
                    cfg.get("reasoning_guidances", cfg.get("reasoning_guidance")),
                    default_reasoning_guidances,
                )

                for reasoning_guidance in guidance_values:
                    for answer_obj, prompt_iter in _iter_rows_for_config(
                        bif_file=bif_file,
                        prompt_style=prompt_style,
                        obs_per_prompt=int(obs_n),
                        int_per_combo=int(int_n),
                        num_prompts_per_config=int(num_prompts_per_config),
                        shuffles_per_graph=int(shuffles_per_graph),
                        seed=int(seed),
                        anonymize=bool(anonymize),
                        wrapper_mode=wrapper_mode,
                        append_format_hint=bool(append_format_hint),
                        reasoning_guidance=str(reasoning_guidance),
                        col_order=col_order,
                        col_perms=int(col_perms),
                    ):
                        answer_json = json.dumps(answer_obj, ensure_ascii=False)
                        for row in prompt_iter:
                            writer.writerow(
                                {
                                    "dataset": graph_name,
                                    "bif_file": str(bif_file),
                                    "prompt_style": prompt_style,
                                    "reasoning_guidance": str(reasoning_guidance),
                                    "anonymize": int(bool(anonymize)),
                                    "obs_per_prompt": int(obs_n),
                                    "int_per_combo": int(int_n),
                                    "data_idx": int(row["data_idx"]),
                                    "shuffle_idx": int(row["shuffle_idx"]),
                                    "given_edges": row.get("given_edges"),
                                    "prompt_text": row["prompt_text"],
                                    "answer": answer_json,
                                }
                            )
                            wrote += 1

    print(f"[done] wrote={wrote} output={out_path}")


if __name__ == "__main__":
    main()
