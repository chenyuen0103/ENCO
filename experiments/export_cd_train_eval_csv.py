#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any

from generate_prompts_names_only import iter_names_only_prompts_in_memory


def _iter_prompt_rows_for_config(
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
    thinking_tags: bool,
):
    is_names_only = obs_per_prompt == 0 and int_per_combo == 0
    if is_names_only:
        return iter_names_only_prompts_in_memory(
            bif_file=bif_file,
            num_prompts=num_prompts,
            seed=seed,
            col_order=col_order,
            anonymize=anonymize,
            causal_rules=causal_rules,
            thinking_tags=thinking_tags,
        )

    from generate_prompts import iter_prompts_in_memory

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
        thinking_tags=thinking_tags,
    )


def _load_configs(config_file: Path) -> list[dict[str, Any]]:
    payload = json.loads(config_file.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        configs = payload.get("configs", [])
    elif isinstance(payload, list):
        configs = payload
    else:
        raise ValueError("--config-file must contain a list or an object with key 'configs'")
    if not configs:
        raise ValueError(f"No configs found in {config_file}")
    return [dict(cfg) for cfg in configs]


def _export_rows(
    *,
    bif_file: Path,
    configs: list[dict[str, Any]],
    num_prompts: int,
    seed: int,
    out_csv: Path,
    split_name: str,
) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "config_idx",
        "data_idx",
        "shuffle_idx",
        "prompt_style",
        "anonymize",
        "obs_per_prompt",
        "int_per_combo",
        "row_order",
        "col_order",
        "seed",
        "prompt_text",
        "answer",
    ]

    count = 0
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for config_idx, cfg in enumerate(configs):
            prompt_style = str(cfg.get("style", cfg.get("prompt_style", "")))
            if not prompt_style:
                raise ValueError(
                    f"Config {config_idx} missing required 'style'/'prompt_style': {cfg}"
                )
            anonymize = bool(cfg.get("anonymize", False))
            obs_per_prompt = int(cfg["obs_per_prompt"])
            int_per_combo = int(cfg["int_per_combo"])
            row_order = str(cfg.get("row_order", "random"))
            col_order = str(cfg.get("col_order", "original"))
            shuffles_per_graph = int(cfg.get("shuffles_per_graph", 1))

            _base_name, answer_obj, prompt_iter = _iter_prompt_rows_for_config(
                bif_file=str(bif_file),
                num_prompts=num_prompts,
                shuffles_per_graph=shuffles_per_graph,
                seed=seed,
                prompt_style=prompt_style,
                obs_per_prompt=obs_per_prompt,
                int_per_combo=int_per_combo,
                row_order=row_order,
                col_order=col_order,
                anonymize=anonymize,
                causal_rules=bool(cfg.get("causal_rules", False)),
                give_steps=bool(cfg.get("give_steps", False)),
                def_int=bool(cfg.get("def_int", False)),
                intervene_vars=str(cfg.get("intervene_vars", "all")),
                thinking_tags=bool(cfg.get("thinking_tags", True)),
            )

            answer_json = json.dumps(answer_obj, ensure_ascii=False)
            for row in prompt_iter:
                writer.writerow(
                    {
                        "split": split_name,
                        "config_idx": config_idx,
                        "data_idx": int(row.get("data_idx", 0)),
                        "shuffle_idx": int(row.get("shuffle_idx", 0)),
                        "prompt_style": prompt_style,
                        "anonymize": int(anonymize),
                        "obs_per_prompt": obs_per_prompt,
                        "int_per_combo": int_per_combo,
                        "row_order": row_order,
                        "col_order": col_order,
                        "seed": seed,
                        "prompt_text": row["prompt_text"],
                        "answer": answer_json,
                    }
                )
                count += 1

    return count


def _load_prompt_set(csv_path: Path) -> set[str]:
    prompts: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.add(str(row.get("prompt_text", "")))
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export leak-free causal-discovery train/eval CSVs from an in-memory config file."
    )
    ap.add_argument("--bif-file", type=Path, required=True)
    ap.add_argument("--config-file", type=Path, required=True)
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--train-seed", type=int, required=True)
    ap.add_argument("--eval-seed", type=int, required=True)
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--eval-csv", type=Path, required=True)
    args = ap.parse_args()

    bif_file = args.bif_file.resolve(strict=True)
    config_file = args.config_file.resolve(strict=True)
    configs = _load_configs(config_file)

    n_train = _export_rows(
        bif_file=bif_file,
        configs=configs,
        num_prompts=int(args.num_prompts),
        seed=int(args.train_seed),
        out_csv=args.train_csv.resolve(),
        split_name="train",
    )
    n_eval = _export_rows(
        bif_file=bif_file,
        configs=configs,
        num_prompts=int(args.num_prompts),
        seed=int(args.eval_seed),
        out_csv=args.eval_csv.resolve(),
        split_name="eval",
    )

    train_prompts = _load_prompt_set(args.train_csv.resolve())
    eval_prompts = _load_prompt_set(args.eval_csv.resolve())
    overlap = train_prompts & eval_prompts
    if overlap:
        raise SystemExit(
            f"Train/eval prompt overlap detected: {len(overlap)} duplicated prompt_text rows. "
            "Choose different seeds or configs."
        )

    print(
        json.dumps(
            {
                "train_rows": n_train,
                "eval_rows": n_eval,
                "train_csv": str(args.train_csv.resolve()),
                "eval_csv": str(args.eval_csv.resolve()),
                "prompt_overlap": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
