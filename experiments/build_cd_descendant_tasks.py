#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from generate_prompts import (
    format_prompt_descendants_no_data,
    format_prompt_descendants_summary,
    iter_prompts_in_memory,
)


DEFAULT_CURRICULUM: List[Dict[str, Any]] = [
    {
        "name": "stage_1_named_obs50_int10",
        "obs_per_prompt": 50,
        "int_per_combo": 10,
        "anonymize": False,
        "num_prompts": 32,
        "shuffles_per_graph": 1,
    },
    {
        "name": "stage_2_named_obs100_int10",
        "obs_per_prompt": 100,
        "int_per_combo": 10,
        "anonymize": False,
        "num_prompts": 32,
        "shuffles_per_graph": 1,
    },
    {
        "name": "stage_3_named_obs100_int50",
        "obs_per_prompt": 100,
        "int_per_combo": 50,
        "anonymize": False,
        "num_prompts": 32,
        "shuffles_per_graph": 1,
    },
    {
        "name": "stage_4_anon_obs100_int50",
        "obs_per_prompt": 100,
        "int_per_combo": 50,
        "anonymize": True,
        "num_prompts": 32,
        "shuffles_per_graph": 1,
    },
    {
        "name": "stage_5_named_obs200_int100",
        "obs_per_prompt": 200,
        "int_per_combo": 100,
        "anonymize": False,
        "num_prompts": 48,
        "shuffles_per_graph": 1,
    },
]


def _set_csv_field_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10_000_000)


def _descendants_from_adj(adj: List[List[int]], src_idx: int) -> List[int]:
    seen = set()
    stack = [src_idx]
    while stack:
        u = stack.pop()
        for v, has_edge in enumerate(adj[u]):
            if int(has_edge) != 1 or v in seen or v == src_idx:
                continue
            seen.add(v)
            stack.append(v)
    return sorted(seen)


def _load_curriculum(path: Path | None) -> List[Dict[str, Any]]:
    if path is None:
        return list(DEFAULT_CURRICULUM)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        stages = payload.get("stages")
    else:
        stages = payload
    if not isinstance(stages, list) or not stages:
        raise ValueError("Curriculum file must contain a JSON list or an object with key 'stages'.")
    return [dict(stage) for stage in stages]


def _stage_slug(idx: int, stage: Dict[str, Any]) -> str:
    name = str(stage.get("name") or "").strip()
    if name:
        return name
    obs_n = int(stage.get("obs_per_prompt", 0))
    int_n = int(stage.get("int_per_combo", 0))
    anon = "anon" if bool(stage.get("anonymize", False)) else "named"
    return f"stage_{idx}_{anon}_obs{obs_n}_int{int_n}"


def _enumerate_no_data_interventions(
    *,
    variables: List[str],
    state_names: List[List[str]] | None,
    intervene_vars: str,
) -> List[tuple[str, str]]:
    if not variables:
        return []

    if intervene_vars.lower() in {"", "none"}:
        return []
    if intervene_vars.lower() == "all":
        targets = list(variables)
    else:
        requested = {s.strip() for s in intervene_vars.split(",") if s.strip()}
        targets = [v for v in variables if v in requested]

    name_to_idx = {v: i for i, v in enumerate(variables)}
    pairs: List[tuple[str, str]] = []
    for target in targets:
        idx = name_to_idx[target]
        values = list((state_names[idx] if state_names and idx < len(state_names) else []) or [])
        if not values:
            values = ["0", "1"]
        for value in values:
            pairs.append((target, str(value)))
    return pairs


def _build_stage_rows(
    *,
    bif_file: str,
    stage: Dict[str, Any],
    default_num_prompts: int,
    seed: int,
    default_intervene_vars: str,
    default_causal_rules: bool,
    default_def_int: bool,
) -> List[Dict[str, Any]]:
    num_prompts = int(stage.get("num_prompts", default_num_prompts))
    shuffles_per_graph = int(stage.get("shuffles_per_graph", 1))
    stage_seed = int(stage.get("seed", seed))
    row_order = str(stage.get("row_order", "random"))
    col_order = str(stage.get("col_order", "original"))
    intervene_vars = str(stage.get("intervene_vars", default_intervene_vars))
    causal_rules = bool(stage.get("causal_rules", default_causal_rules))
    def_int = bool(stage.get("def_int", default_def_int))
    anonymize = bool(stage.get("anonymize", False))
    no_data = bool(stage.get("no_data", False))
    obs_per_prompt = int(stage.get("obs_per_prompt", 0))
    int_per_combo = int(stage.get("int_per_combo", 0))

    if not no_data and int_per_combo <= 0:
        raise ValueError("Each descendant-task stage must use int_per_combo > 0.")

    _, answer_obj, prompt_iter = iter_prompts_in_memory(
        bif_file=bif_file,
        num_prompts=num_prompts,
        shuffles_per_graph=shuffles_per_graph,
        seed=stage_seed,
        prompt_style="summary",
        obs_per_prompt=obs_per_prompt,
        int_per_combo=int_per_combo,
        row_order=row_order,
        col_order=col_order,
        anonymize=anonymize,
        causal_rules=causal_rules,
        give_steps=False,
        def_int=def_int,
        intervene_vars=intervene_vars,
    )

    variables = [str(v) for v in answer_obj["variables"]]
    adj = answer_obj["adjacency_matrix"]
    descendants_map = {
        variables[i]: [variables[j] for j in _descendants_from_adj(adj, i)]
        for i in range(len(variables))
    }

    rows: List[Dict[str, Any]] = []
    for item in prompt_iter:
        item_variables = [str(v) for v in item.get("variables", variables)]
        dataset_name = str(item.get("dataset_name", Path(bif_file).stem))
        obs_rows_num = list(item.get("obs_rows_num") or [])
        int_groups_num = item.get("int_groups_num") or {}
        state_names = item.get("state_names") or None
        if no_data:
            intervention_pairs = _enumerate_no_data_interventions(
                variables=item_variables,
                state_names=state_names,
                intervene_vars=intervene_vars,
            )
            intervention_iter = [
                ((ivar, ival), [])
                for ivar, ival in intervention_pairs
            ]
        else:
            intervention_iter = sorted(
                int_groups_num.items(),
                key=lambda kv: (str(kv[0][0]), str(kv[0][1])),
            )

        for (ivar, ival), intervention_rows_num in intervention_iter:
            target = str(ivar)
            descendants = descendants_map.get(target, [])
            if no_data:
                prompt_text = format_prompt_descendants_no_data(
                    item_variables,
                    dataset_name=dataset_name,
                    intervention_target=target,
                    intervention_value=str(ival),
                    state_names=state_names,
                    include_causal_rules=causal_rules,
                    include_def_int=def_int,
                    anonymize=bool(item.get("anonymize", anonymize)),
                )
            else:
                prompt_text = format_prompt_descendants_summary(
                    item_variables,
                    dataset_name=dataset_name,
                    intervention_target=target,
                    intervention_value=str(ival),
                    intervention_rows_num=list(intervention_rows_num),
                    obs_rows_num=obs_rows_num,
                    state_names=state_names,
                    include_causal_rules=causal_rules,
                    include_def_int=def_int,
                    anonymize=bool(item.get("anonymize", anonymize)),
                )
            answer = {
                "target": target,
                "descendants": descendants,
            }
            rows.append(
                {
                    "data_idx": int(item.get("data_idx", -1)),
                    "shuffle_idx": int(item.get("shuffle_idx", -1)),
                    "intervention_target": target,
                    "intervention_value": str(ival),
                    "prompt_text": prompt_text,
                    "answer": json.dumps(answer, ensure_ascii=False),
                    "obs_per_prompt": obs_per_prompt,
                    "int_per_combo": int_per_combo,
                    "anonymize": int(anonymize),
                }
            )
    return rows


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create descendant-prediction task CSVs and a curriculum manifest from a causal graph."
    )
    p.add_argument("--bif-file", type=str, required=True, help="Path to the source BIF graph.")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for stage CSVs and manifest.")
    p.add_argument(
        "--curriculum-file",
        type=str,
        default=None,
        help="Optional JSON curriculum spec. Defaults to a built-in five-stage curriculum.",
    )
    p.add_argument("--seed", type=int, default=0, help="Base random seed.")
    p.add_argument("--num-prompts", type=int, default=32, help="Default prompts per stage when not overridden.")
    p.add_argument(
        "--intervene-vars",
        type=str,
        default="all",
        help="Intervention variable mode passed to iter_prompts_in_memory (default: all).",
    )
    p.add_argument("--causal-rules", dest="causal_rules", action="store_true")
    p.add_argument("--no-causal-rules", dest="causal_rules", action="store_false")
    p.set_defaults(causal_rules=True)
    p.add_argument("--def-int", dest="def_int", action="store_true")
    p.add_argument("--no-def-int", dest="def_int", action="store_false")
    p.set_defaults(def_int=True)
    p.add_argument(
        "--no-data",
        dest="no_data",
        action="store_true",
        help="Default stage setting: emit descendant prompts without observational or interventional summaries.",
    )
    p.add_argument("--with-data", dest="no_data", action="store_false")
    p.set_defaults(no_data=False)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    _set_csv_field_limit()

    bif_path = Path(args.bif_file).resolve(strict=True)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    curriculum = _load_curriculum(Path(args.curriculum_file).resolve() if args.curriculum_file else None)

    manifest: Dict[str, Any] = {
        "task": "cd_descendants",
        "bif_file": str(bif_path),
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "stages": [],
    }

    fieldnames = [
        "data_idx",
        "shuffle_idx",
        "intervention_target",
        "intervention_value",
        "prompt_text",
        "answer",
        "obs_per_prompt",
        "int_per_combo",
        "anonymize",
    ]

    combined_rows: List[Dict[str, Any]] = []
    for idx, stage in enumerate(curriculum, start=1):
        stage_name = _stage_slug(idx, stage)
        rows = _build_stage_rows(
            bif_file=str(bif_path),
            stage={**stage, "no_data": bool(stage.get("no_data", args.no_data))},
            default_num_prompts=int(args.num_prompts),
            seed=int(args.seed),
            default_intervene_vars=str(args.intervene_vars),
            default_causal_rules=bool(args.causal_rules),
            default_def_int=bool(args.def_int),
        )
        stage_csv = out_dir / f"{stage_name}.csv"
        with stage_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        if rows:
            example_path = out_dir / f"{stage_name}_example_prompt.txt"
            example_path.write_text(rows[0]["prompt_text"] + "\n", encoding="utf-8")

        manifest["stages"].append(
            {
                "name": stage_name,
                "csv_path": str(stage_csv),
                "num_rows": len(rows),
                "config": stage,
            }
        )
        combined_rows.extend([{**row, "stage_name": stage_name} for row in rows])
        print(f"[stage] {stage_name}: wrote {len(rows)} rows -> {stage_csv}")

    combined_csv = out_dir / "all_stages.csv"
    with combined_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage_name", *fieldnames])
        writer.writeheader()
        writer.writerows(combined_rows)

    manifest["combined_csv"] = str(combined_csv)
    manifest_path = out_dir / "curriculum_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
