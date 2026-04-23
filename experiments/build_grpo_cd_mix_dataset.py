#!/usr/bin/env python3
import argparse
import csv
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
        default="0,50,100,200,500",
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
        "--cot-hint",
        action="store_true",
        help=(
            "Legacy alias for chat-style prompt wrapping. This maps to wrapper_mode=chat "
            "and does not change the staged reasoning instructions."
        ),
    )
    args = ap.parse_args()

    graph_names = _parse_name_list(args.graph_names)
    obs_values = _parse_int_list(args.obs_values)
    int_values = _parse_int_list(args.int_values)

    out_path = args.output_csv.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "bif_file",
        "prompt_style",
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
            bif_file = (args.graphs_dir / f"{graph_name}.bif").resolve()
            if not bif_file.exists():
                raise SystemExit(f"Missing BIF file: {bif_file}")

            for obs_n in obs_values:
                for int_n in int_values:
                    if obs_n == 0 and int_n == 0 and not args.include_names_only:
                        continue

                    for answer_obj, prompt_iter in _iter_rows_for_config(
                        bif_file=bif_file,
                        prompt_style=args.prompt_style,
                        obs_per_prompt=int(obs_n),
                        int_per_combo=int(int_n),
                        num_prompts_per_config=int(args.num_prompts_per_config),
                        shuffles_per_graph=int(args.shuffles_per_graph),
                        seed=int(args.seed),
                        anonymize=bool(args.anonymize),
                        wrapper_mode=("chat" if args.cot_hint else None),
                        append_format_hint=bool(args.append_format_hint),
                        col_order=args.col_order,
                        col_perms=int(args.col_perms),
                    ):
                        answer_json = json.dumps(answer_obj, ensure_ascii=False)
                        for row in prompt_iter:
                            writer.writerow(
                                {
                                    "dataset": graph_name,
                                    "bif_file": str(bif_file),
                                    "prompt_style": args.prompt_style,
                                    "anonymize": int(bool(args.anonymize)),
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
