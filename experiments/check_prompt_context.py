#!/usr/bin/env python3
"""
Check when prompts exceed a model's context window.

Typical use: estimate matrix-prompt token counts across (obs_n, int_n) grid and
report the first configuration that exceeds the model's context.

This script generates prompts in-memory (no API calls) and counts tokens using
count_openai_tokens() from query_api.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from query_api import count_openai_tokens


_DEFAULT_CONTEXT_WINDOWS: dict[str, int] = {
    # Source: OpenAI model docs. Keep a small mapping and allow overrides via CLI.
    "gpt-5-mini": 400_000,
    "gpt-5": 400_000,
}


def _parse_ints(xs: Iterable[str]) -> list[int]:
    out: list[int] = []
    for x in xs:
        for tok in str(x).replace(",", " ").split():
            if tok.strip():
                out.append(int(tok))
    return out


def _context_window_for(model: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    key = model.split("/")[-1]
    return int(_DEFAULT_CONTEXT_WINDOWS.get(key, 128_000))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bif-file", required=True)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--context-window", type=int, default=None)
    ap.add_argument(
        "--prompt-style",
        default="matrix",
        choices=["cases", "matrix", "summary", "payload", "payload_topk"],
    )
    ap.add_argument("--anonymize", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffles-per-graph", type=int, default=1)
    ap.add_argument("--row-order", default="random", choices=["random", "sorted", "reverse"])
    ap.add_argument("--col-order", default="original", choices=["original", "reverse", "random", "topo", "reverse_topo"])
    ap.add_argument("--obs-sizes", nargs="*", default=["0", "100", "1000", "5000", "8000"])
    ap.add_argument("--int-sizes", nargs="*", default=["0", "50", "100", "200", "500"])
    ap.add_argument("--num-prompts", type=int, default=1)
    ap.add_argument("--stop-at-first-exceed", action="store_true")
    args = ap.parse_args()

    ctx = _context_window_for(args.model, args.context_window)
    obs_sizes = _parse_ints(args.obs_sizes)
    int_sizes = _parse_ints(args.int_sizes)

    try:
        from generate_prompts import iter_prompts_in_memory
    except Exception as e:
        raise SystemExit(
            "Failed to import generate_prompts.iter_prompts_in_memory (likely missing torch). "
            "Run this inside the same environment you use for prompt generation."
        ) from e

    dataset = Path(args.bif_file).stem
    print(json.dumps({"dataset": dataset, "model": args.model, "context_window": ctx, "prompt_style": args.prompt_style}))

    exceeded_any = False
    for obs_n in obs_sizes:
        for int_n in int_sizes:
            if args.prompt_style in {"payload", "payload_topk"} and obs_n == 0 and int_n > 0:
                # payload requires observational rows
                continue
            base_name, _, it = iter_prompts_in_memory(
                bif_file=args.bif_file,
                num_prompts=int(args.num_prompts),
                shuffles_per_graph=int(args.shuffles_per_graph),
                seed=int(args.seed),
                prompt_style=str(args.prompt_style),
                obs_per_prompt=int(obs_n),
                int_per_combo=int(int_n),
                row_order=str(args.row_order),
                col_order=str(args.col_order),
                anonymize=bool(args.anonymize),
                causal_rules=False,
                give_steps=False,
                def_int=False,
                intervene_vars="all" if int(int_n) > 0 else "none",
            )
            try:
                first = next(it)
            except StopIteration:
                continue
            prompt = first["prompt_text"]
            toks = int(count_openai_tokens(args.model, prompt))
            over = toks > ctx
            exceeded_any = exceeded_any or over
            status = "EXCEEDS" if over else "ok"
            print(f"{status}\tobs={obs_n}\tint={int_n}\ttokens={toks}\tbase={base_name}")
            if over and args.stop_at_first_exceed:
                return 2

    return 2 if exceeded_any else 0


if __name__ == "__main__":
    raise SystemExit(main())
