#!/usr/bin/env python3
"""
Status helper for Experiment 1 runs.

Given the grid definition (obs/int sizes, prompt styles, anonymization) and expected
replicates (num_prompts, shuffles_per_graph), this script reports which response CSVs:
  - are done (all expected rows completed with non-error raw_response + non-empty prediction),
  - are in progress (file exists but some rows missing/errored), or
  - have not started (file missing).

Designed to work for both:
  - in-memory runs (run from experiments/; responses live in experiments/responses/<dataset>/)
  - file-based runs (same output location when invoked from experiments/)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class FileStatus:
    path: Path
    expected_rows: int
    completed_rows: int
    error_rows: int
    missing_rows: int
    eval_done: bool

    @property
    def state(self) -> str:
        if not self.path.exists():
            return "not_started"
        if self.missing_rows == 0 and self.error_rows == 0:
            return "done"
        return "in_progress"


def _parse_int_list(values: list[str]) -> list[int]:
    out: list[int] = []
    for part in values:
        for tok in part.replace(",", " ").split():
            if tok:
                out.append(int(tok))
    return out


def _safe_model_tag(model: str) -> str:
    tag = model.split("/")[-1]
    for ch in (":", " "):
        tag = tag.replace(ch, "_")
    return tag


def _base_name(
    *,
    obs_n: int,
    int_n: int,
    shuf_n: int,
    prompt_style: str,
    anonymize: bool,
) -> str:
    # Tags order matches experiments/generate_prompts.py iter_prompts_in_memory()
    tags: list[str] = []
    if anonymize:
        tags.append("anon")
    if prompt_style in {"matrix", "summary"}:
        tags.append(prompt_style)
    extra_suffix = ("_" + "_".join(tags)) if tags else ""
    return f"prompts_obs{obs_n}_int{int_n}_shuf{shuf_n}{extra_suffix}"


def _response_csv_path(
    *,
    responses_dir: Path,
    base_name: str,
    model: str,
) -> Path:
    base = base_name.replace("prompts", "responses", 1)
    stem = Path(base).stem
    safe_model = _safe_model_tag(model)
    if safe_model not in stem:
        stem = f"{stem}_{safe_model}"
    return responses_dir / f"{stem}.csv"


def _compute_file_status(path: Path, *, num_prompts: int, shuf_n: int) -> FileStatus:
    expected_keys = {(i, j) for i in range(num_prompts) for j in range(shuf_n)}
    expected_rows = len(expected_keys)

    if not path.exists():
        return FileStatus(
            path=path,
            expected_rows=expected_rows,
            completed_rows=0,
            error_rows=0,
            missing_rows=expected_rows,
            eval_done=False,
        )

    completed_keys: set[tuple[int, int]] = set()
    error_rows = 0

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                di = int(row.get("data_idx", -1))
                si = int(row.get("shuffle_idx", -1))
            except Exception:
                continue
            raw = (row.get("raw_response") or "").lstrip()
            pred = (row.get("prediction") or "").strip()
            is_error = raw.startswith("[ERROR]")
            if is_error:
                error_rows += 1
                continue
            if pred:
                completed_keys.add((di, si))

    missing_keys = expected_keys - completed_keys
    eval_done = path.with_suffix(path.suffix + ".summary.json").exists()
    return FileStatus(
        path=path,
        expected_rows=expected_rows,
        completed_rows=len(completed_keys),
        error_rows=error_rows,
        missing_rows=len(missing_keys),
        eval_done=eval_done,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--responses-dir",
        type=Path,
        default=None,
        help="Override responses directory (default: experiments/responses/<dataset>/).",
    )
    ap.add_argument("--model", action="append", default=["gpt-5-mini"], help="Repeatable.")
    ap.add_argument("--styles", nargs="*", default=["summary"], help='Any of: cases, matrix, summary.')
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--shuffles-per-graph", type=int, default=1)
    ap.add_argument("--obs-sizes", nargs="*", default=["0", "100", "1000", "5000", "8000"])
    ap.add_argument("--int-sizes", nargs="*", default=["0", "50", "100", "200", "500"])
    ap.add_argument("--include-anon", action="store_true", help="Also include anonymized runs.")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON to stdout.")
    args = ap.parse_args()

    experiments_dir = Path(__file__).resolve().parent
    responses_dir = args.responses_dir or (experiments_dir / "responses" / args.dataset)

    obs_sizes = _parse_int_list(list(args.obs_sizes))
    int_sizes = _parse_int_list(list(args.int_sizes))
    styles = [s.strip().lower() for s in args.styles if s.strip()]
    models = list(args.model or [])
    if not models:
        models = ["gpt-5-mini"]

    anon_opts = [False, True] if args.include_anon else [False]

    rows: list[dict] = []
    totals = {"done": 0, "in_progress": 0, "not_started": 0}

    for model in models:
        for style in styles:
            for anon in anon_opts:
                for obs_n in obs_sizes:
                    for int_n in int_sizes:
                        # Match the experiment constraint: skip the empty config (names-only is a different generator).
                        if obs_n == 0 and int_n == 0:
                            continue
                        base = _base_name(
                            obs_n=obs_n,
                            int_n=int_n,
                            shuf_n=int(args.shuffles_per_graph),
                            prompt_style=style,
                            anonymize=anon,
                        )
                        csv_path = _response_csv_path(responses_dir=responses_dir, base_name=base, model=model)
                        st = _compute_file_status(
                            csv_path,
                            num_prompts=int(args.num_prompts),
                            shuf_n=int(args.shuffles_per_graph),
                        )
                        totals[st.state] += 1
                        rows.append(
                            {
                                "dataset": args.dataset,
                                "model": _safe_model_tag(model),
                                "prompt_style": style,
                                "anonymize": int(anon),
                                "obs_n": obs_n,
                                "int_n": int_n,
                                "shuf_n": int(args.shuffles_per_graph),
                                "path": str(csv_path),
                                "state": st.state,
                                "expected_rows": st.expected_rows,
                                "completed_rows": st.completed_rows,
                                "missing_rows": st.missing_rows,
                                "error_rows": st.error_rows,
                                "eval_done": int(st.eval_done),
                            }
                        )

    if args.json:
        print(json.dumps({"totals": totals, "rows": rows}, ensure_ascii=False, indent=2))
        return 0

    print(f"[status] dataset={args.dataset} responses_dir={responses_dir}")
    print(f"[status] totals: done={totals['done']} in_progress={totals['in_progress']} not_started={totals['not_started']}")
    # Print only non-done by default for quick debugging.
    for r in rows:
        if r["state"] == "done":
            continue
        print(
            f"- {r['state']}: obs={r['obs_n']} int={r['int_n']} style={r['prompt_style']} anon={r['anonymize']} "
            f"rows={r['completed_rows']}/{r['expected_rows']} err_rows={r['error_rows']} eval={bool(r['eval_done'])} "
            f"path={r['path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

