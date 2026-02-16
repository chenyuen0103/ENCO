#!/usr/bin/env python3
"""
Convert an existing `summary_hist_rows` prompt (which contains full joint histograms)
into other prompt styles without re-sampling from the Bayesian network.

This is useful in environments where torch/graph sampling is unavailable, but you
still want to compare LLM performance across prompt formats on the *same* underlying
empirical data.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


_DATA_SHUF_RE = re.compile(r"_data(?P<data>\d+)_shuf(?P<shuf>\d+)\.txt$", re.IGNORECASE)


def _parse_dataset_name(lines: List[str]) -> str:
    # e.g., "network named sachs."
    for line in lines[:20]:
        m = re.search(r"network named\s+([A-Za-z0-9_\-]+)", line)
        if m:
            return m.group(1)
    return "unknown"


def _parse_flags(text: str) -> Tuple[bool, bool, bool]:
    include_def_int = ("--- INTERVENTION NOTES ---" in text)
    include_causal_rules = ("--- CAUSAL INFERENCE REMINDERS ---" in text)
    include_give_steps = ("You may follow these steps silently" in text)
    return include_causal_rules, include_give_steps, include_def_int


def _parse_variables_and_states(lines: List[str]) -> Tuple[List[str], List[List[str]]]:
    variables: List[str] = []
    state_names: List[List[str]] = []
    in_vars = False
    for line in lines:
        if line.strip() == "--- VARIABLES (ORDER MATTERS) ---":
            in_vars = True
            continue
        if in_vars and line.strip().startswith("--- "):
            break
        if not in_vars:
            continue

        m = re.match(r"^\s*(\d+):\s*([^\s]+)(?:\s+states=(\{.*\}))?\s*$", line)
        if not m:
            continue
        idx = int(m.group(1))
        name = m.group(2)
        states_raw = m.group(3)

        # Ensure list sizes.
        while len(variables) <= idx:
            variables.append("")
            state_names.append([])
        variables[idx] = name

        if states_raw:
            mapping = json.loads(states_raw)
            ordered = []
            for k in sorted(mapping.keys(), key=lambda x: int(x)):
                ordered.append(str(mapping[k]))
            state_names[idx] = ordered

    variables = [v for v in variables if v]
    if len(state_names) < len(variables):
        state_names.extend([[] for _ in range(len(variables) - len(state_names))])
    state_names = state_names[: len(variables)]
    return variables, state_names


def _parse_obs_hist(lines: List[str]) -> List[Tuple[List[int], int]]:
    for line in lines:
        if line.startswith("obs_hist="):
            payload = line.split("=", 1)[1].strip()
            data = json.loads(payload)
            out: List[Tuple[List[int], int]] = []
            for x, c in data:
                out.append(([int(v) for v in x], int(c)))
            return out
    raise ValueError("Could not find obs_hist=... in prompt text.")


def _parse_int_groups(lines: List[str]) -> Dict[Tuple[str, str], List[Tuple[List[int], int]]]:
    out: Dict[Tuple[str, str], List[Tuple[List[int], int]]] = {}
    for line in lines:
        if not line.startswith("do("):
            continue
        m = re.match(r"^do\((?P<ivar>[^=]+)=(?P<ival>[^\)]+)\):\s*(?P<json>\{.*\})\s*$", line)
        if not m:
            continue
        ivar = m.group("ivar").strip()
        ival = m.group("ival").strip()
        payload = json.loads(m.group("json"))
        hist = payload.get("hist", [])
        items: List[Tuple[List[int], int]] = []
        for x, c in hist:
            items.append(([int(v) for v in x], int(c)))
        out[(ivar, str(ival))] = items
    return out


def _expand_hist(items: List[Tuple[List[int], int]]) -> List[List[float]]:
    rows: List[List[float]] = []
    for x, c in items:
        xf = [float(v) for v in x]
        for _ in range(int(c)):
            rows.append(xf)
    return rows


def _infer_prompt_csv_and_answer_path(in_prompt_txt: Path) -> Tuple[Path, str]:
    # prompts/.../prompt_txt/<BASE>_data0_shuf0.txt  => prompts/.../<BASE>.csv
    stem = in_prompt_txt.name
    m = _DATA_SHUF_RE.search(stem)
    if not m:
        raise ValueError(f"Input prompt_txt filename does not match *_dataN_shufM.txt: {in_prompt_txt}")
    base = stem[: m.start()]  # strip suffix including _data..._shuf...txt
    prompts_dir = in_prompt_txt.parent.parent
    in_csv = prompts_dir / f"{base}.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Could not find prompt CSV next to prompt_txt: {in_csv}")
    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Prompt CSV is empty: {in_csv}")
    answer_path = rows[0].get("answer_path")
    if not answer_path:
        raise ValueError(f"Prompt CSV missing answer_path: {in_csv}")
    return in_csv, answer_path


def _derive_out_base(in_base: str, out_style: str) -> str:
    # e.g. prompts_obs200_int50_shuf1_summary_hist_rows -> prompts_obs200_int50_shuf1_summary
    if in_base.endswith("_summary_hist_rows"):
        return in_base[: -len("_summary_hist_rows")] + f"_{out_style}"
    # fallback: append style tag
    if in_base.endswith(f"_{out_style}"):
        return in_base
    return f"{in_base}_{out_style}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert summary_hist_rows prompt_txt into other prompt styles.")
    ap.add_argument("--in-prompt-txt", required=True, help="Path to *_summary_hist_rows_data*_shuf*.txt")
    ap.add_argument(
        "--out-styles",
        default="summary,summary_joint",
        help="Comma-separated list of output styles (default: summary,summary_joint).",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for prompt CSVs (default: sibling of input prompt_txt dir).",
    )
    args = ap.parse_args()

    in_prompt_txt = Path(args.in_prompt_txt)
    text = in_prompt_txt.read_text(encoding="utf-8")
    lines = text.splitlines()

    dataset_name = _parse_dataset_name(lines)
    include_causal_rules, include_give_steps, include_def_int = _parse_flags(text)
    variables, state_names = _parse_variables_and_states(lines)

    obs_hist = _parse_obs_hist(lines)
    int_groups_hist = _parse_int_groups(lines)
    obs_rows_num = _expand_hist(obs_hist)
    int_groups_num: Dict[Tuple[str, str], List[List[float]]] = {
        k: _expand_hist(v) for k, v in int_groups_hist.items()
    }

    # Determine base name / indices from filename
    m = _DATA_SHUF_RE.search(in_prompt_txt.name)
    assert m is not None
    data_idx = int(m.group("data"))
    shuf_idx = int(m.group("shuf"))
    in_base = in_prompt_txt.name[: m.start()]

    _, answer_path = _infer_prompt_csv_and_answer_path(in_prompt_txt)

    out_styles = [s.strip() for s in str(args.out_styles).split(",") if s.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else in_prompt_txt.parent.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prompt_txt_dir = out_dir / "prompt_txt"
    out_prompt_txt_dir.mkdir(parents=True, exist_ok=True)

    # Import formatting functions (requires only numpy/json; graph sampling is not needed).
    from generate_prompts import (  # type: ignore
        format_prompt_summary_full_joint,
        format_prompt_summary_stats,
    )

    for style in out_styles:
        if style not in {"summary", "summary_joint"}:
            raise ValueError(f"Unsupported out style: {style}")
        out_base = _derive_out_base(in_base, style)

        if style == "summary":
            prompt_text = format_prompt_summary_stats(
                variables,
                dataset_name=dataset_name,
                obs_rows_num=obs_rows_num,
                int_groups_num=int_groups_num,
                include_causal_rules=include_causal_rules,
                include_give_steps=include_give_steps,
                include_def_int=include_def_int,
            )
        else:
            prompt_text = format_prompt_summary_full_joint(
                variables,
                dataset_name=dataset_name,
                obs_rows_num=obs_rows_num,
                int_groups_num=int_groups_num,
                state_names=state_names if any(state_names) else None,
                include_causal_rules=include_causal_rules,
                include_give_steps=include_give_steps,
                include_def_int=include_def_int,
            )

        out_prompt_txt = out_prompt_txt_dir / f"{out_base}_data{data_idx}_shuf{shuf_idx}.txt"
        out_prompt_txt.write_text(prompt_text, encoding="utf-8")

        out_csv = out_dir / f"{out_base}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["data_idx", "shuffle_idx", "prompt_path", "answer_path", "given_edges"],
            )
            w.writeheader()
            w.writerow(
                {
                    "data_idx": data_idx,
                    "shuffle_idx": shuf_idx,
                    "prompt_path": str(out_prompt_txt.resolve()),
                    "answer_path": answer_path,
                    "given_edges": "",
                }
            )

        print(f"[ok] Wrote {style} prompt_txt: {out_prompt_txt}")
        print(f"[ok] Wrote {style} prompt CSV  : {out_csv}")


if __name__ == "__main__":
    main()
