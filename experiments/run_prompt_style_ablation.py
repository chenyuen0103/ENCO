#!/usr/bin/env python3
"""
Generate a single `summary_hist_rows` prompt and deterministically convert it into
a `summary` prompt (same underlying empirical data).

This driver always uses `--intervene-vars all` when `--int-per-combo > 0` to avoid
accidentally generating interventions for only a subset of variables.

This is useful for prompt-style ablations without needing to resample per style.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _iter_hist_rows_prompts(prompt_txt_dir: Path) -> list[Path]:
    if not prompt_txt_dir.exists():
        return []
    return sorted(prompt_txt_dir.glob("*_summary_hist_rows_data*_shuf*.txt"))


def _file_stats(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "bytes": len(text.encode("utf-8")),
        "lines": text.count("\n") + (0 if text.endswith("\n") or not text else 1),
    }


def _write_sizes_csv(out_path: Path, rows: Iterable[dict[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["style", "bytes", "lines", "path"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser(description="Prompt-style ablation driver (summary vs summary_joint).")
    ap.add_argument("--bif-file", required=True, help="Path to a .bif file (e.g. causal_graphs/.../sachs.bif).")
    ap.add_argument("--obs-per-prompt", type=int, default=200)
    ap.add_argument("--int-per-combo", type=int, default=50)
    ap.add_argument("--num-prompts", type=int, default=1)
    ap.add_argument("--shuffles-per-graph", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--anonymize", action="store_true")
    ap.add_argument("--out-dir", required=True, help="Output directory (will create prompt_txt/).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing converted prompt files.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    generate_script = repo_root / "experiments" / "generate_prompts.py"
    convert_script = repo_root / "experiments" / "convert_summary_hist_rows.py"
    if not generate_script.exists():
        raise SystemExit(f"generate script not found: {generate_script}")
    if not convert_script.exists():
        raise SystemExit(f"convert script not found: {convert_script}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intervene_vars = "all" if int(args.int_per_combo) > 0 else "none"

    # 1) Generate summary_hist_rows once (source of truth for empirical data).
    gen_cmd = [
        sys.executable,
        str(generate_script),
        "--bif-file",
        str(args.bif_file),
        "--num-prompts",
        str(args.num_prompts),
        "--shuffles-per-graph",
        str(args.shuffles_per_graph),
        "--obs-per-prompt",
        str(args.obs_per_prompt),
        "--int-per-combo",
        str(args.int_per_combo),
        "--intervene-vars",
        intervene_vars,
        "--seed",
        str(args.seed),
        "--prompt-style",
        "summary_hist_rows",
        "--out-dir",
        str(out_dir),
    ]
    if args.anonymize:
        gen_cmd.append("--anonymize")
    print("[running]", " ".join(gen_cmd))
    subprocess.run(gen_cmd, check=True)

    prompt_txt_dir = out_dir / "prompt_txt"
    hist_rows_prompts = _iter_hist_rows_prompts(prompt_txt_dir)
    if not hist_rows_prompts:
        raise SystemExit(f"No *_summary_hist_rows_data*_shuf*.txt found under: {prompt_txt_dir}")

    # 2) Convert each hist_rows prompt into summary in the same out_dir.
    for in_prompt_txt in hist_rows_prompts:
        if not args.overwrite:
            stem = in_prompt_txt.name.replace("_summary_hist_rows_", "_summary_")
            maybe_summary = prompt_txt_dir / stem
            if maybe_summary.exists():
                print(f"[skip] Already converted: {in_prompt_txt.name}")
                continue

        conv_cmd = [
            sys.executable,
            str(convert_script),
            "--in-prompt-txt",
            str(in_prompt_txt),
            "--out-styles",
            "summary",
            "--out-dir",
            str(out_dir),
        ]
        print("[running]", " ".join(conv_cmd))
        subprocess.run(conv_cmd, check=True)

    # 3) Print a compact size report.
    size_rows: list[dict[str, object]] = []
    for style in ("summary_hist_rows", "summary"):
        for p in sorted(prompt_txt_dir.glob(f"*_{style}_data*_shuf*.txt")):
            st = _file_stats(p)
            size_rows.append({"style": style, "bytes": st["bytes"], "lines": st["lines"], "path": str(p)})

    _write_sizes_csv(out_dir / "prompt_sizes.csv", size_rows)

    print("\n=== Prompt sizes ===")
    for r in size_rows:
        print(f'{r["style"]:>16}  {r["bytes"]:>8} bytes  {r["lines"]:>4} lines  {r["path"]}')
    print(f"\n[ok] Wrote: {out_dir / 'prompt_sizes.csv'}")


if __name__ == "__main__":
    main()
