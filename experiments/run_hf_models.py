#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


_PROMPT_RE = re.compile(r"prompts_obs(?P<obs>\d+)_int(?P<int>\d+)_shuf(?P<shuf>\d+)", re.IGNORECASE)


def _infer_datasets(base_root: Path) -> list[str]:
    if not base_root.exists():
        return []
    return sorted([p.name for p in base_root.iterdir() if p.is_dir() and not p.name.startswith(".")])


def _iter_prompt_csvs(search_root: Path, pattern: str) -> list[Path]:
    if not search_root.exists():
        return []
    return sorted([p for p in search_root.rglob(pattern) if p.is_file()])


def _resolve_base_roots(base_root_arg: Path) -> list[Path]:
    if base_root_arg.is_absolute():
        return [base_root_arg]

    candidate1 = (Path(__file__).parent / base_root_arg).resolve()
    candidate2 = (Path(__file__).parent.parent / base_root_arg).resolve()
    candidates = [candidate1, candidate2]

    if base_root_arg == Path("prompts"):
        candidates.extend([
            (Path(__file__).parent / "prompts" / "experiment1").resolve(),
            (Path(__file__).parent.parent / "experiments" / "prompts" / "experiment1").resolve(),
        ])

    seen: set[Path] = set()
    out: list[Path] = []
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _parse_obs_int_shuf(path: Path) -> Optional[tuple[int, int, int]]:
    m = _PROMPT_RE.search(path.stem)
    if not m:
        return None
    return (int(m.group("obs")), int(m.group("int")), int(m.group("shuf")))


def _matches_filter(value: int, allowed: Optional[Iterable[int]]) -> bool:
    if not allowed:
        return True
    return value in set(allowed)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run query_hf.py over generated prompt CSVs.")
    ap.add_argument("--base-root", default="prompts")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--pattern", default="prompts_obs*_int*_shuf*.csv")
    ap.add_argument("--obs", type=int, action="append", default=None)
    ap.add_argument("--intval", type=int, action="append", default=None)
    ap.add_argument("--shuf", type=int, action="append", default=None)
    ap.add_argument("--skip-obs-int-zero", action="store_true")
    ap.set_defaults(skip_obs_int_zero=True)

    ap.add_argument("--model", action="append", default=None, help="HF model to run (repeatable).")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--hf-device-map", default="auto")
    ap.add_argument("--hf-dtype", default="auto")
    ap.add_argument("--hf-batch-size", type=int, default=1)
    ap.add_argument("--hf-trust-remote-code", dest="hf_trust_remote_code", action="store_true")
    ap.add_argument("--no-hf-trust-remote-code", dest="hf_trust_remote_code", action="store_false")
    ap.set_defaults(hf_trust_remote_code=True)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    if not args.model:
        args.model = ["Qwen/Qwen3-4B-Thinking-2507"]

    base_root_candidates = _resolve_base_roots(Path(args.base_root))
    datasets: list[str] = []
    for root in base_root_candidates:
        ds = _infer_datasets(root)
        if ds:
            datasets = ds
            break
    if args.dataset is None:
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            ds_display = ", ".join(datasets) if datasets else "(none found)"
            raise SystemExit(f"Please pass --dataset. Found datasets under {base_root_candidates[0]}: {ds_display}")
    else:
        dataset = args.dataset

    search_roots = [root / dataset for root in base_root_candidates]
    csv_paths = sorted({p for root in search_roots for p in _iter_prompt_csvs(root, args.pattern)})

    selected: list[Path] = []
    skipped_by_filter = 0
    skipped_by_rule = 0
    for p in csv_paths:
        parsed = _parse_obs_int_shuf(p)
        if parsed is None:
            skipped_by_filter += 1
            continue
        obs, intval, shuf = parsed
        if args.skip_obs_int_zero and obs == 0 and intval == 0:
            skipped_by_rule += 1
            continue
        if not _matches_filter(obs, args.obs):
            skipped_by_filter += 1
            continue
        if not _matches_filter(intval, args.intval):
            skipped_by_filter += 1
            continue
        if not _matches_filter(shuf, args.shuf):
            skipped_by_filter += 1
            continue
        selected.append(p)

    print(f"[info] Dataset: {dataset}")
    print(f"[info] Search roots: {', '.join(str(p) for p in search_roots)}")
    print(f"[info] Discovered CSVs: {len(csv_paths)} (pattern={args.pattern})")
    print(f"[info] Selected CSVs: {len(selected)} (skipped_by_filter={skipped_by_filter}, skipped_by_rule={skipped_by_rule})")

    if args.dry_run:
        for p in selected:
            print(p)
        return

    query_script = (Path(__file__).parent / "query_hf.py").resolve()
    if not query_script.exists():
        raise SystemExit(f"query script not found: {query_script}")

    total_ran = 0
    for csv_path in selected:
        for model in args.model:
            cmd = [
                sys.executable,
                str(query_script),
                "--csv",
                str(csv_path),
                "--model",
                model,
                "--temperature",
                str(args.temperature),
                "--hf-device-map",
                args.hf_device_map,
                "--hf-dtype",
                args.hf_dtype,
                "--hf-batch-size",
                str(args.hf_batch_size),
            ]
            if args.max_new_tokens is not None:
                cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
            if args.hf_trust_remote_code:
                cmd.append("--hf-trust-remote-code")
            else:
                cmd.append("--no-hf-trust-remote-code")
            if args.overwrite:
                cmd.append("--overwrite")
            print("\n[running]", " ".join(cmd))
            subprocess.run(cmd, check=True)
            total_ran += 1

    print("\n=== Summary ===")
    print(f"CSV files discovered     : {len(csv_paths)}")
    print(f"CSV files selected       : {len(selected)}")
    print(f"Skipped by rule          : {skipped_by_rule}")
    print(f"Skipped by filter/parse  : {skipped_by_filter}")
    print(f"Model runs executed      : {total_ran}")


if __name__ == "__main__":
    main()
