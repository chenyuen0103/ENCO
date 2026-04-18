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

    # Compatibility fallback during migration from prompts/experiment1/<dataset>
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
    allowed_set = set(allowed)
    return value in allowed_set


def main() -> None:
    ap = argparse.ArgumentParser(description="Run query_gemini.py over generated prompt CSVs.")
    ap.add_argument(
        "--base-root",
        default="prompts",
        help="Root directory containing dataset subfolders (default: prompts).",
    )
    ap.add_argument(
        "--dataset",
        default=None,
        help="Dataset subfolder under base-root (e.g., asia). If omitted and multiple exist, exits with choices.",
    )
    ap.add_argument(
        "--pattern",
        default="prompts_obs*_int*_shuf*.csv",
        help="Filename glob to discover prompt CSVs under the dataset folder.",
    )
    ap.add_argument(
        "--obs",
        type=int,
        action="append",
        default=None,
        help="Filter to a specific observational size (repeatable).",
    )
    ap.add_argument(
        "--intval",
        type=int,
        action="append",
        default=None,
        help="Filter to a specific interventional size (repeatable).",
    )
    ap.add_argument(
        "--shuf",
        type=int,
        action="append",
        default=None,
        help="Filter to a specific shuffle count (repeatable).",
    )
    ap.add_argument(
        "--skip-obs-int-zero",
        action="store_true",
        help="Skip any CSV whose name encodes obs=0 and int=0 (default: True).",
    )
    ap.set_defaults(skip_obs_int_zero=True)

    ap.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model to run (repeatable).",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (passed to query_gemini.py).",
    )
    ap.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "openai", "gemini", "vllm"],
        help=(
            "Provider passed to query_gemini.py. "
            "Use 'openai'/'gemini' for API backends, or 'vllm' for local vLLM."
        ),
    )
    ap.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=None,
        help="Optional vLLM tensor parallel size (number of GPUs).",
    )
    ap.add_argument(
        "--vllm-dtype",
        default=None,
        help="Optional vLLM dtype (auto, float16/fp16, bfloat16/bf16, float32/fp32).",
    )
    ap.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Optional vLLM max_model_len.",
    )
    ap.add_argument(
        "--vllm-gpu-mem-util",
        type=float,
        default=None,
        help="Optional vLLM gpu_memory_utilization (0.0-1.0).",
    )
    ap.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="Pass --vllm-enforce-eager to vLLM backend.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Pass --overwrite to query_gemini.py.")
    ap.add_argument("--dry-run", action="store_true", help="List matching CSVs without running.")

    args = ap.parse_args()
    if not args.model:
        args.model = ["gpt-5-mini"]

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
            raise SystemExit(
                f"Please pass --dataset. Found datasets under {base_root_candidates[0]}: {ds_display}"
            )
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

    if args.provider == "vllm":
        query_script = (Path(__file__).parent / "query_vllm.py").resolve()
    else:
        query_script = (Path(__file__).parent / "query_gemini.py").resolve()
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
                "--provider",
                args.provider,
            ]
            if args.vllm_tensor_parallel_size is not None:
                cmd.extend(["--vllm-tensor-parallel-size", str(args.vllm_tensor_parallel_size)])
            if args.vllm_dtype is not None:
                cmd.extend(["--vllm-dtype", args.vllm_dtype])
            if args.vllm_max_model_len is not None:
                cmd.extend(["--vllm-max-model-len", str(args.vllm_max_model_len)])
            if args.vllm_gpu_mem_util is not None:
                cmd.extend(["--vllm-gpu-mem-util", str(args.vllm_gpu_mem_util)])
            if args.vllm_enforce_eager:
                cmd.append("--vllm-enforce-eager")
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
