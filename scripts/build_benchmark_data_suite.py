#!/usr/bin/env python3
"""Build the paper benchmark-data CSV suite.

This is a convenience wrapper around ``scripts/build_benchmark_data.py``.  It
keeps the real-graph benchmark data and the recommended controlled synthetic
challenge family in one reproducible command.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_benchmark_data.py"
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "eval_configs.json"


@dataclass(frozen=True)
class DatasetJob:
    dataset: str
    graph_path: Path
    output_group: str


SMALL_REAL_GRAPHS = [
    ("cancer", "causal_graphs/real_data/small_graphs/cancer.bif"),
    ("earthquake", "causal_graphs/real_data/small_graphs/earthquake.bif"),
    ("asia", "causal_graphs/real_data/small_graphs/asia.bif"),
    ("sachs", "causal_graphs/real_data/small_graphs/sachs.bif"),
    ("child", "causal_graphs/real_data/small_graphs/child.bif"),
]
LARGE_REAL_GRAPHS = [
    ("alarm", "causal_graphs/real_data/small_graphs/alarm.bif"),
    ("diabetes", "causal_graphs/real_data/large_graphs/diabetes.bif"),
    ("pigs", "causal_graphs/real_data/large_graphs/pigs.bif"),
]
DEFAULT_SYNTHETIC_TYPES = ["chain", "collider", "bidiag", "jungle", "random", "full"]


def _split_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build real and controlled-synthetic benchmark-data CSVs.")
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "benchmark_data" / "graphs")
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config-indexes", default=None, help="Optional comma-separated config subset passed through to each build.")
    parser.add_argument("--skip-real", action="store_true", help="Do not build small/large real graph CSVs.")
    parser.add_argument("--skip-synthetic", action="store_true", help="Do not build controlled synthetic challenge CSVs.")
    parser.add_argument(
        "--synthetic-types",
        default=",".join(DEFAULT_SYNTHETIC_TYPES),
        help="Comma-separated synthetic topologies to include.",
    )
    parser.add_argument(
        "--synthetic-seeds",
        default="42",
        help="Comma-separated synthetic graph seeds to include. Default builds one representative seed per topology.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing CSVs and manifests.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def _real_jobs() -> list[DatasetJob]:
    jobs: list[DatasetJob] = []
    for dataset, graph_rel in SMALL_REAL_GRAPHS:
        jobs.append(DatasetJob(dataset=dataset, graph_path=REPO_ROOT / graph_rel, output_group="small"))
    for dataset, graph_rel in LARGE_REAL_GRAPHS:
        jobs.append(DatasetJob(dataset=dataset, graph_path=REPO_ROOT / graph_rel, output_group="large"))
    return jobs


def _synthetic_jobs(*, synthetic_types: list[str], synthetic_seeds: list[str]) -> list[DatasetJob]:
    jobs: list[DatasetJob] = []
    for graph_type in synthetic_types:
        for seed in synthetic_seeds:
            dataset = f"synthetic_{graph_type}_25_{seed}"
            graph_path = REPO_ROOT / "causal_graphs" / "synthetic_graphs" / f"graph_{graph_type}_25_{seed}.npz"
            jobs.append(DatasetJob(dataset=dataset, graph_path=graph_path, output_group="synthetic"))
    return jobs


def _build_command(args: argparse.Namespace, job: DatasetJob) -> tuple[list[str], Path]:
    output_csv = args.output_root / job.output_group / f"{job.dataset}.csv"
    cmd = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--bif-file",
        str(job.graph_path),
        "--dataset",
        job.dataset,
        "--config-file",
        str(args.config_file),
        "--output-csv",
        str(output_csv),
        "--num-prompts",
        str(args.num_prompts),
        "--seed",
        str(args.seed),
    ]
    if args.config_indexes:
        cmd.extend(["--config-indexes", args.config_indexes])
    return cmd, output_csv


def main() -> int:
    args = _parse_args()
    jobs: list[DatasetJob] = []
    if not args.skip_real:
        jobs.extend(_real_jobs())
    if not args.skip_synthetic:
        jobs.extend(
            _synthetic_jobs(
                synthetic_types=_split_csv(args.synthetic_types),
                synthetic_seeds=_split_csv(args.synthetic_seeds),
            )
        )

    if not jobs:
        raise SystemExit("No datasets selected.")

    for job in jobs:
        if not job.graph_path.exists():
            raise SystemExit(f"Missing graph for dataset {job.dataset}: {job.graph_path}")
        cmd, output_csv = _build_command(args, job)
        manifest_json = output_csv.with_suffix(output_csv.suffix + ".manifest.json")
        if output_csv.exists() and manifest_json.exists() and not args.overwrite:
            print(f"[skip] {job.dataset}: {output_csv}")
            continue
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        print("[run] " + " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
