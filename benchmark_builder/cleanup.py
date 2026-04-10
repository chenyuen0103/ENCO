from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .schema import BenchmarkSpec


@dataclass(frozen=True)
class CleanupTarget:
    label: str
    path: Path


def collect_prompt_cleanup_targets(*, spec: BenchmarkSpec, repo_root: Path, include_examples: bool) -> list[CleanupTarget]:
    targets: list[CleanupTarget] = []

    benchmark_prompt_dir = repo_root / "experiments" / "prompts" / "benchmarks" / spec.name
    targets.append(CleanupTarget(label="benchmark_prompts", path=benchmark_prompt_dir))

    if include_examples and spec.execution.prompt_retention != "none":
        if spec.execution.example_prompt_dir:
            example_dir = Path(spec.execution.example_prompt_dir)
            if not example_dir.is_absolute():
                example_dir = (repo_root / example_dir).resolve()
            targets.append(CleanupTarget(label="example_prompts", path=example_dir))
        else:
            for dataset in spec.datasets:
                targets.append(
                    CleanupTarget(
                        label=f"example_prompts:{dataset.name}",
                        path=repo_root / "experiments" / "prompts" / dataset.name / "example_prompts",
                    )
                )

    deduped: list[CleanupTarget] = []
    seen: set[Path] = set()
    for target in targets:
        if target.path in seen:
            continue
        seen.add(target.path)
        deduped.append(target)
    return deduped


def delete_cleanup_targets(targets: list[CleanupTarget]) -> list[Path]:
    deleted: list[Path] = []
    for target in targets:
        if not target.path.exists():
            continue
        shutil.rmtree(target.path)
        deleted.append(target.path)
    return deleted
