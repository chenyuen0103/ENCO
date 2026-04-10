from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .interfaces import BaselineAdapter
from .schema import BaselineSpec, BenchmarkSpec, DatasetSpec, PromptCellSpec


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    if dry_run:
        print("[dry-run]", " ".join(str(c) for c in cmd))
        return
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), check=True)


@dataclass
class ClassicalBaselineAdapter(BaselineAdapter):
    repo_root: Path
    method_name: str

    def applies_to(self, baseline: BaselineSpec, cell: PromptCellSpec) -> bool:
        if not baseline.enabled:
            return False
        scope = baseline.scope.lower()
        if scope == "observational":
            return cell.int_per_combo == 0 and cell.obs_per_prompt > 0
        if scope == "interventional":
            return cell.int_per_combo > 0
        return True

    def run(
        self,
        *,
        baseline: BaselineSpec,
        dataset: DatasetSpec,
        graph_path: Path,
        cell: PromptCellSpec,
        spec: BenchmarkSpec,
        dry_run: bool,
    ) -> Path:
        sample_size_obs = baseline.sample_size_obs if baseline.sample_size_obs is not None else cell.obs_per_prompt
        out_csv = self.repo_root / "experiments" / "responses" / dataset.name / (
            f"predictions_obs{sample_size_obs}_int0_{self.method_name}.csv"
        )
        cmd = [
            "python3",
            "run_classical_baselines.py",
            "--method",
            self.method_name,
            "--graph_files",
            str(graph_path),
            "--sample_size_obs",
            str(sample_size_obs),
            "--seed",
            str(baseline.seed if baseline.seed is not None else spec.seed),
            "--out_dir",
            "responses",
        ]
        if self.method_name == "PC":
            cmd.extend(
                [
                    "--pc-variant",
                    baseline.pc_variant,
                    "--pc-ci-test",
                    baseline.pc_ci_test,
                    "--pc-significance-level",
                    str(baseline.pc_significance_level),
                    "--pc-max-cond-vars",
                    str(baseline.pc_max_cond_vars),
                ]
            )
        elif self.method_name == "GES":
            cmd.extend(
                [
                    "--ges-scoring-method",
                    baseline.ges_scoring_method,
                    "--ges-min-improvement",
                    str(baseline.ges_min_improvement),
                ]
            )
        _run(cmd, cwd=self.repo_root / "experiments", dry_run=dry_run)
        return out_csv


@dataclass
class ENCOBaselineAdapter(BaselineAdapter):
    repo_root: Path

    def applies_to(self, baseline: BaselineSpec, cell: PromptCellSpec) -> bool:
        if not baseline.enabled:
            return False
        scope = baseline.scope.lower()
        if scope == "observational":
            return cell.int_per_combo == 0 and cell.obs_per_prompt > 0
        if scope == "interventional":
            return cell.int_per_combo > 0
        return True

    def run(
        self,
        *,
        baseline: BaselineSpec,
        dataset: DatasetSpec,
        graph_path: Path,
        cell: PromptCellSpec,
        spec: BenchmarkSpec,
        dry_run: bool,
    ) -> Path:
        checkpoint_dir = baseline.checkpoint_dir or (
            f"experiments/checkpoints/benchmarks/{spec.name}/{dataset.name}/obs{cell.obs_per_prompt}_int{cell.int_per_combo}/ENCO"
        )
        sample_size_obs = baseline.sample_size_obs if baseline.sample_size_obs is not None else cell.obs_per_prompt
        sample_size_inters = baseline.sample_size_inters if baseline.sample_size_inters is not None else cell.int_per_combo
        cmd = [
            "python3",
            "run_exported_graphs.py",
            "--graph_files",
            str(graph_path),
            "--sample_size_obs",
            str(sample_size_obs),
            "--sample_size_inters",
            str(sample_size_inters),
            "--max_inters",
            str(baseline.max_inters),
            "--seed",
            str(baseline.seed if baseline.seed is not None else spec.seed),
            "--checkpoint_dir",
            checkpoint_dir,
        ]
        _run(cmd, cwd=self.repo_root / "experiments", dry_run=dry_run)
        dataset_name = graph_path.stem
        return self.repo_root / "experiments" / "responses" / dataset_name / (
            f"predictions_obs{sample_size_obs}_int{sample_size_inters}_ENCO.csv"
        )


def build_baseline_adapters(repo_root: Path) -> dict[str, BaselineAdapter]:
    return {
        "ENCO": ENCOBaselineAdapter(repo_root=repo_root),
        "PC": ClassicalBaselineAdapter(repo_root=repo_root, method_name="PC"),
        "GES": ClassicalBaselineAdapter(repo_root=repo_root, method_name="GES"),
    }
