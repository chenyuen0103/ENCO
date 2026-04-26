from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .interfaces import BaselineAdapter
from .schema import BaselineSpec, BenchmarkSpec, DatasetSpec, PromptCellSpec

PYTHON_EXE = sys.executable or "python3"


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    if dry_run:
        print("[dry-run]", " ".join(str(c) for c in cmd))
        return
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), check=True)


def _reuse_if_exists(path: Path, *, dry_run: bool) -> bool:
    if dry_run:
        return False
    if path.exists():
        print(f"[reuse] {path}")
        return True
    return False


def _is_names_only_cell(cell: PromptCellSpec) -> bool:
    return cell.style == "names_only" or (cell.obs_per_prompt == 0 and cell.int_per_combo == 0)


@dataclass
class ClassicalBaselineAdapter(BaselineAdapter):
    repo_root: Path
    method_name: str

    def applies_to(self, baseline: BaselineSpec, cell: PromptCellSpec) -> bool:
        if _is_names_only_cell(cell):
            return False
        if not baseline.enabled:
            return False
        scope = baseline.scope.lower()
        if scope == "observational":
            return cell.int_per_combo == 0 and cell.obs_per_prompt > 0
        if scope == "interventional":
            return cell.int_per_combo > 0
        return True

    def dedupe_key(
        self,
        *,
        baseline: BaselineSpec,
        dataset: DatasetSpec,
        cell: PromptCellSpec,
        entry: dict[str, Any],
    ) -> tuple[Any, ...]:
        sample_size_obs = baseline.sample_size_obs if baseline.sample_size_obs is not None else cell.obs_per_prompt
        return (baseline.name, dataset.name, sample_size_obs)

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
            PYTHON_EXE,
            "scripts/run_classical.py",
            "--method",
            self.method_name,
            "--graph_files",
            str(graph_path),
            "--sample_size_obs",
            str(sample_size_obs),
            "--seed",
            str(baseline.seed if baseline.seed is not None else spec.seed),
            "--out_dir",
            str(self.repo_root / "experiments" / "responses"),
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
        if not _reuse_if_exists(out_csv, dry_run=dry_run):
            _run(cmd, cwd=self.repo_root, dry_run=dry_run)
        return out_csv


@dataclass
class ENCOBaselineAdapter(BaselineAdapter):
    repo_root: Path

    def applies_to(self, baseline: BaselineSpec, cell: PromptCellSpec) -> bool:
        if _is_names_only_cell(cell):
            return False
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
            PYTHON_EXE,
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
        dataset_name = graph_path.stem
        out_csv = self.repo_root / "experiments" / "responses" / dataset_name / (
            f"predictions_obs{sample_size_obs}_int{sample_size_inters}_ENCO.csv"
        )
        if not _reuse_if_exists(out_csv, dry_run=dry_run):
            _run(cmd, cwd=self.repo_root / "experiments", dry_run=dry_run)
        return out_csv


@dataclass
class ExternalLLMBaselineAdapter(BaselineAdapter):
    repo_root: Path
    method_name: str

    def applies_to(self, baseline: BaselineSpec, cell: PromptCellSpec) -> bool:
        if not baseline.enabled:
            return False
        if cell.naming_regime == "anonymized":
            return False
        if self.method_name == "CausalLLMPrompt":
            return _is_names_only_cell(cell)
        if self.method_name == "JiralerspongBFS":
            return (not _is_names_only_cell(cell)) and cell.style == "summary" and cell.int_per_combo == 0 and cell.obs_per_prompt > 0
        if self.method_name == "TakayamaSCP":
            return (not _is_names_only_cell(cell)) and cell.int_per_combo == 0 and cell.obs_per_prompt > 0
        if self.method_name == "CausalLLMData":
            return (not _is_names_only_cell(cell)) and cell.style == "summary"
        return False

    def _effective_sample_size_inters(self, baseline: BaselineSpec, cell: PromptCellSpec) -> int:
        if self.method_name in {"TakayamaSCP", "JiralerspongBFS", "CausalLLMPrompt"}:
            return 0
        return baseline.sample_size_inters if baseline.sample_size_inters is not None else cell.int_per_combo

    def dedupe_key(
        self,
        *,
        baseline: BaselineSpec,
        dataset: DatasetSpec,
        cell: PromptCellSpec,
        entry: dict[str, Any],
    ) -> tuple[Any, ...]:
        sample_size_obs = baseline.sample_size_obs if baseline.sample_size_obs is not None else cell.obs_per_prompt
        if self.method_name in {"TakayamaSCP", "JiralerspongBFS"}:
            return (baseline.name, dataset.name, sample_size_obs, entry["naming_regime"])
        if self.method_name == "CausalLLMPrompt":
            return (baseline.name, dataset.name, "names_only")
        return (baseline.name, dataset.name, entry["config_name"])

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
        sample_size_inters = self._effective_sample_size_inters(baseline, cell)
        naming_regime = "names_only" if _is_names_only_cell(cell) else cell.naming_regime
        naming_suffix = ""
        if naming_regime == "anonymized":
            naming_suffix = "_anon"
        elif naming_regime == "names_only":
            naming_suffix = "_names_only"
        out_csv = self.repo_root / "experiments" / "responses" / dataset.name / (
            f"predictions_obs{sample_size_obs}_int{sample_size_inters}_{self.method_name}{naming_suffix}.csv"
        )
        model_name = baseline.model or next((model.name for model in spec.models if model.enabled), spec.models[0].name)
        if self.method_name == "TakayamaSCP":
            cmd = [
                PYTHON_EXE,
                "scripts/takayama_scd.py",
                "--graph_files",
                str(graph_path),
                "--sample_size_obs",
                str(sample_size_obs),
                "--seed",
                str(baseline.seed if baseline.seed is not None else spec.seed),
                "--out_dir",
                str(self.repo_root / "experiments" / "responses"),
                "--model",
                model_name,
                "--provider",
                baseline.provider,
                "--temperature",
                str(baseline.temperature),
                "--num_samples",
                str(baseline.num_samples),
                "--takayama_pattern",
                str(baseline.takayama_pattern),
                "--bootstrap_samples",
                str(baseline.takayama_bootstrap_samples),
                "--naming_regime",
                naming_regime,
            ]
            if baseline.max_new_tokens is not None:
                cmd.extend(["--max_new_tokens", str(baseline.max_new_tokens)])
        else:
            cmd = [
                PYTHON_EXE,
                "scripts/run_external_llm.py",
                "--method",
                self.method_name,
                "--graph_files",
                str(graph_path),
                "--sample_size_obs",
                str(sample_size_obs),
                "--sample_size_inters",
                str(sample_size_inters),
                "--seed",
                str(baseline.seed if baseline.seed is not None else spec.seed),
                "--out_dir",
                str(self.repo_root / "experiments" / "responses"),
                "--model",
                model_name,
                "--provider",
                baseline.provider,
                "--temperature",
                str(baseline.temperature),
                "--num_samples",
                str(baseline.num_samples),
                "--edge_threshold",
                str(baseline.edge_threshold),
                "--prompt_mode",
                ("summary" if self.method_name == "JiralerspongBFS" else ("names_only" if _is_names_only_cell(cell) else "summary")),
                "--naming_regime",
                naming_regime,
            ]
            if baseline.max_new_tokens is not None:
                cmd.extend(["--max_new_tokens", str(baseline.max_new_tokens)])
        if not _reuse_if_exists(out_csv, dry_run=dry_run):
            _run(cmd, cwd=self.repo_root, dry_run=dry_run)
        return out_csv


def build_baseline_adapters(repo_root: Path) -> dict[str, BaselineAdapter]:
    return {
        "ENCO": ENCOBaselineAdapter(repo_root=repo_root),
        "PC": ClassicalBaselineAdapter(repo_root=repo_root, method_name="PC"),
        "GES": ClassicalBaselineAdapter(repo_root=repo_root, method_name="GES"),
        "TakayamaSCP": ExternalLLMBaselineAdapter(repo_root=repo_root, method_name="TakayamaSCP"),
        "JiralerspongBFS": ExternalLLMBaselineAdapter(repo_root=repo_root, method_name="JiralerspongBFS"),
        "CausalLLMPrompt": ExternalLLMBaselineAdapter(repo_root=repo_root, method_name="CausalLLMPrompt"),
        "CausalLLMData": ExternalLLMBaselineAdapter(repo_root=repo_root, method_name="CausalLLMData"),
    }
