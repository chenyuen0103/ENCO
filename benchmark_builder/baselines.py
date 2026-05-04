from __future__ import annotations

import csv
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .interfaces import BaselineAdapter
from .schema import BaselineSpec, BenchmarkSpec, DatasetSpec, PromptCellSpec

PYTHON_EXE = sys.executable or "python3"
TAKAYAMA_DEFAULT_MODEL = "gpt-4.1"
NEURAL_CAUSAL_LLM_METHODS = {"CausalLLMDataNeural"}


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


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    old_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(sys.maxsize)
        with path.open("r", encoding="utf-8", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    finally:
        csv.field_size_limit(old_limit)


def _reuse_if_has_rows(path: Path, *, dry_run: bool, min_rows: int) -> bool:
    if dry_run:
        return False
    if not path.exists():
        return False
    rows = _csv_row_count(path)
    if rows >= min_rows:
        print(f"[reuse] {path} rows={rows}")
        return True
    print(f"[rerun] {path} rows={rows} < required={min_rows}")
    return False


def _effective_seed(baseline: BaselineSpec, spec: BenchmarkSpec) -> int:
    return int(baseline.seed if baseline.seed is not None else spec.seed)


def _seed_suffix(seed: int) -> str:
    return f"_seed{int(seed)}"


def _checkpoint_dir_with_seed(base: str, seed: int) -> str:
    return f"{str(base).rstrip('/')}{_seed_suffix(seed)}"


def _unseeded_variant_path(path: Path) -> Path:
    name = re.sub(r"_seed\d+(?=(?:_(?:anon|names_only))?\.csv$)", "", path.name)
    return path.with_name(name)


def _materialize_existing_csv_copy(*, source_csv: Path, out_csv: Path, dry_run: bool, min_rows: int = 1) -> bool:
    if not source_csv.exists():
        return False
    rows = _csv_row_count(source_csv)
    if rows < min_rows:
        print(f"[skip:unseeded] {source_csv} rows={rows} < required={min_rows}")
        return False
    if dry_run:
        print(f"[dry-run][reuse:unseeded] {source_csv} -> {out_csv} rows={rows}")
        return True
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_csv, out_csv)
    print(f"[reuse:unseeded] {source_csv} -> {out_csv} rows={rows}")
    return True


def _model_filename_tags(model_name: str) -> list[str]:
    tags: list[str] = []
    for raw in (model_name, model_name.split("/")[-1]):
        tag = str(raw or "").strip()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def _prompt_count_from_name(path: Path) -> int:
    match = re.search(r"_p(\d+)_", path.name)
    return int(match.group(1)) if match else 0


def _find_legacy_external_baseline_csv(
    *,
    responses_dir: Path,
    method_name: str,
    model_name: str,
    sample_size_obs: int,
    sample_size_inters: int,
) -> Path | None:
    tags = _model_filename_tags(model_name)
    candidates: list[Path] = []

    if method_name == "CausalLLMData" and sample_size_inters == 0:
        for tag in tags:
            candidates.extend(sorted(responses_dir.glob(f"responses_obs{sample_size_obs}_int0_shuf1_p*_summary_joint_{tag}.csv")))
            candidates.extend(sorted(responses_dir.glob(f"responses_obs{sample_size_obs}_int0_shuf1_p*_summary_{tag}.csv")))
        if not candidates:
            return None
        candidates = sorted(
            candidates,
            key=lambda p: (
                0 if "_summary_joint_" in p.name else 1,
                -_prompt_count_from_name(p),
                p.name,
            ),
        )
        return candidates[0]

    if method_name == "CausalLLMPrompt" and sample_size_obs == 0 and sample_size_inters == 0:
        for tag in tags:
            exact = responses_dir / f"responses_names_only_{tag}.csv"
            if exact.exists():
                candidates.append(exact)
            candidates.extend(sorted(responses_dir.glob(f"responses_names_only_p*_{tag}.csv")))
        if not candidates:
            return None
        candidates = sorted(
            candidates,
            key=lambda p: (
                -_prompt_count_from_name(p),
                0 if "_p" not in p.stem else 1,
                p.name,
            ),
        )
        return candidates[0]

    return None


def _materialize_legacy_external_baseline_copy(
    *,
    legacy_csv: Path,
    out_csv: Path,
    method_name: str,
    model_name: str,
    provider: str,
    naming_regime: str,
    sample_size_obs: int,
    sample_size_inters: int,
    dry_run: bool,
    min_rows: int = 1,
) -> bool:
    with legacy_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return False
    if len(rows) < min_rows:
        print(f"[skip:legacy] {legacy_csv} rows={len(rows)} < required={min_rows}")
        return False
    if dry_run:
        print(f"[dry-run][reuse:legacy] {legacy_csv} -> {out_csv} rows={len(rows)}")
        return True

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "model",
                "provider",
                "naming_regime",
                "obs_n",
                "int_n",
                "raw_response",
                "answer",
                "prediction",
                "valid",
            ],
        )
        writer.writeheader()
        for row in rows:
            valid_raw = row.get("valid", 1)
            try:
                valid = int(float(valid_raw))
            except Exception:
                valid = 1
            writer.writerow(
                {
                    "method": method_name,
                    "model": model_name,
                    "provider": provider,
                    "naming_regime": naming_regime,
                    "obs_n": sample_size_obs,
                    "int_n": sample_size_inters,
                    "raw_response": row.get("raw_response", "") or "",
                    "answer": row.get("answer", "") or "",
                    "prediction": row.get("prediction", "") or "",
                    "valid": valid,
                }
            )
    print(f"[reuse:legacy] {legacy_csv} -> {out_csv}")
    return True


def _is_names_only_cell(cell: PromptCellSpec) -> bool:
    return cell.style == "names_only" or (cell.obs_per_prompt == 0 and cell.int_per_combo == 0)


def _takayama_backend_slug(raw: str | None) -> str:
    backend = (raw or "pc").strip().lower()
    aliases = {
        "pc": "pc",
        "exact": "exact_search",
        "exactsearch": "exact_search",
        "exact_search": "exact_search",
        "es": "exact_search",
        "lingam": "direct_lingam",
        "directlingam": "direct_lingam",
        "direct_lingam": "direct_lingam",
    }
    return aliases.get(backend, backend)


def _takayama_backend_suffix(raw: str | None) -> str:
    backend = _takayama_backend_slug(raw)
    if backend == "pc":
        return ""
    if backend == "exact_search":
        return "_ExactSearch"
    if backend == "direct_lingam":
        return "_DirectLiNGAM"
    return f"_{backend}"


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
        seed = _effective_seed(baseline, spec)
        num_prompts = max(1, int(getattr(spec, "num_prompts", 1)))
        out_csv = self.repo_root / "experiments" / "responses" / dataset.name / (
            f"predictions_obs{sample_size_obs}_int0_{self.method_name}{_seed_suffix(seed)}.csv"
        )
        legacy_out_csv = _unseeded_variant_path(out_csv)
        if _reuse_if_has_rows(out_csv, dry_run=dry_run, min_rows=num_prompts):
            return out_csv
        if _materialize_existing_csv_copy(source_csv=legacy_out_csv, out_csv=out_csv, dry_run=dry_run, min_rows=num_prompts):
            return out_csv
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
            str(seed),
            "--num_prompts",
            str(num_prompts),
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
        seed = _effective_seed(baseline, spec)
        num_prompts = max(1, int(getattr(spec, "num_prompts", 1)))
        checkpoint_dir_base = baseline.checkpoint_dir or (
            f"experiments/checkpoints/benchmarks/{spec.name}/{dataset.name}/obs{cell.obs_per_prompt}_int{cell.int_per_combo}/ENCO"
        )
        checkpoint_dir = _checkpoint_dir_with_seed(checkpoint_dir_base, seed)
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
            str(seed),
            "--num_prompts",
            str(num_prompts),
            "--checkpoint_dir",
            checkpoint_dir,
        ]
        dataset_name = graph_path.stem
        out_csv = self.repo_root / "experiments" / "responses" / dataset_name / (
            f"predictions_obs{sample_size_obs}_int{sample_size_inters}_ENCO{_seed_suffix(seed)}.csv"
        )
        legacy_out_csv = _unseeded_variant_path(out_csv)
        if _reuse_if_has_rows(out_csv, dry_run=dry_run, min_rows=num_prompts):
            return out_csv
        if _materialize_existing_csv_copy(source_csv=legacy_out_csv, out_csv=out_csv, dry_run=dry_run, min_rows=num_prompts):
            return out_csv
        _run(cmd, cwd=self.repo_root / "experiments", dry_run=dry_run)
        return out_csv


@dataclass
class ExternalLLMBaselineAdapter(BaselineAdapter):
    repo_root: Path
    method_name: str

    def applies_to(self, baseline: BaselineSpec, cell: PromptCellSpec) -> bool:
        if not baseline.enabled:
            return False
        if self.method_name in NEURAL_CAUSAL_LLM_METHODS:
            return (not _is_names_only_cell(cell)) and cell.obs_per_prompt > 0 and cell.int_per_combo == 0
        if cell.naming_regime == "anonymized":
            return False
        if self.method_name == "CausalLLMPrompt":
            return _is_names_only_cell(cell)
        if self.method_name == "JiralerspongPairwise":
            return cell.style == "summary" and cell.int_per_combo == 0 and cell.obs_per_prompt > 0
        if self.method_name == "JiralerspongBFS":
            return (not _is_names_only_cell(cell)) and cell.style == "summary" and cell.int_per_combo == 0 and cell.obs_per_prompt > 0
        if self.method_name == "TakayamaSCP":
            return (not _is_names_only_cell(cell)) and cell.int_per_combo == 0 and cell.obs_per_prompt > 0
        if self.method_name == "CausalLLMData":
            return (not _is_names_only_cell(cell)) and cell.style == "summary"
        return False

    def _effective_sample_size_inters(self, baseline: BaselineSpec, cell: PromptCellSpec) -> int:
        if self.method_name in {"TakayamaSCP", "JiralerspongBFS", "JiralerspongPairwise", "CausalLLMPrompt"} | NEURAL_CAUSAL_LLM_METHODS:
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
        if self.method_name in NEURAL_CAUSAL_LLM_METHODS:
            return (baseline.name, dataset.name, sample_size_obs)
        if self.method_name in {"TakayamaSCP", "JiralerspongBFS"}:
            extra = ()
            if self.method_name == "TakayamaSCP":
                extra = (
                    _takayama_backend_slug(baseline.takayama_backend),
                    int(baseline.takayama_pattern),
                )
            return (baseline.name, dataset.name, sample_size_obs, entry["naming_regime"], *extra)
        if self.method_name == "JiralerspongPairwise":
            return (baseline.name, dataset.name, entry["config_name"])
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
        seed = _effective_seed(baseline, spec)
        num_prompts = max(1, int(getattr(spec, "num_prompts", 1)))
        naming_regime = "names_only" if _is_names_only_cell(cell) else cell.naming_regime
        naming_suffix = ""
        if naming_regime == "anonymized":
            naming_suffix = "_anon"
        elif naming_regime == "names_only":
            naming_suffix = "_names_only"
        takayama_suffix = ""
        if self.method_name == "TakayamaSCP":
            takayama_suffix = f"{_takayama_backend_suffix(baseline.takayama_backend)}_p{int(baseline.takayama_pattern)}"
        out_csv = self.repo_root / "experiments" / "responses" / dataset.name / (
            f"predictions_obs{sample_size_obs}_int{sample_size_inters}_{self.method_name}{takayama_suffix}{_seed_suffix(seed)}{naming_suffix}.csv"
        )
        if self.method_name == "TakayamaSCP":
            model_name = baseline.model or TAKAYAMA_DEFAULT_MODEL
        else:
            model_name = baseline.model or next((model.name for model in spec.models if model.enabled), spec.models[0].name)
        expected_rows = 1 if self.method_name in NEURAL_CAUSAL_LLM_METHODS else num_prompts
        if _reuse_if_has_rows(out_csv, dry_run=dry_run, min_rows=expected_rows):
            return out_csv
        legacy_out_csv = _unseeded_variant_path(out_csv)
        if _materialize_existing_csv_copy(source_csv=legacy_out_csv, out_csv=out_csv, dry_run=dry_run, min_rows=expected_rows):
            return out_csv

        if self.method_name in {"CausalLLMData", "CausalLLMPrompt"}:
            legacy_csv = _find_legacy_external_baseline_csv(
                responses_dir=self.repo_root / "experiments" / "responses" / dataset.name,
                method_name=self.method_name,
                model_name=model_name,
                sample_size_obs=sample_size_obs,
                sample_size_inters=sample_size_inters,
            )
            if legacy_csv is not None and _materialize_legacy_external_baseline_copy(
                legacy_csv=legacy_csv,
                out_csv=out_csv,
                method_name=self.method_name,
                model_name=model_name,
                provider=baseline.provider,
                naming_regime=naming_regime,
                sample_size_obs=sample_size_obs,
                sample_size_inters=sample_size_inters,
                min_rows=num_prompts,
                dry_run=dry_run,
            ):
                return out_csv

        if self.method_name in NEURAL_CAUSAL_LLM_METHODS:
            checkpoint_base = baseline.checkpoint_dir or (
                f"experiments/checkpoints/benchmarks/{spec.name}/{dataset.name}/obs{sample_size_obs}_int0/{self.method_name}"
            )
            checkpoint_path = self.repo_root / f"{checkpoint_base}{_seed_suffix(seed)}.pth"
            cmd = [
                PYTHON_EXE,
                "scripts/run_causal_llm_neural.py",
                "--graph_files",
                str(graph_path),
                "--sample_size_obs",
                str(sample_size_obs),
                "--seed",
                str(seed),
                "--out_dir",
                str(self.repo_root / "experiments" / "responses"),
                "--out_csv",
                str(out_csv),
                "--method_name",
                self.method_name,
                "--model_path",
                str(checkpoint_path),
                "--num_epochs",
                str(baseline.causal_llm_epochs),
                "--batch_size",
                str(baseline.causal_llm_batch_size),
                "--epsilon",
                str(baseline.causal_llm_epsilon),
                "--l1_lambda",
                str(baseline.causal_llm_l1_lambda),
            ]
        elif self.method_name == "TakayamaSCP":
            cmd = [
                PYTHON_EXE,
                "scripts/takayama_scd.py",
                "--graph_files",
                str(graph_path),
                "--sample_size_obs",
                str(sample_size_obs),
                "--seed",
                str(seed),
                "--num_prompts",
                str(num_prompts),
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
                "--backend",
                _takayama_backend_slug(baseline.takayama_backend),
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
                str(seed),
                "--num_prompts",
                str(num_prompts),
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
        _run(cmd, cwd=self.repo_root, dry_run=dry_run)
        return out_csv


def build_baseline_adapters(repo_root: Path) -> dict[str, BaselineAdapter]:
    return {
        "ENCO": ENCOBaselineAdapter(repo_root=repo_root),
        "PC": ClassicalBaselineAdapter(repo_root=repo_root, method_name="PC"),
        "GES": ClassicalBaselineAdapter(repo_root=repo_root, method_name="GES"),
        "TakayamaSCP": ExternalLLMBaselineAdapter(repo_root=repo_root, method_name="TakayamaSCP"),
        "JiralerspongBFS": ExternalLLMBaselineAdapter(repo_root=repo_root, method_name="JiralerspongBFS"),
        "JiralerspongPairwise": ExternalLLMBaselineAdapter(repo_root=repo_root, method_name="JiralerspongPairwise"),
        "CausalLLMPrompt": ExternalLLMBaselineAdapter(repo_root=repo_root, method_name="CausalLLMPrompt"),
        "CausalLLMData": ExternalLLMBaselineAdapter(repo_root=repo_root, method_name="CausalLLMData"),
        "CausalLLMDataNeural": ExternalLLMBaselineAdapter(repo_root=repo_root, method_name="CausalLLMDataNeural"),
    }
