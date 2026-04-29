from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baselines import build_baseline_adapters
from .evaluation import EvalScriptEvaluator, contamination_audit, direct_csv_summary, write_csv
from .graph_io import file_sha256, materialize_graph_source
from .interfaces import CommandPlan
from .registry import BenchmarkRegistry
from .schema import BenchmarkSpec, DatasetSpec, ModelSpec, PromptCellSpec, load_benchmark_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = sys.executable or "python3"


def _run(plan: CommandPlan, *, dry_run: bool) -> None:
    if dry_run:
        print("[dry-run]", " ".join(str(c) for c in plan.cmd))
        return
    subprocess.run([str(c) for c in plan.cmd], cwd=str(plan.cwd), check=True)


def _reuse_existing(path: Path, *, dry_run: bool, label: str) -> bool:
    if dry_run:
        return False
    if path.exists():
        print(f"[reuse:{label}] {path}")
        return True
    return False


def _reuse_artifact_if_current(path: Path, *, source: Path, dry_run: bool, label: str) -> bool:
    if dry_run:
        return False
    if not path.exists():
        return False
    if source.exists() and path.stat().st_mtime < source.stat().st_mtime:
        print(f"[rerun:{label}] {path} older than {source}")
        return False
    print(f"[reuse:{label}] {path}")
    return True


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_cell_config_name(cell: PromptCellSpec) -> str:
    parts = [cell.style]
    parts.append("anon" if cell.anonymize else "real")
    parts.append(f"obs{cell.obs_per_prompt}")
    parts.append(f"int{cell.int_per_combo}")
    if cell.reasoning_guidance != "staged":
        parts.append(f"reason{cell.reasoning_guidance}")
    if cell.row_order != "random":
        parts.append(f"row{cell.row_order}")
    if cell.col_order != "original":
        parts.append(f"col{cell.col_order}")
    if cell.causal_rules:
        parts.append("rules")
    if cell.give_steps:
        parts.append("steps")
    if cell.def_int:
        parts.append("defint")
    return "_".join(parts)


def _prompt_base_name(*, cell: PromptCellSpec, num_prompts: int, shuffles_per_graph: int) -> str:
    tags: list[str] = []
    if cell.anonymize:
        tags.append("anon")
    if cell.causal_rules:
        tags.append("rules")
    if cell.give_steps:
        tags.append("steps")
    if cell.reasoning_guidance != "staged":
        tags.append(f"reason{cell.reasoning_guidance}")
    if cell.style == "summary":
        tags.append("summary")
    elif cell.style == "matrix":
        tags.append("matrix")
    if cell.row_order != "random":
        tags.append(f"row{cell.row_order}")
    if cell.col_order != "original":
        tags.append(f"col{cell.col_order}")
    extra_suffix = ("_" + "_".join(tags)) if tags else ""
    return (
        f"prompts_obs{cell.obs_per_prompt}"
        f"_int{cell.int_per_combo}"
        f"_shuf{shuffles_per_graph}"
        f"_p{num_prompts}{extra_suffix}"
    )


def _response_name_for_prompt(prompt_csv: Path, model_name: str) -> str:
    stem = prompt_csv.stem
    if stem.startswith("prompts_"):
        stem = "responses_" + stem[len("prompts_") :]
    return f"{stem}_{model_name}.csv"


def _response_name_for_base(base_name: str, model_name: str) -> str:
    return _response_name_for_prompt(Path(base_name).with_suffix(".csv"), model_name)


def _per_row_metrics_path(response_csv: Path) -> Path:
    return Path(str(response_csv) + ".per_row.csv")


def _config_hash(spec: BenchmarkSpec) -> str:
    payload = json.dumps(spec.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _merge_response_entries(existing: list[dict[str, Any]], new_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for entry in existing + new_entries:
        key = "|".join(
            str(entry.get(part) or "")
            for part in (
                "benchmark",
                "dataset",
                "config_name",
                "prompt_style",
                "naming_regime",
                "reasoning_guidance",
                "obs_n",
                "int_n",
                "system",
                "system_kind",
                "model",
                "baseline",
                "takayama_backend",
                "takayama_pattern",
            )
        )
        merged[key] = entry
    return list(merged.values())


def _baseline_prefers_anonymized(baseline_name: str) -> bool:
    return baseline_name in {"PC", "GES", "ENCO"}


@dataclass
class BenchmarkRunner:
    spec: BenchmarkSpec
    repo_root: Path = REPO_ROOT

    @property
    def run_root(self) -> Path:
        return self.repo_root / self.spec.output_root / self.spec.name

    @property
    def experiments_dir(self) -> Path:
        return self.repo_root / "experiments"

    def build(self, *, dry_run: bool = False) -> dict[str, Any]:
        prompt_entries: list[dict[str, Any]] = []
        config_hash = _config_hash(self.spec)
        for dataset in self.spec.datasets:
            graph_path = materialize_graph_source(
                dataset,
                repo_root=self.repo_root,
                run_root=self.run_root,
                seed=self.spec.seed,
                dry_run=dry_run,
            )
            for cell in self.spec.prompt_cells:
                if not cell.enabled:
                    continue
                prompt_entries.append(self._build_prompt_cell(dataset=dataset, graph_path=graph_path, cell=cell, dry_run=dry_run))
            if self.spec.names_only.enabled:
                prompt_entries.append(self._build_names_only(dataset=dataset, graph_path=graph_path, dry_run=dry_run))

        bundle = {
            "schema_version": "prompt_bundle/v1",
            "benchmark": self.spec.name,
            "config_hash": config_hash,
            "prompt_storage": self.spec.execution.prompt_storage,
            "prompt_retention": self.spec.execution.prompt_retention,
            "entries": prompt_entries,
        }
        if not dry_run:
            _json_dump(self.run_root / "prompt_bundle.json", bundle)
        return bundle

    def _build_prompt_cell(
        self,
        *,
        dataset: DatasetSpec,
        graph_path: Path,
        cell: PromptCellSpec,
        dry_run: bool,
    ) -> dict[str, Any]:
        config_name = _build_cell_config_name(cell)
        base_name = _prompt_base_name(
            cell=cell,
            num_prompts=self.spec.num_prompts,
            shuffles_per_graph=self.spec.shuffles_per_graph,
        )
        out_dir = self.experiments_dir / "prompts" / "benchmarks" / self.spec.name / dataset.name / config_name
        prompt_csv = out_dir / f"{base_name}.csv"
        if self.spec.execution.prompt_storage == "disk":
            cmd = [
                PYTHON_EXE,
                "generate_prompts.py",
                "--graph-file",
                str(graph_path),
                "--out-dir",
                str(out_dir),
                "--num-prompts",
                str(self.spec.num_prompts),
                "--seed",
                str(self.spec.seed),
                "--shuffles-per-graph",
                str(self.spec.shuffles_per_graph),
                "--prompt-style",
                cell.style,
                "--obs-per-prompt",
                str(cell.obs_per_prompt),
                "--int-per-combo",
                str(cell.int_per_combo),
                "--row-order",
                cell.row_order,
                "--col-order",
                cell.col_order,
                "--intervene-vars",
                cell.intervene_vars,
                "--reasoning-guidance",
                cell.reasoning_guidance,
            ]
            if cell.anonymize:
                cmd.append("--anonymize")
            if cell.causal_rules:
                cmd.append("--causal-rules")
            if cell.give_steps:
                cmd.append("--give-steps")
            if cell.def_int:
                cmd.append("--def-int")
            if not _reuse_existing(prompt_csv, dry_run=dry_run, label="prompt"):
                _run(CommandPlan(label=f"build:{dataset.name}:{config_name}", cmd=cmd, cwd=self.experiments_dir), dry_run=dry_run)
        return {
            "dataset": dataset.name,
            "kind": "prompt_cell",
            "config_name": config_name,
            "prompt_style": cell.style,
            "naming_regime": cell.naming_regime,
            "reasoning_guidance": cell.reasoning_guidance,
            "obs_n": cell.obs_per_prompt,
            "int_n": cell.int_per_combo,
            "graph_file": str(graph_path),
            "graph_sha256": file_sha256(graph_path) if graph_path.exists() else None,
            "prompt_basename": base_name,
            "prompt_storage": self.spec.execution.prompt_storage,
            "prompt_csv": str(prompt_csv) if self.spec.execution.prompt_storage == "disk" else None,
            "prompt_csv_sha256": file_sha256(prompt_csv) if self.spec.execution.prompt_storage == "disk" and prompt_csv.exists() else None,
        }

    def _build_names_only(self, *, dataset: DatasetSpec, graph_path: Path, dry_run: bool) -> dict[str, Any]:
        out_dir = self.experiments_dir / "prompts" / "benchmarks" / self.spec.name / dataset.name / "names_only"
        base_name = f"prompts_names_only_p{self.spec.num_prompts}"
        if self.spec.names_only.col_order != "original":
            base_name += f"_col{self.spec.names_only.col_order.capitalize()}"
        prompt_csv = out_dir / f"{base_name}.csv"
        if self.spec.execution.prompt_storage == "disk":
            cmd = [
                PYTHON_EXE,
                "-m",
                "cd_generation.names_only",
                "--graph-file",
                str(graph_path),
                "--out-dir",
                str(out_dir),
                "--num-prompts",
                str(self.spec.num_prompts),
                "--seed",
                str(self.spec.seed),
                "--col-order",
                self.spec.names_only.col_order,
            ]
            if not _reuse_existing(prompt_csv, dry_run=dry_run, label="prompt"):
                _run(CommandPlan(label=f"build:{dataset.name}:names_only", cmd=cmd, cwd=self.experiments_dir), dry_run=dry_run)
        return {
            "dataset": dataset.name,
            "kind": "names_only",
            "config_name": "names_only",
            "prompt_style": "names_only",
            "naming_regime": "names_only",
            "obs_n": 0,
            "int_n": 0,
            "graph_file": str(graph_path),
            "graph_sha256": file_sha256(graph_path) if graph_path.exists() else None,
            "prompt_basename": base_name,
            "prompt_storage": self.spec.execution.prompt_storage,
            "prompt_csv": str(prompt_csv) if self.spec.execution.prompt_storage == "disk" else None,
            "prompt_csv_sha256": file_sha256(prompt_csv) if self.spec.execution.prompt_storage == "disk" and prompt_csv.exists() else None,
        }

    def run_models(self, *, dry_run: bool = False, overwrite: bool = False) -> list[dict[str, Any]]:
        prompt_bundle = self._load_prompt_bundle(dry_run=dry_run)
        if self.spec.execution.prompt_storage == "in_memory":
            return self._run_models_in_memory(prompt_bundle=prompt_bundle, dry_run=dry_run, overwrite=overwrite)
        evaluator = EvalScriptEvaluator(repo_root=self.repo_root)
        response_entries: list[dict[str, Any]] = []
        for entry in prompt_bundle["entries"]:
            prompt_csv = Path(str(entry["prompt_csv"]))
            for model in self.spec.models:
                if not model.enabled:
                    continue
                response_csv = self.experiments_dir / "responses" / entry["dataset"] / _response_name_for_prompt(prompt_csv, model.name)
                per_row_path = _per_row_metrics_path(response_csv)
                query_cmd = [
                    PYTHON_EXE,
                    "query_api.py",
                    "--csv",
                    str(prompt_csv),
                    "--model",
                    model.name,
                    "--temperature",
                    str(model.temperature),
                    "--provider",
                    model.provider,
                ]
                if overwrite:
                    query_cmd.append("--overwrite")
                if overwrite or not _reuse_existing(response_csv, dry_run=dry_run, label="response"):
                    _run(CommandPlan(label=f"query:{entry['dataset']}:{model.name}:{entry['config_name']}", cmd=query_cmd, cwd=self.experiments_dir), dry_run=dry_run)
                if not _reuse_artifact_if_current(per_row_path, source=response_csv, dry_run=dry_run, label="eval"):
                    _run(evaluator.build_eval_command(response_csv=response_csv, evaluator=self.spec.evaluator), dry_run=dry_run)
                response_entries.append(
                    self._response_record(
                        entry=entry,
                        response_csv=response_csv,
                        system=model.name,
                        system_kind="model",
                        extra={
                            "model": model.name,
                            "provider": model.provider,
                            "temperature": model.temperature,
                            "backend_version": model.backend_version,
                        },
                    )
                )
        if not dry_run:
            existing_entries: list[dict[str, Any]] = []
            bundle_path = self.run_root / "response_bundle.json"
            if bundle_path.exists():
                existing_entries = json.loads(bundle_path.read_text(encoding="utf-8")).get("entries", [])
            _json_dump(
                bundle_path,
                {
                    "schema_version": "response_bundle/v1",
                    "benchmark": self.spec.name,
                    "entries": _merge_response_entries(existing_entries, response_entries),
                },
            )
        return response_entries

    def _run_models_in_memory(
        self,
        *,
        prompt_bundle: dict[str, Any],
        dry_run: bool,
        overwrite: bool,
    ) -> list[dict[str, Any]]:
        evaluator = EvalScriptEvaluator(repo_root=self.repo_root)
        response_entries: list[dict[str, Any]] = []
        grouped: dict[str, list[dict[str, Any]]] = {}
        datasets = {dataset.name: dataset for dataset in self.spec.datasets}
        for entry in prompt_bundle["entries"]:
            grouped.setdefault(str(entry["dataset"]), []).append(entry)

        for dataset_name, entries in grouped.items():
            dataset = datasets[dataset_name]
            graph_path = Path(str(entries[0]["graph_file"]))
            config_rows: list[dict[str, Any]] = []
            for entry in entries:
                if entry["kind"] == "names_only":
                    config_rows.append(
                        {
                            "style": "summary",
                            "anonymize": False,
                            "obs_per_prompt": 0,
                            "int_per_combo": 0,
                            "reasoning_guidance": "staged",
                            "row_order": "random",
                            "col_order": self.spec.names_only.col_order,
                            "shuffles_per_graph": 1,
                        }
                    )
                    continue
                config_rows.append(
                    {
                        "style": entry["prompt_style"],
                        "anonymize": entry["naming_regime"] == "anonymized",
                        "obs_per_prompt": entry["obs_n"],
                        "int_per_combo": entry["int_n"],
                        "reasoning_guidance": entry.get("reasoning_guidance", "staged"),
                        "row_order": next(
                            cell.row_order
                            for cell in self.spec.prompt_cells
                            if cell.style == entry["prompt_style"]
                            and cell.obs_per_prompt == entry["obs_n"]
                            and cell.int_per_combo == entry["int_n"]
                            and cell.naming_regime == entry["naming_regime"]
                            and cell.reasoning_guidance == entry.get("reasoning_guidance", "staged")
                        ),
                        "col_order": next(
                            cell.col_order
                            for cell in self.spec.prompt_cells
                            if cell.style == entry["prompt_style"]
                            and cell.obs_per_prompt == entry["obs_n"]
                            and cell.int_per_combo == entry["int_n"]
                            and cell.naming_regime == entry["naming_regime"]
                            and cell.reasoning_guidance == entry.get("reasoning_guidance", "staged")
                        ),
                        "shuffles_per_graph": self.spec.shuffles_per_graph,
                    }
                )

            config_path = self.run_root / "configs" / f"{dataset_name}_in_memory_config.json"
            if not dry_run:
                _json_dump(config_path, {"configs": config_rows})
            cmd = [
                PYTHON_EXE,
                "run_experiment1_in_memory.py",
                "--bif-file",
                str(graph_path),
                "--config-file",
                str(config_path),
                "--num-prompts",
                str(self.spec.num_prompts),
                "--seed",
                str(self.spec.seed),
                "--responses-root",
                str(self.experiments_dir / "responses"),
            ]
            for model in self.spec.models:
                if model.enabled:
                    cmd.extend(["--model", model.name])
            if overwrite:
                cmd.append("--overwrite")
            if self.spec.execution.prompt_retention == "none":
                cmd.append("--no-save-example-prompt")
            else:
                cmd.append("--save-example-prompt")
                if self.spec.execution.example_prompt_dir:
                    cmd.extend(["--example-prompt-dir", self.spec.execution.example_prompt_dir])
            _run(CommandPlan(label=f"in_memory:{dataset_name}", cmd=cmd, cwd=self.experiments_dir), dry_run=dry_run)

            for entry in entries:
                for model in self.spec.models:
                    if not model.enabled:
                        continue
                    response_csv = self.experiments_dir / "responses" / dataset_name / _response_name_for_base(
                        str(entry["prompt_basename"]),
                        model.name,
                    )
                    per_row_path = _per_row_metrics_path(response_csv)
                    if not _reuse_artifact_if_current(per_row_path, source=response_csv, dry_run=dry_run, label="eval"):
                        _run(evaluator.build_eval_command(response_csv=response_csv, evaluator=self.spec.evaluator), dry_run=dry_run)
                    response_entries.append(
                        self._response_record(
                            entry=entry,
                            response_csv=response_csv,
                            system=model.name,
                            system_kind="model",
                            extra={
                                "model": model.name,
                                "provider": model.provider,
                                "temperature": model.temperature,
                                "backend_version": model.backend_version,
                                "prompt_storage": "in_memory",
                                "prompt_retention": self.spec.execution.prompt_retention,
                            },
                        )
                    )

        if not dry_run:
            existing_entries: list[dict[str, Any]] = []
            bundle_path = self.run_root / "response_bundle.json"
            if bundle_path.exists():
                existing_entries = json.loads(bundle_path.read_text(encoding="utf-8")).get("entries", [])
            _json_dump(
                bundle_path,
                {
                    "schema_version": "response_bundle/v1",
                    "benchmark": self.spec.name,
                    "entries": _merge_response_entries(existing_entries, response_entries),
                },
            )
        return response_entries

    def run_baselines(self, *, dry_run: bool = False) -> list[dict[str, Any]]:
        adapters = build_baseline_adapters(self.repo_root)
        prompt_bundle = self._load_prompt_bundle(dry_run=dry_run)
        datasets = {dataset.name: dataset for dataset in self.spec.datasets}
        evaluator = EvalScriptEvaluator(repo_root=self.repo_root)
        response_entries: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for baseline in self.spec.baselines:
            adapter = adapters.get(baseline.name)
            if adapter is None:
                raise RuntimeError(
                    f"Baseline `{baseline.name}` is enabled in config `{self.spec.name}` but no adapter is registered."
                )

            ordered_entries = sorted(
                prompt_bundle["entries"],
                key=lambda entry: (
                    0 if (_baseline_prefers_anonymized(baseline.name) and entry.get("naming_regime") == "anonymized") else 1,
                    str(entry.get("dataset")),
                    str(entry.get("config_name")),
                ),
            )

            for entry in ordered_entries:
                if entry["kind"] == "names_only":
                    cell = PromptCellSpec(
                        style="names_only",
                        obs_per_prompt=0,
                        int_per_combo=0,
                        anonymize=False,
                        enabled=True,
                    )
                else:
                    cell = next(
                        cell
                        for cell in self.spec.prompt_cells
                        if cell.style == entry["prompt_style"]
                        and cell.obs_per_prompt == entry["obs_n"]
                        and cell.int_per_combo == entry["int_n"]
                        and cell.naming_regime == entry["naming_regime"]
                        and cell.reasoning_guidance == entry.get("reasoning_guidance", "staged")
                    )
                if not adapter.applies_to(baseline, cell):
                    continue
                dataset = datasets[entry["dataset"]]
                graph_path = Path(entry["graph_file"])
                dedupe_fn = getattr(adapter, "dedupe_key", None)
                if callable(dedupe_fn):
                    dedupe_key = dedupe_fn(baseline=baseline, dataset=dataset, cell=cell, entry=entry)
                else:
                    dedupe_key = (baseline.name, dataset.name, cell.obs_per_prompt, cell.int_per_combo)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                response_csv = adapter.run(
                    baseline=baseline,
                    dataset=dataset,
                    graph_path=graph_path,
                    cell=cell,
                    spec=self.spec,
                    dry_run=dry_run,
                )
                per_row_path = _per_row_metrics_path(response_csv)
                if not _reuse_artifact_if_current(per_row_path, source=response_csv, dry_run=dry_run, label="eval"):
                    _run(evaluator.build_eval_command(response_csv=response_csv, evaluator=self.spec.evaluator), dry_run=dry_run)
                response_entries.append(
                    self._response_record(
                        entry=entry,
                        response_csv=response_csv,
                        system=baseline.name,
                        system_kind="baseline",
                        extra={
                            "baseline": baseline.name,
                            "library": baseline.library or "internal",
                            "takayama_backend": baseline.takayama_backend if baseline.name == "TakayamaSCP" else None,
                            "takayama_pattern": baseline.takayama_pattern if baseline.name == "TakayamaSCP" else None,
                        },
                    )
                )
        if not dry_run:
            bundle_path = self.run_root / "response_bundle.json"
            existing_entries: list[dict[str, Any]] = []
            if bundle_path.exists():
                existing_entries = json.loads(bundle_path.read_text(encoding="utf-8")).get("entries", [])
            _json_dump(
                bundle_path,
                {
                    "schema_version": "response_bundle/v1",
                    "benchmark": self.spec.name,
                    "entries": _merge_response_entries(existing_entries, response_entries),
                },
            )
        return response_entries

    def _load_prompt_bundle(self, *, dry_run: bool) -> dict[str, Any]:
        bundle_path = self.run_root / "prompt_bundle.json"
        if bundle_path.exists():
            return json.loads(bundle_path.read_text(encoding="utf-8"))
        if dry_run:
            return self.build(dry_run=True)
        raise FileNotFoundError(
            f"Prompt bundle not found at {bundle_path}. Run the `build` step first."
        )

    def _response_record(
        self,
        *,
        entry: dict[str, Any],
        response_csv: Path,
        system: str,
        system_kind: str,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": "run_provenance/v1",
            "config_hash": _config_hash(self.spec),
            "parser_version": self.spec.provenance.parser_version,
            "evaluator_version": self.spec.provenance.evaluator_version,
            "benchmark": self.spec.name,
            "dataset": entry["dataset"],
            "config_name": entry["config_name"],
            "prompt_style": entry["prompt_style"],
            "naming_regime": entry["naming_regime"],
            "reasoning_guidance": entry.get("reasoning_guidance", "staged"),
            "obs_n": entry["obs_n"],
            "int_n": entry["int_n"],
            "system": system,
            "system_kind": system_kind,
            "prompt_storage": entry.get("prompt_storage", self.spec.execution.prompt_storage),
            "prompt_basename": entry.get("prompt_basename"),
            "prompt_csv": entry.get("prompt_csv"),
            "prompt_csv_sha256": entry.get("prompt_csv_sha256"),
            "response_csv": str(response_csv),
            "response_csv_sha256": file_sha256(response_csv) if response_csv.exists() else None,
            **extra,
        }

    def summarize(self) -> dict[str, Path]:
        bundle_path = self.run_root / "response_bundle.json"
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        summary_rows: list[dict[str, Any]] = []
        consensus_rows: list[dict[str, Any]] = []
        for entry in bundle.get("entries", []):
            response_csv = Path(entry["response_csv"])
            if not response_csv.exists():
                continue
            metrics = direct_csv_summary(response_csv=response_csv, evaluator=self.spec.evaluator)
            base = {
                "benchmark": self.spec.name,
                "dataset": entry["dataset"],
                "config_name": entry["config_name"],
                "prompt_style": entry["prompt_style"],
                "naming_regime": entry["naming_regime"],
                "reasoning_guidance": entry.get("reasoning_guidance", "staged"),
                "obs_n": entry["obs_n"],
                "int_n": entry["int_n"],
                "system": entry["system"],
                "system_kind": entry["system_kind"],
                "model": entry.get("model"),
                "provider": entry.get("provider"),
                "baseline": entry.get("baseline"),
                "takayama_backend": entry.get("takayama_backend"),
                "takayama_pattern": entry.get("takayama_pattern"),
                "response_csv": entry["response_csv"],
            }
            summary_rows.append({**base, **metrics})
            consensus_rows.append(
                {
                    **base,
                    "consensus_tau": metrics.get("consensus_tau"),
                    "consensus_f1": metrics.get("consensus_f1"),
                    "consensus_shd": metrics.get("consensus_shd"),
                    "consensus_accuracy": metrics.get("consensus_accuracy"),
                    "consensus_precision": metrics.get("consensus_precision"),
                    "consensus_recall": metrics.get("consensus_recall"),
                    "nhd_consensus": metrics.get("nhd_consensus"),
                    "nhd_ratio_consensus": metrics.get("nhd_ratio_consensus"),
                }
            )

        eval_csv = self.run_root / "evaluation_summary.csv"
        cons_csv = self.run_root / "consensus_summary.csv"
        audit_csv = self.run_root / "contamination_audit.csv"
        write_csv(eval_csv, summary_rows)
        write_csv(cons_csv, consensus_rows)
        write_csv(audit_csv, contamination_audit(summary_rows))
        return {
            "evaluation_summary": eval_csv,
            "consensus_summary": cons_csv,
            "contamination_audit": audit_csv,
        }


def load_runner(identifier: str) -> BenchmarkRunner:
    spec = load_benchmark_spec(BenchmarkRegistry().resolve(identifier))
    return BenchmarkRunner(spec=spec)
