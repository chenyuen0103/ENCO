#!/usr/bin/env python3
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmark_builder.baselines import build_baseline_adapters
from benchmark_builder.registry import BenchmarkRegistry
from benchmark_builder.runner import _prompt_base_name
from benchmark_builder.schema import BaselineSpec, DatasetSpec, PromptCellSpec, load_benchmark_spec


class TestBenchmarkSpec(unittest.TestCase):
    def test_reference_suite_loads(self) -> None:
        spec = load_benchmark_spec("benchmark_specs/reference_suite.json")
        self.assertEqual(spec.name, "reference_suite")
        self.assertEqual([dataset.name for dataset in spec.datasets], ["sachs", "child", "alarm"])
        self.assertEqual(len(spec.models), 4)
        self.assertTrue(spec.names_only.enabled)
        self.assertEqual([baseline.name for baseline in spec.baselines], ["PC", "GES", "ENCO"])
        pc = next(b for b in spec.baselines if b.name == "PC")
        ges = next(b for b in spec.baselines if b.name == "GES")
        self.assertEqual(pc.pc_variant, "stable")
        self.assertEqual(pc.pc_ci_test, "chi_square")
        self.assertEqual(ges.ges_scoring_method, "bic-d")

    def test_legacy_paper_slice_normalizes(self) -> None:
        spec = load_benchmark_spec("paper_slices/sachs_main.json")
        self.assertEqual(len(spec.datasets), 1)
        self.assertEqual(spec.datasets[0].name, "sachs")
        self.assertEqual(spec.models[0].name, "gpt-5-mini")
        self.assertEqual(
            [baseline.name for baseline in spec.baselines],
            ["PC", "GES", "ENCO", "CausalLLMPrompt", "JiralerspongBFS", "TakayamaSCP"],
        )

    def test_registry_resolves_named_config(self) -> None:
        registry = BenchmarkRegistry()
        path = registry.resolve("authoring_demo")
        self.assertTrue(str(path).endswith("benchmark_specs/authoring_demo.json"))

    def test_authoring_demo_uses_in_memory_execution(self) -> None:
        spec = load_benchmark_spec("benchmark_specs/authoring_demo.json")
        self.assertEqual(spec.execution.prompt_storage, "in_memory")
        self.assertEqual(spec.execution.prompt_retention, "example")

    def test_prompt_basename_matches_generator_convention(self) -> None:
        spec = load_benchmark_spec("benchmark_specs/reference_suite.json")
        cell = next(
            cell
            for cell in spec.prompt_cells
            if cell.style == "summary"
            and cell.obs_per_prompt == 100
            and cell.int_per_combo == 0
            and not cell.anonymize
        )
        self.assertEqual(
            _prompt_base_name(cell=cell, num_prompts=spec.num_prompts, shuffles_per_graph=spec.shuffles_per_graph),
            "prompts_obs100_int0_shuf1_p5_reasonconcise_summary",
        )

    def test_external_llm_adapters_bind_to_expected_configs(self) -> None:
        adapters = build_baseline_adapters(Path(".").resolve())
        names_only = PromptCellSpec(style="names_only", obs_per_prompt=0, int_per_combo=0)
        observational = PromptCellSpec(style="summary", obs_per_prompt=100, int_per_combo=0)
        anonymized_observational = PromptCellSpec(
            style="summary",
            obs_per_prompt=100,
            int_per_combo=0,
            anonymize=True,
        )
        summary = PromptCellSpec(style="summary", obs_per_prompt=100, int_per_combo=50)
        matrix = PromptCellSpec(style="matrix", obs_per_prompt=100, int_per_combo=50)
        self.assertFalse(adapters["TakayamaSCP"].applies_to(BaselineSpec(name="TakayamaSCP"), names_only))
        self.assertTrue(adapters["TakayamaSCP"].applies_to(BaselineSpec(name="TakayamaSCP"), observational))
        self.assertFalse(adapters["TakayamaSCP"].applies_to(BaselineSpec(name="TakayamaSCP"), anonymized_observational))
        self.assertFalse(adapters["JiralerspongPairwise"].applies_to(BaselineSpec(name="JiralerspongPairwise"), names_only))
        self.assertTrue(adapters["JiralerspongPairwise"].applies_to(BaselineSpec(name="JiralerspongPairwise"), observational))
        self.assertFalse(adapters["JiralerspongPairwise"].applies_to(BaselineSpec(name="JiralerspongPairwise"), anonymized_observational))
        self.assertFalse(adapters["JiralerspongBFS"].applies_to(BaselineSpec(name="JiralerspongBFS"), names_only))
        self.assertTrue(adapters["JiralerspongBFS"].applies_to(BaselineSpec(name="JiralerspongBFS"), observational))
        self.assertFalse(adapters["JiralerspongBFS"].applies_to(BaselineSpec(name="JiralerspongBFS"), anonymized_observational))
        self.assertTrue(adapters["CausalLLMPrompt"].applies_to(BaselineSpec(name="CausalLLMPrompt"), names_only))
        self.assertFalse(adapters["TakayamaSCP"].applies_to(BaselineSpec(name="TakayamaSCP"), summary))
        self.assertFalse(adapters["JiralerspongBFS"].applies_to(BaselineSpec(name="JiralerspongBFS"), summary))
        self.assertFalse(adapters["JiralerspongPairwise"].applies_to(BaselineSpec(name="JiralerspongPairwise"), summary))
        self.assertTrue(adapters["CausalLLMData"].applies_to(BaselineSpec(name="CausalLLMData"), summary))
        self.assertFalse(adapters["CausalLLMData"].applies_to(BaselineSpec(name="CausalLLMData"), anonymized_observational))
        self.assertFalse(adapters["CausalLLMData"].applies_to(BaselineSpec(name="CausalLLMData"), matrix))

    def test_causal_llm_data_reuses_legacy_summary_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            responses_dir = repo_root / "experiments" / "responses" / "sachs"
            responses_dir.mkdir(parents=True)
            legacy_csv = responses_dir / "responses_obs100_int0_shuf1_p3_summary_joint_gpt-5-mini.csv"
            with legacy_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "answer",
                        "prediction",
                        "raw_response",
                        "valid",
                        "data_idx",
                        "shuffle_idx",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "answer": "[[0, 1], [0, 0]]",
                        "prediction": "[[0, 1], [0, 0]]",
                        "raw_response": '{"adjacency_matrix": [[0, 1], [0, 0]]}',
                        "valid": "1",
                        "data_idx": "0",
                        "shuffle_idx": "0",
                    }
                )

            adapter = build_baseline_adapters(repo_root)["CausalLLMData"]
            out_csv = adapter.run(
                baseline=BaselineSpec(name="CausalLLMData", model="gpt-5-mini", provider="openai"),
                dataset=DatasetSpec(name="sachs", graph_source="bif", graph_path="/tmp/sachs.bif"),
                graph_path=Path("/tmp/sachs.bif"),
                cell=PromptCellSpec(style="summary", obs_per_prompt=100, int_per_combo=0),
                spec=SimpleNamespace(seed=42, models=[SimpleNamespace(name="gpt-5-mini", enabled=True)]),
                dry_run=False,
            )
            self.assertTrue(out_csv.exists())
            with out_csv.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["method"], "CausalLLMData")
            self.assertEqual(row["model"], "gpt-5-mini")
            self.assertEqual(row["provider"], "openai")
            self.assertEqual(row["naming_regime"], "real")
            self.assertEqual(row["obs_n"], "100")
            self.assertEqual(row["int_n"], "0")

    def test_causal_llm_prompt_reuses_legacy_names_only_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            responses_dir = repo_root / "experiments" / "responses" / "sachs"
            responses_dir.mkdir(parents=True)
            legacy_csv = responses_dir / "responses_names_only_gpt-5-mini.csv"
            with legacy_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "answer",
                        "prediction",
                        "raw_response",
                        "valid",
                        "data_idx",
                        "shuffle_idx",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "answer": "[[0, 1], [0, 0]]",
                        "prediction": "[[0, 1], [0, 0]]",
                        "raw_response": "A -> B",
                        "valid": "1",
                        "data_idx": "0",
                        "shuffle_idx": "0",
                    }
                )

            adapter = build_baseline_adapters(repo_root)["CausalLLMPrompt"]
            out_csv = adapter.run(
                baseline=BaselineSpec(name="CausalLLMPrompt", model="gpt-5-mini", provider="openai"),
                dataset=DatasetSpec(name="sachs", graph_source="bif", graph_path="/tmp/sachs.bif"),
                graph_path=Path("/tmp/sachs.bif"),
                cell=PromptCellSpec(style="names_only", obs_per_prompt=0, int_per_combo=0),
                spec=SimpleNamespace(seed=42, models=[SimpleNamespace(name="gpt-5-mini", enabled=True)]),
                dry_run=False,
            )
            self.assertTrue(out_csv.exists())
            with out_csv.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["method"], "CausalLLMPrompt")
            self.assertEqual(row["model"], "gpt-5-mini")
            self.assertEqual(row["provider"], "openai")
            self.assertEqual(row["naming_regime"], "names_only")
            self.assertEqual(row["obs_n"], "0")
            self.assertEqual(row["int_n"], "0")


if __name__ == "__main__":
    unittest.main()
