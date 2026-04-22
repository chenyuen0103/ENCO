#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path

from benchmark_builder.baselines import build_baseline_adapters
from benchmark_builder.registry import BenchmarkRegistry
from benchmark_builder.runner import _prompt_base_name
from benchmark_builder.schema import BaselineSpec, PromptCellSpec, load_benchmark_spec


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
        self.assertEqual(spec.baselines[0].name, "ENCO")

    def test_registry_resolves_named_manifest(self) -> None:
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
            if cell.style == "summary_joint"
            and cell.obs_per_prompt == 100
            and cell.int_per_combo == 0
            and not cell.anonymize
        )
        self.assertEqual(
            _prompt_base_name(cell=cell, num_prompts=spec.num_prompts, shuffles_per_graph=spec.shuffles_per_graph),
            "prompts_obs100_int0_shuf1_p5_thinktags_summary_joint",
        )

    def test_external_llm_adapters_bind_to_expected_configs(self) -> None:
        adapters = build_baseline_adapters(Path(".").resolve())
        names_only = PromptCellSpec(style="names_only", obs_per_prompt=0, int_per_combo=0)
        observational = PromptCellSpec(style="summary_joint", obs_per_prompt=100, int_per_combo=0)
        summary = PromptCellSpec(style="summary_joint", obs_per_prompt=100, int_per_combo=50)
        matrix = PromptCellSpec(style="matrix", obs_per_prompt=100, int_per_combo=50)
        self.assertFalse(adapters["TakayamaSCP"].applies_to(BaselineSpec(name="TakayamaSCP"), names_only))
        self.assertTrue(adapters["TakayamaSCP"].applies_to(BaselineSpec(name="TakayamaSCP"), observational))
        self.assertFalse(adapters["JiralerspongBFS"].applies_to(BaselineSpec(name="JiralerspongBFS"), names_only))
        self.assertTrue(adapters["JiralerspongBFS"].applies_to(BaselineSpec(name="JiralerspongBFS"), observational))
        self.assertTrue(adapters["CausalLLMPrompt"].applies_to(BaselineSpec(name="CausalLLMPrompt"), names_only))
        self.assertFalse(adapters["TakayamaSCP"].applies_to(BaselineSpec(name="TakayamaSCP"), summary))
        self.assertFalse(adapters["JiralerspongBFS"].applies_to(BaselineSpec(name="JiralerspongBFS"), summary))
        self.assertTrue(adapters["CausalLLMData"].applies_to(BaselineSpec(name="CausalLLMData"), summary))
        self.assertFalse(adapters["CausalLLMData"].applies_to(BaselineSpec(name="CausalLLMData"), matrix))


if __name__ == "__main__":
    unittest.main()
