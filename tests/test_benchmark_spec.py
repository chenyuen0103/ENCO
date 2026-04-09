#!/usr/bin/env python3
from __future__ import annotations

import unittest

from benchmark_builder.registry import BenchmarkRegistry
from benchmark_builder.schema import load_benchmark_spec


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


if __name__ == "__main__":
    unittest.main()
