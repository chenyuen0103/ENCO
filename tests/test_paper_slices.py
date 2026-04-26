#!/usr/bin/env python3
"""Validate frozen paper slice configs."""

from __future__ import annotations

import unittest
from pathlib import Path

from benchmark_builder.schema import load_benchmark_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
SLICES_DIR = REPO_ROOT / "paper_slices"


class TestPaperSlices(unittest.TestCase):
    def test_configs_load(self) -> None:
        configs = sorted(SLICES_DIR.glob("*.json"))
        self.assertGreaterEqual(len(configs), 3)
        for path in configs:
            spec = load_benchmark_spec(path)
            self.assertTrue(spec.name)
            self.assertEqual(len(spec.datasets), 1)
            self.assertTrue(spec.models)
            self.assertTrue(spec.role)

    def test_sachs_main_matches_current_paper_roster(self) -> None:
        spec = load_benchmark_spec(SLICES_DIR / "sachs_main.json")
        self.assertEqual(spec.datasets[0].name, "sachs")
        self.assertTrue(spec.names_only.enabled)
        baseline_names = [baseline.name for baseline in spec.baselines]
        self.assertEqual(
            baseline_names,
            ["PC", "GES", "ENCO", "CausalLLMPrompt", "JiralerspongBFS", "TakayamaSCP"],
        )

        styles = [cell.style for cell in spec.prompt_cells]
        self.assertEqual(styles, ["summary", "matrix", "summary", "summary"])
        anonymized = [cell.anonymize for cell in spec.prompt_cells]
        self.assertEqual(anonymized, [False, False, False, True])
        ints = [cell.int_per_combo for cell in spec.prompt_cells]
        self.assertEqual(ints, [50, 50, 0, 0])

        scope_by_name = {baseline.name: baseline.scope for baseline in spec.baselines}
        self.assertEqual(scope_by_name["PC"], "observational")
        self.assertEqual(scope_by_name["GES"], "observational")
        self.assertEqual(scope_by_name["ENCO"], "interventional")
        self.assertEqual(scope_by_name["TakayamaSCP"], "observational")

    def test_cancer_configs_are_smoke_only(self) -> None:
        for name in ["cancer_smoke_summary.json", "cancer_smoke_names_only.json"]:
            spec = load_benchmark_spec(SLICES_DIR / name)
            self.assertEqual(spec.datasets[0].name, "cancer")
            self.assertEqual(spec.role, "smoke")

    def test_compact_breadth_slices_exist(self) -> None:
        expected = {
            "asia_compact.json": ("asia", ["PC", "GES", "ENCO", "CausalLLMPrompt", "TakayamaSCP"]),
            "child_compact.json": ("child", ["PC", "GES", "ENCO", "CausalLLMPrompt"]),
            "alarm_compact.json": ("alarm", ["PC", "GES", "ENCO", "CausalLLMPrompt"]),
        }
        for filename, (dataset_name, baseline_names) in expected.items():
            spec = load_benchmark_spec(SLICES_DIR / filename)
            self.assertEqual(spec.datasets[0].name, dataset_name)
            self.assertEqual(spec.role, "control")
            self.assertTrue(spec.names_only.enabled)
            self.assertEqual([baseline.name for baseline in spec.baselines], baseline_names)


if __name__ == "__main__":
    unittest.main()
