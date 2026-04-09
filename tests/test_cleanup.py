#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path

from benchmark_builder.cleanup import collect_prompt_cleanup_targets
from benchmark_builder.schema import load_benchmark_spec


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPromptCleanup(unittest.TestCase):
    def test_reference_suite_targets_benchmark_prompt_dir_only_by_default(self) -> None:
        spec = load_benchmark_spec(REPO_ROOT / "benchmark_specs" / "reference_suite.json")
        targets = collect_prompt_cleanup_targets(spec=spec, repo_root=REPO_ROOT, include_examples=False)
        self.assertEqual(len(targets), 1)
        self.assertTrue(str(targets[0].path).endswith("experiments/prompts/benchmarks/reference_suite"))

    def test_authoring_demo_example_prompt_targets_are_included(self) -> None:
        spec = load_benchmark_spec(REPO_ROOT / "benchmark_specs" / "authoring_demo.json")
        targets = collect_prompt_cleanup_targets(spec=spec, repo_root=REPO_ROOT, include_examples=True)
        labels = {target.label for target in targets}
        self.assertIn("benchmark_prompts", labels)
        self.assertIn("example_prompts:asia_demo", labels)
        self.assertIn("example_prompts:synthetic_demo_10", labels)


if __name__ == "__main__":
    unittest.main()
