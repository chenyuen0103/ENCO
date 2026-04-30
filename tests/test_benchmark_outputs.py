#!/usr/bin/env python3
from __future__ import annotations

import unittest

from benchmark_builder.evaluation import attach_contrastive_metrics, contamination_audit


class TestBenchmarkOutputs(unittest.TestCase):
    def test_contamination_audit_computes_gaps(self) -> None:
        rows = [
            {"dataset": "sachs", "prompt_style": "summary", "model": "gpt-5-mini", "system": "gpt-5-mini", "naming_regime": "real", "obs_n": 100, "int_n": 50, "consensus_f1": 0.7, "avg_skeleton_f1": 0.6},
            {"dataset": "sachs", "prompt_style": "summary", "model": "gpt-5-mini", "system": "gpt-5-mini", "naming_regime": "anonymized", "obs_n": 100, "int_n": 50, "consensus_f1": 0.5, "avg_skeleton_f1": 0.45},
            {"dataset": "sachs", "prompt_style": "summary", "model": "gpt-5-mini", "system": "gpt-5-mini", "naming_regime": "names_only", "obs_n": 100, "int_n": 50, "consensus_f1": 0.2, "avg_skeleton_f1": 0.3},
        ]
        audit = contamination_audit(rows)
        self.assertEqual(len(audit), 1)
        self.assertAlmostEqual(audit[0]["real_minus_anonymized"], 0.2)
        self.assertAlmostEqual(audit[0]["real_minus_names_only"], 0.5)
        self.assertAlmostEqual(audit[0]["real_minus_anonymized_avg_skeleton_f1"], 0.15)
        self.assertAlmostEqual(audit[0]["real_minus_names_only_avg_skeleton_f1"], 0.3)

    def test_attach_contrastive_metrics_adds_named_representation_and_budget_gaps(self) -> None:
        rows = [
            {
                "dataset": "sachs",
                "model": "gpt-5-mini",
                "parsed_model": "gpt-5-mini",
                "prompt_style": "names_only",
                "is_names_only": 1,
                "anonymize": 0,
                "avg_f1": 0.2,
                "avg_skeleton_f1": 0.25,
                "avg_ancestor_f1": 0.3,
                "acyclic_rate": 1.0,
            },
            {
                "dataset": "sachs",
                "model": "gpt-5-mini",
                "parsed_model": "gpt-5-mini",
                "prompt_style": "summary",
                "is_names_only": 0,
                "anonymize": 0,
                "obs_n": 100,
                "int_n": 50,
                "avg_f1": 0.5,
                "avg_skeleton_f1": 0.55,
                "avg_ancestor_f1": 0.6,
                "acyclic_rate": 1.0,
            },
            {
                "dataset": "sachs",
                "model": "gpt-5-mini",
                "parsed_model": "gpt-5-mini",
                "prompt_style": "summary",
                "is_names_only": 0,
                "anonymize": 1,
                "obs_n": 100,
                "int_n": 50,
                "avg_f1": 0.35,
                "avg_skeleton_f1": 0.4,
                "avg_ancestor_f1": 0.45,
                "acyclic_rate": 0.9,
            },
            {
                "dataset": "sachs",
                "model": "gpt-5-mini",
                "parsed_model": "gpt-5-mini",
                "prompt_style": "matrix",
                "is_names_only": 0,
                "anonymize": 0,
                "obs_n": 100,
                "int_n": 50,
                "avg_f1": 0.42,
                "avg_skeleton_f1": 0.46,
                "avg_ancestor_f1": 0.5,
                "acyclic_rate": 0.95,
            },
            {
                "dataset": "sachs",
                "model": "gpt-5-mini",
                "parsed_model": "gpt-5-mini",
                "prompt_style": "summary",
                "is_names_only": 0,
                "anonymize": 0,
                "obs_n": 200,
                "int_n": 50,
                "avg_f1": 0.58,
                "avg_skeleton_f1": 0.62,
                "avg_ancestor_f1": 0.66,
                "acyclic_rate": 1.0,
            },
            {
                "dataset": "sachs",
                "model": "gpt-5-mini",
                "parsed_model": "gpt-5-mini",
                "prompt_style": "summary",
                "is_names_only": 0,
                "anonymize": 0,
                "obs_n": 200,
                "int_n": 100,
                "avg_f1": 0.63,
                "avg_skeleton_f1": 0.68,
                "avg_ancestor_f1": 0.7,
                "acyclic_rate": 1.0,
            },
        ]

        attach_contrastive_metrics(rows)

        summary_real = rows[1]
        self.assertEqual(summary_real["naming_regime"], "real")
        self.assertAlmostEqual(summary_real["names_only_avg_f1"], 0.2)
        self.assertAlmostEqual(summary_real["avg_f1_minus_names_only"], 0.3)
        self.assertAlmostEqual(summary_real["real_minus_anonymized_avg_f1"], 0.15)
        self.assertAlmostEqual(summary_real["summary_minus_matrix_avg_f1"], 0.08)

        higher_obs = rows[4]
        self.assertEqual(higher_obs["prev_obs_n"], 100)
        self.assertAlmostEqual(higher_obs["obs_budget_gain_avg_f1"], 0.08)

        higher_int = rows[5]
        self.assertEqual(higher_int["prev_int_n"], 50)
        self.assertAlmostEqual(higher_int["int_budget_gain_avg_f1"], 0.05)


if __name__ == "__main__":
    unittest.main()
