#!/usr/bin/env python3
from __future__ import annotations

import unittest

from benchmark_builder.evaluation import contamination_audit


class TestBenchmarkOutputs(unittest.TestCase):
    def test_contamination_audit_computes_gaps(self) -> None:
        rows = [
            {"dataset": "sachs", "prompt_style": "summary_joint", "model": "gpt-5-mini", "system": "gpt-5-mini", "naming_regime": "real", "obs_n": 100, "int_n": 50, "consensus_f1": 0.7},
            {"dataset": "sachs", "prompt_style": "summary_joint", "model": "gpt-5-mini", "system": "gpt-5-mini", "naming_regime": "anonymized", "obs_n": 100, "int_n": 50, "consensus_f1": 0.5},
            {"dataset": "sachs", "prompt_style": "summary_joint", "model": "gpt-5-mini", "system": "gpt-5-mini", "naming_regime": "names_only", "obs_n": 100, "int_n": 50, "consensus_f1": 0.2}
        ]
        audit = contamination_audit(rows)
        self.assertEqual(len(audit), 1)
        self.assertAlmostEqual(audit[0]["real_minus_anonymized"], 0.2)
        self.assertAlmostEqual(audit[0]["real_minus_names_only"], 0.5)


if __name__ == "__main__":
    unittest.main()
