#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from experiments.run_takayama_scd import (
    _build_first_prompt,
    _probability_to_pk,
    _directed_prediction_from_pc_signed,
)


class TestTakayamaSCD(unittest.TestCase):
    def test_probability_to_pk_thresholds(self) -> None:
        probs = np.array(
            [
                [0.0, 0.96, 0.5],
                [0.02, 0.0, 0.99],
                [0.5, 0.01, 0.0],
            ]
        )
        pk = _probability_to_pk(probs)
        self.assertEqual(pk[0, 1], 1)
        self.assertEqual(pk[1, 0], 0)
        self.assertEqual(pk[0, 2], -1)
        self.assertEqual(pk[2, 1], 0)

    def test_directed_prediction_from_pc_signed_transposes_orientation(self) -> None:
        signed = np.array(
            [
                [0, 1, 0],
                [0, 0, 0],
                [1, 0, 0],
            ],
            dtype=int,
        )
        pred = _directed_prediction_from_pc_signed(signed)
        self.assertEqual(pred.tolist(), [[0, 0, 1], [1, 0, 0], [0, 0, 0]])

    def test_first_prompt_pattern2_mentions_bootstrap_probabilities(self) -> None:
        signed = np.zeros((2, 2), dtype=int)
        directed = np.array([[0.0, 0.9], [0.0, 0.0]])
        undirected = np.zeros((2, 2), dtype=float)
        prompt = _build_first_prompt(
            dataset_name="toy",
            labels=["A", "B"],
            i=0,
            j=1,
            pattern=2,
            pc_adj_signed=signed,
            pc_directed_prob=directed,
            pc_undirected_prob=undirected,
            anonymized=False,
        )
        self.assertIn("bootstrap probability", prompt)
        self.assertIn("PC(Peter-Clerk)", prompt)


if __name__ == "__main__":
    unittest.main()
