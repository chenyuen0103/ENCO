#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from experiments.run_external_llm_baselines import (
    _aggregate_sampled_dags,
    _extract_name_list,
    _project_dag,
)


class TestExternalLLMBaselines(unittest.TestCase):
    def test_extract_name_list_reads_answer_block_json(self) -> None:
        text = '<think>brief</think><answer>{"roots": ["A", "B", "junk"]}</answer>'
        self.assertEqual(_extract_name_list(text, key="roots", allowed=["A", "B", "C"]), ["A", "B"])

    def test_project_dag_breaks_simple_cycle(self) -> None:
        mat = np.array(
            [
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
            ],
            dtype=int,
        )
        dag = _project_dag(mat)
        self.assertEqual(int(dag[0, 1]), 1)
        self.assertEqual(int(dag[1, 2]), 1)
        self.assertEqual(int(dag[2, 0]), 0)

    def test_aggregate_sampled_dags_uses_threshold_and_dag_projection(self) -> None:
        mats = [
            np.array([[0, 1], [0, 0]], dtype=int),
            np.array([[0, 1], [0, 0]], dtype=int),
            np.array([[0, 0], [1, 0]], dtype=int),
        ]
        dag = _aggregate_sampled_dags(mats, threshold=0.5)
        self.assertEqual(dag.tolist(), [[0, 1], [0, 0]])


if __name__ == "__main__":
    unittest.main()
