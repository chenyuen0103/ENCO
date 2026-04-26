#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments import run_external_llm_baselines as external
from experiments.run_external_llm_baselines import (
    _aggregate_sampled_dags,
    _bfs_children_prompt,
    _bfs_roots_prompt,
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

    def test_bfs_prompts_include_evidence_context_when_provided(self) -> None:
        roots = _bfs_roots_prompt(dataset_name="sachs", remaining=["A", "B"], context="OBS")
        children = _bfs_children_prompt(dataset_name="sachs", source="A", candidates=["B"], context="OBS")
        self.assertIn("OBS", roots)
        self.assertIn("observational evidence summary", roots)
        self.assertIn("OBS", children)

    def test_run_jiralerspong_bfs_uses_summary_context(self) -> None:
        with mock.patch.object(
            external,
            "_build_data_prompt",
            return_value=(["A", "B"], {}, "OBS"),
        ) as build_data, mock.patch.object(
            external,
            "_call_model",
            side_effect=[
                '<answer>{"roots": ["A"]}</answer>',
                '<answer>{"children": ["B"]}</answer>',
            ],
        ) as call_model:
            adj, variables, _raw = external._run_jiralerspong_bfs(
                graph_path=Path("/tmp/sachs.bif"),
                sample_size_obs=100,
                sample_size_inters=0,
                prompt_mode="summary",
                model_name="gpt-5-mini",
                provider="openai",
                temperature=0.0,
                max_new_tokens=None,
                seed=0,
                anonymize=False,
                hf_pipe=None,
            )
        build_data.assert_called_once()
        first_prompt = call_model.call_args_list[0].kwargs["prompt"]
        self.assertIn("OBS", first_prompt)
        self.assertEqual(variables, ["A", "B"])
        self.assertEqual(adj.tolist(), [[0, 1], [0, 0]])


if __name__ == "__main__":
    unittest.main()
