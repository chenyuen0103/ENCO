#!/usr/bin/env python3
from __future__ import annotations

import json
import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments import run_external_llm_baselines as external
from experiments.run_external_llm_baselines import (
    _aggregate_sampled_dags,
    _bfs_corr_prompt,
    _bfs_initial_messages,
    _extract_name_list,
    _extract_pairwise_choice,
    _pairwise_prompt,
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

    def test_bfs_initial_messages_match_reference_prompts(self) -> None:
        asia = _bfs_initial_messages("asia")
        self.assertEqual(asia[0]["content"], "You are an expert on lung diseases.")
        self.assertIn("construct a causal graph", asia[1]["content"])
        sachs = _bfs_initial_messages("sachs")
        self.assertEqual(sachs[0]["content"], "You are an expert on intracellular protein signaling pathways.")
        self.assertIn("intracellular protein signaling research", sachs[1]["content"])

    def test_bfs_corr_prompt_formats_pairwise_correlations(self) -> None:
        corr = np.array([[1.0, 0.25], [0.25, 1.0]], dtype=float)
        prompt = _bfs_corr_prompt("A", corr, ["A", "B"])
        self.assertIn("Pearson correlation coefficient between A and other variables", prompt)
        self.assertIn("B: 0.25", prompt)

    def test_pairwise_prompt_includes_pearson_when_provided(self) -> None:
        prompt = _pairwise_prompt(
            user_prompt="You are a helpful assistant to a lung disease expert.",
            src="A",
            dst="B",
            all_variables=[
                {"name": "A", "symbol": "A", "description": "desc A"},
                {"name": "B", "symbol": "B", "description": "desc B"},
            ],
            known_edges_text='- "A" causes "C".',
            pearson_corr=0.25,
        )
        self.assertIn("Here are the causal relationships you know so far", prompt)
        self.assertIn("Pearson correlation coefficient", prompt)
        self.assertIn("0.25", prompt)

    def test_extract_pairwise_choice_prefers_answer_block(self) -> None:
        self.assertEqual(_extract_pairwise_choice("<Answer>B</Answer>"), "B")

    def test_run_jiralerspong_bfs_uses_summary_context(self) -> None:
        call_messages = mock.Mock(
            side_effect=[
                "<Answer>A</Answer>",
                "<Answer>B</Answer>",
                "<Answer></Answer>",
            ],
        )
        with mock.patch.dict(
            external._run_jiralerspong_bfs.__globals__,
            {
                "_load_variable_metadata": mock.Mock(
                    return_value=[
                        {"name": "A", "symbol": "A", "description": "desc A"},
                        {"name": "B", "symbol": "B", "description": "desc B"},
                    ]
                ),
                "_load_observational_array": mock.Mock(
                    return_value=(np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float), ["A", "B"])
                ),
                "_call_model_messages": call_messages,
            },
        ):
            adj, variables, raw = external._run_jiralerspong_bfs(
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
        first_messages = call_messages.call_args_list[0].kwargs["messages"]
        self.assertIn("desc A", first_messages[1]["content"])
        root_stage = json.loads(raw[1])
        self.assertIn("Select variables that are caused by A.", root_stage["prompt"])
        self.assertIn("B: 1.00", root_stage["prompt"])
        self.assertEqual(variables, ["A", "B"])
        self.assertEqual(adj.tolist(), [[0, 1], [0, 0]])

    def test_run_jiralerspong_pairwise_uses_observational_stats_in_summary_mode(self) -> None:
        call_messages = mock.Mock(return_value="<Answer>A</Answer>")
        with mock.patch.dict(
            external._run_jiralerspong_pairwise.__globals__,
            {
                "_load_observational_array": mock.Mock(
                    return_value=(np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float), ["A", "B"])
                ),
                "_load_variable_metadata": mock.Mock(
                    return_value=[
                        {"name": "A", "symbol": "A", "description": "desc A"},
                        {"name": "B", "symbol": "B", "description": "desc B"},
                    ]
                ),
                "_call_model_messages": call_messages,
            },
        ):
            adj, variables, raw = external._run_jiralerspong_pairwise(
                graph_path=Path("/tmp/sachs.bif"),
                sample_size_obs=100,
                prompt_mode="summary",
                model_name="gpt-5-mini",
                provider="openai",
                temperature=0.0,
                max_new_tokens=None,
                seed=0,
                anonymize=False,
                hf_pipe=None,
            )
        messages = call_messages.call_args.kwargs["messages"]
        prompt = messages[-1]["content"]
        self.assertEqual(messages[0]["content"], "You are an expert on intracellular protein signaling pathways.")
        self.assertIn("Here are the causal relationships you know so far", prompt)
        self.assertIn("Pearson correlation coefficient", prompt)
        self.assertEqual(variables, ["A", "B"])
        if 'A. "A" causes "B".' in prompt:
            self.assertEqual(adj.tolist(), [[0, 1], [0, 0]])
        else:
            self.assertEqual(adj.tolist(), [[0, 0], [1, 0]])
        self.assertEqual(len(raw), 1)

    def test_run_jiralerspong_pairwise_retries_until_parseable_choice(self) -> None:
        call_messages = mock.Mock(side_effect=["", "<Answer>B</Answer>"])
        with mock.patch.dict(
            external._run_jiralerspong_pairwise.__globals__,
            {
                "_load_observational_array": mock.Mock(
                    return_value=(np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float), ["A", "B"])
                ),
                "_load_variable_metadata": mock.Mock(
                    return_value=[
                        {"name": "A", "symbol": "A", "description": "desc A"},
                        {"name": "B", "symbol": "B", "description": "desc B"},
                    ]
                ),
                "_call_model_messages": call_messages,
            },
        ):
            adj, variables, raw = external._run_jiralerspong_pairwise(
                graph_path=Path("/tmp/sachs.bif"),
                sample_size_obs=100,
                prompt_mode="summary",
                model_name="gpt-5-mini",
                provider="openai",
                temperature=0.0,
                max_new_tokens=None,
                seed=0,
                anonymize=False,
                hf_pipe=None,
            )
        self.assertEqual(call_messages.call_count, 2)
        self.assertEqual(variables, ["A", "B"])
        self.assertEqual(len(raw), 1)
        if 'A. "A" causes "B".' in call_messages.call_args.kwargs["messages"][-1]["content"]:
            self.assertEqual(adj.tolist(), [[0, 0], [1, 0]])
        else:
            self.assertEqual(adj.tolist(), [[0, 1], [0, 0]])

    def test_run_jiralerspong_pairwise_fails_after_unusable_responses(self) -> None:
        call_messages = mock.Mock(side_effect=["", "[ERROR] nope", "no answer tag"])
        with mock.patch.dict(
            external._run_jiralerspong_pairwise.__globals__,
            {
                "_load_observational_array": mock.Mock(
                    return_value=(np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float), ["A", "B"])
                ),
                "_load_variable_metadata": mock.Mock(
                    return_value=[
                        {"name": "A", "symbol": "A", "description": "desc A"},
                        {"name": "B", "symbol": "B", "description": "desc B"},
                    ]
                ),
                "_call_model_messages": call_messages,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "usable <Answer>A/B/C</Answer>"):
                external._run_jiralerspong_pairwise(
                    graph_path=Path("/tmp/sachs.bif"),
                    sample_size_obs=100,
                    prompt_mode="summary",
                    model_name="gpt-5-mini",
                    provider="openai",
                    temperature=0.0,
                    max_new_tokens=None,
                    seed=0,
                    anonymize=False,
                    hf_pipe=None,
                )
        self.assertEqual(call_messages.call_count, 3)

    def test_write_prediction_rows_preserves_multiple_replicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "predictions.csv"
            external._write_prediction_rows(
                out_csv=out_csv,
                rows=[
                    {
                        "method": "CausalLLMData",
                        "model": "m",
                        "provider": "p",
                        "naming_regime": "real",
                        "obs_n": 100,
                        "int_n": 0,
                        "raw_response": "r0",
                        "answer": "[[0]]",
                        "prediction": "[[0]]",
                        "valid": 1,
                    },
                    {
                        "method": "CausalLLMData",
                        "model": "m",
                        "provider": "p",
                        "naming_regime": "real",
                        "obs_n": 100,
                        "int_n": 0,
                        "raw_response": "r1",
                        "answer": "[[0]]",
                        "prediction": "[[0]]",
                        "valid": 1,
                    },
                ],
            )
            with out_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["raw_response"] for row in rows], ["r0", "r1"])


if __name__ == "__main__":
    unittest.main()
