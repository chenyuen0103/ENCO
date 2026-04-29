#!/usr/bin/env python3
from __future__ import annotations

import unittest
import csv
import tempfile
from pathlib import Path

import numpy as np

from experiments.run_takayama_scd import (
    _build_first_prompt,
    _chat_text_completion,
    _load_checkpoint_payload,
    _probability_to_pk,
    _read_completed_replicate_rows,
    _directed_prediction_from_pc_signed,
    _prediction_row,
    _write_prediction_rows,
)


class TestTakayamaSCD(unittest.TestCase):
    def test_chat_completion_omits_temperature_for_gpt5_mini(self) -> None:
        class FakeChoice:
            message = type("Message", (), {"content": "ok"})()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeCompletions:
            def __init__(self) -> None:
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return FakeResponse()

        class FakeClient:
            def __init__(self) -> None:
                self.chat = type("Chat", (), {"completions": FakeCompletions()})()

        client = FakeClient()
        result = _chat_text_completion(
            provider="openai",
            client=client,
            model_name="gpt-5-mini",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.0,
            max_new_tokens=8,
        )

        self.assertEqual(result["text"], "ok")
        self.assertNotIn("temperature", client.chat.completions.kwargs)
        self.assertEqual(client.chat.completions.kwargs["max_completion_tokens"], 8)

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
            backend="pc",
            dataset_name="toy",
            labels=["A", "B"],
            i=0,
            j=1,
            pattern=2,
            adjacency_matrix=signed,
            primary_prob=directed,
            secondary_prob=undirected,
            anonymized=False,
        )
        self.assertIn("bootstrap probability", prompt)
        self.assertIn("PC(Peter-Clerk)", prompt)

    def test_write_prediction_rows_preserves_multiple_replicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "takayama.csv"
            rows = [
                _prediction_row(
                    backend="pc",
                    pattern=2,
                    model_name="m",
                    provider="openai",
                    naming_regime="real",
                    sample_size_obs=100,
                    answer=np.zeros((2, 2), dtype=int),
                    prediction=np.zeros((2, 2), dtype=int),
                    transcript=[{"rep": 0}],
                    probability_matrix=np.zeros((2, 2), dtype=float),
                    prior_matrix=np.zeros((2, 2), dtype=int),
                    replicate_index=0,
                    replicate_seed=42,
                ),
                _prediction_row(
                    backend="pc",
                    pattern=2,
                    model_name="m",
                    provider="openai",
                    naming_regime="real",
                    sample_size_obs=100,
                    answer=np.zeros((2, 2), dtype=int),
                    prediction=np.ones((2, 2), dtype=int),
                    transcript=[{"rep": 1}],
                    probability_matrix=np.ones((2, 2), dtype=float),
                    prior_matrix=np.ones((2, 2), dtype=int),
                    replicate_index=1,
                    replicate_seed=1042,
                ),
            ]
            _write_prediction_rows(out_csv=out_csv, rows=rows)
            with out_csv.open(newline="", encoding="utf-8") as handle:
                loaded = list(csv.DictReader(handle))
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["method"], "TakayamaSCP")
        self.assertEqual(loaded[0]["replicate_index"], "0")
        self.assertEqual(loaded[1]["replicate_seed"], "1042")
        self.assertIn('"rep": 1', loaded[1]["raw_response"])

    def test_completed_replicate_rows_ignores_legacy_rows_without_replicate_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "takayama.csv"
            rows = [
                _prediction_row(
                    backend="pc",
                    pattern=2,
                    model_name="m",
                    provider="openai",
                    naming_regime="real",
                    sample_size_obs=100,
                    answer=np.zeros((2, 2), dtype=int),
                    prediction=np.zeros((2, 2), dtype=int),
                    transcript=[],
                    probability_matrix=np.zeros((2, 2), dtype=float),
                    prior_matrix=np.zeros((2, 2), dtype=int),
                    replicate_index=1,
                    replicate_seed=1042,
                )
            ]
            _write_prediction_rows(out_csv=out_csv, rows=rows)
            loaded = _read_completed_replicate_rows(out_csv)
        self.assertEqual(sorted(loaded), [1])
        self.assertEqual(loaded[1]["replicate_seed"], "1042")

    def test_mismatched_checkpoint_is_archived_and_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "run.checkpoint.json"
            checkpoint.write_text(
                '{"version": 1, "run_signature": {"num_samples": 5}, "completed_pairs": [{"x": 1}], "current_pair": null}',
                encoding="utf-8",
            )
            payload = _load_checkpoint_payload(checkpoint, {"num_samples": 1})
            stale_files = list(Path(tmpdir).glob("run.checkpoint.json.stale-*"))

        self.assertEqual(payload["completed_pairs"], [])
        self.assertEqual(payload["run_signature"], {"num_samples": 1})
        self.assertEqual(len(stale_files), 1)


if __name__ == "__main__":
    unittest.main()
