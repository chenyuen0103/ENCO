#!/usr/bin/env python3
from __future__ import annotations

import unittest

try:
    from experiments.generate_prompts import format_prompt_cb_matrix
except ModuleNotFoundError as exc:
    format_prompt_cb_matrix = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(format_prompt_cb_matrix is None, f"prompt generation dependencies unavailable: {IMPORT_ERROR}")
class TestPromptAnonymization(unittest.TestCase):
    def test_anonymized_matrix_prompt_omits_dataset_name(self) -> None:
        prompt = format_prompt_cb_matrix(
            variables=["X1", "X2"],
            all_rows=[],
            dataset_name="sachs",
            anonymize=True,
        )

        self.assertIn(
            "The following are empirical distributions computed from data sampled from an unknown Bayesian network.",
            prompt,
        )
        self.assertNotIn("sachs", prompt)
        self.assertNotIn("Bayesian network named", prompt)

    def test_named_matrix_prompt_keeps_dataset_name(self) -> None:
        prompt = format_prompt_cb_matrix(
            variables=["Akt", "Erk"],
            all_rows=[],
            dataset_name="sachs",
            anonymize=False,
        )

        self.assertIn("Bayesian network named sachs", prompt)


if __name__ == "__main__":
    unittest.main()
