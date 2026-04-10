#!/usr/bin/env python3
"""Validate frozen paper slice manifests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SLICES_DIR = REPO_ROOT / "paper_slices"


class TestPaperSlices(unittest.TestCase):
    def test_manifests_load(self) -> None:
        manifests = sorted(SLICES_DIR.glob("*.json"))
        self.assertGreaterEqual(len(manifests), 3)
        for path in manifests:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.assertIn("name", data)
            self.assertIn("dataset", data)
            self.assertIn("model", data)
            self.assertIn("role", data)

    def test_sachs_main_has_enco_and_two_cells(self) -> None:
        path = SLICES_DIR / "sachs_main.json"
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(data["dataset"], "sachs")
        self.assertTrue(data["enco"]["enabled"])
        styles = [cell["style"] for cell in data["prompt_cells"]]
        self.assertEqual(styles, ["summary_joint", "matrix"])

    def test_cancer_manifests_are_smoke_only(self) -> None:
        for name in ["cancer_smoke_summary.json", "cancer_smoke_names_only.json"]:
            with (SLICES_DIR / name).open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.assertEqual(data["dataset"], "cancer")
            self.assertEqual(data["role"], "smoke")


if __name__ == "__main__":
    unittest.main()
