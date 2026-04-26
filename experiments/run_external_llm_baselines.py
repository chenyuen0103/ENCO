#!/usr/bin/env python3
"""Compatibility shim for experiments/baselines/run_external_llm.py."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "baselines" / "run_external_llm.py"),
        run_name="__main__",
    )
