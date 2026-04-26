#!/usr/bin/env python3
"""Compatibility shim for experiments/baselines/run_classical.py."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "baselines" / "run_classical.py"),
        run_name="__main__",
    )
