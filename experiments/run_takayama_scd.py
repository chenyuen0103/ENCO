#!/usr/bin/env python3
"""Compatibility shim for experiments/baselines/takayama_scd.py."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "baselines" / "takayama_scd.py"),
        run_name="__main__",
    )
