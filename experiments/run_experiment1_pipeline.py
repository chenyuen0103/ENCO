#!/usr/bin/env python3
"""Compatibility shim for experiments/pipelines/run_cd_eval_pipeline.py."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "pipelines" / "run_cd_eval_pipeline.py"),
        run_name="__main__",
    )
