#!/usr/bin/env python3
"""Run API or HF models over already-generated prompt CSV files.

This is the consolidated entry point for the old run_api_models.py and
run_hf_models.py prompt-file workflows. New config-based evaluations should
prefer scripts/eval_cd_configs.py.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main(default_backend: str | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", choices=["api", "hf"], default=default_backend or "api")
    known, remaining = parser.parse_known_args()

    script_name = "run_hf_models.py" if known.backend == "hf" else "run_api_models.py"
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "experiments" / "legacy" / script_name
    sys.argv = [str(script_path), *remaining]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
