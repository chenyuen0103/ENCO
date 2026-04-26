#!/usr/bin/env python3
"""Compatibility shim for experiments/run_prompt_csv_models.py --backend api."""
from run_prompt_csv_models import main


if __name__ == "__main__":
    main(default_backend="api")
