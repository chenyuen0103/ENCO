#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = [
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen2-0.5B-Instruct",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run HF querying on causal discovery prompt CSVs for Qwen models. "
            "Thin wrapper around run_hf_models.py."
        )
    )
    ap.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "HF model id/path (repeatable). "
            "If omitted, defaults to Qwen3-4B-Thinking-2507 and Qwen2-0.5B-Instruct."
        ),
    )
    args, passthrough = ap.parse_known_args()

    models = args.model or DEFAULT_MODELS
    runner = (Path(__file__).parent / "run_hf_models.py").resolve()
    if not runner.exists():
        raise SystemExit(f"runner script not found: {runner}")

    cmd = [sys.executable, str(runner)]
    for model in models:
        cmd.extend(["--model", model])
    cmd.extend(passthrough)

    print("[running]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
