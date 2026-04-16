#!/usr/bin/env python3
"""
eval_cd_rewards_on_jsonl.py  — compatibility shim

This script's functionality has been merged into eval_sft_on_jsonl.py.
GRPO reward computation is now the default; use --no-compute-rewards to skip it.

Equivalent call:
    python experiments/eval_sft_on_jsonl.py \\
        --model <model_path> \\
        --jsonl <eval.jsonl> \\
        [--n 50] [--max-new-tokens 2048] ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_sft_on_jsonl import main

if __name__ == "__main__":
    main()
