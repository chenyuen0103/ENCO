#!/usr/bin/env python3
"""
collect_permuted_sft_data.py  — compatibility shim

This script's functionality has been merged into collect_format_sft_data.py
as Mode C (--perm-csv).

Equivalent call:
    python experiments/collect_format_sft_data.py \\
        --perm-csv \\
        [--rows-per-source 5] [--max-perms 500] \\
        [--data-dir experiments/data] [--graph-filter cancer earthquake] ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from collect_format_sft_data import main

if __name__ == "__main__":
    # Inject --perm-csv if not already present so bare invocations of this
    # script default to the permutation mode they originally expected.
    if "--perm-csv" not in sys.argv:
        sys.argv.insert(1, "--perm-csv")
    main()
