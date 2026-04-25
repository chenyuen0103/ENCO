#!/usr/bin/env python3
"""
Compatibility shim for the renamed prompt/answer CSV generator.

Use:
    python experiments/generate_prompt_answer_csv.py ...
"""
from generate_prompt_answer_csv import main


if __name__ == "__main__":
    main()
