#!/usr/bin/env python3
import argparse
import itertools
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run generate_prompts.py for all combinations of "
            "--anonymize / --causal-rules / --give-steps."
        )
    )
    ap.add_argument(
        "--script",
        default="generate_prompts.py",
        help="Path to the prompt-generating script (default: generate_prompts.py in this directory).",
    )
    # Common args to forward to generate_prompts.py
    ap.add_argument(
        "--bif-file",
        default="/home/yuen_chen/ENCO/causal_graphs/real_data/small_graphs/cancer.bif",
        help="Forwarded to generate_prompts.py --bif-file.",
    )
    ap.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Forwarded to generate_prompts.py --num-prompts.",
    )
    ap.add_argument(
        "--obs-per-prompt",
        type=int,
        default=200,
        help="Forwarded to generate_prompts.py --obs-per-prompt.",
    )
    ap.add_argument(
        "--int-per-combo",
        type=int,
        default=3,
        help="Forwarded to generate_prompts.py --int-per-combo.",
    )
    ap.add_argument(
        "--intervene-vars",
        default="all",
        help="Forwarded to generate_prompts.py --intervene-vars.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Forwarded to generate_prompts.py --seed.",
    )
    ap.add_argument(
        "--out-dir",
        default="./out/cancer",
        help="Forwarded to generate_prompts.py --out-dir.",
    )
    ap.add_argument(
        "--shuffles-per-graph",
        type=int,
        default=5,
        help="Forwarded to generate_prompts.py --shuffles-per-graph.",
    )
    args = ap.parse_args()

    script_path = Path(args.script).resolve()
    if not script_path.exists():
        sys.exit(f"Generator script not found: {script_path}")

    # All combinations of the three boolean flags:
    # anonymize ∈ {False, True}
    # causal_rules ∈ {False, True}
    # give_steps ∈ {False, True}
    combos = list(itertools.product([False, True], repeat=3))

    for anonymize, causal_rules, give_steps in combos:
        flag_str = (
            f"anonymize={anonymize}, "
            f"causal_rules={causal_rules}, "
            f"give_steps={give_steps}"
        )
        print(f"\n=== Running combination: {flag_str} ===")

        cmd = [
            sys.executable,
            str(script_path),
            "--bif-file", args.bif_file,
            "--num-prompts", str(args.num_prompts),
            "--obs-per-prompt", str(args.obs_per_prompt),
            "--int-per-combo", str(args.int_per_combo),
            "--intervene-vars", args.intervene_vars,
            "--seed", str(args.seed),
            "--out-dir", args.out_dir,
            "--shuffles-per-graph", str(args.shuffles_per_graph),
        ]

        # Add flags only when True so the generator's own suffix logic works
        if anonymize:
            cmd.append("--anonymize")
        if causal_rules:
            cmd.append("--causal-rules")
        if give_steps:
            cmd.append("--give-steps")

        print("Command:", " ".join(cmd))
        # Run and fail fast if any combination errors
        subprocess.run(cmd, check=True)

    print("\nAll flag combinations completed.")


if __name__ == "__main__":
    main()
