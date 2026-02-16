#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path


def main():
    # Match the output directory of the names-only generator
    # base_dir = Path("prompts_names_only/asia")
    base_dirs = [
        # Path("prompts_names_only/asia"),
        # Path("prompts_names_only/sachs"),
        # Path("prompts_names_only/alarm"),
        Path("prompts_names_only/cancer"),
        # Path("prompts_names_only/child"),
        # Path("prompts_names_only/earthquake"),
        # Path("prompts_names_only/pigs"),
        # Path("prompts_names_only/diabetes"),
    ]


    # This must match how you called generate_prompts_names_only.py
    # (default in the generator is shuffles_per_graph=1)
    shuf_values = [1]

    # anonymize / rules / steps combinations
    anonymize_opts = [False, True]
    rule_step_opts = [
        (False, False),  # ""
        # (True,  False),  # "_rules"
        # (False, True),   # "_steps"
        # (True,  True),   # "_rules_steps"
    ]

    # If you ever use given_edge_frac>0, add suffixes like "_gedge20" here
    given_edge_suffixes = [""]  # e.g. ["", "_gedge20"]

    # API models to run
    models = [
        "gpt-5-mini",
        # "gpt-4o-mini",
        # "gemini-2.5-flash",
    ]

    temperature = "0.0"

    total_planned = 0
    total_ran = 0
    total_skipped_missing_csv = 0
    for base_dir in base_dirs:
        for shuf in shuf_values:
            for anonymize in anonymize_opts:
                for use_rules, use_steps in rule_step_opts:
                    for edge_suffix in given_edge_suffixes:
                        # Rebuild tags in the SAME ORDER as the names-only generator:
                        # anon, rules, steps
                        tags = []
                        if anonymize:
                            tags.append("anon")
                        if use_rules:
                            tags.append("rules")
                        if use_steps:
                            tags.append("steps")

                        extra_suffix = "_" + "_".join(tags) if tags else ""

                        # Must match base_name in generate_prompts_names_only.py:
                        # base_name = f"prompts_names_only_shuf{shuf}{edge_tag}{extra_suffix}"
                        filename = f"prompts_names_only_shuf{shuf}{edge_suffix}{extra_suffix}.csv"
                        csv_path = base_dir / filename

                        total_planned += 1

                        if not csv_path.exists():
                            print(f"[skip] CSV not found: {csv_path}")
                            total_skipped_missing_csv += 1
                            continue

                        for model in models:
                            cmd = [
                                "python",
                                "query_gemini.py",
                                "--csv", str(csv_path),
                                "--model", model,
                                "--temperature", temperature,
                                # "--overwrite",  # uncomment if you want to redo runs
                            ]
                            print("\n[running]", " ".join(cmd))
                            subprocess.run(cmd, check=True)
                            total_ran += 1

    print("\n=== Summary ===")
    print(f"CSV combinations considered : {total_planned}")
    print(f"CSV files missing           : {total_skipped_missing_csv}")
    print(f"Model runs executed         : {total_ran}")


if __name__ == "__main__":
    main()
