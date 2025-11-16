#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

def main():
    base_dir = Path("prompts/cancer")

    # Corresponds to prompts_obs{0,200}
    obs_values = [5000]

    # Corresponds to _int{0,3}
    int_values = [200]

    # Corresponds to { _anon, '' }
    # anon_suffixes = ["", "_anon"]
    anon_suffixes = ["_anon"]

    # Corresponds to { _rules, _steps, _rules_steps, '' }
    # rule_suffixes = ["", "_rules", "_steps", "_rules_steps"]
    given_edges_suffixes = [ "" ]
    rule_suffixes = [""]
    # rule_suffixes = ["_rules_steps"]

    # Models to run
    models = [
        # "gemini-2.5-flash",
        # "gpt-5-mini",
        "gpt-4o-mini",
    ]

    temperature = "0.0"

    total_planned = 0
    total_ran = 0
    total_skipped_missing_csv = 0
    total_skipped_by_rule = 0

    for obs in obs_values:
        for intval in int_values:
            if obs == 0 and intval == 0:
                total_skipped_by_rule += 1
                continue
            for anon in anon_suffixes:
                for rule in rule_suffixes:
                    for given_edges in given_edges_suffixes:
                        filename = f"prompts_obs{obs}_int{intval}_shuf3{given_edges}{anon}{rule}.csv"
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
                            # "--overwrite",
                        ]
                        print("\n[running]", " ".join(cmd))
                        # Run and propagate errors if something fails
                        subprocess.run(cmd, check=True)
                        total_ran += 1

    print("\n=== Summary ===")
    print(f"CSV combos (obs/int/anon/rule) considered : {total_planned + total_skipped_missing_csv}")
    print(f"Skipped due to obs=int=0                  : {total_skipped_by_rule}")
    print(f"CSV combinations planned (after rule)     : {total_planned}")
    print(f"CSV files missing                         : {total_skipped_missing_csv}")
    print(f"Model runs executed                       : {total_ran}")
if __name__ == "__main__":
    main()
