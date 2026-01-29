#!/usr/bin/env python3
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(
        description="Generate prompts for Experiment 1 (Data Representation & Robustness)."
    )
    ap.add_argument(
        "--script",
        default="generate_prompts.py",
        help="Path to the standard prompt-generating script.",
    )
    ap.add_argument(
        "--bif-file",
        default="../causal_graphs/real_data/small_graphs/cancer.bif",
        help="Path to the BIF file.",
    )
    ap.add_argument(
        "--num-prompts",
        type=int,
        default=5,
        help="Number of random prompts (replicates) per configuration.",
    )
    ap.add_argument(
        "--shuffles-per-graph",
        type=int,
        action="append",
        default=[1],
        help=(
            "How many independent row-order shuffles to generate per base sampled dataset "
            "(repeatable). Encoded in filenames as '_shufN'."
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = ap.parse_args()

    # Resolve default script path
    script_arg = Path(args.script)
    # Interpret relative paths w.r.t. this file so defaults work when run from repo root
    if script_arg.is_absolute():
        default_script_path = script_arg
    else:
        default_script_path = (Path(__file__).parent / script_arg).resolve()
    if not default_script_path.exists():
        print(f"Error: Default script not found at {default_script_path}")
        sys.exit(1)

    # Resolve names-only script path (always next to this orchestrator)
    names_only_script = Path(__file__).parent / "generate_prompts_names_only.py"

    # ==========================================
    # EXPERIMENT 1 GRID DEFINITION
    # ==========================================
    
    # 1. Data Representation
    styles = ["cases", "matrix"]
    
    # 2. Semantics
    anonymize_opts = [False, True]
    
    # 3. Data Volume (Observational)
    obs_sizes = [0, 100, 1000, 5000, 8000]
    
    # 4. Data Volume (Interventional)
    int_sizes = [0, 50, 100, 200, 500] 

    # 5. Robustness: Ordering
    row_order_opts = ["random", "sorted", "reverse"]
    col_order_opts = ["original", "reverse", "random", "topo", "reverse_topo"]

    shuf_values = [int(x) for x in (args.shuffles_per_graph or [1])]

    # Create the Cartesian Product
    flag_combos = itertools.product(
        styles, 
        anonymize_opts, 
        obs_sizes,
        int_sizes,
        row_order_opts,
        col_order_opts,
        shuf_values,
    )

    print(f"--- Starting Experiment 1 Generation ---")
    print(f"BIF File: {args.bif_file}")
    
    count = 0
    for style, anon, obs_n, int_n, row_ord, col_ord, shuf_n in flag_combos:
        
        # --- DEFINITIONS ---
        is_names_only = (obs_n == 0 and int_n == 0)

        # Names-only generator currently ignores shuffles-per-graph and would overwrite outputs
        # if we varied it; keep a single value here.
        if is_names_only and shuf_n != 1:
            continue
        
        # --- NEW BASELINE DEFINITION ---
        # We perform expensive robustness checks (Topo/Reverse) ONLY on this config.
        # N=5000, Int=200, Cases, Real Names
        is_robustness_baseline = (
            obs_n == 5000 and 
            int_n == 200 and 
            style == "cases" and 
            anon is False
        )

        # --- FILTER 1: Names Only Constraints ---
        if is_names_only:
            # For names only, we ignore Data Size, Row Order, Anonymization
            if row_ord != "random": continue
            if anon is True: continue
            if style != "cases": continue 
            
            current_script_path = names_only_script

        # --- FILTER 2: Data Experiments Constraints ---
        else:
            current_script_path = default_script_path
            
            # Skip invalid empty data configs that aren't the official "Names Only" run
            if obs_n == 0 and int_n == 0: continue 

            # --- FILTER 3: Robustness (Ordering) ---
            # If this is NOT the baseline, force standard ordering.
            is_non_default_ordering = (row_ord != "random" or col_ord != "original")
            
            if is_non_default_ordering and not is_robustness_baseline:
                continue

            # --- FILTER 4: Context Window Safety ---
            # 'Cases' style with N=5000/8000 is massive. 
            # We allow it ONLY if it is the specific baseline we want to test.
            # Otherwise, we skip it to save time/tokens.
            if obs_n >= 5000 and style == "cases" and not is_robustness_baseline:
                continue

        # ----------------------------------------------------

        # Construct Output Directory Name
        parts = [style]
        parts.append("anon" if anon else "real")
        parts.append(f"obs{obs_n}")
        parts.append(f"int{int_n}")
        
        if row_ord != "random":
            parts.append(f"row{row_ord}")
        if col_ord != "original":
            parts.append(f"col{col_ord}")

        config_name = "_".join(parts)
        out_dir = Path(f"prompts/experiment1/cancer") / config_name

        print(
            f"[{count+1}] Generating: {config_name} (shuffles_per_graph={shuf_n}, Script: {current_script_path.name}) ..."
        )

        # Build Command
        cmd = [
            sys.executable,
            str(current_script_path),
            "--bif-file", args.bif_file,
            "--num-prompts", str(args.num_prompts),
            "--shuffles-per-graph", str(shuf_n),
            "--seed", str(args.seed),
            
            # Grid Variables
            "--prompt-style", style,
            "--obs-per-prompt", str(obs_n),
            "--int-per-combo", str(int_n),
            "--row-order", row_ord,
            "--col-order", col_ord,
            
            # Output
            "--out-dir", str(out_dir)
        ]

        if anon:
            cmd.append("--anonymize")
        
        if int_n > 0:
            cmd.extend(["--intervene-vars", "all"]) 
        else:
            cmd.extend(["--intervene-vars", "none"])

        # Execute
        try:
            if not current_script_path.exists():
                 raise FileNotFoundError(f"Script {current_script_path} not found.")

            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error running config {config_name}: {e}")
            sys.exit(1)
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
        
        count += 1

    print(f"\n=== generation complete. {count} configurations generated. ===")

if __name__ == "__main__":
    main()
