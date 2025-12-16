import os, sys
import pandas as pd
import numpy as np
from pathlib import Path
import csv
import json
import subprocess
import re

# Display settings: do not abbreviate DataFrame output
pd.set_option("display.max_rows", None)        # show all rows
pd.set_option("display.max_columns", None)     # show all columns
pd.set_option("display.width", None)           # don't wrap to fit console width
pd.set_option("display.max_colwidth", None)    # don't truncate column contents
from pathlib import Path
import csv
import json
import subprocess
import re

dataset_name = "asia"  # change as needed
BASE_DIR = Path(f"responses/{dataset_name}")
SUMMARY_CSV = BASE_DIR / "eval_summary.csv"

def parse_given_edges_tag(path: Path):
    """
    Parse given-edges info from filenames like:
      ..._gedge20_...
    Returns (has_given_edges, given_edge_frac, given_edge_pct)
      has_given_edges: 0/1
      given_edge_frac: float or None
      given_edge_pct:  int or None
    """
    stem = path.stem  # e.g. responses_obs200_int3_shuf3_gedge20_anon_gpt-4o-mini
    m = re.search(r"_gedge(\d+)", stem)
    if not m:
        return 0, None, None
    pct = int(m.group(1))
    frac = pct / 100.0
    return 1, frac, pct

def count_nonempty(colname, rows):
    return sum(1 for r in rows if (r.get(colname) or "").strip())

def count_valid_flag(rows):
    # valid column is expected to be 1/0 or truthy/falsy
    return sum(1 for r in rows if str(r.get("valid", "")).strip() in {"1", "true", "True"})

def count_error_raw(rows):
    # count rows where raw_response contains "[ERROR]"
    return sum(
        1
        for r in rows
        if "[ERROR]" in (r.get("raw_response") or "")
    )


def evaluate_and_write_summary(base_dir: Path = BASE_DIR, summary_csv: Path = SUMMARY_CSV):
    """
    Scan response CSVs, run evaluate.py on complete ones, and write a combined summary CSV.
    """
    incomplete_files = []
    complete_files = []
    file_stats: dict[Path, dict[str, int]] = {}

    if not base_dir.exists():
        print(f"Base directory not found: {base_dir.resolve()}")
        return

    for csv_path in sorted(base_dir.rglob("*.csv")):
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
        except Exception as e:
            print(f"[ERROR] Failed to read {csv_path}: {e}")
            incomplete_files.append((csv_path, "read_error"))
            continue

        n_rows = len(rows)

        # Detect ENCO vs LLM responses by filename
        is_enco = "ENCO" in csv_path.name

        # Which column (if any) is the "prompt-like" column? (for logging only)
        if "prompt" in fieldnames:
            prompt_col = "prompt"
        elif "prompt_path" in fieldnames:
            prompt_col = "prompt_path"
        else:
            prompt_col = None

        has_raw = "raw_response" in fieldnames

        if is_enco:
            # ENCO files: no prompt/raw_response expected.
            n_prompt = 0
            n_raw = 0
            n_valid = 0
            n_error = 0

            complete = n_rows > 0
            status = "OK" if complete else "INCOMPLETE"
            print(
                f"{csv_path} -> rows={n_rows}, (ENCO: skipping prompt/raw checks) "
                f"valid_flags={n_valid} [{status}]"
            )

        else:
            # Normal LLM response files (must have raw_response)
            if not has_raw:
                # We cannot evaluate without raw_response column
                n_prompt = count_nonempty(prompt_col, rows) if prompt_col else 0
                n_raw = 0
                n_valid = 0
                n_error = 0

                complete = False
                status = "INCOMPLETE"
                reason = "missing raw_response column"
                print(f"{csv_path} -> rows={n_rows}, prompts={n_prompt}, "
                      f"raw_responses={n_raw}, error_responses={n_error}, "
                      f"valid_flags={n_valid} [{status}] ({reason})")

                file_stats[csv_path] = {
                    "n_rows": n_rows,
                    "completed_rows": n_raw,
                    "valid_flag_rows": n_valid,
                    "error_raw_responses": n_error,
                }
                incomplete_files.append((csv_path, reason))
                continue

            # We *do* have raw_response; use it for completeness
            n_prompt = count_nonempty(prompt_col, rows) if prompt_col else 0
            n_raw = count_nonempty("raw_response", rows)
            n_valid = count_valid_flag(rows) if "valid" in fieldnames else 0
            n_error = count_error_raw(rows)

            # Completeness = every row has some raw_response
            complete = (n_raw == n_rows)
            status = "OK" if complete else "INCOMPLETE"

            print(
                f"{csv_path} -> rows={n_rows}, prompts={n_prompt}, "
                f"raw_responses={n_raw}, error_responses={n_error}, "
                f"valid_flags={n_valid} [{status}]"
            )

        # Store stats for later use in the summary CSV
        file_stats[csv_path] = {
            "n_rows": n_rows,
            "completed_rows": n_raw,
            "valid_flag_rows": n_valid,
            "error_raw_responses": n_error,
        }

        if complete:
            complete_files.append(csv_path)
        else:
            # Explicit reason matching our completeness rule
            if is_enco:
                reason = f"rows={n_rows}, ENCO_file"
            else:
                reason = (
                    f"rows={n_rows}, raw_responses={n_raw}, "
                    f"has_raw={has_raw}"
                )
            incomplete_files.append((csv_path, reason))

    # --------- Delete files with only [ERROR] raw_responses ---------
    all_error_files = []
    for csv_path_err, stats in list(file_stats.items()):
        n_rows_err = stats["n_rows"]
        n_error_err = stats["error_raw_responses"]

        # For ENCO, n_error_err == 0, so this will never trigger
        if n_rows_err > 0 and n_rows_err == n_error_err:
            print(f"[CLEANUP] Deleting {csv_path_err} (all {n_rows_err} raw_responses contain '[ERROR]').")
            all_error_files.append(csv_path_err)

            # Remove from complete_files if it was marked complete
            if csv_path_err in complete_files:
                complete_files.remove(csv_path_err)

            # Delete the CSV itself
            try:
                csv_path_err.unlink()
            except FileNotFoundError:
                pass

            # Optionally delete any sidecar files created in previous runs
            jsonl_path = csv_path_err.with_suffix(".jsonl")
            summary_path = csv_path_err.with_suffix(csv_path_err.suffix + ".summary.json")
            for sidecar in (jsonl_path, summary_path):
                try:
                    sidecar.unlink()
                except FileNotFoundError:
                    pass

    # --------- Completeness summary to stdout ---------
    print("\n=== Completeness Summary ===")
    if not incomplete_files:
        print(f"All CSV files in {base_dir}/ appear complete.")
    else:
        print("These files look incomplete:")
        for path, reason in incomplete_files:
            print(f"- {path}: {reason}")

    # --------- Run evaluate.py on complete files & collect metrics ---------
    summary_rows = []

    for csv_path in complete_files:
        print(f"\nEvaluating {csv_path}")
        proc = subprocess.run(
            ["python", "evaluate.py", "--csv", str(csv_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"[ERROR] evaluate.py failed on {csv_path}")
            print(proc.stdout)
            print(proc.stderr)
            continue

        # evaluate.py should have created <file>.summary.json
        summary_path = csv_path.with_suffix(csv_path.suffix + ".summary.json")
        if not summary_path.exists():
            print(f"[WARN] Summary JSON not found for {csv_path}: {summary_path}")
            continue

        try:
            with summary_path.open("r", encoding="utf-8") as f_sum:
                metrics = json.load(f_sum)
        except Exception as e:
            print(f"[ERROR] Failed to read summary JSON for {csv_path}: {e}")
            continue

        # Build one combined row: file name + completeness stats + eval metrics
        base_stats = file_stats[csv_path]

        # Rename potentially confusing keys from evaluate.py
        metrics["eval_num_rows"] = metrics.pop("num_rows", None)
        metrics["eval_valid_rows"] = metrics.pop("valid_rows", None)

        # Decode given-edges info from filename
        has_given_edges, given_edge_frac, given_edge_pct = parse_given_edges_tag(csv_path)

        row = {
            "file": str(csv_path),
            "num_rows": base_stats["n_rows"],
            "completed": base_stats["completed_rows"],        # count(raw_response nonempty)
            "valid": base_stats["valid_flag_rows"],           # from 'valid' column
            "error_raw_responses": base_stats["error_raw_responses"],  # rows with "[ERROR]" in raw_response

            # new columns:
            "given_edges": has_given_edges,       # 0/1 flag
            "given_edge_frac": given_edge_frac,   # e.g. 0.2 for _gedge20
            "given_edge_pct": given_edge_pct,     # e.g. 20 for _gedge20
        }
        row.update(metrics)

        summary_rows.append(row)

    # --------- Write global metrics summary CSV ---------
    if summary_rows:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        # Collect union of all keys in case some metrics differ
        all_keys = []
        for r in summary_rows:
            for k in r.keys():
                if k not in all_keys:
                    all_keys.append(k)

        with summary_csv.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=all_keys)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"\nEvaluation summary written to: {summary_csv.resolve()}")
    else:
        print("\nNo complete files were successfully evaluated; no summary CSV written.")

    return summary_rows

import math

def build_tables(summary_csv: Path = SUMMARY_CSV, matrix_only: bool = False, per_obs_int: bool = False):
    df = pd.read_csv(summary_csv)
    df = df[~df['file'].str.contains('4B', case=False) &
            ~df['file'].str.contains('2024-07-18', case=False)]

    if matrix_only:
        matrix_mask = df["file"].str.contains("matrix", case=False)
        df = df[matrix_mask | df["file"].str.contains("ENCO", case=False)]

    def extract_obs_int(path_str: str):
        m = re.search(r"obs(\d+)_int(\d+)", path_str)
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    # Keep only rows that have an obs/int pattern in the filename
    df["obs_int"] = df["file"].astype(str).apply(extract_obs_int)
    df = df[~df["obs_int"].isna()].copy()

    # --- 0. Keep only anon LLM runs + ENCO baseline --------------------
    file_str = df["file"].astype(str)
    is_anon = file_str.str.contains("_anon", regex=False)
    is_enco = file_str.str.contains("ENCO", regex=False)  # non-anon ENCO runs
    # df = df[is_anon | is_enco].reset_index(drop=True)

    def detect_variant(path_str: str) -> str:
        name = Path(path_str).name.lower()
        if "rules_steps" in name:
            return "rules_steps"
        if "rules" in name:
            return "rules"
        if "steps" in name:
            return "steps"
        return "base"

    df["variant"] = df["file"].astype(str).apply(detect_variant)

    # Make life easier: ensure 'valid' exists & integer
    if "valid" in df.columns:
        df["valid"] = df["valid"].fillna(0).astype(int)
    else:
        df["valid"] = 0

    def render_tables(df_slice: pd.DataFrame, obs_int_pair=None):
        df_local = df_slice.copy()

        # --- 2. Metric lookup helpers ----------------------------------

        def matches_setting(path_series: pd.Series, include: list[str], exclude: list[str]) -> pd.Series:
            m = pd.Series(True, index=path_series.index)
            for inc in include:
                m &= path_series.str.contains(inc, regex=False)
            for exc in exclude:
                m &= ~path_series.str.contains(exc, regex=False)
            return m

        def lookup_metrics(setting_spec, model_tag,
                           variant=None,
                           min_valid=1,
                           given_edge_count=None):
            """
            setting_spec: (label, include_list, exclude_list)
            model_tag:    e.g. "gpt-4o-mini" or "ENCO"
            variant:      one of {"base", "rules", "steps", "rules_steps"} or None
            given_edge_count: if not None, require df['given_edge_count'] == this value
            """
            _, include_tags, exclude_tags = setting_spec
            file_series = df_local["file"].astype(str)
            m = matches_setting(file_series, include_tags, exclude_tags)

            # model name constraint
            m &= file_series.str.contains(model_tag, regex=False)

            # only apply valid filter if requested
            if min_valid > 0:
                m &= df_local["valid"] >= min_valid

            # variant constraint (base / rules / steps / rules_steps)
            if variant is not None:
                m &= (df_local["variant"] == variant)

            # given-edge constraint
            if given_edge_count is not None and "given_edge_count" in df_local.columns:
                m &= (df_local["given_edge_count"] == given_edge_count)

            subset = df_local[m]
            if subset.empty:
                return None, None

            row = subset.iloc[0]   # or subset.mean() if you prefer
            return row["avg_f1"], row["avg_shd"]

        def fmt(x, ndigits=2):
            if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
                return ""
            return f"{x:.{ndigits}f}"

        # ---------------------------------------------------------------
        # 3. Settings (Obs/Inter combinations)
        # ---------------------------------------------------------------

        # (display label, list of substrings to include, list to exclude)
        SETTINGS = [
            ("Anonymized", ["_anon"], []),
            ("Non-anonymized", [], ["_anon"]),
        ]

        # ---------------------------------------------------------------
        # 4. Extract model tags from filenames (LLMs only; ENCO is special)
        # ---------------------------------------------------------------

        def extract_model_tag(path_str: str) -> str:
            """Get the model part from a path, e.g.
               '...responses_obs0_int200_shuf3_anon_Qwen3-32B.csv' -> 'Qwen3-32B'
               '...predictions_obs5000_int200_ENCO.csv'             -> 'ENCO'
            """
            stem = Path(path_str).stem    # e.g. 'responses_obs0_int200_shuf3_anon_Qwen3-32B'
            parts = stem.split("_")
            return parts[-1]

        # Only look at response files for LLM models (ENCO is in predictions files)
        mask_llm = df_local["file"].str.contains("responses", regex=False) & \
                   ~df_local["file"].str.contains("ENCO", regex=False)
        df_models = df_local[mask_llm].copy()
        df_models["model"] = df_models["file"].apply(extract_model_tag)

        all_models = sorted(df_models["model"].unique())
        llm_models = all_models  # ENCO handled separately

        PRETTY_NAME = {
            "gemini-2.5-flash": "Gemini-2.5-Flash",
            # add others if you care; fall back to tag itself
        }
        def pretty_model_name(model_tag: str) -> str:
            return PRETTY_NAME.get(model_tag, model_tag)

        # ---------------------------------------------------------------
        # 5. Variant blocks (sections)
        # ---------------------------------------------------------------
        num_columns = len(SETTINGS) * 2 + 1  # SHD + F1 per setting + label
        VARIANTS = [
            ("base",        rf"\multicolumn{{{num_columns}}}{{l}}{{\textbf{{\textit{{Zero-Shot LLMs}}}}}} \\"),
            ("rules",       rf"\multicolumn{{{num_columns}}}{{l}}{{\textbf{{\textit{{Zero-Shot LLMs + Causality Rules}}}}}} \\"),
            ("steps",       rf"\multicolumn{{{num_columns}}}{{l}}{{\textbf{{\textit{{Zero-Shot LLMs + CD Steps}}}}}} \\"),
            ("rules_steps", rf"\multicolumn{{{num_columns}}}{{l}}{{\textbf{{\textit{{Zero-Shot LLMs + Causality Rules + CD Steps}}}}}} \\"),
        ]

        ROWS = []

        # 5.a ENCO (pure causal discovery method, non-anon predictions_*.csv)
        num_columns = len(SETTINGS) * 2 + 1  # SHD + F1 per setting + label
        ROWS.append({
            "section": rf"\multicolumn{{{num_columns}}}{{l}}{{\textbf{{\textit{{Pure Causal Discovery Method}}}}}} \\",
            "label":   "ENCO",
            "model_tag": "ENCO",
            "variant": "base",
            "given_edge_count": None,   # no given-edge prior
        })

        # 5.b LLM rows for each variant + model
        for variant_key, section_header in VARIANTS:
            first_in_section = True
            for model_tag in llm_models:
                ROWS.append({
                    "section":   section_header if first_in_section else None,
                    "label":     pretty_model_name(model_tag),
                    "model_tag": model_tag,
                    "variant":   variant_key,
                    "given_edge_count": None,   # normal zero-shot / rules / steps runs
                })
                first_in_section = False

        # 5.c NEW: Zero-Shot LLMs + Given One Edge
        # pick models that have given_edge_count == 1 somewhere
        if "given_edge_count" in df_local.columns:
            mask_given = (
                (df_local["given_edge_count"] == 1) &
                df_local["file"].str.contains("responses", regex=False) &
                ~df_local["file"].str.contains("ENCO", regex=False)
            )
            df_given = df_local[mask_given].copy()
            given_models = sorted(df_given["file"].apply(extract_model_tag).unique())
        else:
            given_models = []

        first_in_section = True

        for model_tag in given_models:
            ROWS.append({
                "section": rf"\multicolumn{{{num_columns}}}{{l}}{{\textbf{{\textit{{Zero-Shot LLMs + Given One Edge}}}}}} \\"
                           if first_in_section else None,
                "label":   pretty_model_name(model_tag),
                "model_tag": model_tag,
                "variant": "base",          # assume these are base + given-edge
                "given_edge_count": 1,      # key for filtering in lookup_metrics
            })
            first_in_section = False

        # ---------------------------------------------------------------
        # 6. Build LaTeX lines (F1 / SHD table)
        # ---------------------------------------------------------------
        lines = []

        # Determine obs/int context for captions
        caption_pair = obs_int_pair
        if caption_pair is None:
            pairs_here = sorted({p for p in df_local["obs_int"].dropna()})
            if len(pairs_here) == 1:
                caption_pair = pairs_here[0]
        caption_context = ""
        if caption_pair:
            obs_val, int_val = caption_pair
            caption_context = f" (obs={obs_val}, int={int_val})"

        lines.append(r"\begin{table}[ht!]")
        lines.append(r"\centering")
        lines.append(r"\setlength{\tabcolsep}{6pt}")
        lines.append(
            rf"\caption{{Causal discovery performance on the \textit{{{dataset_name.capitalize()}}} causal graph{caption_context}.}}"
        )
        # lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(r"\begin{tabular}{" + "l" + "cc"*len(SETTINGS) + "}")
        lines.append(r"\toprule")
        # Dynamic headers based on SETTINGS
        header_parts = []
        cmid_parts = []
        col_idx = 2
        for label, _, _ in SETTINGS:
            header_parts.append(rf"\multicolumn{{2}}{{c}}{{\textbf{{{label}}}}}")
            cmid_parts.append(rf"\cmidrule(lr){{{col_idx}-{col_idx+1}}}")
            col_idx += 2
        header_line = " & " + " & ".join(header_parts) + r" \\"
        lines.append(header_line)
        lines.append("".join(cmid_parts))

        metric_headers = ["\\textbf{Method}"]
        for _ in SETTINGS:
            metric_headers.extend([r"\textbf{SHD}~$\downarrow$", r"\textbf{F1}~$\uparrow$"])
        lines.append(" & ".join(metric_headers) + r" \\")

        lines.append(r"\midrule")

        for rowdef in ROWS:
            # Optional section header
            if rowdef.get("section"):
                lines.append(rowdef["section"])

            label       = rowdef["label"]
            model_tag   = rowdef["model_tag"]
            variant_key = rowdef["variant"]
            given_edge_count = rowdef.get("given_edge_count", None)

            # ENCO: don't require valid>=1, and ignore variant filter
            is_enco_row = (model_tag == "ENCO")
            min_valid = 0 if is_enco_row else 1
            variant_for_lookup = None if is_enco_row else variant_key

            cells = []
            for setting_spec in SETTINGS:
                f1, shd = lookup_metrics(
                    setting_spec,
                    model_tag=model_tag,
                    variant=variant_for_lookup,
                    min_valid=min_valid,
                    given_edge_count=given_edge_count,
                )
                cells.append(fmt(shd, ndigits=2))
                cells.append(fmt(f1))

            # If *all* entries for this row are empty, skip the row entirely
            if all(c == "" for c in cells):
                continue

            row_tex = " & ".join([label] + cells) + r" \\"

            lines.append(row_tex)
            lines.append(r"\addlinespace[0.8ex]")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        # lines.append(r"}")
        lines.append(r"\end{table}")

        latex_table = "\n".join(lines)
        print(latex_table)

        # ---------------------------------------------------------------
        # 7. Valid-ratio table, same sections (including Given One Edge)
        # ---------------------------------------------------------------

        def lookup_valid_counts(setting_spec, model_tag,
                                variant=None,
                                min_rows=1,
                                given_edge_count=None):
            """
            Returns (valid, num_rows) or (None, None) if not found.
            """
            _, include_tags, exclude_tags = setting_spec
            file_series = df_local["file"].astype(str)
            m = matches_setting(file_series, include_tags, exclude_tags)

            # constrain to this model
            m &= df_local["file"].astype(str).str.contains(model_tag, regex=False)

            # constrain to variant if given
            if variant is not None:
                m &= (df_local["variant"] == variant)

            # constrain to given_edge_count if given
            if given_edge_count is not None and "given_edge_count" in df_local.columns:
                m &= (df_local["given_edge_count"] == given_edge_count)

            subset = df_local[m]
            if subset.empty:
                return None, None

            row = subset.iloc[0]
            num_rows = row.get("num_rows", 0)
            valid    = row.get("valid", 0)

            if not num_rows or num_rows < min_rows:
                return None, None

            return int(valid), int(num_rows)

        valid_lines = []

        valid_lines.append(r"\begin{table}[ht!]")
        valid_lines.append(r"\centering")
        valid_lines.append(r"\setlength{\tabcolsep}{6pt}")
        valid_lines.append(
            rf"\caption{{Valid adjacency extraction ratio for each method and data setting on the \textit{{{dataset_name.capitalize()}}} graph{caption_context}.}}"
        )
        # valid_lines.append(r"\resizebox{\textwidth}{!}{%")
        valid_lines.append(r"\begin{tabular}{" + "l" + "c"*len(SETTINGS) + "}")
        valid_lines.append(r"\toprule")
        valid_header = [""] + [rf"\textbf{{{label}}}" for label, _, _ in SETTINGS]
        valid_lines.append(" & ".join(valid_header) + r" \\")
        valid_lines.append(r"\midrule")

        for rowdef in ROWS:
            # Optional section header – adapt from 7 columns to 4
            if rowdef.get("section"):
                sec = rowdef["section"]
                # adjust multicolumn to current width (1 label + len(SETTINGS) columns)
                sec = sec.replace(r"\multicolumn{7}", rf"\multicolumn{{{1+len(SETTINGS)}}}")
                valid_lines.append(sec)

            label       = rowdef["label"]
            model_tag   = rowdef["model_tag"]
            variant_key = rowdef["variant"]
            given_edge_count = rowdef.get("given_edge_count", None)

            cells = []
            for setting_spec in SETTINGS:
                valid_count, num_rows = lookup_valid_counts(
                    setting_spec,
                    model_tag=model_tag,
                    variant=variant_key if model_tag != "ENCO" else None,
                    given_edge_count=given_edge_count,
                )

                if valid_count is None or num_rows is None:
                    cells.append("")  # empty cell
                else:
                    cells.append(rf"$\frac{{{valid_count}}}{{{num_rows}}}$")

            # Skip row entirely if ALL settings are empty
            if all(c == "" for c in cells):
                continue

            row_tex = " & ".join([label] + cells) + r" \\"
            valid_lines.append(row_tex)
            valid_lines.append(r"\addlinespace[0.8ex]")

        valid_lines.append(r"\bottomrule")
        valid_lines.append(r"\end{tabular}")
        # valid_lines.append(r"}")
        valid_lines.append(r"\end{table}")

        latex_valid_table = "\n".join(valid_lines)
        print(latex_valid_table)

        return latex_table, latex_valid_table

    obs_int_pairs = sorted({pair for pair in df["obs_int"].dropna()})

    if per_obs_int and obs_int_pairs:
        results = []
        for obs_val, int_val in obs_int_pairs:
            df_pair = df[df["obs_int"] == (obs_val, int_val)]
            if df_pair.empty:
                continue
            print(f"\n=== Tables for obs{obs_val}_int{int_val} ===")
            results.append(((obs_val, int_val),) + render_tables(df_pair, obs_int_pair=(obs_val, int_val)))
        return results

    return render_tables(df)


if __name__ == "__main__":
    # Uncomment to regenerate eval_summary.csv from raw responses
    # evaluate_and_write_summary()
    build_tables(matrix_only=True, per_obs_int=True)
