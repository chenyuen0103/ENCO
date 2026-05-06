#!/usr/bin/env python3
"""Count prompt configurations per graph/model/data-budget cell.

The grouping key is:

  graph, model, obs_n, int_n

Everything else that distinguishes one evaluated response artifact from another
is treated as the prompt configuration. The script writes:

  experiments/out/prompt_config_counts.csv
  experiments/out/prompt_config_inventory.csv
  experiments/out/prompt_config_rankings.csv
  experiments/out/prompt_config_top_configs.csv
  experiments/out/prompt_config_missing_valid_groups.csv

By default it scans current and legacy response roots, then deduplicates repeated
copies of the same response artifact by graph/model/obs/int/response basename.
The default scope is the 10 base LLMs used in the paper figures with
obs_n <= 5000 and int_n <= 200.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSE_ROOTS = [
    Path("scripts/responses"),
    Path("experiments/responses"),
    Path("experiments/experiments/responses"),
    Path("responses"),
]
DEFAULT_COUNTS_OUT = Path("experiments/out/prompt_config_counts.csv")
DEFAULT_INVENTORY_OUT = Path("experiments/out/prompt_config_inventory.csv")
DEFAULT_RANKINGS_OUT = Path("experiments/out/prompt_config_rankings.csv")
DEFAULT_TOP_OUT = Path("experiments/out/prompt_config_top_configs.csv")
DEFAULT_MISSING_VALID_OUT = Path("experiments/out/prompt_config_missing_valid_groups.csv")
DEFAULT_MAX_OBS = 5000
DEFAULT_MAX_INT = 200

PAPER_MODEL_ORDER = [
    "GPT-5 Mini",
    "GPT-5.2 Pro",
    "Qwen3-4B",
    "Qwen3-30B-A3B",
    "Qwen2.5-7B",
    "Qwen2.5-14B",
    "Qwen2.5-72B",
    "Llama-3.1-8B",
    "Llama-3.1-8B-Inst.",
    "Llama-3.1-70B",
]
PAPER_MODEL_DISPLAY_BY_RAW = {
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-5.2-pro": "GPT-5.2 Pro",
    "Qwen3-4B-Thinking-2507": "Qwen3-4B",
    "Qwen3-30B-A3B-Thinking-2507": "Qwen3-30B-A3B",
    "Qwen2.5-7B-Instruct-1M": "Qwen2.5-7B",
    "Qwen2.5-14B-Instruct-1M": "Qwen2.5-14B",
    "Qwen2.5-72B-Instruct-AWQ": "Qwen2.5-72B",
    "Meta-Llama-3.1-8B": "Llama-3.1-8B",
    "Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B-Inst.",
    "Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "Qwen/Qwen3-4B-Thinking-2507": "Qwen3-4B",
    "Qwen/Qwen3-30B-A3B-Thinking-2507": "Qwen3-30B-A3B",
    "Qwen/Qwen2.5-7B-Instruct-1M": "Qwen2.5-7B",
    "Qwen/Qwen2.5-14B-Instruct-1M": "Qwen2.5-14B",
    "Qwen/Qwen2.5-72B-Instruct-AWQ": "Qwen2.5-72B",
    "meta-llama/Meta-Llama-3.1-8B": "Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B-Inst.",
    "meta-llama/Llama-3.1-70B-Instruct": "Llama-3.1-70B",
}


RESP_RE = re.compile(
    r"^responses_obs(?P<obs>\d+)_int(?P<int>\d+)_shuf(?P<shuf>\d+)(?P<suffix>.*)$",
    flags=re.IGNORECASE,
)
PRED_RE = re.compile(
    r"^predictions_obs(?P<obs>\d+)_int(?P<int>\d+)_(?P<method>.+)$",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses-root",
        action="append",
        type=Path,
        default=[],
        help="Response root containing <graph>/eval_summary.csv. Repeatable.",
    )
    parser.add_argument(
        "--counts-out",
        type=Path,
        default=DEFAULT_COUNTS_OUT,
        help="Output CSV with one row per graph/model/obs/int group.",
    )
    parser.add_argument(
        "--inventory-out",
        type=Path,
        default=DEFAULT_INVENTORY_OUT,
        help="Long-form output CSV with one row per prompt configuration.",
    )
    parser.add_argument(
        "--rankings-out",
        type=Path,
        default=DEFAULT_RANKINGS_OUT,
        help="Long-form output CSV with per-group valid-rate and F1 ranks.",
    )
    parser.add_argument(
        "--top-out",
        type=Path,
        default=DEFAULT_TOP_OUT,
        help="Compact output CSV with the top valid-rate and top-F1 config per group.",
    )
    parser.add_argument(
        "--missing-valid-out",
        type=Path,
        default=DEFAULT_MISSING_VALID_OUT,
        help="Output CSV listing groups with fewer than --min-valid-configs valid prompt configs.",
    )
    parser.add_argument(
        "--graphs",
        nargs="*",
        default=None,
        help="Optional graph filter. Defaults to every eval_summary.csv found.",
    )
    parser.add_argument(
        "--include-sachs-old",
        action="store_true",
        help="Include sachs_old directories. Default excludes them from paper-style counts.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Count every discovered model instead of the 10 base paper models.",
    )
    parser.add_argument(
        "--max-obs",
        type=int,
        default=DEFAULT_MAX_OBS,
        help="Drop rows with obs_n greater than this value. Use a negative value to disable.",
    )
    parser.add_argument(
        "--max-int",
        type=int,
        default=DEFAULT_MAX_INT,
        help="Drop rows with int_n greater than this value. Use a negative value to disable.",
    )
    parser.add_argument(
        "--min-valid-runs",
        type=int,
        default=1,
        help="Minimum summed valid_rows required per group.",
    )
    parser.add_argument(
        "--fail-on-missing-valid",
        action="store_true",
        help="Exit nonzero when any group has fewer than --min-valid-configs valid prompt configs.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Maximum prompt config examples to include in the compact count table.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def read_eval_summaries(roots: Iterable[Path], graphs: set[str] | None, include_sachs_old: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*/eval_summary.csv")):
            graph = path.parent.name
            if graph.startswith("_"):
                continue
            if graph == "sachs_old" and not include_sachs_old:
                continue
            if graphs is not None and graph not in graphs:
                continue
            try:
                frame = pd.read_csv(path)
            except Exception as exc:
                print(f"[warn] failed to read {path}: {exc}")
                continue
            frame["summary_path"] = str(path)
            frame["response_root"] = str(root)
            if "dataset" not in frame.columns:
                frame["dataset"] = graph
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No eval_summary.csv files found in requested response roots.")
    return pd.concat(frames, ignore_index=True, sort=False)


def numeric_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def text_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="object")
    return df[col].fillna(default).astype(str)


def basename_series(df: pd.DataFrame) -> pd.Series:
    response_csv = text_series(df, "response_csv")
    return response_csv.map(lambda value: Path(value).name if value else "")


def f1_series(df: pd.DataFrame) -> pd.Series:
    f1 = numeric_series(df, "avg_f1")
    alt = numeric_series(df, "avg_F1")
    return f1.where(f1.notna(), alt)


def normalize_stem_for_parse(stem: str) -> str:
    out = stem
    out = out.replace("summary_joint", "summary")
    out = out.replace("summary_join", "summary")
    out = out.replace("thinktags_cothint", "wrapchat_fmthint")
    out = out.replace("cothint_thinktags", "wrapchat_fmthint")
    out = out.replace("thinktags", "wrapchat")
    out = out.replace("cothint", "fmthint")
    out = out.replace("respthink_answer", "")
    out = out.replace("wrapplain", "")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def normalize_paper_model(raw_model: str) -> str:
    raw = str(raw_model).strip()
    if raw in PAPER_MODEL_DISPLAY_BY_RAW:
        return PAPER_MODEL_DISPLAY_BY_RAW[raw]
    basename = Path(raw).name
    if basename in PAPER_MODEL_DISPLAY_BY_RAW:
        return PAPER_MODEL_DISPLAY_BY_RAW[basename]
    return ""


def infer_model_from_basename(basename: str) -> str:
    stem = normalize_stem_for_parse(Path(str(basename)).stem)
    if not stem:
        return "unknown"

    m_names = re.match(
        r"^responses_names_only(?:_p\d+)?"
        r"(?P<tags>(?:_(?:anon|rules|steps|wrapchat|fmthint|reason(?:concise|none)|shuf\d+|"
        r"row[A-Za-z0-9]+|col[A-Za-z0-9]+))*)_(?P<model>.+)$",
        stem,
        flags=re.IGNORECASE,
    )
    if m_names:
        return m_names.group("model").strip()

    m_pred = PRED_RE.match(stem)
    if m_pred:
        method = m_pred.group("method")
        method = re.sub(r"_seed\d+$", "", method)
        for suffix in ("_anon", "_names_only"):
            if method.endswith(suffix):
                method = method[: -len(suffix)]
        return method.strip() or "unknown"

    m_resp = RESP_RE.match(stem)
    if not m_resp:
        return "unknown"

    tokens = [token for token in m_resp.group("suffix").strip("_").split("_") if token]
    tag_re = re.compile(
        r"^(p\d+|anon|rules|steps|wrapchat|fmthint|reason(?:concise|none)|"
        r"row[A-Za-z0-9]+|col[A-Za-z0-9]+|summary|matrix|cases|payload|topk|probs|gedge\d+)$",
        flags=re.IGNORECASE,
    )
    while tokens and tag_re.match(tokens[0]):
        tokens.pop(0)
    return "_".join(tokens).strip() or "unknown"


def normalize_inventory(raw: pd.DataFrame, paper_models_only: bool) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    out["graph"] = text_series(raw, "dataset").str.strip()
    out["model"] = text_series(raw, "model").str.strip()
    out["prompt_style"] = text_series(raw, "prompt_style")
    out["anonymize"] = text_series(raw, "anonymize", "0")
    out["reasoning_guidance"] = text_series(raw, "reasoning_guidance")
    out["wrapper_mode"] = text_series(raw, "wrapper_mode")
    out["append_format_hint"] = text_series(raw, "append_format_hint")
    out["row_order"] = text_series(raw, "row_order")
    out["col_order"] = text_series(raw, "col_order")
    out["give_steps"] = text_series(raw, "give_steps")
    out["causal_rules"] = text_series(raw, "causal_rules")
    out["config"] = text_series(raw, "config")
    out["response_csv"] = text_series(raw, "response_csv")
    out["response_basename"] = basename_series(raw)
    inferred_model = out["response_basename"].map(infer_model_from_basename)
    missing_model = out["model"].eq("") | out["model"].str.lower().eq("unknown")
    out.loc[missing_model & inferred_model.ne("unknown"), "model"] = inferred_model[missing_model & inferred_model.ne("unknown")]
    out["model_raw"] = out["model"]
    out["paper_model"] = out["model_raw"].map(normalize_paper_model)
    if paper_models_only:
        out = out[out["paper_model"].ne("")].copy()
        out["model"] = out["paper_model"]
    out["summary_path"] = text_series(raw, "summary_path")
    out["response_root"] = text_series(raw, "response_root")

    obs = numeric_series(raw, "obs_n")
    inter = numeric_series(raw, "int_n")
    is_names_only = out["prompt_style"].eq("names_only")
    out["obs_n"] = obs.mask(obs.isna() & is_names_only, 0)
    out["int_n"] = inter.mask(inter.isna() & is_names_only, 0)
    out["obs_n"] = out["obs_n"].astype("Int64")
    out["int_n"] = out["int_n"].astype("Int64")

    out["num_rows"] = numeric_series(raw, "num_rows").astype("Int64")
    out["valid_rows"] = numeric_series(raw, "valid_rows").astype("Int64")
    out["valid_rate"] = numeric_series(raw, "valid_rate")
    out["avg_f1"] = f1_series(raw)
    out["avg_shd"] = numeric_series(raw, "avg_shd")
    missing_valid_rate = out["valid_rate"].isna() & out["num_rows"].notna() & (out["num_rows"] > 0)
    out.loc[missing_valid_rate, "valid_rate"] = (
        out.loc[missing_valid_rate, "valid_rows"].astype(float) / out.loc[missing_valid_rate, "num_rows"].astype(float)
    )

    # One evaluated response artifact is the practical prompt-configuration unit.
    # The basename preserves tags such as prompt style, anonymization, shuffling,
    # prompt count, wrapper, format hint, ordering, and model-specific suffixes.
    out["prompt_config_key"] = out["response_basename"]
    no_basename = out["prompt_config_key"].eq("")
    out.loc[no_basename, "prompt_config_key"] = out.loc[no_basename, "config"]

    keep = (
        out["graph"].ne("")
        & out["model"].ne("")
        & out["obs_n"].notna()
        & out["int_n"].notna()
        & out["prompt_config_key"].ne("")
    )
    out = out[keep].copy()

    # Deduplicate copies of the same artifact across legacy and canonical roots.
    dedup_cols = ["graph", "model", "obs_n", "int_n", "prompt_config_key"]
    out = (
        out.sort_values(["graph", "model", "obs_n", "int_n", "prompt_config_key", "response_root"])
        .drop_duplicates(dedup_cols, keep="first")
        .reset_index(drop=True)
    )
    return out


def compact_examples(values: pd.Series, max_examples: int) -> str:
    unique = [str(value) for value in values.dropna().astype(str).unique() if value]
    shown = unique[:max_examples]
    suffix = "" if len(unique) <= max_examples else f"; ... (+{len(unique) - max_examples} more)"
    return "; ".join(shown) + suffix


def build_counts(inventory: pd.DataFrame, max_examples: int) -> pd.DataFrame:
    group_cols = ["graph", "model", "obs_n", "int_n"]
    counts = (
        inventory.groupby(group_cols, dropna=False)
        .agg(
            prompt_config_count=("prompt_config_key", "nunique"),
            total_valid_rows=("valid_rows", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
            valid_prompt_config_count=("valid_rows", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            source_root_count=("response_root", "nunique"),
            response_file_count=("response_basename", "nunique"),
            prompt_styles=("prompt_style", lambda s: ",".join(sorted(x for x in s.dropna().astype(str).unique() if x))),
            anonymize_values=("anonymize", lambda s: ",".join(sorted(x for x in s.dropna().astype(str).unique() if x))),
            wrapper_modes=("wrapper_mode", lambda s: ",".join(sorted(x for x in s.dropna().astype(str).unique() if x))),
            format_hint_values=("append_format_hint", lambda s: ",".join(sorted(x for x in s.dropna().astype(str).unique() if x))),
            prompt_config_examples=("prompt_config_key", lambda s: compact_examples(s, max_examples)),
        )
        .reset_index()
    )
    best_valid = pick_best(inventory, "valid_rate", require_score=True).rename(
        columns={
            "prompt_config_key": "best_valid_config",
            "valid_rate": "best_valid_rate",
            "avg_f1": "best_valid_avg_f1",
            "response_basename": "best_valid_response",
        }
    )
    best_f1 = pick_best(inventory, "avg_f1", require_score=True).rename(
        columns={
            "prompt_config_key": "best_f1_config",
            "avg_f1": "best_f1",
            "valid_rate": "best_f1_valid_rate",
            "response_basename": "best_f1_response",
        }
    )
    best_cols = ["graph", "model", "obs_n", "int_n"]
    counts = counts.merge(
        best_valid[
            best_cols + ["best_valid_rate", "best_valid_avg_f1", "best_valid_config", "best_valid_response"]
        ],
        on=best_cols,
        how="left",
    )
    counts = counts.merge(
        best_f1[best_cols + ["best_f1", "best_f1_valid_rate", "best_f1_config", "best_f1_response"]],
        on=best_cols,
        how="left",
    )
    counts["has_valid_run"] = counts["total_valid_rows"].ge(1)
    return counts.sort_values(["graph", "model", "obs_n", "int_n"]).reset_index(drop=True)


def pick_best(inventory: pd.DataFrame, score_col: str, require_score: bool) -> pd.DataFrame:
    group_cols = ["graph", "model", "obs_n", "int_n"]
    work = inventory.copy()
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    if require_score:
        work = work[work["_score"].notna()].copy()
    if work.empty:
        return inventory.iloc[0:0].copy()
    tie_cols = ["avg_f1", "valid_rate", "valid_rows", "prompt_config_key"]
    ascending = [True, True, True, True, False, False, False, False, True]
    return (
        work.sort_values(group_cols + ["_score"] + tie_cols, ascending=ascending, na_position="last")
        .drop_duplicates(group_cols, keep="first")
        .drop(columns=["_score"])
        .reset_index(drop=True)
    )


def build_rankings(inventory: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["graph", "model", "obs_n", "int_n"]
    rankings = inventory.copy()

    def rank_nonmissing(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        ranks = numeric.rank(method="min", ascending=False, na_option="keep")
        return ranks.where(numeric.notna()).astype("Int64")

    rankings["valid_rate_rank"] = rankings.groupby(group_cols, dropna=False)["valid_rate"].transform(rank_nonmissing)
    rankings["f1_rank"] = rankings.groupby(group_cols, dropna=False)["avg_f1"].transform(rank_nonmissing)
    rankings["is_top_valid_rate"] = rankings["valid_rate"].notna() & rankings["valid_rate_rank"].eq(1)
    rankings["is_top_f1"] = rankings["avg_f1"].notna() & rankings["f1_rank"].eq(1)
    return rankings.sort_values(
        ["graph", "model", "obs_n", "int_n", "valid_rate_rank", "f1_rank", "prompt_config_key"],
        na_position="last",
    ).reset_index(drop=True)


def build_top_configs(inventory: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["graph", "model", "obs_n", "int_n"]
    output_cols = group_cols + [
        "model_raw",
        "paper_model",
        "criterion",
        "criterion_value",
        "valid_rate",
        "avg_f1",
        "avg_shd",
        "valid_rows",
        "num_rows",
        "prompt_style",
        "anonymize",
        "reasoning_guidance",
        "wrapper_mode",
        "append_format_hint",
        "row_order",
        "col_order",
        "give_steps",
        "causal_rules",
        "prompt_config_key",
        "response_basename",
        "response_csv",
        "summary_path",
    ]
    best_valid = pick_best(inventory, "valid_rate", require_score=True).copy()
    best_valid["criterion"] = "top_valid_rate"
    best_valid["criterion_value"] = best_valid["valid_rate"]
    best_f1 = pick_best(inventory, "avg_f1", require_score=True).copy()
    best_f1["criterion"] = "top_f1"
    best_f1["criterion_value"] = best_f1["avg_f1"]
    top = pd.concat([best_valid, best_f1], ignore_index=True, sort=False)
    return top[output_cols].sort_values(group_cols + ["criterion"]).reset_index(drop=True)


def build_missing_valid_groups(counts: pd.DataFrame, inventory: pd.DataFrame, min_valid_runs: int) -> pd.DataFrame:
    group_cols = ["graph", "model", "obs_n", "int_n"]
    missing = counts[counts["total_valid_rows"].lt(min_valid_runs)].copy()
    if missing.empty:
        return missing

    attempted = (
        inventory.groupby(group_cols, dropna=False)
        .agg(
            attempted_configs=("prompt_config_key", lambda s: compact_examples(s, 20)),
            attempted_prompt_styles=(
                "prompt_style",
                lambda s: ",".join(sorted(x for x in s.dropna().astype(str).unique() if x)),
            ),
            attempted_wrappers=(
                "wrapper_mode",
                lambda s: ",".join(sorted(x for x in s.dropna().astype(str).unique() if x)),
            ),
            attempted_format_hints=(
                "append_format_hint",
                lambda s: ",".join(sorted(x for x in s.dropna().astype(str).unique() if x)),
            ),
            attempted_col_orders=(
                "col_order",
                lambda s: ",".join(sorted(x for x in s.dropna().astype(str).unique() if x)),
            ),
            attempted_anonymize_values=(
                "anonymize",
                lambda s: ",".join(sorted(x for x in s.dropna().astype(str).unique() if x)),
            ),
        )
        .reset_index()
    )
    missing = missing.merge(attempted, on=group_cols, how="left")
    front = group_cols + [
        "prompt_config_count",
        "total_valid_rows",
        "valid_prompt_config_count",
        "best_valid_rate",
        "best_valid_config",
        "attempted_prompt_styles",
        "attempted_wrappers",
        "attempted_format_hints",
        "attempted_col_orders",
        "attempted_anonymize_values",
        "attempted_configs",
    ]
    rest = [col for col in missing.columns if col not in front]
    return missing[front + rest].sort_values(group_cols).reset_index(drop=True)


def apply_budget_filter(inventory: pd.DataFrame, max_obs: int, max_int: int) -> pd.DataFrame:
    out = inventory
    if max_obs >= 0:
        out = out[out["obs_n"] <= max_obs]
    if max_int >= 0:
        out = out[out["int_n"] <= max_int]
    return out.reset_index(drop=True)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path}")


def main() -> None:
    args = parse_args()
    roots = [resolve(path) for path in (args.responses_root or DEFAULT_RESPONSE_ROOTS)]
    graphs = {graph.lower() for graph in args.graphs} if args.graphs else None

    raw = read_eval_summaries(roots, graphs, args.include_sachs_old)
    inventory = normalize_inventory(raw, paper_models_only=not args.all_models)
    inventory = apply_budget_filter(inventory, args.max_obs, args.max_int)
    counts = build_counts(inventory, args.max_examples)
    rankings = build_rankings(inventory)
    top_configs = build_top_configs(inventory)
    missing_valid = build_missing_valid_groups(counts, inventory, args.min_valid_runs)

    write_csv(resolve(args.inventory_out), inventory)
    write_csv(resolve(args.rankings_out), rankings)
    write_csv(resolve(args.top_out), top_configs)
    write_csv(resolve(args.missing_valid_out), missing_valid)
    write_csv(resolve(args.counts_out), counts)
    print(
        f"[summary] groups={len(counts)} prompt_configs={len(inventory)} "
        f"graphs={counts['graph'].nunique()} models={counts['model'].nunique()}"
    )
    print(f"[summary] missing_valid_groups={len(missing_valid)} min_valid_runs={args.min_valid_runs}")
    if args.fail_on_missing_valid and not missing_valid.empty:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
