#!/usr/bin/env python3
"""Collect best prompt configs per graph/model/budget/style/anonymization group.

Default scope matches the paper audit:
  - current and legacy <graph>/eval_summary.csv files
  - 10 base paper LLMs only
  - obs_n <= 5000 and int_n <= 200
  - summary/matrix data prompts must use p5
  - sachs_old excluded

For each group

  graph, model, obs_n, int_n, prompt_style, anonymize

the script first restricts to configs with the top valid_rate, then chooses the
best result among those configs by avg_f1, valid_rows, lower avg_shd, and a
deterministic filename tie-break.
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
DEFAULT_PER_GRAPH_ROOT = Path("scripts/responses")
DEFAULT_BEST_OUT = Path("experiments/out/best_prompt_configs_by_valid_rate.csv")
DEFAULT_TOP_VALID_OUT = Path("experiments/out/top_valid_rate_prompt_configs.csv")
PER_GRAPH_BEST_NAME = "best_prompt_configs_by_valid_rate.csv"
PER_GRAPH_TOP_VALID_NAME = "top_valid_rate_prompt_configs.csv"
DEFAULT_MAX_OBS = 5000
DEFAULT_MAX_INT = 200
DEFAULT_DATA_PROMPT_COUNT = 5

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
        help="Root containing <graph>/eval_summary.csv. Repeatable. Defaults to current and legacy roots.",
    )
    parser.add_argument(
        "--per-graph-root",
        type=Path,
        default=DEFAULT_PER_GRAPH_ROOT,
        help="Root where per-graph selected CSVs are written.",
    )
    parser.add_argument(
        "--best-out",
        type=Path,
        default=DEFAULT_BEST_OUT,
        help="Combined CSV with one selected best config per group.",
    )
    parser.add_argument(
        "--top-valid-out",
        type=Path,
        default=DEFAULT_TOP_VALID_OUT,
        help="Combined CSV with all configs tied for top valid_rate per group.",
    )
    parser.add_argument(
        "--graphs",
        nargs="*",
        default=None,
        help="Optional graph filter. Defaults to every graph under --responses-root.",
    )
    parser.add_argument(
        "--include-sachs-old",
        action="store_true",
        help="Include sachs_old. Default excludes it.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Include every model instead of only the 10 base paper models.",
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
        "--data-prompt-count",
        type=int,
        default=DEFAULT_DATA_PROMPT_COUNT,
        help="For summary/matrix prompts, keep only response files tagged with this p-count. Use negative to disable.",
    )
    parser.add_argument(
        "--no-per-graph",
        action="store_true",
        help="Only write combined outputs; do not write per-graph CSVs under scripts/responses/<graph>/.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def numeric_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def text_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="object")
    return df[col].fillna(default).astype(str)


def flag_series(df: pd.DataFrame, col: str, default: int = 0) -> pd.Series:
    raw = text_series(df, col, str(default)).str.strip().str.lower()
    numeric = pd.to_numeric(raw, errors="coerce")
    out = pd.Series(default, index=df.index, dtype="Int64")
    out.loc[numeric.notna()] = numeric.loc[numeric.notna()].round().astype("Int64")
    out.loc[raw.isin({"true", "yes", "y", "t"})] = 1
    out.loc[raw.isin({"false", "no", "n", "f", ""})] = 0
    return out


def f1_series(df: pd.DataFrame) -> pd.Series:
    f1 = numeric_series(df, "avg_f1")
    alt = numeric_series(df, "avg_F1")
    return f1.where(f1.notna(), alt)


def infer_prompt_count(basename: object) -> int | pd.NA:
    match = re.search(r"(?:^|_)p(?P<count>\d+)(?:_|$)", str(basename))
    if not match:
        return pd.NA
    return int(match.group("count"))


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


def normalize_paper_model(raw_model: str) -> str:
    raw = str(raw_model).strip()
    if raw in PAPER_MODEL_DISPLAY_BY_RAW:
        return PAPER_MODEL_DISPLAY_BY_RAW[raw]
    basename = Path(raw).name
    if basename in PAPER_MODEL_DISPLAY_BY_RAW:
        return PAPER_MODEL_DISPLAY_BY_RAW[basename]
    return ""


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
            frame["graph_dir"] = graph
            frame["summary_path"] = str(path)
            frame["response_root"] = str(root)
            if "dataset" not in frame.columns:
                frame["dataset"] = graph
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No eval_summary.csv files found under requested response roots")
    return pd.concat(frames, ignore_index=True, sort=False)


def normalize_rows(
    raw: pd.DataFrame,
    paper_models_only: bool,
    max_obs: int,
    max_int: int,
    data_prompt_count: int,
) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    out["graph"] = text_series(raw, "dataset")
    out["graph_dir"] = text_series(raw, "graph_dir")
    out["prompt_style"] = text_series(raw, "prompt_style")
    out["anonymize"] = flag_series(raw, "anonymize", 0)
    out["reasoning_guidance"] = text_series(raw, "reasoning_guidance")
    out["wrapper_mode"] = text_series(raw, "wrapper_mode")
    out["append_format_hint"] = flag_series(raw, "append_format_hint", 0)
    out["row_order"] = text_series(raw, "row_order")
    out["col_order"] = text_series(raw, "col_order")
    out["give_steps"] = flag_series(raw, "give_steps", 0)
    out["causal_rules"] = flag_series(raw, "causal_rules", 0)
    out["config"] = text_series(raw, "config")
    out["response_csv"] = text_series(raw, "response_csv")
    out["response_basename"] = out["response_csv"].map(lambda value: Path(value).name if value else "")
    out["prompt_count"] = out["response_basename"].map(infer_prompt_count).astype("Int64")
    out["summary_path"] = text_series(raw, "summary_path")
    out["response_root"] = text_series(raw, "response_root")

    out["model_raw"] = text_series(raw, "model").str.strip()
    inferred_model = out["response_basename"].map(infer_model_from_basename)
    missing_model = out["model_raw"].eq("") | out["model_raw"].str.lower().eq("unknown")
    out.loc[missing_model & inferred_model.ne("unknown"), "model_raw"] = inferred_model[
        missing_model & inferred_model.ne("unknown")
    ]
    out["paper_model"] = out["model_raw"].map(normalize_paper_model)
    out["model"] = out["paper_model"].where(out["paper_model"].ne(""), out["model_raw"])
    if paper_models_only:
        out = out[out["paper_model"].ne("")].copy()

    obs = numeric_series(raw, "obs_n")
    inter = numeric_series(raw, "int_n")
    is_names_only = out["prompt_style"].eq("names_only")
    out["obs_n"] = obs.mask(obs.isna() & is_names_only, 0).astype("Int64")
    out["int_n"] = inter.mask(inter.isna() & is_names_only, 0).astype("Int64")
    if max_obs >= 0:
        out = out[out["obs_n"] <= max_obs].copy()
    if max_int >= 0:
        out = out[out["int_n"] <= max_int].copy()

    out["num_rows"] = numeric_series(raw, "num_rows").astype("Int64")
    out["valid_rows"] = numeric_series(raw, "valid_rows").astype("Int64")
    out["valid_rate"] = numeric_series(raw, "valid_rate")
    missing_valid_rate = out["valid_rate"].isna() & out["num_rows"].notna() & (out["num_rows"] > 0)
    out.loc[missing_valid_rate, "valid_rate"] = (
        out.loc[missing_valid_rate, "valid_rows"].astype(float)
        / out.loc[missing_valid_rate, "num_rows"].astype(float)
    )

    metric_cols = [
        "avg_accuracy",
        "avg_precision",
        "avg_recall",
        "avg_shd",
        "acyclic_rate",
        "format_rate",
        "consensus_f1",
        "consensus_shd",
        "avg_skeleton_f1",
        "avg_orientation_accuracy",
        "avg_vstruct_f1",
        "avg_ancestor_f1",
    ]
    out["avg_f1"] = f1_series(raw)
    for col in metric_cols:
        out[col] = numeric_series(raw, col)

    out["prompt_config_key"] = out["response_basename"]
    no_basename = out["prompt_config_key"].eq("")
    out.loc[no_basename, "prompt_config_key"] = out.loc[no_basename, "config"]

    keep = (
        out["graph"].ne("")
        & out["model"].ne("")
        & out["obs_n"].notna()
        & out["int_n"].notna()
        & out["prompt_style"].ne("")
        & out["prompt_config_key"].ne("")
    )
    out = out[keep].copy()

    if data_prompt_count >= 0:
        is_data_prompt = out["prompt_style"].isin(["summary", "matrix"])
        out = out[(~is_data_prompt) | out["prompt_count"].eq(data_prompt_count)].copy()

    dedup_cols = ["graph", "model", "obs_n", "int_n", "prompt_style", "anonymize", "prompt_config_key"]
    out = (
        out.sort_values(dedup_cols + ["summary_path"])
        .drop_duplicates(dedup_cols, keep="first")
        .reset_index(drop=True)
    )
    return out


def sort_for_reporting(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_model_order"] = out["model"].map({model: i for i, model in enumerate(PAPER_MODEL_ORDER)}).fillna(10_000)
    out = out.sort_values(
        ["graph", "_model_order", "model", "obs_n", "int_n", "prompt_style", "anonymize", "prompt_config_key"],
        na_position="last",
    )
    return out.drop(columns=["_model_order"])


def select_top_valid_configs(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["graph", "model", "obs_n", "int_n", "prompt_style", "anonymize"]
    work = rows.copy()
    work["top_valid_rate"] = work.groupby(group_cols, dropna=False)["valid_rate"].transform("max")
    is_top = work["valid_rate"].eq(work["top_valid_rate"]) | (
        work["valid_rate"].isna() & work["top_valid_rate"].isna()
    )
    top = work[is_top].copy()
    top["num_top_valid_rate_configs"] = top.groupby(group_cols, dropna=False)["prompt_config_key"].transform("nunique")
    return top


def select_best_from_top_valid(rows: pd.DataFrame, top: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["graph", "model", "obs_n", "int_n", "prompt_style", "anonymize"]
    group_stats = (
        rows.groupby(group_cols, dropna=False)
        .agg(
            prompt_config_count=("prompt_config_key", "nunique"),
            total_valid_rows=("valid_rows", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
            valid_prompt_config_count=("valid_rows", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        )
        .reset_index()
    )

    selected = (
        top.sort_values(
            group_cols + ["valid_rate", "avg_f1", "valid_rows", "avg_shd", "prompt_config_key"],
            ascending=[True, True, True, True, True, True, False, False, False, True, True],
            na_position="last",
        )
        .drop_duplicates(group_cols, keep="first")
        .reset_index(drop=True)
    )
    selected = selected.merge(group_stats, on=group_cols, how="left")
    selected["has_valid_run"] = selected["total_valid_rows"].ge(1)
    selected["selection_rule"] = "top_valid_rate_then_best_f1"
    return selected


def output_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "graph",
        "model",
        "model_raw",
        "paper_model",
        "obs_n",
        "int_n",
        "prompt_style",
        "anonymize",
        "prompt_count",
        "selection_rule",
        "prompt_config_count",
        "num_top_valid_rate_configs",
        "total_valid_rows",
        "valid_prompt_config_count",
        "has_valid_run",
        "valid_rate",
        "avg_f1",
        "avg_shd",
        "avg_accuracy",
        "avg_precision",
        "avg_recall",
        "acyclic_rate",
        "format_rate",
        "consensus_f1",
        "consensus_shd",
        "avg_skeleton_f1",
        "avg_orientation_accuracy",
        "avg_vstruct_f1",
        "avg_ancestor_f1",
        "valid_rows",
        "num_rows",
        "reasoning_guidance",
        "wrapper_mode",
        "append_format_hint",
        "row_order",
        "col_order",
        "give_steps",
        "causal_rules",
        "config",
        "prompt_config_key",
        "response_basename",
        "response_csv",
        "response_root",
        "summary_path",
    ]
    return [col for col in preferred if col in df.columns] + [col for col in df.columns if col not in preferred]


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path}")


def write_per_graph(root: Path, best: pd.DataFrame, top: pd.DataFrame) -> None:
    for graph_dir, graph_best in best.groupby("graph_dir", dropna=False):
        if not graph_dir:
            continue
        graph_root = root / str(graph_dir)
        write_csv(graph_root / PER_GRAPH_BEST_NAME, graph_best[output_columns(graph_best)])
    for graph_dir, graph_top in top.groupby("graph_dir", dropna=False):
        if not graph_dir:
            continue
        graph_root = root / str(graph_dir)
        write_csv(graph_root / PER_GRAPH_TOP_VALID_NAME, graph_top[output_columns(graph_top)])


def main() -> None:
    args = parse_args()
    roots = [resolve(path) for path in (args.responses_root or DEFAULT_RESPONSE_ROOTS)]
    per_graph_root = resolve(args.per_graph_root)
    graphs = {graph.lower() for graph in args.graphs} if args.graphs else None

    raw = read_eval_summaries(roots, graphs, args.include_sachs_old)
    rows = normalize_rows(
        raw,
        paper_models_only=not args.all_models,
        max_obs=args.max_obs,
        max_int=args.max_int,
        data_prompt_count=args.data_prompt_count,
    )
    top_valid = select_top_valid_configs(rows)
    best = select_best_from_top_valid(rows, top_valid)

    group_cols = ["graph", "model", "obs_n", "int_n", "prompt_style", "anonymize"]
    top_valid_keys = set(
        tuple(row[col] for col in group_cols + ["prompt_config_key"])
        for _, row in best[group_cols + ["prompt_config_key"]].iterrows()
    )
    top_valid = top_valid.copy()
    top_valid["is_selected_best_result"] = [
        tuple(row[col] for col in group_cols + ["prompt_config_key"]) in top_valid_keys
        for _, row in top_valid[group_cols + ["prompt_config_key"]].iterrows()
    ]

    best = sort_for_reporting(best)
    top_valid = sort_for_reporting(top_valid)

    write_csv(resolve(args.best_out), best[output_columns(best)])
    write_csv(resolve(args.top_valid_out), top_valid[output_columns(top_valid)])
    if not args.no_per_graph:
        write_per_graph(per_graph_root, best, top_valid)

    print(
        f"[summary] selected_groups={len(best)} top_valid_rows={len(top_valid)} "
        f"graphs={best['graph'].nunique()} models={best['model'].nunique()} "
        f"missing_valid_groups={int((~best['has_valid_run']).sum())}"
    )


if __name__ == "__main__":
    main()
