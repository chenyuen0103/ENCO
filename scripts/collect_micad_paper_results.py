#!/usr/bin/env python3
"""Collect MICAD paper-ready result tables from evaluated response summaries.

This script reads condition-level summaries under scripts/responses/<graph>/
and creates a smaller, auditable set of CSV/LaTeX tables and plot-data files
for the paper.  It deliberately keeps provenance columns so every table entry
can be traced back to the selected source artifact.

Default input:
  scripts/responses/{cancer,earthquake,asia,sachs}/eval_summary.csv

Default output:
  experiments/out/micad_paper/

The cross-graph evidence ladder includes both real-name and anonymized
summary/matrix columns so plotting can emit both Figure 4 variants.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


GRAPH_ORDER = ["cancer", "earthquake", "asia", "sachs"]
CONDITION_ORDER = ["names_only", "real+summary", "anon+summary", "real+matrix", "anon+matrix"]
DATA_ONLY_METHODS = ["PC", "GES", "ENCO"]


@dataclass(frozen=True)
class ModelInfo:
    method: str
    display: str
    family: str
    size_label: str
    size_value: float
    openness: str


MODEL_INFOS = [
    ModelInfo("gpt-5-mini", "GPT-5 mini", "OpenAI", "mini", 1.0, "closed"),
    ModelInfo("gpt-5.2-pro", "GPT-5.2 pro", "OpenAI", "pro", 2.0, "closed"),
    ModelInfo("Qwen/Qwen3-4B-Thinking-2507", "Qwen3-4B", "Qwen3", "4B", 4.0, "open"),
    ModelInfo("Qwen/Qwen3-30B-A3B-Thinking-2507", "Qwen3-30B-A3B", "Qwen3", "30B-A3B", 30.0, "open"),
    ModelInfo("Qwen/Qwen2.5-7B-Instruct-1M", "Qwen2.5-7B", "Qwen2.5", "7B", 7.0, "open"),
    ModelInfo("Qwen/Qwen2.5-14B-Instruct-1M", "Qwen2.5-14B", "Qwen2.5", "14B", 14.0, "open"),
    ModelInfo("Qwen/Qwen2.5-72B-Instruct-AWQ", "Qwen2.5-72B", "Qwen2.5", "72B", 72.0, "open"),
    ModelInfo("meta-llama/Meta-Llama-3.1-8B", "Llama-3.1-8B", "Llama 3.1", "8B", 8.0, "open"),
    ModelInfo("meta-llama/Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B-Inst.", "Llama 3.1", "8B-Inst.", 8.1, "open"),
    ModelInfo("meta-llama/Llama-3.1-70B-Instruct", "Llama-3.1-70B", "Llama 3.1", "70B", 70.0, "open"),
]

MODEL_INFO_BY_METHOD = {m.method: m for m in MODEL_INFOS}
DISPLAY_BY_METHOD = {m.method: m.display for m in MODEL_INFOS} | {
    "PC": "PC",
    "GES": "GES",
    "ENCO": "ENCO",
    "CausalLLMPrompt_names_only": "Causal-LLM prompt",
    "JiralerspongBFS": "BFS-style LLM",
    "TakayamaSCP": "Takayama SCP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["responses", "metrics-csv"],
        default="responses",
        help="Use scripts/responses/* summaries by default; metrics-csv is a legacy fallback.",
    )
    parser.add_argument(
        "--responses-root",
        type=Path,
        default=Path("scripts/responses"),
        help="Root containing per-graph eval_summary.csv files.",
    )
    parser.add_argument(
        "--baseline-fallback-roots",
        nargs="*",
        type=Path,
        default=[Path("experiments/responses_old")],
        help=(
            "Old response roots used only for missing PC/GES/ENCO baseline rows. "
            "LLM rows are never read from these roots."
        ),
    )
    parser.add_argument(
        "--llm-fallback-roots",
        nargs="*",
        type=Path,
        default=[Path("experiments/responses_old"), Path("experiments/responses")],
        help=(
            "Old response roots used only for supplemental GPT-5.2-Pro thinktags rows. "
            "Current scripts_eval_summary rows are preferred on ties."
        ),
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("experiments/out/micad_eval_results/all_condition_metrics.csv"),
        help="Legacy input condition-level metrics CSV used only with --source metrics-csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/out/micad_paper"),
        help="Output directory for paper-ready tables and plot data.",
    )
    parser.add_argument("--graph", default="sachs", choices=GRAPH_ORDER, help="Primary graph for headline tables.")
    parser.add_argument("--obs", type=int, default=5000, help="Primary observational budget N.")
    parser.add_argument("--inter", type=int, default=200, help="Primary interventional budget M.")
    parser.add_argument(
        "--dedup-policy",
        choices=["max-valid", "best-valid-rate", "best-f1", "first"],
        default="max-valid",
        help="How to choose among duplicate source artifacts for one cell.",
    )
    parser.add_argument(
        "--variant-contains",
        default=None,
        help="Optional substring filter for LLM prompt variants; baselines and names_only are always kept.",
    )
    return parser.parse_args()


def normalize_model_name(model: object) -> str:
    name = "" if pd.isna(model) else str(model)
    if name in MODEL_INFO_BY_METHOD:
        return name
    candidates = [
        f"Qwen/{name}",
        f"meta-llama/{name}",
    ]
    for candidate in candidates:
        if candidate in MODEL_INFO_BY_METHOD:
            return candidate
    return name


def normalize_prompt_style(style: object) -> str:
    text = "" if pd.isna(style) else str(style).strip().lower()
    text = text.replace("summary_joint", "summary").replace("summary_join", "summary")
    if text in {"names_only", "summary", "matrix"}:
        return text
    return text


def parse_baseline_summary_path(path: Path) -> tuple[int, int, str] | None:
    match = re.match(
        r"predictions_obs(?P<obs>\d+)_int(?P<inter>\d+)_(?P<method>PC|GES|ENCO)\.csv\.summary\.json$",
        path.name,
    )
    if not match:
        return None
    return int(match.group("obs")), int(match.group("inter")), match.group("method")


def parse_llm_summary_path(path: Path) -> tuple[int, int, str, str, str] | None:
    name = path.name
    if name.endswith(".summary.json"):
        name = name.removesuffix(".summary.json")
    elif name.endswith(".per_row.csv"):
        name = name.removesuffix(".per_row.csv")
    else:
        return None
    if name.endswith(".csv"):
        name = name.removesuffix(".csv")

    match = re.match(
        r"responses_obs(?P<obs>\d+)_int(?P<inter>\d+)_shuf\d+_p\d+_(?P<middle>.+)_(?P<model>[^_]+(?:_[^_]+)*)$",
        name,
    )
    if not match:
        return None
    middle = match.group("middle")
    if "_summary" in f"_{middle}":
        fmt = "summary"
    elif "_matrix" in f"_{middle}":
        fmt = "matrix"
    else:
        return None
    semantic = "anon" if "anon" in middle.split("_") else "real"
    return int(match.group("obs")), int(match.group("inter")), semantic, fmt, normalize_model_name(match.group("model"))


def summary_row_from_payload(
    *,
    graph: str,
    method: str,
    obs: int,
    inter: int,
    semantic: str,
    fmt: str,
    path: Path,
    payload: dict,
    source_kind: str,
) -> dict:
    condition = f"{semantic}+{fmt}" if semantic in {"real", "anon"} else semantic
    return {
        "graph": graph,
        "method": method,
        "obs": obs,
        "inter": inter,
        "semantic": semantic,
        "format": fmt,
        "condition": condition,
        "mean_f1": payload.get("avg_f1"),
        "mean_shd": payload.get("avg_shd"),
        "std_f1": payload.get("var_f1_sd"),
        "std_shd": payload.get("var_shd_sd"),
        "valid_rows": payload.get("valid_rows"),
        "n_rows": payload.get("num_rows"),
        "path": str(path),
        "source_kind": source_kind,
        "variant": path.name,
    }


def summarize_per_row(path: Path) -> dict:
    per_row = pd.read_csv(path)
    valid_col = "f1" if "f1" in per_row.columns else None
    valid_rows = int(per_row[valid_col].notna().sum()) if valid_col else 0
    return {
        "num_rows": int(len(per_row)),
        "valid_rows": valid_rows,
        "avg_f1": pd.to_numeric(per_row.get("f1"), errors="coerce").mean(),
        "avg_shd": pd.to_numeric(per_row.get("shd"), errors="coerce").mean(),
        "var_f1_sd": pd.to_numeric(per_row.get("f1"), errors="coerce").std(),
        "var_shd_sd": pd.to_numeric(per_row.get("shd"), errors="coerce").std(),
    }


def load_old_llm_fallbacks(roots: list[Path], existing: pd.DataFrame) -> pd.DataFrame:
    existing_keys = set(
        existing.loc[~existing["method"].isin(DATA_ONLY_METHODS), ["graph", "method", "obs", "inter", "condition"]]
        .itertuples(index=False, name=None)
    )
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for graph in GRAPH_ORDER:
            graph_dir = root / graph
            if not graph_dir.exists():
                continue
            candidates = list(graph_dir.glob("responses_obs*_int*_*.csv.summary.json"))
            candidates.extend(graph_dir.glob("responses_obs*_int*_*.csv.per_row.csv"))
            for path in sorted(candidates):
                parsed = parse_llm_summary_path(path)
                if parsed is None:
                    continue
                obs, inter, semantic, fmt, method = parsed
                if method not in MODEL_INFO_BY_METHOD:
                    continue
                condition = f"{semantic}+{fmt}"
                key = (graph, method, obs, inter, condition)
                prefer_old_thinktags = method == "gpt-5.2-pro" and "thinktags" in path.name.lower()
                if not prefer_old_thinktags:
                    continue
                if path.name.endswith(".summary.json"):
                    try:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as exc:
                        print(f"[warn] Could not parse LLM fallback summary {path}: {exc}")
                        continue
                else:
                    payload = summarize_per_row(path)
                rows.append(
                    summary_row_from_payload(
                        graph=graph,
                        method=method,
                        obs=obs,
                        inter=inter,
                        semantic=semantic,
                        fmt=fmt,
                        path=path,
                        payload=payload,
                        source_kind="preferred_old_llm_thinktags" if prefer_old_thinktags else "old_llm_summary",
                    )
                )
    if not rows:
        return existing.iloc[0:0].copy()
    out = pd.DataFrame(rows)
    for col in ["obs", "inter", "mean_f1", "mean_shd", "std_f1", "std_shd", "valid_rows", "n_rows"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[out["mean_f1"].notna()].copy()
    out["method_display"] = out["method"].map(display_method)
    out["is_data_only"] = False
    out["is_llm_model"] = out["method"].isin(MODEL_INFO_BY_METHOD)
    return out


def load_old_baseline_fallbacks(roots: list[Path], existing: pd.DataFrame) -> pd.DataFrame:
    existing_keys = set(
        existing.loc[existing["method"].isin(DATA_ONLY_METHODS), ["graph", "method", "obs", "inter"]]
        .itertuples(index=False, name=None)
    )
    rows = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for graph in GRAPH_ORDER:
            graph_dir = root / graph
            if not graph_dir.exists():
                continue
            for path in sorted(graph_dir.glob("predictions_obs*_int*_*.csv.summary.json")):
                parsed = parse_baseline_summary_path(path)
                if parsed is None:
                    continue
                obs, inter, method = parsed
                key = (graph, method, obs, inter)
                if key in existing_keys or key in seen:
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    print(f"[warn] Could not parse baseline summary {path}: {exc}")
                    continue
                rows.append(
                    {
                        "graph": graph,
                        "method": method,
                        "obs": obs,
                        "inter": inter,
                        "semantic": "data_only",
                        "format": "data_only",
                        "condition": f"{method} data-only",
                        "mean_f1": payload.get("avg_f1"),
                        "mean_shd": payload.get("avg_shd"),
                        "std_f1": payload.get("var_f1_sd"),
                        "std_shd": payload.get("var_shd_sd"),
                        "valid_rows": payload.get("valid_rows"),
                        "n_rows": payload.get("num_rows"),
                        "path": str(path),
                        "source_kind": "old_baseline_summary",
                        "variant": path.name,
                    }
                )
                seen.add(key)

    if not rows:
        return existing.iloc[0:0].copy()
    out = pd.DataFrame(rows)
    for col in ["obs", "inter", "mean_f1", "mean_shd", "std_f1", "std_shd", "valid_rows", "n_rows"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[out["mean_f1"].notna()].copy()
    out["method_display"] = out["method"].map(display_method)
    out["is_data_only"] = True
    out["is_llm_model"] = False
    return out


def load_response_summaries(
    responses_root: Path,
    variant_contains: str | None,
    baseline_fallback_roots: list[Path],
    llm_fallback_roots: list[Path],
) -> pd.DataFrame:
    frames = []
    for graph in GRAPH_ORDER:
        path = responses_root / graph / "eval_summary.csv"
        if not path.exists():
            print(f"[warn] Missing response summary: {path}")
            continue
        src = pd.read_csv(path)
        src["graph"] = src.get("dataset", graph).fillna(graph).astype(str)
        src = src[src["graph"].eq(graph)].copy()
        src["method"] = src["model"].map(normalize_model_name)
        src["obs"] = pd.to_numeric(src.get("obs_n", 0), errors="coerce").fillna(0).astype(int)
        src["inter"] = pd.to_numeric(src.get("int_n", 0), errors="coerce").fillna(0).astype(int)
        src["format"] = src.get("prompt_style", "").map(normalize_prompt_style)
        src["semantic"] = np.where(
            src["format"].eq("names_only"),
            "names_only",
            np.where(pd.to_numeric(src.get("anonymize", 0), errors="coerce").fillna(0).astype(int).eq(1), "anon", "real"),
        )
        src["condition"] = src.apply(infer_condition, axis=1)
        src["mean_f1"] = pd.to_numeric(src.get("avg_f1", np.nan), errors="coerce")
        src["mean_shd"] = pd.to_numeric(src.get("avg_shd", np.nan), errors="coerce")
        src["std_f1"] = pd.to_numeric(src.get("var_f1_sd", src.get("avg_f1_sd", np.nan)), errors="coerce")
        src["std_shd"] = pd.to_numeric(src.get("var_shd_sd", src.get("avg_shd_sd", np.nan)), errors="coerce")
        src["n_rows"] = pd.to_numeric(src.get("num_rows", np.nan), errors="coerce")
        src["valid_rows"] = pd.to_numeric(src.get("valid_rows", np.nan), errors="coerce")
        src["path"] = src.get("response_csv", "").fillna("").astype(str)
        src["variant"] = src["path"].map(lambda p: Path(str(p)).name)
        src["source_kind"] = "scripts_eval_summary"
        frames.append(src)

    if not frames:
        raise FileNotFoundError(f"No eval_summary.csv files found under {responses_root}")

    df = pd.concat(frames, ignore_index=True, sort=False)
    keep_cols = [
        "graph",
        "method",
        "obs",
        "inter",
        "semantic",
        "format",
        "condition",
        "mean_f1",
        "mean_shd",
        "std_f1",
        "std_shd",
        "valid_rows",
        "n_rows",
        "path",
        "source_kind",
        "variant",
    ]
    df = df[keep_cols].copy()
    df = df[df["graph"].isin(GRAPH_ORDER)].copy()
    df["method_display"] = df["method"].map(display_method)
    df["is_data_only"] = df["method"].isin(DATA_ONLY_METHODS)
    df["is_llm_model"] = df["method"].isin(MODEL_INFO_BY_METHOD)

    if variant_contains:
        variant = df.get("variant", pd.Series("", index=df.index)).astype(str)
        keep = (
            df["is_data_only"]
            | df["condition"].eq("names_only")
            | variant.str.contains(variant_contains, case=False, na=False)
        )
        df = df[keep].copy()
    fallback = load_old_baseline_fallbacks(baseline_fallback_roots, df)
    if not fallback.empty:
        print(f"[info] Added {len(fallback)} missing PC/GES/ENCO baseline rows from old response roots.")
        df = pd.concat([df, fallback], ignore_index=True, sort=False)
    llm_fallback = load_old_llm_fallbacks(llm_fallback_roots, df)
    if not llm_fallback.empty:
        print(f"[info] Added {len(llm_fallback)} supplemental LLM rows from old response roots.")
        df = pd.concat([df, llm_fallback], ignore_index=True, sort=False)
    return df


def load_metrics(path: Path, variant_contains: str | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {path}")
    df = pd.read_csv(path)
    required = {"graph", "method", "obs", "inter", "semantic", "format", "condition", "mean_f1", "mean_shd"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df = df[df["graph"].isin(GRAPH_ORDER)].copy()
    for col in ["obs", "inter", "mean_f1", "mean_shd", "std_f1", "std_shd", "valid_rows", "n_rows"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["condition"] = df["condition"].fillna(df.apply(infer_condition, axis=1))
    df["method_display"] = df["method"].map(display_method)
    df["is_data_only"] = df["method"].isin(DATA_ONLY_METHODS)
    df["is_llm_model"] = df["method"].isin(MODEL_INFO_BY_METHOD)

    if variant_contains:
        variant = df.get("variant", pd.Series("", index=df.index)).astype(str)
        keep = (
            df["is_data_only"]
            | df["condition"].eq("names_only")
            | variant.str.contains(variant_contains, case=False, na=False)
        )
        df = df[keep].copy()
    return df


def infer_condition(row: pd.Series) -> str:
    semantic = str(row.get("semantic", ""))
    fmt = str(row.get("format", ""))
    if semantic == "names_only" or fmt == "names_only":
        return "names_only"
    if fmt == "data_only":
        return f"{row.get('method', 'data')} data-only"
    if semantic in {"real", "anon"} and fmt:
        return f"{semantic}+{fmt}"
    return fmt or semantic or "unknown"


def display_method(method: str) -> str:
    if method in DISPLAY_BY_METHOD:
        return DISPLAY_BY_METHOD[method]
    return method.replace("Qwen/", "").replace("meta-llama/", "")


def source_priority(source_kind: object) -> int:
    source = str(source_kind)
    if source == "per_row":
        return 2
    if source == "summary_json":
        return 1
    if source == "scripts_eval_summary":
        return 3
    if source == "old_baseline_summary":
        return 2
    if source == "preferred_old_llm_thinktags":
        return 1
    if source == "old_llm_summary":
        return 1
    return 0


def canonicalize(df: pd.DataFrame, dedup_policy: str) -> pd.DataFrame:
    df = df.copy()
    df["_source_priority"] = df.get("source_kind", "").map(source_priority)
    df["_valid_rows"] = df.get("valid_rows", 0).fillna(0)
    df["_n_rows"] = df.get("n_rows", 0).fillna(0)
    df["_valid_rate"] = np.where(df["_n_rows"] > 0, df["_valid_rows"] / df["_n_rows"], np.nan)
    df["_path"] = df.get("path", "").fillna("").astype(str)

    group_cols = ["graph", "method", "obs", "inter", "condition"]
    rows = []
    for _, sub in df.groupby(group_cols, dropna=False, sort=False):
        if dedup_policy == "best-f1":
            chosen = sub.sort_values(
                ["mean_f1", "_valid_rows", "_source_priority", "_n_rows", "_path"],
                ascending=[False, False, False, False, True],
            ).iloc[0]
        elif dedup_policy == "best-valid-rate":
            chosen = sub.sort_values(
                [
                    "_valid_rate",
                    "_valid_rows",
                    "_source_priority",
                    "mean_f1",
                    "_n_rows",
                    "_path",
                ],
                ascending=[False, False, False, False, False, True],
                na_position="last",
            ).iloc[0]
        elif dedup_policy == "first":
            chosen = sub.sort_values("_path").iloc[0]
        else:
            chosen = sub.sort_values(
                ["_valid_rows", "_source_priority", "_n_rows", "mean_f1", "_path"],
                ascending=[False, False, False, False, True],
            ).iloc[0]
        rows.append(chosen)
    out = pd.DataFrame(rows).drop(
        columns=[
            c
            for c in ["_source_priority", "_valid_rows", "_n_rows", "_valid_rate", "_path"]
            if c in df
        ]
    )
    out = out.sort_values(["graph", "method", "obs", "inter", "condition"]).reset_index(drop=True)
    return out


def get_cell(df: pd.DataFrame, graph: str, method: str, condition: str, obs: int, inter: int) -> pd.Series | None:
    sub = df[(df.graph == graph) & (df.method == method) & (df.condition == condition)]
    if condition != "names_only":
        sub = sub[(sub.obs == obs) & (sub.inter == inter)]
    if sub.empty:
        return None
    return sub.iloc[0]


def score(row: pd.Series | None, col: str = "mean_f1") -> float:
    if row is None:
        return np.nan
    val = row.get(col, np.nan)
    return float(val) if pd.notna(val) else np.nan


def standard_error(row: pd.Series | None, value_col: str = "std_f1") -> float:
    if row is None:
        return np.nan
    std = score(row, value_col)
    valid_rows = score(row, "valid_rows")
    if not pd.notna(std) or not pd.notna(valid_rows) or valid_rows <= 1:
        return np.nan
    return std / np.sqrt(valid_rows)


def fmt_f1_shd(f1: float, shd: float) -> str:
    if pd.isna(f1):
        return "--"
    if pd.isna(shd):
        return f"{f1:.2f}"
    return f"{f1:.2f} ({shd:.1f})"


def best_data_only(df: pd.DataFrame, graph: str, obs: int, inter: int) -> pd.Series | None:
    sub = df[(df.graph == graph) & (df.method.isin(DATA_ONLY_METHODS)) & (df.obs == obs) & (df.inter == inter)]
    if sub.empty:
        return None
    return sub.sort_values(["mean_f1", "valid_rows"], ascending=[False, False]).iloc[0]


def collect_evidence_ladder(df: pd.DataFrame, graph: str, obs: int, inter: int) -> pd.DataFrame:
    rows = []
    for info in MODEL_INFOS:
        cells = {cond: get_cell(df, graph, info.method, cond, obs, inter) for cond in CONDITION_ORDER}
        base = best_data_only(df, graph, obs, inter)
        values = {cond: score(row) for cond, row in cells.items()}
        shds = {cond: score(row, "mean_shd") for cond, row in cells.items()}
        ses = {cond: standard_error(row, "std_f1") for cond, row in cells.items()}
        valid_counts = {cond: score(row, "valid_rows") for cond, row in cells.items()}
        total_counts = {cond: score(row, "n_rows") for cond, row in cells.items()}
        best_data_f1 = score(base)
        llm_candidates = [values.get("real+summary", np.nan), values.get("real+matrix", np.nan)]
        finite_candidates = [v for v in llm_candidates if pd.notna(v)]
        best_llm_f1 = max(finite_candidates) if finite_candidates else np.nan
        rows.append(
            {
                "graph": graph,
                "method": info.method,
                "model": info.display,
                "family": info.family,
                "size_label": info.size_label,
                "size_value": info.size_value,
                "obs": obs,
                "inter": inter,
                "names_only_f1": values["names_only"],
                "real_summary_f1": values["real+summary"],
                "real_matrix_f1": values["real+matrix"],
                "anon_summary_f1": values["anon+summary"],
                "anon_matrix_f1": values["anon+matrix"],
                "names_only_f1_se": ses["names_only"],
                "real_summary_f1_se": ses["real+summary"],
                "real_matrix_f1_se": ses["real+matrix"],
                "anon_summary_f1_se": ses["anon+summary"],
                "anon_matrix_f1_se": ses["anon+matrix"],
                "names_only_valid_rows": valid_counts["names_only"],
                "real_summary_valid_rows": valid_counts["real+summary"],
                "real_matrix_valid_rows": valid_counts["real+matrix"],
                "anon_summary_valid_rows": valid_counts["anon+summary"],
                "anon_matrix_valid_rows": valid_counts["anon+matrix"],
                "names_only_n_rows": total_counts["names_only"],
                "real_summary_n_rows": total_counts["real+summary"],
                "real_matrix_n_rows": total_counts["real+matrix"],
                "anon_summary_n_rows": total_counts["anon+summary"],
                "anon_matrix_n_rows": total_counts["anon+matrix"],
                "names_only_shd": shds["names_only"],
                "real_summary_shd": shds["real+summary"],
                "real_matrix_shd": shds["real+matrix"],
                "anon_summary_shd": shds["anon+summary"],
                "anon_matrix_shd": shds["anon+matrix"],
                "best_llm_f1": best_llm_f1,
                "best_data_only_method": None if base is None else base.method,
                "best_data_only_f1": best_data_f1,
                "gap_to_data_only": best_data_f1 - best_llm_f1 if pd.notna(best_data_f1) and pd.notna(best_llm_f1) else np.nan,
                "best_data_format": best_format(values["real+summary"], values["real+matrix"]),
            }
        )
    return pd.DataFrame(rows)


def best_format(summary: float, matrix: float) -> str:
    if pd.isna(summary) and pd.isna(matrix):
        return "--"
    if pd.isna(summary):
        return "matrix"
    if pd.isna(matrix):
        return "summary"
    if abs(summary - matrix) < 1e-9:
        return "tie"
    return "matrix" if matrix > summary else "summary"


def collect_cross_graph(df: pd.DataFrame, obs: int, inter: int) -> pd.DataFrame:
    rows = []
    for graph in GRAPH_ORDER:
        ladder = collect_evidence_ladder(df, graph, obs, inter)
        rows.append(ladder)
    return pd.concat(rows, ignore_index=True)


def collect_anonymized_cross_graph(cross: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "graph",
        "method",
        "model",
        "family",
        "size_label",
        "size_value",
        "obs",
        "inter",
        "names_only_f1",
        "anon_summary_f1",
        "anon_matrix_f1",
        "names_only_f1_se",
        "anon_summary_f1_se",
        "anon_matrix_f1_se",
        "names_only_valid_rows",
        "anon_summary_valid_rows",
        "anon_matrix_valid_rows",
        "names_only_n_rows",
        "anon_summary_n_rows",
        "anon_matrix_n_rows",
        "names_only_shd",
        "anon_summary_shd",
        "anon_matrix_shd",
        "best_data_only_method",
        "best_data_only_f1",
    ]
    return cross[[col for col in columns if col in cross.columns]].copy()


def collect_contrast_metrics(cross: pd.DataFrame, canonical: pd.DataFrame, obs: int, inter: int) -> pd.DataFrame:
    rows = []
    for _, row in cross.iterrows():
        inter_summary = get_cell(canonical, row.graph, row.method, "real+summary", 0, inter)
        inter_matrix = get_cell(canonical, row.graph, row.method, "real+matrix", 0, inter)
        obs_summary = get_cell(canonical, row.graph, row.method, "real+summary", obs, 0)
        obs_matrix = get_cell(canonical, row.graph, row.method, "real+matrix", obs, 0)
        names_only_f1 = row.names_only_f1
        contrasts = {
            "summary_gain": row.real_summary_f1 - names_only_f1,
            "matrix_gain": row.real_matrix_f1 - names_only_f1,
            "interventional_summary_gain": score(inter_summary) - names_only_f1,
            "interventional_matrix_gain": score(inter_matrix) - names_only_f1,
            "observational_summary_gain": score(obs_summary) - names_only_f1,
            "observational_matrix_gain": score(obs_matrix) - names_only_f1,
            "format_gap_matrix_minus_summary": row.real_matrix_f1 - row.real_summary_f1,
            "anonymization_drop_summary": row.real_summary_f1 - row.anon_summary_f1,
            "anonymization_drop_matrix": row.real_matrix_f1 - row.anon_matrix_f1,
            "gap_to_data_only": row.gap_to_data_only,
        }
        for metric, value in contrasts.items():
            rows.append(
                {
                    "graph": row.graph,
                    "method": row.method,
                    "model": row.model,
                    "family": row.family,
                    "size_label": row.size_label,
                    "obs": row.obs,
                    "inter": row.inter,
                    "metric": metric,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def collect_model_size_ladder(cross: pd.DataFrame, graph: str) -> pd.DataFrame:
    rows = cross[cross.graph.eq(graph) & cross.family.isin(["Qwen2.5", "Qwen3", "Llama 3.1"])].copy()
    return rows.sort_values(["family", "size_value", "model"])


def collect_classical_baselines(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[df.method.isin(DATA_ONLY_METHODS)].copy()
    rows = rows[["graph", "method", "method_display", "obs", "inter", "mean_f1", "mean_shd", "path"]]
    return rows.sort_values(["graph", "obs", "inter", "method"])


def collect_selected_configs(canonical: pd.DataFrame) -> pd.DataFrame:
    rows = canonical.copy()
    rows["valid_rate"] = np.where(
        pd.to_numeric(rows["n_rows"], errors="coerce").fillna(0) > 0,
        pd.to_numeric(rows["valid_rows"], errors="coerce") / pd.to_numeric(rows["n_rows"], errors="coerce"),
        np.nan,
    )
    cols = [
        "graph",
        "method",
        "method_display",
        "obs",
        "inter",
        "condition",
        "semantic",
        "format",
        "valid_rows",
        "n_rows",
        "valid_rate",
        "mean_f1",
        "mean_shd",
        "source_kind",
        "variant",
        "path",
    ]
    return rows[[col for col in cols if col in rows.columns]].sort_values(
        ["graph", "method_display", "obs", "inter", "condition"]
    )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path}")


def latex_escape(text: object) -> str:
    s = "" if pd.isna(text) else str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


def write_latex_table(
    df: pd.DataFrame,
    path: Path,
    columns: list[str],
    headers: list[str],
    caption: str,
    label: str,
    align: str,
) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append("--" if pd.isna(val) else f"{val:.2f}")
            else:
                vals.append(latex_escape(val))
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {path}")


def make_tables(out_dir: Path, ladder: pd.DataFrame, size_ladder: pd.DataFrame, classical: pd.DataFrame, graph: str) -> None:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    headline = ladder.copy()
    headline = headline[
        [
            "model",
            "family",
            "names_only_f1",
            "real_summary_f1",
            "real_matrix_f1",
            "anon_summary_f1",
            "anon_matrix_f1",
            "best_data_only_method",
            "best_data_only_f1",
            "gap_to_data_only",
        ]
    ]
    write_latex_table(
        headline,
        table_dir / "headline_evidence_ladder.tex",
        ["model", "names_only_f1", "real_summary_f1", "real_matrix_f1", "best_data_only_method", "best_data_only_f1", "gap_to_data_only"],
        ["Model", "Names", "Summary", "Matrix", "Best classical", "Classical F1", "Gap"],
        f"Headline evidence-use ladder on {graph.capitalize()}. Values are directed-edge F1 at the selected data budget.",
        "tab:headline_evidence_ladder",
        "lrrrrrr",
    )

    size_table = size_ladder[
        ["family", "model", "names_only_f1", "real_summary_f1", "real_matrix_f1", "best_data_format"]
    ].copy()
    write_latex_table(
        size_table,
        table_dir / "model_size_ladder.tex",
        ["family", "model", "names_only_f1", "real_summary_f1", "real_matrix_f1", "best_data_format"],
        ["Family", "Model", "Names", "Summary", "Matrix", "Best format"],
        f"Within-family model-size comparison on {graph.capitalize()}.",
        "tab:model_size_ladder",
        "llrrrl",
    )

    classical_short = classical[classical.graph.eq(graph)].copy()
    if classical_short.empty:
        print(
            f"[warn] No classical baseline rows for {graph} in the selected source; "
            "skipping classical_baselines.tex"
        )
    else:
        classical_short["budget"] = classical_short.apply(lambda r: f"N={int(r.obs)}, M={int(r.inter)}", axis=1)
        write_latex_table(
            classical_short.head(30),
            table_dir / "classical_baselines.tex",
            ["method", "budget", "mean_f1", "mean_shd"],
            ["Method", "Budget", "F1", "SHD"],
            f"Classical causal discovery baselines on {graph.capitalize()}.",
            "tab:classical_baselines",
            "llrr",
        )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "responses":
        raw = load_response_summaries(
            args.responses_root,
            args.variant_contains,
            args.baseline_fallback_roots,
            args.llm_fallback_roots,
        )
        input_source = str(args.responses_root / "{graph}" / "eval_summary.csv")
    else:
        raw = load_metrics(args.metrics_csv, args.variant_contains)
        input_source = str(args.metrics_csv)

    canonical = canonicalize(raw, args.dedup_policy)
    cross = collect_cross_graph(canonical, args.obs, args.inter)
    anon_cross = collect_anonymized_cross_graph(cross)
    ladder = cross[cross.graph.eq(args.graph)].copy()
    contrasts = collect_contrast_metrics(cross, canonical, args.obs, args.inter)
    size_ladder = collect_model_size_ladder(cross, args.graph)
    classical = collect_classical_baselines(canonical)
    selected_configs = collect_selected_configs(canonical)

    write_csv(canonical, args.out_dir / "canonical_condition_results.csv")
    write_csv(selected_configs, args.out_dir / "paper_selected_configs.csv")
    write_csv(cross, args.out_dir / "paper_cross_graph_evidence_ladder.csv")
    write_csv(anon_cross, args.out_dir / "paper_cross_graph_evidence_ladder_anonymized.csv")
    write_csv(ladder, args.out_dir / "paper_headline_evidence_ladder.csv")
    write_csv(contrasts, args.out_dir / "paper_contrast_metrics.csv")
    write_csv(size_ladder, args.out_dir / "paper_model_size_ladder.csv")
    write_csv(classical, args.out_dir / "paper_classical_baselines.csv")

    make_tables(args.out_dir, ladder, size_ladder, classical, args.graph)

    manifest = pd.DataFrame(
        [
            {"key": "source", "value": args.source},
            {"key": "input", "value": input_source},
            {"key": "baseline_fallback_roots", "value": ", ".join(str(p) for p in args.baseline_fallback_roots)},
            {"key": "llm_fallback_roots", "value": ", ".join(str(p) for p in args.llm_fallback_roots)},
            {"key": "old_thinktags_preferred", "value": "false"},
            {"key": "graph", "value": args.graph},
            {"key": "obs", "value": args.obs},
            {"key": "inter", "value": args.inter},
            {"key": "dedup_policy", "value": args.dedup_policy},
            {"key": "variant_contains", "value": args.variant_contains or ""},
            {"key": "canonical_rows", "value": len(canonical)},
        ]
    )
    write_csv(manifest, args.out_dir / "paper_results_manifest.csv")


if __name__ == "__main__":
    main()
