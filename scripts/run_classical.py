#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_builder.graph_io import load_causal_graph
from experiments.evaluate import eval_pair

DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "responses"


warnings.filterwarnings("ignore", category=FutureWarning, module=r"pgmpy\..*")


def _load_observational_array(graph: Any, sample_size_obs: int, seed: int) -> tuple[np.ndarray, list[str]]:
    if sample_size_obs <= 0:
        raise SystemExit("Classical observational baselines require --sample_size_obs > 0.")

    var_names = [v.name for v in graph.variables]
    if hasattr(graph, "data_obs"):
        arr = np.asarray(graph.data_obs)
        if arr.shape[0] < sample_size_obs:
            raise SystemExit(
                f"Requested sample_size_obs={sample_size_obs}, but dataset only has {arr.shape[0]} observational samples."
            )
        rng = np.random.default_rng(seed)
        idx = rng.choice(arr.shape[0], size=sample_size_obs, replace=False)
        return arr[idx], var_names

    if not hasattr(graph, "sample"):
        raise SystemExit("Graph object does not expose observational data or sampling.")

    np.random.seed(seed)
    arr = graph.sample(batch_size=sample_size_obs, as_array=True)
    return np.asarray(arr), var_names


def _adjacency_from_edges(var_names: list[str], edges: list[tuple[str, str]]) -> np.ndarray:
    name_to_idx = {name: idx for idx, name in enumerate(var_names)}
    adj = np.zeros((len(var_names), len(var_names)), dtype=int)
    for src, dst in edges:
        if src in name_to_idx and dst in name_to_idx:
            adj[name_to_idx[src], name_to_idx[dst]] = 1
    return adj


def _coerce_discrete_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # pgmpy selects discrete-vs-continuous scores from pandas dtypes. The sampled
    # ENCO benchmark graphs are categorical, but they arrive as int64 arrays, which
    # newer pgmpy versions may misclassify as continuous for GES scoring.
    discrete_df = df.copy()
    for col in discrete_df.columns:
        discrete_df[col] = discrete_df[col].astype("category")
    return discrete_df


def _run_pc(df: pd.DataFrame, *, variant: str, ci_test: str, significance_level: float, max_cond_vars: int) -> np.ndarray:
    try:
        from pgmpy.causal_discovery import PC
    except ImportError:
        try:
            from pgmpy.estimators import PC  # type: ignore[no-redef]
        except ImportError as exc:
            raise SystemExit("PC baseline requires `pgmpy`. Install it with `pip install pgmpy`.") from exc
    est = PC(
        variant=variant,
        ci_test=ci_test,
        return_type="dag",
        significance_level=significance_level,
        max_cond_vars=max_cond_vars,
        show_progress=False,
    )
    est.fit(df)
    return _adjacency_from_edges(list(df.columns), list(est.causal_graph_.edges()))


def _run_ges(df: pd.DataFrame, *, scoring_method: str, min_improvement: float) -> np.ndarray:
    try:
        from pgmpy.causal_discovery import GES
    except ImportError:
        try:
            from pgmpy.estimators import GES  # type: ignore[no-redef]
        except ImportError as exc:
            raise SystemExit("GES baseline requires `pgmpy`. Install it with `pip install pgmpy`.") from exc
    est = GES(scoring_method=scoring_method, return_type="dag", min_improvement=min_improvement)
    est.fit(df)
    return _adjacency_from_edges(list(df.columns), list(est.causal_graph_.edges()))


def _prediction_row(
    *,
    method: str,
    graph_name: str,
    num_obs: int,
    answer: np.ndarray,
    prediction: np.ndarray,
    replicate_index: int,
    replicate_seed: int,
) -> dict[str, Any]:
    metrics = eval_pair(answer, prediction)
    return {
        "method": method,
        "graph": graph_name,
        "num_obs": num_obs,
        "num_inters": 0,
        "answer": json.dumps(answer.tolist(), ensure_ascii=False),
        "prediction": json.dumps(prediction.tolist(), ensure_ascii=False),
        "replicate_index": replicate_index,
        "replicate_seed": replicate_seed,
        **metrics,
    }


def _write_prediction_rows(*, out_csv: Path, rows: list[dict[str, Any]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run representative classical causal discovery baselines.")
    parser.add_argument(
        "--method",
        required=True,
        choices=["PC", "GES"],
        help="Classical baseline to run. ENCO remains available via run_exported_graphs.py.",
    )
    parser.add_argument(
        "--graph_files",
        type=str,
        nargs="+",
        required=True,
        help="Graph files to evaluate (.bif, .pt, or .npz).",
    )
    parser.add_argument("--sample_size_obs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--pc-variant", choices=["orig", "stable", "parallel"], default="stable")
    parser.add_argument("--pc-ci-test", default="chi_square")
    parser.add_argument("--pc-significance-level", type=float, default=0.01)
    parser.add_argument("--pc-max-cond-vars", type=int, default=5)
    parser.add_argument("--ges-scoring-method", default="bic-d")
    parser.add_argument("--ges-min-improvement", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_prompts <= 0:
        raise SystemExit("--num_prompts must be > 0.")
    for graph_file in args.graph_files:
        graph_path = Path(graph_file).resolve()
        graph = load_causal_graph(graph_path)
        answer = np.asarray(graph.adj_matrix).astype(int)
        np.fill_diagonal(answer, 0)
        dataset_name = graph_path.stem
        out_csv = Path(args.out_dir) / dataset_name / f"predictions_obs{args.sample_size_obs}_int0_{args.method}_seed{int(args.seed)}.csv"
        rows: list[dict[str, Any]] = []

        for prompt_idx in range(args.num_prompts):
            seed_i = int(args.seed) + prompt_idx * 1000
            data_obs, var_names = _load_observational_array(graph, args.sample_size_obs, seed_i)
            df = _coerce_discrete_dataframe(pd.DataFrame(data_obs, columns=var_names))

            if args.method == "PC":
                prediction = _run_pc(
                    df,
                    variant=args.pc_variant,
                    ci_test=args.pc_ci_test,
                    significance_level=args.pc_significance_level,
                    max_cond_vars=args.pc_max_cond_vars,
                )
            else:
                prediction = _run_ges(
                    df,
                    scoring_method=args.ges_scoring_method,
                    min_improvement=args.ges_min_improvement,
                )
            rows.append(
                _prediction_row(
                    method=args.method,
                    graph_name=dataset_name,
                    num_obs=args.sample_size_obs,
                    answer=answer,
                    prediction=prediction,
                    replicate_index=prompt_idx,
                    replicate_seed=seed_i,
                )
            )

        _write_prediction_rows(out_csv=out_csv, rows=rows)
        print(f"[{args.method}] wrote {out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
