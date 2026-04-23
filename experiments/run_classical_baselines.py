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
        return arr[:sample_size_obs], var_names

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
        from pgmpy.estimators import PC
    except Exception as exc:
        raise SystemExit("PC baseline requires `pgmpy`. Install it with `pip install pgmpy`.") from exc
    est = PC(df)
    model = est.estimate(
        variant=variant,
        ci_test=ci_test,
        return_type="dag",
        significance_level=significance_level,
        max_cond_vars=max_cond_vars,
        show_progress=False,
    )
    return _adjacency_from_edges(list(df.columns), list(model.edges()))


def _run_ges(df: pd.DataFrame, *, scoring_method: str, min_improvement: float) -> np.ndarray:
    try:
        from pgmpy.estimators import GES
    except Exception as exc:
        raise SystemExit("GES baseline requires `pgmpy`. Install it with `pip install pgmpy`.") from exc
    est = GES(df)
    model = est.estimate(scoring_method=scoring_method, min_improvement=min_improvement, debug=False)
    return _adjacency_from_edges(list(df.columns), list(model.edges()))


def _write_prediction_csv(
    *,
    out_csv: Path,
    method: str,
    graph_name: str,
    num_obs: int,
    answer: np.ndarray,
    prediction: np.ndarray,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics = eval_pair(answer, prediction)
    row = {
        "method": method,
        "graph": graph_name,
        "num_obs": num_obs,
        "num_inters": 0,
        "answer": json.dumps(answer.tolist(), ensure_ascii=False),
        "prediction": json.dumps(prediction.tolist(), ensure_ascii=False),
        **metrics,
    }
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


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
    for graph_file in args.graph_files:
        graph_path = Path(graph_file).resolve()
        graph = load_causal_graph(graph_path)
        data_obs, var_names = _load_observational_array(graph, args.sample_size_obs, args.seed)
        df = _coerce_discrete_dataframe(pd.DataFrame(data_obs, columns=var_names))

        answer = np.asarray(graph.adj_matrix).astype(int)
        np.fill_diagonal(answer, 0)

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

        dataset_name = graph_path.stem
        out_csv = Path(args.out_dir) / dataset_name / f"predictions_obs{args.sample_size_obs}_int0_{args.method}.csv"
        _write_prediction_csv(
            out_csv=out_csv,
            method=args.method,
            graph_name=dataset_name,
            num_obs=args.sample_size_obs,
            answer=answer,
            prediction=prediction,
        )
        print(f"[{args.method}] wrote {out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "responses"
