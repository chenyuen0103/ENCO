#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from benchmark_builder.graph_io import load_causal_graph


def _require_openai():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise SystemExit(f"TakayamaSCP requires the OpenAI Python SDK: {exc}") from exc
    api_key = os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise SystemExit("TakayamaSCP requires OPENAI_API_KEY for OpenAI chat-completions logprobs.")
    return OpenAI(api_key=api_key)


def _require_causallearn():
    try:
        from causallearn.graph.GraphNode import GraphNode  # noqa: F401
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    except Exception as exc:
        raise SystemExit(
            "TakayamaSCP requires `causal-learn`. Install it with `pip install causal-learn`."
        ) from exc
    return pc, BackgroundKnowledge


def _load_observational_dataframe(graph_path: Path, sample_size_obs: int, seed: int):
    if sample_size_obs <= 0:
        raise SystemExit("TakayamaSCP requires --sample_size_obs > 0.")
    graph = load_causal_graph(graph_path)
    var_names = [str(v.name) for v in graph.variables]

    if hasattr(graph, "data_obs"):
        arr = np.asarray(graph.data_obs)
        if arr.shape[0] < sample_size_obs:
            raise SystemExit(
                f"Requested sample_size_obs={sample_size_obs}, but graph only exposes {arr.shape[0]} observational rows."
            )
        arr = arr[:sample_size_obs]
    elif hasattr(graph, "sample"):
        np.random.seed(seed)
        arr = np.asarray(graph.sample(batch_size=sample_size_obs, as_array=True))
    else:
        raise SystemExit("Graph object does not expose observational data or sampling for TakayamaSCP.")

    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    scaler = StandardScaler()
    x = scaler.fit_transform(arr)
    df = pd.DataFrame(x, columns=var_names)
    answer = np.asarray(graph.adj_matrix).astype(int)
    np.fill_diagonal(answer, 0)
    return df, var_names, answer


def _pc_signed_adjacency(cg: Any) -> np.ndarray:
    num_nodes = len(cg.G.nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if cg.G.graph[i][j] == 1 and cg.G.graph[j][i] == -1:
                adj_matrix[i, j] = 1  # j -> i
            elif cg.G.graph[i][j] == -1 and cg.G.graph[j][i] == -1:
                adj_matrix[i, j] = -1  # undirected
            elif cg.G.graph[i][j] == 1 and cg.G.graph[j][i] == 1:
                adj_matrix[i, j] = 2  # bidirected
    return adj_matrix


def _bootstrap_pc_probabilities(df: Any, n_sampling: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.utils import resample

    pc, _BackgroundKnowledge = _require_causallearn()
    num_nodes = df.shape[1]
    directed = np.zeros((num_nodes, num_nodes), dtype=float)
    undirected = np.zeros((num_nodes, num_nodes), dtype=float)

    for _ in range(n_sampling):
        boot = resample(df)
        cg = pc(boot.to_numpy(), independence_test_method="fisherz", verbose=False, show_progress=False)
        mat = _pc_signed_adjacency(cg)
        directed += (mat == 1).astype(float)
        undirected += (mat == -1).astype(float)

    directed /= float(n_sampling)
    undirected /= float(n_sampling)
    return directed, undirected


def _system_intro(dataset_name: str, labels: list[str], *, anonymized: bool) -> str:
    variables = ", ".join(labels[:-1]) + f", and {labels[-1]}" if len(labels) > 1 else labels[0]
    if anonymized:
        return f"We want to carry out causal inference in an anonymized system, considering {variables} as variables."
    return f"We want to carry out causal inference in the system called '{dataset_name}', considering {variables} as variables."


def _first_template_text(src: str, dst: str) -> str:
    return (
        "Then, your task is to interpret this result from a domain knowledge perspective and determine whether "
        "this statistically suggested hypothesis is plausible in the context of the domain.\n"
        f"Please provide an explanation that leverages your expert knowledge on the causal relationship between {src} and {dst}, "
        "and assess the naturalness of this causal discovery result.\n"
        "Your response should consider the relevant factors and provide a reasoned explanation based on your understanding of the domain."
    )


def _all_edges_pattern1_pc(adj_matrix: np.ndarray, labels: list[str]) -> str:
    lines = ["All of the directed edges suggested by the statistic causal discovery are below:", "-----"]
    n = adj_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if j != i and adj_matrix[i, j] == 1:
                lines.append(f"{labels[j]} → {labels[i]}")
    lines.extend(["-----", "In additon to the directed edges above, all of the undirected edges suggested by the statistic causal discovery are below:", "-----"])
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j] == -1:
                lines.append(f"{labels[j]} － {labels[i]}")
    lines.extend(["-----"])
    return "\n".join(lines)


def _all_edges_pattern2_pc(directed: np.ndarray, undirected: np.ndarray, labels: list[str]) -> str:
    lines = ["All of the directed edges with non-zero bootstrap probabilities suggested by the statistic causal discovery are below:", "-----"]
    n = directed.shape[0]
    for i in range(n):
        for j in range(n):
            if j != i and directed[i, j] != 0:
                lines.append(f"{labels[j]} → {labels[i]} (bootstrap probability = {directed[i, j]})")
    lines.extend(["-----", "In additon to the directed edges above, all of the undirected edges suggested by the statistic causal discovery are below:", "-----"])
    for i in range(n):
        for j in range(i + 1, n):
            if undirected[i, j] != 0:
                lines.append(f"{labels[j]} ― {labels[i]} (bootstrap probability = {undirected[i, j]})")
    lines.extend(["-----"])
    return "\n".join(lines)


def _causal_text_pattern1_pc(adj_matrix: np.ndarray, i: int, j: int, labels: list[str]) -> str:
    if adj_matrix[i, j] == 0:
        return f"there may be no direct impact of a change in {labels[j]} on {labels[i]}."
    if adj_matrix[i, j] == 1:
        return f"there may be a direct impact of a change in {labels[j]} on {labels[i]}."
    return f"there may be a direct causal relationship between {labels[j]} and {labels[i]}, although the direction has not been determined."


def _causal_text_pattern2_pc(directed: np.ndarray, undirected: np.ndarray, i: int, j: int, labels: list[str]) -> str:
    d = float(directed[i, j])
    u = float(undirected[i, j])
    if d == 0 and u == 0:
        return f"there may be no direct impact of a change in {labels[j]} on {labels[i]}."
    if d != 0 and u == 0:
        return f"there may be a direct impact of a change in {labels[j]} on {labels[i]} with a bootstrap probability of {d}."
    if d == 0 and u != 0:
        return (
            f"there may be a direct causal relationship between {labels[j]} and {labels[i]} "
            f"with a bootstrap probability of {u}, although the direction has not been determined."
        )
    return (
        f"there may be a direct impact of a change in {labels[j]} on {labels[i]} with a bootstrap probability of {d}. "
        f"In addition, it has also been shown above that there may be a direct causal relationship between {labels[j]} and {labels[i]} "
        f"with a bootstrap probability of {u}, although the direction has not completely been determined."
    )


def _build_first_prompt(
    *,
    dataset_name: str,
    labels: list[str],
    i: int,
    j: int,
    pattern: int,
    pc_adj_signed: np.ndarray,
    pc_directed_prob: np.ndarray,
    pc_undirected_prob: np.ndarray,
    anonymized: bool,
) -> str:
    q1_1 = _system_intro(dataset_name, labels, anonymized=anonymized)
    q1_2 = "First, we have conducted the statistical causal discovery with PC(Peter-Clerk) algorithm, using a fully standardized dataset."
    q1_3 = "According to the results shown above, it has been determined that"
    if pattern == 1:
        all_edges = _all_edges_pattern1_pc(pc_adj_signed, labels)
        causal_text = _causal_text_pattern1_pc(pc_adj_signed, i, j, labels)
    elif pattern == 2:
        all_edges = _all_edges_pattern2_pc(pc_directed_prob, pc_undirected_prob, labels)
        causal_text = _causal_text_pattern2_pc(pc_directed_prob, pc_undirected_prob, i, j, labels)
    else:
        raise SystemExit(f"Unsupported Takayama PC pattern: {pattern}. Use 1 or 2.")
    template = _first_template_text(labels[j], labels[i])
    return f"{q1_1}\n{q1_2}\n{all_edges}\n{q1_3} {causal_text}\n{template}"


def _build_second_prompt(first_prompt: str, first_answer: str, *, src: str, dst: str) -> str:
    return (
        "An expert was asked the question below:\n"
        f"{first_prompt}\n"
        "Then, the expert replied with its domain knowledge:\n"
        f"{first_answer}\n"
        "Considering objectively this discussion above,"
        f"if {src} is modified, will it have a direct or indirect impact on {dst}?\n"
        "Please answer this question with <yes> or <no>.\n"
        "No answers except these two responses are needed."
    )


def _extract_yes_no_prob(response: Any) -> tuple[float, float, str]:
    try:
        content_items = response.choices[0].logprobs.content
    except Exception:
        content_items = None
    prob_yes = 0.0
    prob_no = 0.0
    if content_items:
        first = content_items[0]
        for item in getattr(first, "top_logprobs", []) or []:
            tok = str(getattr(item, "token", "")).strip().lower()
            lp = float(getattr(item, "logprob", float("-inf")))
            if tok == "yes":
                prob_yes += math.exp(lp)
            elif tok == "no":
                prob_no += math.exp(lp)
    text = ""
    try:
        text = response.choices[0].message.content or ""
    except Exception:
        pass
    if prob_yes == 0.0 and prob_no == 0.0:
        low = text.strip().lower()
        if low.startswith("<yes>") or low.startswith("yes"):
            prob_yes = 1.0
        elif low.startswith("<no>") or low.startswith("no"):
            prob_no = 1.0
    return prob_yes, prob_no, text


def _llm_probability_matrix(
    *,
    client: Any,
    model_name: str,
    labels: list[str],
    dataset_name: str,
    pattern: int,
    pc_adj_signed: np.ndarray,
    pc_directed_prob: np.ndarray,
    pc_undirected_prob: np.ndarray,
    temperature: float,
    max_new_tokens: int | None,
    num_samples: int,
    anonymized: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n = len(labels)
    mean_matrix = np.zeros((n, n), dtype=float)
    transcript: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            first_prompt = _build_first_prompt(
                dataset_name=dataset_name,
                labels=labels,
                i=i,
                j=j,
                pattern=pattern,
                pc_adj_signed=pc_adj_signed,
                pc_directed_prob=pc_directed_prob,
                pc_undirected_prob=pc_undirected_prob,
                anonymized=anonymized,
            )
            first_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for causal inference."},
                    {"role": "user", "content": first_prompt},
                ],
                temperature=temperature,
                max_tokens=max_new_tokens or 3000,
            )
            first_answer = first_response.choices[0].message.content or ""
            second_prompt = _build_second_prompt(first_prompt, first_answer, src=labels[j], dst=labels[i])
            probs: list[float] = []
            sample_logs: list[dict[str, Any]] = []
            for trial in range(max(1, num_samples)):
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for causal inference."},
                        {"role": "user", "content": second_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_new_tokens or 1500,
                    logprobs=True,
                    top_logprobs=5,
                )
                prob_yes, prob_no, raw_text = _extract_yes_no_prob(response)
                probs.append(prob_yes)
                sample_logs.append(
                    {
                        "trial": trial + 1,
                        "prob_yes": prob_yes,
                        "prob_no": prob_no,
                        "response": raw_text,
                    }
                )
            mean_matrix[i, j] = float(np.mean(probs)) if probs else 0.0
            transcript.append(
                {
                    "effect_idx": i,
                    "cause_idx": j,
                    "effect": labels[i],
                    "cause": labels[j],
                    "first_prompt": first_prompt,
                    "first_answer": first_answer,
                    "second_prompt": second_prompt,
                    "samples": sample_logs,
                    "mean_prob_yes": mean_matrix[i, j],
                }
            )
    return mean_matrix, transcript


def _probability_to_pk(probability: np.ndarray) -> np.ndarray:
    pk = np.empty(probability.shape, dtype=int)
    n = probability.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                pk[i, j] = -1
            else:
                if probability[i, j] < 0.05:
                    pk[i, j] = 0
                elif probability[i, j] > 0.95:
                    pk[i, j] = 1
                else:
                    pk[i, j] = -1
    return pk


def _pk_to_background_knowledge(pk_matrix: np.ndarray, df: Any) -> Any:
    pc, BackgroundKnowledge = _require_causallearn()
    cg = pc(df.to_numpy(), independence_test_method="fisherz", verbose=False, show_progress=False)
    nodes = cg.G.get_nodes()
    bk = BackgroundKnowledge()
    n = pk_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if pk_matrix[i, j] == 1:
                bk.add_required_by_node(nodes[j], nodes[i])
            elif pk_matrix[i, j] == 0:
                bk.add_forbidden_by_node(nodes[j], nodes[i])
    return bk


def _run_pc_with_background_knowledge(df: Any, pk_matrix: Optional[np.ndarray]) -> np.ndarray:
    pc, _BackgroundKnowledge = _require_causallearn()
    kwargs: dict[str, Any] = {
        "independence_test_method": "fisherz",
        "verbose": False,
        "show_progress": False,
    }
    if pk_matrix is not None:
        kwargs["background_knowledge"] = _pk_to_background_knowledge(pk_matrix, df)
    cg = pc(df.to_numpy(), **kwargs)
    return _pc_signed_adjacency(cg)


def _directed_prediction_from_pc_signed(pc_signed: np.ndarray) -> np.ndarray:
    return (pc_signed == 1).astype(int).T


def _write_prediction_csv(
    *,
    out_csv: Path,
    model_name: str,
    provider: str,
    naming_regime: str,
    sample_size_obs: int,
    answer: np.ndarray,
    prediction: np.ndarray,
    transcript: list[dict[str, Any]],
    probability_matrix: np.ndarray,
    pk_matrix: np.ndarray,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "model",
                "provider",
                "naming_regime",
                "obs_n",
                "int_n",
                "raw_response",
                "answer",
                "prediction",
                "valid",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "TakayamaSCP",
                "model": model_name,
                "provider": provider,
                "naming_regime": naming_regime,
                "obs_n": sample_size_obs,
                "int_n": 0,
                "raw_response": json.dumps(
                    {
                        "probability_matrix_effect_by_cause": probability_matrix.tolist(),
                        "pk_matrix_effect_by_cause": pk_matrix.tolist(),
                        "transcript": transcript,
                    },
                    ensure_ascii=False,
                ),
                "answer": json.dumps(np.asarray(answer, dtype=int).tolist(), ensure_ascii=False),
                "prediction": json.dumps(np.asarray(prediction, dtype=int).tolist(), ensure_ascii=False),
                "valid": 1,
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Faithful PC-based Takayama SCP baseline runner.")
    parser.add_argument("--graph_files", type=str, nargs="+", required=True)
    parser.add_argument("--sample_size_obs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="responses")
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--provider", type=str, default="auto")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--bootstrap_samples", type=int, default=100)
    parser.add_argument("--takayama_pattern", type=int, default=2)
    parser.add_argument("--naming_regime", choices=["real", "anonymized"], default="real")
    args = parser.parse_args()

    provider = args.provider
    if provider == "auto":
        provider = "openai" if "gpt" in args.model.lower() or "o3-" in args.model.lower() or "o1-" in args.model.lower() else provider
    if provider != "openai":
        raise SystemExit("Faithful TakayamaSCP currently supports only OpenAI chat-completions with logprobs.")
    client = _require_openai()

    for graph_file in args.graph_files:
        graph_path = Path(graph_file).resolve()
        df, var_names, answer = _load_observational_dataframe(graph_path, args.sample_size_obs, args.seed)
        pc_signed = _run_pc_with_background_knowledge(df, pk_matrix=None)
        directed_prob, undirected_prob = _bootstrap_pc_probabilities(df, args.bootstrap_samples)

        if args.naming_regime == "anonymized":
            labels = [f"X{i+1}" for i in range(len(var_names))]
        else:
            labels = list(var_names)

        probability_matrix, transcript = _llm_probability_matrix(
            client=client,
            model_name=args.model,
            labels=labels,
            dataset_name=graph_path.stem,
            pattern=args.takayama_pattern,
            pc_adj_signed=pc_signed,
            pc_directed_prob=directed_prob,
            pc_undirected_prob=undirected_prob,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            num_samples=args.num_samples,
            anonymized=args.naming_regime == "anonymized",
        )
        pk_matrix = _probability_to_pk(probability_matrix)
        constrained_pc_signed = _run_pc_with_background_knowledge(df, pk_matrix=pk_matrix)
        prediction = _directed_prediction_from_pc_signed(constrained_pc_signed)

        naming_suffix = "_anon" if args.naming_regime == "anonymized" else ""
        out_csv = Path(args.out_dir) / graph_path.stem / (
            f"predictions_obs{args.sample_size_obs}_int0_TakayamaSCP{naming_suffix}.csv"
        )
        _write_prediction_csv(
            out_csv=out_csv,
            model_name=args.model,
            provider=provider,
            naming_regime=args.naming_regime,
            sample_size_obs=args.sample_size_obs,
            answer=answer,
            prediction=prediction,
            transcript=transcript,
            probability_matrix=probability_matrix,
            pk_matrix=pk_matrix,
        )
        print(f"[TakayamaSCP] wrote {out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
