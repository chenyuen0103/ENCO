#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
for _path in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from benchmark_builder.graph_io import load_causal_graph

DEFAULT_OUT_DIR = EXPERIMENTS_DIR / "responses"
TAKAYAMA_BACKENDS = {"pc", "exact_search", "direct_lingam"}
DEFAULT_FIRST_STAGE_MAX_NEW_TOKENS = 384
DEFAULT_SECOND_STAGE_MAX_NEW_TOKENS = 8


def _progress(message: str) -> None:
    print(f"[TakayamaSCP {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _normalize_backend(raw: str) -> str:
    backend = (raw or "pc").strip().lower()
    aliases = {
        "pc": "pc",
        "exact": "exact_search",
        "exactsearch": "exact_search",
        "exact_search": "exact_search",
        "es": "exact_search",
        "lingam": "direct_lingam",
        "directlingam": "direct_lingam",
        "direct_lingam": "direct_lingam",
    }
    backend = aliases.get(backend, backend)
    if backend not in TAKAYAMA_BACKENDS:
        raise SystemExit(f"Unsupported Takayama backend: {raw}")
    return backend


def _backend_suffix(backend: str) -> str:
    if backend == "pc":
        return ""
    if backend == "exact_search":
        return "_ExactSearch"
    if backend == "direct_lingam":
        return "_DirectLiNGAM"
    return f"_{backend}"


def _require_openai():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise SystemExit(f"TakayamaSCP requires the OpenAI Python SDK: {exc}") from exc
    api_key = os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise SystemExit("TakayamaSCP requires OPENAI_API_KEY for OpenAI chat-completions logprobs.")
    return OpenAI(api_key=api_key)


def _require_illinois():
    api_key = os.getenv("ILLINOIS_CHAT_API") or ""
    if not api_key:
        raise SystemExit("TakayamaSCP with provider=illinois requires ILLINOIS_CHAT_API.")
    return {
        "url": "https://chat.illinois.edu/api/chat-api/chat",
        "api_key": api_key,
        "course_name": os.getenv("ILLINOIS_CHAT_COURSE", "llm_cd"),
    }


def _require_causallearn():
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    except Exception as exc:
        raise SystemExit(
            "TakayamaSCP requires `causal-learn`. Install it with `pip install causal-learn`."
        ) from exc
    return pc, bic_exact_search, BackgroundKnowledge


def _require_lingam():
    try:
        import lingam  # type: ignore
    except Exception as exc:
        raise SystemExit("TakayamaSCP DirectLiNGAM backend requires `lingam`.") from exc
    return lingam


def _load_observational_frames(
    graph_path: Path,
    sample_size_obs: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray]:
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

    raw_df = pd.DataFrame(arr, columns=var_names)
    scaler = StandardScaler()
    x = scaler.fit_transform(arr)
    std_df = pd.DataFrame(x, columns=var_names)

    answer = np.asarray(graph.adj_matrix).astype(int)
    np.fill_diagonal(answer, 0)
    return raw_df, std_df, var_names, answer


def _pc_signed_adjacency(cg: Any) -> np.ndarray:
    num_nodes = len(cg.G.nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if cg.G.graph[i][j] == 1 and cg.G.graph[j][i] == -1:
                adj_matrix[i, j] = 1
            elif cg.G.graph[i][j] == -1 and cg.G.graph[j][i] == -1:
                adj_matrix[i, j] = -1
            elif cg.G.graph[i][j] == 1 and cg.G.graph[j][i] == 1:
                adj_matrix[i, j] = 2
    return adj_matrix


def _bootstrap_pc_probabilities(df: pd.DataFrame, n_sampling: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.utils import resample

    pc, _bic_exact_search, _BackgroundKnowledge = _require_causallearn()
    num_nodes = df.shape[1]
    directed = np.zeros((num_nodes, num_nodes), dtype=float)
    undirected = np.zeros((num_nodes, num_nodes), dtype=float)

    _progress(f"bootstrap PC start: samples={n_sampling}, vars={num_nodes}")
    report_every = max(1, n_sampling // 5)
    for idx in range(n_sampling):
        boot = resample(df)
        cg = pc(boot.to_numpy(), independence_test_method="fisherz", verbose=False, show_progress=False)
        mat = _pc_signed_adjacency(cg)
        directed += (mat == 1).astype(float)
        undirected += (mat == -1).astype(float)
        done = idx + 1
        if done == 1 or done == n_sampling or done % report_every == 0:
            _progress(f"bootstrap PC progress: {done}/{n_sampling}")

    directed /= float(n_sampling)
    undirected /= float(n_sampling)
    _progress("bootstrap PC done")
    return directed, undirected


def _run_exact_search(df: pd.DataFrame, *, super_graph: Optional[np.ndarray] = None) -> np.ndarray:
    _pc, bic_exact_search, _BackgroundKnowledge = _require_causallearn()
    dag_est, _search_stats = bic_exact_search(df.to_numpy(), super_graph=super_graph, verbose=False)
    return np.asarray(dag_est)


def _bootstrap_exact_search_probabilities(df: pd.DataFrame, n_sampling: int) -> np.ndarray:
    from sklearn.utils import resample

    num_nodes = df.shape[1]
    directed = np.zeros((num_nodes, num_nodes), dtype=float)
    _progress(f"bootstrap Exact Search start: samples={n_sampling}, vars={num_nodes}")
    report_every = max(1, n_sampling // 5)
    for idx in range(n_sampling):
        boot = resample(df)
        dag_est = _run_exact_search(boot, super_graph=None)
        directed += (np.asarray(dag_est) != 0).astype(float)
        done = idx + 1
        if done == 1 or done == n_sampling or done % report_every == 0:
            _progress(f"bootstrap Exact Search progress: {done}/{n_sampling}")
    directed /= float(n_sampling)
    _progress("bootstrap Exact Search done")
    return directed


def _run_direct_lingam(df: pd.DataFrame, *, prior_knowledge: Optional[np.ndarray] = None) -> np.ndarray:
    lingam = _require_lingam()
    model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
    model.fit(df)
    return np.asarray(model.adjacency_matrix_)


def _bootstrap_direct_lingam_probabilities(df: pd.DataFrame, n_sampling: int) -> np.ndarray:
    lingam = _require_lingam()
    _progress(f"bootstrap DirectLiNGAM start: samples={n_sampling}, vars={df.shape[1]}")
    model = lingam.DirectLiNGAM(prior_knowledge=None)
    model.fit(df)
    bootstrap = model.bootstrap(df, n_sampling=n_sampling)
    probs = np.asarray(bootstrap.get_probabilities(min_causal_effect=0.01), dtype=float)
    _progress("bootstrap DirectLiNGAM done")
    return probs


def _system_intro(dataset_name: str, labels: list[str], *, anonymized: bool) -> str:
    variables = ", ".join(labels[:-1]) + f", and {labels[-1]}" if len(labels) > 1 else labels[0]
    context = "in an anonymized system" if anonymized else f"in the system called '{dataset_name}'"
    return f"We want to carry out causal inference {context}, considering {variables} as variables."


def _dataset_explanation(dataset_name: str, *, anonymized: bool) -> str:
    return "an anonymized dataset" if anonymized else f"the {dataset_name} dataset"


def _first_template_text(src: str, dst: str) -> str:
    return (
        "Then, your task is to interpret this result from a domain knowledge perspective and determine whether "
        "this statistically suggested hypothesis is plausible in the context of the domain.\n"
        f"Please provide an explanation that leverages your expert knowledge on the causal relationship between {src} and {dst}, "
        "and assess the naturalness of this causal discovery result.\n"
        "Your response should consider the relevant factors and provide a reasoned explanation based on your understanding of the domain."
    )


def _q1_2_text(backend: str, dataset_name: str, *, anonymized: bool) -> str:
    explanation = _dataset_explanation(dataset_name, anonymized=anonymized)
    if backend == "pc":
        return "First, we have conducted the statistical causal discovery with PC(Peter-Clerk) algorithm, using a fully standardized dataset."
    if backend == "exact_search":
        return f"First, we have conducted the statistical causal discovery with Exact Search algorithm, using a fully standardized dataset on {explanation}."
    if backend == "direct_lingam":
        return f"First, we have conducted the statistical causal discovery with LiNGAM(Linear Non-Gaussian Acyclic Model) algorithm, using a fully standardized dataset on {explanation}."
    raise SystemExit(f"Unsupported backend in q1_2_text: {backend}")


def _all_edges_pattern1(adjacency_matrix: np.ndarray, labels: list[str]) -> str:
    lines = ["All of the edges suggested by the statistical causal discovery are below:", "-----"]
    n = adjacency_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if j == i or adjacency_matrix[i, j] == 0:
                continue
            lines.append(f"{labels[j]} → {labels[i]}")
    lines.extend(["-----"])
    return "\n".join(lines)


def _causal_text_pattern1(adjacency_matrix: np.ndarray, i: int, j: int, labels: list[str]) -> str:
    if adjacency_matrix[i, j] == 0:
        return f"there may be no direct impact of a change in {labels[j]} on {labels[i]}."
    return f"there may be a direct impact of a change in {labels[j]} on {labels[i]}."


def _all_edges_pattern2(boot_prob: np.ndarray, labels: list[str]) -> str:
    lines = ["All of the edges with non-zero bootstrap probabilities suggested by the statistical causal discovery are below:", "-----"]
    n = boot_prob.shape[0]
    for i in range(n):
        for j in range(n):
            if j == i or boot_prob[i, j] == 0:
                continue
            lines.append(f"{labels[j]} → {labels[i]} (bootstrap probability = {boot_prob[i, j]})")
    lines.extend(["-----"])
    return "\n".join(lines)


def _causal_text_pattern2(boot_prob: np.ndarray, i: int, j: int, labels: list[str]) -> str:
    if boot_prob[i, j] == 0:
        return f"there may be no direct impact of a change in {labels[j]} on {labels[i]}."
    return f"there may be a direct impact of a change in {labels[j]} on {labels[i]} with a bootstrap probability of {boot_prob[i, j]}."


def _all_edges_pattern3(adjacency_matrix: np.ndarray, labels: list[str]) -> str:
    lines = ["All of the edges and their coefficients of the structural causal model suggested by the statistical causal discovery are below:", "-----"]
    n = adjacency_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if j == i or adjacency_matrix[i, j] == 0:
                continue
            lines.append(f"{labels[j]} → {labels[i]} (coefficient = {adjacency_matrix[i, j]})")
    lines.extend(["-----"])
    return "\n".join(lines)


def _causal_text_pattern3(adjacency_matrix: np.ndarray, i: int, j: int, labels: list[str]) -> str:
    if adjacency_matrix[i, j] == 0:
        return f"there may be no direct impact of a change in {labels[j]} on {labels[i]}."
    return f"there may be a direct impact of a change in {labels[j]} on {labels[i]} with a causal coefficient of {adjacency_matrix[i, j]}."


def _all_edges_pattern4(adjacency_matrix: np.ndarray, boot_prob: np.ndarray, labels: list[str]) -> str:
    lines = ["All of the edges with non-zero bootstrap probabilities and their coefficients of the structural causal model suggested by the statistical causal discovery are below:", "-----"]
    n = boot_prob.shape[0]
    for i in range(n):
        for j in range(n):
            if j == i or boot_prob[i, j] == 0:
                continue
            lines.append(
                f"{labels[j]} → {labels[i]} (coefficient = {adjacency_matrix[i, j]}, bootstrap probability = {boot_prob[i, j]})"
            )
    lines.extend(["-----"])
    return "\n".join(lines)


def _causal_text_pattern4(adjacency_matrix: np.ndarray, boot_prob: np.ndarray, i: int, j: int, labels: list[str]) -> str:
    if boot_prob[i, j] == 0:
        return f"there may be no direct impact of a change in {labels[j]} on {labels[i]}."
    if adjacency_matrix[i, j] == 0:
        return (
            f"there may be a direct impact of a change in {labels[j]} on {labels[i]} "
            f"with a bootstrap probability of {boot_prob[i, j]}, but the coefficient is likely to be {adjacency_matrix[i, j]}."
        )
    return (
        f"there may be a direct impact of a change in {labels[j]} on {labels[i]} "
        f"with a bootstrap probability of {boot_prob[i, j]}, and the coefficient is likely to be {adjacency_matrix[i, j]}."
    )


def _all_edges_pattern1_pc(adjacency_matrix: np.ndarray, labels: list[str]) -> str:
    lines = ["All of the directed edges suggested by the statistic causal discovery are below:", "-----"]
    n = adjacency_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if j != i and adjacency_matrix[i, j] == 1:
                lines.append(f"{labels[j]} → {labels[i]}")
    lines.extend(["-----", "In additon to the directed edges above, all of the undirected edges suggested by the statistic causal discovery are below:", "-----"])
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i, j] == -1:
                lines.append(f"{labels[j]} － {labels[i]}")
    lines.extend(["-----"])
    return "\n".join(lines)


def _causal_text_pattern1_pc(adjacency_matrix: np.ndarray, i: int, j: int, labels: list[str]) -> str:
    if adjacency_matrix[i, j] == 1:
        return f"there may be a direct impact of a change in {labels[j]} on {labels[i]}."
    if adjacency_matrix[i, j] == -1 or adjacency_matrix[j, i] == -1:
        return f"there may be a direct causal relationship between {labels[j]} and {labels[i]}, although the direction has not been determined."
    return f"there may be no direct impact of a change in {labels[j]} on {labels[i]}."


def _all_edges_pattern2_pc(boot_prob_directed: np.ndarray, boot_prob_undirected: np.ndarray, labels: list[str]) -> str:
    lines = ["All of the directed edges with non-zero bootstrap probabilities suggested by the statistic causal discovery are below:", "-----"]
    n = boot_prob_directed.shape[0]
    for i in range(n):
        for j in range(n):
            if j == i or boot_prob_directed[i, j] == 0:
                continue
            lines.append(f"{labels[j]} → {labels[i]} (bootstrap probability = {boot_prob_directed[i, j]})")
    lines.extend(["-----", "In additon to the directed edges above, all of the undirected edges suggested by the statistic causal discovery are below:", "-----"])
    for i in range(n):
        for j in range(i + 1, n):
            if boot_prob_undirected[i, j] != 0:
                lines.append(f"{labels[j]} ― {labels[i]} (bootstrap probability = {boot_prob_undirected[i, j]})")
    lines.extend(["-----"])
    return "\n".join(lines)


def _causal_text_pattern2_pc(
    boot_prob_directed: np.ndarray,
    boot_prob_undirected: np.ndarray,
    i: int,
    j: int,
    labels: list[str],
) -> str:
    d = float(boot_prob_directed[i, j])
    u = float(boot_prob_undirected[i, j])
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
    backend: str,
    pattern: int,
    dataset_name: str,
    labels: list[str],
    i: int,
    j: int,
    adjacency_matrix: np.ndarray,
    primary_prob: Optional[np.ndarray],
    secondary_prob: Optional[np.ndarray],
    anonymized: bool,
) -> str:
    q1_1 = _system_intro(dataset_name, labels, anonymized=anonymized)
    q1_2 = _q1_2_text(backend, dataset_name, anonymized=anonymized)
    q1_3 = "According to the results shown above, it has been determined that"
    template = _first_template_text(labels[j], labels[i])

    if backend == "pc":
        if pattern == 1:
            all_edges = _all_edges_pattern1_pc(adjacency_matrix, labels)
            causal_text = _causal_text_pattern1_pc(adjacency_matrix, i, j, labels)
        elif pattern == 2:
            if primary_prob is None or secondary_prob is None:
                raise SystemExit("Takayama PC pattern 2 requires directed and undirected bootstrap probabilities.")
            all_edges = _all_edges_pattern2_pc(primary_prob, secondary_prob, labels)
            causal_text = _causal_text_pattern2_pc(primary_prob, secondary_prob, i, j, labels)
        else:
            raise SystemExit("Takayama PC backend supports patterns 1 and 2.")
        return f"{q1_1}\n{q1_2}\n{all_edges}\n{q1_3} {causal_text}\n{template}"

    if backend == "exact_search":
        if pattern == 1:
            all_edges = _all_edges_pattern1(adjacency_matrix, labels)
            causal_text = _causal_text_pattern1(adjacency_matrix, i, j, labels)
        elif pattern == 2:
            if primary_prob is None:
                raise SystemExit("Takayama Exact Search pattern 2 requires bootstrap probabilities.")
            # Mirror the upstream notebook's pattern-2 template selection.
            all_edges = _all_edges_pattern3(primary_prob, labels)
            causal_text = _causal_text_pattern2(primary_prob, i, j, labels)
        else:
            raise SystemExit("Takayama Exact Search backend supports patterns 1 and 2.")
        return f"{q1_1}\n{q1_2}\n{all_edges}\n{q1_3} {causal_text}\n{template}"

    if backend == "direct_lingam":
        if pattern == 0:
            return (
                f"{q1_1}\n"
                f"If {labels[j]} is modified, will it have a direct impact on {labels[i]}?\n"
                f"Please provide an explanation that leverages your expert knowledge on the causal relationship between {labels[j]} and {labels[i]}.\n"
                "Your response should consider the relevant factors and provide a reasoned explanation based on your understanding of the domain."
            )
        if pattern == 1:
            all_edges = _all_edges_pattern1(adjacency_matrix, labels)
            causal_text = _causal_text_pattern1(adjacency_matrix, i, j, labels)
        elif pattern == 2:
            if primary_prob is None:
                raise SystemExit("Takayama DirectLiNGAM pattern 2 requires bootstrap probabilities.")
            # Mirror the upstream notebook's pattern-2 template selection.
            all_edges = _all_edges_pattern3(primary_prob, labels)
            causal_text = _causal_text_pattern3(primary_prob, i, j, labels)
        elif pattern == 3:
            all_edges = _all_edges_pattern2(adjacency_matrix, labels)
            causal_text = _causal_text_pattern2(adjacency_matrix, i, j, labels)
        elif pattern == 4:
            if primary_prob is None:
                raise SystemExit("Takayama DirectLiNGAM pattern 4 requires bootstrap probabilities.")
            all_edges = _all_edges_pattern4(adjacency_matrix, primary_prob, labels)
            causal_text = _causal_text_pattern4(adjacency_matrix, primary_prob, i, j, labels)
        else:
            raise SystemExit("Takayama DirectLiNGAM backend supports patterns 0, 1, 2, 3, and 4.")
        return f"{q1_1}\n{q1_2}\n{all_edges}\n{q1_3} {causal_text}\n{template}"

    raise SystemExit(f"Unsupported backend in first prompt builder: {backend}")


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


def _yes_no_label_from_text(text: str) -> Optional[str]:
    normalized = (text or "").strip().lower()
    normalized = normalized.removeprefix("<").strip()
    normalized = normalized.removesuffix(">").strip()
    normalized = normalized.strip(" .,:;!?\"'")
    if normalized.startswith("yes"):
        return "yes"
    if normalized.startswith("no"):
        return "no"
    return None


def _extract_yes_no_prob(response: Any) -> tuple[float, float, str]:
    text = ""
    try:
        text = response.choices[0].message.content or ""
    except Exception:
        pass
    text_label = _yes_no_label_from_text(text)
    if text_label == "yes":
        return 1.0, 0.0, text
    if text_label == "no":
        return 0.0, 1.0, text

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
            tok = tok.removeprefix("<").removesuffix(">").strip(" .,:;!?\"'")
            lp = float(getattr(item, "logprob", float("-inf")))
            if tok == "yes":
                prob_yes += math.exp(lp)
            elif tok == "no":
                prob_no += math.exp(lp)
    return prob_yes, prob_no, text


def _fmt_prob(value: float) -> str:
    return f"{float(value):.12e}"


def _resolve_stage_token_limits(
    *,
    shared_max_new_tokens: int | None,
    max_new_tokens_first: int | None,
    max_new_tokens_second: int | None,
) -> tuple[int, int]:
    if shared_max_new_tokens is not None:
        first = int(max_new_tokens_first) if max_new_tokens_first is not None else int(shared_max_new_tokens)
        second = int(max_new_tokens_second) if max_new_tokens_second is not None else int(shared_max_new_tokens)
        return first, second
    return (
        int(max_new_tokens_first) if max_new_tokens_first is not None else DEFAULT_FIRST_STAGE_MAX_NEW_TOKENS,
        int(max_new_tokens_second) if max_new_tokens_second is not None else DEFAULT_SECOND_STAGE_MAX_NEW_TOKENS,
    )


def _extract_text_from_illinois_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("content", "response", "text", "message", "output"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            choice0 = choices[0]
            if isinstance(choice0, dict):
                message = choice0.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                content = choice0.get("content")
                if isinstance(content, str):
                    return content
        data = payload.get("data")
        if data is not None:
            return _extract_text_from_illinois_payload(data)
    if isinstance(payload, list) and payload:
        for item in payload:
            text = _extract_text_from_illinois_payload(item)
            if text:
                return text
    return ""


def _chat_text_completion(
    *,
    provider: str,
    client: Any,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_new_tokens: int | None,
    logprobs: bool = False,
    top_logprobs: int = 20,
) -> dict[str, Any]:
    if provider == "openai":
        completion_tokens = max_new_tokens or (1500 if logprobs else 3000)
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "logprobs": logprobs if logprobs else None,
            "top_logprobs": top_logprobs if logprobs else None,
        }
        if "gpt-5" in model_name.lower():
            kwargs["max_completion_tokens"] = completion_tokens
        else:
            kwargs["max_tokens"] = completion_tokens
        response = client.chat.completions.create(**kwargs)
        text = ""
        try:
            text = response.choices[0].message.content or ""
        except Exception:
            pass
        return {"raw": response, "text": text}
    if provider == "illinois":
        payload = {
            "model": model_name,
            "messages": messages,
            "api_key": client["api_key"],
            "course_name": client["course_name"],
            "temperature": temperature,
            "stream": False,
        }
        max_attempts = 6
        retry_statuses = {429, 500, 502, 503, 504}
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(client["url"], json=payload, timeout=180)
                if response.status_code in retry_statuses:
                    raise requests.HTTPError(
                        f"{response.status_code} Server Error: {response.reason}",
                        response=response,
                    )
                response.raise_for_status()
                raw = response.json()
                return {"raw": raw, "text": _extract_text_from_illinois_payload(raw)}
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                should_retry = (
                    isinstance(exc, (requests.Timeout, requests.ConnectionError))
                    or status_code in retry_statuses
                )
                if not should_retry or attempt == max_attempts:
                    raise
                sleep_s = min(60.0, 2.0 ** (attempt - 1))
                _progress(
                    "Illinois API transient failure "
                    f"(attempt {attempt}/{max_attempts}, status={status_code or 'network'}); "
                    f"retrying in {sleep_s:.1f}s"
                )
                time.sleep(sleep_s)
    raise SystemExit(f"Unsupported provider for TakayamaSCP: {provider}")


def _extract_yes_no_prob_from_text(text: str) -> tuple[float, float]:
    label = _yes_no_label_from_text(text)
    if label == "yes":
        return 1.0, 0.0
    if label == "no":
        return 0.0, 1.0
    return 0.0, 0.0


def _checkpoint_path_for_output(out_csv: Path) -> Path:
    return out_csv.with_suffix(out_csv.suffix + ".checkpoint.json")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _build_run_signature(
    *,
    graph_path: Path,
    backend: str,
    model_name: str,
    provider: str,
    labels: list[str],
    sample_size_obs: int,
    naming_regime: str,
    temperature: float,
    max_new_tokens: int | None,
    max_new_tokens_first: int,
    max_new_tokens_second: int,
    num_samples: int,
    bootstrap_samples: int,
    pattern: int,
) -> dict[str, Any]:
    return {
        "graph_file": str(graph_path),
        "graph_stem": graph_path.stem,
        "backend": backend,
        "model": model_name,
        "provider": provider,
        "labels": list(labels),
        "sample_size_obs": int(sample_size_obs),
        "naming_regime": naming_regime,
        "temperature": float(temperature),
        "max_new_tokens": max_new_tokens,
        "max_new_tokens_first": int(max_new_tokens_first),
        "max_new_tokens_second": int(max_new_tokens_second),
        "num_samples": int(num_samples),
        "bootstrap_samples": int(bootstrap_samples),
        "takayama_pattern": int(pattern),
    }


def _load_checkpoint_payload(checkpoint_path: Path, run_signature: dict[str, Any]) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {
            "version": 1,
            "run_signature": run_signature,
            "completed_pairs": [],
            "current_pair": None,
        }
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise SystemExit(f"Unsupported checkpoint version in {checkpoint_path}. Delete it and rerun.")
    saved_signature = payload.get("run_signature")
    if saved_signature != run_signature:
        raise SystemExit(
            f"Checkpoint {checkpoint_path} does not match this run configuration. "
            "Delete it and rerun, or use a different output path."
        )
    payload.setdefault("completed_pairs", [])
    payload.setdefault("current_pair", None)
    return payload


def _checkpoint_snapshot(
    *,
    run_signature: dict[str, Any],
    completed_pairs: list[dict[str, Any]],
    current_pair: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "run_signature": run_signature,
        "completed_pairs": completed_pairs,
        "current_pair": current_pair,
    }


def _restore_checkpoint_state(
    checkpoint_payload: dict[str, Any],
    n: int,
) -> tuple[np.ndarray, list[dict[str, Any]], set[tuple[int, int]], dict[str, Any] | None]:
    mean_matrix = np.zeros((n, n), dtype=float)
    transcript: list[dict[str, Any]] = []
    completed_pairs: set[tuple[int, int]] = set()
    for saved in checkpoint_payload.get("completed_pairs", []):
        entry = dict(saved)
        entry["samples"] = [dict(sample) for sample in saved.get("samples", [])]
        i = int(entry["effect_idx"])
        j = int(entry["cause_idx"])
        if not (0 <= i < n and 0 <= j < n) or i == j:
            raise SystemExit("Checkpoint contains an invalid completed pair entry.")
        mean_prob = float(entry.get("mean_prob_yes", 0.0))
        entry["mean_prob_yes"] = mean_prob
        mean_matrix[i, j] = mean_prob
        transcript.append(entry)
        completed_pairs.add((i, j))
    current_pair = checkpoint_payload.get("current_pair")
    if current_pair is not None:
        current_pair = dict(current_pair)
        current_pair["samples"] = [dict(sample) for sample in current_pair.get("samples", [])]
    return mean_matrix, transcript, completed_pairs, current_pair


def _llm_probability_matrix(
    *,
    provider: str,
    client: Any,
    model_name: str,
    backend: str,
    labels: list[str],
    dataset_name: str,
    pattern: int,
    adjacency_matrix: np.ndarray,
    primary_prob: Optional[np.ndarray],
    secondary_prob: Optional[np.ndarray],
    temperature: float,
    max_new_tokens_first: int,
    max_new_tokens_second: int,
    num_samples: int,
    anonymized: bool,
    checkpoint_path: Path,
    run_signature: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n = len(labels)
    total_pairs = n * (n - 1)
    checkpoint_payload = _load_checkpoint_payload(checkpoint_path, run_signature)
    mean_matrix, transcript, completed_pairs, current_pair = _restore_checkpoint_state(checkpoint_payload, n)
    if completed_pairs or current_pair is not None:
        current_desc = ""
        if current_pair is not None:
            current_desc = (
                f" current_pair=({current_pair.get('cause_idx')}->{current_pair.get('effect_idx')})"
                f" samples={len(current_pair.get('samples', []))}"
            )
        print(f"[TakayamaSCP] resuming from {checkpoint_path} completed_pairs={len(completed_pairs)}{current_desc}")
    _progress(
        f"LLM pairwise phase start: backend={backend}, dataset={dataset_name}, vars={n}, total_pairs={total_pairs}, "
        f"completed_pairs={len(completed_pairs)}, samples_per_pair={max(1, num_samples)}"
    )

    def save_checkpoint(*, current_pair_state: dict[str, Any] | None) -> None:
        _atomic_write_json(
            checkpoint_path,
            _checkpoint_snapshot(
                run_signature=run_signature,
                completed_pairs=transcript,
                current_pair=current_pair_state,
            ),
        )

    for i in range(n):
        for j in range(n):
            if i == j or (i, j) in completed_pairs:
                continue
            pair_done = len(completed_pairs) + 1
            _progress(
                f"pair {pair_done}/{total_pairs}: cause={labels[j]} -> effect={labels[i]} "
                f"(indices {j}->{i})"
            )
            first_prompt = _build_first_prompt(
                backend=backend,
                pattern=pattern,
                dataset_name=dataset_name,
                labels=labels,
                i=i,
                j=j,
                adjacency_matrix=adjacency_matrix,
                primary_prob=primary_prob,
                secondary_prob=secondary_prob,
                anonymized=anonymized,
            )
            if current_pair is not None and int(current_pair.get("effect_idx", -1)) == i and int(current_pair.get("cause_idx", -1)) == j:
                pair_state = current_pair
                if not pair_state.get("first_prompt"):
                    pair_state["first_prompt"] = first_prompt
            else:
                pair_state = {
                    "effect_idx": i,
                    "cause_idx": j,
                    "effect": labels[i],
                    "cause": labels[j],
                    "first_prompt": first_prompt,
                    "first_answer": "",
                    "second_prompt": "",
                    "samples": [],
                }
                current_pair = pair_state
                save_checkpoint(current_pair_state=current_pair)

            if not pair_state.get("first_answer"):
                _progress(f"pair {pair_done}/{total_pairs}: requesting first-stage explanation")
                first_result = _chat_text_completion(
                    provider=provider,
                    client=client,
                    model_name=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for causal inference."},
                        {"role": "user", "content": pair_state["first_prompt"]},
                    ],
                    temperature=temperature,
                    max_new_tokens=max_new_tokens_first,
                )
                pair_state["first_answer"] = first_result["text"] or ""
                pair_state["second_prompt"] = _build_second_prompt(
                    pair_state["first_prompt"],
                    pair_state["first_answer"],
                    src=labels[j],
                    dst=labels[i],
                )
                current_pair = pair_state
                save_checkpoint(current_pair_state=current_pair)
            elif not pair_state.get("second_prompt"):
                pair_state["second_prompt"] = _build_second_prompt(
                    pair_state["first_prompt"],
                    pair_state["first_answer"],
                    src=labels[j],
                    dst=labels[i],
                )
                current_pair = pair_state
                save_checkpoint(current_pair_state=current_pair)

            sample_logs = [dict(sample) for sample in pair_state.get("samples", [])]
            probs = [float(sample.get("prob_yes", 0.0)) for sample in sample_logs]
            for trial in range(len(sample_logs), max(1, num_samples)):
                _progress(f"pair {pair_done}/{total_pairs}: second-stage sample {trial + 1}/{max(1, num_samples)}")
                second_result = _chat_text_completion(
                    provider=provider,
                    client=client,
                    model_name=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for causal inference."},
                        {"role": "user", "content": pair_state["second_prompt"]},
                    ],
                    temperature=temperature,
                    max_new_tokens=max_new_tokens_second,
                    logprobs=(provider == "openai"),
                    top_logprobs=20,
                )
                raw_text = second_result["text"] or ""
                if provider == "openai":
                    prob_yes, prob_no, raw_text = _extract_yes_no_prob(second_result["raw"])
                else:
                    prob_yes, prob_no = _extract_yes_no_prob_from_text(raw_text)
                probs.append(prob_yes)
                sample_logs.append(
                    {
                        "trial": trial + 1,
                        "prob_yes": prob_yes,
                        "prob_no": prob_no,
                        "response": raw_text,
                    }
                )
                pair_state["samples"] = sample_logs
                pair_state["mean_prob_yes"] = float(np.mean(probs)) if probs else 0.0
                _progress(
                    f"pair {pair_done}/{total_pairs}: sample {trial + 1} done, "
                    f"prob_yes={_fmt_prob(prob_yes)}, running_mean={_fmt_prob(pair_state['mean_prob_yes'])}"
                )
                current_pair = pair_state
                save_checkpoint(current_pair_state=current_pair)

            mean_matrix[i, j] = float(np.mean(probs)) if probs else 0.0
            completed_entry = {
                "effect_idx": i,
                "cause_idx": j,
                "effect": labels[i],
                "cause": labels[j],
                "first_prompt": pair_state["first_prompt"],
                "first_answer": pair_state["first_answer"],
                "second_prompt": pair_state["second_prompt"],
                "samples": sample_logs,
                "mean_prob_yes": mean_matrix[i, j],
            }
            transcript.append(completed_entry)
            completed_pairs.add((i, j))
            _progress(
                f"pair {pair_done}/{total_pairs}: completed, mean_prob_yes={_fmt_prob(mean_matrix[i, j])}, "
                f"completed_pairs={len(completed_pairs)}/{total_pairs}"
            )
            current_pair = None
            save_checkpoint(current_pair_state=None)
    _progress("LLM pairwise phase done")
    return mean_matrix, transcript


def _probability_to_pk(probability: np.ndarray) -> np.ndarray:
    pk = np.empty(probability.shape, dtype=int)
    n = probability.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                pk[i, j] = -1
            elif probability[i, j] < 0.05:
                pk[i, j] = 0
            elif probability[i, j] > 0.95:
                pk[i, j] = 1
            else:
                pk[i, j] = -1
    return pk


def _probability_to_super_structure(probability: np.ndarray) -> np.ndarray:
    super_graph = np.empty(probability.shape, dtype=int)
    n = probability.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                super_graph[i, j] = 0
            elif probability[i, j] < 0.05:
                super_graph[i, j] = 0
            else:
                super_graph[i, j] = 1
    return super_graph


def _pk_to_background_knowledge(pk_matrix: np.ndarray, df: pd.DataFrame) -> Any:
    pc, _bic_exact_search, BackgroundKnowledge = _require_causallearn()
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


def _run_pc_with_background_knowledge(df: pd.DataFrame, pk_matrix: Optional[np.ndarray]) -> np.ndarray:
    pc, _bic_exact_search, _BackgroundKnowledge = _require_causallearn()
    kwargs: dict[str, Any] = {
        "independence_test_method": "fisherz",
        "verbose": False,
        "show_progress": False,
    }
    if pk_matrix is not None:
        kwargs["background_knowledge"] = _pk_to_background_knowledge(pk_matrix, df)
    cg = pc(df.to_numpy(), **kwargs)
    return _pc_signed_adjacency(cg)


def _binary_prediction_from_effect_by_cause(effect_by_cause: np.ndarray) -> np.ndarray:
    return (np.asarray(effect_by_cause) != 0).astype(int).T


def _directed_prediction_from_pc_signed(pc_signed: np.ndarray) -> np.ndarray:
    return (np.asarray(pc_signed) == 1).astype(int).T


def _run_backend_with_prior(
    *,
    backend: str,
    pattern: int,
    std_df: pd.DataFrame,
    probability_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if backend == "pc":
        pk_matrix = _probability_to_pk(probability_matrix)
        constrained_pc_signed = _run_pc_with_background_knowledge(std_df, pk_matrix=pk_matrix)
        prediction = _directed_prediction_from_pc_signed(constrained_pc_signed)
        return prediction, pk_matrix

    if backend == "exact_search":
        super_graph = _probability_to_super_structure(probability_matrix)
        dag_est = _run_exact_search(std_df, super_graph=super_graph)
        prediction = _binary_prediction_from_effect_by_cause(dag_est)
        return prediction, super_graph

    if backend == "direct_lingam":
        pk_matrix = _probability_to_pk(probability_matrix)
        prior_knowledge = probability_matrix if pattern == 4 else pk_matrix
        try:
            adjacency = _run_direct_lingam(std_df, prior_knowledge=prior_knowledge)
        except Exception:
            if pattern != 4:
                raise
            _progress("DirectLiNGAM rejected raw pattern-4 prior matrix; falling back to thresholded prior knowledge")
            adjacency = _run_direct_lingam(std_df, prior_knowledge=pk_matrix)
            prior_knowledge = pk_matrix
        prediction = _binary_prediction_from_effect_by_cause(adjacency)
        return prediction, np.asarray(prior_knowledge)

    raise SystemExit(f"Unsupported backend in constrained run: {backend}")


def _write_prediction_csv(
    *,
    out_csv: Path,
    backend: str,
    pattern: int,
    model_name: str,
    provider: str,
    naming_regime: str,
    sample_size_obs: int,
    answer: np.ndarray,
    prediction: np.ndarray,
    transcript: list[dict[str, Any]],
    probability_matrix: np.ndarray,
    prior_matrix: np.ndarray,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "backend",
                "pattern",
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
                "backend": backend,
                "pattern": pattern,
                "model": model_name,
                "provider": provider,
                "naming_regime": naming_regime,
                "obs_n": sample_size_obs,
                "int_n": 0,
                "raw_response": json.dumps(
                    {
                        "backend": backend,
                        "pattern": pattern,
                        "probability_matrix_effect_by_cause": probability_matrix.tolist(),
                        "prior_matrix_effect_by_cause": np.asarray(prior_matrix).tolist(),
                        "transcript": transcript,
                    },
                    ensure_ascii=False,
                ),
                "answer": json.dumps(np.asarray(answer, dtype=int).tolist(), ensure_ascii=False),
                "prediction": json.dumps(np.asarray(prediction, dtype=int).tolist(), ensure_ascii=False),
                "valid": 1,
            }
        )


def _repair_probability_matrix_from_transcript(raw_response: dict[str, Any], n: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    probability_matrix = np.zeros((n, n), dtype=float)
    repaired_transcript: list[dict[str, Any]] = []
    for raw_entry in raw_response.get("transcript", []):
        entry = dict(raw_entry)
        i = int(entry["effect_idx"])
        j = int(entry["cause_idx"])
        if not (0 <= i < n and 0 <= j < n) or i == j:
            continue
        repaired_samples: list[dict[str, Any]] = []
        probs: list[float] = []
        for raw_sample in entry.get("samples", []):
            sample = dict(raw_sample)
            response_text = str(sample.get("response", ""))
            prob_yes, prob_no = _extract_yes_no_prob_from_text(response_text)
            if prob_yes == 0.0 and prob_no == 0.0:
                prob_yes = float(sample.get("prob_yes", 0.0))
                prob_no = float(sample.get("prob_no", 0.0))
            sample["prob_yes"] = float(prob_yes)
            sample["prob_no"] = float(prob_no)
            repaired_samples.append(sample)
            probs.append(float(prob_yes))
        mean_prob = float(np.mean(probs)) if probs else float(entry.get("mean_prob_yes", 0.0))
        entry["samples"] = repaired_samples
        entry["mean_prob_yes"] = mean_prob
        probability_matrix[i, j] = mean_prob
        repaired_transcript.append(entry)
    return probability_matrix, repaired_transcript


def _repair_existing_csv(
    *,
    csv_path: Path,
    backend: str,
    pattern: int,
    std_df: pd.DataFrame,
    answer: np.ndarray,
) -> None:
    if not csv_path.exists():
        raise SystemExit(f"Cannot repair missing CSV: {csv_path}")
    csv.field_size_limit(10_000_000)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise SystemExit(f"Expected exactly one row in Takayama CSV, found {len(rows)}: {csv_path}")
    row = rows[0]
    raw_response = json.loads(row.get("raw_response", "") or "{}")
    n = int(answer.shape[0])
    probability_matrix, transcript = _repair_probability_matrix_from_transcript(raw_response, n)
    prediction, prior_matrix = _run_backend_with_prior(
        backend=backend,
        pattern=pattern,
        std_df=std_df,
        probability_matrix=probability_matrix,
    )
    _write_prediction_csv(
        out_csv=csv_path,
        backend=backend,
        pattern=pattern,
        model_name=row.get("model", ""),
        provider=row.get("provider", ""),
        naming_regime=row.get("naming_regime", "real"),
        sample_size_obs=int(float(row.get("obs_n", 0) or 0)),
        answer=answer,
        prediction=prediction,
        transcript=transcript,
        probability_matrix=probability_matrix,
        prior_matrix=prior_matrix,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Takayama SCP runner with PC, Exact Search, and DirectLiNGAM backends.")
    parser.add_argument("--graph_files", type=str, nargs="+", required=True)
    parser.add_argument("--sample_size_obs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--provider", type=str, default="auto")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--max_new_tokens_first", type=int, default=None)
    parser.add_argument("--max_new_tokens_second", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--bootstrap_samples", type=int, default=100)
    parser.add_argument("--backend", choices=sorted(TAKAYAMA_BACKENDS), default="pc")
    parser.add_argument("--takayama_pattern", type=int, default=2)
    parser.add_argument("--naming_regime", choices=["real", "anonymized"], default="real")
    parser.add_argument(
        "--repair_existing_csv",
        type=str,
        default=None,
        help="Repair a completed Takayama CSV from its saved transcript without making API calls.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend = _normalize_backend(args.backend)
    max_new_tokens_first, max_new_tokens_second = _resolve_stage_token_limits(
        shared_max_new_tokens=args.max_new_tokens,
        max_new_tokens_first=args.max_new_tokens_first,
        max_new_tokens_second=args.max_new_tokens_second,
    )

    provider = args.provider
    client = None
    if not args.repair_existing_csv:
        if provider == "auto":
            if "gpt" in args.model.lower() or "o3-" in args.model.lower() or "o1-" in args.model.lower():
                provider = "openai"
            else:
                provider = "illinois"
        if provider == "openai":
            client = _require_openai()
        elif provider == "illinois":
            client = _require_illinois()
        else:
            raise SystemExit("TakayamaSCP supports provider=openai or provider=illinois.")

    for graph_file in args.graph_files:
        graph_path = Path(graph_file).resolve()
        _progress(
            f"start graph={graph_path.stem}, backend={backend}, provider={provider}, model={args.model}, "
            f"obs={args.sample_size_obs}, pattern={args.takayama_pattern}, naming={args.naming_regime}, "
            f"max_first={max_new_tokens_first}, max_second={max_new_tokens_second}"
        )
        _progress("loading observational dataframe")
        _raw_df, std_df, var_names, answer = _load_observational_frames(graph_path, args.sample_size_obs, args.seed)
        _progress(f"loaded dataframe: rows={std_df.shape[0]}, cols={std_df.shape[1]}")
        if args.repair_existing_csv:
            csv_path = Path(args.repair_existing_csv)
            _progress(f"repairing existing CSV from saved transcript: {csv_path}")
            _repair_existing_csv(
                csv_path=csv_path,
                backend=backend,
                pattern=args.takayama_pattern,
                std_df=std_df,
                answer=answer,
            )
            _progress(f"repaired {csv_path.resolve()}")
            continue

        if backend == "pc":
            _progress("running unconstrained PC")
            adjacency_matrix = _run_pc_with_background_knowledge(std_df, pk_matrix=None)
            _progress("unconstrained PC done")
            primary_prob, secondary_prob = _bootstrap_pc_probabilities(std_df, args.bootstrap_samples)
        elif backend == "exact_search":
            _progress("running unconstrained Exact Search")
            adjacency_matrix = _run_exact_search(std_df, super_graph=None)
            _progress("unconstrained Exact Search done")
            primary_prob = _bootstrap_exact_search_probabilities(std_df, args.bootstrap_samples)
            secondary_prob = None
        elif backend == "direct_lingam":
            _progress("running unconstrained DirectLiNGAM")
            adjacency_matrix = _run_direct_lingam(std_df, prior_knowledge=None)
            _progress("unconstrained DirectLiNGAM done")
            primary_prob = _bootstrap_direct_lingam_probabilities(std_df, args.bootstrap_samples)
            secondary_prob = None
        else:
            raise SystemExit(f"Unsupported backend: {backend}")

        if args.naming_regime == "anonymized":
            labels = [f"X{i+1}" for i in range(len(var_names))]
        else:
            labels = list(var_names)

        naming_suffix = "_anon" if args.naming_regime == "anonymized" else ""
        out_csv = Path(args.out_dir) / graph_path.stem / (
            f"predictions_obs{args.sample_size_obs}_int0_TakayamaSCP{_backend_suffix(backend)}_p{int(args.takayama_pattern)}{naming_suffix}.csv"
        )
        checkpoint_path = _checkpoint_path_for_output(out_csv)
        run_signature = _build_run_signature(
            graph_path=graph_path,
            backend=backend,
            model_name=args.model,
            provider=provider,
            labels=labels,
            sample_size_obs=args.sample_size_obs,
            naming_regime=args.naming_regime,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            max_new_tokens_first=max_new_tokens_first,
            max_new_tokens_second=max_new_tokens_second,
            num_samples=args.num_samples,
            bootstrap_samples=args.bootstrap_samples,
            pattern=args.takayama_pattern,
        )
        probability_matrix, transcript = _llm_probability_matrix(
            provider=provider,
            client=client,
            model_name=args.model,
            backend=backend,
            labels=labels,
            dataset_name=graph_path.stem,
            pattern=args.takayama_pattern,
            adjacency_matrix=np.asarray(adjacency_matrix),
            primary_prob=(np.asarray(primary_prob) if primary_prob is not None else None),
            secondary_prob=(np.asarray(secondary_prob) if secondary_prob is not None else None),
            temperature=args.temperature,
            max_new_tokens_first=max_new_tokens_first,
            max_new_tokens_second=max_new_tokens_second,
            num_samples=args.num_samples,
            anonymized=args.naming_regime == "anonymized",
            checkpoint_path=checkpoint_path,
            run_signature=run_signature,
        )
        _progress("running constrained backend with derived prior")
        prediction, prior_matrix = _run_backend_with_prior(
            backend=backend,
            pattern=args.takayama_pattern,
            std_df=std_df,
            probability_matrix=probability_matrix,
        )
        _progress("constrained backend done; writing output csv")
        _write_prediction_csv(
            out_csv=out_csv,
            backend=backend,
            pattern=args.takayama_pattern,
            model_name=args.model,
            provider=provider,
            naming_regime=args.naming_regime,
            sample_size_obs=args.sample_size_obs,
            answer=answer,
            prediction=prediction,
            transcript=transcript,
            probability_matrix=probability_matrix,
            prior_matrix=prior_matrix,
        )
        checkpoint_path.unlink(missing_ok=True)
        _progress(f"wrote {out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
