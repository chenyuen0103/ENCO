#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import copy
import json
import random
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
for _path in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from benchmark_builder.graph_io import load_causal_graph

DEFAULT_OUT_DIR = EXPERIMENTS_DIR / "responses"

try:
    from experiments.generate_prompts import iter_prompts_in_memory
    from experiments.cd_generation.names_only import iter_names_only_prompts_in_memory
    from experiments.query_api import (
        ANSWER_RE,
        build_hf_pipeline,
        call_gemini,
        call_hf_textgen_batch,
        call_openai,
        extract_adjacency_matrix,
        is_gemini_model,
        is_openai_model,
    )
except ModuleNotFoundError:
    from generate_prompts import iter_prompts_in_memory
    from cd_generation.names_only import iter_names_only_prompts_in_memory
    from query_api import (
        ANSWER_RE,
        build_hf_pipeline,
        call_gemini,
        call_hf_textgen_batch,
        call_openai,
        extract_adjacency_matrix,
        is_gemini_model,
        is_openai_model,
    )


METHODS = {"TakayamaSCP", "JiralerspongBFS", "JiralerspongPairwise", "CausalLLMPrompt", "CausalLLMData"}

_PAIRWISE_INIT_PROMPTS: dict[str, dict[str, str]] = {
    "neuropathic": {
        "system": "You are an expert on neuropathic pain diagnosis.",
        "user": "You are a helpful assistant to a neuropathic pain diagnosis expert.",
    },
    "asia": {
        "system": "You are an expert on lung diseases.",
        "user": "You are a helpful assistant to a lung disease expert.",
    },
    "child": {
        "system": "You are an expert on children's diseases.",
        "user": "You are a helpful assistant to a children's disease expert.",
    },
    "alarm": {
        "system": "You are an expert on alarm systems for patient monitoring.",
        "user": "You are a helpful assistant to an expert on alarm systems for patient monitoring.",
    },
    "insurance": {
        "system": "You are an expert on car insurance risks.",
        "user": "You are a helpful assistant to expert on car insurance risks.",
    },
    "sachs": {
        "system": "You are an expert on intracellular protein signaling pathways.",
        "user": "You are a helpful assistant to an expert on intracellular protein signaling pathways.",
    },
}

_BFS_INIT_MESSAGES: dict[str, list[dict[str, str]]] = {
    "asia": [
        {"role": "system", "content": "You are an expert on lung diseases."},
        {
            "role": "user",
            "content": "You are a helpful assistant to experts in lung diesease research. Our goal is to construct a causal graph between the following variables.\n",
        },
    ],
    "child": [
        {"role": "system", "content": "You are an expert on children's diseases."},
        {
            "role": "user",
            "content": "You are a helpful assistant to experts in children's diesease research. Our goal is to construct a causal graph between the following variables.\n",
        },
    ],
    "sachs": [
        {"role": "system", "content": "You are an expert on intracellular protein signaling pathways."},
        {
            "role": "user",
            "content": "You are a helpful assistant to experts in intracellular protein signaling research. Our goal is to construct a causal graph between the following variables.\n",
        },
    ],
}

_BFS_PROMPT_INIT = (
    "Now you are going to use the data to construct a causal graph. "
    "You will start with identifying the variable(s) that are unaffected by any other variables.\n"
)
_BFS_PROMPT_FORMAT = (
    'Think step by step. Then, provide your final answer (variable names only) within the tags '
    '<Answer>...</Answer>, seperated by ", ". '
)


def _resolve_provider(provider: str, model_name: str) -> str:
    if provider != "auto":
        return provider
    if is_gemini_model(model_name):
        return "gemini"
    if is_openai_model(model_name):
        return "openai"
    return "hf"


def _extract_answer_text(text: str) -> str:
    match = ANSWER_RE.search(text or "")
    return match.group(1).strip() if match else (text or "").strip()


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    candidate = _extract_answer_text(text)
    for source in (candidate, text or ""):
        source = source.strip()
        if not source:
            continue
        try:
            parsed = json.loads(source)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        match = re.search(r"\{.*\}", source, flags=re.S)
        if not match:
            continue
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _extract_name_list(text: str, *, key: str, allowed: list[str]) -> list[str]:
    parsed = _extract_json_object(text)
    if not parsed:
        return []
    raw = parsed.get(key, [])
    if not isinstance(raw, list):
        return []
    allowed_set = set(allowed)
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        name = str(item).strip()
        if name in allowed_set and name not in seen:
            out.append(name)
            seen.add(name)
    return out


def _empty_dag(n: int) -> np.ndarray:
    return np.zeros((n, n), dtype=int)


def _load_observational_array(graph_path: Path, sample_size_obs: int, seed: int) -> tuple[np.ndarray, list[str]]:
    if sample_size_obs <= 0:
        raise ValueError("Observational pairwise prompting requires sample_size_obs > 0.")
    graph = load_causal_graph(graph_path)
    variables = [str(v.name) for v in graph.variables]
    if hasattr(graph, "data_obs"):
        arr = np.asarray(graph.data_obs)
        if arr.shape[0] < sample_size_obs:
            raise ValueError(
                f"Requested sample_size_obs={sample_size_obs}, but dataset only exposes {arr.shape[0]} rows."
            )
        return np.asarray(arr[:sample_size_obs]), variables
    if hasattr(graph, "sample"):
        np.random.seed(seed)
        return np.asarray(graph.sample(batch_size=sample_size_obs, as_array=True)), variables
    raise ValueError("Graph object does not expose observational data or sampling.")


def _load_variable_metadata(graph_path: Path) -> list[dict[str, str]]:
    graph = load_causal_graph(graph_path)
    out: list[dict[str, str]] = []
    for var in graph.variables:
        name = str(getattr(var, "name", ""))
        symbol = str(getattr(var, "symbol", name) or name)
        description = str(getattr(var, "description", name) or name)
        out.append({"name": name, "symbol": symbol, "description": description})
    return out


def _would_create_cycle(adj: np.ndarray, src: int, dst: int) -> bool:
    if src == dst:
        return True
    stack = [dst]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node == src:
            return True
        if node in seen:
            continue
        seen.add(node)
        children = np.where(adj[node] > 0)[0].tolist()
        stack.extend(children)
    return False


def _greedy_dag_from_weights(weights: np.ndarray, threshold: float) -> np.ndarray:
    n = weights.shape[0]
    out = np.zeros((n, n), dtype=int)
    candidates: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = float(weights[i, j])
            if w >= threshold:
                candidates.append((w, i, j))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, i, j in candidates:
        if not _would_create_cycle(out, i, j):
            out[i, j] = 1
    return out


def _project_dag(mat: np.ndarray) -> np.ndarray:
    weights = np.asarray(mat, dtype=float)
    return _greedy_dag_from_weights(weights, threshold=0.5)


def _aggregate_sampled_dags(mats: list[np.ndarray], threshold: float) -> np.ndarray:
    if not mats:
        raise ValueError("Need at least one adjacency matrix to aggregate.")
    stacked = np.stack([np.asarray(mat, dtype=float) for mat in mats], axis=0)
    weights = stacked.mean(axis=0)
    np.fill_diagonal(weights, 0.0)
    return _greedy_dag_from_weights(weights, threshold=threshold)


def _call_model(
    *,
    prompt: str,
    model_name: str,
    provider: str,
    temperature: float,
    max_new_tokens: int | None,
    hf_pipe: Any,
) -> str:
    if provider == "gemini":
        return call_gemini(model_name, prompt, temperature=temperature)
    if provider == "openai":
        return call_openai(
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            max_retries=0,
            request_timeout=6000.0,
        )
    if provider == "hf":
        if hf_pipe is None:
            hf_pipe = build_hf_pipeline(model_name)
        outputs = call_hf_textgen_batch(
            hf_pipe,
            [prompt],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            batch_size=1,
        )
        return outputs[0] if outputs else "[ERROR] Empty HF output batch."
    return "[ERROR] Unknown provider"


def _render_messages_as_prompt(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for item in messages:
        role = str(item.get("role", "user")).capitalize()
        content = str(item.get("content", ""))
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def _call_model_messages(
    *,
    messages: list[dict[str, str]],
    model_name: str,
    provider: str,
    temperature: float,
    max_new_tokens: int | None,
    hf_pipe: Any,
) -> str:
    if provider == "openai":
        try:
            from openai import OpenAI
        except Exception as exc:
            return f"[ERROR] ImportError: {exc}"
        try:
            client = OpenAI()
            use_max_completion_tokens = "gpt-5" in model_name.lower()
            req: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }
            if max_new_tokens is not None:
                token_arg = "max_completion_tokens" if use_max_completion_tokens else "max_tokens"
                req[token_arg] = max_new_tokens
            resp = client.chat.completions.create(**req)
            msg = getattr(resp.choices[0], "message", None)
            content = getattr(msg, "content", "") or ""
            return str(content)
        except Exception as exc:
            return f"[ERROR] {type(exc).__name__}: {exc}"
    return _call_model(
        prompt=_render_messages_as_prompt(messages),
        model_name=model_name,
        provider=provider,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        hf_pipe=hf_pipe,
    )


def _load_graph_context(graph_path: Path) -> tuple[Any, list[str], np.ndarray]:
    graph = load_causal_graph(graph_path)
    variables = [str(v.name) for v in graph.variables]
    answer = np.asarray(graph.adj_matrix).astype(int)
    np.fill_diagonal(answer, 0)
    return graph, variables, answer


def _build_names_only_prompt(
    *,
    graph_path: Path,
    num_prompts: int,
    seed: int,
    anonymize: bool,
) -> tuple[list[str], dict[str, Any], str]:
    _base_name, answer_obj, prompt_iter = iter_names_only_prompts_in_memory(
        bif_file=str(graph_path),
        num_prompts=num_prompts,
        seed=seed,
        col_order="original",
        anonymize=anonymize,
        causal_rules=False,
    )
    first = next(prompt_iter)
    variables = [str(v) for v in answer_obj["variables"]]
    return variables, answer_obj, str(first["prompt_text"])


def _build_data_prompt(
    *,
    graph_path: Path,
    sample_size_obs: int,
    sample_size_inters: int,
    seed: int,
    anonymize: bool,
) -> tuple[list[str], dict[str, Any], str]:
    _base_name, answer_obj, prompt_iter = iter_prompts_in_memory(
        bif_file=str(graph_path),
        num_prompts=1,
        shuffles_per_graph=1,
        seed=seed,
        prompt_style="summary",
        obs_per_prompt=sample_size_obs,
        int_per_combo=sample_size_inters,
        row_order="random",
        col_order="original",
        anonymize=anonymize,
        causal_rules=False,
        give_steps=False,
        def_int=False,
        intervene_vars="all",
    )
    first = next(prompt_iter)
    prompt_text = (
        "Use the full observational and interventional evidence below to infer the entire DAG in one step.\n\n"
        + str(first["prompt_text"])
    )
    variables = [str(v) for v in answer_obj["variables"]]
    return variables, answer_obj, prompt_text


def _semantic_full_graph_prompt(prompt_text: str) -> str:
    return (
        "Infer the full directed acyclic causal graph in one step using only the variable names and domain semantics.\n"
        "Prefer a sparse DAG and avoid indirect edges when possible.\n\n"
        + prompt_text
    )


def _new_previous_edges() -> dict[str, dict[str, list[str]]]:
    return {}


def _add_node(previous_edges: dict[str, dict[str, list[str]]], node: str) -> None:
    if node not in previous_edges:
        previous_edges[node] = {"incoming": [], "outgoing": []}


def _add_edge_pairwise(previous_edges: dict[str, dict[str, list[str]]], head: str, tail: str, choice: str) -> None:
    _add_node(previous_edges, head)
    _add_node(previous_edges, tail)
    if choice == "A":
        previous_edges[head]["outgoing"].append(tail)
        previous_edges[tail]["incoming"].append(head)
    elif choice == "B":
        previous_edges[tail]["outgoing"].append(head)
        previous_edges[head]["incoming"].append(tail)


def _get_previous_relevant_edges_string(
    previous_edges: dict[str, dict[str, list[str]]],
    head: str,
    tail: str,
) -> str:
    output = ""
    if head in previous_edges:
        for node in previous_edges[head]["outgoing"]:
            output += f"{head} causes {node}.\n"
        for node in previous_edges[head]["incoming"]:
            output += f"{node} causes {head}.\n"
    if tail in previous_edges:
        for node in previous_edges[tail]["outgoing"]:
            output += f"{tail} causes {node}.\n"
        for node in previous_edges[tail]["incoming"]:
            output += f"{node} causes {tail}.\n"
    return output


def _previous_edges_to_adjacency(previous_edges: dict[str, dict[str, list[str]]], variables: list[str]) -> np.ndarray:
    adj = np.zeros((len(variables), len(variables)), dtype=int)
    index_by_name = {name: idx for idx, name in enumerate(variables)}
    for head, info in previous_edges.items():
        if head not in index_by_name:
            continue
        for node in info["outgoing"]:
            if node in index_by_name:
                adj[index_by_name[head], index_by_name[node]] = 1
    return adj


def _pairwise_prompt(
    *,
    user_prompt: str,
    src: str,
    dst: str,
    all_variables: list[dict[str, str]],
    known_edges_text: str,
    pearson_corr: float | None,
) -> str:
    descriptions = "\n".join(f'{item["name"]}: {item["description"]}' for item in all_variables)
    stats_block = ""
    if pearson_corr is not None:
        stats_block = f'To help you, the Pearson correlation coefficient between "{src}" and "{dst}" is {float(pearson_corr):.2f}\n'
    return (
        f"{user_prompt} \n"
        "Here is a description of the causal variables in this causal graph:\n"
        f"{descriptions}\n\n"
        "Here are the causal relationships you know so far:\n"
        f"{known_edges_text}\n"
        f'We are interested in the causal relationship between "{src}" and "{dst}".\n'
        f"{stats_block}"
        "Which cause-and-effect relationship is more likely?\n"
        f'A. "{src}" causes "{dst}".\n'
        f'B. "{dst}" causes "{src}".\n'
        f'C. There is no causal relationship between "{src}" and "{dst}".\n'
        "Let’s work this out in a step by step way to be sure that we have the right answer. "
        "Then provide your final answer within the tags <Answer>A/B/C</Answer>."
    )


def _extract_pairwise_choice(text: str) -> str | None:
    explicit = re.search(r"<\s*Answer\s*>\s*([ABC])\s*<\s*/\s*Answer\s*>", text or "", flags=re.I)
    if explicit:
        return explicit.group(1).upper()
    candidate = _extract_answer_text(text).strip()
    upper = candidate.upper()
    if upper in {"A", "B", "C"}:
        return upper
    match = re.search(r"\b([ABC])\b", upper)
    if match:
        return match.group(1)
    low = candidate.lower()
    if "no causal relationship" in low:
        return "C"
    if "causes a change in" in low:
        first = re.search(r"changing\s+(.+?)\s+causes a change in\s+(.+?)(?:\.|$)", low)
        if first:
            return "A"
    return None


def _extract_answer_list(text: str) -> list[str]:
    match = re.search(r"<\s*Answer\s*>(.*?)<\s*/\s*Answer\s*>", text or "", flags=re.I | re.S)
    if not match:
        return []
    return [part.strip() for part in match.group(1).split(", ")]


def _sanitize_bfs_nodes(answer_nodes: list[str], *, nodes: list[str], independent_nodes: list[str]) -> list[str]:
    out: list[str] = []
    for node in answer_nodes:
        if not node:
            continue
        if node in independent_nodes:
            continue
        if node not in nodes:
            continue
        if node not in out:
            out.append(node)
    return out


def _bfs_corr_prompt(to_visit: str, corr: np.ndarray, variables: list[str]) -> str:
    idx = variables.index(to_visit)
    prompt = f"Addtionally, the Pearson correlation coefficient between {to_visit} and other variables are as follows:\n"
    for j, var in enumerate(variables):
        if var == to_visit:
            continue
        prompt += f"{var}: {float(corr[j, idx]):.2f}\n"
    return prompt


def _bfs_initial_messages(dataset_name: str) -> list[dict[str, str]]:
    if dataset_name in _BFS_INIT_MESSAGES:
        return copy.deepcopy(_BFS_INIT_MESSAGES[dataset_name])
    return [
        {"role": "system", "content": "You are an expert on causal systems."},
        {
            "role": "user",
            "content": "You are a helpful assistant to experts constructing a causal graph between the following variables.\n",
        },
    ]


def _run_causal_llm_prompt(
    *,
    graph_path: Path,
    model_name: str,
    provider: str,
    temperature: float,
    max_new_tokens: int | None,
    seed: int,
    anonymize: bool,
    hf_pipe: Any,
) -> tuple[np.ndarray, list[str], list[str]]:
    variables, answer_obj, base_prompt = _build_names_only_prompt(
        graph_path=graph_path,
        num_prompts=1,
        seed=seed,
        anonymize=anonymize,
    )
    prompt = _semantic_full_graph_prompt(base_prompt)
    raw = _call_model(
        prompt=prompt,
        model_name=model_name,
        provider=provider,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        hf_pipe=hf_pipe,
    )
    adj = extract_adjacency_matrix(raw, fallback_variables=variables)
    if adj is None:
        adj = _empty_dag(len(variables))
    return np.asarray(adj, dtype=int), variables, [raw]


def _run_takayama_scp(
    *,
    graph_path: Path,
    model_name: str,
    provider: str,
    temperature: float,
    max_new_tokens: int | None,
    seed: int,
    num_samples: int,
    edge_threshold: float,
    anonymize: bool,
    hf_pipe: Any,
) -> tuple[np.ndarray, list[str], list[str]]:
    variables, _answer_obj, base_prompt = _build_names_only_prompt(
        graph_path=graph_path,
        num_prompts=1,
        seed=seed,
        anonymize=anonymize,
    )
    prompt = _semantic_full_graph_prompt(base_prompt)
    mats: list[np.ndarray] = []
    raws: list[str] = []
    sample_temp = temperature if temperature > 0 else 0.7
    for _ in range(max(1, num_samples)):
        raw = _call_model(
            prompt=prompt,
            model_name=model_name,
            provider=provider,
            temperature=sample_temp,
            max_new_tokens=max_new_tokens,
            hf_pipe=hf_pipe,
        )
        raws.append(raw)
        adj = extract_adjacency_matrix(raw, fallback_variables=variables)
        if adj is not None:
            mats.append(np.asarray(adj, dtype=int))
    if not mats:
        return _empty_dag(len(variables)), variables, raws
    return _aggregate_sampled_dags(mats, threshold=edge_threshold), variables, raws


def _run_jiralerspong_bfs(
    *,
    graph_path: Path,
    sample_size_obs: int,
    sample_size_inters: int,
    prompt_mode: str,
    model_name: str,
    provider: str,
    temperature: float,
    max_new_tokens: int | None,
    seed: int,
    anonymize: bool,
    hf_pipe: Any,
) -> tuple[np.ndarray, list[str], list[str]]:
    variable_cards = _load_variable_metadata(graph_path)
    dataset_name = graph_path.stem
    variables = [item["name"] for item in variable_cards]
    corr = None
    if prompt_mode == "summary":
        obs, obs_variables = _load_observational_array(graph_path, sample_size_obs=sample_size_obs, seed=seed)
        variables = list(obs_variables)
        card_by_name = {item["name"]: item for item in variable_cards}
        variable_cards = [card_by_name.get(name, {"name": name, "symbol": name, "description": name}) for name in variables]
        corr = np.corrcoef(obs, rowvar=False)
    raw_transcript: list[str] = []
    message_history = _bfs_initial_messages(dataset_name)
    message_history[1]["content"] += "".join(f'{item["name"]}: {item["description"]}\n' for item in variable_cards)
    message_history[1]["content"] += _BFS_PROMPT_INIT + _BFS_PROMPT_FORMAT

    initial_raw = _call_model_messages(
        messages=message_history,
        model_name=model_name,
        provider=provider,
        temperature=temperature,
        max_new_tokens=max_new_tokens or 4095,
        hf_pipe=hf_pipe,
    )
    raw_transcript.append(
        json.dumps({"stage": "init", "messages": copy.deepcopy(message_history), "response": initial_raw}, ensure_ascii=False)
    )
    message_history.append({"role": "assistant", "content": initial_raw})
    independent_nodes = _extract_answer_list(initial_raw)
    unvisited_nodes = list(variables)
    for node in independent_nodes:
        if node in unvisited_nodes:
            unvisited_nodes.remove(node)
    frontier: list[str] = []
    predict_graph: dict[str, list[str]] = {}

    for to_visit in independent_nodes:
        prompt = "Given " + ", ".join(independent_nodes) + " is(are) not affected by any other variable"
        if len(predict_graph) == 0:
            prompt += ".\n"
        else:
            prompt += " and the following causal relationships.\n"
            for head, tails in predict_graph.items():
                if len(tails) > 0:
                    prompt += f"{head} causes " + ", ".join(tails) + "\n"
        prompt += f"Select variables that are caused by {to_visit}.\n"
        if corr is not None:
            prompt += _bfs_corr_prompt(to_visit, corr, variables)
        prompt += _BFS_PROMPT_FORMAT
        message_history.append({"role": "user", "content": prompt})
        raw = _call_model_messages(
            messages=message_history,
            model_name=model_name,
            provider=provider,
            temperature=temperature,
            max_new_tokens=max_new_tokens or 4095,
            hf_pipe=hf_pipe,
        )
        raw_transcript.append(
            json.dumps({"stage": "root_children", "source": to_visit, "prompt": prompt, "response": raw}, ensure_ascii=False)
        )
        message_history.append({"role": "assistant", "content": raw})
        answer = _sanitize_bfs_nodes(_extract_answer_list(raw), nodes=variables, independent_nodes=independent_nodes)
        predict_graph[to_visit] = answer
        for node in answer:
            if node in unvisited_nodes and node not in frontier:
                frontier.append(node)

    while len(frontier) > 0:
        to_visit = frontier.pop(0)
        if to_visit in unvisited_nodes:
            unvisited_nodes.remove(to_visit)
        prompt = "Given " + ", ".join(independent_nodes) + " is(are) not affected by any other variable and the following causal relationships.\n"
        for head, tails in predict_graph.items():
            if len(tails) > 0:
                prompt += f"{head} causes " + ", ".join(tails) + "\n"
        prompt += f"Select variables that are caused by {to_visit}.\n"
        if corr is not None:
            prompt += _bfs_corr_prompt(to_visit, corr, variables)
        prompt += _BFS_PROMPT_FORMAT
        message_history.append({"role": "user", "content": prompt})
        raw = _call_model_messages(
            messages=message_history,
            model_name=model_name,
            provider=provider,
            temperature=temperature,
            max_new_tokens=max_new_tokens or 4095,
            hf_pipe=hf_pipe,
        )
        raw_transcript.append(
            json.dumps({"stage": "frontier_children", "source": to_visit, "prompt": prompt, "response": raw}, ensure_ascii=False)
        )
        message_history.append({"role": "assistant", "content": raw})
        answer = _sanitize_bfs_nodes(_extract_answer_list(raw), nodes=variables, independent_nodes=independent_nodes)
        predict_graph[to_visit] = answer
        for node in answer:
            if node in unvisited_nodes and node not in frontier:
                frontier.append(node)

    adj = np.zeros((len(variables), len(variables)), dtype=int)
    index_by_name = {name: idx for idx, name in enumerate(variables)}
    for head, tails in predict_graph.items():
        if head not in index_by_name:
            continue
        for node in tails:
            if node in index_by_name:
                adj[index_by_name[head], index_by_name[node]] = 1

    return adj, variables, raw_transcript


def _run_jiralerspong_pairwise(
    *,
    graph_path: Path,
    sample_size_obs: int,
    prompt_mode: str,
    model_name: str,
    provider: str,
    temperature: float,
    max_new_tokens: int | None,
    seed: int,
    anonymize: bool,
    hf_pipe: Any,
) -> tuple[np.ndarray, list[str], list[str]]:
    dataset_name = graph_path.stem
    prompts = _PAIRWISE_INIT_PROMPTS.get(
        dataset_name,
        {
            "system": "You are an expert on causal systems.",
            "user": "You are a helpful assistant to a causal discovery expert.",
        },
    )
    variable_metadata = _load_variable_metadata(graph_path)
    by_name = {item["name"]: item for item in variable_metadata}
    obs, variables = _load_observational_array(graph_path, sample_size_obs=sample_size_obs, seed=seed)
    corr = np.corrcoef(obs, rowvar=False) if prompt_mode == "summary" else None
    rng = random.Random(seed)
    variable_cards = [by_name.get(name, {"name": name, "symbol": name, "description": name}) for name in variables]
    previous_edges = _new_previous_edges()
    raw_transcript: list[str] = []
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            head = variables[i]
            tail = variables[j]
            if rng.random() >= 0.5:
                head, tail = tail, head
            known_edges_text = _get_previous_relevant_edges_string(previous_edges, head, tail)
            pearson = None
            if corr is not None:
                pearson = float(corr[variables.index(head), variables.index(tail)])
            query = _pairwise_prompt(
                user_prompt=prompts["user"],
                src=head,
                dst=tail,
                all_variables=variable_cards,
                known_edges_text=known_edges_text,
                pearson_corr=pearson,
            )
            messages = [
                {"role": "system", "content": prompts["system"]},
                {"role": "user", "content": query},
            ]
            raw = _call_model_messages(
                messages=messages,
                model_name=model_name,
                provider=provider,
                temperature=temperature,
                max_new_tokens=max_new_tokens or 2048,
                hf_pipe=hf_pipe,
            )
            choice = _extract_pairwise_choice(raw)
            if choice is not None:
                _add_edge_pairwise(previous_edges, head, tail, choice)
            record = {
                "pair": [head, tail],
                "pearson_corr": pearson,
                "known_edges_text": known_edges_text,
                "messages": messages,
                "response": raw,
                "choice": choice,
            }
            raw_transcript.append(json.dumps(record, ensure_ascii=False))
    return _previous_edges_to_adjacency(previous_edges, variables), variables, raw_transcript


def _run_causal_llm_data(
    *,
    graph_path: Path,
    sample_size_obs: int,
    sample_size_inters: int,
    model_name: str,
    provider: str,
    temperature: float,
    max_new_tokens: int | None,
    seed: int,
    anonymize: bool,
    hf_pipe: Any,
) -> tuple[np.ndarray, list[str], list[str]]:
    variables, _answer_obj, prompt = _build_data_prompt(
        graph_path=graph_path,
        sample_size_obs=sample_size_obs,
        sample_size_inters=sample_size_inters,
        seed=seed,
        anonymize=anonymize,
    )
    raw = _call_model(
        prompt=prompt,
        model_name=model_name,
        provider=provider,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        hf_pipe=hf_pipe,
    )
    adj = extract_adjacency_matrix(raw, fallback_variables=variables)
    if adj is None:
        adj = _empty_dag(len(variables))
    return np.asarray(adj, dtype=int), variables, [raw]


def _write_prediction_csv(
    *,
    out_csv: Path,
    method: str,
    model_name: str,
    provider: str,
    sample_size_obs: int,
    sample_size_inters: int,
    naming_regime: str,
    answer: np.ndarray,
    prediction: np.ndarray,
    raw_responses: list[str],
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
                "method": method,
                "model": model_name,
                "provider": provider,
                "naming_regime": naming_regime,
                "obs_n": sample_size_obs,
                "int_n": sample_size_inters,
                "raw_response": json.dumps(raw_responses, ensure_ascii=False),
                "answer": json.dumps(np.asarray(answer, dtype=int).tolist(), ensure_ascii=False),
                "prediction": json.dumps(np.asarray(prediction, dtype=int).tolist(), ensure_ascii=False),
                "valid": 1,
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark-native external LLM causal-discovery baselines.")
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--graph_files", type=str, nargs="+", required=True)
    parser.add_argument("--sample_size_obs", type=int, default=100)
    parser.add_argument("--sample_size_inters", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--provider", type=str, default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--edge_threshold", type=float, default=0.5)
    parser.add_argument("--prompt_mode", choices=["names_only", "summary", "summary_joint"], default="names_only")
    parser.add_argument("--naming_regime", choices=["real", "anonymized", "names_only"], default="real")
    args = parser.parse_args()

    if args.prompt_mode == "summary_joint":
        args.prompt_mode = "summary"

    names_only_methods = {"TakayamaSCP", "CausalLLMPrompt"}
    if args.method in names_only_methods and args.prompt_mode != "names_only":
        raise SystemExit(f"{args.method} expects --prompt_mode names_only.")
    if args.method == "JiralerspongBFS" and args.prompt_mode != "summary":
        raise SystemExit("JiralerspongBFS expects --prompt_mode summary.")
    if args.method == "JiralerspongBFS" and args.sample_size_inters != 0:
        raise SystemExit("JiralerspongBFS is observational-only in this implementation.")
    if args.method == "JiralerspongPairwise" and args.prompt_mode != "summary":
        raise SystemExit("JiralerspongPairwise expects --prompt_mode summary to match the original implementation.")
    if args.method == "JiralerspongPairwise" and args.sample_size_inters != 0:
        raise SystemExit("JiralerspongPairwise summary mode is observational-only in this implementation.")
    if (args.method in names_only_methods or args.method == "JiralerspongBFS") and args.naming_regime not in {"real", "names_only", "anonymized"}:
        raise SystemExit(f"Unsupported naming regime for {args.method}: {args.naming_regime}")
    if args.method == "JiralerspongPairwise" and args.naming_regime not in {"real", "anonymized"}:
        raise SystemExit(f"Unsupported naming regime for {args.method}: {args.naming_regime}")
    if args.method == "CausalLLMData" and args.prompt_mode != "summary":
        raise SystemExit("CausalLLMData expects --prompt_mode summary.")
    if args.method == "CausalLLMData" and args.naming_regime == "names_only":
        raise SystemExit("CausalLLMData does not support --naming_regime names_only.")

    provider = _resolve_provider(args.provider, args.model)
    hf_pipe = None
    if provider == "hf":
        hf_pipe = build_hf_pipeline(args.model)

    for graph_file in args.graph_files:
        graph_path = Path(graph_file).resolve()
        _graph, _base_variables, answer = _load_graph_context(graph_path)
        anonymize = args.naming_regime == "anonymized"

        if args.method == "TakayamaSCP":
            prediction, _variables, raw_responses = _run_takayama_scp(
                graph_path=graph_path,
                model_name=args.model,
                provider=provider,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                num_samples=args.num_samples,
                edge_threshold=args.edge_threshold,
                anonymize=anonymize,
                hf_pipe=hf_pipe,
            )
        elif args.method == "JiralerspongBFS":
            prediction, _variables, raw_responses = _run_jiralerspong_bfs(
                graph_path=graph_path,
                sample_size_obs=args.sample_size_obs,
                sample_size_inters=args.sample_size_inters,
                prompt_mode=args.prompt_mode,
                model_name=args.model,
                provider=provider,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                anonymize=anonymize,
                hf_pipe=hf_pipe,
            )
        elif args.method == "JiralerspongPairwise":
            prediction, _variables, raw_responses = _run_jiralerspong_pairwise(
                graph_path=graph_path,
                sample_size_obs=args.sample_size_obs,
                prompt_mode=args.prompt_mode,
                model_name=args.model,
                provider=provider,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                anonymize=anonymize,
                hf_pipe=hf_pipe,
            )
        elif args.method == "CausalLLMPrompt":
            prediction, _variables, raw_responses = _run_causal_llm_prompt(
                graph_path=graph_path,
                model_name=args.model,
                provider=provider,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                anonymize=anonymize,
                hf_pipe=hf_pipe,
            )
        elif args.method == "CausalLLMData":
            prediction, _variables, raw_responses = _run_causal_llm_data(
                graph_path=graph_path,
                sample_size_obs=args.sample_size_obs,
                sample_size_inters=args.sample_size_inters,
                model_name=args.model,
                provider=provider,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                anonymize=anonymize,
                hf_pipe=hf_pipe,
            )
        else:
            raise SystemExit(f"Unsupported method: {args.method}")

        naming_suffix = ""
        if args.naming_regime == "anonymized":
            naming_suffix = "_anon"
        elif args.naming_regime == "names_only":
            naming_suffix = "_names_only"
        out_csv = Path(args.out_dir) / graph_path.stem / (
            f"predictions_obs{args.sample_size_obs}_int{args.sample_size_inters}_{args.method}{naming_suffix}.csv"
        )
        _write_prediction_csv(
            out_csv=out_csv,
            method=args.method,
            model_name=args.model,
            provider=provider,
            sample_size_obs=args.sample_size_obs,
            sample_size_inters=args.sample_size_inters,
            naming_regime=args.naming_regime,
            answer=answer,
            prediction=prediction,
            raw_responses=raw_responses,
        )
        print(f"[{args.method}] wrote {out_csv.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
