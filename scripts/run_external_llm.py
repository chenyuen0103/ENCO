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
        _model_requires_default_temperature,
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
        _model_requires_default_temperature,
    )


METHODS = {
    "TakayamaSCP",
    "JiralerspongBFS",
    "JiralerspongPairwise",
    "CausalLLMPrompt",
    "CausalLLMData",
    "CausalLLMTrainableData",
}

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
        rng = np.random.default_rng(seed)
        idx = rng.choice(arr.shape[0], size=sample_size_obs, replace=False)
        return np.asarray(arr[idx]), variables
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


def _standardize_numeric_data(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D data array, got shape {arr.shape}.")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (arr - mean) / std


def _prune_causal_llm_scores_by_linear_coef(
    scores: np.ndarray,
    data: np.ndarray,
    *,
    edge_threshold: float = 0.5,
    max_edges: int | None = None,
) -> np.ndarray:
    """Prune candidate src->dst scores using linear-regression coefficients.

    This follows the upstream CausalLLM data baseline's pruning idea, but keeps
    this repo's adjacency convention: rows are sources, columns are targets.
    """
    from sklearn.linear_model import LinearRegression

    score_arr = np.asarray(scores, dtype=float)
    data_arr = _standardize_numeric_data(data)
    if score_arr.ndim != 2 or score_arr.shape[0] != score_arr.shape[1]:
        raise ValueError(f"Expected square score matrix, got shape {score_arr.shape}.")
    n = score_arr.shape[0]
    if data_arr.shape[1] != n:
        raise ValueError(f"Data has {data_arr.shape[1]} variables, but scores are {n}x{n}.")
    np.fill_diagonal(score_arr, 0.0)

    coef_weights = np.zeros((n, n), dtype=float)
    reg = LinearRegression()
    for dst in range(n):
        parent_mask = score_arr[:, dst] > float(edge_threshold)
        parent_mask[dst] = False
        parent_indices = np.where(parent_mask)[0]
        if len(parent_indices) == 0:
            continue
        reg.fit(data_arr[:, parent_indices], data_arr[:, dst])
        for src, coef in zip(parent_indices, reg.coef_):
            coef_weights[src, dst] = abs(float(coef))

    positives = coef_weights[coef_weights > 0.0]
    if positives.size == 0:
        return _empty_dag(n)
    keep = max(1, min(int(max_edges if max_edges is not None else n), positives.size))
    cutoff = np.sort(positives)[::-1][keep - 1]
    selected = np.where(coef_weights >= cutoff, coef_weights, 0.0)
    np.fill_diagonal(selected, 0.0)
    return _greedy_dag_from_weights(selected, threshold=1e-12)


class CausalLLMTrainableData:
    """Cleaned implementation of the upstream trainable CausalLLM data baseline."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        *,
        model_path: str | Path | None = None,
        hidden_size: int = 512,
        intermediate_size: int = 1024,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        learning_rate: float = 2e-5,
        l1_lambda: float = 0.01,
        device: str | None = None,
    ) -> None:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from transformers import LlamaConfig, LlamaModel

        if int(num_attention_heads) <= 0 or int(num_hidden_layers) <= 0:
            raise ValueError("num_attention_heads and num_hidden_layers must be positive.")
        if int(hidden_size) % int(num_attention_heads) != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        self.torch = torch
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim if output_dim is not None else input_dim * input_dim)
        self.model_path = Path(model_path) if model_path else None
        self.l1_lambda = float(l1_lambda)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        output_dim_local = self.output_dim

        class _Backbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                config = LlamaConfig(
                    hidden_size=int(hidden_size),
                    intermediate_size=int(intermediate_size),
                    num_hidden_layers=int(num_hidden_layers),
                    num_attention_heads=int(num_attention_heads),
                    max_position_embeddings=512,
                    vocab_size=32000,
                )
                self.llama = LlamaModel(config)
                for param in self.llama.parameters():
                    param.requires_grad = False
                self.input_projection = nn.Linear(int(input_dim), config.hidden_size)
                self.output_projection = nn.Linear(config.hidden_size, int(output_dim_local))

            def forward(self, x: Any) -> Any:
                x = self.input_projection(x.to(torch.float32))
                outputs = self.llama(inputs_embeds=x)
                return self.output_projection(outputs.last_hidden_state)

        self.model = _Backbone().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(learning_rate))
        self.criterion = nn.BCELoss()
        if self.model_path and self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()

    def learn(self, data: np.ndarray, *, num_epochs: int = 10, batch_size: int = 32, epsilon: float = 0.1, seed: int = 0) -> None:
        data_arr = _standardize_numeric_data(data)
        if data_arr.shape[1] != self.input_dim:
            raise ValueError(f"Data has {data_arr.shape[1]} variables, expected {self.input_dim}.")
        torch = self.torch
        rng = np.random.default_rng(seed)
        torch.manual_seed(int(seed))
        self.model.train()
        for _epoch in range(max(1, int(num_epochs))):
            for _ in range(max(1, int(batch_size))):
                row = data_arr[int(rng.integers(data_arr.shape[0]))]
                state = torch.tensor(row, dtype=torch.float32, device=self.device).view(1, 1, self.input_dim)
                logits = self.model(state).view(-1)
                probs = torch.sigmoid(logits)
                if rng.random() < float(epsilon):
                    random_probs = torch.rand_like(probs)
                    probs = probs + (random_probs - probs).detach()
                target = torch.zeros_like(probs)
                loss = self.criterion(probs, target)
                if self.l1_lambda > 0:
                    l1_norm = sum(param.abs().sum() for param in self.model.parameters() if param.requires_grad)
                    loss = loss + self.l1_lambda * l1_norm
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.model.parameters():
                    if param.requires_grad and param.grad is not None:
                        param.grad.data.clamp_(-1.0, 1.0)
                self.optimizer.step()
        if self.model_path:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)

    def causal_scores(self, data: np.ndarray) -> np.ndarray:
        data_arr = _standardize_numeric_data(data)
        torch = self.torch
        src = torch.tensor(data_arr.mean(axis=0), dtype=torch.float32, device=self.device).view(1, 1, self.input_dim)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(src).view(self.input_dim, self.input_dim)
            scores = torch.sigmoid(logits).detach().cpu().numpy()
        np.fill_diagonal(scores, 0.0)
        return scores

    def causal_matrix(self, data: np.ndarray, *, edge_threshold: float = 0.5, max_edges: int | None = None) -> np.ndarray:
        scores = self.causal_scores(data)
        return _prune_causal_llm_scores_by_linear_coef(
            scores,
            data,
            edge_threshold=edge_threshold,
            max_edges=max_edges,
        )


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
            client = OpenAI(timeout=180.0, max_retries=2)
            use_max_completion_tokens = "gpt-5" in model_name.lower()
            req: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
            }
            if not _model_requires_default_temperature(model_name):
                req.update(
                    {
                        "temperature": temperature,
                        "top_p": 1,
                        "frequency_penalty": 0,
                        "presence_penalty": 0,
                    }
                )
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


# -- CausalLLM dataset-specific prompts (Roy et al. 2023) ----------------------
# Exact prompt from the paper's repo for the Sachs signaling network.
# Capitalisation differences (p38/P38, PLCg/Plcg, pIP3/PIP3) are resolved by
# case-insensitive matching at parse time.
SACHS_CAUSAL_LLM_PROMPT = """\
You are an *intelligent causal discovery agent* tasked with mapping how signaling molecules interact in the Sachs dataset to form a causal signaling network. These molecules influence one another through biochemical processes like activation, inhibition, or enzymatic transformation, ultimately leading to downstream cellular responses.

### **Important Rules:**
- Each signaling molecule may have *multiple incoming edges* to reflect how upstream molecules influence its activity.
- Some molecules act as *critical intermediaries* (e.g., converting signals or amplifying responses) and may have both *incoming and outgoing edges*.
- The causal DAG should faithfully represent known causal relationships in the Sachs dataset based on experimental data and biological knowledge.

### **Features:**

1. **Akt**: A kinase involved in cell survival pathways, regulating processes like metabolism, proliferation, and apoptosis.
2. **Erk**: Extracellular signal-regulated kinase, part of the MAP kinase pathway, essential for cell division and differentiation.
3. **Jnk**: c-Jun N-terminal kinase, associated with stress response and apoptosis signaling.
4. **p38**: A stress-activated protein kinase involved in responses to inflammation and environmental stress.
5. **PIP2**: Phosphatidylinositol 4,5-bisphosphate, a phospholipid precursor involved in signal transduction and membrane dynamics.
6. **PIP3**: Phosphatidylinositol 3,4,5-trisphosphate, generated by PI3K and a key regulator of Akt signaling.
7. **PKA**: Protein kinase A, a cAMP-dependent kinase that regulates metabolic and gene transcription processes.
8. **PKC**: Protein kinase C, involved in regulating various cellular functions, including gene expression and membrane signaling.
9. **PLCg**: Phospholipase C gamma, an enzyme that hydrolyzes PIP2 into IP3 and DAG, key molecules in calcium signaling.
10. **Raf**: A kinase that acts upstream of MEK and Erk in the MAPK/ERK signaling pathway, influencing cell growth and survival.
11. **pIP3**: Phosphorylated inositol triphosphate, linked to calcium signaling and involved in cellular communication.

---

### **Output Example:**

### **Step 1: Finding the Edges**

Here are the identified edges, focusing on how the signaling molecules influence one another:

1. **Edge (PIP2 → PIP3):** PIP2 is phosphorylated by PI3K to form PIP3, marking a key step in activating the Akt signaling pathway.
2.........
..
.
.

---

### **Step 2: Reflect back on each edge to see if it matches Domain Knowledge and give the finalized set of edges.**

---

**Output format:**
Provide a list of edges in the format specified above. For example:
```
1. (A, B) : Explanation of why A causes B.
2. (C, D) : Explanation of why C causes D.
...
```\
"""

# Lookup: dataset stem → dataset-specific prompt (only Sachs so far)
_CAUSAL_LLM_PROMPTS: dict[str, str] = {
    "sachs": SACHS_CAUSAL_LLM_PROMPT,
}


def _semantic_full_graph_prompt(prompt_text: str) -> str:
    return (
        "Infer the full directed acyclic causal graph in one step using only the variable names and domain semantics.\n"
        "Prefer a sparse DAG and avoid indirect edges when possible.\n\n"
        + prompt_text
    )


def _parse_causal_llm_two_step(raw: str, variables: list[str]) -> np.ndarray:
    """Parse the two-step Roy et al. output; keep only edges confirmed in both steps.

    Step 1 uses arrow notation  (A → B) or (A -> B).
    Step 2 uses comma notation  (A, B).
    An edge is accepted iff it appears in both steps (case-insensitive variable match).
    """
    name_to_idx: dict[str, int] = {v.lower(): i for i, v in enumerate(variables)}

    def _idx(name: str) -> int | None:
        return name_to_idx.get(name.lower())

    # Step 1 — arrow edges anywhere in the full output
    step1: set[tuple[int, int]] = set()
    for src, tgt in re.findall(r'\((\w+)\s*(?:→|->)\s*(\w+)\)', raw):
        si, ti = _idx(src), _idx(tgt)
        if si is not None and ti is not None and si != ti:
            step1.add((si, ti))

    # Step 2 — comma edges in the text after the "Step 2" heading
    parts = re.split(r'step\s*2', raw, flags=re.IGNORECASE)
    step2_text = parts[1] if len(parts) > 1 else ""
    step2: set[tuple[int, int]] = set()
    for src, tgt in re.findall(r'\((\w+),\s*(\w+)\)', step2_text):
        si, ti = _idx(src), _idx(tgt)
        if si is not None and ti is not None and si != ti:
            step2.add((si, ti))

    n = len(variables)
    adj = np.zeros((n, n), dtype=int)
    for edge in step1 & step2:
        adj[edge[0], edge[1]] = 1
    return adj


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


def _checkpoint_path_for_output(out_csv: Path) -> Path:
    return out_csv.with_suffix(out_csv.suffix + ".checkpoint.json")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _pairwise_checkpoint_signature(
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
) -> dict[str, Any]:
    return {
        "graph_file": str(graph_path.resolve()),
        "sample_size_obs": int(sample_size_obs),
        "prompt_mode": prompt_mode,
        "model": model_name,
        "provider": provider,
        "temperature": float(temperature),
        "max_new_tokens": max_new_tokens,
        "seed": int(seed),
        "anonymize": bool(anonymize),
    }


def _load_pairwise_checkpoint(path: Path, signature: dict[str, Any]) -> list[str]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("signature") != signature:
        raise SystemExit(
            f"Checkpoint {path} does not match this run configuration. "
            "Move it aside or use the same command to resume."
        )
    records = payload.get("raw_transcript", [])
    if not isinstance(records, list):
        raise SystemExit(f"Malformed checkpoint raw_transcript in {path}")
    return [str(record) for record in records]


def _replay_pairwise_edges(raw_transcript: list[str]) -> dict[str, dict[str, list[str]]]:
    previous_edges = _new_previous_edges()
    for raw in raw_transcript:
        try:
            record = json.loads(raw)
        except Exception:
            continue
        pair = record.get("pair")
        choice = record.get("choice")
        if isinstance(pair, list) and len(pair) == 2 and choice in {"A", "B", "C"}:
            _add_edge_pairwise(previous_edges, str(pair[0]), str(pair[1]), str(choice))
    return previous_edges


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


def _response_preview(text: str, limit: int = 300) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


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
    # Use dataset-specific Roy et al. prompt when available (real names only).
    # Fall back to generic semantic prompt for anonymized runs or unknown datasets.
    dataset_key = graph_path.stem.lower()
    if not anonymize and dataset_key in _CAUSAL_LLM_PROMPTS:
        prompt = _CAUSAL_LLM_PROMPTS[dataset_key]
        use_two_step_parser = True
    else:
        prompt = _semantic_full_graph_prompt(base_prompt)
        use_two_step_parser = False

    raw = _call_model(
        prompt=prompt,
        model_name=model_name,
        provider=provider,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        hf_pipe=hf_pipe,
    )
    if use_two_step_parser:
        adj = _parse_causal_llm_two_step(raw, variables)
    else:
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
    print(
        f"[JiralerspongBFS] start dataset={dataset_name} seed={seed} "
        f"obs={sample_size_obs} vars={len(variables)}",
        flush=True,
    )
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
    print(
        f"[JiralerspongBFS] init done seed={seed} independent={len(independent_nodes)} "
        f"{independent_nodes}",
        flush=True,
    )
    unvisited_nodes = list(variables)
    for node in independent_nodes:
        if node in unvisited_nodes:
            unvisited_nodes.remove(node)
    frontier: list[str] = []
    predict_graph: dict[str, list[str]] = {}

    for root_idx, to_visit in enumerate(independent_nodes, start=1):
        print(
            f"[JiralerspongBFS] root {root_idx}/{len(independent_nodes)} seed={seed}: {to_visit}",
            flush=True,
        )
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
        print(
            f"[JiralerspongBFS] root {root_idx}/{len(independent_nodes)} done seed={seed}: "
            f"children={len(answer)} frontier={len(frontier)}",
            flush=True,
        )

    while len(frontier) > 0:
        to_visit = frontier.pop(0)
        print(
            f"[JiralerspongBFS] frontier seed={seed}: visiting={to_visit} remaining={len(frontier)}",
            flush=True,
        )
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
        print(
            f"[JiralerspongBFS] frontier done seed={seed}: {to_visit} "
            f"children={len(answer)} frontier={len(frontier)}",
            flush=True,
        )

    adj = np.zeros((len(variables), len(variables)), dtype=int)
    index_by_name = {name: idx for idx, name in enumerate(variables)}
    for head, tails in predict_graph.items():
        if head not in index_by_name:
            continue
        for node in tails:
            if node in index_by_name:
                adj[index_by_name[head], index_by_name[node]] = 1

    print(
        f"[JiralerspongBFS] done dataset={dataset_name} seed={seed} "
        f"edges={int(adj.sum())}",
        flush=True,
    )
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
    checkpoint_path: Path | None = None,
    partial_out_csv: Path | None = None,
    answer: np.ndarray | None = None,
    naming_regime: str = "real",
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
    signature = _pairwise_checkpoint_signature(
        graph_path=graph_path,
        sample_size_obs=sample_size_obs,
        prompt_mode=prompt_mode,
        model_name=model_name,
        provider=provider,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        seed=seed,
        anonymize=anonymize,
    )
    raw_transcript = _load_pairwise_checkpoint(checkpoint_path, signature) if checkpoint_path else []
    previous_edges = _replay_pairwise_edges(raw_transcript)
    completed_pairs = len(raw_transcript)
    total_pairs = len(variables) * (len(variables) - 1) // 2
    pair_idx = 0
    if completed_pairs:
        print(
            f"[JiralerspongPairwise] resuming from {checkpoint_path} completed_pairs={completed_pairs}",
            file=sys.stderr,
            flush=True,
        )
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            pair_idx += 1
            head = variables[i]
            tail = variables[j]
            if rng.random() >= 0.5:
                head, tail = tail, head
            if pair_idx <= completed_pairs:
                continue
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
            raw = ""
            choice: str | None = None
            max_attempts = 3 if provider == "openai" else 1
            for attempt in range(1, max_attempts + 1):
                print(
                    f"[JiralerspongPairwise] pair {pair_idx}/{total_pairs}: querying {head} vs {tail} attempt={attempt}",
                    file=sys.stderr,
                    flush=True,
                )
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
                    break
                reason = "empty response" if not str(raw).strip() else f"unparseable response: {_response_preview(raw)}"
                print(
                    f"[JiralerspongPairwise] pair {pair_idx}/{total_pairs}: {reason}, retrying",
                    file=sys.stderr,
                    flush=True,
                )
            if choice is None:
                checkpoint_msg = f" Checkpoint preserved at {checkpoint_path}." if checkpoint_path is not None else ""
                raise RuntimeError(
                    f"JiralerspongPairwise failed to get a usable <Answer>A/B/C</Answer> "
                    f"for pair {pair_idx}/{total_pairs}: {head} vs {tail} after {max_attempts} attempt(s). "
                    f"Last response: {_response_preview(raw) or '<empty>'}.{checkpoint_msg}"
                )
            print(
                f"[JiralerspongPairwise] pair {pair_idx}/{total_pairs}: {head} vs {tail} choice={choice}",
                file=sys.stderr,
                flush=True,
            )
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
            if checkpoint_path is not None:
                _atomic_write_json(
                    checkpoint_path,
                    {
                        "version": 1,
                        "signature": signature,
                        "raw_transcript": raw_transcript,
                    },
                )
            if partial_out_csv is not None and answer is not None:
                partial_prediction = _previous_edges_to_adjacency(previous_edges, variables)
                _write_prediction_csv(
                    out_csv=partial_out_csv,
                    method="JiralerspongPairwise",
                    model_name=model_name,
                    provider=provider,
                    sample_size_obs=sample_size_obs,
                    sample_size_inters=0,
                    naming_regime=naming_regime,
                    answer=answer,
                    prediction=partial_prediction,
                    raw_responses=raw_transcript,
                )
    prediction = _previous_edges_to_adjacency(previous_edges, variables)
    if checkpoint_path is not None:
        checkpoint_path.unlink(missing_ok=True)
    return prediction, variables, raw_transcript


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


def _run_causal_llm_trainable_data(
    *,
    graph_path: Path,
    sample_size_obs: int,
    sample_size_inters: int,
    seed: int,
    num_epochs: int,
    batch_size: int,
    epsilon: float,
    edge_threshold: float,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    learning_rate: float,
    l1_lambda: float,
) -> tuple[np.ndarray, list[str], list[str]]:
    if sample_size_inters != 0:
        raise ValueError("CausalLLMTrainableData is observational-only to match the upstream implementation.")
    obs, variables = _load_observational_array(graph_path, sample_size_obs=sample_size_obs, seed=seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
    model = CausalLLMTrainableData(
        input_dim=len(variables),
        output_dim=len(variables) * len(variables),
        hidden_size=hidden_size,
        intermediate_size=max(hidden_size * 2, hidden_size),
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        learning_rate=learning_rate,
        l1_lambda=l1_lambda,
    )
    model.learn(obs, num_epochs=num_epochs, batch_size=batch_size, epsilon=epsilon, seed=seed)
    prediction = np.asarray(
        model.causal_matrix(obs, edge_threshold=edge_threshold, max_edges=len(variables)),
        dtype=int,
    )
    np.fill_diagonal(prediction, 0)
    raw = json.dumps(
        {
            "implementation": "CausalLLMTrainableData",
            "source": "devharish1371 Causal-LLM Dag_generation and model_evaluation/causal_llm.py",
            "sample_size_obs": int(sample_size_obs),
            "sample_size_inters": int(sample_size_inters),
            "seed": int(seed),
            "num_epochs": int(num_epochs),
            "batch_size": int(batch_size),
            "epsilon": float(epsilon),
            "edge_threshold": float(edge_threshold),
            "hidden_size": int(hidden_size),
            "num_hidden_layers": int(num_hidden_layers),
            "num_attention_heads": int(num_attention_heads),
            "learning_rate": float(learning_rate),
            "l1_lambda": float(l1_lambda),
        },
        ensure_ascii=False,
    )
    return prediction, variables, [raw]


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
    _write_prediction_rows(
        out_csv=out_csv,
        rows=[
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
        ],
    )


def _write_prediction_rows(*, out_csv: Path, rows: list[dict[str, Any]]) -> None:
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
                "replicate_index",
                "replicate_seed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_completed_replicate_rows(out_csv: Path) -> dict[int, dict[str, Any]]:
    if not out_csv.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    with out_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                rep = int(row.get("replicate_index", ""))
            except Exception:
                continue
            rows[rep] = dict(row)
    return rows


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
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--edge_threshold", type=float, default=0.5)
    parser.add_argument("--causal_llm_epochs", type=int, default=10)
    parser.add_argument("--causal_llm_batch_size", type=int, default=32)
    parser.add_argument("--causal_llm_epsilon", type=float, default=0.1)
    parser.add_argument("--causal_llm_hidden_size", type=int, default=512)
    parser.add_argument("--causal_llm_layers", type=int, default=8)
    parser.add_argument("--causal_llm_heads", type=int, default=8)
    parser.add_argument("--causal_llm_lr", type=float, default=2e-5)
    parser.add_argument("--causal_llm_l1", type=float, default=0.01)
    parser.add_argument("--prompt_mode", choices=["names_only", "summary", "summary_joint"], default="names_only")
    parser.add_argument("--naming_regime", choices=["real", "anonymized", "names_only"], default="real")
    args = parser.parse_args()

    if args.prompt_mode == "summary_joint":
        args.prompt_mode = "summary"
    if args.num_prompts <= 0:
        raise SystemExit("--num_prompts must be > 0.")

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
    if args.method == "CausalLLMTrainableData" and args.prompt_mode != "summary":
        raise SystemExit("CausalLLMTrainableData expects --prompt_mode summary.")
    if args.method == "CausalLLMTrainableData" and args.sample_size_inters != 0:
        raise SystemExit("CausalLLMTrainableData is observational-only in the upstream implementation.")
    if args.method == "CausalLLMTrainableData" and args.naming_regime == "names_only":
        raise SystemExit("CausalLLMTrainableData does not support --naming_regime names_only.")
    if args.method == "CausalLLMTrainableData" and args.causal_llm_hidden_size % args.causal_llm_heads != 0:
        raise SystemExit("--causal_llm_hidden_size must be divisible by --causal_llm_heads.")

    provider = "local" if args.method == "CausalLLMTrainableData" else _resolve_provider(args.provider, args.model)
    hf_pipe = None
    if provider == "hf":
        hf_pipe = build_hf_pipeline(args.model)

    for graph_file in args.graph_files:
        graph_path = Path(graph_file).resolve()
        _graph, _base_variables, answer = _load_graph_context(graph_path)
        anonymize = args.naming_regime == "anonymized"
        naming_suffix = ""
        if args.naming_regime == "anonymized":
            naming_suffix = "_anon"
        elif args.naming_regime == "names_only":
            naming_suffix = "_names_only"
        out_csv = Path(args.out_dir) / graph_path.stem / (
            f"predictions_obs{args.sample_size_obs}_int{args.sample_size_inters}_{args.method}_seed{int(args.seed)}{naming_suffix}.csv"
        )

        rows_by_replicate = _read_completed_replicate_rows(out_csv)
        for prompt_idx in range(args.num_prompts):
            if prompt_idx in rows_by_replicate:
                print(
                    f"[{args.method}] replicate {prompt_idx + 1}/{args.num_prompts} "
                    f"already present in {out_csv}; skipping",
                    flush=True,
                )
                continue
            seed_i = int(args.seed) + prompt_idx * 1000
            print(
                f"[{args.method}] replicate {prompt_idx + 1}/{args.num_prompts} start seed={seed_i}",
                flush=True,
            )
            if args.method == "TakayamaSCP":
                prediction, _variables, raw_responses = _run_takayama_scp(
                    graph_path=graph_path,
                    model_name=args.model,
                    provider=provider,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    seed=seed_i,
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
                    seed=seed_i,
                    anonymize=anonymize,
                    hf_pipe=hf_pipe,
                )
            elif args.method == "JiralerspongPairwise":
                checkpoint_path = _checkpoint_path_for_output(out_csv)
                if args.num_prompts > 1:
                    checkpoint_path = out_csv.with_suffix(out_csv.suffix + f".rep{prompt_idx}.checkpoint.json")
                prediction, _variables, raw_responses = _run_jiralerspong_pairwise(
                    graph_path=graph_path,
                    sample_size_obs=args.sample_size_obs,
                    prompt_mode=args.prompt_mode,
                    model_name=args.model,
                    provider=provider,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    seed=seed_i,
                    anonymize=anonymize,
                    hf_pipe=hf_pipe,
                    checkpoint_path=checkpoint_path,
                    partial_out_csv=(out_csv if args.num_prompts == 1 else None),
                    answer=answer,
                    naming_regime=args.naming_regime,
                )
            elif args.method == "CausalLLMPrompt":
                prediction, _variables, raw_responses = _run_causal_llm_prompt(
                    graph_path=graph_path,
                    model_name=args.model,
                    provider=provider,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    seed=seed_i,
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
                    seed=seed_i,
                    anonymize=anonymize,
                    hf_pipe=hf_pipe,
                )
            elif args.method == "CausalLLMTrainableData":
                prediction, _variables, raw_responses = _run_causal_llm_trainable_data(
                    graph_path=graph_path,
                    sample_size_obs=args.sample_size_obs,
                    sample_size_inters=args.sample_size_inters,
                    seed=seed_i,
                    num_epochs=args.causal_llm_epochs,
                    batch_size=args.causal_llm_batch_size,
                    epsilon=args.causal_llm_epsilon,
                    edge_threshold=args.edge_threshold,
                    hidden_size=args.causal_llm_hidden_size,
                    num_hidden_layers=args.causal_llm_layers,
                    num_attention_heads=args.causal_llm_heads,
                    learning_rate=args.causal_llm_lr,
                    l1_lambda=args.causal_llm_l1,
                )
            else:
                raise SystemExit(f"Unsupported method: {args.method}")

            rows_by_replicate[prompt_idx] = {
                "method": args.method,
                "model": args.model,
                "provider": provider,
                "naming_regime": args.naming_regime,
                "obs_n": args.sample_size_obs,
                "int_n": args.sample_size_inters,
                "raw_response": json.dumps(raw_responses, ensure_ascii=False),
                "answer": json.dumps(np.asarray(answer, dtype=int).tolist(), ensure_ascii=False),
                "prediction": json.dumps(np.asarray(prediction, dtype=int).tolist(), ensure_ascii=False),
                "valid": 1,
                "replicate_index": prompt_idx,
                "replicate_seed": seed_i,
            }
            _write_prediction_rows(out_csv=out_csv, rows=[rows_by_replicate[idx] for idx in sorted(rows_by_replicate)])
            print(
                f"[{args.method}] replicate {prompt_idx + 1}/{args.num_prompts} done; "
                f"rows={len(rows_by_replicate)}/{args.num_prompts}",
                flush=True,
            )

        _write_prediction_rows(out_csv=out_csv, rows=[rows_by_replicate[idx] for idx in sorted(rows_by_replicate)])
        print(f"[{args.method}] wrote {out_csv.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
