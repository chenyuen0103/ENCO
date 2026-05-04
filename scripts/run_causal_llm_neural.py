#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_builder.graph_io import load_causal_graph

DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "responses"


def _require_neural_deps():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.linear_model import LinearRegression
        from transformers import LlamaConfig, LlamaModel
        import networkx as nx
    except Exception as exc:
        raise SystemExit(
            "CausalLLMDataNeural requires torch, transformers, scikit-learn, and networkx. "
            "Install the missing dependency before running this baseline."
        ) from exc
    return torch, nn, optim, LinearRegression, LlamaConfig, LlamaModel, nx


def _load_observational_array(graph: Any, sample_size_obs: int, seed: int) -> tuple[np.ndarray, list[str], list[int] | None]:
    if sample_size_obs <= 0:
        raise SystemExit("CausalLLMDataNeural requires --sample_size_obs > 0.")
    variables = [str(v.name) for v in graph.variables]
    if hasattr(graph, "data_obs"):
        arr = np.asarray(graph.data_obs)
        if arr.shape[0] < sample_size_obs:
            raise SystemExit(
                f"Requested sample_size_obs={sample_size_obs}, but dataset only exposes {arr.shape[0]} rows."
            )
        rng = np.random.default_rng(seed)
        row_idx = rng.choice(arr.shape[0], size=sample_size_obs, replace=False)
        return np.asarray(arr[row_idx], dtype=np.float32), variables, row_idx.astype(int).tolist()
    if hasattr(graph, "sample"):
        np.random.seed(seed)
        return np.asarray(graph.sample(batch_size=sample_size_obs, as_array=True), dtype=np.float32), variables, None
    raise SystemExit("Graph object does not expose observational data or sampling.")


def _fit_predict_causal_llm(
    data: np.ndarray,
    *,
    seed: int,
    num_epochs: int,
    batch_size: int,
    epsilon: float,
    l1_lambda: float,
    model_path: Path | None,
) -> np.ndarray:
    torch, nn, optim, LinearRegression, LlamaConfig, LlamaModel, nx = _require_neural_deps()
    np.random.seed(seed)
    torch.manual_seed(seed)

    class CausalDiscoveryModel(nn.Module):
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.config = LlamaConfig(
                hidden_size=512,
                intermediate_size=1024,
                num_hidden_layers=8,
                num_attention_heads=8,
                max_position_embeddings=512,
                vocab_size=32000,
            )
            self.llama = LlamaModel(self.config)
            for param in self.llama.parameters():
                param.requires_grad = False
            self.input_projection = nn.Linear(input_dim, self.config.hidden_size)
            self.output_projection = nn.Linear(self.config.hidden_size, output_dim)

        def forward(self, x):
            x = x.to(torch.float32)
            x = self.input_projection(x)
            outputs = self.llama(inputs_embeds=x)
            return self.output_projection(outputs.last_hidden_state)

    def ensure_acyclic(adj_matrix):
        if adj_matrix.dim() != 2:
            raise ValueError("ensure_acyclic expects a 2D adjacency matrix.")
        device = adj_matrix.device
        arr = adj_matrix.detach().cpu().numpy().copy()
        n = arr.shape[0]
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n))
        for i in range(n):
            for j in range(n):
                if i != j and arr[i, j] > 0:
                    graph.add_edge(i, j, weight=float(arr[i, j]))
        while True:
            try:
                cycle_edges = nx.find_cycle(graph, orientation="original")
            except nx.NetworkXNoCycle:
                break
            edge_to_remove = min(cycle_edges, key=lambda edge: graph[edge[0]][edge[1]]["weight"])
            u, v = edge_to_remove[0], edge_to_remove[1]
            graph.remove_edge(u, v)
            arr[u, v] = 0.0
        return torch.from_numpy(arr).float().to(device)

    def calculate_threshold(weights: np.ndarray, n_edges: int) -> float:
        nonzero = np.abs(weights[np.abs(weights) > 0]).reshape(-1)
        if nonzero.size == 0:
            return float("inf")
        sorted_weights = np.sort(nonzero)[::-1]
        idx = min(max(int(n_edges), 1), sorted_weights.size) - 1
        return float(sorted_weights[idx])

    def graph_pruned_by_coef(graph_batch: np.ndarray, x: np.ndarray) -> np.ndarray:
        d = int(graph_batch.shape[0])
        reg = LinearRegression()
        coeff_rows: list[np.ndarray] = []
        for child_idx in range(d):
            parent_mask = np.abs(graph_batch[child_idx]) > 0.5
            parent_mask[child_idx] = False
            if int(parent_mask.sum()) == 0:
                coeff_rows.append(np.zeros(d, dtype=float))
                continue
            reg.fit(x[:, parent_mask], x[:, child_idx])
            row = np.zeros(d, dtype=float)
            row[np.where(parent_mask)[0]] = reg.coef_
            coeff_rows.append(row)
        weights = np.asarray(coeff_rows, dtype=float)
        threshold = calculate_threshold(weights, d)
        pruned = ((np.abs(weights) >= threshold) & (np.abs(weights) > 0)).astype(np.float32)
        np.fill_diagonal(pruned, 0.0)
        return pruned

    class GraphEnvironment:
        def __init__(self, synthetic_data: np.ndarray):
            self.synthetic_data = synthetic_data
            self.current_state_index = 0

        def get_next_state(self) -> np.ndarray:
            self.current_state_index = (self.current_state_index + 1) % len(self.synthetic_data)
            return self.synthetic_data[self.current_state_index]

    class CausalDiscoveryLLM:
        def __init__(self, input_dim: int, output_dim: int, checkpoint: Path | None):
            self.model = CausalDiscoveryModel(input_dim, output_dim)
            self.optimizer = optim.Adam(self.model.parameters(), lr=2e-5)
            self.criterion = nn.BCELoss()
            self.checkpoint = checkpoint
            if checkpoint is not None and checkpoint.exists():
                self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
                self.model.eval()

        def learn(self, train_data: np.ndarray) -> None:
            env = GraphEnvironment(train_data)
            self.model.train()
            for _epoch in range(int(num_epochs)):
                for _ in range(int(batch_size)):
                    state = env.get_next_state()
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
                    action_probs = torch.sigmoid(self.model(state_tensor).squeeze(0).squeeze(0))
                    if np.random.rand() < epsilon:
                        action_probs = torch.rand_like(action_probs)
                    d = int(np.sqrt(action_probs.size(0)))
                    action_probs_dag = ensure_acyclic(action_probs.view(d, d))
                    target = torch.zeros_like(action_probs_dag.view(-1))
                    loss = self.criterion(action_probs_dag.view(-1), target)
                    l1_norm = sum(param.abs().sum() for param in self.model.parameters())
                    loss = loss + float(l1_lambda) * l1_norm
                    self.optimizer.zero_grad()
                    loss.backward()
                    for param in self.model.parameters():
                        if param.requires_grad and param.grad is not None:
                            param.grad.data.clamp_(-1, 1)
                    self.optimizer.step()
            if self.checkpoint is not None:
                self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), self.checkpoint)

        def causal_matrix(self, x: np.ndarray) -> np.ndarray:
            data_tensor = torch.tensor(x, dtype=torch.float32)
            src = data_tensor.mean(dim=0, keepdim=True).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                adj_output = self.model(src).squeeze(0).squeeze(0)
                d = x.shape[1]
                adj_probs = torch.sigmoid(adj_output).view(d, d)
                adj_probs = adj_probs * (1 - torch.eye(d, device=adj_probs.device))
                pruned = graph_pruned_by_coef(adj_probs.detach().cpu().numpy(), x)
                dag = ensure_acyclic(torch.tensor(pruned, dtype=torch.float32, device=adj_probs.device))
            return dag.cpu().numpy()

    input_dim = int(data.shape[1])
    learner = CausalDiscoveryLLM(input_dim=input_dim, output_dim=input_dim * input_dim, checkpoint=model_path)
    learner.learn(data)
    child_parent = learner.causal_matrix(data)
    prediction = child_parent.T.astype(int)
    np.fill_diagonal(prediction, 0)
    return prediction


def _prediction_row(
    *,
    method: str,
    model_name: str,
    sample_size_obs: int,
    answer: np.ndarray,
    prediction: np.ndarray,
    metadata: dict[str, Any],
    replicate_index: int,
    replicate_seed: int,
) -> dict[str, Any]:
    return {
        "method": method,
        "model": model_name,
        "provider": "local",
        "naming_regime": "data_only",
        "obs_n": sample_size_obs,
        "int_n": 0,
        "raw_response": json.dumps(metadata, ensure_ascii=False),
        "answer": json.dumps(np.asarray(answer, dtype=int).tolist(), ensure_ascii=False),
        "prediction": json.dumps(np.asarray(prediction, dtype=int).tolist(), ensure_ascii=False),
        "valid": 1,
        "replicate_index": replicate_index,
        "replicate_seed": replicate_seed,
    }


def _write_prediction_rows(*, out_csv: Path, rows: list[dict[str, Any]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _replicate_checkpoint_path(model_path: Path | None, replicate_index: int, num_replicates: int) -> Path | None:
    if model_path is None:
        return None
    if num_replicates <= 1:
        return model_path
    return model_path.with_name(f"{model_path.stem}_rep{replicate_index}{model_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the upstream-style trainable Causal-LLM data baseline.")
    parser.add_argument("--graph_files", type=str, nargs="+", required=True)
    parser.add_argument("--sample_size_obs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--method_name", default="CausalLLMDataNeural")
    parser.add_argument("--model_name", default="CausalDiscoveryLLM")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--num_replicates", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--l1_lambda", type=float, default=0.01)
    parser.add_argument("--out_csv", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.graph_files) > 1 and args.out_csv:
        raise SystemExit("--out_csv can only be used with a single --graph_files entry.")
    if args.num_replicates <= 0:
        raise SystemExit("--num_replicates must be > 0.")
    for graph_file in args.graph_files:
        graph_path = Path(graph_file).resolve()
        graph = load_causal_graph(graph_path)
        answer = np.asarray(graph.adj_matrix).astype(int)
        np.fill_diagonal(answer, 0)
        model_path = Path(args.model_path).resolve() if args.model_path else None
        out_csv = Path(args.out_csv) if args.out_csv else Path(args.out_dir) / graph_path.stem / (
            f"predictions_obs{args.sample_size_obs}_int0_{args.method_name}_seed{int(args.seed)}.csv"
        )
        rows: list[dict[str, Any]] = []
        for replicate_index in range(int(args.num_replicates)):
            replicate_seed = int(args.seed) + replicate_index * 1000
            data, _variables, row_indices = _load_observational_array(graph, args.sample_size_obs, replicate_seed)
            checkpoint = _replicate_checkpoint_path(model_path, replicate_index, int(args.num_replicates))
            print(
                f"[{args.method_name}] replicate {replicate_index + 1}/{args.num_replicates} "
                f"start seed={replicate_seed}",
                flush=True,
            )
            prediction = _fit_predict_causal_llm(
                data,
                seed=replicate_seed,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                epsilon=args.epsilon,
                l1_lambda=args.l1_lambda,
                model_path=checkpoint,
            )
            metadata = {
                "source": "devharish1371 Causal-LLM Dag_generation and model_evaluation/causal_llm.py",
                "implementation": "trainable frozen-LLaMA projection with acyclicity projection and regression pruning",
                "seed": replicate_seed,
                "base_seed": int(args.seed),
                "replicate_index": replicate_index,
                "num_replicates": int(args.num_replicates),
                "num_epochs": int(args.num_epochs),
                "batch_size": int(args.batch_size),
                "epsilon": float(args.epsilon),
                "l1_lambda": float(args.l1_lambda),
                "sample_size_obs": int(args.sample_size_obs),
                "sampled_row_indices": row_indices,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "output_orientation": "transposed from upstream child-parent regression convention to repo parent-child adjacency convention",
            }
            rows.append(
                _prediction_row(
                    method=args.method_name,
                    model_name=args.model_name,
                    sample_size_obs=args.sample_size_obs,
                    answer=answer,
                    prediction=prediction,
                    metadata=metadata,
                    replicate_index=replicate_index,
                    replicate_seed=replicate_seed,
                )
            )
            _write_prediction_rows(out_csv=out_csv, rows=rows)
            print(
                f"[{args.method_name}] replicate {replicate_index + 1}/{args.num_replicates} done; "
                f"rows={len(rows)}/{args.num_replicates}",
                flush=True,
            )
        print(f"[{args.method_name}] wrote {out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
