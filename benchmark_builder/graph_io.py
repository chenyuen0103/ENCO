from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .schema import DatasetSpec


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_causal_graph(path: str | Path):
    graph_path = Path(path)
    suffix = graph_path.suffix.lower()
    if suffix == ".bif":
        from causal_graphs.graph_real_world import load_graph_file

        return load_graph_file(str(graph_path))
    if suffix == ".pt":
        from causal_graphs.graph_definition import CausalDAG

        return CausalDAG.load_from_file(str(graph_path))
    if suffix == ".npz":
        from causal_graphs.graph_export import load_graph as load_dataset_graph

        return load_dataset_graph(str(graph_path))
    raise ValueError(f"Unsupported graph file type: {graph_path}")


def materialize_graph_source(dataset: DatasetSpec, *, repo_root: Path, run_root: Path, seed: int, dry_run: bool) -> Path:
    if dataset.graph_source != "synthetic":
        graph_path = Path(dataset.graph_path)
        if not graph_path.is_absolute():
            graph_path = (repo_root / graph_path).resolve()
        return graph_path

    graph_dir = run_root / "graphs"
    graph_path = graph_dir / f"{dataset.name}.pt"
    if graph_path.exists() or dry_run:
        return graph_path
    graph_dir.mkdir(parents=True, exist_ok=True)

    from causal_graphs.graph_export import create_graph

    params: dict[str, Any] = dict(dataset.graph_params)
    graph = create_graph(
        num_vars=int(params.get("num_vars", 10)),
        num_categs=int(params.get("num_categs", 10)),
        edge_prob=float(params.get("edge_prob", 0.3)),
        graph_type=params.get("graph_type", "random"),
        num_latents=int(params.get("num_latents", 0)),
        deterministic=bool(params.get("deterministic", False)),
        seed=int(params.get("seed", seed)),
    )
    graph.save_to_file(str(graph_path))
    return graph_path
