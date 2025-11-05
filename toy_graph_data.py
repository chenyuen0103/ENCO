# toy_graph_data.py
import numpy as np
import pandas as pd
import torch

from causal_graphs.graph_definition import CausalDAG, CausalVariable
from causal_graphs.variable_distributions import DiscreteProbDist, CategoricalDist, LeafCategDist
from causal_discovery.datasets import ObservationalCategoricalData

def make_random_dag(num_vars=5, edge_prob=0.3, num_categories=3):
    # --- Step 1: Create random DAG adjacency ---
    A = np.zeros((num_vars, num_vars), dtype=bool)
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            if np.random.rand() < edge_prob:
                A[i, j] = True

    # --- Step 2: Create variables with discrete probability distributions ---
    variables = []
    for i in range(num_vars):
        # Each variable has num_categories categories, uniform distribution initially
        prob_dist = CategoricalDist(num_categs=num_categories, prob_func=LeafCategDist(num_categs=num_categories))
        variables.append(CausalVariable(name=f"X{i}", prob_dist=prob_dist))

    # --- Step 3: Build graph ---
    graph = CausalDAG(variables=variables, adj_matrix=A)
    return graph

def main():
    # 1. Make random graph
    graph = make_random_dag(num_vars=6, edge_prob=0.4, num_categories=3)
    print(graph)

    # 2. Simulate observational dataset
    obs_dataset = ObservationalCategoricalData(graph, dataset_size=2000)
    print("Variables:", obs_dataset.var_names)
    print("Dataset shape:", obs_dataset.data.shape)

    # 3. Inspect a few samples
    for i in range(3):
        row = obs_dataset[i].tolist()
        print(f"Sample {i}:", dict(zip(obs_dataset.var_names, row)))

    # 4. Save tabular data as CSV
    arr = obs_dataset.data.cpu().numpy()
    df = pd.DataFrame(arr, columns=obs_dataset.var_names)
    df.to_csv("synthetic_obs.csv", index=False)
    print("Saved observational data to synthetic_obs.csv")

if __name__ == "__main__":
    main()
