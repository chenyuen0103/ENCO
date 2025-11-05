import pandas as pd
import torch
import numpy as np

from causal_discovery.enco import ENCO
from causal_discovery.datasets import ObservationalCategoricalData
from causal_graphs.graph_definition import CausalDAG, CausalVariable, CausalDAGDataset
from causal_graphs.variable_distributions import CategoricalDist, LeafCategDist
from causal_graphs.graph_visualization import visualize_graph

def main():
    # 1. Load data from CSV
    df = pd.read_csv("synthetic_obs.csv")
    data = torch.from_numpy(df.values).long()
    var_names = df.columns.tolist()
    num_vars = len(var_names)
    
    # Create a dataset object
    # We need to create a dummy graph object to initialize the dataset
    # The actual graph structure is unknown to the discovery algorithm.
    variables = []
    for i in range(num_vars):
        # The number of categories can be inferred from the data
        num_categories = len(df.iloc[:, i].unique())
        prob_dist = CategoricalDist(num_categs=num_categories, prob_func=LeafCategDist(num_categs=num_categories))
        variables.append(CausalVariable(name=var_names[i], prob_dist=prob_dist))
    
    # Create a dummy adjacency matrix (all zeros - no edges initially) 
    dummy_adj_matrix = np.zeros((num_vars, num_vars), dtype=np.int32)
    
    # Create minimal dummy interventional data (1 sample per variable)
    # ENCO expects some interventional data, so we create minimal dummy data
    dummy_data_int = np.zeros((num_vars, 1, num_vars), dtype=np.int32)
    
    # Create dataset graph with the loaded data
    discovery_graph = CausalDAGDataset(
        adj_matrix=dummy_adj_matrix,
        data_obs=data.numpy().astype(np.int32),
        data_int=dummy_data_int
    )

    # 2. Initialize ENCO
    discovery_model = ENCO(
        graph=discovery_graph,
        hidden_dims=[64],
        lr_model=5e-3,
        lr_gamma=2e-2,
        lr_theta=1e-1,
        model_iters=1000,
        graph_iters=100,
        batch_size=128,
        GF_num_graphs=50,
        lambda_sparse=0.004
    )

    # 3. Run causal discovery
    discovery_model.discover_graph(num_epochs=10)

    # 4. Get and print the discovered graph
    discovered_adj_matrix = discovery_model.get_binary_adjmatrix().detach().cpu().numpy()
    
    print("Discovered Adjacency Matrix:")
    print(discovered_adj_matrix)

    # 5. Visualize the discovered graph
    pred_graph = CausalDAGDataset(
        adj_matrix=discovered_adj_matrix.astype(np.int32),
        data_obs=data.numpy().astype(np.int32),
        data_int=np.zeros((num_vars, 1, num_vars), dtype=np.int32)
    )
    visualize_graph(pred_graph, filename="discovered_graph.svg")
    print("Saved discovered graph to discovered_graph.svg")

if __name__ == "__main__":
    main()
