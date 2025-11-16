import json
import os
from datetime import datetime
import sys
sys.path.append("../")
from pathlib import Path
from causal_graphs.graph_visualization import visualize_graph
from causal_graphs.graph_export import load_graph
from causal_graphs.graph_real_world import load_graph_file
from causal_graphs.graph_definition import CausalDAG
from causal_discovery.utils import set_cluster
from experiments.utils import set_seed, get_basic_parser, test_graph


if __name__ == '__main__':
    parser = get_basic_parser()
    parser.add_argument('--graph_files', type=str, nargs='+', default=["../causal_graphs/real_data/small_graphs/cancer.bif"],
                        help='Graph files to apply ENCO to. Files must be .pt, .npz, or .bif files.')
    args = parser.parse_args()

    # Basic checkpoint directory creation
    current_date = datetime.now()
    if args.checkpoint_dir is None or len(args.checkpoint_dir) == 0:
        checkpoint_dir = "checkpoints/%02d_%02d_%02d__%02d_%02d_%02d/" % (
            current_date.year, current_date.month, current_date.day, current_date.hour, current_date.minute, current_date.second)
    else:
        checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    set_cluster(args.cluster)

    for gindex, graph_path in enumerate(args.graph_files):
        # Seed setting for reproducibility
        set_seed(args.seed)
        # Load graph
        if graph_path.endswith(".bif"):
            graph = load_graph_file(graph_path)
        elif graph_path.endswith(".pt"):
            graph = CausalDAG.load_from_file(graph_path)
        elif graph_path.endswith(".npz"):
            graph = load_graph(graph_path)
        else:
            assert False, "Unknown file extension for " + graph_path
        graph_name = graph_path.split("/")[-1].rsplit(".", 1)[0]
        if graph_name.startswith("graph_"):
            graph_name = graph_name.split("graph_")[-1]
        file_id = "%s_%s" % (str(gindex+1).zfill(3), graph_name)
        # Visualize graph
        if graph.num_vars <= 100:
            figsize = max(3, graph.num_vars ** 0.7)
            visualize_graph(graph,
                            filename=os.path.join(checkpoint_dir, "graph_%s.pdf" % (file_id)),
                            figsize=(figsize, figsize),
                            layout="circular" if graph.num_vars < 40 else "graphviz")
        s = "== Testing graph \"%s\" ==" % graph_name
        print("="*len(s)+"\n"+s+"\n"+"="*len(s))
        # Start structure learning
        # test_graph(graph, args, checkpoint_dir, file_id)
                # Start structure learning (now returns predicted adjacency)
        A_pred, metrics = test_graph(graph, args, checkpoint_dir, file_id)  # (N,N) int

        # ---------- NEW: export predictions to CSV for evaluation ----------
        # ground truth adjacency
        A_true = graph.adj_matrix.astype(int)

        # derive obs / inter counts
        num_obs = int(args.sample_size_obs)
        if args.sample_size_inters <= 0:
            num_inters = 0
        else:
            if args.max_inters < 0:
                num_inters = graph.num_vars
            else:
                num_inters = int(args.max_inters)

        # we’ll only special-case the cancer graph here; adjust if you want more
        date_name = graph_path.split("/")[-1].rsplit(".", 1)[0]
        out_root = Path("responses/") / date_name
        out_root.mkdir(parents=True, exist_ok=True)

        out_csv = out_root / f"predictions_obs{num_obs}_int{args.sample_size_inters}_ENCO.csv"

        import csv
        row = {
            "method": "ENCO",
            "graph": graph_name,
            "num_obs": num_obs,
            "num_inters": num_inters,
            "answer": json.dumps(A_true.tolist(), ensure_ascii=False),
            "prediction": json.dumps(A_pred.tolist(), ensure_ascii=False),
            **metrics
        }
        fieldnames = list(row.keys())
        with out_csv.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

        print(f"[ENCO] Wrote predictions to: {out_csv.resolve()}")
