import json
import os
import csv
from datetime import datetime
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
import numpy as np
from causal_graphs.graph_export import load_graph
from causal_graphs.graph_real_world import load_graph_file
from causal_graphs.graph_definition import CausalDAG
from causal_discovery.utils import set_cluster
from experiments.utils import set_seed, get_basic_parser, test_graph


def _read_completed_replicate_rows(out_csv: Path) -> dict[int, dict]:
    if not out_csv.exists():
        return {}
    with out_csv.open("r", encoding="utf-8", newline="") as handle:
        rows: dict[int, dict] = {}
        for row in csv.DictReader(handle):
            try:
                rep = int(row.get("replicate_index", ""))
            except Exception:
                continue
            rows[rep] = dict(row)
        return rows


def _write_prediction_rows(out_csv: Path, rows: list[dict]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with out_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = get_basic_parser()
    parser.add_argument('--graph_files', type=str, nargs='+', default=["../causal_graphs/real_data/small_graphs/asia.bif"],
                        help='Graph files to apply ENCO to. Files must be .pt, .npz, or .bif files.')
    parser.add_argument('--num_prompts', type=int, default=1,
                        help='Number of independently seeded data replicates to evaluate.')
    args = parser.parse_args()
    if args.num_prompts <= 0:
        raise SystemExit("--num_prompts must be > 0.")
    base_seed = int(args.seed)

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
        graph_name = graph_path.split("/")[-1].rsplit(".", 1)[0]
        date_name = graph_name
        if graph_name.startswith("graph_"):
            graph_name = graph_name.split("graph_")[-1]
        num_obs = int(args.sample_size_obs)
        out_root = Path(__file__).resolve().parent / "responses" / date_name
        out_csv = out_root / f"predictions_obs{num_obs}_int{args.sample_size_inters}_ENCO_seed{base_seed}.csv"
        rows_by_replicate = _read_completed_replicate_rows(out_csv)

        for prompt_idx in range(args.num_prompts):
            if prompt_idx in rows_by_replicate:
                print(f"[ENCO] replicate {prompt_idx + 1}/{args.num_prompts} already present in {out_csv}; skipping")
                continue
            seed_i = base_seed + prompt_idx * 1000
            args.seed = seed_i
            set_seed(seed_i)
            if graph_path.endswith(".bif"):
                graph = load_graph_file(graph_path)
            elif graph_path.endswith(".pt"):
                graph = CausalDAG.load_from_file(graph_path)
            elif graph_path.endswith(".npz"):
                graph = load_graph(graph_path)
            else:
                assert False, "Unknown file extension for " + graph_path
            file_id = "%s_%s_rep%s" % (str(gindex+1).zfill(3), graph_name, prompt_idx)

            A_true = graph.adj_matrix.astype(int)
            np.fill_diagonal(A_true, 0)
            n = int(graph.num_vars)
            e = int(A_true.sum())
            denom = max(1, n * (n - 1))
            density = float(e) / float(denom)
            print(
                f"[info] Graph stats: name={graph_name} rep={prompt_idx} seed={seed_i} "
                f"nodes={n} edges={e} density={density:.4f} file={graph_path}"
            )

            if graph.num_vars <= 100:
                figsize = max(3, graph.num_vars ** 0.7)
                try:
                    from causal_graphs.graph_visualization import visualize_graph  # lazy import (matplotlib)

                    visualize_graph(graph,
                                    filename=os.path.join(checkpoint_dir, "graph_%s.pdf" % (file_id)),
                                    figsize=(figsize, figsize),
                                    layout="circular" if graph.num_vars < 40 else "graphviz")
                except Exception as e:
                    print(f"[warn] Skipping graph visualization (failed to import/render): {e}", file=sys.stderr)
            s = '== Testing graph "%s" replicate %s/%s ==' % (graph_name, prompt_idx + 1, args.num_prompts)
            print("="*len(s)+"\n"+s+"\n"+"="*len(s))
            A_pred, metrics = test_graph(graph, args, checkpoint_dir, file_id)  # (N,N) int

            if args.sample_size_inters <= 0:
                num_inters = 0
            elif args.max_inters < 0:
                num_inters = graph.num_vars
            else:
                num_inters = int(args.max_inters)

            rows_by_replicate[prompt_idx] = {
                "method": "ENCO",
                "graph": graph_name,
                "num_obs": num_obs,
                "num_inters": num_inters,
                "answer": json.dumps(A_true.tolist(), ensure_ascii=False),
                "prediction": json.dumps(A_pred.tolist(), ensure_ascii=False),
                "replicate_index": prompt_idx,
                "replicate_seed": seed_i,
                **metrics
            }
            _write_prediction_rows(out_csv, [rows_by_replicate[idx] for idx in sorted(rows_by_replicate)])

        print(f"[ENCO] Wrote predictions to: {out_csv.resolve()}")
