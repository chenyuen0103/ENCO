#!/usr/bin/env bash
set -euo pipefail
# Graphs are dynamically generated here to save disk memory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}/experiments"
SEED="${SEED:-42}"

DONE_DIR="${REPO_ROOT}/experiments/checkpoints/enco_baselines/.done"
DONE_FILE="${DONE_DIR}/experiment_scale_seed${SEED}.done"
if [[ -f "${DONE_FILE}" && "${FORCE:-0}" != "1" ]]; then
  echo "[skip] scale baseline already done (${DONE_FILE}). Set FORCE=1 to rerun."
  exit 0
fi
mkdir -p "${DONE_DIR}"

# Graph with 100 variables
python run_generated_graphs.py --cluster \
                               --graph_type random_max_10 \
                               --num_vars 100 \
                               --edge_prob 0.08 \
                               --lambda_sparse 0.01 \
                               --max_graph_stacking 50 \
                               --model_iters 2000 \
                               --use_theta_only_stage \
                               --theta_only_iters 1000 \
                               --sample_size_obs 100000 \
                               --sample_size_inters 4096 \
                               --seed "${SEED}"
# Graph with 200 variables
python run_generated_graphs.py --cluster \
                               --graph_type random_max_10 \
                               --num_vars 200 \
                               --edge_prob 0.04 \
                               --lambda_sparse 0.01 \
                               --max_graph_stacking 20 \
                               --model_iters 2000 \
                               --use_theta_only_stage \
                               --theta_only_iters 1000 \
                               --sample_size_obs 100000 \
                               --sample_size_inters 4096 \
                               --seed "${SEED}"
# Graph with 400 variables
python run_generated_graphs.py --cluster \
                               --graph_type random_max_10 \
                               --num_vars 400 \
                               --edge_prob 0.02 \
                               --lambda_sparse 0.01 \
                               --max_graph_stacking 10 \
                               --model_iters 4000 \
                               --use_theta_only_stage \
                               --theta_only_iters 2000 \
                               --sample_size_obs 100000 \
                               --sample_size_inters 4096 \
                               --seed "${SEED}"
# Graph with 1000 variables
python run_generated_graphs.py --cluster \
                               --graph_type random_max_10 \
                               --num_vars 1000 \
                               --edge_prob 0.008 \
                               --lambda_sparse 0.01 \
                               --max_graph_stacking 2 \
                               --batch_size 64 \
                               --GF_num_batches 2 \
                               --model_iters 4000 \
                               --use_theta_only_stage \
                               --theta_only_iters 2000 \
                               --sample_size_obs 100000 \
                               --sample_size_inters 4096 \
                               --seed "${SEED}"

touch "${DONE_FILE}"
