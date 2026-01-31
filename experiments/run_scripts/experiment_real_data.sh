#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}/experiments"
SEED="${SEED:-42}"

DONE_DIR="${REPO_ROOT}/experiments/checkpoints/enco_baselines/.done"
DONE_FILE="${DONE_DIR}/experiment_real_data_seed${SEED}.done"
if [[ -f "${DONE_FILE}" && "${FORCE:-0}" != "1" ]]; then
  echo "[skip] real-data baseline already done (${DONE_FILE}). Set FORCE=1 to rerun."
  exit 0
fi
mkdir -p "${DONE_DIR}"

# Small graphs with less than 100 variables
python run_exported_graphs.py --graph_files ../causal_graphs/real_data/small_graphs/*.bif \
                              --lambda_sparse 0.002 \
                              --num_epochs 100 \
                              --sample_size_obs 50000 \
                              --sample_size_inters 512 \
                              --seed "${SEED}"
# Large graphs with more than 100 variables
python run_exported_graphs.py --cluster \
                              --graph_files ../causal_graphs/real_data/large_graphs/*.bif \
                              --num_epochs 50 \
                              --lambda_sparse 0.02 \
                              --max_graph_stacking 10 \
                              --model_iters 4000 \
                              --use_theta_only_stage \
                              --theta_only_iters 2000 \
                              --sample_size_obs 100000 \
                              --sample_size_inters 4096 \
                              --seed "${SEED}"

touch "${DONE_FILE}"
