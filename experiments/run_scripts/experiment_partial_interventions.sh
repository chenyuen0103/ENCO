#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}/experiments"
SEED="${SEED:-42}"

DONE_DIR="${REPO_ROOT}/experiments/checkpoints/enco_baselines/.done"
DONE_FILE="${DONE_DIR}/experiment_partial_interventions_seed${SEED}.done"
if [[ -f "${DONE_FILE}" && "${FORCE:-0}" != "1" ]]; then
  echo "[skip] partial-interventions baseline already done (${DONE_FILE}). Set FORCE=1 to rerun."
  exit 0
fi
mkdir -p "${DONE_DIR}"

# Change the value of 'max_inters' to the number of variables to use interventional data for
python run_exported_graphs.py --graph_files ../causal_graphs/synthetic_graphs_partial/*_42.npz \
                              --weight_decay 4e-5 \
                              --max_inters 10 \
                              --lambda_sparse 0.002 \
                              --seed "${SEED}"

touch "${DONE_FILE}"
