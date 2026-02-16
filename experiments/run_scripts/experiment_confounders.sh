#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}/experiments"
SEED="${SEED:-42}"

DONE_DIR="${REPO_ROOT}/experiments/checkpoints/enco_baselines/.done"
DONE_FILE="${DONE_DIR}/experiment_confounders_seed${SEED}.done"
if [[ -f "${DONE_FILE}" && "${FORCE:-0}" != "1" ]]; then
  echo "[skip] confounders baseline already done (${DONE_FILE}). Set FORCE=1 to rerun."
  exit 0
fi
mkdir -p "${DONE_DIR}"

python run_exported_graphs.py --graph_files ../causal_graphs/confounder_graphs/*.npz \
                              --sample_size_obs 5000 \
                              --sample_size_inters 512 \
                              --weight_decay 1e-4 \
                              --seed "${SEED}"

touch "${DONE_FILE}"
