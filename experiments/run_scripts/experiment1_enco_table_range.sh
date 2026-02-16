#!/usr/bin/env bash
set -euo pipefail

# ENCO baselines aligned with Experiment 1 "table range" (Data Volume grid).
#
# Defaults match experiments/generate_prompt_files.py:
#   N (obs)  : 0,100,1000,5000,8000
#   M (int)  : 0,50,100,200,500   (samples per intervention)
#
# Usage (from repo root or anywhere):
#   SEED=0 bash experiments/run_scripts/experiment1_enco_table_range.sh
#
# Override a single graph:
#   BIF_FILE=causal_graphs/real_data/small_graphs/sachs.bif SEED=0 \
#     bash experiments/run_scripts/experiment1_enco_table_range.sh
#
# Run over ALL real_data graphs (small + large):
#   ALL_REAL_DATA=1 SEED=0 bash experiments/run_scripts/experiment1_enco_table_range.sh
#   # or restrict:
#   ALL_REAL_DATA=1 REAL_DATA_SCOPE=small SEED=0 bash experiments/run_scripts/experiment1_enco_table_range.sh
#
# Resume:
# - Skips a run if the expected predictions CSV already exists, unless FORCE=1.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}/experiments"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "[error] Python cannot import 'torch' (PyTorch), which ENCO requires." >&2
  echo "[hint] Activate/install the ENCO environment (see ${REPO_ROOT}/environment.yml or ${REPO_ROOT}/requirements.txt)." >&2
  exit 2
fi

SEED="${SEED:-42}"
FORCE="${FORCE:-0}"

ALL_REAL_DATA="${ALL_REAL_DATA:-0}"
REAL_DATA_SCOPE="${REAL_DATA_SCOPE:-all}" # small|large|all (only used when ALL_REAL_DATA=1)

# ENCO hyperparameters:
# We default to the same settings as the paper scripts for small vs large graphs.
# You can override any of these via env vars to reduce memory usage / runtime.
N_LARGE_THRESHOLD="${N_LARGE_THRESHOLD:-100}"

# Small-graph defaults (matches experiment_real_data.sh small-graphs block)
NUM_EPOCHS_SMALL="${NUM_EPOCHS_SMALL:-100}"
LAMBDA_SPARSE_SMALL="${LAMBDA_SPARSE_SMALL:-0.002}"
BATCH_SIZE_SMALL="${BATCH_SIZE_SMALL:-128}"
HIDDEN_SIZE_SMALL="${HIDDEN_SIZE_SMALL:-64}"
GF_NUM_BATCHES_SMALL="${GF_NUM_BATCHES_SMALL:-1}"
MAX_GRAPH_STACKING_SMALL="${MAX_GRAPH_STACKING_SMALL:-50}"
MODEL_ITERS_SMALL="${MODEL_ITERS_SMALL:-1000}"

# Large-graph defaults (matches experiment_real_data.sh large-graphs block, but with safer memory defaults)
NUM_EPOCHS_LARGE="${NUM_EPOCHS_LARGE:-50}"
LAMBDA_SPARSE_LARGE="${LAMBDA_SPARSE_LARGE:-0.02}"
BATCH_SIZE_LARGE="${BATCH_SIZE_LARGE:-64}"
HIDDEN_SIZE_LARGE="${HIDDEN_SIZE_LARGE:-32}"
GF_NUM_BATCHES_LARGE="${GF_NUM_BATCHES_LARGE:-2}"
MAX_GRAPH_STACKING_LARGE="${MAX_GRAPH_STACKING_LARGE:-5}"
MODEL_ITERS_LARGE="${MODEL_ITERS_LARGE:-4000}"
THETA_ONLY_ITERS_LARGE="${THETA_ONLY_ITERS_LARGE:-2000}"

_run_sweep_for_bif() {
  local bif_file="$1"
  local dataset
  dataset="$(basename "${bif_file}" .bif)"

  echo "[info] ---- table-range sweep graph=${dataset} ----"

  local n_vars n_edges
  read -r n_vars n_edges < <(
    python - <<PY
import sys
import numpy as np
sys.path.insert(0, r"${REPO_ROOT}")
from causal_graphs.graph_real_world import load_graph_file
g = load_graph_file("${bif_file}")
A = g.adj_matrix.astype(int)
np.fill_diagonal(A, 0)
print(int(g.num_vars), int(A.sum()))
PY
  )
  echo "[info] graph=${dataset} nodes=${n_vars} edges=${n_edges}"

  local -a EXTRA_ARGS=()
  if [[ "${n_vars}" -gt "${N_LARGE_THRESHOLD}" ]]; then
    EXTRA_ARGS+=(
      --cluster
      --num_epochs "${NUM_EPOCHS_LARGE}"
      --lambda_sparse "${LAMBDA_SPARSE_LARGE}"
      --batch_size "${BATCH_SIZE_LARGE}"
      --hidden_size "${HIDDEN_SIZE_LARGE}"
      --GF_num_batches "${GF_NUM_BATCHES_LARGE}"
      --max_graph_stacking "${MAX_GRAPH_STACKING_LARGE}"
      --model_iters "${MODEL_ITERS_LARGE}"
      --use_theta_only_stage
      --theta_only_iters "${THETA_ONLY_ITERS_LARGE}"
    )
  else
    EXTRA_ARGS+=(
      --num_epochs "${NUM_EPOCHS_SMALL}"
      --lambda_sparse "${LAMBDA_SPARSE_SMALL}"
      --batch_size "${BATCH_SIZE_SMALL}"
      --hidden_size "${HIDDEN_SIZE_SMALL}"
      --GF_num_batches "${GF_NUM_BATCHES_SMALL}"
      --max_graph_stacking "${MAX_GRAPH_STACKING_SMALL}"
      --model_iters "${MODEL_ITERS_SMALL}"
    )
  fi

  local N M OUT_CSV CKPT_DIR
  for N in "${OBS_SIZES[@]}"; do
    for M in "${INT_SIZES[@]}"; do
      # Skip the empty-data case
      if [[ "${N}" == "0" && "${M}" == "0" ]]; then
        continue
      fi

      OUT_CSV="responses/${dataset}/predictions_obs${N}_int${M}_ENCO.csv"
      if [[ -f "${OUT_CSV}" && "${FORCE}" != "1" ]]; then
        echo "[skip] exists: ${OUT_CSV}"
        continue
      fi

      CKPT_DIR="checkpoints/enco_table_range/${dataset}/obs${N}_int${M}_seed${SEED}"
      echo "[run] dataset=${dataset} obs=${N} int=${M} seed=${SEED}"
      python run_exported_graphs.py \
        --graph_files "${bif_file}" \
        --sample_size_obs "${N}" \
        --sample_size_inters "${M}" \
        --max_inters -1 \
        --seed "${SEED}" \
        --checkpoint_dir "${CKPT_DIR}" \
        "${EXTRA_ARGS[@]}"
    done
  done
}

OBS_SIZES=(${OBS_SIZES:-0 100 1000 5000 8000})
INT_SIZES=(${INT_SIZES:-0 50 100 200 500})

echo "[info] ENCO table-range sweep: seed=${SEED} force=${FORCE}"
if [[ "${ALL_REAL_DATA}" == "1" ]]; then
  echo "[info] ALL_REAL_DATA=1 scope=${REAL_DATA_SCOPE}"
else
  echo "[info] Single-graph mode (set ALL_REAL_DATA=1 to sweep all real_data graphs)"
fi

if [[ "${ALL_REAL_DATA}" == "1" ]]; then
  declare -a bif_files=()
  if [[ "${REAL_DATA_SCOPE}" == "small" || "${REAL_DATA_SCOPE}" == "all" ]]; then
    for f in "${REPO_ROOT}"/causal_graphs/real_data/small_graphs/*.bif; do
      [[ -f "${f}" ]] && bif_files+=("${f}")
    done
  fi
  if [[ "${REAL_DATA_SCOPE}" == "large" || "${REAL_DATA_SCOPE}" == "all" ]]; then
    for f in "${REPO_ROOT}"/causal_graphs/real_data/large_graphs/*.bif; do
      [[ -f "${f}" ]] && bif_files+=("${f}")
    done
  fi

  if [[ "${#bif_files[@]}" -eq 0 ]]; then
    echo "[error] No .bif files found under causal_graphs/real_data/{small_graphs,large_graphs}." >&2
    echo "[hint] You may need to run ./download_datasets.sh first." >&2
    exit 2
  fi

  for f in "${bif_files[@]}"; do
    _run_sweep_for_bif "${f}"
  done
else
  # Path can be relative to repo root or absolute.
  BIF_FILE="${BIF_FILE:-causal_graphs/real_data/small_graphs/cancer.bif}"
  if [[ "${BIF_FILE}" != /* ]]; then
    BIF_FILE="${REPO_ROOT}/${BIF_FILE}"
  fi
  if [[ ! -f "${BIF_FILE}" ]]; then
    echo "[error] BIF_FILE not found: ${BIF_FILE}" >&2
    exit 2
  fi

  _run_sweep_for_bif "${BIF_FILE}"
fi

echo "[done] ENCO table-range sweep complete."
