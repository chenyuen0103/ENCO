#!/usr/bin/env bash
set -euo pipefail

# Run all non-Qwen2.5 models from the current eval sweep across the graph suite.
#
# Usage:
#   bash experiments/run_scripts/run_non_qwen25_all_graphs.sh
#
# Set FORCE=1 to overwrite existing response CSVs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-./experiments/configs/eval_configs.json}"
GRAPHS_DIR="${GRAPHS_DIR:-causal_graphs/real_data/small_graphs}"
GRAPHS=(${GRAPHS:-cancer earthquake asia sachs})

MAX_MODEL_LEN="${MAX_MODEL_LEN:-258048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
FORCE="${FORCE:-0}"

SMALL_GPU_A="${SMALL_GPU_A:-4}"
SMALL_GPU_B="${SMALL_GPU_B:-5}"
LARGE_GPU_ARG="${LARGE_GPU_ARG:-4,5}"

small_models=(
  "Qwen/Qwen3-4B-Thinking-2507"
  "meta-llama/Meta-Llama-3.1-8B"
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

large_models=(
  "meta-llama/Llama-3.1-70B-Instruct"
  "Qwen/Qwen3-30B-A3B-Thinking-2507"
)

EXTRA_ARGS=()
if [[ "${FORCE}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

run_model() {
  local gpu_arg="$1"
  local tp="$2"
  local model="$3"
  local graph="$4"

  echo "[run] graph=${graph} model=${model} gpu=${gpu_arg} tp=${tp}"
  CUDA_VISIBLE_DEVICES="${gpu_arg}" python3 experiments/run_experiment1_in_memory.py \
    --bif-file "${GRAPHS_DIR}/${graph}.bif" \
    --config-file "${CONFIG}" \
    --model "${model}" \
    --provider vllm \
    --temperature 0 \
    --vllm-dtype bf16 \
    --vllm-tensor-parallel-size "${tp}" \
    --vllm-max-new-tokens "${MAX_NEW_TOKENS}" \
    --vllm-batch-size "${BATCH_SIZE}" \
    --vllm-gpu-mem-util "${GPU_MEM_UTIL}" \
    --vllm-max-model-len "${MAX_MODEL_LEN}" \
    "${EXTRA_ARGS[@]}"
}

echo "[info] Running non-Qwen2.5 models across graphs"
echo "[info] GRAPHS=${GRAPHS[*]}"
echo "[info] small GPUs=${SMALL_GPU_A},${SMALL_GPU_B}; large GPUs=${LARGE_GPU_ARG}"

for graph in "${GRAPHS[@]}"; do
  for ((i=0; i<${#small_models[@]}; i+=2)); do
    run_model "${SMALL_GPU_A}" 1 "${small_models[i]}" "${graph}" &
    pid1=$!

    if (( i + 1 < ${#small_models[@]} )); then
      run_model "${SMALL_GPU_B}" 1 "${small_models[i+1]}" "${graph}" &
      pid2=$!
      wait "${pid1}" "${pid2}"
    else
      wait "${pid1}"
    fi
  done

  for model in "${large_models[@]}"; do
    run_model "${LARGE_GPU_ARG}" 2 "${model}" "${graph}"
  done
done

echo "[done] Completed non-Qwen2.5 model sweep."
