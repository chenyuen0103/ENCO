#!/usr/bin/env bash
set -euo pipefail

# Run one Qwen2.5 model across the four graph suite.
#
# Usage:
#   MODEL="Qwen/Qwen2.5-14B-Instruct-1M" bash experiments/run_scripts/run_qwen25_all_graphs.sh
#   MODEL="Qwen/Qwen2.5-7B-Instruct-1M"  GPU_ARG=5 bash experiments/run_scripts/run_qwen25_all_graphs.sh
#
# Notes:
# - Qwen2.5 long-context vLLM runs in this environment need DualChunk attention
#   and eager mode.
# - Set FORCE=1 to overwrite existing response CSVs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct-1M}"
CONFIG="${CONFIG:-./experiments/configs/eval_configs.json}"
GRAPHS_DIR="${GRAPHS_DIR:-causal_graphs/real_data/small_graphs}"
GRAPHS=(${GRAPHS:-cancer earthquake asia sachs})

GPU_ARG="${GPU_ARG:-4}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-128000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
FORCE="${FORCE:-0}"

EXTRA_ARGS=()
if [[ "${FORCE}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

echo "[info] Running Qwen2.5 model across graphs"
echo "[info] MODEL=${MODEL}"
echo "[info] GRAPHS=${GRAPHS[*]}"
echo "[info] GPU_ARG=${GPU_ARG} TP=${TP} MAX_MODEL_LEN=${MAX_MODEL_LEN}"

for graph in "${GRAPHS[@]}"; do
  echo "[run] graph=${graph} model=${MODEL}"
  VLLM_ATTENTION_BACKEND=DUAL_CHUNK_FLASH_ATTN \
  VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  CUDA_VISIBLE_DEVICES="${GPU_ARG}" python3 experiments/run_experiment1_in_memory.py \
    --bif-file "${GRAPHS_DIR}/${graph}.bif" \
    --config-file "${CONFIG}" \
    --model "${MODEL}" \
    --provider vllm \
    --temperature 0 \
    --vllm-dtype bf16 \
    --vllm-tensor-parallel-size "${TP}" \
    --vllm-max-new-tokens "${MAX_NEW_TOKENS}" \
    --vllm-batch-size "${BATCH_SIZE}" \
    --vllm-gpu-mem-util "${GPU_MEM_UTIL}" \
    --vllm-max-model-len "${MAX_MODEL_LEN}" \
    --vllm-enforce-eager \
    "${EXTRA_ARGS[@]}"
done

echo "[done] Completed ${MODEL}."
