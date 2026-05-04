#!/usr/bin/env bash
set -euo pipefail

# Run one configured model across cancer, earthquake, asia, and sachs.
#
# Usage:
#   MODEL="Qwen/Qwen2.5-14B-Instruct-1M" bash experiments/run_scripts/run_model_all_graphs.sh
#   MODEL="meta-llama/Llama-3.1-70B-Instruct" GPU_ARG=4,5 TP=2 FORCE=1 bash experiments/run_scripts/run_model_all_graphs.sh
#
# Defaults come from experiments/configs/model_context_windows.json:
# - recommended_max_model_len
# - vLLM env vars
# - vLLM flags

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${MODEL:-${1:-}}"
if [[ -z "${MODEL}" ]]; then
  echo "Usage: MODEL='model/id' bash $0" >&2
  echo "   or: bash $0 'model/id'" >&2
  exit 2
fi

MODEL_CONTEXT_FILE="${MODEL_CONTEXT_FILE:-./experiments/configs/model_context_windows.json}"
CONFIG="${CONFIG:-./experiments/configs/eval_configs.json}"
GRAPHS_DIR="${GRAPHS_DIR:-causal_graphs/real_data/small_graphs}"
GRAPHS=(${GRAPHS:-cancer earthquake asia sachs})

GPU_ARG="${GPU_ARG:-4}"
TP="${TP:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
FORCE="${FORCE:-0}"

readarray -t MODEL_META < <(
  MODEL="${MODEL}" MODEL_CONTEXT_FILE="${MODEL_CONTEXT_FILE}" python3 - <<'PY'
import json
import os
import shlex
import sys
from pathlib import Path

model_id = os.environ["MODEL"]
path = Path(os.environ["MODEL_CONTEXT_FILE"])
data = json.loads(path.read_text())
entry = next((m for m in data.get("models", []) if m.get("model") == model_id), None)
if entry is None:
    known = "\n".join(f"  - {m.get('model')}" for m in data.get("models", []))
    raise SystemExit(f"Unknown model in {path}: {model_id}\nKnown models:\n{known}")

provider = entry.get("provider") or "vllm"
max_len = entry.get("recommended_max_model_len")
if provider == "vllm" and max_len is None:
    max_len = entry.get("advertised_context_window_tokens")
if provider == "vllm" and max_len is None:
    raise SystemExit(f"Model {model_id} has no recommended or advertised context window.")

env = entry.get("vllm_env") or {}
flags = entry.get("vllm_flags") or []
print(f"PROVIDER={shlex.quote(str(provider))}")
print(f"MAX_MODEL_LEN={shlex.quote(str(max_len or ''))}")
print("MODEL_ENV_PREFIX=" + shlex.quote(" ".join(f"{k}={v}" for k, v in env.items())))
print("MODEL_FLAGS=" + shlex.quote(" ".join(str(flag) for flag in flags)))
PY
)

for line in "${MODEL_META[@]}"; do
  eval "${line}"
done

EXTRA_ARGS=()
if [[ "${FORCE}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

if [[ "${PROVIDER}" != "vllm" ]]; then
  echo "[error] This launcher currently supports vLLM models only; ${MODEL} has provider=${PROVIDER}." >&2
  exit 2
fi

echo "[info] MODEL=${MODEL}"
echo "[info] GRAPHS=${GRAPHS[*]}"
echo "[info] GPU_ARG=${GPU_ARG} TP=${TP}"
echo "[info] MAX_MODEL_LEN=${MAX_MODEL_LEN} MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "[info] MODEL_ENV_PREFIX=${MODEL_ENV_PREFIX:-<none>}"
echo "[info] MODEL_FLAGS=${MODEL_FLAGS:-<none>}"

for graph in "${GRAPHS[@]}"; do
  echo "[run] graph=${graph} model=${MODEL}"
  # shellcheck disable=SC2086
  env ${MODEL_ENV_PREFIX} CUDA_VISIBLE_DEVICES="${GPU_ARG}" \
    python3 experiments/run_experiment1_in_memory.py \
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
      ${MODEL_FLAGS} \
      "${EXTRA_ARGS[@]}"
done

echo "[done] Completed ${MODEL}."
