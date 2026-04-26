#!/usr/bin/env bash
set -u

cd /home/yuen_chen/ENCO

GPU_ID="${GPU_ID:-2}"
PYTHON_BIN="${PYTHON_BIN:-/home/yuen_chen/.conda/envs/enco/bin/python}"
RUN_DIR="${RUN_DIR:-experiments/checkpoints/qwen3_4b_sft_5way_v4_2gpu}"
EVAL_JSONL="${EVAL_JSONL:-experiments/data/format_sft_5way_v4_small.jsonl}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/reward_eval_rollouts}"
N="${N:-64}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-42}"

mkdir -p "$OUT_DIR"

models=(
  "$RUN_DIR/checkpoint-100"
  "$RUN_DIR/checkpoint-200"
  "$RUN_DIR/checkpoint-236"
  "$RUN_DIR"
)

for model in "${models[@]}"; do
  name="$(basename "$model")"
  if [[ "$model" == "$RUN_DIR" ]]; then
    name="final_adapter"
  fi

  out_jsonl="$OUT_DIR/${name}_rollouts.jsonl"
  log_path="$OUT_DIR/${name}_rollouts.log"

  echo "[$(date -Is)] [eval] $model -> $out_jsonl" | tee "$log_path"

  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
    experiments/eval_cd_rewards_on_jsonl.py \
    --model "$model" \
    --jsonl "$EVAL_JSONL" \
    --n "$N" \
    --seed "$SEED" \
    --num-rollouts "$NUM_ROLLOUTS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --dtype bf16 \
    --device-map auto \
    --output-jsonl "$out_jsonl" 2>&1 | tee -a "$log_path"

  status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "[$(date -Is)] [error] $model failed with status $status" | tee -a "$log_path"
  else
    echo "[$(date -Is)] [done] $model" | tee -a "$log_path"
  fi
done
