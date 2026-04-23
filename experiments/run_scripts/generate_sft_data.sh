#!/usr/bin/env bash

# Re-exec under bash if launched via `sh ...`.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BIF_FILE="${BIF_FILE:-${REPO_ROOT}/causal_graphs/real_data/small_graphs/sachs.bif}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/experiments/configs/sachs_qwen_configs.json}"

NUM_PROMPTS="${NUM_PROMPTS:-20}"
TRAIN_SEED="${TRAIN_SEED:-42}"
EVAL_SEED="${EVAL_SEED:-1337}"

TRAIN_CSV="${TRAIN_CSV:-${REPO_ROOT}/experiments/data/grpo_sachs_train.csv}"
EVAL_CSV="${EVAL_CSV:-${REPO_ROOT}/experiments/data/sft_eval_child.csv}"

MIX_ANON_CSV="${MIX_ANON_CSV:-${REPO_ROOT}/experiments/data/grpo_mix_anon.csv}"
MIX_NAMED_CSV="${MIX_NAMED_CSV:-${REPO_ROOT}/experiments/data/grpo_mix_named.csv}"

REASONING_TARGET="${REASONING_TARGET:-stages}"
WRAPPER_MODE="${WRAPPER_MODE:-chat}"

TRAIN_ROWS_PER_SOURCE="${TRAIN_ROWS_PER_SOURCE:-1725}"
EVAL_ROWS_PER_SOURCE="${EVAL_ROWS_PER_SOURCE:-999999}"

TRAIN_JSONL="${TRAIN_JSONL:-${REPO_ROOT}/experiments/data/format_sft_stages_v4_mixed.jsonl}"
EVAL_JSONL="${EVAL_JSONL:-${REPO_ROOT}/experiments/data/sft_eval.jsonl}"

echo "============================================"
echo " Generating SFT data"
echo " Repo root:          ${REPO_ROOT}"
echo " BIF file:           ${BIF_FILE}"
echo " Config file:        ${CONFIG_FILE}"
echo " Train seed:         ${TRAIN_SEED}"
echo " Eval seed:          ${EVAL_SEED}"
echo " Reasoning target:   ${REASONING_TARGET}"
echo " Wrapper mode:       ${WRAPPER_MODE}"
echo " Train JSONL:        ${TRAIN_JSONL}"
echo " Eval JSONL:         ${EVAL_JSONL}"
echo "============================================"

cd "${REPO_ROOT}"

python experiments/export_cd_train_eval_csv.py \
  --bif-file "${BIF_FILE}" \
  --config-file "${CONFIG_FILE}" \
  --num-prompts "${NUM_PROMPTS}" \
  --train-seed "${TRAIN_SEED}" \
  --eval-seed "${EVAL_SEED}" \
  --train-csv "${TRAIN_CSV}" \
  --eval-csv "${EVAL_CSV}"

python experiments/collect_format_sft_data.py \
  --output "${TRAIN_JSONL}" \
  --csv "${MIX_ANON_CSV}:grpo_mix_anon" \
  --csv "${MIX_NAMED_CSV}:grpo_mix_named" \
  --prompt-col prompt_text \
  --answer-col answer \
  --reasoning-target "${REASONING_TARGET}" \
  --wrapper-mode "${WRAPPER_MODE}" \
  --n-per-source "${TRAIN_ROWS_PER_SOURCE}" \
  --seed "${TRAIN_SEED}"

python experiments/collect_format_sft_data.py \
  --output "${EVAL_JSONL}" \
  --csv "${EVAL_CSV}:sft_eval_child" \
  --prompt-col prompt_text \
  --answer-col answer \
  --reasoning-target "${REASONING_TARGET}" \
  --wrapper-mode "${WRAPPER_MODE}" \
  --n-per-source "${EVAL_ROWS_PER_SOURCE}" \
  --seed "${EVAL_SEED}"

echo ""
echo "Done."
echo "  train csv:  ${TRAIN_CSV}"
echo "  eval csv:   ${EVAL_CSV}"
echo "  train jsonl:${TRAIN_JSONL}"
echo "  eval jsonl: ${EVAL_JSONL}"
