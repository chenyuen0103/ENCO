#!/usr/bin/env bash

# Re-exec under bash if launched via `sh ...`.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BIF_FILE="${BIF_FILE:-${REPO_ROOT}/causal_graphs/real_data/small_graphs/sachs.bif}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/experiments/configs/sft_eval.json}"

NUM_PROMPTS="${NUM_PROMPTS:-20}"
EVAL_SEED="${EVAL_SEED:-1337}"
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
echo " Eval seed:          ${EVAL_SEED}"
echo " Reasoning target:   ${REASONING_TARGET}"
echo " Wrapper mode:       ${WRAPPER_MODE}"
echo " Train JSONL:        ${TRAIN_JSONL}"
echo " Eval JSONL:         ${EVAL_JSONL}"
echo "============================================"

cd "${REPO_ROOT}"

python experiments/generate_prompt_answer_csv.py \
  --config-file "${CONFIG_FILE}" \
  --graphs-dir "$(dirname "${BIF_FILE}")" \
  --graph-names "$(basename "${BIF_FILE}" .bif)" \
  --num-prompts-per-config "${NUM_PROMPTS}" \
  --seed "${EVAL_SEED}" \
  --output-csv "${EVAL_CSV}"

python experiments/generate_reasoning.py \
  --output "${TRAIN_JSONL}" \
  --csv "${MIX_ANON_CSV}:grpo_mix_anon" \
  --csv "${MIX_NAMED_CSV}:grpo_mix_named" \
  --prompt-col prompt_text \
  --answer-col answer \
  --reasoning-target "${REASONING_TARGET}" \
  --wrapper-mode "${WRAPPER_MODE}" \
  --n-per-source "${TRAIN_ROWS_PER_SOURCE}" \
  --seed 42

python experiments/generate_reasoning.py \
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
echo "  eval csv:   ${EVAL_CSV}"
echo "  train jsonl:${TRAIN_JSONL}"
echo "  eval jsonl: ${EVAL_JSONL}"
