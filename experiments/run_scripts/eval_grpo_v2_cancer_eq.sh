#!/usr/bin/env bash
# eval_grpo_v2_cancer_eq.sh
#
# Evaluate grpo_v2_cancer_eq on Sachs at two obs/int configurations:
#   (1) obs=5000, int=200
#   (2) obs=1000, int=50
#
# Results are appended to experiments/out/experiment1/sachs_summary.csv
# for comparison in collect_result.ipynb.
#
# Usage (from repo root):
#   bash experiments/run_scripts/eval_grpo_v2_cancer_eq.sh
#
# Set NPROC to use multiple GPUs (default: 1)
#   NPROC=2 bash experiments/run_scripts/eval_grpo_v2_cancer_eq.sh

# Re-exec under bash if the script was launched via `sh ...`.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXPERIMENTS_DIR="${REPO_ROOT}/experiments"

# MODEL="${EXPERIMENTS_DIR}/checkpoints/grpo_v2_cancer_eq" 
MODEL="Qwen/Qwen3-4B-Thinking-2507" 
BIF="${REPO_ROOT}/causal_graphs/real_data/small_graphs/sachs.bif"
RESPONSES_ROOT="${EXPERIMENTS_DIR}/responses"
SUMMARY_CSV="${EXPERIMENTS_DIR}/out/experiment1/sachs_summary.csv"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
RUN_EXPERIMENT_SCRIPT="${EXPERIMENTS_DIR}/run_experiment1_in_memory.py"
EVALUATE_SCRIPT="${EXPERIMENTS_DIR}/evaluate.py"
CONFIG_FILE="${SCRIPT_DIR}/configs/grpo_v2_cancer_eq_sachs_both_anon.json"
CONFIG_COT_HINT="$(
python - "${CONFIG_FILE}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)

cfgs = payload.get("configs") if isinstance(payload, dict) else payload
vals = {bool(c.get("cot_hint", False)) for c in cfgs if isinstance(c, dict)}
if vals == {True}:
    print("1")
elif vals in ({False}, set()):
    print("0")
else:
    print("mixed")
PY
)"

if [[ "${CONFIG_COT_HINT}" == "mixed" ]]; then
    echo "[error] Mixed cot_hint values in ${CONFIG_FILE}. This wrapper expects a consistent setting across configs." >&2
    exit 1
fi

COT_HINT_FLAG=()
COT_HINT_TAG=""
if [[ "${CONFIG_COT_HINT}" == "1" ]]; then
    COT_HINT_TAG="cothint_"
fi

# ── Helper: evaluate one generated CSV ────────────────────────────────────
# anon_flag: "" for real variable names, "--anonymize" for X1/X2/... names
eval_only() {
    local obs="$1"
    local int_n="$2"
    local anon_flag="${3:-}"   # optional: "--anonymize"

    local anon_label=""
    local anon_suffix=""
    if [[ "${anon_flag}" == "--anonymize" ]]; then
        anon_label=" [anonymized]"
        anon_suffix="anon_"
    fi

    echo ""
    echo "--- Config: obs=${obs}, int=${int_n}${anon_label} ---"

    if [[ ! -f "${EVALUATE_SCRIPT}" ]]; then
        echo "[error] Could not find evaluate.py at: ${EVALUATE_SCRIPT}" >&2
        return 1
    fi

    # Derive the exact filename that run_experiment1_in_memory.py writes.
    local resp_csv="${RESPONSES_ROOT}/sachs/responses_obs${obs}_int${int_n}_shuf1_p${NUM_PROMPTS}_${anon_suffix}thinktags_${COT_HINT_TAG}summary_joint_grpo_v2_cancer_eq.csv"

    echo "[eval] Computing metrics: ${resp_csv}"
    python "${EVALUATE_SCRIPT}" \
        --csv "${resp_csv}" \
        --summary-csv "${SUMMARY_CSV}"

    echo "[done] obs=${obs}, int=${int_n}${anon_label}"
}

main() {
    echo "============================================"
    echo " Evaluating: grpo_v2_cancer_eq"
    echo " Model:      ${MODEL}"
    echo " BIF:        ${BIF}"
    echo " Prompts:    ${NUM_PROMPTS}"
    echo " CoT Hint:   ${CONFIG_COT_HINT}"
    echo " Config:     ${CONFIG_FILE}"
    echo "============================================"

    if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo "[error] Could not find config file at: ${CONFIG_FILE}" >&2
        return 1
    fi

    echo ""
    echo "--- Running combined anonymized + non-anonymized batch via --config-file ---"
    python "${RUN_EXPERIMENT_SCRIPT}" \
        --bif-file "${BIF}" \
        --model "${MODEL}" \
        --provider hf \
        --num-prompts "${NUM_PROMPTS}" \
        --prompt-style summary_joint \
        --thinking-tags \
        --config-file "${CONFIG_FILE}" \
        --temperature 0.0 \
        --hf-max-new-tokens 8192 \
        --responses-root "${RESPONSES_ROOT}"

    # ── Run both configurations × both anonymization settings ─────────────
    # Non-anon: direct comparison with existing SFT baseline (real variable names)
    eval_only 5000 200
    eval_only 1000 50

    # Anon: shows improvement over SFT-anon baseline (X1/X2/... variable names,
    #       which is the setting grpo_v2_cancer_eq was trained on)
    eval_only 5000 200 "--anonymize"
    eval_only 1000 50 "--anonymize"

    echo ""
    echo "============================================"
    echo " All done. Results appended to:"
    echo " ${SUMMARY_CSV}"
    echo "============================================"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
