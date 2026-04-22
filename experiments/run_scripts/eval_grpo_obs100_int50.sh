#!/usr/bin/env bash
# eval_grpo_v2_cancer_eq.sh
#
# Evaluate grpo_v2_cancer_eq on Sachs at two obs/int configurations:
#   (1) obs=5000, int=200
#   (2) obs=1000, int=50
#
# Results are appended to experiments/responses/sachs/sachs_summary.csv
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
# MODEL="Qwen/Qwen3-4B-Thinking-2507" 
MODEL="${EXPERIMENTS_DIR}/checkpoints/causal_grpo_sachs_v1/checkpoint-400"
BIF="${REPO_ROOT}/causal_graphs/real_data/small_graphs/sachs.bif"
RESPONSES_ROOT="${EXPERIMENTS_DIR}/responses"
SUMMARY_CSV="${RESPONSES_ROOT}/sachs/sachs_summary.csv"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
RUN_EXPERIMENT_SCRIPT="${EXPERIMENTS_DIR}/run_experiment1_in_memory.py"
EVALUATE_SCRIPT="${EXPERIMENTS_DIR}/evaluate.py"
CONFIG_FILE="${SCRIPT_DIR}/configs/obs100_int50.json"
MODEL_TAG="${MODEL##*/}"
MODEL_TAG="${MODEL_TAG//:/_}"
MODEL_TAG="${MODEL_TAG// /_}"
CONFIG_COT_HINT="$(
python - "${CONFIG_FILE}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)

cfgs = payload.get("configs") if isinstance(payload, dict) else payload
def wrapper_mode(cfg):
    if not isinstance(cfg, dict):
        return "plain"
    mode = cfg.get("wrapper_mode")
    if mode is not None:
        mode = str(mode).strip().lower()
        return "chat" if mode == "chat" else "plain"
    return "chat" if bool(cfg.get("cot_hint", False)) else "plain"

vals = {wrapper_mode(c) for c in cfgs if isinstance(c, dict)}
if vals == {"chat"}:
    print("chat")
elif vals in ({"plain"}, set()):
    print("plain")
else:
    print("mixed")
PY
)"

LEGACY_WRAPPER_TAGS=("")
if [[ "${CONFIG_COT_HINT}" == "chat" ]]; then
    LEGACY_WRAPPER_TAGS=("cothint_")
elif [[ "${CONFIG_COT_HINT}" == "mixed" ]]; then
    LEGACY_WRAPPER_TAGS=("" "cothint_")
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

    local found_any=0
    local cot_tag
    for cot_tag in "${LEGACY_WRAPPER_TAGS[@]}"; do
        # Derive the exact filename that run_experiment1_in_memory.py writes.
        local resp_csv="${RESPONSES_ROOT}/sachs/responses_obs${obs}_int${int_n}_shuf1_p${NUM_PROMPTS}_${anon_suffix}thinktags_${cot_tag}summary_joint_${MODEL_TAG}.csv"
        if [[ ! -f "${resp_csv}" ]]; then
            continue
        fi

        found_any=1
        local cot_label="legacy_chat_wrapper=plain"
        if [[ "${cot_tag}" == "cothint_" ]]; then
            cot_label="legacy_chat_wrapper=chat"
        fi
        echo "[eval] Computing metrics (${cot_label}): ${resp_csv}"
        python "${EVALUATE_SCRIPT}" \
            --csv "${resp_csv}" \
            --summary-csv "${SUMMARY_CSV}"
    done

    if [[ "${found_any}" -eq 0 ]]; then
        echo "[error] No response CSV found for obs=${obs}, int=${int_n}${anon_label}. Expected one of:" >&2
        for cot_tag in "${LEGACY_WRAPPER_TAGS[@]}"; do
            echo "  - ${RESPONSES_ROOT}/sachs/responses_obs${obs}_int${int_n}_shuf1_p${NUM_PROMPTS}_${anon_suffix}thinktags_${cot_tag}summary_joint_${MODEL_TAG}.csv" >&2
        done
        return 1
    fi

    echo "[done] obs=${obs}, int=${int_n}${anon_label}"
}

main() {
    echo "============================================"
    echo " Evaluating: grpo_v2_cancer_eq"
    echo " Model:      ${MODEL}"
    echo " Model tag:  ${MODEL_TAG}"
    echo " BIF:        ${BIF}"
    echo " Prompts:    ${NUM_PROMPTS}"
    echo " Chat Wrap:  ${CONFIG_COT_HINT}"
    echo " Config:     ${CONFIG_FILE}"
    echo "============================================"

    if [[ "${CONFIG_COT_HINT}" == "mixed" ]]; then
        echo "[info] Mixed legacy chat-wrapper configs detected; evaluating both plain and chat legacy outputs when present."
    fi

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
        --config-file "${CONFIG_FILE}" \
        --temperature 0.0 \
        --hf-max-new-tokens 8192 \
        --responses-root "${RESPONSES_ROOT}"

    # ── Run both configurations × both anonymization settings ─────────────
    # Non-anon: direct comparison with existing SFT baseline (real variable names)
    eval_only 100 10
    eval_only 100 10

    # Anon: shows improvement over SFT-anon baseline (X1/X2/... variable names,
    #       which is the setting grpo_v2_cancer_eq was trained on)
    eval_only 100 10 "--anonymize"
    eval_only 100 10 "--anonymize"

    echo ""
    echo "============================================"
    echo " All done. Results appended to:"
    echo " ${SUMMARY_CSV}"
    echo "============================================"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
