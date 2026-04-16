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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}/experiments"

MODEL="${REPO_ROOT}/experiments/checkpoints/grpo_v2_cancer_eq"
BIF="${REPO_ROOT}/causal_graphs/real_data/small_graphs/sachs.bif"
RESPONSES_ROOT="${REPO_ROOT}/experiments/responses"
SUMMARY_CSV="${REPO_ROOT}/experiments/out/experiment1/sachs_summary.csv"
NUM_PROMPTS="${NUM_PROMPTS:-5}"

echo "============================================"
echo " Evaluating: grpo_v2_cancer_eq"
echo " Model:      ${MODEL}"
echo " BIF:        ${BIF}"
echo " Prompts:    ${NUM_PROMPTS}"
echo "============================================"

# Build a single config file so HF model loading happens once inside one
# run_experiment1_in_memory.py process and can reuse its in-memory pipeline cache.
CONFIG_FILE="$(mktemp "${TMPDIR:-/tmp}/eval_grpo_v2_cancer_eq.XXXXXX.json")"
trap 'rm -f "${CONFIG_FILE}"' EXIT

cat > "${CONFIG_FILE}" <<'JSON'
{
  "configs": [
    {
      "prompt_style": "summary_joint",
      "anonymize": false,
      "obs_per_prompt": 5000,
      "int_per_combo": 200,
      "row_order": "random",
      "col_order": "original",
      "shuffles_per_graph": 1
    },
    {
      "prompt_style": "summary_joint",
      "anonymize": false,
      "obs_per_prompt": 1000,
      "int_per_combo": 50,
      "row_order": "random",
      "col_order": "original",
      "shuffles_per_graph": 1
    },
    {
      "prompt_style": "summary_joint",
      "anonymize": true,
      "obs_per_prompt": 5000,
      "int_per_combo": 200,
      "row_order": "random",
      "col_order": "original",
      "shuffles_per_graph": 1
    },
    {
      "prompt_style": "summary_joint",
      "anonymize": true,
      "obs_per_prompt": 1000,
      "int_per_combo": 50,
      "row_order": "random",
      "col_order": "original",
      "shuffles_per_graph": 1
    }
  ]
}
JSON

echo ""
echo "--- Running combined config batch via --config-file ---"
python run_experiment1_in_memory.py \
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

    # Derive the exact filename that run_experiment1_in_memory.py writes
    RESP_CSV="${RESPONSES_ROOT}/sachs/responses_obs${obs}_int${int_n}_shuf1_p${NUM_PROMPTS}_${anon_suffix}thinktags_summary_joint_grpo_v2_cancer_eq.csv"

    echo "[eval] Computing metrics: ${RESP_CSV}"
    python evaluate.py \
        --csv "${RESP_CSV}" \
        --summary-csv "${SUMMARY_CSV}"

    echo "[done] obs=${obs}, int=${int_n}${anon_label}"
}

# ── Run both configurations × both anonymization settings ─────────────────
# Non-anon: direct comparison with existing SFT baseline (real variable names)
eval_only 5000 200
eval_only 1000 50

# Anon: shows improvement over SFT-anon baseline (X1/X2/... variable names,
#       which is the setting grpo_v2_cancer_eq was trained on)
eval_only 5000 200 "--anonymize"
eval_only 1000 50  "--anonymize"

echo ""
echo "============================================"
echo " All done. Results appended to:"
echo " ${SUMMARY_CSV}"
echo "============================================"
