#!/usr/bin/env bash
set -euo pipefail

# Runs the ENCO baseline scripts (paper baselines) in a single sequence.
#
# Usage:
#   SEED=0 bash experiments/run_scripts/run_all_enco_baselines.sh
#
# Notes:
# - Most scripts require datasets downloaded via ./download_datasets.sh (repo root).
# - experiment_scale.sh does not require downloaded datasets.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED="${SEED:-42}"

echo "[info] Running ENCO baselines with SEED=${SEED}"
if [[ "${FORCE:-0}" == "1" ]]; then
  echo "[info] FORCE=1 set; will rerun even if done markers exist."
fi


SEED="${SEED}" bash "${SCRIPT_DIR}/experiment_real_data.sh"
SEED="${SEED}" bash "${SCRIPT_DIR}/experiment_synthetic.sh"
SEED="${SEED}" bash "${SCRIPT_DIR}/experiment_confounders.sh"
SEED="${SEED}" bash "${SCRIPT_DIR}/experiment_continuous.sh"
SEED="${SEED}" bash "${SCRIPT_DIR}/experiment_partial_interventions.sh"
SEED="${SEED}" bash "${SCRIPT_DIR}/experiment_scale.sh"

echo "[done] Completed ENCO baseline suite."
