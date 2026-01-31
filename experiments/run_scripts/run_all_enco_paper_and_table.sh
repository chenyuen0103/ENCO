#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper: run both
#  1) ENCO paper baselines (experiments/run_scripts/experiment_*.sh suite)
#  2) ENCO table-range sweep for Experiment 1
#
# Usage:
#   SEED=0 bash experiments/run_scripts/run_all_enco_paper_and_table.sh
#
# Rerun everything:
#   SEED=0 FORCE=1 bash experiments/run_scripts/run_all_enco_paper_and_table.sh
#
# Table-range overrides:
#   BIF_FILE=causal_graphs/real_data/small_graphs/sachs.bif SEED=0 \
#     bash experiments/run_scripts/run_all_enco_paper_and_table.sh
#
# Table-range sweep over all real_data graphs:
#   ALL_REAL_DATA=1 REAL_DATA_SCOPE=all SEED=0 \
#     bash experiments/run_scripts/run_all_enco_paper_and_table.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"

echo "[info] Running ENCO paper baselines + table-range sweep (SEED=${SEED}, FORCE=${FORCE})"

SEED="${SEED}" FORCE="${FORCE}" bash "${SCRIPT_DIR}/run_all_enco_baselines.sh"
SEED="${SEED}" FORCE="${FORCE}" ALL_REAL_DATA="${ALL_REAL_DATA:-0}" REAL_DATA_SCOPE="${REAL_DATA_SCOPE:-}" \
  BIF_FILE="${BIF_FILE:-}" DATASET="${DATASET:-}" \
  OBS_SIZES="${OBS_SIZES:-}" INT_SIZES="${INT_SIZES:-}" \
  bash "${SCRIPT_DIR}/experiment1_enco_table_range.sh"

echo "[done] Completed ENCO paper + table-range runs."
