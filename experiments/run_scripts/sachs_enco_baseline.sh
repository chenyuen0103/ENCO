#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper to run the ENCO "table-range" baseline sweep on Sachs.
#
# Usage:
#   SEED=42 bash experiments/run_scripts/sachs_enco_baseline.sh
#
# Optional overrides:
#   OBS_SIZES="0 100 1000 5000 8000" INT_SIZES="0 50 100 200 500" SEED=0 FORCE=1 \
#     bash experiments/run_scripts/sachs_enco_baseline.sh
#
# Outputs (relative to repo root):
# - checkpoints: experiments/checkpoints/enco_table_range/sachs/...
# - predictions: experiments/responses/sachs/predictions_obs{N}_int{M}_ENCO.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BIF_FILE="${BIF_FILE:-causal_graphs/real_data/small_graphs/sachs.bif}"
export SEED="${SEED:-0}"
export FORCE="${FORCE:-0}"

# Forward optional sweeps if provided
export OBS_SIZES="${OBS_SIZES:-0 100 1000 5000 8000}"
export INT_SIZES="${INT_SIZES:-0 50 100 200 500}"

bash "${SCRIPT_DIR}/experiment1_enco_table_range.sh"

