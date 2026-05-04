#!/usr/bin/env bash
set -euo pipefail

# Evaluate/analyze all response CSVs for graph BIFs.
#
# Usage:
#   bash scripts/evaluate_all_results.sh
#
# Optional overrides:
#   GRAPHS_DIR=causal_graphs/real_data/small_graphs \
#   RESPONSE_ROOTS="./experiments/responses ./scripts/responses ./responses" \
#   STEPS=evaluate,analyze \
#   PYTHON=python3 \
#   bash scripts/evaluate_all_results.sh
#
# Any extra arguments are forwarded to scripts/run_cd_eval_pipeline.py:
#   bash scripts/evaluate_all_results.sh --dry-run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GRAPHS_DIR="${GRAPHS_DIR:-causal_graphs/real_data/small_graphs}"
RESPONSE_ROOTS="${RESPONSE_ROOTS:-./experiments/responses ./scripts/responses ./responses}"
STEPS="${STEPS:-evaluate,analyze}"
PYTHON="${PYTHON:-python3}"

shopt -s nullglob
graph_paths=("${GRAPHS_DIR}"/*.bif)
shopt -u nullglob

if [[ ${#graph_paths[@]} -eq 0 ]]; then
  echo "No .bif graph files found under ${GRAPHS_DIR}" >&2
  exit 1
fi

processed=0
for bif_file in "${graph_paths[@]}"; do
  dataset="$(basename "${bif_file}" .bif)"
  response_args=()

  for response_root in ${RESPONSE_ROOTS}; do
    response_dir="${response_root}/${dataset}"
    [[ -d "${response_dir}" ]] || continue
    response_args+=(--responses-dir "${response_dir}")
  done

  if [[ ${#response_args[@]} -eq 0 ]]; then
    echo "Skipping ${dataset}: no response directories found"
    continue
  fi

  echo "Evaluating/analyzing ${dataset}"
  "${PYTHON}" scripts/run_cd_eval_pipeline.py \
    --bif-file "${bif_file}" \
    --dataset "${dataset}" \
    --steps "${STEPS}" \
    "${response_args[@]}" \
    "$@"

  processed=$((processed + 1))
done

if [[ ${processed} -eq 0 ]]; then
  echo "No graph response directories found under ${RESPONSE_ROOTS}" >&2
  exit 1
fi

echo "Done. Processed ${processed} dataset(s)."
