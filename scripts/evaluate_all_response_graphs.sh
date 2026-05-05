#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESPONSES_ROOT="${1:-${REPO_ROOT}/scripts/responses}"
BIF_ROOT="${BIF_ROOT:-${REPO_ROOT}/causal_graphs/real_data/small_graphs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIPELINE="${REPO_ROOT}/scripts/run_cd_eval_pipeline.py"

if [[ ! -d "${RESPONSES_ROOT}" ]]; then
  echo "[error] Responses directory not found: ${RESPONSES_ROOT}" >&2
  exit 1
fi

for response_dir in "${RESPONSES_ROOT}"/*; do
  [[ -d "${response_dir}" ]] || continue

  graph="$(basename "${response_dir}")"
  bif_file="${BIF_ROOT}/${graph}.bif"

  if [[ ! -f "${bif_file}" ]]; then
    echo "[skip] ${graph}: no BIF file at ${bif_file}"
    continue
  fi

  echo "[run] ${graph}"
  (
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" scripts/run_cd_eval_pipeline.py \
      --bif-file "${bif_file}" \
      --dataset "${graph}" \
      --steps evaluate,analyze \
      --responses-dir "${response_dir}"
  )
done
