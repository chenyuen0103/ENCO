#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-enco-llm}"
PYTHON_VERSION="${2:-3.12}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

METHOD="${SETUP_METHOD:-conda}" # "conda" (default) or "venv"
VENV_DIR="${VENV_DIR:-.venv}"  # used when METHOD=venv (or conda is missing)

if [[ "${METHOD}" != "conda" && "${METHOD}" != "venv" ]]; then
  echo "[error] Unknown SETUP_METHOD: ${METHOD} (expected: conda|venv)" >&2
  exit 2
fi

if [[ "${METHOD}" == "conda" ]] && ! command -v conda >/dev/null 2>&1; then
  echo "[warn] conda not found; falling back to venv at ${VENV_DIR}" >&2
  METHOD="venv"
fi

if [[ "${METHOD}" == "conda" ]]; then
  # Enable `conda activate` in non-interactive shells
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"

  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[info] conda env already exists: ${ENV_NAME}"
  else
    echo "[info] creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
  fi

  conda activate "${ENV_NAME}"
  ACTIVATE_HINT="conda activate ${ENV_NAME}"

  echo "[info] installing system-like deps via conda (graphviz)"
  conda install -c conda-forge graphviz -y
else
  echo "[info] creating venv at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}" 2>/dev/null || python -m venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  ACTIVATE_HINT="source ${VENV_DIR}/bin/activate"

  if ! command -v dot >/dev/null 2>&1; then
    echo "[warn] graphviz 'dot' not found on PATH. Some plots/layouts may not work." >&2
    echo "[hint] Install graphviz via your OS package manager, or use conda: SETUP_METHOD=conda ./setup.sh" >&2
  fi
fi

echo "[info] installing python deps via pip (requirements.txt)"
python -m pip install --upgrade pip
if [[ ! -f requirements.txt ]]; then
  echo "[error] requirements.txt not found in ${SCRIPT_DIR}" >&2
  echo "[hint] If you cloned an older version of the repo, run: git pull" >&2
  exit 1
fi
python -m pip install -r requirements.txt

cat <<EOF

[done] Environment is ready.

Next:
  ${ACTIVATE_HINT}
  # Experiment 1 (LLM prompt pipeline) dry-run from repo root:
  python experiments/run_experiment1_pipeline.py --dry-run \\
    --bif-file causal_graphs/real_data/small_graphs/cancer.bif \\
    --dataset cancer \\
    --model gpt-5-mini \\
    --shuffles-per-graph 1

LLM notes:
  - Export OPENAI_API_KEY for OpenAI runs.
  - Export GOOGLE_API_KEY (or GEMINI_API_KEY) for Gemini runs.

Torch notes:
  - This installs a default torch build from pip. If you want GPU PyTorch, install the
    correct CUDA-enabled torch build for your system instead.
EOF
