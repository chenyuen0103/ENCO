#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-enco-llm}"
PYTHON_VERSION="${2:-3.9}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda not found. Install Miniconda/Anaconda first, then re-run." >&2
  exit 1
fi

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

echo "[info] installing system-like deps via conda (graphviz)"
conda install -c conda-forge graphviz -y

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
  conda activate ${ENV_NAME}
  # ENCO baseline example:
  cd experiments
  python run_exported_graphs.py --graph_files ../causal_graphs/real_data/small_graphs/cancer.bif

LLM notes:
  - Export OPENAI_API_KEY for OpenAI runs.
  - Export GOOGLE_API_KEY (or GEMINI_API_KEY) for Gemini runs.

Torch notes:
  - This installs a default torch build from pip. If you want GPU PyTorch, install the
    correct CUDA-enabled torch build for your system instead.
EOF
