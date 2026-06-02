#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${HOME}/.venvs/react-baseline"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_DIR}/requirements.txt"

alfworld-download

echo
echo "Text-only ALFWorld setup complete."
echo "Data directory: ${HOME}/.cache/alfworld"
echo "Activate later with: source ${VENV_DIR}/bin/activate"
