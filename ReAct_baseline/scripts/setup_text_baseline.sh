#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${HOME}/.venvs/react-baseline"

PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3.9; do
    if command -v "${candidate}" >/dev/null 2>&1; then
        PYTHON_BIN="${candidate}"
        break
    fi
done

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "TextWorld requires Python 3.9, 3.10, 3.11, or 3.12."
    echo "Install a supported Python version and rerun this script."
    exit 1
fi

echo "Using ${PYTHON_BIN}: $("${PYTHON_BIN}" --version)"
rm -rf "${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_DIR}/requirements.txt"

alfworld-download

echo
echo "Text-only ALFWorld setup complete."
echo "Data directory: ${HOME}/.cache/alfworld"
echo "Activate later with: source ${VENV_DIR}/bin/activate"
