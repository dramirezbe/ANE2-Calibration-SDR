#!/usr/bin/env bash
# install.sh - Set up a Python virtual environment and install dependencies
# Usage: bash install.sh

set -e

VENV_DIR=".venv"

echo "Creating Python virtual environment in '${VENV_DIR}'..."
python3 -m venv "${VENV_DIR}"

echo "Activating virtual environment..."
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
