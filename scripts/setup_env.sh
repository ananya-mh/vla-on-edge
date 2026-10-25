#!/usr/bin/env bash
# Create (if needed) and activate the virtual environment. Use: source scripts/setup_env.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
if [ ! -d "$REPO_ROOT/.venv" ]; then
  python3.11 -m venv "$REPO_ROOT/.venv"
fi
source "$REPO_ROOT/.venv/bin/activate"
pip install -q -r "$REPO_ROOT/requirements.txt"
