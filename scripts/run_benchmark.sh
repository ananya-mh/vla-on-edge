#!/usr/bin/env bash
# Run one model on one experiment. All arguments are passed through, e.g.:
#   bash scripts/run_benchmark.sh --model dummy:bf16 --experiment smoke_test --sim mock [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "No virtual environment found. Run: source scripts/setup_env.sh" >&2
  exit 1
fi
"$PYTHON" -m vla_on_edge.run_benchmark "$@"
