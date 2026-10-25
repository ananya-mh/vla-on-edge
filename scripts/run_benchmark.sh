#!/usr/bin/env bash
# Run one model on one experiment. All arguments are passed through, e.g.:
#   bash scripts/run_benchmark.sh --model dummy:bf16 --experiment smoke_test --sim mock [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.run_benchmark "$@"
