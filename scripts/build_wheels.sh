#!/usr/bin/env bash
# Builds wheels for our two packages into dist/wheels/. AlpaSim's driver container
# installs these (--offline --no-deps) from the read-only repo mount, so it needs
# no network and writes nothing into the repo. Re-run after changing our code.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WHEELS="$REPO/dist/wheels"
rm -rf "$WHEELS"
uv build --quiet --wheel --out-dir "$WHEELS" "$REPO"
uv build --quiet --wheel --out-dir "$WHEELS" "$REPO/plugins/alpasim_vla_on_edge"
ls "$WHEELS"
