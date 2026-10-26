#!/usr/bin/env bash
# Runs the GPU benchmark in three stages; each stops at the first failure.
#   1: one scene with AlpaSim's built-in alpamayo1_5 driver, unmodified (sanity baseline)
#   2: the same scene through our logging wrapper, then check our logs exist and parse
#   3: every scene in configs/scenarios.yaml (alpasim.scenes) through the wrapper,
#      then build the report
# Each scene is its own AlpaSim run (our logs cannot tell rollouts apart).
#
# Usage: bash scripts/run_stages.sh 1|2|3|all     (after scripts/setup_vm.sh)
# Results: results/vm_<RUN_ID>/stage<N>/<model>/<scene>/ (copied after each scene,
# also on failure), stage reports, and results/report.md after stage 3.
# Set RUN_ID to reuse one results folder across separate stage invocations, and
# RESULTS_ROOT to write somewhere other than results/ (used by the tests).
set -euo pipefail

# setup_vm.sh may have installed uv and Rust here; a new shell might not have them on PATH.
export PATH="$PATH:$HOME/.local/bin:$HOME/.cargo/bin"  # appended: an existing uv wins
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ALPASIM="$REPO/third_party/alpasim"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WORK="${WORKDIR:-$HOME/bench}/runs/$RUN_ID"   # AlpaSim writes here (large)
RESULTS_ROOT="${RESULTS_ROOT:-$REPO/results}"
RESULTS="$RESULTS_ROOT/vm_$RUN_ID"            # copied here after each scene
PYTHON="$REPO/.venv/bin/python"

fail() { echo "STAGE FAILED: $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

[[ -x "$ALPASIM/.venv/bin/python" && -x "$PYTHON" ]] || fail "run scripts/setup_vm.sh first"

config_value() {
  "$PYTHON" -c "
from vla_on_edge.utils.config import CONFIG_DIR, load_alpasim_scenes
config = load_alpasim_scenes(CONFIG_DIR)
print($1)"
}

# run_alpasim <log_dir> <scene_id> [extra Hydra overrides...]
run_alpasim() {
  local log_dir="$1" scene="$2"
  shift 2
  (cd "$ALPASIM" && VLA_ON_EDGE_REPO="$REPO" uv run --no-sync alpasim_wizard \
      deploy=local topology=1gpu driver=alpamayo1_5 \
      wizard.log_dir="$log_dir" "scenes.scene_ids=[$scene]" "$@")
}

WRAPPER_OVERRIDES=("hydra.searchpath=[file://$REPO/configs/alpasim]" "+vla_on_edge=alpamayo1_5")

# copy_results <log_dir> <stage> <model> <scene>
copy_results() {
  [[ -d "$1" ]] || { echo "no outputs to copy from $1"; return 0; }
  local destination="$RESULTS/$2/$3/$4"
  mkdir -p "$destination"
  cp -a "$1/." "$destination/"
  echo "copied to $destination"
}

# run_scene <stage> <model> <scene> <check_run flags> [extra Hydra overrides...]
run_scene() {
  local stage="$1" model="$2" scene="$3" check_flags="$4"
  shift 4
  local log_dir="$WORK/$stage/$model/$scene"
  step "$stage: $model on $scene"
  if ! run_alpasim "$log_dir" "$scene" "$@"; then
    copy_results "$log_dir" "$stage" "$model" "$scene"
    fail "AlpaSim run failed for $scene; logs in $RESULTS/$stage/$model/$scene/txt-logs"
  fi
  copy_results "$log_dir" "$stage" "$model" "$scene"
  # shellcheck disable=SC2086  # check_flags is empty or one flag
  "$PYTHON" -m vla_on_edge.eval.check_run "$log_dir" $check_flags || fail "outputs for $scene are incomplete"
}

# stage_report <stage> [output path]
stage_report() {
  "$PYTHON" -m vla_on_edge.eval.report "$RESULTS/$1" --output "${2:-$RESULTS/$1/report.md}"
}

stage_1() {
  run_scene stage1 alpamayo1_5_builtin "$(config_value "config['smoke_scene']")" ""
  stage_report stage1
}

stage_2() {
  local scene log_dir
  scene="$(config_value "config['smoke_scene']")"
  # Generate and check the wrapper's configs before spending GPU time on them.
  log_dir="$WORK/stage2_config_check"
  run_alpasim "$log_dir" "$scene" "${WRAPPER_OVERRIDES[@]}" wizard.run_method=NONE || fail "config generation failed"
  "$ALPASIM/.venv/bin/python" -m alpasim_vla_on_edge.check_config "$log_dir" || fail "wrapper configs are wrong"
  run_scene stage2 alpamayo1_5 "$scene" --plugin-logs "${WRAPPER_OVERRIDES[@]}"
  stage_report stage2
}

stage_3() {
  local scene
  for scene in $(config_value "' '.join(scene['id'] for scene in config['scenes'])"); do
    run_scene stage3 alpamayo1_5 "$scene" --plugin-logs "${WRAPPER_OVERRIDES[@]}"
  done
  stage_report stage3
  # Written in place (not copied) so its relative chart and video links resolve.
  stage_report stage3 "$RESULTS_ROOT/report.md"
  echo "report: $RESULTS_ROOT/report.md"
}

case "${1:-}" in
  1) stage_1 ;;
  2) stage_2 ;;
  3) stage_3 ;;
  all) stage_1 && stage_2 && stage_3 ;;
  *) fail "usage: $0 1|2|3|all" ;;
esac
echo "DONE: stage ${1} (results in $RESULTS)"
