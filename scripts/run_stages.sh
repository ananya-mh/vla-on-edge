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
# Set RUN_ID to reuse one results folder across separate stage invocations (stage 3
# uses stage 2's timing for its estimate), SCENE_TIMEOUT_S to change the per-scene
# timeout (default 3600 s), and RESULTS_ROOT to write somewhere other than results/.
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
SCENE_TIMEOUT_S="${SCENE_TIMEOUT_S:-3600}"

fail() { echo "STAGE FAILED: $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

[[ -x "$ALPASIM/.venv/bin/python" && -x "$PYTHON" ]] || fail "run scripts/setup_vm.sh first"
# The driver container runs our code from wheels, not a checkout, so record the
# commit here; the AlpaSim config passes these into the container for run metadata.
VLA_ON_EDGE_GIT_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
VLA_ON_EDGE_GIT_DIRTY="$([[ -n "$(git -C "$REPO" status --porcelain)" ]] && echo true || echo false)"
export VLA_ON_EDGE_GIT_COMMIT VLA_ON_EDGE_GIT_DIRTY
# The driver container installs these wheels; rebuild so it runs the current code.
bash "$REPO/scripts/build_wheels.sh" > /dev/null || fail "could not build our wheels"

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

# run_with_timeout <seconds> <log_dir> <command...>
# Runs the command in its own process group. On timeout, stops that run's containers
# (killing only the wizard would leave them holding the GPU), then the whole group,
# and returns 124.
run_with_timeout() {
  local seconds="$1" log_dir="$2"
  shift 2
  local marker="$log_dir.timed_out" pid watcher status=0
  rm -f "$marker"
  set -m  # job control: the background job gets its own process group
  "$@" &
  pid=$!
  set +m
  (
    sleep "$seconds"
    touch "$marker"
    echo "TIMEOUT after ${seconds} s: stopping the run" >&2
    if [[ -f "$log_dir/docker-compose.yaml" ]]; then
      docker compose -f "$log_dir/docker-compose.yaml" down --timeout 30 > /dev/null 2>&1 || true
    fi
    kill -TERM -- "-$pid" 2> /dev/null || true
  ) &
  watcher=$!
  wait "$pid" || status=$?
  pkill -P "$watcher" 2> /dev/null || true  # its sleep
  kill "$watcher" 2> /dev/null || true
  wait "$watcher" 2> /dev/null || true
  if [[ -f "$marker" ]]; then
    rm -f "$marker"
    return 124
  fi
  return "$status"
}

# record_seconds <name> <seconds>: keep stage timings for later estimates.
record_seconds() {
  mkdir -p "$RESULTS"
  echo "$1=$2" >> "$RESULTS/timings.txt"
  echo "$1: $(( $2 / 60 )) min $(( $2 % 60 )) s"
}

# recorded_seconds <name>: the last recorded value, or empty. Always succeeds, so a
# missing value does not stop the script under set -e.
recorded_seconds() {
  if [[ -f "$RESULTS/timings.txt" ]]; then
    { grep "^$1=" "$RESULTS/timings.txt" || true; } | tail -1 | cut -d= -f2
  fi
}

# Log nvidia-smi memory (per GPU and per container) for the whole invocation; the
# monitor writes gpu_memory_peaks.json when stopped, including on failure.
start_gpu_monitor() {
  mkdir -p "$RESULTS"
  "$PYTHON" "$REPO/scripts/gpu_monitor.py" "$RESULTS" --interval-s "${GPU_MONITOR_INTERVAL_S:-5}" &
  GPU_MONITOR_PID=$!
  trap 'kill -TERM "$GPU_MONITOR_PID" 2> /dev/null || true; wait "$GPU_MONITOR_PID" 2> /dev/null || true' EXIT
}

set_stage() {
  mkdir -p "$RESULTS"
  echo "$1" > "$RESULTS/current_stage"
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
  step "$stage: $model on $scene (timeout ${SCENE_TIMEOUT_S} s)"
  local status=0 start=$SECONDS
  mkdir -p "$(dirname "$log_dir")"
  run_with_timeout "$SCENE_TIMEOUT_S" "$log_dir" run_alpasim "$log_dir" "$scene" "$@" || status=$?
  if [[ "$status" -eq 124 ]]; then
    copy_results "$log_dir" "$stage" "$model" "$scene"
    fail "$scene timed out after ${SCENE_TIMEOUT_S} s (raise SCENE_TIMEOUT_S); logs in $RESULTS/$stage/$model/$scene"
  elif [[ "$status" -ne 0 ]]; then
    copy_results "$log_dir" "$stage" "$model" "$scene"
    fail "AlpaSim run failed for $scene; logs in $RESULTS/$stage/$model/$scene/txt-logs"
  fi
  record_seconds "${stage}_${scene}" $(( SECONDS - start ))
  copy_results "$log_dir" "$stage" "$model" "$scene"
  # shellcheck disable=SC2086  # check_flags is empty or one flag
  "$PYTHON" -m vla_on_edge.eval.check_run "$log_dir" $check_flags || fail "outputs for $scene are incomplete"
}

# stage_report <stage> [output path]
stage_report() {
  "$PYTHON" -m vla_on_edge.eval.report "$RESULTS/$1" --output "${2:-$RESULTS/$1/report.md}"
}

stage_1() {
  set_stage stage1
  run_scene stage1 alpamayo1_5_builtin "$(config_value "config['smoke_scene']")" ""
  stage_report stage1
}

stage_2() {
  local scene log_dir start=$SECONDS
  set_stage stage2
  scene="$(config_value "config['smoke_scene']")"
  # Generate and check the wrapper's configs before spending GPU time on them.
  log_dir="$WORK/stage2_config_check"
  run_alpasim "$log_dir" "$scene" "${WRAPPER_OVERRIDES[@]}" wizard.run_method=NONE || fail "config generation failed"
  "$ALPASIM/.venv/bin/python" -m alpasim_vla_on_edge.check_config "$log_dir" || fail "wrapper configs are wrong"
  run_scene stage2 alpamayo1_5 "$scene" --plugin-logs "${WRAPPER_OVERRIDES[@]}"
  stage_report stage2
  record_seconds stage2_total $(( SECONDS - start ))
}

# Print how long stage 3 should take, from stage 2's one-scene time (same RUN_ID).
estimate_stage_3() {
  local scene_count="$1" per_scene
  per_scene="$(recorded_seconds stage2_total)"
  if [[ -z "$per_scene" ]]; then
    echo "No stage 2 timing for RUN_ID=$RUN_ID; run stage 2 with the same RUN_ID for an estimate."
    return
  fi
  local total=$(( scene_count * per_scene ))
  echo "Estimated stage 3 time: ~$(( total / 60 )) min ($scene_count scenes x ~$(( per_scene / 60 )) min each, from stage 2)."
  echo "Per-scene timeout: ${SCENE_TIMEOUT_S} s."
}

stage_3() {
  local scene scenes start=$SECONDS
  set_stage stage3
  scenes="$(config_value "' '.join(scene['id'] for scene in config['scenes'])")"
  step "Stage 3: $(wc -w <<< "$scenes" | tr -d ' ') scenes"
  estimate_stage_3 "$(wc -w <<< "$scenes" | tr -d ' ')"
  for scene in $scenes; do
    run_scene stage3 alpamayo1_5 "$scene" --plugin-logs "${WRAPPER_OVERRIDES[@]}"
  done
  record_seconds stage3_total $(( SECONDS - start ))
  stage_report stage3
  # Written in place (not copied) so its relative chart and video links resolve.
  stage_report stage3 "$RESULTS_ROOT/report.md"
  echo "report: $RESULTS_ROOT/report.md"
}

start_gpu_monitor
case "${1:-}" in
  1) stage_1 ;;
  2) stage_2 ;;
  3) stage_3 ;;
  all) stage_1 && stage_2 && stage_3 ;;
  *) fail "usage: $0 1|2|3|all" ;;
esac
echo "DONE: stage ${1} (results in $RESULTS)"
