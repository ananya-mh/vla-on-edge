#!/usr/bin/env bash
# Runs the GPU benchmark for one model in three stages; each stops at the first failure.
#   1: one scene with AlpaSim's built-in driver for the model, unmodified (sanity
#      baseline; only its container environment is set, to stay offline)
#   2: the same scene through our logging wrapper, then check our logs exist and parse
#   3: every scene in configs/scenarios.yaml (alpasim.scenes) through the wrapper,
#      then build the report
# Each scene is its own AlpaSim run (our logs cannot tell rollouts apart).
#
# Usage: [MODEL=...] [GPU_LAYOUT=...] bash scripts/run_stages.sh 1|2|3|all   (after scripts/setup_vm.sh)
#   MODEL       alpamayo1_5 | alpamayo2_super (default: the one setup_vm.sh prepared,
#               from vm.env; else alpamayo1_5)
#   GPU_LAYOUT  1gpu: every service on GPU 0 (AlpaSim's topology/1gpu.yaml)
#               2gpu: the model alone on GPU 0, renderer and physics on GPU 1
#                     (configs/alpasim/topology/vla_on_edge_2gpu.yaml)
#               (default: from vm.env; else 2gpu if nvidia-smi lists 2+ GPUs, otherwise 1gpu)
# Results: results/vm_<RUN_ID>/stage<N>/<model>/<scene>/ (copied after each scene,
# also on failure), stage reports, and results/report.md after stage 3.
# Set RUN_ID to reuse one results folder across separate stage invocations (stage 3
# uses stage 2's timing for its estimate), SCENE_TIMEOUT_S to change the per-scene
# timeout (default 3600 s), and RESULTS_ROOT to write somewhere other than results/.
# AlpaSim's run directories go to $WORKDIR/runs; on a VM set up by setup_vm.sh,
# WORKDIR, HF_HOME, MODEL, and GPU_LAYOUT come from its vm.env (on the data disk,
# next to this repo).
set -euo pipefail

# setup_vm.sh may have installed uv and Rust here; a new shell might not have them on PATH.
export PATH="$PATH:$HOME/.local/bin:$HOME/.cargo/bin"  # appended: an existing uv wins
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# setup_vm.sh records where it put things (DATA_DIR, WORKDIR, HF_HOME, UV_CACHE_DIR);
# HF_HOME matters here because AlpaSim's wizard mounts it into the driver container.
if [[ -f "$REPO/../vm.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO/../vm.env"
fi
ALPASIM="$REPO/third_party/alpasim"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WORK="${WORKDIR:-$HOME/bench}/runs/$RUN_ID"   # AlpaSim writes here (large)
RESULTS_ROOT="${RESULTS_ROOT:-$REPO/results}"
RESULTS="$RESULTS_ROOT/vm_$RUN_ID"            # copied here after each scene
PYTHON="$REPO/.venv/bin/python"
SCENE_TIMEOUT_S="${SCENE_TIMEOUT_S:-3600}"
MODEL="${MODEL:-alpamayo1_5}"
# AlpaSim's driver config for each models.yaml key (src/wizard/configs/driver/); our
# wrapper overrides are configs/alpasim/vla_on_edge/<MODEL>.yaml.
case "$MODEL" in
  alpamayo1_5) ALPASIM_DRIVER=alpamayo1_5 ;;
  alpamayo2_super) ALPASIM_DRIVER=alpamayo2 ;;
  *) echo "STAGE FAILED: unknown MODEL=$MODEL (alpamayo1_5 | alpamayo2_super)" >&2; exit 1 ;;
esac
if [[ -z "${GPU_LAYOUT:-}" ]]; then
  GPU_LAYOUT=1gpu
  [[ "$(nvidia-smi -L 2> /dev/null | grep -c '^GPU' || true)" -ge 2 ]] && GPU_LAYOUT=2gpu
fi
case "$GPU_LAYOUT" in
  1gpu) TOPOLOGY=1gpu ;;
  2gpu) TOPOLOGY=vla_on_edge_2gpu ;;
  *) echo "STAGE FAILED: unknown GPU_LAYOUT=$GPU_LAYOUT (1gpu | 2gpu)" >&2; exit 1 ;;
esac
# AlpaSim's wizard runs docker compose in the foreground. Compose v5.6.1 froze after
# "Created" with its interactive menu and ANSI output on, so turn both off (and the
# wizard gets stdin from /dev/null in run_alpasim).
export COMPOSE_MENU=false COMPOSE_ANSI=never

# The data disk may be ephemeral (wiped when the VM stops), so say how to copy results off.
copy_off_reminder() {
  [[ -d "$RESULTS" ]] || return 0
  local host
  host="$(hostname -I 2> /dev/null | awk '{print $1}' || true)"  # Linux only; never fail here
  echo
  echo "REMINDER: copy the results off this VM before stopping it (an ephemeral disk is wiped). From your machine:"
  echo "  rsync -avz --progress $(whoami)@${host:-<vm-ip>}:$RESULTS/ ./results/vm_$RUN_ID/"
  echo "  (or: scp -r $(whoami)@${host:-<vm-ip>}:$RESULTS ./results/)"
  echo "  Use the VM's public IP if the address above is a private one."
}

fail() { echo "STAGE FAILED: $*" >&2; copy_off_reminder; exit 1; }
step() { echo; echo "==> $*"; }

[[ -x "$ALPASIM/.venv/bin/python" && -x "$PYTHON" ]] || fail "run scripts/setup_vm.sh first"
# The driver container runs our code from wheels, not a checkout, so record the
# commit here; the AlpaSim config passes these into the container for run metadata.
VLA_ON_EDGE_GIT_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
VLA_ON_EDGE_GIT_DIRTY="$([[ -n "$(git -C "$REPO" status --porcelain)" ]] && echo true || echo false)"
export VLA_ON_EDGE_GIT_COMMIT VLA_ON_EDGE_GIT_DIRTY
# Compose's behavior differs by version (see COMPOSE_MENU above), so record it: the
# wrapper's run metadata gets it via the container environment, and every stage
# (including stage 1, which has no wrapper) gets it in host_versions.txt.
VLA_ON_EDGE_COMPOSE_VERSION="$(docker compose version --short 2> /dev/null || echo unknown)"
export VLA_ON_EDGE_COMPOSE_VERSION
record_host_versions() {
  mkdir -p "$RESULTS"
  {
    echo "invocation=$(date -u +%Y-%m-%dT%H:%M:%SZ) stage=${1:-} model=$MODEL gpu_layout=$GPU_LAYOUT"
    echo "docker_compose=$VLA_ON_EDGE_COMPOSE_VERSION"
    echo "docker=$(docker version --format '{{.Server.Version}}' 2> /dev/null || echo unknown)"
    echo "nvidia_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2> /dev/null | head -1 || echo unknown)"
  } >> "$RESULTS/host_versions.txt"
}
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
      deploy=local topology="$TOPOLOGY" driver="$ALPASIM_DRIVER" \
      wizard.log_dir="$log_dir" "scenes.scene_ids=[$scene]" "$@" < /dev/null)
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

# hydra.searchpath also makes our topology (vla_on_edge_2gpu) visible to AlpaSim.
WRAPPER_OVERRIDES=("hydra.searchpath=[file://$REPO/configs/alpasim]" "+vla_on_edge=$MODEL")
# Stage 1: AlpaSim's driver as is, only run offline against the mounted cache.
BUILTIN_OVERRIDES=("hydra.searchpath=[file://$REPO/configs/alpasim]" "+vla_on_edge=builtin_offline")

# check_configs <stage> <scene> <extra check_config flags> <Hydra overrides...>
# Generates the run's configs without running and checks them (model, GPU layout,
# offline cache, our wrapper's mounts) before GPU time is spent.
check_configs() {
  local stage="$1" scene="$2" check_flags="$3"
  shift 3
  local log_dir="$WORK/${stage}_config_check"
  run_alpasim "$log_dir" "$scene" "$@" wizard.run_method=NONE || fail "config generation failed"
  # shellcheck disable=SC2086  # check_flags is empty or one flag
  "$ALPASIM/.venv/bin/python" -m alpasim_vla_on_edge.check_config "$log_dir" --model "$MODEL" --layout "$GPU_LAYOUT" \
    $check_flags || fail "$stage configs are wrong"
}

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
  "$PYTHON" -m vla_on_edge.eval.check_run "$log_dir" --model "$MODEL" $check_flags || fail "outputs for $scene are incomplete"
}

# stage_report <stage> [output path]
stage_report() {
  "$PYTHON" -m vla_on_edge.eval.report "$RESULTS/$1" --output "${2:-$RESULTS/$1/report.md}"
}

stage_1() {
  local scene
  set_stage stage1
  scene="$(config_value "config['smoke_scene']")"
  check_configs stage1 "$scene" --builtin "${BUILTIN_OVERRIDES[@]}"
  run_scene stage1 "${MODEL}_builtin" "$scene" "" "${BUILTIN_OVERRIDES[@]}"
  stage_report stage1
}

stage_2() {
  local scene start=$SECONDS
  set_stage stage2
  scene="$(config_value "config['smoke_scene']")"
  check_configs stage2 "$scene" "" "${WRAPPER_OVERRIDES[@]}"
  run_scene stage2 "$MODEL" "$scene" --plugin-logs "${WRAPPER_OVERRIDES[@]}"
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
    run_scene stage3 "$MODEL" "$scene" --plugin-logs "${WRAPPER_OVERRIDES[@]}"
  done
  record_seconds stage3_total $(( SECONDS - start ))
  stage_report stage3
  # Written in place (not copied) so its relative chart and video links resolve.
  stage_report stage3 "$RESULTS_ROOT/report.md"
  echo "report: $RESULTS_ROOT/report.md"
}

start_gpu_monitor
record_host_versions "${1:-}"
case "${1:-}" in
  1) stage_1 && copy_off_reminder ;;
  2) stage_2 && copy_off_reminder ;;
  3) stage_3 && copy_off_reminder ;;
  all) stage_1 && copy_off_reminder && stage_2 && copy_off_reminder && stage_3 && copy_off_reminder ;;
  *) fail "usage: $0 1|2|3|all" ;;
esac
echo "DONE: stage ${1}, $MODEL, $GPU_LAYOUT (results in $RESULTS)"
