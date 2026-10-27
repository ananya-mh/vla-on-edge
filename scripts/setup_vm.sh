#!/usr/bin/env bash
# One-time setup of a fresh cloud GPU VM, so that scripts/run_stages.sh only executes.
#
# Needs: a VM (not a container-based GPU pod) with Docker (compose + buildx plugins,
# usable without sudo), NVIDIA driver 580 or newer, and the NVIDIA Container Toolkit.
# On a fresh Ubuntu 24.04 VM, scripts/bootstrap_vm.sh installs these (then reboot).
# Moving Docker's data-root (once) needs passwordless sudo.
#
# Everything large lives under DATA_DIR: Docker's images and build cache, the
# Hugging Face and uv caches, our repo and AlpaSim (whose data/ holds the scenes),
# AlpaSim's run directories, and our results. On Hyperstack that is the ephemeral
# disk, which is wiped when the VM stops: copy results off first, and rerun this
# script after a restart.
#
# Usage:
#   HF_TOKEN=hf_... [MODEL=alpamayo2_super] bash setup_vm.sh   # e.g. curl the file onto the VM first
# Options (environment variables):
#   MODEL       alpamayo1_5 | alpamayo2_super: which model to download and check
#               (default: alpamayo1_5). Recorded in vm.env for run_stages.sh.
#   GPU_LAYOUT  1gpu | 2gpu (default: 2gpu if the VM has 2+ GPUs, else 1gpu; see
#               run_stages.sh). Recorded in vm.env.
#   DATA_DIR    large disk for everything above  (default: the ephemeral mount, e.g.
#               /ephemeral; else $HOME/vla_data if / has enough free space)
#   REPO_URL    our repo                          (default: GitHub origin below)
#   REPO_REF    branch, tag, or commit to check out (default: main)
#
# Steps: check prerequisites -> install uv and Rust -> clone our repo and AlpaSim at
# the pinned commit -> set up AlpaSim's environment -> install our plugin (--no-deps)
# -> set up our report environment -> check Hugging Face access -> download
# the model files (offline-checked) -> download the scenes, pull and build images ->
# check configs. Before all that: pick DATA_DIR and move Docker's data-root there.
# Every step fails fast with a message saying what to fix.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ananya-mh/vla-on-edge.git}"
REPO_REF="${REPO_REF:-main}"
ALPASIM_URL="https://github.com/NVlabs/alpasim.git"
ALPASIM_COMMIT="affc2eab209fa43bdfa2f26c0f8d437922d78a68"  # the commit everything was built against
# 580 = CUDA 13. AlpaSim's physics uses warp-lang 1.18.0 (third_party/alpasim/uv.lock),
# which failed on 570 on our first VM; its renderer needs CUDA 12.8 (570+,
# docs/ONBOARDING.md), so 580 covers both.
MIN_DRIVER_MAJOR=580
MODEL="${MODEL:-alpamayo1_5}"
GPU_LAYOUT="${GPU_LAYOUT:-}"
# Free space needed on DATA_DIR. Alpamayo 1.5: 200 GB for AlpaSim's image and build
# cache, the renderer image, the model files (22 GB), uv's cache, 8 scenes, and
# rollouts with videos. Alpamayo 2 Super: the same with its 71.6 GB checkpoint
# instead (about 250 GB), rounded up to 300 GB, which also leaves room to cache 1.5.
case "$MODEL" in
  alpamayo1_5) MIN_FREE_DISK_GB=200 ;;
  alpamayo2_super) MIN_FREE_DISK_GB=300 ;;
  *) echo "SETUP FAILED: unknown MODEL=$MODEL (alpamayo1_5 | alpamayo2_super)" >&2; exit 1 ;;
esac
# Where VM providers mount a local scratch disk (Hyperstack: /ephemeral, per
# docs.hyperstack.cloud/docs/storage/ephemeral).
EPHEMERAL_MOUNTS=(/ephemeral /mnt/ephemeral)
MIN_GPU_MEMORY_MIB=80000      # 80 GB GPUs (A100/H100 80GB)
# Hugging Face repos each model's driver loads; scripts/prepare_hf_cache.py says what
# each is for and downloads only what is used. Gated status per the Hugging Face API
# on 2026-10-26: Cosmos-Reason2-8B and the NuRec dataset are gated (accept the terms
# on their pages); Alpamayo-1.5-10B, Alpamayo2-Super, and the Qwen3-VL repos are not,
# although AlpaSim's docs/TUTORIAL.md says to accept Alpamayo 1.5's terms.
if [[ "$MODEL" == alpamayo1_5 ]]; then
  MODEL_REPOS=("nvidia/Alpamayo-1.5-10B" "nvidia/Cosmos-Reason2-8B" "Qwen/Qwen3-VL-8B-Instruct"
               "Qwen/Qwen3-VL-2B-Instruct")
  ALPASIM_DRIVER=alpamayo1_5
else
  MODEL_REPOS=("nvidia/Alpamayo2-Super")  # self-contained: see prepare_hf_cache.py
  ALPASIM_DRIVER=alpamayo2
fi
SCENE_DATASET="nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"  # scenes (wizard/scenes/csv_utils.py)

# setup_data_dir sets WORKDIR="$DATA_DIR/bench", REPO="$WORKDIR/vla-on-edge", and
# ALPASIM="$REPO/third_party/alpasim".
DATA_DIR="${DATA_DIR:-}"
# Same Compose settings as run_stages.sh (v5.6.1 froze with its menu and ANSI output on).
export COMPOSE_MENU=false COMPOSE_ANSI=never

fail() { echo "SETUP FAILED: $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

check_prerequisites() {
  step "Checking prerequisites"
  for tool in git curl docker nvidia-smi; do
    command -v "$tool" > /dev/null || fail "'$tool' is not installed (on a fresh VM, run scripts/bootstrap_vm.sh first)"
  done
  docker info > /dev/null 2>&1 || fail "docker does not run without sudo; run: sudo usermod -aG docker \$USER, then log in again"
  docker compose version > /dev/null 2>&1 || fail "the docker compose plugin is missing"
  docker buildx version > /dev/null 2>&1 || fail "the docker buildx plugin is missing"

  local driver_major gpu_memory gpu_count
  driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
  [[ "$driver_major" -ge "$MIN_DRIVER_MAJOR" ]] || fail "NVIDIA driver $driver_major is too old: need $MIN_DRIVER_MAJOR+ \
(CUDA 13; AlpaSim's physics uses Warp 1.18.0). Don't upgrade the driver in place (570 -> 580 \
broke on a package conflict); start a new VM with a $MIN_DRIVER_MAJOR+ driver, or one with no driver and run scripts/bootstrap_vm.sh"
  gpu_memory="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
  [[ "$gpu_memory" -ge "$MIN_GPU_MEMORY_MIB" ]] || echo "WARNING: GPU 0 has ${gpu_memory} MiB; the plan assumes 80 GB"
  gpu_count="$(nvidia-smi -L | grep -c '^GPU' || true)"
  if [[ -z "$GPU_LAYOUT" ]]; then
    GPU_LAYOUT=1gpu
    [[ "$gpu_count" -ge 2 ]] && GPU_LAYOUT=2gpu
  fi
  case "$GPU_LAYOUT" in
    1gpu) TOPOLOGY=1gpu ;;
    2gpu) TOPOLOGY=vla_on_edge_2gpu
          [[ "$gpu_count" -ge 2 ]] || fail "GPU_LAYOUT=2gpu needs 2 GPUs; this VM has $gpu_count" ;;
    *) fail "unknown GPU_LAYOUT=$GPU_LAYOUT (1gpu | 2gpu)" ;;
  esac
  if [[ "$MODEL" == alpamayo2_super && "$GPU_LAYOUT" == 1gpu ]]; then
    # Model card: 72,115 MiB peak on one H100 (7 cameras); AlpaSim uses 4. Measured on
    # VM 1: renderer 1.6 GiB. Physics on GPU 0 too: a few GiB of headroom at best.
    echo "WARNING: Alpamayo 2 Super (~70.4 GiB peak per its card), the renderer (1.6 GiB on VM 1), and"
    echo "         physics will share GPU 0's 80 GiB (untested). If it runs out of memory, use a 2-GPU VM."
  fi
  [[ -n "${HF_TOKEN:-}" ]] || fail "HF_TOKEN is not set (create one at https://huggingface.co/settings/tokens)"
  echo "OK: driver $driver_major, $gpu_count GPU(s), GPU 0 ${gpu_memory} MiB; MODEL=$MODEL, GPU_LAYOUT=$GPU_LAYOUT"
}

free_gb() {
  df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'
}

# Print the first mounted ephemeral disk, or nothing.
detect_ephemeral_mount() {
  local mount
  for mount in "${EPHEMERAL_MOUNTS[@]}"; do
    if mountpoint -q "$mount" 2> /dev/null; then
      echo "$mount"
      return
    fi
  done
}

# Pick DATA_DIR, make it ours, check its free space, and write $WORKDIR/vm.env so
# run_stages.sh uses the same locations.
setup_data_dir() {
  step "Choosing the data disk"
  if [[ -z "$DATA_DIR" ]]; then
    DATA_DIR="$(detect_ephemeral_mount)"
  fi
  if [[ -z "$DATA_DIR" ]]; then
    [[ "$(free_gb /)" -ge "$MIN_FREE_DISK_GB" ]] || fail "no ephemeral disk mounted at ${EPHEMERAL_MOUNTS[*]} \
and only $(free_gb /) GB free on /; need ${MIN_FREE_DISK_GB} GB. Set DATA_DIR=/path/on/a/large/disk"
    DATA_DIR="$HOME/vla_data"
    echo "No ephemeral disk found; / has room, so using $DATA_DIR"
  fi
  if ! mkdir -p "$DATA_DIR" 2> /dev/null || [[ ! -w "$DATA_DIR" ]]; then
    # Ephemeral mounts are usually owned by root.
    { sudo -n mkdir -p "$DATA_DIR" && sudo -n chown "$(id -u):$(id -g)" "$DATA_DIR"; } \
      || fail "cannot write to $DATA_DIR; run: sudo chown \$USER $DATA_DIR"
  fi
  local free
  free="$(free_gb "$DATA_DIR")"
  [[ "$free" -ge "$MIN_FREE_DISK_GB" ]] || fail "only ${free} GB free in $DATA_DIR; need ${MIN_FREE_DISK_GB} GB"

  WORKDIR="$DATA_DIR/bench"
  REPO="$WORKDIR/vla-on-edge"
  ALPASIM="$REPO/third_party/alpasim"
  # HF_HOME is read by huggingface_hub, AlpaSim's setup script, and its wizard (which
  # mounts it into the driver container); UV_CACHE_DIR holds the torch wheels.
  export HF_HOME="$DATA_DIR/huggingface" UV_CACHE_DIR="$DATA_DIR/uv-cache"
  mkdir -p "$WORKDIR" "$HF_HOME" "$UV_CACHE_DIR"
  # Values already in the environment win, so tests and one-off overrides still work.
  cat > "$WORKDIR/vm.env" << ENV
# Written by scripts/setup_vm.sh; sourced by scripts/run_stages.sh.
export DATA_DIR="\${DATA_DIR:-$DATA_DIR}"
export WORKDIR="\${WORKDIR:-$WORKDIR}"
export HF_HOME="\${HF_HOME:-$HF_HOME}"
export UV_CACHE_DIR="\${UV_CACHE_DIR:-$UV_CACHE_DIR}"
export MODEL="\${MODEL:-$MODEL}"
export GPU_LAYOUT="\${GPU_LAYOUT:-$GPU_LAYOUT}"
ENV
  echo "OK: DATA_DIR=$DATA_DIR (${free} GB free)"
}

# Docker keeps images and build cache in its data-root (default /var/lib/docker, on
# the small root disk), so point it at DATA_DIR. Existing images are not moved.
move_docker_data_root() {
  step "Putting Docker's data-root on the data disk"
  local target="$DATA_DIR/docker"
  if [[ "$(docker info --format '{{.DockerRootDir}}')" == "$target" ]]; then
    echo "OK: already $target"
    return
  fi
  sudo -n true 2> /dev/null || fail "moving Docker's data-root to $target needs passwordless sudo; \
or set \"data-root\": \"$target\" in /etc/docker/daemon.json yourself and restart Docker"
  sudo mkdir -p "$target"
  # Merge into daemon.json: the NVIDIA Container Toolkit keeps its runtime there.
  # The containerd image store (Docker 29's default for new installs) keeps images
  # under /var/lib/containerd instead of data-root, so turn it off.
  sudo python3 - "$target" << 'EOF' || fail "could not update /etc/docker/daemon.json"
import json
import os
import sys

path = "/etc/docker/daemon.json"
config = {}
if os.path.exists(path) and os.path.getsize(path) > 0:
    with open(path) as file:
        config = json.load(file)
config["data-root"] = sys.argv[1]
config.setdefault("features", {})["containerd-snapshotter"] = False
with open(path, "w") as file:
    json.dump(config, file, indent=2)
EOF
  sudo systemctl restart docker || fail "Docker did not restart; check: sudo journalctl -u docker"
  [[ "$(docker info --format '{{.DockerRootDir}}')" == "$target" ]] || fail "Docker's data-root is still not $target"
  echo "OK: Docker data-root $target, storage driver $(docker info --format '{{.Driver}}')"
}

check_container_gpu() {
  step "Checking that containers can see the GPU"
  docker run --rm --gpus all ubuntu:22.04 nvidia-smi > /dev/null 2>&1 \
    || fail "containers cannot see the GPU: install/configure the NVIDIA Container Toolkit"
  echo "OK"
}

install_tools() {
  step "Installing uv and Rust (if missing)"
  if ! command -v uv > /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh || fail "could not install uv"
    export PATH="$HOME/.local/bin:$PATH"
  fi
  if ! command -v cargo > /dev/null; then
    # AlpaSim's utils_rs package is a Rust extension; its setup script needs cargo.
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y || fail "could not install Rust"
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
  fi
  echo "OK: $(uv --version), $(cargo --version)"
}

clone_repos() {
  step "Cloning our repo ($REPO_REF) and AlpaSim ($ALPASIM_COMMIT)"
  if [[ ! -d "$REPO/.git" ]]; then
    git clone "$REPO_URL" "$REPO" || fail "could not clone $REPO_URL (private repo? use a token URL)"
  fi
  git -C "$REPO" fetch --quiet origin
  git -C "$REPO" checkout --quiet "$REPO_REF" || fail "ref $REPO_REF not found (push it first?)"

  if [[ ! -d "$ALPASIM/.git" ]]; then
    git init --quiet "$ALPASIM"
    git -C "$ALPASIM" remote add origin "$ALPASIM_URL"
  fi
  git -C "$ALPASIM" fetch --quiet --depth 1 origin "$ALPASIM_COMMIT" || fail "could not fetch AlpaSim $ALPASIM_COMMIT"
  git -C "$ALPASIM" checkout --quiet FETCH_HEAD
  echo "OK: $(git -C "$REPO" rev-parse --short HEAD), AlpaSim $(git -C "$ALPASIM" rev-parse --short HEAD)"
}

setup_alpasim() {
  step "Setting up AlpaSim's environment (its own setup_local_env.sh) and installing our plugin"
  # The script must be sourced; a subshell keeps its cd and exports out of ours.
  # It reads unset variables (e.g. HF_HOME) and reports errors via `return 1`, so it
  # runs without our strict mode and we check its exit status instead.
  (cd "$ALPASIM" && set +eu && source setup_local_env.sh) || fail "AlpaSim's setup_local_env.sh failed"
  # AlpaSim's environment already has every dependency; resolving again fails on its
  # git dependencies, so install without deps. `uv sync` there would remove these.
  uv pip install --python "$ALPASIM/.venv/bin/python" --no-deps \
    -e "$REPO" -e "$REPO/plugins/alpasim_vla_on_edge" || fail "could not install our plugin into AlpaSim's venv"
  mkdir -p "$ALPASIM/data/drivers"  # mounted into the driver container; must exist
  echo "OK: AlpaSim environment ready"
}

setup_report_env() {
  step "Setting up our environment (Python 3.11) for checks and the report"
  uv venv --quiet --python 3.11 "$REPO/.venv" || fail "could not create $REPO/.venv"
  uv pip install --quiet --python "$REPO/.venv/bin/python" -r "$REPO/requirements.txt" -e "$REPO" \
    || fail "could not install requirements.txt"
  echo "OK: $REPO/.venv"
}

check_hf_access() {
  step "Checking Hugging Face access to the gated models and scene dataset"
  "$ALPASIM/.venv/bin/python" - "${MODEL_REPOS[@]}" "$SCENE_DATASET" << 'EOF' || fail "Hugging Face access check failed (see above)"
import sys
from huggingface_hub import auth_check, whoami
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

print("logged in as", whoami()["name"])
*models, dataset = sys.argv[1:]
problems = []
for repo_id, repo_type in [(model, "model") for model in models] + [(dataset, "dataset")]:
    try:
        auth_check(repo_id, repo_type=repo_type)
        print("access ok:", repo_id)
    except GatedRepoError:
        problems.append(f"accept the terms at https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}")
    except RepositoryNotFoundError:
        problems.append(f"{repo_id} not found or the token cannot see it")
for problem in problems:
    print("PROBLEM:", problem)
sys.exit(1 if problems else 0)
EOF
}

download_models() {
  step "Caching the files $MODEL's driver loads, then checking they load offline"
  # Alpamayo 1.5: its checkpoint, plus config/processor files of Cosmos-Reason2-8B and
  # two Qwen3-VL repos. Alpamayo 2 Super: its checkpoint only (71.6 GB; the check loads
  # it on the GPU). Fails if anything would need a download.
  "$ALPASIM/.venv/bin/python" "$REPO/scripts/prepare_hf_cache.py" "$MODEL" || fail "model download or offline check failed"
}

scene_ids() {
  "$REPO/.venv/bin/python" -c "
from vla_on_edge.utils.config import CONFIG_DIR, load_alpasim_scenes
print(','.join(scene['id'] for scene in load_alpasim_scenes(CONFIG_DIR)['scenes']))"
}

prepare_scenes_and_image() {
  step "Building our wheels (installed offline in the driver container)"
  bash "$REPO/scripts/build_wheels.sh" || fail "could not build our wheels"

  step "Downloading the scenes and generating AlpaSim's configs (no simulation yet)"
  local prepare_dir="$WORKDIR/prepare"
  VLA_ON_EDGE_GIT_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
  export VLA_ON_EDGE_GIT_COMMIT  # checked by check_config below
  rm -rf "$prepare_dir"
  (cd "$ALPASIM" && VLA_ON_EDGE_REPO="$REPO" uv run --no-sync alpasim_wizard \
      deploy=local topology="$TOPOLOGY" driver="$ALPASIM_DRIVER" \
      "hydra.searchpath=[file://$REPO/configs/alpasim]" "+vla_on_edge=$MODEL" \
      wizard.run_method=NONE wizard.log_dir="$prepare_dir" "scenes.scene_ids=[$(scene_ids)]") \
    || fail "AlpaSim's wizard could not prepare the run (scene download or config error; see above)"
  "$ALPASIM/.venv/bin/python" -m alpasim_vla_on_edge.check_config "$prepare_dir" --model "$MODEL" --layout "$GPU_LAYOUT" \
    || fail "generated configs are wrong"

  step "Pulling prebuilt images (NVIDIA's NuRec renderer, Prometheus)"
  docker compose -f "$prepare_dir/docker-compose.yaml" pull --ignore-buildable < /dev/null \
    || fail "image pull failed; if nvcr.io refused it, run: docker login nvcr.io (user \$oauthtoken, password: an NGC API key)"

  step "Building AlpaSim's Docker image (long; needs network)"
  docker compose -f "$prepare_dir/docker-compose.yaml" build < /dev/null || fail "docker image build failed"
}

main() {
  check_prerequisites
  setup_data_dir
  move_docker_data_root
  check_container_gpu
  install_tools
  clone_repos
  setup_alpasim
  setup_report_env
  check_hf_access
  download_models
  prepare_scenes_and_image
  echo
  echo "SETUP DONE for $MODEL ($GPU_LAYOUT). Everything is under $DATA_DIR."
  echo "Next: cd $REPO && bash scripts/run_stages.sh 1   (then 2, then 3; MODEL and GPU_LAYOUT come from vm.env)"
  echo "If $DATA_DIR is an ephemeral disk, stopping the VM wipes it: copy results off first."
}

main "$@"
