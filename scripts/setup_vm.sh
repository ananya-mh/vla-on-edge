#!/usr/bin/env bash
# One-time setup of a fresh cloud GPU VM, so that scripts/run_stages.sh only executes.
#
# Needs: a VM (not a container-based GPU pod) with Docker (compose + buildx plugins,
# usable without sudo), the NVIDIA driver, and the NVIDIA Container Toolkit.
#
# Usage:
#   HF_TOKEN=hf_... bash setup_vm.sh            # e.g. curl the file onto the VM first
# Options (environment variables):
#   WORKDIR   where to put everything            (default: $HOME/bench)
#   REPO_URL  our repo                           (default: GitHub origin below)
#   REPO_REF  branch, tag, or commit to check out (default: main)
#
# Steps: check prerequisites -> install uv and Rust -> clone our repo and AlpaSim at
# the pinned commit -> set up AlpaSim's environment -> install our plugin (--no-deps)
# -> set up our report environment -> check Hugging Face access -> download
# Alpamayo 1.5 -> download the scenes and build AlpaSim's image -> check configs.
# Every step fails fast with a message saying what to fix.
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/bench}"
REPO_URL="${REPO_URL:-https://github.com/ananya-mh/vla-on-edge.git}"
REPO_REF="${REPO_REF:-main}"
ALPASIM_URL="https://github.com/NVlabs/alpasim.git"
ALPASIM_COMMIT="affc2eab209fa43bdfa2f26c0f8d437922d78a68"  # the commit everything was built against
MIN_DRIVER_MAJOR=570          # AlpaSim's renderer container uses CUDA 12.8 (docs/ONBOARDING.md)
MIN_FREE_DISK_GB=200          # AlpaSim image + Alpamayo 1.5 (22 GB) + Cosmos-Reason2 + scenes + rollouts
MIN_GPU_MEMORY_MIB=80000      # plan: Alpamayo 1.5 and the renderer on one 80 GB GPU
# Gated status as of 2026-10-26 (Hugging Face API): Cosmos-Reason2-8B is gated;
# Alpamayo-1.5-10B is not, though AlpaSim's docs/TUTORIAL.md says to accept its terms.
MODELS=("nvidia/Alpamayo-1.5-10B" "nvidia/Cosmos-Reason2-8B")
SCENE_DATASET="nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"     # gated (scenes/csv_utils.py)
# Loaded by AlpaSim's driver at startup via alpamayo1_5.helper.get_processor
# (alpamayo1_5/helper.py BASE_PROCESSOR_NAME); public. Only its small files are needed.
PROCESSOR_REPO="Qwen/Qwen3-VL-2B-Instruct"
NON_WEIGHT_FILES='["*.json", "*.txt", "*.jinja", "*.model", "tokenizer*"]'

REPO="$WORKDIR/vla-on-edge"
ALPASIM="$REPO/third_party/alpasim"

fail() { echo "SETUP FAILED: $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

check_prerequisites() {
  step "Checking prerequisites"
  for tool in git curl docker nvidia-smi; do
    command -v "$tool" > /dev/null || fail "'$tool' is not installed"
  done
  docker info > /dev/null 2>&1 || fail "docker does not run without sudo; run: sudo usermod -aG docker \$USER, then log in again"
  docker compose version > /dev/null 2>&1 || fail "the docker compose plugin is missing"
  docker buildx version > /dev/null 2>&1 || fail "the docker buildx plugin is missing"

  local driver_major gpu_memory free_gb
  driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
  [[ "$driver_major" -ge "$MIN_DRIVER_MAJOR" ]] || fail "NVIDIA driver $driver_major < $MIN_DRIVER_MAJOR (needed for CUDA 12.8)"
  gpu_memory="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
  [[ "$gpu_memory" -ge "$MIN_GPU_MEMORY_MIB" ]] || echo "WARNING: GPU 0 has ${gpu_memory} MiB; the plan assumes 80 GB"
  docker run --rm --gpus all ubuntu:22.04 nvidia-smi > /dev/null 2>&1 \
    || fail "containers cannot see the GPU: install/configure the NVIDIA Container Toolkit"

  mkdir -p "$WORKDIR"
  free_gb="$(df -BG --output=avail "$WORKDIR" | tail -1 | tr -dc '0-9')"
  [[ "$free_gb" -ge "$MIN_FREE_DISK_GB" ]] || fail "only ${free_gb} GB free in $WORKDIR; need ${MIN_FREE_DISK_GB} GB"

  [[ -n "${HF_TOKEN:-}" ]] || fail "HF_TOKEN is not set (create one at https://huggingface.co/settings/tokens)"
  echo "OK: driver $driver_major, GPU ${gpu_memory} MiB, ${free_gb} GB free"
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
  "$ALPASIM/.venv/bin/python" - "${MODELS[@]}" "$SCENE_DATASET" << 'EOF' || fail "Hugging Face access check failed (see above)"
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
  step "Downloading ${MODELS[*]} into the Hugging Face cache (mounted into the driver)"
  "$ALPASIM/.venv/bin/python" -c "
import sys
from huggingface_hub import snapshot_download
for repo_id in sys.argv[1:]:
    print(snapshot_download(repo_id))
" "${MODELS[@]}" || fail "model download failed"

  step "Downloading what the driver would otherwise fetch at startup"
  "$ALPASIM/.venv/bin/python" - "$PROCESSOR_REPO" "$NON_WEIGHT_FILES" << 'EOF' || fail "could not cache the driver's startup files"
import json
import sys
from huggingface_hub import hf_hub_download, snapshot_download

processor_repo, patterns = sys.argv[1], json.loads(sys.argv[2])
print(snapshot_download(processor_repo, allow_patterns=patterns))
# Alpamayo 1.5 loads its base VLM's config and processor from the repo named in its
# own config.json (alpamayo1_5/models/base_model.py, config.vlm_name_or_path).
config = json.load(open(hf_hub_download("nvidia/Alpamayo-1.5-10B", "config.json")))
base_vlm = config.get("vlm_name_or_path")
print("Alpamayo 1.5 base VLM:", base_vlm)
if base_vlm is None:
    sys.exit("config.json has no vlm_name_or_path; check alpamayo1_5's config format")
print(snapshot_download(base_vlm, allow_patterns=patterns))
EOF
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
      deploy=local topology=1gpu driver=alpamayo1_5 \
      "hydra.searchpath=[file://$REPO/configs/alpasim]" +vla_on_edge=alpamayo1_5 \
      wizard.run_method=NONE wizard.log_dir="$prepare_dir" "scenes.scene_ids=[$(scene_ids)]") \
    || fail "AlpaSim's wizard could not prepare the run (scene download or config error; see above)"
  "$ALPASIM/.venv/bin/python" -m alpasim_vla_on_edge.check_config "$prepare_dir" || fail "generated configs are wrong"

  step "Pulling prebuilt images (NVIDIA's NuRec renderer, Prometheus)"
  docker compose -f "$prepare_dir/docker-compose.yaml" pull --ignore-buildable \
    || fail "image pull failed; if nvcr.io refused it, run: docker login nvcr.io (user \$oauthtoken, password: an NGC API key)"

  step "Building AlpaSim's Docker image (long; needs network)"
  docker compose -f "$prepare_dir/docker-compose.yaml" build || fail "docker image build failed"
}

main() {
  check_prerequisites
  install_tools
  clone_repos
  setup_alpasim
  setup_report_env
  check_hf_access
  download_models
  prepare_scenes_and_image
  echo
  echo "SETUP DONE. Next: bash $REPO/scripts/run_stages.sh 1   (then 2, then 3)"
}

main "$@"
