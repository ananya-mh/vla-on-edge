#!/usr/bin/env bash
# Prepares a fresh Ubuntu 24.04 GPU VM for scripts/setup_vm.sh: NVIDIA driver (only
# if none is installed), Docker with compose + buildx, and the NVIDIA Container
# Toolkit, then asks for a reboot. Safe to rerun: each step skips what is present.
#
# Usage (as your normal user, with sudo):
#   bash scripts/bootstrap_vm.sh
#   sudo reboot
#   then: HF_TOKEN=hf_... bash scripts/setup_vm.sh
#
# Never installs a driver over an existing one: upgrading 570 -> 580 in place
# broke on our first VM (the two driver versions' packages conflict). If the VM
# came with a driver older than 580, start a new VM instead.
set -euo pipefail

DRIVER_BRANCH=580   # CUDA 13; see MIN_DRIVER_MAJOR in setup_vm.sh
DRIVER_PACKAGE="nvidia-driver-${DRIVER_BRANCH}-open"
# Ubuntu's prebuilt, signed kernel modules (no DKMS build): one package for the
# running kernel, and the kernel flavour's meta package (e.g. -generic) so the
# modules follow kernel updates. Names per Ubuntu Server's "Install NVIDIA drivers".
MODULES_PACKAGE="linux-modules-nvidia-${DRIVER_BRANCH}-open-$(uname -r)"
MODULES_META_PACKAGE="linux-modules-nvidia-${DRIVER_BRANCH}-open-$(uname -r | sed 's/^[0-9.-]*-//')"
NEEDS_REBOOT=false

fail() { echo "BOOTSTRAP FAILED: $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

check_system() {
  step "Checking the system"
  [[ "$(id -u)" -ne 0 ]] || fail "run as your normal user (it uses sudo), so that user joins the docker group"
  sudo -v || fail "this script needs sudo"
  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
    echo "WARNING: written for Ubuntu 24.04; this is ${PRETTY_NAME:-unknown}"
  fi
  echo "OK: ${PRETTY_NAME:-unknown}, kernel $(uname -r), user $USER"
}

# Print installed NVIDIA driver packages, if any (a driver may be installed but not
# loaded yet, e.g. before a reboot, so nvidia-smi alone is not enough).
installed_driver_packages() {
  dpkg-query -W -f '${Status} ${Package}\n' 'nvidia-driver-*' 'nvidia-dkms-*' 'linux-modules-nvidia-*' 2> /dev/null \
    | awk '$1 == "install" && $3 == "installed" {print $4}' || true
}

ensure_driver() {
  step "NVIDIA driver"
  local version packages
  if version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2> /dev/null | head -1)" && [[ -n "$version" ]]; then
    if [[ "${version%%.*}" -lt "$DRIVER_BRANCH" ]]; then
      fail "driver $version is installed; AlpaSim needs $DRIVER_BRANCH+ (CUDA 13). Not upgrading in place \
(570 -> 580 broke on our first VM): start a new VM with a $DRIVER_BRANCH+ driver, or with none"
    fi
    echo "OK: driver $version already installed; leaving it alone"
    return
  fi
  packages="$(installed_driver_packages)"
  if [[ -n "$packages" ]]; then
    fail "NVIDIA driver packages are installed but nvidia-smi does not work (reboot pending?): \
$(echo "$packages" | tr '\n' ' ')- not installing over them; reboot and rerun, or start a new VM"
  fi

  echo "No NVIDIA driver found; installing $DRIVER_PACKAGE with prebuilt modules ($MODULES_PACKAGE)"
  sudo apt-get update -q
  for package in "$MODULES_PACKAGE" "$DRIVER_PACKAGE"; do
    apt-cache show "$package" > /dev/null 2>&1 || fail "package $package is not available \
(is the 'restricted' component enabled? is this a stock Ubuntu kernel?)"
  done
  local modules=("$MODULES_PACKAGE")
  if [[ "$MODULES_META_PACKAGE" == "$MODULES_PACKAGE" ]]; then
    echo "WARNING: could not tell the kernel flavour from $(uname -r); after a kernel update the driver may not load"
  elif apt-cache show "$MODULES_META_PACKAGE" > /dev/null 2>&1; then
    modules+=("$MODULES_META_PACKAGE")
  else
    echo "WARNING: $MODULES_META_PACKAGE is not available; after a kernel update the driver may not load"
  fi
  # Modules first, then the driver (Ubuntu's documented order), so apt does not
  # build the modules with DKMS instead.
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q "${modules[@]}" || fail "kernel module install failed"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q "$DRIVER_PACKAGE" || fail "driver install failed"
  NEEDS_REBOOT=true
  echo "OK: installed $DRIVER_PACKAGE; it loads after the reboot"
}

ensure_docker() {
  step "Docker (with compose and buildx)"
  if ! command -v docker > /dev/null; then
    # Docker's official convenience script: docker-ce, containerd, and the compose and
    # buildx plugins from Docker's apt repository.
    curl -fsSL https://get.docker.com | sudo sh || fail "Docker install failed"
  fi
  sudo docker compose version > /dev/null 2>&1 || fail "the docker compose plugin is missing (sudo apt-get install docker-compose-plugin)"
  sudo docker buildx version > /dev/null 2>&1 || fail "the docker buildx plugin is missing (sudo apt-get install docker-buildx-plugin)"
  echo "OK: $(sudo docker --version), compose $(sudo docker compose version --short)"
}

ensure_container_toolkit() {
  step "NVIDIA Container Toolkit"
  if ! command -v nvidia-ctk > /dev/null; then
    # NVIDIA's apt repository, as in the toolkit's install guide.
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update -q
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q nvidia-container-toolkit || fail "toolkit install failed"
  fi
  # Registers the nvidia runtime in /etc/docker/daemon.json (merging with what is there).
  sudo nvidia-ctk runtime configure --runtime=docker || fail "could not configure Docker's NVIDIA runtime"
  sudo systemctl restart docker || fail "Docker did not restart; check: sudo journalctl -u docker"
  echo "OK: $(nvidia-ctk --version | head -1)"
}

ensure_docker_group() {
  step "docker group"
  if id -nG "$USER" | tr ' ' '\n' | grep -qx docker; then
    echo "OK: $USER is already in the docker group"
    return
  fi
  sudo usermod -aG docker "$USER" || fail "could not add $USER to the docker group"
  NEEDS_REBOOT=true  # group membership only applies to new logins
  echo "OK: added $USER to the docker group"
}

main() {
  check_system
  ensure_driver
  ensure_docker
  ensure_container_toolkit
  ensure_docker_group
  echo
  if [[ "$NEEDS_REBOOT" == true ]]; then
    echo "BOOTSTRAP DONE. Reboot now: sudo reboot"
    echo "After logging back in, check: nvidia-smi && docker run --rm --gpus all ubuntu:22.04 nvidia-smi"
  else
    echo "BOOTSTRAP DONE. Nothing needs a reboot."
  fi
  echo "Then: HF_TOKEN=hf_... bash scripts/setup_vm.sh"
}

main "$@"
