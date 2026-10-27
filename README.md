# vla-on-edge

## GPU VM requirements

- A full VM, not a container-based GPU pod (AlpaSim starts its own containers with
  docker compose). We use Hyperstack, 1x A100 80 GB PCIe.
- Ubuntu 24.04, with NVIDIA driver 580 or newer (CUDA 13, needed by the Warp version
  in AlpaSim's physics service), or with no driver. Never upgrade a driver in place;
  start a new VM instead.
- Docker with the compose and buildx plugins, and the NVIDIA Container Toolkit.
  `scripts/bootstrap_vm.sh` installs these (and the driver, if there is none); then reboot.
- Passwordless sudo (bootstrap, and moving Docker's data-root once).
- At least 200 GB free on the data disk (`DATA_DIR`, default: the ephemeral disk at
  /ephemeral, which is wiped when the VM stops).
- A Hugging Face read token, with the terms accepted for `nvidia/Cosmos-Reason2-8B`
  and the `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` dataset.

Steps: `bash scripts/bootstrap_vm.sh`, `sudo reboot`, `HF_TOKEN=... bash scripts/setup_vm.sh`,
then `bash scripts/run_stages.sh 1` (then 2, then 3) from the repo copy that setup prints.
Problems we hit and their fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
