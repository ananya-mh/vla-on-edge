# vla-on-edge

## GPU VM requirements

- A full VM, not a container-based GPU pod (AlpaSim starts its own containers with
  docker compose). We use Hyperstack A100 80 GB PCIe VMs.
- GPUs, by model:
  - Alpamayo 1.5: one 80 GB GPU. Measured on our first VM: driver 21.3 GiB, renderer 1.6 GiB.
  - Alpamayo 2 Super: two 80 GB GPUs recommended (`GPU_LAYOUT=2gpu`: the model alone on
    GPU 0, the renderer and physics on GPU 1). One GPU may work (`GPU_LAYOUT=1gpu`, untested):
    the model card measured a 72,115 MiB (70.4 GiB) peak on an H100 with 7 cameras (AlpaSim
    uses 4), plus 1.6 GiB for the renderer and some for physics, out of 80 GiB. AlpaSim's
    own Alpamayo eval topology gives each driver a GPU of its own.
- Ubuntu 24.04, with NVIDIA driver 580 or newer (CUDA 13, needed by the Warp version
  in AlpaSim's physics service), or with no driver. Never upgrade a driver in place;
  start a new VM instead.
- Docker with the compose and buildx plugins, and the NVIDIA Container Toolkit.
  `scripts/bootstrap_vm.sh` installs these (and the driver, if there is none); then reboot.
- Passwordless sudo (bootstrap, and moving Docker's data-root once).
- Free space on the data disk (`DATA_DIR`, default: the ephemeral disk at /ephemeral,
  which is wiped when the VM stops): 200 GB for Alpamayo 1.5, 300 GB for Alpamayo 2 Super
  (its checkpoint alone is 71.6 GB).
- A Hugging Face read token, with the terms accepted for the
  `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` dataset (scenes) and, for Alpamayo 1.5,
  `nvidia/Cosmos-Reason2-8B`. `nvidia/Alpamayo2-Super` is not gated.

Steps: `bash scripts/bootstrap_vm.sh`, `sudo reboot`,
`HF_TOKEN=... MODEL=alpamayo2_super bash scripts/setup_vm.sh` (or `MODEL=alpamayo1_5`, the default),
then `bash scripts/run_stages.sh 1` (then 2, then 3) from the repo copy that setup prints.
Problems we hit and their fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
