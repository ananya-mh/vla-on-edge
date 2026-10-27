# Troubleshooting: getting AlpaSim + Alpamayo running

Problems we hit, in the order we hit them.

## Plugin install failed in AlpaSim's environment
- Symptom: `uv pip install` of our plugin failed while resolving AlpaSim's git dependencies.
- Cause: AlpaSim's environment already has every dependency, and resolving them again fails.
- Fix: install with `--no-deps` (setup_vm.sh). Don't run `uv sync` there: it removes our plugin.

## `import src` loaded the wrong package
- Symptom: inside AlpaSim's driver container, our imports resolved to AlpaSim's code.
- Cause: our package was called `src`, and the container's working directory has AlpaSim's own `src/`.
- Fix: renamed the package to `vla_on_edge` (9d78b27).

## Installing our code in the container needed network and left root-owned files
- Symptom: the driver container's startup install tried to download, and wrote root-owned build files into our repo.
- Cause: it built our packages from the mounted source tree, as root.
- Fix: build wheels on the host and install them offline, with the repo mounted read-only (c4360c9).

## The GPU provider couldn't run Docker
- Symptom: RunPod pods can't run Docker containers with GPU access.
- Cause: AlpaSim starts its own containers with docker compose, so it needs a full VM, not a container.
- Fix: switched to a Hyperstack VM (1x A100 80 GB PCIe).

## The VM had no Docker
- Symptom: `docker` was missing, and after installing it, it still needed sudo.
- Cause: the image didn't include Docker or the NVIDIA Container Toolkit, and joining the docker group only takes effect at a new login.
- Fix: installed Docker and the toolkit by hand, then rebooted. scripts/bootstrap_vm.sh now does this on a fresh VM.

## The root disk was too small
- Symptom: the 100 GB root disk can't hold the images, model files, scenes and rollouts.
- Cause: the VM's large disk is a separate 750 GB ephemeral disk at /ephemeral, which is wiped when the VM stops.
- Fix: everything goes under DATA_DIR, by default /ephemeral (b49178e). That includes Docker's data-root (with containerd's image store turned off, so images go there too) and the Hugging Face and uv caches. Copy results off before stopping the VM.

## Docker Compose froze after "Created"
- Symptom: Compose v5.6.1 stopped after creating the containers; its process state was T (stopped).
- Cause: its interactive menu and terminal output.
- Fix: COMPOSE_MENU=false, COMPOSE_ANSI=never, and stdin from /dev/null (run_stages.sh, e3ebccb).

## Stage 1's built-in driver crashed with 401 GatedRepoError
- Symptom: the driver failed on `nvidia/Cosmos-Reason2-8B/processor_config.json`.
- Cause: the driver went online with no token. The file doesn't exist, which is normally fine, but a gated repo answers 401 instead of 404 without a token.
- Fix: run the built-in driver offline against our cache with HF_HUB_OFFLINE=1 (configs/alpasim/vla_on_edge/builtin_offline.yaml, e3ebccb).

## The offline load failed while building Alpamayo's config
- Symptom: OSError in `AutoProcessor.from_pretrained`, although every file we cached was present.
- Cause: a Transformers log message builds a default config, which loads the processor of `Qwen/Qwen3-VL-8B-Instruct`, a model Alpamayo doesn't otherwise use.
- Fix: cache that repo's config and processor files, no weights (b2f1a04). prepare_hf_cache.py now prints every file Transformers asks for.

## The physics service failed with driver 570
- Symptom: physics failed at startup on the 570 NVIDIA driver.
- Cause: AlpaSim's image ships Warp 1.18.0, which needs a CUDA 13 driver (580 or newer).
- Fix: setup_vm.sh now requires driver 580 or newer.

## Upgrading the driver in place broke it
- Symptom: upgrading from 570 to 580 failed with conflicts between the two driver versions' packages.
- Cause: the packages for the two driver versions overlap.
- Fix: start a new VM with driver 580 or newer, or with no driver. bootstrap_vm.sh installs `nvidia-driver-580-open` only when no driver is present, and never over an existing one.

## AlpaSim's 2-GPU topology would load Alpamayo 2 Super three times
- Symptom: (found before running) `topology=2gpu` puts 3 driver replicas on GPU 0.
- Cause: it is sized for small models; 2 Super needs about 70 GiB per copy.
- Fix: our `topology=vla_on_edge_2gpu` (configs/alpasim/topology/): the model alone on GPU 0, the rest on GPU 1. check_config checks each service's GPU.
