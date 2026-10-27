# Troubleshooting

Problems we hit getting AlpaSim and Alpamayo running, in the order we hit them.
Commit hashes point to the fix.

## Plugin install failed in AlpaSim's environment
- Symptom: installing our plugin with `uv pip install` failed while resolving AlpaSim's git dependencies.
- Cause: AlpaSim's environment already has every dependency, and resolving them again fails.
- Fix: install with `--no-deps` (setup_vm.sh). Don't run `uv sync` there: it removes our plugin.

## `import src` loaded the wrong package
- Symptom: inside AlpaSim's driver container, our imports loaded AlpaSim's code.
- Cause: our package was named `src`, and the container's working directory has AlpaSim's own `src/`.
- Fix: renamed our package to `vla_on_edge` (9d78b27).

## Installing our code in the container needed the network and left root-owned files
- Symptom: at startup the driver container tried to download packages, and wrote root-owned build files into our repo.
- Cause: it built our packages from the mounted source folder, as root.
- Fix: build wheels (ready-made install files) on the host and install them offline, with the repo mounted read-only (c4360c9).

## The GPU provider couldn't run Docker
- Symptom: RunPod's GPU pods can't start Docker containers.
- Cause: a pod is itself a container, and AlpaSim starts its own containers with docker compose.
- Fix: use a full VM (Hyperstack, A100 80 GB).

## The VM had no Docker
- Symptom: `docker` was missing, and once installed it needed sudo.
- Cause: the VM image has no Docker or NVIDIA Container Toolkit, and a new docker group membership applies only after you log in again.
- Fix: `scripts/bootstrap_vm.sh` installs both and asks for a reboot.

## The root disk was too small
- Symptom: the 100 GB root disk can't hold the images, model files, scenes, and runs.
- Cause: the VM's large disk is a separate 750 GB temporary disk at /ephemeral, which is wiped when the VM stops.
- Fix: everything goes under `DATA_DIR` (default /ephemeral), including Docker's image storage and the Hugging Face cache (b49178e). Copy results off before stopping the VM.

## Docker Compose froze after "Created"
- Symptom: Compose v5.6.1 stopped right after creating the containers. Its process state was "stopped".
- Cause: its interactive menu and colored terminal output.
- Fix: `COMPOSE_MENU=false`, `COMPOSE_ANSI=never`, and input from /dev/null (e3ebccb).

## Stage 1's driver crashed with "401 GatedRepoError"
- Symptom: the driver failed on `nvidia/Cosmos-Reason2-8B/processor_config.json`.
- Cause: the container went online with no token. The file doesn't exist, which is normally fine, but a gated model answers "401 not allowed" instead of "404 not found".
- Fix: the driver runs offline (`HF_HUB_OFFLINE=1`) against our checked cache (e3ebccb).

## The offline load failed while building Alpamayo 1.5's config
- Symptom: an OSError in `AutoProcessor.from_pretrained`, although every file we cached was present.
- Cause: a Transformers log message builds a default config, which loads the processor of `Qwen/Qwen3-VL-8B-Instruct`, a model Alpamayo 1.5 doesn't otherwise use.
- Fix: cache that repo's small config files too (b2f1a04). `prepare_hf_cache.py` can print every file Transformers asks for.

## Physics failed with NVIDIA driver 570
- Symptom: the physics service crashed at startup.
- Cause: AlpaSim's physics uses Warp 1.18.0, which needs CUDA 13, i.e. driver 580 or newer.
- Fix: `setup_vm.sh` now refuses drivers older than 580 (357f584).

## Upgrading the driver in place broke it
- Symptom: upgrading from 570 to 580 failed with package conflicts.
- Cause: the two driver versions' packages overlap.
- Fix: start a new VM with driver 580+ or no driver. `bootstrap_vm.sh` installs a driver only when none is present.

## AlpaSim's 2-GPU layout would load Alpamayo 2 Super three times
- Symptom: found before running. AlpaSim's `topology=2gpu` puts 3 copies of the driver on GPU 0.
- Cause: that layout is sized for small models. Alpamayo 2 Super alone peaked at 69.5 GiB.
- Fix: our `topology=vla_on_edge_2gpu` puts one copy alone on GPU 0, and the renderer and physics on GPU 1 (b4ef6d5).

## The report's worst-case text contradicted its tables
- Symptom: a scene that scored 1.00 was described as "no hard failure (low progress)", with a "first collision" at 13.8 s.
- Cause: the text read raw per-step events. AlpaSim had stopped scoring at 10.4 s, when the car was 4 m from the recorded drive, so later collisions didn't count.
- Fix: the text now applies AlpaSim's cutoffs, and marks each event as at fault or not, and scored or not (b0e8308).

## The report lists its own `charts` folder as a model (open)
- Symptom: stage 2's `report.md` has a `charts` row with no data.
- Cause: the report treats every folder in the stage directory as a model, including the `charts/` folder it writes there.
- Fix: not fixed yet. Ignore the row for now.
