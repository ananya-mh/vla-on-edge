# vla-on-edge

A benchmark that lets open self-driving models drive in a closed-loop simulator and
compares them on driving quality, speed, and GPU memory. "Closed-loop" means the
model's chosen path moves the car, and the next camera images show where the car
ended up, so mistakes add up the way they would on a real road. We use NVIDIA's
AlpaSim simulator and NVIDIA's Alpamayo models: Alpamayo 2 Super (34B parameters)
now, and Alpamayo 1.5 (10B) next.

## Requirements

- **Machine:** a full cloud VM, not a container-based GPU pod. AlpaSim starts its own Docker containers.
- **GPUs:** two 80 GB GPUs for Alpamayo 2 Super; one is enough for Alpamayo 1.5. We use NVIDIA A100 80 GB.
- **OS and driver:** Ubuntu 24.04 with NVIDIA driver 580 or newer, or with no driver at all.
  Never upgrade the driver in place; start a new VM instead.
- **Docker:** Docker with the compose and buildx plugins, plus the NVIDIA Container Toolkit
  (lets containers use the GPU). `bootstrap_vm.sh` installs all of these.
- **sudo:** passwordless sudo.
- **Disk:** 300 GB free for Alpamayo 2 Super, 200 GB for Alpamayo 1.5. By default this goes on
  the VM's large temporary disk, which is wiped when the VM stops.
- **Hugging Face:** a read token. You must accept the terms for the scene dataset
  `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`, and also for `nvidia/Cosmos-Reason2-8B` if you run Alpamayo 1.5.

## How to run

On the VM, from a clone of this repo:

```bash
bash scripts/bootstrap_vm.sh     # installs the NVIDIA driver (only if none), Docker, and the container toolkit
sudo reboot                      # loads the driver and your docker group membership
HF_TOKEN=hf_... MODEL=alpamayo2_super bash scripts/setup_vm.sh
                                 # downloads the code, model, and scenes; builds images; checks everything offline
cd /ephemeral/bench/vla-on-edge  # the copy setup made on the data disk (it prints the path)
bash scripts/run_stages.sh 1     # one scene with AlpaSim's own driver, unchanged: is the setup sound?
bash scripts/run_stages.sh 2     # the same scene through our logging plugin: do our logs work?
bash scripts/run_stages.sh 3     # all 8 scenes through the plugin, then the report
```

`MODEL` is `alpamayo2_super` or `alpamayo1_5`. The GPU layout is picked from the
number of GPUs; set `GPU_LAYOUT=1gpu` or `2gpu` to choose it yourself. Each stage
stops at the first problem and says what to fix. To keep one results folder across
stages, run each stage with the same `RUN_ID=...`.

## Results and videos

- Everything goes to `results/vm_<RUN_ID>/` in the repo copy on the VM.
- Each scene's AlpaSim output is in `stage<N>/<model>/<scene>/`. Our step logs are in `vla_on_edge/steps.jsonl`.
- Videos are in `rollouts/<scene>/<rollout>/*.mp4`. Each scene gets a top-down map with the
  front camera, and, in stages 2 and 3, the front camera with the model's reasoning text.
- Each stage writes `report.md` (tables, charts, worst cases), and stage 3 also writes `results/report.md`.
- After each stage the script prints an `rsync` command. Copy results off before stopping the VM.

## More

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): the parts and why they are built this way.
- [docs/REPORT.md](docs/REPORT.md): the results write-up (draft).
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md): problems we hit and how each was fixed.
