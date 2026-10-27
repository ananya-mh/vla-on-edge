# Results (draft)

> **Status:** draft. Stages 1 and 2 have run for Alpamayo 2 Super, each on one scene.
> Anything marked **[TODO]** will come from the generated
> `results/vm_<RUN_ID>/stage3/report.md` once stage 3 has run. The numbers shown
> come from run `vm_20261027_195851`; none are estimated.

## Goal

Measure how well open self-driving models drive when their decisions actually
move the car (closed-loop), and what that costs in time and GPU memory. The
first model is NVIDIA's Alpamayo 2 Super; Alpamayo 1.5 follows on the same setup.
How the pieces fit: [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup

- **Simulator:** NVIDIA AlpaSim (commit `affc2eab`) with the NuRec renderer.
  NuRec draws camera images from 3D reconstructions of real recorded drives.
- **Scenes:** 8 on-road scenes from NVIDIA's public set (suite `public_2601`), listed in
  `configs/scenarios.yaml`. Two are AlpaSim's own reference scenes; six are a seeded
  random sample. We don't yet know how hard each one is.
- **Each run:** one rollout per scene. A rollout is about 20 s of simulated time at
  10 steps per second. For the first 1.7 s the car follows the recorded drive; then
  the model takes over.
- **Model input and output:** four cameras (front wide, front telephoto, left and
  right cross views) and recent motion in; a reasoning sentence and a 6.4 s
  trajectory (64 points, 0.1 s apart) out.
- **Model settings:** bf16, batch size 1, one trajectory sample, AlpaSim's default settings.
- **Hardware:** Hyperstack VM with 2× NVIDIA A100 80 GB PCIe, NVIDIA driver 580.178.04,
  Docker 29.9.2, Compose 5.6.1. The model runs alone on GPU 0; the renderer and physics share GPU 1.
- **Scoring:** AlpaSim's own. A scene scores 0 on an at-fault collision (front or side)
  or on going off-road; otherwise it scores min(progress / 0.8, 1). AlpaSim stops scoring
  at the first collision of any kind, or once the car is 4 m from the recorded drive.
- **Our additions:** jerk (how abruptly acceleration changes), latency per decision,
  GPU memory, and validator flags. Flags mark trajectories that break speed or
  acceleration limits: speed 15 m/s, acceleration 2 m/s², braking 6 m/s². They are
  recorded only; the trajectory is never changed.

## Results

### Stages 1 and 2: the same scene, twice (Alpamayo 2 Super)

Stage 1 runs AlpaSim's driver unchanged; stage 2 runs it through our logging plugin.

| | Stage 1 | Stage 2 |
|---|---|---|
| Scene score | 1.00 | 0.00 |
| Outcome | no at-fault collision or off-road | at-fault collision (front) at 11.4 s |
| Progress | 1.00 | 1.00 |
| Wall-clock time per scene | 844 s | 814 s |
| Latency per decision, p50 / p90 (n) | not logged | 3.16 s / 4.17 s (n = 182) |
| Jerk p95 / max | not logged | 3.6 / 9.4 m/s³ |
| Validator flags | not logged | 21% of 182 decisions |
| GPU 0 peak, driver (nvidia-smi) | 69.5 GiB | 69.5 GiB |
| PyTorch peak allocated / reserved | not logged | 68.4 / 68.9 GiB |
| GPU 1 peak, renderer + physics | 9.3 GiB | 9.3 GiB |

The same model on the same scene scored 1.00 once and 0.00 once. Our plugin returns
the model's trajectories unchanged. The model samples its reasoning and trajectory
at random: AlpaSim's default config sets `force_determinism: false`. So one rollout
per scene is not enough to rank models. See [Limitations](#limitations).

### Stage 3: all scenes, through our plugin

**[TODO]** Copy the Headline and Per-scenario tables from
`results/vm_<RUN_ID>/stage3/report.md`:

| Model | Scenes | Driving score | Collision (at fault) | Off-road | Progress | Jerk p95 | Latency p50 / p90 (n) | GPU peak | Validator flags |
|---|---|---|---|---|---|---|---|---|---|
| Alpamayo 2 Super | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** | **[TODO]** |
| Alpamayo 1.5 | **[TODO]** | | | | | | | | |

**[TODO]** Charts: score by scene, latency distribution, validator flags (from the generated report's `charts/`).

### Experiment: AlpaSim's Alpamayo 2 comparison settings (not the benchmark default)

The benchmark uses AlpaSim's default driver settings: 4 cameras, one trajectory
sample, always the first one. NVIDIA's own Alpamayo 2 comparison preset instead
uses six cameras (adding rear left and right), six candidates per step, and
`CLOSEST_LATERAL`. That rule drives the candidate closest sideways to the previous
plan, which should steady the plan-to-plan swings we saw. One run on the stage 2
crash scene, through our plugin (`run_stages.sh experiment`):

| | Default (stage 2) | Comparison settings |
|---|---|---|
| Scene score | 0.00 | **[TODO]** |
| Latency per decision, p50 / p90 (n) | 3.16 / 4.17 s (182) | **[TODO]** |
| GPU 0 peak, driver | 69.5 GiB | **[TODO]** |
| GPU 1 peak, renderer + physics | 9.3 GiB | **[TODO]** |

One run is one random draw, like every other single run here.

## Notable drives

**Stage 1, scene `clipgt-01d503d4…`: perfect score, but the car left the recorded path.**
- At 10.4 s the car was 4 m or more from the recorded drive, so AlpaSim stopped scoring there.
- After that, the car was hit from behind (12.8 s), then hit something in front (13.8 s)
  and at the side (18.7 s).
- None of these collisions count toward the score, because they happened after scoring stopped.
- We re-rendered the reasoning-overlay video from this run's logs. At 9 s it shows the
  reasoning "Keep distance to the lead vehicle because it is slowing ahead", and
  visible rendering artifacts, such as a car that seems to float above the road.

So the score says the model drove well while it was being scored. It says nothing
about what happened after the car left the recorded path.

**Stage 2, the same scene: at-fault front collision at 11.4 s.** AlpaSim stopped
scoring there. A rear collision followed at 14.4 s, which was not scored. In the
0.2 s before the collision, the model's reasoning switched between three plans:
- "Slow down to create a usable gap for merging left because the right lane ahead is
  blocked by a stopped vehicle and queued traffic."
- "Slow down to create a gap on the left for a lane change because the adjacent left
  lane is occupied by passing traffic while the lead traffic ahead is stopped/slow."
- "Keep distance to the lead vehicle because traffic ahead is congested."

The reasoning-overlay video for this drive is in the stage 2 results.

**[TODO]** The three lowest-scoring drives from stage 3, each with what happened, the
model's reasoning just before, and a link to the reasoning-overlay video.

## Limitations

- **On-road scenes only.** All scenes are public-road drives; none are off-road.
- **Few scenes, one random rollout each.** Eight scenes, one rollout each. The model
  samples at random, and the same scene scored 1.00 in stage 1 and 0.00 in stage 2. Differences
  between models will be noisy, and the scenes were not picked for difficulty.
- **Scoring stops early.** AlpaSim stops scoring at the first collision or once the car
  drifts 4 m from the recorded drive. A high score can hide what happened later, as in
  stage 1.
- **Rendering artifacts.** The renderer can only reconstruct what was recorded.
  Once the car moves away from the recorded path, images can be blurred or wrong.
  The model sees these images, so some failures may come from the renderer rather
  than the model.
- **A100, not H100.** NVIDIA tested Alpamayo on H100 GPUs. Our latency on the A100 is not
  comparable to NVIDIA's numbers, and is comparable only between runs with the same GPU
  layout.
- **Latency is model time only.** It is the time per decision in the driver, about 3 s
  on the A100. It doesn't include rendering or physics. The simulator waits for each
  decision, so slow decisions don't hurt the driving score.
- **Validator limits are first guesses.** They are not calibrated against real driving data yet.
- **Default settings.** Each model runs with AlpaSim's defaults (one sample, no extra
  guidance). Other settings may drive better or worse.

## Next steps

1. Run stage 3 for Alpamayo 2 Super; fill in the **[TODO]** sections from the generated report.
2. Run each scene several times (each as its own run), or turn on AlpaSim's `force_determinism`, so results can be repeated.
3. Run the same three stages for Alpamayo 1.5 and compare the two models.
4. Screen a larger set of scenes and swap in harder ones (dense traffic, unprotected turns).
5. Calibrate the validator limits from these runs.
6. Later: add general vision-language models through the agent layer (see [ARCHITECTURE.md](ARCHITECTURE.md)).
