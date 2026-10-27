# Results: Alpamayo 2 Super in AlpaSim

All numbers come from four result folders and from our analysis of AlpaSim's own logs
(`rollout.asl`):
- `results/vm_20261027_195851/`: stages 1 and 2
- `results/vm_20261027_210046/`: stage 3
- `results/vm_exp_6cam_1/`: the experiment

None are estimated. How the pieces fit: [ARCHITECTURE.md](ARCHITECTURE.md).

## Goal

Measure how well an open self-driving model drives when its own decisions move the
car (closed-loop), and what that costs in time and GPU memory. This report covers
NVIDIA's Alpamayo 2 Super. Alpamayo 1.5 is set up on the same pipeline but not run yet.

## Setup

- **Simulator:** NVIDIA AlpaSim (commit `affc2eab`) with the NuRec renderer. NuRec draws
  camera images from 3D reconstructions of real recorded drives.
- **Other vehicles** replay the recording. They don't react to our car.
- **Scenes:** 8 on-road scenes from NVIDIA's public set (`public_2601`), listed in
  `configs/scenarios.yaml`. Two are AlpaSim's own reference scenes; six are a seeded random sample.
- **A run:** one rollout per scene, about 20 s of simulated time at 10 steps per second.
  For the first 1.7 s the car follows the recorded drive; then the model drives.
- **Model settings (the benchmark default, which is AlpaSim's own):**
  - inputs: four cameras (front wide, front telephoto, left and right cross views) and recent motion
  - one sampled trajectory per step, always driven as is
  - bf16, batch size 1
  - output: a reasoning sentence and a 6.4 s trajectory (64 points)
- **Hardware:** 2× NVIDIA A100 80 GB PCIe, NVIDIA driver 580.178.04. The model runs alone on
  GPU 0; the renderer and physics share GPU 1.
- **Scoring (AlpaSim's own):**
  - A scene scores 0 on an at-fault collision (the other vehicle touches our car's front or side) or on going off-road.
  - Otherwise it scores min(progress / 0.8, 1).
  - AlpaSim **stops scoring** at the first collision of any kind, or once our car is 4 m or more from the recorded drive. Nothing after that counts.
- **Our additions:**
  - per-decision latency
  - GPU memory
  - jerk: how abruptly acceleration changes, in m/s³
  - validator flags: trajectories over our speed or acceleration limits. They are recorded only; trajectories are never changed.

## Results

### Headline: 8 scenes, default settings (stage 3)

| Metric | Value |
|---|---|
| Mean scene score | 0.80 |
| At-fault collisions (scored) | 0 of 8 scenes |
| Off-road | 1 of 8 scenes |
| Mean progress | 0.80 |
| Scoring stopped early by drift (≥ 4 m from the recorded drive) | 5 of 8 scenes |
| Jerk, 95th percentile | 3.0 m/s³ |
| Latency per decision, p50 / p90 / p99 | 3.13 / 3.74 / 4.55 s (n = 1,451) |
| GPU 0 peak, driver (nvidia-smi) | 69.3–69.5 GiB per scene |
| GPU 1 peak | renderer 3.5–9.6 GiB, physics 0.5–0.6 GiB per scene |
| Wall-clock time | 784–877 s per scene, 6,642 s for all 8 |

| Scene | Score | Progress | What happened |
|---|---|---|---|
| `01d503d4` (crash scene) | 1.00 | 1.00 | Scoring stopped by drift at 10.6 s; unscored collisions after it (side 11.2 s, front 11.4 s, rear 12.5 s) |
| `02eadd92` | 1.00 | 0.87 | Scored to the end |
| `235c5235` | 1.00 | 1.00 | Scoring stopped by drift at 19.4 s, near the end |
| `28e3a9ca` | 0.57 | 0.46 | Scoring stopped by drift at 7.5 s |
| `73ee2cdb` | 1.00 | 1.00 | Scoring stopped by drift at 19.9 s, at the end |
| `b6f09ef3` | 1.00 | 0.97 | Scored to the end |
| `b988494a` | 0.79 | 0.63 | Scoring stopped by drift at 12.4 s |
| `eda66f79` | 0.00 | 0.49 | Off-road at 8.8 s, in a work zone |

Validator flags were raised on 40% of decisions (speed 531, acceleration 78, waypoint
timing 13, deceleration 1). Most are speed flags against our first-guess 15 m/s limit,
which is too low for these roads, so this rate says little yet.

### The crash scene, three default runs

The same scene ran in stages 1, 2, and 3 with identical settings. Only the random
sampling differed: AlpaSim picks a new seed every run, and the model samples randomly.

| Run | Score | What happened |
|---|---|---|
| Stage 1 (AlpaSim's driver, no plugin) | 1.00 | Scoring stopped by drift at 10.4 s; unscored collisions at 12.8 s (rear), 13.8 s (front), 18.7 s (side) |
| Stage 2 (our plugin) | 0.00 | At-fault front collision at 11.4 s (scored) |
| Stage 3 (our plugin) | 1.00 | Scoring stopped by drift at 10.6 s; unscored collisions from 11.2 s |

- **Scores:** mean 0.67, spread (sample standard deviation) 0.58. Progress was 1.00 in all three.
- **Every run hit another vehicle at the front or side**, between 11.2 s and 13.8 s. AlpaSim counted it in one run only.
- **Latency per decision, p50:** 3.16 s (stage 2) and 3.15 s (stage 3). Stage 1 has no plugin logs.
- **Our plugin is not the cause.** All 182 trajectories in stage 2 match what AlpaSim received, to 17 µm, with identical timestamps and reasoning text.

### Experiment: NVIDIA's comparison settings on the crash scene (single trial)

AlpaSim ships NVIDIA's own Alpamayo 2 comparison settings, and they differ from the default in four ways:
- six cameras, adding rear left and rear right
- six trajectory candidates per step, sampled three at a time
- the `CLOSEST_LATERAL` rule: drive the candidate closest sideways to the previous plan
- a memory allocator setting (`expandable_segments`)

We ran them once on the crash scene, through our plugin (`run_stages.sh experiment`).
**This is a single trial: one random draw, not a measured improvement.**

| | Stage 2 (default) | Stage 3 (default) | Experiment |
|---|---|---|---|
| Scene score | 0.00 | 1.00 | 1.00 |
| Progress | 1.00 | 1.00 | 0.98 |
| How scoring ended | at-fault front collision, 11.4 s | drift, 10.6 s | hit from behind, 11.3 s |
| Unscored collisions after that | rear, 14.4 s | side 11.2 s, front 11.4 s, rear 12.5 s | front, 11.7 s |
| Latency per decision, p50 / p90 (n) | 3.16 / 4.17 s (182) | 3.15 / 3.87 s (182) | 8.29 / 10.66 s (182) |
| GPU 0 peak, driver (nvidia-smi) | 69.5 GiB | 69.5 GiB | 73.5 GiB |
| PyTorch peak allocated / reserved | 68.4 / 68.9 GiB | 68.4 / 68.9 GiB | 72.5 / 72.9 GiB |
| Jerk p95 / max | 3.6 / 9.4 m/s³ | 7.1 / 16.9 m/s³ | 5.8 / 12.2 m/s³ |
| Plan swing, 1.7–8 s (see below) | 3.3 m | 2.6 m | 0.8 m |
| Wall-clock time | 814 s | 796 s | 1,811 s |

"Plan swing" is how far the end point of the planned path moves sideways from one step
to the next, on average.

- **The plans were much steadier.** The swing was 0.8 m instead of about 3 m. After 8 s the
  car held its lane and braked to 0.4 m/s, with straight plans that matched its "keep
  distance" reasoning.
- **The cost:** each decision took 2.6 times as long, and the driver needed 4 GiB more GPU memory.
- **Jerk was no better:** p95 was 5.8 m/s³, between the two default runs.
- **The 1.00 is misleading here too.** A replayed truck, which cannot react, drove into the
  nearly stopped car from behind. That ended scoring, and a front contact 0.4 s later was never scored.

### Latency against real time

At 10 decisions per second, a car needs each decision in under 0.1 s.
- **Default settings:** p50 is 3.13 s, about 31 times too slow.
- **Comparison settings:** p50 is 8.29 s, about 83 times too slow.

This doesn't affect the scores, because the simulator waits for each decision.
It is the gap that compression and smaller GPUs would have to close.

## Key findings

1. **Single-sample plans are unstable.** With one sample per step, the plan's end point
   jumped sideways by about 3 m between steps on average. Just before the stage 2 crash it
   jumped by 7 m on average. The car followed each new plan, so the swings added up to a lane
   change into an occupied lane. With six candidates and `CLOSEST_LATERAL`, the swing fell
   to 0.8 m.
2. **The reasoning text and the plan often disagree.** Just before the stage 2 crash, the
   text kept saying "Keep distance to the lead vehicle because traffic ahead is congested",
   while the plans pointed up to 18 m to the left. Read the reasoning as a hint, not as an
   explanation of the trajectory.
3. **The renderer breaks down when the car leaves the recorded path.**
   - Once the car drifts, the scene is drawn from viewpoints the reconstruction never saw: smeared cars and floating shapes.
   - The reasoning in those moments mentions "limited visibility from rain".
   - We can't tell how much these images caused the later mistakes, but some failures after drift may be the renderer's, not the model's.
4. **A score of 1.00 can hide a crash.**
   - AlpaSim stopped scoring early because of drift in 5 of 8 scenes.
   - In the crash scene, two of the three default runs scored 1.00 and then collided after scoring stopped.
   - The experiment's 1.00 also ended with a front contact just after scoring stopped.
   - Our report now marks every drift cut, and lists each later collision as at fault or not, and scored or not.
5. **Traffic doesn't react.** Other vehicles replay the recording, so a car that does
   something other than the recorded driver can be hit by vehicles that never brake. That
   happened in the experiment.

## What to show

1. **The stage 2 crash, with reasoning** (findings 1, 2 and 3):
   `results/vm_20261027_195851/stage2/alpamayo2_super/clipgt-01d503d4-449b-46fc-8d78-9085e70d3554/rollouts/clipgt-01d503d4-449b-46fc-8d78-9085e70d3554/58329a0c-d245-11f1-b05a-09dde8096c31/clipgt-01d503d4-449b-46fc-8d78-9085e70d3554_58329a0c-d245-11f1-b05a-09dde8096c31_camera_front_wide_120fov_reasoning_overlay.mp4`.
   - Watch 9–11.5 s.
   - At 11.1 s the text says "Keep distance…" while the plotted plan curves about 18 m left, and the cars around are smeared.
2. **The work-zone off-road, a model error with a clean image:**
   `results/vm_20261027_210046/stage3/alpamayo2_super/clipgt-eda66f79-fa63-418c-97e6-14b91a0c615f/rollouts/clipgt-eda66f79-fa63-418c-97e6-14b91a0c615f/16085c40-d257-11f1-a65b-454b237da872/clipgt-eda66f79-fa63-418c-97e6-14b91a0c615f_16085c40-d257-11f1-a65b-454b237da872_camera_front_wide_120fov_reasoning_overlay.mp4`.
   - At 8.4 s the model plans a right lane change through the cones ("Change lanes to the right due to work-zone cones…").
   - It leaves the road at 8.8 s.
3. **The experiment, top-down view** (findings 4 and 5):
   `results/vm_exp_6cam_1/experiment_alpamayo2_comparison_6cam/alpamayo2_super/clipgt-01d503d4-449b-46fc-8d78-9085e70d3554/rollouts/clipgt-01d503d4-449b-46fc-8d78-9085e70d3554/80d9aaca-d25b-11f1-abd7-63e2c7763985/clipgt-01d503d4-449b-46fc-8d78-9085e70d3554_80d9aaca-d25b-11f1-abd7-63e2c7763985_camera_front_wide_120fov_default.mp4`.
   - Watch 10–12 s: the car holds its lane and slows almost to a stop, and the replayed truck behind runs into it.
   - At 11.6 s the metrics panel shows `collision_rear` 1 but `collision_at_fault` 0.

## Limitations

- **On-road scenes only.** All scenes are public-road drives; none are off-road.
- **Few scenes, few runs.**
  - Eight scenes with one run each, plus three runs of one scene. That scene scored 1.00, 0.00 and 1.00.
  - Per-scene scores are close to a coin flip, and the experiment is a single trial.
- **Scoring stops early.** Drift and the first collision end scoring, so later events don't count.
- **Rendering artifacts** once the car leaves the recorded path.
- **Non-reactive traffic.** Other vehicles replay the recording.
- **A100, not H100.**
  - NVIDIA tested Alpamayo 2 Super on H100 only.
  - Our latency isn't comparable to NVIDIA's.
  - It is comparable only between our runs with the same GPU layout.
- **Latency is model time only.** It is measured in the driver and excludes rendering and physics.
- **Default settings.** The benchmark uses AlpaSim's default four-camera, one-sample setting.
  NVIDIA's comparison setting looks steadier but is 2.6 times slower, and we tried it once.
- **Validator limits are first guesses.** For example, the 15 m/s speed limit flags normal highway driving.

## Next steps

1. **Repeat runs.** Run stage 3 two or three more times and combine the passes
   (`vla_on_edge.eval.combine`), to report a mean and spread per scene. Repeat the
   comparison settings the same way before drawing conclusions from them.
2. **Alpamayo 1.5.** Run the same three stages for Alpamayo 1.5 on one GPU, and compare it with 2 Super.
3. **General VLMs.** Add general vision-language models through the agent layer (see [ARCHITECTURE.md](ARCHITECTURE.md)).
4. **Compression.** Try quantization, pruning, and distillation. Measure how latency and memory move towards the
   0.1 s real-time budget, and what that costs in driving score.
5. **Off-road scenes.** Build or find off-road scenes; all current scenes are on-road.
6. **Calibration.** Calibrate the validator limits, and screen more scenes to swap in harder ones.
