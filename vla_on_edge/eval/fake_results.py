"""Writes fake AlpaSim outputs plus fake plugin logs, to test the report offline.

The files follow the schemas in AlpaSim's source (third_party/alpasim, commit
affc2eab), so the report reads them exactly as it will read a real run:
- rollouts/<scene>/<rollout>/metrics.parquet: long format, one row per (metric,
  timestep), written by eval.data.create_metrics_dataframe
- aggregate/results-summary.json: eval.aggregation.processing.write_results_summary_json,
  with rows truncated and scored the way AlpaSim does (aggregation/scene_score.py)
- aggregate/metrics_unprocessed.parquet and metrics_results.parquet
- vla_on_edge/steps.jsonl: our plugin's records (log_format_version 2)

Each scene gets a "story" (clean drive, collision, off-road, ...) so the report has
something to rank. Usage: python -m vla_on_edge.eval.fake_results <out_dir>
"""

import hashlib
import json
import math
import random
import sys
import uuid
from pathlib import Path

import polars as pl

from vla_on_edge.utils.config import CONFIG_DIR, load_alpasim_scenes
from vla_on_edge.utils.logging import LOG_FORMAT_VERSION, append_jsonl

STEP_US = 100_000  # AlpaSim's default 10 Hz control step
N_STEPS = 200  # runtime.simulation_config.n_sim_steps default
FORCE_GT_US = 1_700_000  # runtime.simulation_config.force_gt_duration_us default
START_US = 10_000_000_000
RUN_NAME = "LR-fake"

# What happens in each fake scene, in scene-list order.
STORIES = [
    {"event": None, "event_s": None, "progress": 0.97, "speed": 9.0},
    {"event": "collision_front", "event_s": 8.3, "progress": 0.45, "speed": 10.0},
    {"event": "offroad", "event_s": 12.1, "progress": 0.62, "speed": 8.5},
    {"event": None, "event_s": None, "progress": 0.71, "speed": 6.0},
    {"event": "collision_lateral", "event_s": 15.4, "progress": 0.80, "speed": 9.5},
    {"event": None, "event_s": None, "progress": 0.99, "speed": 9.2},
    {"event": None, "event_s": None, "progress": 0.88, "speed": 8.8},
    {"event": None, "event_s": None, "progress": 0.93, "speed": 9.1},
]
REASONING = [
    "Lead vehicle ahead is slowing; keep distance and follow the lane.",
    "Clear road ahead; maintain speed and stay centered in the lane.",
    "Pedestrian near the crosswalk on the right; prepare to yield.",
    "Lane curves left; reduce speed slightly through the curve.",
]


def is_after_event(story: dict, t_s: float) -> bool:
    return story["event"] is not None and t_s >= story["event_s"]


def metric_series(story: dict, t_s: float, duration_s: float) -> dict[str, tuple[str, float]]:
    """Return {metric name: (time aggregation, value)} at time t_s."""
    fraction = t_s / duration_s
    progress = story["progress"] * fraction
    drifting = story["event"] == "offroad" and t_s > story["event_s"] - 3.0
    return {
        "collision_front": ("max", float(story["event"] == "collision_front" and is_after_event(story, t_s))),
        "collision_rear": ("max", 0.0),
        "collision_lateral": ("max", float(story["event"] == "collision_lateral" and is_after_event(story, t_s))),
        "collision_any": ("max", float(story["event"] in ("collision_front", "collision_lateral")
                                       and is_after_event(story, t_s))),
        "offroad": ("max", float(story["event"] == "offroad" and is_after_event(story, t_s))),
        "wrong_lane": ("max", 0.0),
        "min_distance_to_lane_boundary_m": ("min", 0.2 if drifting else 0.9),
        "min_distance_to_obstacle_m": ("min", max(0.0, 12.0 - t_s) if story["event"] == "collision_front" else 6.0),
        "open_loop_collision": ("mean", 0.0),
        "progress": ("last", progress),
        "progress_rel_to_total": ("last", progress),
        "progress_rel": ("min", progress),
        "dist_to_gt_trajectory": ("max", 2.5 if drifting else 0.3 + 0.1 * math.sin(t_s)),
        "dist_to_gt_location": ("max", 0.5 + 0.1 * math.sin(t_s)),
        "dist_traveled_m": ("last", story["speed"] * t_s),
        "gt_dist_traveled_m": ("last", 9.0 * t_s),
        "plan_deviation": ("mean", 0.15),
        "img_is_black": ("max", 0.0),
        "safety_monitor_triggered": ("max", 0.0),
        "eval_relevant": ("max", float(t_s * 1e6 >= FORCE_GT_US)),
    }


def rollout_rows(story: dict, scene_id: str, rollout_id: str, run_uuid: str) -> list[dict]:
    """Return the long-format rows of one rollout's metrics.parquet."""
    rows = []
    duration_s = N_STEPS * STEP_US / 1e6
    for step in range(N_STEPS):
        t_s = step * STEP_US / 1e6
        for name, (aggregation, value) in metric_series(story, t_s, duration_s).items():
            rows.append({"name": name, "timestamps_us": START_US + step * STEP_US, "values": float(value),
                         "valid": True, "time_aggregation": aggregation, "clipgt_id": scene_id,
                         "rollout_id": rollout_id, "run_uuid": run_uuid, "run_name": RUN_NAME})
    return rows


def aggregate_rollout(rows: list[dict]) -> dict:
    """Reduce one rollout to per-metric values the way AlpaSim's aggregation does.

    Keep timesteps from the first eval_relevant one, add the combined events, cut
    after the first collision/off-road or dist_to_gt_trajectory >= 4 m, then apply
    each metric's time aggregation (aggregation/processing.py, modifiers.py).
    """
    wide = pl.DataFrame(rows).pivot(on="name", index="timestamps_us", values="values").sort("timestamps_us")
    aggregations = {row["name"]: row["time_aggregation"] for row in rows}
    wide = wide.filter(pl.col("timestamps_us") >= wide.filter(pl.col("eval_relevant") > 0)["timestamps_us"].min())
    wide = wide.with_columns(
        offroad_or_collision=((pl.col("collision_any") > 0) | (pl.col("offroad") > 0)).cast(pl.Float64),
        collision_at_fault=((pl.col("collision_front") > 0) | (pl.col("collision_lateral") > 0)).cast(pl.Float64),
    )
    wide = wide.with_columns(
        offroad_or_collision_at_fault=((pl.col("collision_at_fault") > 0) | (pl.col("offroad") > 0)).cast(pl.Float64),
        duration_frac_20s=(pl.col("timestamps_us") - pl.col("timestamps_us").min()) / 20e6,
    )
    for condition in (pl.col("offroad_or_collision") > 0, pl.col("dist_to_gt_trajectory") >= 4.0):
        events = wide.filter(condition)
        if len(events):
            wide = wide.filter(pl.col("timestamps_us") <= events["timestamps_us"].min())
    aggregations.update(offroad_or_collision="max", collision_at_fault="max",
                        offroad_or_collision_at_fault="max", duration_frac_20s="last")
    result = {}
    for name, aggregation in aggregations.items():
        column = wide[name]
        result[name] = float(column[-1] if aggregation == "last" else getattr(column, aggregation)())
    result["progress_clipped_rel"] = result["progress_rel_to_total"]
    return result


def score_rollout(metrics: dict) -> dict:
    """AlpaSim's scene score with default config (threshold 0.8, min GT distance 5 m)."""
    if metrics["gt_dist_traveled_m"] < 5.0:
        progress_score = 1.0
    else:
        progress_score = min(min(max(metrics["progress_clipped_rel"], 0.0), 1.0) / 0.8, 1.0)
    failure_reason = None
    if metrics["collision_at_fault"] != 0.0:
        failure_reason = "collision_at_fault"
    elif metrics["offroad"] != 0.0:
        failure_reason = "offroad"
    score = 0.0 if failure_reason else progress_score
    return {"score": score, "progress_score": progress_score, "failure_reason": failure_reason}


def summary_row(metrics: dict, scene_id: str, rollout_id: str, run_uuid: str) -> dict:
    """One entry of results-summary.json's `rollouts` list."""
    scored = score_rollout(metrics)
    passed = scored["failure_reason"] is None
    return {
        "run_uuid": run_uuid, "run_name": RUN_NAME, "clipgt_id": scene_id, "rollout_id": rollout_id,
        "metrics": {**metrics, "run_name": RUN_NAME, "run_uuid": run_uuid, "clipgt_id": scene_id,
                    "rollout_id": rollout_id},
        "status": "pass" if passed else "fail", "passed": passed, "score": scored["score"],
        "failure_reason": scored["failure_reason"],
        "score_metrics": {
            "progress_clipped_rel": metrics["progress_clipped_rel"], "progress_rel": metrics["progress_rel"],
            "progress_score": scored["progress_score"], "collision_at_fault": metrics["collision_at_fault"],
            "offroad": metrics["offroad"], "dist_to_gt_trajectory": metrics["dist_to_gt_trajectory"],
            "gt_dist_traveled_m": metrics["gt_dist_traveled_m"],
        },
    }


def write_alpasim_outputs(log_dir: Path, scene_id: str, story: dict) -> None:
    """Write one scene's AlpaSim log dir: per-rollout parquet and the aggregate files."""
    run_uuid = str(uuid.uuid4())
    rollout_id = str(uuid.uuid1())
    rows = rollout_rows(story, scene_id, rollout_id, run_uuid)
    rollout_dir = log_dir / "rollouts" / scene_id / rollout_id
    rollout_dir.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(rollout_dir / "metrics.parquet")

    # Aggregation replaces run_uuid with a hash of the runs (force_same_run=True).
    hashed_run = hashlib.sha256(run_uuid.encode()).hexdigest()
    metrics = aggregate_rollout(rows)
    aggregate_dir = log_dir / "aggregate"
    aggregate_dir.mkdir()
    pl.DataFrame(rows).with_columns(run_uuid=pl.lit(hashed_run)).write_parquet(aggregate_dir / "metrics_unprocessed.parquet")
    run_row = {"run_uuid": hashed_run, **metrics, "n_clips": 1, "n_rollouts": 1, "run_name": RUN_NAME}
    pl.DataFrame([run_row]).write_parquet(aggregate_dir / "metrics_results.parquet")
    summary = {
        "schema_version": 1,
        "scene_score_enabled": True,
        "rollouts": [summary_row(metrics, scene_id, rollout_id, hashed_run)],
        "metrics_results": [run_row],
        "score_criteria": {
            "progress_score": "min(clamp(progress_clipped_rel, 0, 1) / 0.8, 1.0)",
            "short_gt_distance_override": "progress_score = 1.0 when gt_dist_traveled_m < 5.0",
            "collision_at_fault": "== 0",
            "offroad": "== 0",
        },
    }
    (aggregate_dir / "results-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def plugin_record(story: dict, step: int, rng: random.Random, model: str = "alpamayo1_5") -> dict:
    """One fake plugin step record, shaped like LoggedModel.build_record."""
    t_s = step * STEP_US / 1e6
    acceleration = 0.4 * math.sin(t_s)
    violations = []
    if story["event"] and abs(t_s - story["event_s"]) < 1.0:
        acceleration = -5.0  # hard braking just before the event
        violations.append(f"deceleration 7.2 m/s^2 around t={t_s:.2f} s exceeds 6.0")
    elif rng.random() < 0.03:
        violations.append(f"speed 15.4 m/s around t=0.25 s exceeds 15.0 m/s")
    waypoints = [{"x": story["speed"] * 0.1 * (i + 1), "y": 0.0, "t": round(0.1 * (i + 1), 1), "z": 0.0,
                  "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]} for i in range(64)]
    return {
        "log_format_version": LOG_FORMAT_VERSION, "model": model, "wall_time_s": 1.8e9 + step,
        "sim_time_us": START_US + step * STEP_US, "inference_seed": step, "command": "STRAIGHT",
        "ego_speed_mps": story["speed"], "ego_acceleration_mps2": acceleration,
        "executed_action": {"waypoints": waypoints}, "validation_violations": violations,
        "num_candidates": 1, "selected_index": 0,
        "reasoning_text": REASONING[(step // 10) % len(REASONING)],
        "latency_s": {"batch": rng.lognormvariate(math.log(0.85), 0.15), "batch_size": 1},
        "gpu_memory_mib": {"peak_allocated": 22800.0 + rng.random() * 50, "peak_reserved": 24100.0},
    }


def write_plugin_logs(log_dir: Path, story: dict, rng: random.Random, model: str = "alpamayo1_5") -> None:
    """Write vla_on_edge/steps.jsonl for the driven (post force-GT) steps of one scene."""
    plugin_dir = log_dir / "vla_on_edge"
    plugin_dir.mkdir()
    first_driven_step = FORCE_GT_US // STEP_US
    for step in range(first_driven_step, N_STEPS):
        append_jsonl(plugin_dir / "steps.jsonl", plugin_record(story, step, rng, model))


def write_fake_run(stage_dir: Path, model: str = "alpamayo1_5", seed: int = 0) -> Path:
    """Write fake results for every configured scene under <stage_dir>/<model>/<scene>/."""
    rng = random.Random(seed)
    scene_ids = [scene["id"] for scene in load_alpasim_scenes(CONFIG_DIR)["scenes"]]
    for scene_id, story in zip(scene_ids, STORIES):
        log_dir = stage_dir / model / scene_id
        write_alpasim_outputs(log_dir, scene_id, story)
        write_plugin_logs(log_dir, story, rng, model)
    return stage_dir


if __name__ == "__main__":
    print(write_fake_run(Path(sys.argv[1])))
