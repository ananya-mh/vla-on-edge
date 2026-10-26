"""Computes benchmark metrics from a finished stage folder.

Expected layout (written by scripts/run_stages.sh, one AlpaSim run per scene):
    <stage_dir>/<model>/<scene_id>/            AlpaSim's wizard.log_dir
        aggregate/results-summary.json         AlpaSim's per-rollout scores and metrics
        rollouts/<scene_id>/<rollout>/metrics.parquet   per-timestep metrics
        vla_on_edge/steps.jsonl                our plugin's per-prediction records

Driving metrics (score, collision, off-road, progress) are AlpaSim's own; we only
read them. Jerk, latency, GPU memory, and validator flags come from our logs,
because AlpaSim does not compute them per model.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from vla_on_edge.utils.logging import read_jsonl

# A percentile is only shown when there are enough samples for it to mean something.
P99_MIN_SAMPLES = 1000
EVENT_METRICS = ("collision_front", "collision_lateral", "offroad")


def rollout_start_us(metrics_parquet: Path) -> int:
    """Return the rollout's first timestamp."""
    return int(pl.read_parquet(metrics_parquet, columns=["timestamps_us"])["timestamps_us"].min())


def first_event_time_us(metrics_parquet: Path) -> int | None:
    """Return the first driven timestep with an at-fault collision or off-road, if any."""
    long = pl.read_parquet(metrics_parquet)
    wide = long.pivot(on="name", index="timestamps_us", values="values").sort("timestamps_us")
    driven = wide.filter(pl.col("eval_relevant") > 0)
    present = [name for name in EVENT_METRICS if name in driven.columns]
    if not present:
        return None
    any_event = pl.any_horizontal([pl.col(name) > 0 for name in present])
    events = driven.filter(any_event)
    return int(events["timestamps_us"].min()) if len(events) else None


def jerk_values(records: list[dict]) -> list[float]:
    """Return |d(acceleration)/dt| between consecutive predictions (m/s^3), in sim time order."""
    timed = sorted((r for r in records if r["sim_time_us"] is not None), key=lambda r: r["sim_time_us"])
    jerks = []
    for previous, current in zip(timed, timed[1:]):
        dt_s = (current["sim_time_us"] - previous["sim_time_us"]) / 1e6
        if dt_s > 0:
            jerks.append(abs(current["ego_acceleration_mps2"] - previous["ego_acceleration_mps2"]) / dt_s)
    return jerks


def violation_rule(message: str) -> str:
    """Turn a violation message into its rule name, e.g. 'speed 16.2 m/s ...' -> 'speed'."""
    return message.split()[0]


def load_scene(log_dir: Path) -> dict:
    """Read one scene's AlpaSim outputs and (if present) our plugin logs."""
    summary = json.loads((log_dir / "aggregate" / "results-summary.json").read_text())
    if len(summary["rollouts"]) != 1:
        raise ValueError(f"{log_dir}: expected exactly one rollout, found {len(summary['rollouts'])}")
    rollout = summary["rollouts"][0]
    rollout_dir = log_dir / "rollouts" / rollout["clipgt_id"] / rollout["rollout_id"]
    metrics_parquet = rollout_dir / "metrics.parquet"
    steps_path = log_dir / "vla_on_edge" / "steps.jsonl"
    return {
        "scene_id": rollout["clipgt_id"],
        "rollout": rollout,
        "rollout_dir": rollout_dir,
        "event_time_us": first_event_time_us(metrics_parquet) if metrics_parquet.exists() else None,
        "start_time_us": rollout_start_us(metrics_parquet) if metrics_parquet.exists() else None,
        "records": read_jsonl(steps_path) if steps_path.exists() else [],
        "videos": sorted(rollout_dir.glob("*.mp4")),
    }


def scene_row(scene: dict) -> dict:
    """Return the per-scenario table row for one scene."""
    rollout, records = scene["rollout"], scene["records"]
    metrics = rollout["metrics"]
    jerks = jerk_values(records)
    latencies = [r["latency_s"]["batch"] for r in records]
    flagged = [r for r in records if r["validation_violations"]]
    return {
        "scene_id": scene["scene_id"],
        "score": rollout["score"],
        "failure_reason": rollout["failure_reason"],
        "progress": metrics.get("progress_clipped_rel"),
        "collision_at_fault": metrics.get("collision_at_fault"),
        "collision_any": metrics.get("collision_any"),
        "offroad": metrics.get("offroad"),
        "jerk_p95": float(np.percentile(jerks, 95)) if jerks else None,
        "jerk_max": max(jerks) if jerks else None,
        "latency_p50_s": float(np.percentile(latencies, 50)) if latencies else None,
        "n_predictions": len(records),
        "flag_rate": len(flagged) / len(records) if records else None,
    }


def latency_summary(latencies: list[float]) -> dict:
    """p50/p90 always, p99 only with enough samples; n is always reported."""
    if not latencies:
        return {"n": 0, "p50": None, "p90": None, "p99": None}
    return {
        "n": len(latencies),
        "p50": float(np.percentile(latencies, 50)),
        "p90": float(np.percentile(latencies, 90)),
        "p99": float(np.percentile(latencies, 99)) if len(latencies) >= P99_MIN_SAMPLES else None,
    }


def mean_or_none(values: list) -> float | None:
    """Return the mean of the non-None values, or None if there are none."""
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def model_summary(scenes: list[dict]) -> dict:
    """Return the headline row for one model over all its scenes."""
    rows = [scene_row(scene) for scene in scenes]
    records = [record for scene in scenes for record in scene["records"]]
    jerks = [value for scene in scenes for value in jerk_values(scene["records"])]
    memories = [r["gpu_memory_mib"] for r in records if r["gpu_memory_mib"] is not None]
    rule_counts: dict[str, int] = {}
    for record in records:
        for rule in {violation_rule(message) for message in record["validation_violations"]}:
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
    return {
        "n_scenes": len(rows),
        "driving_score": mean_or_none([row["score"] for row in rows]),
        "collision_rate": mean_or_none([float(row["collision_at_fault"] > 0) for row in rows]),
        "offroad_rate": mean_or_none([float(row["offroad"] > 0) for row in rows]),
        "progress": mean_or_none([row["progress"] for row in rows]),
        "jerk_p95": float(np.percentile(jerks, 95)) if jerks else None,
        "latency": latency_summary([r["latency_s"]["batch"] for r in records]),
        "gpu_peak_allocated_gib": max(m["peak_allocated"] for m in memories) / 1024 if memories else None,
        "gpu_peak_reserved_gib": max(m["peak_reserved"] for m in memories) / 1024 if memories else None,
        "n_predictions": len(records),
        "flag_rate": sum(1 for r in records if r["validation_violations"]) / len(records) if records else None,
        "flags_by_rule": dict(sorted(rule_counts.items())),
        "rows": rows,
    }


def load_stage(stage_dir: Path) -> dict[str, list[dict]]:
    """Return {model name: [scene, ...]} for every scene folder in a stage."""
    models = {}
    for model_dir in sorted(path for path in stage_dir.iterdir() if path.is_dir()):
        scene_dirs = sorted(path for path in model_dir.iterdir() if (path / "aggregate").is_dir())
        models[model_dir.name] = [load_scene(scene_dir) for scene_dir in scene_dirs]
    return models
