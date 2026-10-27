"""Checks one scene's run folder before trusting it (used after stage 2 on the VM).

Usage: python -m vla_on_edge.eval.check_run <log_dir> [--plugin-logs]
Exits non-zero and prints each problem if anything is missing or malformed.
"""

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from vla_on_edge.utils.config import CONFIG_DIR, load_model_config
from vla_on_edge.utils.logging import LOG_FORMAT_VERSION

METRICS_COLUMNS = ["name", "timestamps_us", "values", "valid", "time_aggregation",
                   "clipgt_id", "rollout_id", "run_uuid", "run_name"]
# Share of our records whose sim time must fall inside AlpaSim's metric time range,
# to confirm both use the same clock (needed to find reasoning near a failure).
MIN_TIME_OVERLAP = 0.9


def check_alpasim_outputs(log_dir: Path) -> tuple[list[str], tuple[int, int] | None]:
    """Return (problems, (first, last) metric timestamp) for AlpaSim's own outputs."""
    summary_path = log_dir / "aggregate" / "results-summary.json"
    if not summary_path.exists():
        return [f"missing {summary_path}"], None
    rollouts = json.loads(summary_path.read_text())["rollouts"]
    if len(rollouts) != 1:
        return [f"expected 1 rollout in results-summary.json, found {len(rollouts)}"], None
    rollout = rollouts[0]
    if rollout["metrics"].get("error"):
        return [f"rollout failed inside AlpaSim: {rollout['metrics']['error']}"], None
    parquet = log_dir / "rollouts" / rollout["clipgt_id"] / rollout["rollout_id"] / "metrics.parquet"
    if not parquet.exists():
        return [f"missing {parquet}"], None
    metrics = pl.read_parquet(parquet)
    if metrics.columns != METRICS_COLUMNS:
        return [f"metrics.parquet columns {metrics.columns} differ from {METRICS_COLUMNS}"], None
    return [], (int(metrics["timestamps_us"].min()), int(metrics["timestamps_us"].max()))


def check_plugin_logs(log_dir: Path, time_range: tuple[int, int] | None, model: str) -> list[str]:
    """Return problems with our plugin's steps.jsonl."""
    steps_path = log_dir / "vla_on_edge" / "steps.jsonl"
    if not steps_path.exists():
        return [f"missing {steps_path} (was the wrapper used and VLA_ON_EDGE_LOG_DIR mounted?)"]
    records = []
    for number, line in enumerate(steps_path.read_text().splitlines(), start=1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            return [f"steps.jsonl line {number} is not JSON: {error}"]
    if not records:
        return ["steps.jsonl is empty"]

    problems = []
    expected_points = load_model_config(CONFIG_DIR, model, None)["trajectory"]["waypoint_count"]
    if any(record["log_format_version"] != LOG_FORMAT_VERSION for record in records):
        problems.append(f"some records are not log_format_version {LOG_FORMAT_VERSION}")
    if any(record["model"] != model for record in records):
        problems.append(f"some records are not from {model} (wrong wrapper?)")
    if any(len(record["executed_action"]["waypoints"]) != expected_points for record in records):
        problems.append(f"some trajectories do not have {expected_points} waypoints")
    if time_range is not None:
        first, last = time_range
        inside = [r for r in records if r["sim_time_us"] is not None and first <= r["sim_time_us"] <= last]
        if len(inside) < MIN_TIME_OVERLAP * len(records):
            problems.append(f"only {len(inside)}/{len(records)} records have sim_time_us inside AlpaSim's "
                            f"metric time range [{first}, {last}]: the clocks may differ")
    return problems


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--plugin-logs", action="store_true", help="also check our wrapper's step logs")
    parser.add_argument("--model", default="alpamayo1_5", help="models.yaml entry the run used")
    args = parser.parse_args()

    problems, time_range = check_alpasim_outputs(args.log_dir)
    if args.plugin_logs:
        problems += check_plugin_logs(args.log_dir, time_range, args.model)
    for problem in problems:
        print(f"PROBLEM: {problem}")
    if problems:
        sys.exit(1)
    print(f"OK: {args.log_dir}")


if __name__ == "__main__":
    main()
