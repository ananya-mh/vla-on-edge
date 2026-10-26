"""Builds the Markdown benchmark report (tables, charts, worst cases) from a stage folder.

Usage: python -m vla_on_edge.eval.report <stage_dir> [--output results/report.md]
Charts are written next to the report, in <output dir>/charts/.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on the VM
import matplotlib.pyplot as plt  # noqa: E402

from vla_on_edge.eval.metrics import P99_MIN_SAMPLES, load_stage, model_summary  # noqa: E402

WORST_CASE_COUNT = 3
REASONING_WINDOW_US = 2_000_000  # show reasoning from the 2 s before the failure
REASONING_SHOWN = 3


def fmt(value, digits: int = 2, suffix: str = "") -> str:
    """Format a number for a table cell; missing values become 'n/a'."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def percent(value) -> str:
    """Format a 0–1 fraction as a percentage; missing values become 'n/a'."""
    return "n/a" if value is None else f"{100 * value:.0f}%"


def headline_table(summaries: dict[str, dict]) -> list[str]:
    """One row per model."""
    lines = [
        "| Model | Scenes | Driving score | Collision (at fault) | Off-road | Progress | Jerk p95 (m/s³) "
        "| Latency p50 / p90 / p99 (s), n | GPU peak alloc / reserved (GiB) | Validator flags |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model, summary in summaries.items():
        latency = summary["latency"]
        p99 = fmt(latency["p99"]) if latency["p99"] is not None else f"— (n<{P99_MIN_SAMPLES})"
        lines.append(
            f"| {model} | {summary['n_scenes']} | {fmt(summary['driving_score'])} "
            f"| {percent(summary['collision_rate'])} | {percent(summary['offroad_rate'])} "
            f"| {fmt(summary['progress'])} | {fmt(summary['jerk_p95'], 1)} "
            f"| {fmt(latency['p50'])} / {fmt(latency['p90'])} / {p99}, n={latency['n']} "
            f"| {fmt(summary['gpu_peak_allocated_gib'], 1)} / {fmt(summary['gpu_peak_reserved_gib'], 1)} "
            f"| {percent(summary['flag_rate'])} of {summary['n_predictions']} |"
        )
    return lines


def scenario_table(model: str, summary: dict) -> list[str]:
    """One row per scene for one model."""
    lines = [
        f"### {model}",
        "",
        "| Scene | Score | Failure | Progress | Collision | Off-road | Jerk p95 / max (m/s³) | Latency p50 (s) | Flags |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| `{row['scene_id']}` | {fmt(row['score'])} | {row['failure_reason'] or '—'} "
            f"| {fmt(row['progress'])} | {'yes' if row['collision_at_fault'] else 'no'} "
            f"| {'yes' if row['offroad'] else 'no'} | {fmt(row['jerk_p95'], 1)} / {fmt(row['jerk_max'], 1)} "
            f"| {fmt(row['latency_p50_s'])} | {percent(row['flag_rate'])} |"
        )
    return lines


def reasoning_near_failure(scene: dict) -> list[tuple[float, str]]:
    """Return (seconds before the failure, text) for the last distinct reasoning texts.

    Uses the 2 s before the first collision/off-road; without one, the end of the run.
    """
    timed = [r for r in scene["records"] if r["sim_time_us"] is not None and r["reasoning_text"]]
    if not timed:
        return []
    end_us = scene["event_time_us"] or max(r["sim_time_us"] for r in timed)
    window = [r for r in timed if end_us - REASONING_WINDOW_US <= r["sim_time_us"] <= end_us]
    shown, seen = [], set()
    for record in sorted(window, key=lambda r: r["sim_time_us"], reverse=True):
        if record["reasoning_text"] not in seen:
            seen.add(record["reasoning_text"])
            shown.append(((end_us - record["sim_time_us"]) / 1e6, record["reasoning_text"]))
        if len(shown) == REASONING_SHOWN:
            break
    return shown


def worst_cases(stage: dict[str, list[dict]], report_dir: Path) -> list[str]:
    """The lowest-scoring rollouts, with what the model said just before failing."""
    scenes = [(model, scene) for model, model_scenes in stage.items() for scene in model_scenes]
    scenes.sort(key=lambda item: (item[1]["rollout"]["score"], item[1]["rollout"]["metrics"].get("progress_clipped_rel", 0)))
    lines = []
    for model, scene in scenes[:WORST_CASE_COUNT]:
        rollout = scene["rollout"]
        lines += ["", f"### {model} on `{scene['scene_id']}`: score {fmt(rollout['score'])}, "
                      f"{rollout['failure_reason'] or 'no hard failure (low progress)'}", ""]
        if scene["event_time_us"] is not None:
            seconds_in = (scene["event_time_us"] - scene["start_time_us"]) / 1e6
            lines.append(f"First collision/off-road {seconds_in:.1f} s into the rollout.")
        for video in scene["videos"]:
            lines.append(f"Video: [{video.name}]({os.path.relpath(video, report_dir)})")
        reasoning = reasoning_near_failure(scene)
        if reasoning:
            lines += ["", "Model reasoning, most recent first:"]
            lines += [f"- {seconds:.1f} s before: \"{text}\"" for seconds, text in reasoning]
        else:
            lines.append("No reasoning text logged (no plugin logs for this run).")
    return lines


def plot_charts(summaries: dict[str, dict], stage: dict[str, list[dict]], chart_dir: Path) -> list[Path]:
    """Score by scenario, latency distribution, and validator flags by rule."""
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    figure, axis = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(len(summaries), 1)
    for index, (model, summary) in enumerate(summaries.items()):
        positions = [i + index * width for i in range(len(summary["rows"]))]
        bars = axis.bar(positions, [row["score"] for row in summary["rows"]], width=width, label=model)
        axis.bar_label(bars, fmt="%.2f")  # a failed scene (score 0) would otherwise be invisible
    first_rows = next(iter(summaries.values()))["rows"]
    axis.set_xticks(range(len(first_rows)), [row["scene_id"][7:15] for row in first_rows], rotation=45)
    axis.set_ylabel("AlpaSim driving score")
    axis.set_ylim(0, 1.05)
    axis.legend()
    paths.append(save(figure, chart_dir / "score_by_scenario.png"))

    figure, axis = plt.subplots(figsize=(6, 4))
    for model, scenes in stage.items():
        latencies = [r["latency_s"]["batch"] for scene in scenes for r in scene["records"]]
        if latencies:
            axis.hist(latencies, bins=40, alpha=0.6, label=f"{model} (n={len(latencies)})")
    axis.set_xlabel("Decision latency (s, batch size 1)")
    axis.set_ylabel("Predictions")
    axis.legend()
    paths.append(save(figure, chart_dir / "latency_distribution.png"))

    figure, axis = plt.subplots(figsize=(6, 4))
    for model, summary in summaries.items():
        rules = summary["flags_by_rule"]
        axis.bar([f"{model}\n{rule}" for rule in rules], list(rules.values()))
    axis.set_ylabel("Predictions flagged (flag-only, never replaced)")
    paths.append(save(figure, chart_dir / "validator_flags.png"))
    return paths


def save(figure, path: Path) -> Path:
    """Save and close a figure."""
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def gpu_monitor_section(stage_dir: Path) -> list[str]:
    """Peak nvidia-smi memory per GPU and per container, if scripts/gpu_monitor.py ran.

    This includes the renderer, unlike the plugin's PyTorch numbers.
    """
    peaks_path = stage_dir.parent / "gpu_memory_peaks.json"
    if not peaks_path.exists():
        return []
    peaks = json.loads(peaks_path.read_text()).get(stage_dir.name, {})
    if not peaks:
        return []
    lines = ["## GPU memory (nvidia-smi peaks)", "", "| GPU or container | Peak (GiB) |", "|---|---|"]
    lines += [f"| {key} | {value / 1024:.1f} |" for key, value in sorted(peaks.items())]
    return lines + [""]


def build_report(stage_dir: Path, output: Path) -> Path:
    """Write the Markdown report and its charts; return the report path."""
    stage = load_stage(stage_dir)
    summaries = {model: model_summary(scenes) for model, scenes in stage.items()}
    report_dir = output.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    charts = plot_charts(summaries, stage, report_dir / "charts")

    lines = [
        "# Driving benchmark report",
        "",
        f"Source: `{stage_dir}`. Driving metrics are AlpaSim's own (scene score: 0 on an at-fault "
        "collision or off-road, otherwise min(progress / 0.8, 1)). Jerk, latency, GPU memory, and "
        "validator flags come from our plugin logs. Latency is per prediction at batch size 1; p99 "
        f"is shown only with at least {P99_MIN_SAMPLES} samples. GPU memory is PyTorch's peak in "
        "the driver process (the renderer runs separately). Validator flags are flag-only: "
        "trajectories are never replaced.",
        "",
        "## Headline",
        "",
        *headline_table(summaries),
        "",
        "## Per scenario",
        "",
    ]
    for model, summary in summaries.items():
        lines += [*scenario_table(model, summary), ""]
    lines += gpu_monitor_section(stage_dir)
    lines += ["## Charts", ""]
    lines += [f"![{path.stem}](charts/{path.name})" for path in charts]
    lines += ["", "## Worst cases", *worst_cases(stage, report_dir), ""]
    output.write_text("\n".join(lines))
    return output


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/report.md"))
    args = parser.parse_args()
    print(build_report(args.stage_dir, args.output))


if __name__ == "__main__":
    main()
