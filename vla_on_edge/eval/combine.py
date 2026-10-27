"""Combines several stage 3 passes (separate RUN_IDs) into one Markdown report.

Usage:
    python -m vla_on_edge.eval.combine results/vm_<A>/stage3 results/vm_<B>/stage3 ... \
        --output results/combined_report.md

Each pass is one rollout per scene, and Alpamayo samples at random, so the same scene
can score 1.00 in one pass and 0.00 in the next. This report shows, per scene and
model, the number of passes and the mean and spread (standard deviation, min–max)
of score, progress, and latency, and how many passes had an at-fault collision,
an off-road, or scoring cut short by drift.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from vla_on_edge.eval.metrics import latency_summary, load_stage, scene_row
from vla_on_edge.eval.report import fmt


def spread(values: list) -> dict:
    """Mean, sample standard deviation (None for fewer than 2), min, max, and n of the non-None values."""
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def fmt_spread(stats: dict, digits: int = 2) -> str:
    """Format as 'mean ± sd (min–max)'; 'n/a' without values."""
    if stats["n"] == 0:
        return "n/a"
    sd = f" ± {stats['sd']:.{digits}f}" if stats["sd"] is not None else ""
    return f"{stats['mean']:.{digits}f}{sd} ({stats['min']:.{digits}f}–{stats['max']:.{digits}f})"


def count(flags: list[bool]) -> str:
    """'k of n' for a list of yes/no outcomes."""
    return f"{sum(flags)} of {len(flags)}"


def pass_commit(scenes: list[dict]) -> str:
    """The git commit our plugin recorded in this pass (from a scene's metadata.json), if any."""
    for scene in scenes:
        metadata_path = scene["rollout_dir"].parents[2] / "vla_on_edge" / "metadata.json"
        if metadata_path.exists():
            return json.loads(metadata_path.read_text())["git"]["commit"][:7]
    return "unknown"


def load_passes(stage_dirs: list[Path]) -> tuple[list[dict], dict[str, dict[str, list[dict]]]]:
    """Return (pass descriptions, {model: {scene_id: [scene row with its records, one per pass]}})."""
    passes, by_model = [], {}
    for pass_index, stage_dir in enumerate(stage_dirs):
        stage = load_stage(stage_dir)
        all_scenes = [scene for scenes in stage.values() for scene in scenes]
        passes.append({"dir": stage_dir, "n_scenes": len(all_scenes), "commit": pass_commit(all_scenes)})
        for model, scenes in stage.items():
            for scene in scenes:
                row = {**scene_row(scene), "records": scene["records"], "pass_index": pass_index}
                by_model.setdefault(model, {}).setdefault(scene["scene_id"], []).append(row)
    return passes, by_model


def scene_lines(model: str, scenes: dict[str, list[dict]]) -> list[str]:
    """One row per scene: passes, score/progress/latency spread, and event counts."""
    lines = [
        f"### {model}",
        "",
        "| Scene | Passes | Score | At-fault collision | Off-road | Progress | Scoring cut by drift | Latency p50 (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for scene_id, rows in sorted(scenes.items()):
        lines.append(
            f"| `{scene_id}` | {len(rows)} | {fmt_spread(spread([r['score'] for r in rows]))} "
            f"| {count([bool(r['collision_at_fault']) for r in rows])} "
            f"| {count([bool(r['offroad']) for r in rows])} "
            f"| {fmt_spread(spread([r['progress'] for r in rows]))} "
            f"| {count([r['drift_cut_s'] is not None for r in rows])} "
            f"| {fmt_spread(spread([r['latency_p50_s'] for r in rows]))} |"
        )
    return lines


def model_line(model: str, scenes: dict[str, list[dict]]) -> str:
    """Headline row: score over all scene-passes, its pass-to-pass spread, event rates, pooled latency."""
    rows = [row for scene_rows in scenes.values() for row in scene_rows]
    scores_by_pass: dict[int, list[float]] = {}
    for row in rows:
        scores_by_pass.setdefault(row["pass_index"], []).append(row["score"])
    per_pass_means = [float(np.mean(scores)) for _, scores in sorted(scores_by_pass.items())]
    latency = latency_summary([r["latency_s"]["batch"] for row in rows for r in row["records"]])
    return (
        f"| {model} | {len(per_pass_means)} | {len(scenes)} | {fmt(float(np.mean([r['score'] for r in rows])))} "
        f"| {fmt_spread(spread(per_pass_means))} | {count([bool(r['collision_at_fault']) for r in rows])} "
        f"| {count([bool(r['offroad']) for r in rows])} | {count([r['drift_cut_s'] is not None for r in rows])} "
        f"| {fmt(latency['p50'])} / {fmt(latency['p90'])}, n={latency['n']} |"
    )


def build_combined_report(stage_dirs: list[Path], output: Path) -> Path:
    """Write the combined report; return its path."""
    passes, by_model = load_passes(stage_dirs)
    lines = [
        "# Combined benchmark report",
        "",
        f"{len(passes)} passes, each one rollout per scene. Spread is shown as mean ± sample standard "
        "deviation (min–max) over passes. Driving metrics are AlpaSim's own; latency is per prediction "
        "from our plugin logs. \"Scoring cut by drift\" counts passes where AlpaSim stopped scoring "
        "because the ego left the recorded drive, so a high score there covers only part of the drive.",
        "",
        "## Passes",
        "",
        "| Pass | Folder | Scenes | Commit |",
        "|---|---|---|---|",
        *[f"| {i + 1} | `{p['dir']}` | {p['n_scenes']} | {p['commit']} |" for i, p in enumerate(passes)],
        "",
        "## Headline",
        "",
        "| Model | Passes | Scenes | Mean score (all scene-passes) | Per-pass mean score | At-fault collision "
        "| Off-road | Scoring cut by drift | Latency p50 / p90 (s), n |",
        "|---|---|---|---|---|---|---|---|---|",
        *[model_line(model, scenes) for model, scenes in by_model.items()],
        "",
        "## Per scenario",
        "",
    ]
    for model, scenes in by_model.items():
        lines += [*scene_lines(model, scenes), ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    return output


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage_dirs", type=Path, nargs="+", help="one stage folder per pass")
    parser.add_argument("--output", type=Path, default=Path("results/combined_report.md"))
    args = parser.parse_args()
    print(build_combined_report(args.stage_dirs, args.output))


if __name__ == "__main__":
    main()
