"""Offline tests of the results pipeline on fake AlpaSim outputs and fake plugin logs."""

import json

import polars as pl
import pytest

from vla_on_edge.eval.check_run import check_alpasim_outputs, check_plugin_logs
from vla_on_edge.eval.fake_results import write_fake_run
from vla_on_edge.eval.fake_results import STORIES, START_US, rollout_rows
from vla_on_edge.eval.metrics import (
    driven_timesteps,
    first_counted_event_us,
    first_event_times_us,
    load_stage,
    model_summary,
    scoring_end,
)
from vla_on_edge.eval.report import build_report, event_lines, outcome


@pytest.fixture(scope="module")
def stage_dir(tmp_path_factory):
    return write_fake_run(tmp_path_factory.mktemp("stage3"))


def scene_dirs(stage_dir):
    return sorted((stage_dir / "alpamayo1_5").iterdir())


def test_fake_metrics_parquet_has_alpasims_long_schema(stage_dir):
    parquet = next(scene_dirs(stage_dir)[0].glob("rollouts/*/*/metrics.parquet"))
    frame = pl.read_parquet(parquet)
    assert frame.columns == ["name", "timestamps_us", "values", "valid", "time_aggregation",
                             "clipgt_id", "rollout_id", "run_uuid", "run_name"]
    assert set(frame["time_aggregation"].unique()) <= {"mean", "median", "max", "min", "last"}


def test_collision_scene_is_failed_and_truncated(stage_dir):
    # The second scene's story is a front collision at 8.3 s with 45% final progress.
    collision_dir = scene_dirs(stage_dir)[1]
    rollout = json.loads((collision_dir / "aggregate/results-summary.json").read_text())["rollouts"][0]
    assert rollout["failure_reason"] == "collision_at_fault"
    assert rollout["score"] == 0.0
    # Truncation at the collision keeps progress at its value then, not the final 0.45.
    assert rollout["metrics"]["progress_clipped_rel"] == pytest.approx(0.45 * 8.3 / 20, abs=0.01)
    parquet = next(collision_dir.glob("rollouts/*/*/metrics.parquet"))
    start = int(pl.read_parquet(parquet)["timestamps_us"].min())
    wide = driven_timesteps(parquet)
    scored_end_us, _ = scoring_end(wide, 4.0)
    assert (scored_end_us - start) / 1e6 == pytest.approx(8.3, abs=0.05)
    assert (first_counted_event_us(first_event_times_us(wide), scored_end_us) - start) / 1e6 == pytest.approx(8.3, abs=0.05)


def test_headline_numbers(stage_dir):
    summary = model_summary(load_stage(stage_dir)["alpamayo1_5"])
    assert summary["n_scenes"] == 8
    assert summary["collision_rate"] == pytest.approx(2 / 8)
    assert summary["offroad_rate"] == pytest.approx(1 / 8)
    assert summary["latency"]["n"] == summary["n_predictions"] > 1000
    assert summary["latency"]["p99"] is not None
    assert summary["gpu_peak_allocated_gib"] == pytest.approx(22.3, abs=0.1)
    assert set(summary["flags_by_rule"]) == {"deceleration", "speed"}


def test_p99_is_withheld_for_small_samples(stage_dir):
    scenes = load_stage(stage_dir)["alpamayo1_5"][:1]
    summary = model_summary(scenes)
    assert summary["latency"]["n"] < 1000
    assert summary["latency"]["p99"] is None


def test_report_has_every_section_and_chart(stage_dir, tmp_path):
    report = build_report(stage_dir, tmp_path / "report.md").read_text()
    for heading in ("## Headline", "## Per scenario", "## Charts", "## Worst cases"):
        assert heading in report
    assert "collision_at_fault" in report and "offroad" in report
    assert "8.3 s into the rollout" in report
    assert "Model reasoning before the first at-fault collision or off-road, most recent first" in report
    for chart in ("score_by_scenario.png", "latency_distribution.png", "validator_flags.png"):
        assert (tmp_path / "charts" / chart).stat().st_size > 0


def test_check_run_accepts_a_good_run(stage_dir):
    log_dir = scene_dirs(stage_dir)[0]
    problems, time_range = check_alpasim_outputs(log_dir)
    assert problems == []
    assert check_plugin_logs(log_dir, time_range, "alpamayo1_5") == []


def test_check_run_reports_missing_plugin_logs(stage_dir, tmp_path):
    problems = check_plugin_logs(tmp_path, None, "alpamayo1_5")
    assert problems and "VLA_ON_EDGE_LOG_DIR" in problems[0]


def test_check_run_detects_a_clock_mismatch(stage_dir):
    log_dir = scene_dirs(stage_dir)[0]
    _, (first, last) = check_alpasim_outputs(log_dir)
    shifted = (first + 10**12, last + 10**12)  # a different clock origin
    problems = check_plugin_logs(log_dir, shifted, "alpamayo1_5")
    assert any("clocks may differ" in problem for problem in problems)


def test_check_run_detects_logs_from_another_model(stage_dir):
    log_dir = scene_dirs(stage_dir)[0]  # written as alpamayo1_5
    _, time_range = check_alpasim_outputs(log_dir)
    problems = check_plugin_logs(log_dir, time_range, "alpamayo2_super")
    assert problems == ["some records are not from alpamayo2_super (wrong wrapper?)"]


def drifted_then_hit_parquet(tmp_path):
    """Like stage 1 on VM 2 (alpamayo2_super): >= 4 m from the recorded drive at 10.4 s,
    then hit from behind at 12.8 s and a front collision at 13.8 s, none of them scored."""
    rows = rollout_rows(STORIES[0], "scene", "rollout", "run")
    for row in rows:
        t_s = (row["timestamps_us"] - START_US) / 1e6
        if row["name"] == "dist_to_gt_trajectory" and t_s >= 10.4:
            row["values"] = 4.2
        if row["name"] in ("collision_rear", "collision_any") and t_s >= 12.8:
            row["values"] = 1.0
        if row["name"] == "collision_front" and t_s >= 13.8:
            row["values"] = 1.0
    path = tmp_path / "metrics.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def test_events_after_alpasim_stops_scoring_are_not_counted(tmp_path):
    wide = driven_timesteps(drifted_then_hit_parquet(tmp_path))
    scored_end_us, reason = scoring_end(wide, 4.0)
    assert (scored_end_us - START_US) / 1e6 == pytest.approx(10.4, abs=0.05)
    assert reason == "drift"
    events = first_event_times_us(wide)
    assert set(events) == {"collision_rear", "collision_front"}
    assert first_counted_event_us(events, scored_end_us) is None


def test_worst_case_text_matches_alpasims_verdict(tmp_path):
    wide = driven_timesteps(drifted_then_hit_parquet(tmp_path))
    scored_end_us, reason = scoring_end(wide, 4.0)
    scene = {"start_time_us": START_US, "scored_end_us": scored_end_us, "scored_end_reason": reason,
             "max_dist_to_gt_m": 4.0, "event_times_us": first_event_times_us(wide)}
    passed = {"failure_reason": None, "metrics": {"progress_clipped_rel": 1.0}}
    assert outcome(passed) == "no at-fault collision or off-road (progress 1.00)"
    assert event_lines(scene) == [
        "AlpaSim stopped scoring at 10.4 s into the rollout: the ego was 4 m or more from the recorded drive, "
        "so what follows is not scored.",
        "- not-at-fault collision (rear) at 12.8 s (after scoring stopped, so not in the score)",
        "- at-fault collision (front) at 13.8 s (after scoring stopped, so not in the score)",
    ]
    assert outcome({"failure_reason": "collision_at_fault", "metrics": {}}) == "failed: at-fault collision"


def test_a_rear_collision_ends_scoring_but_is_not_at_fault(tmp_path):
    rows = rollout_rows(STORIES[0], "scene", "rollout", "run")
    for row in rows:
        if row["name"] in ("collision_rear", "collision_any") and row["timestamps_us"] >= START_US + 9_000_000:
            row["values"] = 1.0
    pl.DataFrame(rows).write_parquet(tmp_path / "metrics.parquet")
    wide = driven_timesteps(tmp_path / "metrics.parquet")
    scored_end_us, reason = scoring_end(wide, 4.0)
    assert reason == "collision" and scored_end_us == START_US + 9_000_000
    assert first_counted_event_us(first_event_times_us(wide), scored_end_us) is None


def test_folders_without_scene_runs_are_not_models(stage_dir, tmp_path):
    report = build_report(stage_dir, tmp_path / "report.md")  # writes charts/ next to the report
    build_report(stage_dir, stage_dir / "report.md")  # and here into the stage folder itself, as run_stages.sh does
    assert list(load_stage(stage_dir)) == ["alpamayo1_5"]
    assert "| charts |" not in (stage_dir / "report.md").read_text() and report.exists()


def test_scenes_cut_short_by_drift_are_flagged(tmp_path):
    from vla_on_edge.eval.report import drift_note

    summary = {"n_scenes": 8, "n_drift_cut": 1}
    note = drift_note({"alpamayo2_super": summary}, 4.0)
    assert "in 1 of 8 scenes AlpaSim stopped scoring early" in note[0] and "4 m" in note[0]
    assert drift_note({"alpamayo2_super": {**summary, "n_drift_cut": 0}}, 4.0) == []


def test_combined_report_shows_passes_and_spread(tmp_path):
    from vla_on_edge.eval.combine import build_combined_report, load_passes, spread

    first = write_fake_run(tmp_path / "vm_a" / "stage3", seed=0)
    second = write_fake_run(tmp_path / "vm_b" / "stage3", seed=1)
    # In pass 2, the clean first scene "scores" 0.0, as when a random rollout crashes.
    clean = sorted((second / "alpamayo1_5").iterdir())[0]
    summary_path = clean / "aggregate/results-summary.json"
    summary = json.loads(summary_path.read_text())
    summary["rollouts"][0]["score"] = 0.0
    summary_path.write_text(json.dumps(summary))

    passes, by_model = load_passes([first, second])
    assert [p["n_scenes"] for p in passes] == [8, 8]
    rows = by_model["alpamayo1_5"][clean.name]
    assert [r["pass_index"] for r in rows] == [0, 1]
    scores = spread([r["score"] for r in rows])  # 1.0 in pass 1, 0.0 in pass 2
    assert (scores["mean"], scores["sd"], scores["min"], scores["max"]) == pytest.approx((0.5, 0.7071, 0.0, 1.0), abs=1e-4)

    report = build_combined_report([first, second], tmp_path / "combined.md").read_text()
    assert "2 passes" in report and f"| `{clean.name}` | 2 | 0.50 ± 0.71 (0.00–1.00) |" in report
    assert "2 of 2" in report  # the fake collision scene collides in both passes
    assert spread([]) == {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
