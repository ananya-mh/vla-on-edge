"""Tests for the mock simulator, plus an end-to-end smoke test of the whole pipeline."""

import math

import pytest

from src.adapters.base import Action, EgoState, MetaAction, Waypoint
from src.run_benchmark import main
from src.sim.mock_env import MockEnv, advance_along_waypoints
from src.utils import config as config_module
from src.utils.logging import read_jsonl

KEEP_LANE = Action(meta_action=MetaAction.KEEP_LANE)


def test_reset_is_deterministic_per_seed(agent_config, scenarios):
    env = MockEnv(agent_config)
    env.reset(scenarios["scattered_obstacles"], seed=3)
    first = list(env.obstacles)
    env.reset(scenarios["scattered_obstacles"], seed=3)
    assert env.obstacles == first
    env.reset(scenarios["scattered_obstacles"], seed=4)
    assert env.obstacles != first


def test_keep_lane_on_open_road_reaches_goal(agent_config, scenarios):
    env = MockEnv(agent_config)
    env.reset(scenarios["open_corridor"], seed=0)
    while not env.done():
        events = env.step(KEEP_LANE)
        assert not events.collided and not events.off_road
    assert env.metrics()["reached_goal"]
    assert env.metrics()["progress_fraction"] == 1.0


def test_keep_lane_into_blocked_lane_collides(agent_config, scenarios):
    env = MockEnv(agent_config)
    env.reset(scenarios["blocked_lane"], seed=0)
    assert env.observe().nearest_obstacle_ahead_m == pytest.approx(60.0 - 2.0 - 1.5)
    while not env.done():
        env.step(KEEP_LANE)
    assert env.metrics()["collided"]


def test_steering_eventually_leaves_the_road(agent_config, scenarios):
    env = MockEnv(agent_config)
    env.reset(scenarios["narrow_corridor"], seed=0)
    off_road = False
    for _ in range(10):
        off_road = off_road or env.step(Action(meta_action=MetaAction.STEER_LEFT)).off_road
    assert off_road
    assert env.ego.heading_rad > 0


def test_advance_uses_heading_for_world_position():
    # Facing +y in the world, moving 4 m forward in the ego frame moves +y in the world.
    ego = EgoState(x=0.0, y=0.0, heading_rad=math.pi / 2, speed_mps=8.0)
    waypoints = [Waypoint(4.0 * (i + 1), 0.0, 0.5 * (i + 1)) for i in range(6)]
    new_ego = advance_along_waypoints(ego, waypoints, 0.5)
    assert new_ego.x == pytest.approx(0.0, abs=1e-9)
    assert new_ego.y == pytest.approx(4.0)
    assert new_ego.speed_mps == pytest.approx(8.0)
    assert new_ego.heading_rad == pytest.approx(math.pi / 2)


def test_camera_frames_have_expected_shape(agent_config, scenarios):
    env = MockEnv(agent_config)
    env.reset(scenarios["blocked_lane"], seed=0)
    env.ego.x = 40.0  # the obstacle at x=60 is now inside the front view
    frames = env.observe().camera_frames
    assert set(frames) == {"front", "rear"}
    size = scenarios["blocked_lane"]["image_size_px"]
    assert frames["front"].shape == (size, size, 3)
    red_pixels = (frames["front"] == (200, 40, 40)).all(axis=2).sum()
    assert red_pixels > 0


@pytest.mark.parametrize("action_format", ["meta_action", "waypoints"])
def test_smoke_run_end_to_end(tmp_path, monkeypatch, action_format):
    """Run the dummy model through the full CLI and check the logs it writes."""
    original_load_yaml = config_module.load_yaml

    def load_yaml_with_format(path):
        data = original_load_yaml(path)
        if path.name == "agent.yaml":
            data["action_format"] = action_format
        return data

    monkeypatch.setattr(config_module, "load_yaml", load_yaml_with_format)
    run_dir = main(["--model", "dummy:bf16", "--experiment", "smoke_test", "--results-dir", str(tmp_path)])

    assert (run_dir / "metadata.json").exists()
    episodes = read_jsonl(run_dir / "episodes.jsonl")
    steps = read_jsonl(run_dir / "steps.jsonl")
    assert len(episodes) == 8
    assert len(steps) == sum(episode["steps"] for episode in episodes)
    # The dummy only produces valid actions, so nothing should fall back.
    assert all(step["fallback_reason"] is None for step in steps)
    assert all(step["prompt_hash"] for step in steps)
    assert all(record["log_format_version"] == 1 for record in steps + episodes)


def test_fake_alpamayo_runs_end_to_end_on_the_mock(tmp_path):
    """Plumbing only: a fixed 64-point native trajectory is validated (flag-only) and logged in full."""
    run_dir = main(["--model", "fake_alpamayo1_5", "--experiment", "smoke_test", "--results-dir", str(tmp_path)])
    steps = read_jsonl(run_dir / "steps.jsonl")
    assert all(step["fallback_reason"] is None for step in steps)
    assert all(step["validation_violations"] == [] for step in steps)
    first = steps[0]["executed_action"]["waypoints"]
    assert len(first) == 64
    assert first[-1]["t"] == pytest.approx(6.4)
    assert first[0]["rotation"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_alpasim_plugin_models_are_rejected_by_run_benchmark(tmp_path):
    with pytest.raises(ValueError, match="runs inside AlpaSim"):
        main(["--model", "alpamayo1_5", "--experiment", "smoke_test", "--results-dir", str(tmp_path)])
