"""Tests for the speed-dependent fallback and safety thresholds."""

import pytest

from src.adapters.base import EgoState, MetaAction, Observation
from src.agent.safety import choose_fallback, is_obstacle_within_stopping_range, stopping_distance_m
from src.utils.config import apply_overrides


def observation_with(speed_mps: float, gap_m: float | None) -> Observation:
    return Observation(
        camera_frames={},
        ego=EgoState(0.0, 0.0, 0.0, speed_mps),
        route=[],
        goal=(100.0, 0.0),
        timestamp_s=0.0,
        step_index=0,
        nearest_obstacle_ahead_m=gap_m,
    )


def test_stopping_distance_grows_with_speed_squared(agent_config):
    # stop_decel_mps2 is 4: v^2 / 8.
    assert stopping_distance_m(0.0, agent_config) == 0.0
    assert stopping_distance_m(8.0, agent_config) == pytest.approx(8.0)
    assert stopping_distance_m(16.0, agent_config) == pytest.approx(32.0)


def test_same_gap_is_safe_when_slow_and_unsafe_when_fast(agent_config):
    margin = 5.0
    assert not is_obstacle_within_stopping_range(observation_with(2.0, 10.0), margin, agent_config)
    assert is_obstacle_within_stopping_range(observation_with(12.0, 10.0), margin, agent_config)


def test_no_obstacle_means_slow_down_fallback(agent_config):
    assert choose_fallback(observation_with(15.0, None), agent_config).meta_action == MetaAction.SLOW_DOWN


def test_fallback_stops_for_obstacle_in_stopping_range(agent_config):
    # 15 m/s needs ~28 m to stop; with a 10 m margin a 30 m gap calls for STOP.
    assert choose_fallback(observation_with(15.0, 30.0), agent_config).meta_action == MetaAction.STOP


def test_experiment_overrides_merge_one_level_deep(agent_config):
    merged = apply_overrides(agent_config, {"safety": {"enabled": True}})
    assert merged["safety"] == {"enabled": True, "stop_margin_m": agent_config["safety"]["stop_margin_m"]}
    assert agent_config["safety"]["enabled"] is False  # the original is not modified
