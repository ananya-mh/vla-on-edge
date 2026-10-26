"""Tests for format and physical-limit validation of parsed actions."""

import pytest

from src.adapters.base import Action, MetaAction, Waypoint
from src.agent.validator import validate_action
from src.controller.trajectory import meta_action_to_waypoints
from src.utils.config import check_trajectory_spec


def constant_speed_waypoints(speed_mps: float, count: int = 6) -> list[Waypoint]:
    return [Waypoint(x=speed_mps * 0.5 * (i + 1), y=0.0, t=0.5 * (i + 1)) for i in range(count)]


def test_meta_action_is_always_valid(agent_config, trajectory_spec):
    assert validate_action(Action(meta_action=MetaAction.ACCELERATE), 15.0, trajectory_spec, agent_config["limits"]) is None


def test_constant_speed_waypoints_are_valid(agent_config, trajectory_spec):
    action = Action(waypoints=constant_speed_waypoints(8.0))
    assert validate_action(action, 8.0, trajectory_spec, agent_config["limits"]) is None


def test_controller_trajectories_are_valid(agent_config, trajectory_spec):
    # Anything the controller produces must pass the validator, or the dummy adapter
    # in waypoints mode would trigger fallbacks.
    for meta_action in MetaAction:
        for speed in (0.0, 8.0, 15.0):
            waypoints = meta_action_to_waypoints(meta_action, speed, agent_config)
            assert validate_action(Action(waypoints=waypoints), speed, trajectory_spec, agent_config["limits"]) is None, (meta_action, speed)


def test_too_few_waypoints_are_rejected(agent_config, trajectory_spec):
    # A truncated output can still be valid JSON with a short list.
    action = Action(waypoints=constant_speed_waypoints(8.0, count=5))
    assert "expected 6 waypoints" in validate_action(action, 8.0, trajectory_spec, agent_config["limits"])


def test_wrong_waypoint_times_are_rejected(agent_config, trajectory_spec):
    waypoints = [Waypoint(x=4.0 * (i + 1), y=0.0, t=1.0 * (i + 1)) for i in range(6)]
    assert "time" in validate_action(Action(waypoints=waypoints), 8.0, trajectory_spec, agent_config["limits"])


def test_excessive_speed_is_rejected(agent_config, trajectory_spec):
    action = Action(waypoints=constant_speed_waypoints(30.0))
    assert "speed" in validate_action(action, 30.0, trajectory_spec, agent_config["limits"])


def test_excessive_acceleration_is_rejected(agent_config, trajectory_spec):
    # From standstill to 8 m/s average over the first half second.
    action = Action(waypoints=constant_speed_waypoints(8.0))
    assert "acceleration" in validate_action(action, 0.0, trajectory_spec, agent_config["limits"])


def test_hard_braking_is_rejected(agent_config, trajectory_spec):
    # 12 m/s, then an immediate stop: all points at the first position.
    waypoints = [Waypoint(x=6.0, y=0.0, t=0.5)] + [Waypoint(x=6.0, y=0.0, t=0.5 * (i + 1)) for i in range(1, 6)]
    assert "deceleration" in validate_action(Action(waypoints=waypoints), 12.0, trajectory_spec, agent_config["limits"])


def test_backwards_motion_is_rejected(agent_config, trajectory_spec):
    waypoints = constant_speed_waypoints(4.0)
    waypoints[3] = Waypoint(x=waypoints[2].x - 1.0, y=0.0, t=waypoints[3].t)
    assert "backwards" in validate_action(Action(waypoints=waypoints), 4.0, trajectory_spec, agent_config["limits"])


def test_non_finite_values_are_rejected(agent_config, trajectory_spec):
    waypoints = constant_speed_waypoints(4.0)
    waypoints[0] = Waypoint(x=float("nan"), y=0.0, t=0.5)
    assert "non-finite" in validate_action(Action(waypoints=waypoints), 4.0, trajectory_spec, agent_config["limits"])


def test_native_trajectory_shape_is_validated_as_is(agent_config):
    # A model with a different native shape (e.g. 10 Hz) is checked against its own
    # spec, without resampling.
    ten_hz_spec = {"waypoint_count": 30, "spacing_s": 0.1, "horizon_s": 3.0}
    waypoints = [Waypoint(x=8.0 * 0.1 * (i + 1), y=0.0, t=0.1 * (i + 1)) for i in range(30)]
    assert validate_action(Action(waypoints=waypoints), 8.0, ten_hz_spec, agent_config["limits"]) is None
    six_point_spec = {"waypoint_count": 6, "spacing_s": 0.5, "horizon_s": 3.0}
    assert "expected 6 waypoints" in validate_action(Action(waypoints=waypoints), 8.0, six_point_spec, agent_config["limits"])


def test_inconsistent_trajectory_spec_is_rejected():
    with pytest.raises(ValueError, match="horizon_s"):
        check_trajectory_spec("bad", {"waypoint_count": 64, "spacing_s": 0.1, "horizon_s": 6.0})
