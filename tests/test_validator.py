"""Tests for format and physical-limit validation of parsed actions."""

import pytest

from vla_on_edge.adapters.base import Action, MetaAction, Waypoint
from vla_on_edge.agent.validator import find_violations, validate_action
from vla_on_edge.controller.trajectory import meta_action_to_waypoints
from vla_on_edge.utils.config import check_trajectory_spec


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


def ten_hz_waypoints(speed_mps: float, count: int = 64) -> list[Waypoint]:
    return [Waypoint(x=speed_mps * 0.1 * (i + 1), y=0.0, t=round(0.1 * (i + 1), 1)) for i in range(count)]


TEN_HZ_SPEC = {"waypoint_count": 64, "spacing_s": 0.1, "horizon_s": 6.4}


def test_window_smooths_position_noise(agent_config):
    # 8 m/s with +-5 cm zig-zag noise: per-0.1 s differences swing about +-1 m/s
    # (+-20 m/s^2), but 0.5 s windows see the true, constant speed.
    noisy = []
    for index, point in enumerate(ten_hz_waypoints(8.0)):
        noise = 0.05 if index % 2 == 0 else -0.05
        noisy.append(Waypoint(x=point.x + noise, y=0.0, t=point.t))
    action = Action(waypoints=noisy)
    assert find_violations(action, 8.0, TEN_HZ_SPEC, agent_config["limits"]) == []

    per_step_limits = {**agent_config["limits"], "speed_window_s": 0.1}
    assert any("acceleration" in v for v in find_violations(action, 8.0, TEN_HZ_SPEC, per_step_limits))


def test_all_broken_rules_are_reported(agent_config, trajectory_spec):
    # 20 m/s from 8 m/s: too fast and too much acceleration, one message each.
    action = Action(waypoints=constant_speed_waypoints(20.0))
    violations = find_violations(action, 8.0, trajectory_spec, agent_config["limits"])
    assert len(violations) == 2
    assert "speed" in violations[0] and "acceleration" in violations[1]


def test_window_longer_than_horizon_is_clamped(agent_config):
    short_spec = {"waypoint_count": 3, "spacing_s": 0.1, "horizon_s": 0.3}
    action = Action(waypoints=ten_hz_waypoints(8.0, count=3))
    assert find_violations(action, 8.0, short_spec, agent_config["limits"]) == []


def test_rotation_is_ignored_by_the_validator(agent_config):
    # A nonsense rotation must not affect validation: only positions are checked.
    bad_rotation = ((9.0, 9.0, 9.0), (9.0, 9.0, 9.0), (9.0, 9.0, 9.0))
    waypoints = [Waypoint(p.x, p.y, p.t, z=0.0, rotation=bad_rotation) for p in ten_hz_waypoints(8.0)]
    assert find_violations(Action(waypoints=waypoints), 8.0, TEN_HZ_SPEC, agent_config["limits"]) == []
