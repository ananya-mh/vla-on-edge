"""Tests for format and physical-limit validation of parsed actions."""

from src.adapters.base import Action, MetaAction, Waypoint
from src.agent.validator import validate_action
from src.controller.trajectory import meta_action_to_waypoints


def constant_speed_waypoints(speed_mps: float, count: int = 6) -> list[Waypoint]:
    return [Waypoint(x=speed_mps * 0.5 * (i + 1), y=0.0, t=0.5 * (i + 1)) for i in range(count)]


def test_meta_action_is_always_valid(agent_config):
    assert validate_action(Action(meta_action=MetaAction.ACCELERATE), 15.0, agent_config) is None


def test_constant_speed_waypoints_are_valid(agent_config):
    action = Action(waypoints=constant_speed_waypoints(8.0))
    assert validate_action(action, 8.0, agent_config) is None


def test_controller_trajectories_are_valid(agent_config):
    # Anything the controller produces must pass the validator, or the dummy adapter
    # in waypoints mode would trigger fallbacks.
    for meta_action in MetaAction:
        for speed in (0.0, 8.0, 15.0):
            waypoints = meta_action_to_waypoints(meta_action, speed, agent_config)
            assert validate_action(Action(waypoints=waypoints), speed, agent_config) is None, (meta_action, speed)


def test_too_few_waypoints_are_rejected(agent_config):
    # A truncated output can still be valid JSON with a short list.
    action = Action(waypoints=constant_speed_waypoints(8.0, count=5))
    assert "expected 6 waypoints" in validate_action(action, 8.0, agent_config)


def test_wrong_waypoint_times_are_rejected(agent_config):
    waypoints = [Waypoint(x=4.0 * (i + 1), y=0.0, t=1.0 * (i + 1)) for i in range(6)]
    assert "time" in validate_action(Action(waypoints=waypoints), 8.0, agent_config)


def test_excessive_speed_is_rejected(agent_config):
    action = Action(waypoints=constant_speed_waypoints(30.0))
    assert "speed" in validate_action(action, 30.0, agent_config)


def test_excessive_acceleration_is_rejected(agent_config):
    # From standstill to 8 m/s average over the first half second.
    action = Action(waypoints=constant_speed_waypoints(8.0))
    assert "acceleration" in validate_action(action, 0.0, agent_config)


def test_hard_braking_is_rejected(agent_config):
    # 12 m/s, then an immediate stop: all points at the first position.
    waypoints = [Waypoint(x=6.0, y=0.0, t=0.5)] + [Waypoint(x=6.0, y=0.0, t=0.5 * (i + 1)) for i in range(1, 6)]
    assert "deceleration" in validate_action(Action(waypoints=waypoints), 12.0, agent_config)


def test_backwards_motion_is_rejected(agent_config):
    waypoints = constant_speed_waypoints(4.0)
    waypoints[3] = Waypoint(x=waypoints[2].x - 1.0, y=0.0, t=waypoints[3].t)
    assert "backwards" in validate_action(Action(waypoints=waypoints), 4.0, agent_config)


def test_non_finite_values_are_rejected(agent_config):
    waypoints = constant_speed_waypoints(4.0)
    waypoints[0] = Waypoint(x=float("nan"), y=0.0, t=0.5)
    assert "non-finite" in validate_action(Action(waypoints=waypoints), 4.0, agent_config)
