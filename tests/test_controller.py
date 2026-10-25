"""Tests for the meta-action → waypoint controller."""

import pytest

from src.adapters.base import Action, MetaAction, Waypoint
from src.controller.trajectory import action_to_waypoints, meta_action_to_waypoints


def test_keep_lane_holds_speed_and_goes_straight(agent_config):
    waypoints = meta_action_to_waypoints(MetaAction.KEEP_LANE, 10.0, agent_config)
    assert [p.t for p in waypoints] == pytest.approx([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    assert [p.x for p in waypoints] == pytest.approx([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    assert all(p.y == pytest.approx(0.0) for p in waypoints)


def test_accelerate_is_capped_at_max_speed(agent_config):
    max_speed = agent_config["limits"]["max_speed_mps"]
    waypoints = meta_action_to_waypoints(MetaAction.ACCELERATE, max_speed, agent_config)
    assert waypoints[-1].x == pytest.approx(max_speed * 3.0)


def test_stop_comes_to_rest(agent_config):
    # 8 m/s at 4 m/s^2 stops after 2 s and 8 m; it must not reverse afterwards.
    waypoints = meta_action_to_waypoints(MetaAction.STOP, 8.0, agent_config)
    assert waypoints[-1].x == pytest.approx(8.0, abs=0.5)
    assert waypoints[-1].x == pytest.approx(waypoints[-2].x)


def test_steering_directions(agent_config):
    left = meta_action_to_waypoints(MetaAction.STEER_LEFT, 8.0, agent_config)
    right = meta_action_to_waypoints(MetaAction.STEER_RIGHT, 8.0, agent_config)
    assert left[-1].y > 0.5
    assert right[-1].y == pytest.approx(-left[-1].y)


def test_stationary_vehicle_stays_put(agent_config):
    waypoints = meta_action_to_waypoints(MetaAction.SLOW_DOWN, 0.0, agent_config)
    assert all(p.x == 0.0 and p.y == 0.0 for p in waypoints)


def test_waypoint_actions_pass_through(agent_config):
    waypoints = [Waypoint(1.0, 0.0, 0.5)]
    assert action_to_waypoints(Action(waypoints=waypoints), 5.0, agent_config) is waypoints
