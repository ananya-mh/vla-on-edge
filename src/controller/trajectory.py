"""Turns a meta-action into a waypoint trajectory, so simulators only need to follow waypoints.

Each meta-action is a constant acceleration and yaw rate held for the whole horizon.
We integrate a simple unicycle model in small time steps and sample it at the
configured waypoint times (e.g. 6 points, 0.5 s apart).
"""

import math

from src.adapters.base import Action, MetaAction, Waypoint


def waypoint_times(agent_config: dict) -> list[float]:
    """Return the expected waypoint times, e.g. [0.5, 1.0, ..., 3.0]."""
    count = agent_config["waypoints"]["count"]
    spacing = agent_config["waypoints"]["spacing_s"]
    return [spacing * (index + 1) for index in range(count)]


def acceleration_for(meta_action: MetaAction, controller_config: dict) -> float:
    """Return the longitudinal acceleration (m/s^2) a meta-action asks for."""
    if meta_action == MetaAction.ACCELERATE:
        return controller_config["accelerate_mps2"]
    if meta_action == MetaAction.SLOW_DOWN:
        return -controller_config["slow_down_decel_mps2"]
    if meta_action == MetaAction.STOP:
        return -controller_config["stop_decel_mps2"]
    return 0.0


def yaw_rate_for(meta_action: MetaAction, controller_config: dict) -> float:
    """Return the yaw rate (rad/s, positive = left) a meta-action asks for."""
    if meta_action == MetaAction.STEER_LEFT:
        return controller_config["steer_yaw_rate_rps"]
    if meta_action == MetaAction.STEER_RIGHT:
        return -controller_config["steer_yaw_rate_rps"]
    return 0.0


def meta_action_to_waypoints(
    meta_action: MetaAction, current_speed_mps: float, agent_config: dict
) -> list[Waypoint]:
    """Integrate the meta-action from the current speed and return ego-frame waypoints."""
    controller_config = agent_config["controller"]
    max_speed = agent_config["limits"]["max_speed_mps"]
    time_step = controller_config["integration_dt_s"]
    acceleration = acceleration_for(meta_action, controller_config)
    yaw_rate = yaw_rate_for(meta_action, controller_config)

    x, y, heading, speed, time = 0.0, 0.0, 0.0, current_speed_mps, 0.0
    waypoints = []
    for target_time in waypoint_times(agent_config):
        while time < target_time - 1e-9:
            step = min(time_step, target_time - time)
            speed = min(max(speed + acceleration * step, 0.0), max_speed)
            heading += yaw_rate * step
            x += speed * math.cos(heading) * step
            y += speed * math.sin(heading) * step
            time += step
        waypoints.append(Waypoint(x=x, y=y, t=target_time))
    return waypoints


def action_to_waypoints(action: Action, current_speed_mps: float, agent_config: dict) -> list[Waypoint]:
    """Return the action's waypoints, converting a meta-action if needed."""
    if action.waypoints is not None:
        return action.waypoints
    return meta_action_to_waypoints(action.meta_action, current_speed_mps, agent_config)
