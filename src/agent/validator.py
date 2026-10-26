"""Checks an Action against the model's trajectory spec and simple physical limits.

A meta-action is always valid (the controller enforces limits itself). A waypoint
list is checked as the model produced it, never resampled: it must have exactly the
model's `waypoint_count` points at its `spacing_s`, and the speed and acceleration it
implies must stay within limits. Rejecting short lists matters because a truncated
model output can still be valid JSON.
"""

import math

from src.adapters.base import Action, Waypoint
from src.controller.trajectory import waypoint_times

# Allowed backwards motion (m) between consecutive waypoints, to absorb rounding.
BACKWARD_TOLERANCE_M = 0.1


def validate_action(action: Action, current_speed_mps: float, trajectory_spec: dict, limits: dict) -> str | None:
    """Return None if the action is valid, otherwise a short reason it is not.

    `trajectory_spec` is the model's `trajectory` entry from models.yaml;
    `limits` is the `limits` section of agent.yaml.
    """
    if action.meta_action is not None:
        return None
    return validate_waypoints(action.waypoints, current_speed_mps, trajectory_spec, limits)


def validate_waypoints(
    waypoints: list[Waypoint], current_speed_mps: float, trajectory_spec: dict, limits: dict
) -> str | None:
    """Return None if the waypoints are complete and physically plausible, else a reason."""
    expected_times = waypoint_times(trajectory_spec["waypoint_count"], trajectory_spec["spacing_s"])
    if len(waypoints) != len(expected_times):
        return f"expected {len(expected_times)} waypoints, got {len(waypoints)}"

    tolerance = limits["waypoint_time_tolerance_s"]
    for point, expected_time in zip(waypoints, expected_times):
        if not all(math.isfinite(value) for value in (point.x, point.y, point.t)):
            return f"non-finite waypoint {point}"
        if abs(point.t - expected_time) > tolerance:
            return f"waypoint time {point.t} does not match expected {expected_time}"

    return check_motion_limits(waypoints, current_speed_mps, limits)


def check_motion_limits(waypoints: list[Waypoint], current_speed_mps: float, limits: dict) -> str | None:
    """Check the speed and acceleration implied by consecutive waypoints."""
    previous = Waypoint(0.0, 0.0, 0.0)
    previous_speed = current_speed_mps
    for index, point in enumerate(waypoints):
        segment_time = point.t - previous.t
        if point.x < previous.x - BACKWARD_TOLERANCE_M:
            return f"waypoint {index} moves backwards"

        speed = math.hypot(point.x - previous.x, point.y - previous.y) / segment_time
        if speed > limits["max_speed_mps"]:
            return f"implied speed {speed:.1f} m/s exceeds {limits['max_speed_mps']} m/s"

        # Segment speeds are averages over the segment, so they sit at segment midpoints.
        # The current speed is instantaneous at t = 0, only half a segment before the
        # first midpoint; later midpoints are a full segment apart.
        time_between_speeds = segment_time / 2 if index == 0 else segment_time
        acceleration = (speed - previous_speed) / time_between_speeds
        if acceleration > limits["max_accel_mps2"]:
            return f"implied acceleration {acceleration:.1f} m/s^2 exceeds {limits['max_accel_mps2']}"
        if acceleration < -limits["max_decel_mps2"]:
            return f"implied deceleration {-acceleration:.1f} m/s^2 exceeds {limits['max_decel_mps2']}"

        previous, previous_speed = point, speed
    return None
