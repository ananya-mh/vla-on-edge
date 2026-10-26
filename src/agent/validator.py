"""Checks an Action against the model's trajectory spec and simple physical limits.

A meta-action is always valid (the controller enforces limits itself). A waypoint
list is checked as the model produced it, never resampled: it must have exactly the
model's `waypoint_count` points at its `spacing_s`, and the speed and acceleration it
implies must stay within limits. Rejecting short lists matters because a truncated
model output can still be valid JSON. Only positions are checked, never rotations.

Speed and acceleration are measured over a window (`limits.speed_window_s`, e.g.
0.5 s) rather than between adjacent waypoints, because finite differences over
0.1 s steps amplify small position noise into large fake accelerations.

What happens to an invalid action is the caller's choice ("validation_mode" in
models.yaml): `enforce` replaces it with the fallback, `flag_only` only logs the
violations. This module never changes an action.
"""

import math

from src.adapters.base import Action, Waypoint
from src.controller.trajectory import waypoint_times

# Allowed backwards motion (m) between consecutive waypoints, to absorb rounding.
BACKWARD_TOLERANCE_M = 0.1


def find_violations(action: Action, current_speed_mps: float, trajectory_spec: dict, limits: dict) -> list[str]:
    """Return every rule the action breaks (empty if valid), at most one message per rule.

    `trajectory_spec` is the model's `trajectory` entry from models.yaml;
    `limits` is the `limits` section of agent.yaml.
    """
    if action.meta_action is not None:
        return []
    return waypoint_violations(action.waypoints, current_speed_mps, trajectory_spec, limits)


def validate_action(action: Action, current_speed_mps: float, trajectory_spec: dict, limits: dict) -> str | None:
    """Return None if the action is valid, otherwise the first violation."""
    violations = find_violations(action, current_speed_mps, trajectory_spec, limits)
    return violations[0] if violations else None


def waypoint_violations(
    waypoints: list[Waypoint], current_speed_mps: float, trajectory_spec: dict, limits: dict
) -> list[str]:
    """Check shape first; motion is only checked once the shape is right."""
    expected_times = waypoint_times(trajectory_spec["waypoint_count"], trajectory_spec["spacing_s"])
    if len(waypoints) != len(expected_times):
        return [f"expected {len(expected_times)} waypoints, got {len(waypoints)}"]

    tolerance = limits["waypoint_time_tolerance_s"]
    for point, expected_time in zip(waypoints, expected_times):
        if not all(math.isfinite(value) for value in (point.x, point.y, point.t)):
            return [f"non-finite waypoint {point}"]
        if abs(point.t - expected_time) > tolerance:
            return [f"waypoint time {point.t} does not match expected {expected_time}"]

    return motion_violations(waypoints, current_speed_mps, trajectory_spec["spacing_s"], limits)


def window_speeds(points: list[Waypoint], step: int) -> list[tuple[float, float]]:
    """Return (center time, average speed) over each window of `step` waypoint intervals."""
    speeds = []
    for start, end in zip(points, points[step:]):
        distance = math.hypot(end.x - start.x, end.y - start.y)
        speeds.append(((start.t + end.t) / 2, distance / (end.t - start.t)))
    return speeds


def window_accelerations(
    speeds: list[tuple[float, float]], current_speed_mps: float, step: int
) -> list[tuple[float, float]]:
    """Return (time, acceleration) between window speeds one full window apart.

    The current speed is instantaneous at t = 0, so it pairs with the first window,
    whose center is only half a window later.
    """
    pairs = [((0.0, current_speed_mps), speeds[0])] + list(zip(speeds, speeds[step:]))
    accelerations = []
    for (start_time, start_speed), (end_time, end_speed) in pairs:
        accelerations.append((end_time, (end_speed - start_speed) / (end_time - start_time)))
    return accelerations


def motion_violations(
    waypoints: list[Waypoint], current_speed_mps: float, spacing_s: float, limits: dict
) -> list[str]:
    """Check backwards motion, windowed speed, and windowed acceleration."""
    violations = []
    points = [Waypoint(0.0, 0.0, 0.0)] + list(waypoints)  # the path starts at the ego now

    for index, (previous, point) in enumerate(zip(points, points[1:])):
        if point.x < previous.x - BACKWARD_TOLERANCE_M:
            violations.append(f"waypoint {index} moves backwards")
            break

    # Number of waypoint intervals per window: at least one (spacing wider than the
    # window), at most the whole path (window longer than the horizon).
    step = min(max(1, round(limits["speed_window_s"] / spacing_s)), len(waypoints))
    speeds = window_speeds(points, step)
    for center_time, speed in speeds:
        if speed > limits["max_speed_mps"]:
            violations.append(
                f"speed {speed:.1f} m/s around t={center_time:.2f} s exceeds {limits['max_speed_mps']} m/s"
            )
            break

    accelerations = window_accelerations(speeds, current_speed_mps, step)
    for center_time, acceleration in accelerations:
        if acceleration > limits["max_accel_mps2"]:
            violations.append(
                f"acceleration {acceleration:.1f} m/s^2 around t={center_time:.2f} s "
                f"exceeds {limits['max_accel_mps2']}"
            )
            break
    for center_time, acceleration in accelerations:
        if acceleration < -limits["max_decel_mps2"]:
            violations.append(
                f"deceleration {-acceleration:.1f} m/s^2 around t={center_time:.2f} s "
                f"exceeds {limits['max_decel_mps2']}"
            )
            break
    return violations
