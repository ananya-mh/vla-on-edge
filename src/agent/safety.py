"""Rule-based safety layer: the fallback action and the safety override.

Both use the simulator's privileged `nearest_obstacle_ahead_m`, which the model never
sees, and a speed-dependent threshold: the distance needed to stop from the current
speed plus a configured margin. They are logged separately: a fallback replaces an
invalid or missing model output, while a safety override replaces a valid action
that would be unsafe.
"""

from src.adapters.base import Action, MetaAction, Observation

SAFETY_OVERRIDE_OBSTACLE = "obstacle_within_stopping_distance"


def stopping_distance_m(speed_mps: float, agent_config: dict) -> float:
    """Return the distance needed to stop from `speed_mps` when braking with STOP."""
    deceleration = agent_config["controller"]["stop_decel_mps2"]
    return speed_mps**2 / (2 * deceleration)


def is_obstacle_within_stopping_range(observation: Observation, margin_m: float, agent_config: dict) -> bool:
    """Return True if an obstacle in the ego's path is within stopping distance + margin."""
    gap = observation.nearest_obstacle_ahead_m
    if gap is None:
        return False
    return gap <= stopping_distance_m(observation.ego.speed_mps, agent_config) + margin_m


def choose_fallback(observation: Observation, agent_config: dict) -> Action:
    """Return STOP if an obstacle is within stopping range, otherwise SLOW_DOWN."""
    margin = agent_config["fallback"]["stop_margin_m"]
    if is_obstacle_within_stopping_range(observation, margin, agent_config):
        return Action(meta_action=MetaAction.STOP)
    return Action(meta_action=MetaAction.SLOW_DOWN)


def apply_safety_check(action: Action, observation: Observation, agent_config: dict) -> tuple[Action, str | None]:
    """Return the action to execute and the override reason (None if not overridden)."""
    safety_config = agent_config["safety"]
    if not safety_config["enabled"]:
        return action, None
    if not is_obstacle_within_stopping_range(observation, safety_config["stop_margin_m"], agent_config):
        return action, None
    if action.meta_action == MetaAction.STOP:
        return action, None
    return Action(meta_action=MetaAction.STOP), SAFETY_OVERRIDE_OBSTACLE
