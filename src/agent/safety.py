"""Rule-based safety layer: the fallback action and the safety override.

Both use the simulator's privileged `nearest_obstacle_ahead_m`, which the model never
sees. They are logged separately: a fallback replaces an invalid or missing model
output, while a safety override replaces a valid action that would be unsafe.
"""

from src.adapters.base import Action, MetaAction, Observation

SAFETY_OVERRIDE_OBSTACLE = "obstacle_within_stop_distance"


def is_obstacle_within(observation: Observation, distance_m: float) -> bool:
    """Return True if an obstacle in the ego's path is at most `distance_m` ahead."""
    gap = observation.nearest_obstacle_ahead_m
    return gap is not None and gap <= distance_m


def choose_fallback(observation: Observation, agent_config: dict) -> Action:
    """Return STOP if an obstacle is close ahead, otherwise SLOW_DOWN."""
    if is_obstacle_within(observation, agent_config["fallback"]["stop_distance_m"]):
        return Action(meta_action=MetaAction.STOP)
    return Action(meta_action=MetaAction.SLOW_DOWN)


def apply_safety_check(action: Action, observation: Observation, agent_config: dict) -> tuple[Action, str | None]:
    """Return the action to execute and the override reason (None if not overridden)."""
    safety_config = agent_config["safety"]
    if not safety_config["enabled"]:
        return action, None
    if not is_obstacle_within(observation, safety_config["stop_distance_m"]):
        return action, None
    if action.meta_action == MetaAction.STOP:
        return action, None
    return Action(meta_action=MetaAction.STOP), SAFETY_OVERRIDE_OBSTACLE
