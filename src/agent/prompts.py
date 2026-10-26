"""Builds the prompt every model sees, from the templates in agent.yaml.

The agent loop calls `build_prompt` once per step and hands the same `Prompt` to
whichever adapter is running, so every model gets the same text and images for the
same observation. The prompt hash lets us check that in the logs.
"""

import hashlib
import math

from src.adapters.base import Action, Observation, Prompt


def describe_history(history: list[Action]) -> str:
    """Format past actions as a single line, oldest first."""
    if not history:
        return "none"
    return "; ".join(action.describe() for action in history)


def goal_bearing_degrees(observation: Observation) -> float:
    """Return the goal direction relative to the ego heading, in degrees (positive = left)."""
    goal_x, goal_y = observation.goal
    absolute_bearing = math.atan2(goal_y - observation.ego.y, goal_x - observation.ego.x)
    relative = absolute_bearing - observation.ego.heading_rad
    # Wrap to [-180, 180) so a goal slightly to the right reads as negative, not ~360.
    return (math.degrees(relative) + 180.0) % 360.0 - 180.0


def output_instructions(agent_config: dict, trajectory_spec: dict) -> str:
    """Return the instructions for the configured action format."""
    action_format = agent_config["action_format"]
    template = agent_config["prompt"]["output_instructions"][action_format]
    return template.format(count=trajectory_spec["waypoint_count"], spacing_s=trajectory_spec["spacing_s"])


def build_prompt(
    observation: Observation, history: list[Action], agent_config: dict, trajectory_spec: dict
) -> Prompt:
    """Fill the prompt templates for this observation and recent history."""
    goal_x, goal_y = observation.goal
    view_names = sorted(observation.camera_frames)
    user_text = agent_config["prompt"]["user_template"].format(
        view_names=", ".join(view_names),
        speed_mps=observation.ego.speed_mps,
        distance_to_goal_m=math.hypot(goal_x - observation.ego.x, goal_y - observation.ego.y),
        goal_bearing_deg=goal_bearing_degrees(observation),
        history=describe_history(history),
        output_instructions=output_instructions(agent_config, trajectory_spec),
    )
    return Prompt(
        system_text=agent_config["prompt"]["system"].strip(),
        user_text=user_text.strip(),
        images={name: observation.camera_frames[name] for name in view_names},
    )


def prompt_hash(prompt: Prompt) -> str:
    """Return a SHA-256 hex digest over the prompt text and image pixels."""
    digest = hashlib.sha256()
    digest.update(prompt.system_text.encode("utf-8"))
    digest.update(prompt.user_text.encode("utf-8"))
    for name, image in prompt.images.items():
        digest.update(name.encode("utf-8"))
        digest.update(image.tobytes())
    return digest.hexdigest()
