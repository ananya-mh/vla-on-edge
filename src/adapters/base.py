"""Core data types shared by every model adapter, the agent loop, and the simulators.

`Observation` is what the simulator shows the agent, `Action` is what the agent sends
back, and `Decision` is what a model adapter returns for one step.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class MetaAction(str, Enum):
    """High-level driving commands a model can choose from."""

    KEEP_LANE = "KEEP_LANE"
    ACCELERATE = "ACCELERATE"
    SLOW_DOWN = "SLOW_DOWN"
    STOP = "STOP"
    STEER_LEFT = "STEER_LEFT"
    STEER_RIGHT = "STEER_RIGHT"


@dataclass(frozen=True)
class Waypoint:
    """A future ego position in the ego frame: meters (x forward, y left), t seconds from now."""

    x: float
    y: float
    t: float


@dataclass
class Action:
    """Either a meta-action or a list of waypoints, never both."""

    meta_action: MetaAction | None = None
    waypoints: list[Waypoint] | None = None

    def __post_init__(self) -> None:
        has_meta_action = self.meta_action is not None
        has_waypoints = self.waypoints is not None
        if has_meta_action == has_waypoints:
            raise ValueError("Action needs exactly one of meta_action or waypoints")

    def to_dict(self) -> dict:
        """Return a JSON-serializable form, used in step logs."""
        if self.meta_action is not None:
            return {"meta_action": self.meta_action.value}
        return {"waypoints": [[point.x, point.y, point.t] for point in self.waypoints]}

    def describe(self) -> str:
        """Return a short human-readable form, used in prompts for the decision history."""
        if self.meta_action is not None:
            return self.meta_action.value
        points = ", ".join(f"({p.x:.1f}, {p.y:.1f}, {p.t:.1f})" for p in self.waypoints)
        return f"waypoints [{points}]"


@dataclass
class EgoState:
    """Ego vehicle pose and speed in the world frame."""

    x: float
    y: float
    heading_rad: float
    speed_mps: float


@dataclass
class Observation:
    """Everything the simulator reports for one step."""

    camera_frames: dict[str, np.ndarray]  # view name -> H x W x 3 uint8 image
    ego: EgoState
    route: list[tuple[float, float]]  # route centerline points, world frame
    goal: tuple[float, float]  # world frame
    timestamp_s: float
    step_index: int
    # Privileged ground truth used only by the safety check and fallback choice.
    # It must never be put in the prompt: models have to perceive obstacles from images.
    nearest_obstacle_ahead_m: float | None = None


@dataclass
class Decision:
    """One model call: the raw text plus token counts and timing.

    The action itself is produced by the agent loop's parser, not by the adapter,
    so that every model's output is parsed by exactly the same code.
    """

    raw_output: str
    prompt_hash: str
    truncated: bool  # the model hit max_tokens before finishing
    total_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ttft_s: float | None = None  # time to first token (prefill)
    decode_s: float | None = None  # time spent generating output tokens


class ModelAdapter:
    """Base class for models driven by the agent loop. Subclasses implement `decide`."""

    def __init__(self, model_config: dict, agent_config: dict) -> None:
        self.model_config = model_config
        self.agent_config = agent_config

    def decide(self, observation: Observation, history: list[Action]) -> Decision:
        """Return the model's raw decision for this observation.

        `history` holds the most recent executed actions, oldest first. Adapters
        should raise TimeoutError if the model does not answer in time.
        """
        raise NotImplementedError
