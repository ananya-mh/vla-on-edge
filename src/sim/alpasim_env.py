"""STUB: SimEnv backed by NVIDIA AlpaSim + a Cosmos world model. Not implemented yet.

The AlpaSim API is deliberately not guessed here. Fill each method in from the AlpaSim
docs or code. Until then, use `mock_env.py`.
"""

from src.adapters.base import Action, Observation
from src.sim.base import SimEnv, StepEvents


class AlpaSimEnv(SimEnv):
    """AlpaSim implementation of the SimEnv interface (stub)."""

    def __init__(self, agent_config: dict) -> None:
        # TODO: connect to the AlpaSim runtime and the world-model service.
        # TODO_VERIFY: how AlpaSim is launched and addressed (process, gRPC, HTTP, ...).
        self.agent_config = agent_config
        raise NotImplementedError("AlpaSimEnv is a stub; use --sim mock")

    def reset(self, scenario: dict, seed: int) -> None:
        # TODO: load the AlpaSim scenario that `scenario` names and seed every source of
        # randomness (traffic, world model sampling) with `seed`. Run in step mode so the
        # simulator waits for each action.
        raise NotImplementedError

    def observe(self) -> Observation:
        # TODO: fetch the world-model camera frames for all views as H x W x 3 uint8
        # arrays, the ego state (world-frame x, y, heading, speed), the route and goal,
        # the sim timestamp, and the step index. Fill nearest_obstacle_ahead_m from
        # ground-truth objects if AlpaSim exposes them (used only by safety/fallback).
        raise NotImplementedError

    def step(self, action: Action) -> StepEvents:
        # TODO: convert the action to the trajectory format AlpaSim expects. Meta-actions
        # go through src.controller.trajectory.action_to_waypoints (ego frame, meters,
        # x forward, y left, t seconds from now). Advance one step and report collision,
        # off-road, and route progress for this step.
        raise NotImplementedError

    def done(self) -> bool:
        # TODO: return AlpaSim's episode-termination signal.
        raise NotImplementedError

    def metrics(self) -> dict:
        # TODO: return AlpaSim's episode summary using the same keys as MockEnv.metrics()
        # where possible (steps, collided, reached_goal, off_road_steps, progress_fraction).
        raise NotImplementedError
