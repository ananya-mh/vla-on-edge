"""STUB: SimEnv backed by the real simulator. Not implemented yet.

The real simulator has not been chosen (the founder called it a "Cosmos simulation";
the exact setup is TBD). Its API is deliberately not guessed here. Fill each method in
once a simulator is chosen. Until then, use `mock_env.py`.
"""

from src.adapters.base import Action, Observation
from src.sim.base import SimEnv, StepEvents


class RealSimEnv(SimEnv):
    """Real-simulator implementation of the SimEnv interface (stub)."""

    def __init__(self, agent_config: dict) -> None:
        # TODO: connect to the chosen simulator (and its world model/renderer, if any).
        # TODO_VERIFY: how the simulator is launched and addressed (process, gRPC, HTTP, ...).
        self.agent_config = agent_config
        raise NotImplementedError("the real simulator is not chosen yet; use --sim mock")

    def reset(self, scenario: dict, seed: int) -> None:
        # TODO: load the simulator scenario that `scenario` names and seed every source
        # of randomness (traffic, rendering/world-model sampling) with `seed`. Run in
        # step mode so the simulator waits for each action.
        raise NotImplementedError

    def observe(self) -> Observation:
        # TODO: fetch the camera frames for all views as H x W x 3 uint8 arrays, the ego
        # state (world-frame x, y, heading, speed), the route and goal, the sim timestamp,
        # and the step index. Fill nearest_obstacle_ahead_m from ground-truth objects if
        # the simulator exposes them (used only by safety/fallback).
        raise NotImplementedError

    def step(self, action: Action) -> StepEvents:
        # TODO: convert the action to the trajectory format the simulator expects.
        # Meta-actions go through src.controller.trajectory.action_to_waypoints (ego
        # frame, meters, x forward, y left, t seconds from now). Advance one step and
        # report collision, off-road, and route progress for this step.
        raise NotImplementedError

    def done(self) -> bool:
        # TODO: return the simulator's episode-termination signal.
        raise NotImplementedError

    def metrics(self) -> dict:
        # TODO: return the simulator's episode summary using the same keys as
        # MockEnv.metrics() where possible (steps, collided, reached_goal,
        # off_road_steps, progress_fraction).
        raise NotImplementedError
