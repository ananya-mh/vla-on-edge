"""The simulator interface. Both the mock simulator and AlpaSim implement it."""

from dataclasses import dataclass

from src.adapters.base import Action, Observation


@dataclass
class StepEvents:
    """What happened during one simulator step. Logged by the agent loop."""

    collided: bool
    off_road: bool
    progress_fraction: float  # 0.0 at the start of the route, 1.0 at the goal


class SimEnv:
    """A closed-loop driving simulator in step mode: it waits for each action."""

    def reset(self, scenario: dict, seed: int) -> None:
        """Start a new episode of `scenario` (a resolved entry from scenarios.yaml)."""
        raise NotImplementedError

    def observe(self) -> Observation:
        """Return the current observation."""
        raise NotImplementedError

    def step(self, action: Action) -> StepEvents:
        """Apply `action`, advance the simulation by one step, and report what happened."""
        raise NotImplementedError

    def done(self) -> bool:
        """Return True when the episode is over (goal reached, collision, or step limit)."""
        raise NotImplementedError

    def metrics(self) -> dict:
        """Return a summary of the episode so far, as reported by the simulator."""
        raise NotImplementedError
