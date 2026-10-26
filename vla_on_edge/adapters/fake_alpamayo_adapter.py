"""A stand-in for Alpamayo on the mock simulator, for testing plumbing only.

It returns the same fixed trajectory every step, shaped like Alpamayo's native
output (64 waypoints, 0.1 s apart, full pose with identity rotation). It says
nothing about Alpamayo's driving quality: the mock's top-down images are not
meaningful Alpamayo input, so it ignores the prompt entirely.
"""

import time

from vla_on_edge.adapters.base import Decision, ModelAdapter, Prompt, Waypoint

IDENTITY_ROTATION = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def fixed_trajectory(speed_mps: float, trajectory_spec: dict) -> list[Waypoint]:
    """Return a straight trajectory at constant speed with the spec's count and spacing."""
    waypoints = []
    for index in range(trajectory_spec["waypoint_count"]):
        time_s = round(trajectory_spec["spacing_s"] * (index + 1), 6)
        waypoints.append(Waypoint(x=speed_mps * time_s, y=0.0, t=time_s, z=0.0, rotation=IDENTITY_ROTATION))
    return waypoints


class FakeAlpamayoAdapter(ModelAdapter):
    """Returns a fixed, valid native trajectory (speed from `fixed_speed_mps` in models.yaml)."""

    def __init__(self, model_config: dict, agent_config: dict) -> None:
        super().__init__(model_config, agent_config)
        self.trajectory = fixed_trajectory(model_config["fixed_speed_mps"], model_config["trajectory"])

    def decide(self, prompt: Prompt) -> Decision:
        """Ignore the prompt and return the fixed trajectory."""
        start = time.perf_counter()
        return Decision(
            raw_output="",
            truncated=False,
            total_s=time.perf_counter() - start,
            native_trajectory=self.trajectory,
        )
