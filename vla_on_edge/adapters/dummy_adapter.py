"""A stand-in model that returns random valid actions as JSON text.

It receives the same prompt as a real model and its output goes through the same
parsing, validation, and logging, so the whole pipeline can run end to end without
a GPU.
"""

import json
import random
import re
import time

from vla_on_edge.adapters.base import Decision, MetaAction, ModelAdapter, Prompt
from vla_on_edge.controller.trajectory import meta_action_to_waypoints

# Like a real model, the dummy reads the current speed from the prompt text.
SPEED_PATTERN = re.compile(r"Speed: ([0-9.]+) m/s")


def read_speed(prompt: Prompt) -> float:
    """Return the current speed stated in the prompt's user text."""
    match = SPEED_PATTERN.search(prompt.user_text)
    if match is None:
        raise ValueError("dummy adapter expects 'Speed: <number> m/s' in the prompt")
    return float(match.group(1))


class DummyAdapter(ModelAdapter):
    """Picks a random meta-action each step (seeded by `seed` in models.yaml)."""

    def __init__(self, model_config: dict, agent_config: dict) -> None:
        super().__init__(model_config, agent_config)
        self.rng = random.Random(model_config.get("seed", 0))

    def decide(self, prompt: Prompt) -> Decision:
        """Return a random action in the configured output format."""
        start = time.perf_counter()
        meta_action = self.rng.choice(list(MetaAction))

        if self.agent_config["action_format"] == "waypoints":
            # Waypoints from the controller are valid by construction.
            waypoints = meta_action_to_waypoints(meta_action, read_speed(prompt), self.agent_config)
            output = {"waypoints": [[round(p.x, 3), round(p.y, 3), p.t] for p in waypoints]}
        else:
            output = {"action": meta_action.value}

        return Decision(raw_output=json.dumps(output), truncated=False, total_s=time.perf_counter() - start)
