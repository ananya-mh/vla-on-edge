"""A stand-in model that returns random valid actions as JSON text.

It goes through the same prompt building, parsing, validation, and logging as a real
model, so the whole pipeline can run end to end without a GPU.
"""

import json
import random
import time

from src.adapters.base import Action, Decision, MetaAction, ModelAdapter, Observation
from src.agent.prompts import build_prompt, prompt_hash
from src.controller.trajectory import meta_action_to_waypoints


class DummyAdapter(ModelAdapter):
    """Picks a random meta-action each step (seeded by `seed` in models.yaml)."""

    def __init__(self, model_config: dict, agent_config: dict) -> None:
        super().__init__(model_config, agent_config)
        self.rng = random.Random(model_config.get("seed", 0))

    def decide(self, observation: Observation, history: list[Action]) -> Decision:
        """Return a random action in the configured output format."""
        start = time.perf_counter()
        prompt = build_prompt(observation, history, self.agent_config)
        meta_action = self.rng.choice(list(MetaAction))

        if self.agent_config["action_format"] == "waypoints":
            # Waypoints from the controller are valid by construction.
            waypoints = meta_action_to_waypoints(meta_action, observation.ego.speed_mps, self.agent_config)
            output = {"waypoints": [[round(p.x, 3), round(p.y, 3), p.t] for p in waypoints]}
        else:
            output = {"action": meta_action.value}

        return Decision(
            raw_output=json.dumps(output),
            prompt_hash=prompt_hash(prompt),
            truncated=False,
            total_s=time.perf_counter() - start,
        )
