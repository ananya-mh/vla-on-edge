"""The observe → reason → act loop that drives one episode with one model.

Per step: observe → build one model-agnostic prompt → call the adapter → parse
strictly → validate → fall back if needed → safety check → step the simulator →
write one JSONL record. Metrics are
computed later from these records, never here.

In milestone (a) the loop talks to the SimEnv directly; milestone (b) puts the MCP
server between them.
"""

import time
from dataclasses import asdict
from pathlib import Path

from src.adapters.base import Action, Decision, ModelAdapter, Observation, Prompt
from src.agent.parser import ParseError, parse_action
from src.agent.prompts import build_prompt, prompt_hash
from src.agent.safety import apply_safety_check, choose_fallback
from src.agent.validator import find_violations
from src.sim.base import SimEnv
from src.utils.logging import append_jsonl


def call_adapter(adapter: ModelAdapter, prompt: Prompt, timeout_s: float) -> tuple[Decision | None, str | None]:
    """Call the adapter and return (decision, timeout_detail).

    timeout_detail is None unless the adapter raised TimeoutError or answered late.
    A late answer is still returned so its raw output can be logged.
    """
    start = time.perf_counter()
    try:
        decision = adapter.decide(prompt)
    except TimeoutError as error:
        return None, str(error) or "adapter raised TimeoutError"
    elapsed = time.perf_counter() - start
    if elapsed > timeout_s:
        return decision, f"took {elapsed:.2f} s, limit is {timeout_s} s"
    return decision, None


def parse_and_validate(
    raw_output: str, observation: Observation, agent_config: dict, model_config: dict
) -> tuple[Action | None, list[str], str | None, str | None]:
    """Return (parsed_action, violations, fallback_reason, fallback_detail).

    parsed_action is None if parsing failed. If parsing worked but validation did
    not, the parsed action is still returned so it can be logged. In `flag_only`
    mode, violations are returned but never cause a fallback.
    """
    try:
        action = parse_action(raw_output, agent_config["action_format"])
    except ParseError as error:
        return None, [], "parse_error", str(error)
    violations = find_violations(
        action, observation.ego.speed_mps, model_config["trajectory"], agent_config["limits"]
    )
    if violations and model_config["validation_mode"] == "enforce":
        return action, violations, "validation_error", violations[0]
    return action, violations, None, None


def run_step(
    env: SimEnv, adapter: ModelAdapter, history: list[Action], agent_config: dict, model_config: dict
) -> tuple[dict, Action]:
    """Run one decision step. Return the step's log record and the executed action.

    `model_config` is the model's resolved models.yaml entry (trajectory shape and
    validation mode).
    """
    observation = env.observe()
    history_length = agent_config["history_length"]
    recent_history = history[-history_length:] if history_length > 0 else []

    step_start = time.perf_counter()
    prompt = build_prompt(observation, recent_history, agent_config, model_config["trajectory"])
    current_prompt_hash = prompt_hash(prompt)
    prompt_build_s = time.perf_counter() - step_start

    adapter_start = time.perf_counter()
    decision, timeout_detail = call_adapter(adapter, prompt, agent_config["timeout_s"])
    adapter_s = time.perf_counter() - adapter_start

    parse_start = time.perf_counter()
    parsed_action, violations, fallback_reason, fallback_detail = None, [], None, None
    if timeout_detail is not None:
        fallback_reason, fallback_detail = "timeout", timeout_detail
    else:
        parsed_action, violations, fallback_reason, fallback_detail = parse_and_validate(
            decision.raw_output, observation, agent_config, model_config
        )
    parse_validate_s = time.perf_counter() - parse_start

    chosen_action = choose_fallback(observation, agent_config) if fallback_reason else parsed_action
    executed_action, safety_override = apply_safety_check(chosen_action, observation, agent_config)
    end_to_end_s = time.perf_counter() - step_start

    events = env.step(executed_action)
    record = {
        "step_index": observation.step_index,
        "timestamp_s": observation.timestamp_s,
        "prompt_hash": current_prompt_hash,
        "raw_output": decision.raw_output if decision else None,
        # The model's full native trajectory, exactly as output (never resampled).
        "parsed_action": parsed_action.to_dict() if parsed_action else None,
        "executed_action": executed_action.to_dict(),
        "validation_violations": violations,
        "fallback_reason": fallback_reason,
        "fallback_detail": fallback_detail,
        "safety_override": safety_override,
        "truncated": decision.truncated if decision else False,
        "prompt_tokens": decision.prompt_tokens if decision else None,
        "completion_tokens": decision.completion_tokens if decision else None,
        "latency_s": {
            "prompt_build": prompt_build_s,
            "adapter": adapter_s,
            "ttft": decision.ttft_s if decision else None,
            "decode": decision.decode_s if decision else None,
            "parse_validate": parse_validate_s,
            "end_to_end": end_to_end_s,
        },
        "ego": asdict(observation.ego),
        "nearest_obstacle_ahead_m": observation.nearest_obstacle_ahead_m,
        "events": asdict(events),
    }
    return record, executed_action


def run_episode(
    env: SimEnv,
    adapter: ModelAdapter,
    scenario: dict,
    seed: int,
    agent_config: dict,
    model_config: dict,
    steps_log_path: Path,
) -> dict:
    """Drive one episode, append a record per step to `steps_log_path`, return the sim's metrics."""
    env.reset(scenario, seed)
    history: list[Action] = []
    while not env.done():
        record, executed_action = run_step(env, adapter, history, agent_config, model_config)
        history.append(executed_action)
        append_jsonl(steps_log_path, {"scenario": scenario["name"], "seed": seed, **record})
    return env.metrics()
