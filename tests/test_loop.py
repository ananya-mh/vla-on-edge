"""Tests for the agent loop's fallback, safety override, and truncation handling."""

from vla_on_edge.adapters.base import Decision, ModelAdapter
from vla_on_edge.agent.loop import run_step
from vla_on_edge.sim.mock_env import MockEnv


class ScriptedAdapter(ModelAdapter):
    """Returns a fixed raw output, optionally marked truncated."""

    def __init__(self, raw_output: str, truncated: bool = False) -> None:
        super().__init__({}, {})
        self.raw_output = raw_output
        self.truncated = truncated

    def decide(self, prompt):
        return Decision(raw_output=self.raw_output, truncated=self.truncated, total_s=0.0)


class TimingOutAdapter(ModelAdapter):
    def __init__(self) -> None:
        super().__init__({}, {})

    def decide(self, prompt):
        raise TimeoutError("no answer")


def place_ego_at_gap(env, gap_m):
    """Move the ego so the blocked_lane obstacle (x=60, radius 2) is `gap_m` ahead."""
    env.ego.x = 60.0 - 2.0 - env.scenario["ego_radius_m"] - gap_m


def make_env(agent_config, scenario):
    env = MockEnv(agent_config)
    env.reset(scenario, seed=0)
    return env


def test_valid_output_is_executed(agent_config, scenarios, model_config):
    env = make_env(agent_config, scenarios["open_corridor"])
    record, action = run_step(env, ScriptedAdapter('{"action": "ACCELERATE"}'), [], agent_config, model_config)
    assert record["fallback_reason"] is None
    assert record["executed_action"] == {"meta_action": "ACCELERATE"}


def test_parse_error_falls_back_to_slow_down_on_open_road(agent_config, scenarios, model_config):
    env = make_env(agent_config, scenarios["open_corridor"])
    record, _ = run_step(env, ScriptedAdapter("I think we should accelerate."), [], agent_config, model_config)
    assert record["fallback_reason"] == "parse_error"
    assert record["parsed_action"] is None
    assert record["executed_action"] == {"meta_action": "SLOW_DOWN"}


def test_fallback_is_stop_near_obstacle(agent_config, scenarios, model_config):
    # At 8 m/s the stopping distance is 8 m, so the fallback threshold is 8 + 10 m.
    agent_config = {**agent_config, "safety": {**agent_config["safety"], "enabled": True}}
    env = make_env(agent_config, scenarios["blocked_lane"])
    place_ego_at_gap(env, 17.0)  # inside the fallback threshold, outside safety's 8 + 5 m
    record, _ = run_step(env, ScriptedAdapter("garbage"), [], agent_config, model_config)
    assert record["fallback_reason"] == "parse_error"
    assert record["safety_override"] is None
    assert record["executed_action"] == {"meta_action": "STOP"}


def test_timeout_falls_back(agent_config, scenarios, model_config):
    env = make_env(agent_config, scenarios["open_corridor"])
    record, _ = run_step(env, TimingOutAdapter(), [], agent_config, model_config)
    assert record["fallback_reason"] == "timeout"
    assert record["raw_output"] is None
    assert record["prompt_hash"]  # the prompt was still built and hashed


def test_adapter_receives_the_loop_built_prompt(agent_config, scenarios, model_config):
    received = []

    class RecordingAdapter(ScriptedAdapter):
        def decide(self, prompt):
            received.append(prompt)
            return super().decide(prompt)

    env = make_env(agent_config, scenarios["blocked_lane"])
    privileged_gap = env.observe().nearest_obstacle_ahead_m
    run_step(env, RecordingAdapter('{"action": "KEEP_LANE"}'), [], agent_config, model_config)
    prompt = received[0]
    assert "Speed: 8.0 m/s" in prompt.user_text
    assert set(prompt.images) == {"front", "rear"}
    # Privileged data must never reach the model.
    assert f"{privileged_gap:.0f}" not in prompt.user_text


def test_validation_error_keeps_parsed_action_in_log(agent_config, scenarios, model_config):
    agent_config = {**agent_config, "action_format": "waypoints"}
    env = make_env(agent_config, scenarios["open_corridor"])
    short_list = '{"waypoints": [[4, 0, 0.5], [8, 0, 1.0]]}'
    record, _ = run_step(env, ScriptedAdapter(short_list, truncated=True), [], agent_config, model_config)
    assert record["fallback_reason"] == "validation_error"
    assert record["parsed_action"] is not None
    assert record["truncated"] is True


def test_truncated_but_valid_output_is_accepted(agent_config, scenarios, model_config):
    env = make_env(agent_config, scenarios["open_corridor"])
    record, _ = run_step(env, ScriptedAdapter('{"action": "KEEP_LANE"}', truncated=True), [], agent_config, model_config)
    assert record["truncated"] is True
    assert record["fallback_reason"] is None
    assert record["executed_action"] == {"meta_action": "KEEP_LANE"}


def test_safety_is_off_by_default(agent_config, scenarios, model_config):
    env = make_env(agent_config, scenarios["blocked_lane"])
    place_ego_at_gap(env, 3.0)
    record, _ = run_step(env, ScriptedAdapter('{"action": "ACCELERATE"}'), [], agent_config, model_config)
    assert record["safety_override"] is None
    assert record["executed_action"] == {"meta_action": "ACCELERATE"}


def test_safety_override_is_logged_separately_from_fallback(agent_config, scenarios, model_config):
    agent_config = {**agent_config, "safety": {**agent_config["safety"], "enabled": True}}
    env = make_env(agent_config, scenarios["blocked_lane"])
    place_ego_at_gap(env, 3.0)
    record, _ = run_step(env, ScriptedAdapter('{"action": "ACCELERATE"}'), [], agent_config, model_config)
    assert record["fallback_reason"] is None
    assert record["parsed_action"] == {"meta_action": "ACCELERATE"}
    assert record["safety_override"] == "obstacle_within_stopping_distance"
    assert record["executed_action"] == {"meta_action": "STOP"}


def test_full_native_trajectory_is_logged(agent_config, scenarios, model_config):
    # A 10 Hz, 6.4 s model: all 64 points must reach the log unchanged.
    agent_config = {**agent_config, "action_format": "waypoints"}
    spec = {"waypoint_count": 64, "spacing_s": 0.1, "horizon_s": 6.4}
    points = [[round(8.0 * 0.1 * (i + 1), 3), 0.0, round(0.1 * (i + 1), 1)] for i in range(64)]
    env = make_env(agent_config, scenarios["open_corridor"])
    raw = '{"waypoints": ' + str(points) + "}"
    record, _ = run_step(env, ScriptedAdapter(raw), [], agent_config, {**model_config, "trajectory": spec})
    assert record["fallback_reason"] is None
    logged = [[p["x"], p["y"], p["t"]] for p in record["parsed_action"]["waypoints"]]
    assert logged == points
    assert record["executed_action"] == record["parsed_action"]


TOO_FAST = '{"waypoints": [[10, 0, 0.5], [20, 0, 1.0], [30, 0, 1.5], [40, 0, 2.0], [50, 0, 2.5], [60, 0, 3.0]]}'


def test_enforce_mode_falls_back_and_logs_all_violations(agent_config, scenarios, model_config):
    agent_config = {**agent_config, "action_format": "waypoints"}
    env = make_env(agent_config, scenarios["open_corridor"])
    record, _ = run_step(env, ScriptedAdapter(TOO_FAST), [], agent_config, model_config)
    assert record["fallback_reason"] == "validation_error"
    assert record["executed_action"] == {"meta_action": "SLOW_DOWN"}
    # 20 m/s from 8 m/s breaks both the speed and the acceleration limits.
    assert len(record["validation_violations"]) == 2
    assert record["fallback_detail"] == record["validation_violations"][0]


def test_flag_only_mode_logs_violations_and_keeps_the_trajectory(agent_config, scenarios, model_config):
    agent_config = {**agent_config, "action_format": "waypoints"}
    env = make_env(agent_config, scenarios["open_corridor"])
    flag_only = {**model_config, "validation_mode": "flag_only"}
    record, action = run_step(env, ScriptedAdapter(TOO_FAST), [], agent_config, flag_only)
    assert record["fallback_reason"] is None
    assert len(record["validation_violations"]) == 2
    assert record["executed_action"] == record["parsed_action"]
    assert action.waypoints[-1].x == 60.0


def test_flag_only_still_falls_back_on_parse_errors(agent_config, scenarios, model_config):
    # Flag-only covers validation; unparseable output has no trajectory to keep.
    env = make_env(agent_config, scenarios["open_corridor"])
    flag_only = {**model_config, "validation_mode": "flag_only"}
    record, _ = run_step(env, ScriptedAdapter("garbage"), [], agent_config, flag_only)
    assert record["fallback_reason"] == "parse_error"
