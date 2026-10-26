"""Tests for the logging wrapper, using a fake inner driver instead of Alpamayo.

Run inside AlpaSim's environment (it provides alpasim_driver), e.g.:
    third_party/alpasim/.venv/bin/python -m pytest plugins/alpasim_vla_on_edge/tests
"""

import json
from importlib.metadata import entry_points

import numpy as np
import pytest
from alpasim_driver.models.base import BaseTrajectoryModel, DriveCommand, ModelPrediction, PredictionInput

from alpasim_vla_on_edge.logged_model import LoggedAlpamayo15, LoggedModel, prediction_to_waypoints
from vla_on_edge.utils.config import CONFIG_DIR, load_model_config, load_yaml


class FakeInnerDriver(BaseTrajectoryModel):
    """Returns a fixed straight 64-point, 10 Hz trajectory at `speed_mps`."""

    MIN_FRAME_HW = (320, 576)

    def __init__(self, speed_mps: float) -> None:
        self.speed_mps = speed_mps
        self.calls = 0

    @classmethod
    def from_config(cls, model_cfg, device, camera_ids, context_length, output_frequency_hz):
        return cls(8.0)

    def _encode_command(self, command):
        return int(command)

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        self.calls += 1
        times = np.arange(1, 65) / 10.0
        positions = np.zeros((1, 64, 3))
        positions[0, :, 0] = self.speed_mps * times
        rotations = np.tile(np.eye(3), (1, 64, 1, 1))
        return ModelPrediction(candidate_positions=positions, candidate_rotations=rotations, reasoning_text="clear road")

    @property
    def camera_ids(self):
        return ["camera_front_wide_120fov"]

    @property
    def context_length(self):
        return 4

    @property
    def output_frequency_hz(self):
        return 10


def make_input(speed_mps: float = 8.0) -> PredictionInput:
    return PredictionInput(
        camera_images={},
        command=DriveCommand.STRAIGHT,
        speed=speed_mps,
        acceleration=0.0,
        ego_pose_history=[],
        inference_seed=7,
        previous_plan=None,
        route=None,
    )


def make_wrapper(inner: BaseTrajectoryModel, tmp_path) -> LoggedModel:
    model_config = load_model_config(CONFIG_DIR, "alpamayo1_5", None)
    limits = load_yaml(CONFIG_DIR / "agent.yaml")["limits"]
    return LoggedAlpamayo15(inner, model_config, limits, tmp_path)


def read_records(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]


def test_predictions_are_returned_unchanged_and_logged(tmp_path):
    inner = FakeInnerDriver(speed_mps=8.0)
    wrapper = make_wrapper(inner, tmp_path)
    expected = inner.predict(make_input())

    predictions = wrapper.predict_batch([make_input(), make_input()])

    assert len(predictions) == 2
    np.testing.assert_array_equal(predictions[0].candidate_positions, expected.candidate_positions)
    records = read_records(tmp_path)
    assert len(records) == 2
    record = records[0]
    assert record["log_format_version"] == 2
    assert record["sim_time_us"] is None  # the fake input has no pose history
    assert record["gpu_memory_mib"] is None  # no CUDA in tests
    assert record["model"] == "alpamayo1_5"
    assert record["validation_violations"] == []
    assert record["reasoning_text"] == "clear road"
    assert record["latency_s"]["batch_size"] == 2
    waypoints = record["executed_action"]["waypoints"]
    assert len(waypoints) == 64
    assert waypoints[-1]["t"] == pytest.approx(6.4)
    assert waypoints[-1]["rotation"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_violations_are_flagged_but_the_trajectory_is_kept(tmp_path):
    # 30 m/s exceeds max_speed_mps (15) and jumps from 8 m/s: flagged, never replaced.
    inner = FakeInnerDriver(speed_mps=30.0)
    wrapper = make_wrapper(inner, tmp_path)

    prediction = wrapper.predict(make_input(speed_mps=8.0))

    assert prediction.selected_positions[-1, 0] == pytest.approx(30.0 * 6.4)
    violations = read_records(tmp_path)[0]["validation_violations"]
    assert any("speed" in violation for violation in violations)
    assert any("acceleration" in violation for violation in violations)


def test_wrapper_delegates_model_properties(tmp_path):
    inner = FakeInnerDriver(speed_mps=8.0)
    wrapper = make_wrapper(inner, tmp_path)
    assert wrapper.camera_ids == inner.camera_ids
    assert wrapper.context_length == 4
    assert wrapper.output_frequency_hz == 10
    assert wrapper.MIN_FRAME_HW == (320, 576)
    assert wrapper._encode_command(DriveCommand.LEFT) == 0


def test_frequency_mismatch_with_models_yaml_is_rejected(tmp_path):
    class FiveHzDriver(FakeInnerDriver):
        @property
        def output_frequency_hz(self):
            return 5

    with pytest.raises(ValueError, match="Hz"):
        make_wrapper(FiveHzDriver(speed_mps=8.0), tmp_path)


def test_timestamps_from_the_prediction_are_used_when_present():
    prediction = FakeInnerDriver(speed_mps=8.0).predict(make_input())
    prediction.model_t0_us = 1_000_000
    prediction.waypoint_timestamps_us = 1_000_000 + np.arange(1, 65) * 100_000
    waypoints = prediction_to_waypoints(prediction, output_frequency_hz=10)
    assert waypoints[0].t == pytest.approx(0.1)
    assert waypoints[-1].t == pytest.approx(6.4)


def test_entry_point_is_registered_with_alpasim():
    names = [entry.name for entry in entry_points(group="alpasim.models")]
    assert "vla_on_edge_alpamayo1_5" in names
    assert "alpamayo1_5" in names  # the built-in model we wrap


def test_simulation_time_prefers_the_model_origin():
    from types import SimpleNamespace

    from alpasim_vla_on_edge.logged_model import simulation_time_us

    prediction = FakeInnerDriver(speed_mps=8.0).predict(make_input())
    with_history = make_input()
    with_history.ego_pose_history = [SimpleNamespace(timestamp_us=100), SimpleNamespace(timestamp_us=200)]
    assert simulation_time_us(with_history, prediction) == 200
    prediction.model_t0_us = 250
    assert simulation_time_us(with_history, prediction) == 250
