"""A thin wrapper around one of AlpaSim's built-in driver models.

AlpaSim calls `predict_batch` on the wrapper; the wrapper calls the real model
(e.g. AlpaSim's own `alpamayo1_5`), validates each returned trajectory in flag-only
mode, writes one JSONL record per prediction, and returns the predictions
unchanged. It never reimplements or alters the wrapped model.

Each wrapped model gets its own entry point (see pyproject.toml), because AlpaSim's
driver ModelConfig has a fixed schema with no room for our settings. The log
directory comes from the VLA_ON_EDGE_LOG_DIR environment variable.
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
from alpasim_driver.models.base import (
    BaseTrajectoryModel,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)
from alpasim_plugins.plugins import models as model_registry

from vla_on_edge.adapters.base import Action, Waypoint
from vla_on_edge.agent.validator import find_violations
from vla_on_edge.utils.config import CONFIG_DIR, load_model_config, load_yaml
from vla_on_edge.utils.logging import LOG_FORMAT_VERSION, append_jsonl
from vla_on_edge.utils.run_metadata import build_run_metadata, save_run_metadata

LOG_DIR_VARIABLE = "VLA_ON_EDGE_LOG_DIR"


def prediction_to_waypoints(prediction: ModelPrediction, output_frequency_hz: int) -> list[Waypoint]:
    """Convert the driven candidate into our Waypoints, keeping the full pose.

    AlpaSim's rig frame is x forward, y left, z up, matching our ego frame.
    Waypoint times come from the prediction when it reports them, otherwise from
    the output frequency (waypoint i is at (i + 1) / frequency seconds).
    """
    positions = prediction.selected_positions
    rotations = prediction.selected_rotations
    timestamps = prediction.waypoint_timestamps_us
    waypoints = []
    for index in range(len(positions)):
        if timestamps is not None and prediction.model_t0_us is not None:
            time_s = (int(timestamps[index]) - prediction.model_t0_us) / 1e6
        else:
            time_s = (index + 1) / output_frequency_hz
        x, y, z = (float(value) for value in positions[index])
        rotation = tuple(tuple(float(value) for value in row) for row in rotations[index])
        waypoints.append(Waypoint(x=x, y=y, t=time_s, z=z, rotation=rotation))
    return waypoints


class LoggedModel(BaseTrajectoryModel):
    """Wraps `inner` and logs every prediction. Subclasses set the two names below."""

    INNER_MODEL_TYPE: str = ""  # AlpaSim entry-point name of the model to wrap
    MODEL_NAME: str = ""  # our models.yaml key for the wrapped model

    def __init__(self, inner: BaseTrajectoryModel, model_config: dict, limits: dict, log_dir: Path) -> None:
        self.inner = inner
        self.model_config = model_config
        self.limits = limits
        self.steps_log_path = log_dir / "steps.jsonl"
        # The driver reads this class-level setting, so mirror the wrapped model's.
        self.MIN_FRAME_HW = inner.MIN_FRAME_HW
        check_frequency_matches(model_config, inner.output_frequency_hz)

    @classmethod
    def from_config(cls, model_cfg, device, camera_ids, context_length, output_frequency_hz) -> "LoggedModel":
        """Build the wrapped AlpaSim model from the same arguments, then wrap it."""
        inner_class = model_registry.get(cls.INNER_MODEL_TYPE)
        inner = inner_class.from_config(model_cfg, device, camera_ids, context_length, output_frequency_hz)
        model_config = load_model_config(CONFIG_DIR, cls.MODEL_NAME, None)
        limits = load_yaml(CONFIG_DIR / "agent.yaml")["limits"]
        log_dir = log_dir_from_environment()
        save_run_metadata(log_dir, build_run_metadata({"model": model_config, "limits": limits}))
        return cls(inner, model_config, limits, log_dir)

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        """Single-input form; AlpaSim's driver itself calls predict_batch."""
        return self.predict_batch([prediction_input])[0]

    def predict_batch(self, prediction_inputs: list[PredictionInput]) -> list[ModelPrediction]:
        """Call the wrapped model, log each prediction, and return them unchanged."""
        start = time.perf_counter()
        predictions = self.inner.predict_batch(prediction_inputs)
        batch_s = time.perf_counter() - start
        for prediction_input, prediction in zip(prediction_inputs, predictions, strict=True):
            record = self.build_record(prediction_input, prediction, batch_s, len(prediction_inputs))
            append_jsonl(self.steps_log_path, record)
        return predictions

    def build_record(
        self, prediction_input: PredictionInput, prediction: ModelPrediction, batch_s: float, batch_size: int
    ) -> dict:
        """Return one step record: the full native trajectory plus flag-only violations."""
        waypoints = prediction_to_waypoints(prediction, self.inner.output_frequency_hz)
        action = Action(waypoints=waypoints)
        violations = find_violations(action, prediction_input.speed, self.model_config["trajectory"], self.limits)
        return {
            "log_format_version": LOG_FORMAT_VERSION,
            "model": self.MODEL_NAME,
            "wall_time_s": time.time(),
            "sim_time_us": simulation_time_us(prediction_input, prediction),
            "inference_seed": prediction_input.inference_seed,
            "command": prediction_input.command.name,
            "ego_speed_mps": prediction_input.speed,
            "ego_acceleration_mps2": prediction_input.acceleration,
            "executed_action": action.to_dict(),  # never replaced: flag-only
            "validation_violations": violations,
            "num_candidates": int(len(prediction.candidate_positions)),
            "selected_index": prediction.selected_index,
            "reasoning_text": prediction.reasoning_text,
            # With batch_size > 1 the per-prediction latency is not separable.
            "latency_s": {"batch": batch_s, "batch_size": batch_size},
            "gpu_memory_mib": gpu_memory_mib(),
        }

    # Everything else is delegated to the wrapped model.
    def _encode_command(self, command: DriveCommand):
        return self.inner._encode_command(command)

    @property
    def camera_ids(self) -> list[str]:
        return self.inner.camera_ids

    @property
    def context_length(self) -> int:
        return self.inner.context_length

    @property
    def output_frequency_hz(self) -> int:
        return self.inner.output_frequency_hz


class LoggedAlpamayo15(LoggedModel):
    """AlpaSim's built-in Alpamayo 1.5 driver, logged."""

    INNER_MODEL_TYPE = "alpamayo1_5"
    MODEL_NAME = "alpamayo1_5"


def simulation_time_us(prediction_input: PredictionInput, prediction: ModelPrediction) -> int | None:
    """Return the simulation time of this prediction, for per-rollout time series (e.g. jerk).

    Prefers the model's own input time origin; otherwise the newest ego pose.
    """
    if prediction.model_t0_us is not None:
        return int(prediction.model_t0_us)
    if prediction_input.ego_pose_history:
        return int(prediction_input.ego_pose_history[-1].timestamp_us)
    return None


def gpu_memory_mib() -> dict | None:
    """Return PyTorch's peak GPU memory in this (driver) process, or None without CUDA.

    This is the model's memory only: the renderer runs in a separate container.
    """
    if not torch.cuda.is_available():
        return None
    mebibyte = 2**20
    return {
        "peak_allocated": torch.cuda.max_memory_allocated() / mebibyte,
        "peak_reserved": torch.cuda.max_memory_reserved() / mebibyte,
    }


def check_frequency_matches(model_config: dict, output_frequency_hz: int) -> None:
    """Fail early if the wrapped model's waypoint rate disagrees with models.yaml."""
    expected_hz = 1.0 / model_config["trajectory"]["spacing_s"]
    if not np.isclose(expected_hz, output_frequency_hz):
        raise ValueError(
            f"{model_config['name']}: models.yaml spacing implies {expected_hz:g} Hz, "
            f"but the wrapped model outputs {output_frequency_hz} Hz"
        )


def log_dir_from_environment() -> Path:
    """Return (and create) the log directory named by VLA_ON_EDGE_LOG_DIR."""
    value = os.environ.get(LOG_DIR_VARIABLE)
    if not value:
        raise RuntimeError(f"set {LOG_DIR_VARIABLE} to the directory where step logs should go")
    log_dir = Path(value)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir
