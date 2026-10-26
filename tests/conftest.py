"""Shared test fixtures: the real configs from configs/, so tests exercise what runs use."""

import pytest

from src.utils.config import CONFIG_DIR, load_model_config, load_scenarios, load_yaml


@pytest.fixture
def agent_config() -> dict:
    return load_yaml(CONFIG_DIR / "agent.yaml")


@pytest.fixture
def scenarios() -> dict[str, dict]:
    return load_scenarios(CONFIG_DIR)


@pytest.fixture
def model_config() -> dict:
    """The dummy model's resolved config (6 points 0.5 s apart, enforce mode)."""
    return load_model_config(CONFIG_DIR, "dummy", None)


@pytest.fixture
def trajectory_spec(model_config) -> dict:
    return model_config["trajectory"]
