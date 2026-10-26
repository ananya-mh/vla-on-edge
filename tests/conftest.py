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
def trajectory_spec() -> dict:
    """The dummy model's trajectory spec: 6 points, 0.5 s apart."""
    return load_model_config(CONFIG_DIR, "dummy", None)["trajectory"]
