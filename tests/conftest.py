"""Shared test fixtures: the real configs from configs/, so tests exercise what runs use."""

import pytest

from src.utils.config import CONFIG_DIR, load_scenarios, load_yaml


@pytest.fixture
def agent_config() -> dict:
    return load_yaml(CONFIG_DIR / "agent.yaml")


@pytest.fixture
def scenarios() -> dict[str, dict]:
    return load_scenarios(CONFIG_DIR)
