"""Checks the fixed AlpaSim scene list in configs/scenarios.yaml."""

import csv
import re
from pathlib import Path

import pytest

from vla_on_edge.utils.config import CONFIG_DIR, load_alpasim_scenes

SCENE_ID = re.compile(r"^clipgt-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
SUITES_CSV = Path(__file__).resolve().parents[1] / "third_party/alpasim/data/scenes/sim_suites.csv"


def test_scene_list_is_well_formed():
    config = load_alpasim_scenes(CONFIG_DIR)
    ids = [scene["id"] for scene in config["scenes"]]
    assert 5 <= len(ids) <= 10
    assert len(set(ids)) == len(ids)
    assert all(SCENE_ID.match(scene_id) for scene_id in ids)
    assert all(scene["reason"] for scene in config["scenes"])
    assert config["smoke_scene"] in ids


@pytest.mark.skipif(not SUITES_CSV.exists(), reason="AlpaSim clone not present")
def test_every_scene_is_in_the_configured_suite():
    config = load_alpasim_scenes(CONFIG_DIR)
    with open(SUITES_CSV) as file:
        in_suite = {row["scene_id"] for row in csv.DictReader(file) if row["test_suite_id"] == config["suite"]}
    assert {scene["id"] for scene in config["scenes"]} <= in_suite
