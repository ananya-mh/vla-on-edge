"""Tests for strict parsing of model output."""

import pytest

from src.adapters.base import MetaAction
from src.agent.parser import ParseError, parse_action


def test_plain_meta_action():
    action = parse_action('{"action": "KEEP_LANE"}', "meta_action")
    assert action.meta_action == MetaAction.KEEP_LANE


def test_code_fence_and_whitespace_are_allowed():
    action = parse_action('\n```json\n{"action": "STOP"}\n```\n', "meta_action")
    assert action.meta_action == MetaAction.STOP


def test_think_block_is_stripped():
    raw = '<think>The road is clear,\nso keep going.</think>\n{"action": "ACCELERATE"}'
    assert parse_action(raw, "meta_action").meta_action == MetaAction.ACCELERATE


def test_unclosed_think_block_is_rejected():
    # Typical of a reasoning model cut off by max_tokens.
    with pytest.raises(ParseError, match="unclosed"):
        parse_action('<think>Let me consider the obstacle on the', "meta_action")


@pytest.mark.parametrize(
    "raw",
    [
        'I will keep my lane. {"action": "KEEP_LANE"}',  # prose around the JSON
        '{"action": "keep_lane"}',  # wrong case
        '{"action": "TURN_AROUND"}',  # unknown action
        '{"action": "STOP", "reason": "obstacle"}',  # extra key
        '["STOP"]',  # not an object
        '{"action": "STOP"',  # truncated JSON
        "",
    ],
)
def test_invalid_meta_action_outputs_are_rejected(raw):
    with pytest.raises(ParseError):
        parse_action(raw, "meta_action")


def test_waypoints_are_parsed():
    action = parse_action('{"waypoints": [[1, 0, 0.5], [2.5, 0.1, 1.0]]}', "waypoints")
    assert [(p.x, p.y, p.t) for p in action.waypoints] == [(1.0, 0.0, 0.5), (2.5, 0.1, 1.0)]


@pytest.mark.parametrize(
    "raw",
    [
        '{"waypoints": []}',
        '{"waypoints": [[1, 0]]}',  # missing t
        '{"waypoints": [[1, 0, "0.5"]]}',  # string number
        '{"waypoints": [[true, 0, 0.5]]}',  # bool is not a number here
        '{"action": "STOP"}',  # wrong format for this config
    ],
)
def test_invalid_waypoint_outputs_are_rejected(raw):
    with pytest.raises(ParseError):
        parse_action(raw, "waypoints")
