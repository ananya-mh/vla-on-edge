"""Strict parsing of raw model text into an Action.

Accepted output, after removing `<think>...</think>` blocks and surrounding whitespace,
is exactly one JSON object, optionally wrapped in a ```json code fence:
    {"action": "KEEP_LANE"}                       when action_format is meta_action
    {"waypoints": [[x, y, t], [x, y, t], ...]}    when action_format is waypoints
Anything else is a parse error. Checking the number of waypoints and physical limits
is the validator's job.
"""

import json
import re

from vla_on_edge.adapters.base import Action, MetaAction, Waypoint

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class ParseError(ValueError):
    """Raised when model output does not match the expected action format."""


def strip_reasoning(raw_text: str) -> str:
    """Remove complete <think>...</think> blocks; reject an unclosed one."""
    text = THINK_BLOCK.sub("", raw_text)
    if "<think>" in text:
        raise ParseError("unclosed <think> block")
    return text.strip()


def parse_action(raw_text: str, action_format: str) -> Action:
    """Parse model output into an Action, or raise ParseError explaining why not."""
    text = strip_reasoning(raw_text)
    fence_match = CODE_FENCE.match(text)
    if fence_match:
        text = fence_match.group(1)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ParseError(f"not valid JSON: {error.msg}") from error
    if not isinstance(data, dict):
        raise ParseError("JSON is not an object")

    if action_format == "meta_action":
        return parse_meta_action(data)
    if action_format == "waypoints":
        return parse_waypoints(data)
    raise ValueError(f"unknown action_format: {action_format}")


def parse_meta_action(data: dict) -> Action:
    """Parse {"action": "<META_ACTION>"}."""
    if set(data) != {"action"}:
        raise ParseError(f"expected exactly the key 'action', got {sorted(data)}")
    name = data["action"]
    if not isinstance(name, str) or name not in MetaAction.__members__:
        raise ParseError(f"unknown meta-action: {name!r}")
    return Action(meta_action=MetaAction(name))


def parse_waypoints(data: dict) -> Action:
    """Parse {"waypoints": [[x, y, t], ...]}."""
    if set(data) != {"waypoints"}:
        raise ParseError(f"expected exactly the key 'waypoints', got {sorted(data)}")
    points = data["waypoints"]
    if not isinstance(points, list) or not points:
        raise ParseError("'waypoints' must be a non-empty list")

    waypoints = []
    for point in points:
        if not isinstance(point, list) or len(point) != 3:
            raise ParseError(f"each waypoint must be [x, y, t], got {point!r}")
        for value in point:
            # bool is a subclass of int in Python, so exclude it explicitly.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ParseError(f"waypoint values must be numbers, got {point!r}")
        waypoints.append(Waypoint(x=float(point[0]), y=float(point[1]), t=float(point[2])))
    return Action(waypoints=waypoints)
