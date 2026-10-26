"""Tests that the step log keeps the full pose of every waypoint."""

from vla_on_edge.adapters.base import Action, Waypoint
from vla_on_edge.utils.logging import append_jsonl, read_jsonl

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def test_full_pose_round_trips_through_the_log(tmp_path):
    waypoints = [Waypoint(x=0.8 * (i + 1), y=0.01 * i, t=0.1 * (i + 1), z=0.02, rotation=IDENTITY) for i in range(64)]
    path = tmp_path / "steps.jsonl"
    append_jsonl(path, {"parsed_action": Action(waypoints=waypoints).to_dict()})

    logged = read_jsonl(path)[0]["parsed_action"]["waypoints"]
    assert len(logged) == 64
    assert logged[5] == {
        "x": waypoints[5].x,
        "y": waypoints[5].y,
        "t": waypoints[5].t,
        "z": 0.02,
        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }


def test_waypoints_without_pose_log_only_position_and_time():
    assert Waypoint(x=1.0, y=2.0, t=0.5).to_dict() == {"x": 1.0, "y": 2.0, "t": 0.5}
