"""A tiny 2D driving world for developing and testing the pipeline without GPUs or the real simulator.

The world is a straight road corridor along the world x axis, centered on y = 0, with
round static obstacles. The goal is the end of the corridor. Camera frames are simple
top-down renders in the ego frame (one looking forward, one looking back).
"""

import math
from dataclasses import dataclass

import numpy as np

from src.adapters.base import Action, EgoState, Observation, Waypoint
from src.controller.trajectory import action_to_waypoints
from src.sim.base import SimEnv, StepEvents

ROAD_COLOR = (90, 90, 90)
OFF_ROAD_COLOR = (150, 120, 80)
OBSTACLE_COLOR = (200, 40, 40)
GOAL_COLOR = (40, 180, 40)

# Half-width of the time window used to estimate heading and speed from waypoints.
FINITE_DIFFERENCE_S = 0.05


@dataclass
class Obstacle:
    """A static round obstacle in the world frame."""

    x: float
    y: float
    radius: float


def position_at(waypoints: list[Waypoint], time_s: float) -> tuple[float, float]:
    """Linearly interpolate the ego-frame position at `time_s` along a waypoint path.

    The path implicitly starts at the ego's current position (0, 0) at t = 0.
    Times outside the path are clamped to its ends.
    """
    points = [Waypoint(0.0, 0.0, 0.0)] + list(waypoints)
    time_s = min(max(time_s, 0.0), points[-1].t)
    for start, end in zip(points, points[1:]):
        if time_s <= end.t:
            fraction = (time_s - start.t) / (end.t - start.t)
            return (
                start.x + fraction * (end.x - start.x),
                start.y + fraction * (end.y - start.y),
            )
    return points[-1].x, points[-1].y


def advance_along_waypoints(ego: EgoState, waypoints: list[Waypoint], step_s: float) -> EgoState:
    """Move the ego along an ego-frame waypoint path for `step_s` seconds and return its new state.

    Heading and speed come from a short central difference around `step_s`, which
    keeps turns accurate even though the path between waypoints is piecewise linear.
    """
    local_x, local_y = position_at(waypoints, step_s)
    before_time = max(step_s - FINITE_DIFFERENCE_S, 0.0)
    after_time = min(step_s + FINITE_DIFFERENCE_S, waypoints[-1].t)
    before_x, before_y = position_at(waypoints, before_time)
    after_x, after_y = position_at(waypoints, after_time)
    delta_x, delta_y = after_x - before_x, after_y - before_y
    distance = math.hypot(delta_x, delta_y)

    elapsed = after_time - before_time
    speed = distance / elapsed if elapsed > 0 else 0.0
    # A stationary vehicle keeps its heading instead of snapping to atan2(0, 0).
    heading_change = math.atan2(delta_y, delta_x) if distance > 1e-6 else 0.0

    cos_heading, sin_heading = math.cos(ego.heading_rad), math.sin(ego.heading_rad)
    return EgoState(
        x=ego.x + local_x * cos_heading - local_y * sin_heading,
        y=ego.y + local_x * sin_heading + local_y * cos_heading,
        heading_rad=ego.heading_rad + heading_change,
        speed_mps=speed,
    )


def create_obstacles(scenario: dict, seed: int) -> list[Obstacle]:
    """Return the scenario's fixed obstacles plus seeded random ones inside the corridor."""
    obstacles = [Obstacle(x, y, radius) for x, y, radius in scenario.get("fixed_obstacles", [])]
    rng = np.random.default_rng(seed)
    half_width = scenario["road_width_m"] / 2
    for _ in range(scenario.get("random_obstacles", 0)):
        radius = float(rng.uniform(1.0, 2.0))
        x = float(rng.uniform(scenario["obstacle_min_x_m"], scenario["road_length_m"] - 20.0))
        y = float(rng.uniform(-half_width + radius, half_width - radius))
        obstacles.append(Obstacle(x, y, radius))
    return obstacles


class MockEnv(SimEnv):
    """Mock simulator implementing the SimEnv interface."""

    def __init__(self, agent_config: dict) -> None:
        # The agent config supplies the controller settings used to turn meta-actions into paths.
        self.agent_config = agent_config
        self.scenario: dict = {}
        self.obstacles: list[Obstacle] = []
        self.ego = EgoState(0.0, 0.0, 0.0, 0.0)
        self.step_index = 0
        self.collided = False
        self.off_road_steps = 0
        self.furthest_x = 0.0

    def reset(self, scenario: dict, seed: int) -> None:
        """Start a new episode with obstacles placed according to `seed`."""
        self.scenario = scenario
        self.obstacles = create_obstacles(scenario, seed)
        self.ego = EgoState(x=0.0, y=0.0, heading_rad=0.0, speed_mps=scenario["initial_speed_mps"])
        self.step_index = 0
        self.collided = False
        self.off_road_steps = 0
        self.furthest_x = 0.0

    def observe(self) -> Observation:
        """Return camera renders, ego state, route, and the privileged obstacle distance."""
        road_length = self.scenario["road_length_m"]
        return Observation(
            camera_frames={"front": self.render_view("front"), "rear": self.render_view("rear")},
            ego=EgoState(self.ego.x, self.ego.y, self.ego.heading_rad, self.ego.speed_mps),
            route=[(0.0, 0.0), (road_length, 0.0)],
            goal=(road_length, 0.0),
            timestamp_s=self.step_index * self.scenario["step_dt_s"],
            step_index=self.step_index,
            nearest_obstacle_ahead_m=self.nearest_obstacle_ahead(),
        )

    def step(self, action: Action) -> StepEvents:
        """Follow the action's path for one step and check for collisions and leaving the road."""
        waypoints = action_to_waypoints(action, self.ego.speed_mps, self.agent_config)
        self.ego = advance_along_waypoints(self.ego, waypoints, self.scenario["step_dt_s"])
        self.step_index += 1
        self.furthest_x = max(self.furthest_x, self.ego.x)

        collided = self.is_colliding()
        off_road = self.is_off_road()
        self.collided = self.collided or collided
        if off_road:
            self.off_road_steps += 1
        return StepEvents(collided=collided, off_road=off_road, progress_fraction=self.progress_fraction())

    def done(self) -> bool:
        """The episode ends on collision, on reaching the goal, or at the step limit."""
        reached_goal = self.ego.x >= self.scenario["road_length_m"]
        out_of_steps = self.step_index >= self.scenario["max_steps"]
        return self.collided or reached_goal or out_of_steps

    def metrics(self) -> dict:
        """Return the episode summary."""
        return {
            "steps": self.step_index,
            "collided": self.collided,
            "reached_goal": self.ego.x >= self.scenario["road_length_m"],
            "off_road_steps": self.off_road_steps,
            "progress_fraction": self.progress_fraction(),
            "final_speed_mps": self.ego.speed_mps,
        }

    def progress_fraction(self) -> float:
        """Return how far along the corridor the ego has ever been, from 0 to 1."""
        return min(max(self.furthest_x / self.scenario["road_length_m"], 0.0), 1.0)

    def is_colliding(self) -> bool:
        """Return True if the ego's footprint circle overlaps any obstacle."""
        ego_radius = self.scenario["ego_radius_m"]
        for obstacle in self.obstacles:
            distance = math.hypot(self.ego.x - obstacle.x, self.ego.y - obstacle.y)
            if distance < ego_radius + obstacle.radius:
                return True
        return False

    def is_off_road(self) -> bool:
        """Return True if any part of the ego's footprint is outside the corridor."""
        half_width = self.scenario["road_width_m"] / 2
        return abs(self.ego.y) + self.scenario["ego_radius_m"] > half_width

    def nearest_obstacle_ahead(self) -> float | None:
        """Return the gap (m) to the nearest obstacle in the ego's straight-ahead path, or None."""
        ego_radius = self.scenario["ego_radius_m"]
        cos_heading, sin_heading = math.cos(self.ego.heading_rad), math.sin(self.ego.heading_rad)
        nearest = None
        for obstacle in self.obstacles:
            delta_x, delta_y = obstacle.x - self.ego.x, obstacle.y - self.ego.y
            forward = delta_x * cos_heading + delta_y * sin_heading
            lateral = -delta_x * sin_heading + delta_y * cos_heading
            if forward <= 0 or abs(lateral) >= ego_radius + obstacle.radius:
                continue
            gap = max(forward - ego_radius - obstacle.radius, 0.0)
            if nearest is None or gap < nearest:
                nearest = gap
        return nearest

    def render_view(self, view_name: str) -> np.ndarray:
        """Render a top-down image in the ego frame, looking forward ("front") or back ("rear")."""
        size = self.scenario["image_size_px"]
        view_range = self.scenario["view_range_m"]
        offsets = (np.arange(size) + 0.5) * (view_range / size)
        if view_name == "front":
            forward = view_range - offsets  # top row is farthest ahead
            left = view_range / 2 - offsets  # left image edge is the ego's left
        else:
            forward = -(view_range - offsets)  # top row is farthest behind
            left = -(view_range / 2 - offsets)  # a rear camera sees the ego's right on its left
        forward_grid, left_grid = np.meshgrid(forward, left, indexing="ij")

        cos_heading, sin_heading = math.cos(self.ego.heading_rad), math.sin(self.ego.heading_rad)
        world_x = self.ego.x + forward_grid * cos_heading - left_grid * sin_heading
        world_y = self.ego.y + forward_grid * sin_heading + left_grid * cos_heading

        image = np.empty((size, size, 3), dtype=np.uint8)
        image[:] = OFF_ROAD_COLOR
        image[np.abs(world_y) <= self.scenario["road_width_m"] / 2] = ROAD_COLOR
        road_length = self.scenario["road_length_m"]
        image[(world_x >= road_length) & (world_x <= road_length + 2.0)] = GOAL_COLOR
        for obstacle in self.obstacles:
            inside = (world_x - obstacle.x) ** 2 + (world_y - obstacle.y) ** 2 <= obstacle.radius**2
            image[inside] = OBSTACLE_COLOR
        return image
