"""A* grid-based trajectory planner."""

from __future__ import annotations

from typing import Any

import numpy as np

from apex_lab.algorithms.base import BasePlanner, CarState, Trajectory


class AStarPlanner(BasePlanner):
    """Heuristic grid search using the A* algorithm.

    Discretises the track occupancy grid and finds the shortest path
    from the car's current cell to the goal cell, then smooths the
    resulting waypoints.
    """

    def __init__(self, resolution: float = 0.5, heuristic: str = "euclidean") -> None:
        """
        Args:
            resolution: Grid cell size in metres.
            heuristic: Distance heuristic — ``"euclidean"`` or ``"manhattan"``.
        """
        self._resolution = resolution
        self._heuristic = heuristic

    @property
    def name(self) -> str:
        return "a_star"

    def plan(self, track: Any, car_state: CarState) -> Trajectory:
        """Return an A* path from *car_state* to the finish line.

        Not yet implemented — returns a straight-line stub.
        """
        raise NotImplementedError("AStarPlanner.plan() is not implemented yet.")

    def reset(self) -> None:
        pass
