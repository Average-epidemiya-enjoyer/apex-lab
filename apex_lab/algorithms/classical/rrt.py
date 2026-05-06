"""Rapidly-exploring Random Tree (RRT) planner."""

from __future__ import annotations

from typing import Any

from apex_lab.algorithms.base import BasePlanner, CarState, Trajectory


class RRTPlanner(BasePlanner):
    """RRT: builds a random tree in configuration space until the goal is reached."""

    def __init__(self, max_iter: int = 5_000, step_size: float = 1.0, seed: int | None = None) -> None:
        """
        Args:
            max_iter: Maximum number of tree expansion iterations.
            step_size: Distance to extend the tree per iteration (metres).
            seed: Random seed for reproducibility.
        """
        self._max_iter = max_iter
        self._step_size = step_size
        self._seed = seed

    @property
    def name(self) -> str:
        return "rrt"

    def plan(self, track: Any, car_state: CarState) -> Trajectory:
        raise NotImplementedError("RRTPlanner.plan() is not implemented yet.")

    def reset(self) -> None:
        pass
