"""Asymptotically optimal RRT* planner."""

from __future__ import annotations

from typing import Any

from apex_lab.algorithms.base import BasePlanner, CarState, Trajectory


class RRTStarPlanner(BasePlanner):
    """RRT*: extends RRT with rewiring to converge toward the optimal path."""

    def __init__(
        self,
        max_iter: int = 10_000,
        step_size: float = 1.0,
        search_radius: float = 5.0,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            max_iter: Maximum number of iterations.
            step_size: Extension distance per iteration (metres).
            search_radius: Neighbourhood radius for rewiring (metres).
            seed: Random seed for reproducibility.
        """
        self._max_iter = max_iter
        self._step_size = step_size
        self._search_radius = search_radius
        self._seed = seed

    @property
    def name(self) -> str:
        return "rrt_star"

    def plan(self, track: Any, car_state: CarState) -> Trajectory:
        raise NotImplementedError("RRTStarPlanner.plan() is not implemented yet.")

    def reset(self) -> None:
        pass
