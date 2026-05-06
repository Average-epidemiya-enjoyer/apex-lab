"""Model Predictive Control planner."""

from __future__ import annotations

from typing import Any

from apex_lab.algorithms.base import BasePlanner, CarState, Trajectory


class MPCPlanner(BasePlanner):
    """Receding-horizon MPC that optimises a short trajectory at each step.

    Solves a constrained quadratic programme over a prediction horizon,
    respecting track boundaries and car dynamics linearised around the
    current state.
    """

    def __init__(
        self,
        horizon: int = 20,
        dt: float = 0.05,
        max_iter: int = 100,
    ) -> None:
        """
        Args:
            horizon: Number of time steps in the prediction horizon.
            dt: Simulation time step in seconds.
            max_iter: Maximum QP solver iterations per planning call.
        """
        self._horizon = horizon
        self._dt = dt
        self._max_iter = max_iter

    @property
    def name(self) -> str:
        return "mpc"

    def plan(self, track: Any, car_state: CarState) -> Trajectory:
        raise NotImplementedError("MPCPlanner.plan() is not implemented yet.")

    def reset(self) -> None:
        pass
