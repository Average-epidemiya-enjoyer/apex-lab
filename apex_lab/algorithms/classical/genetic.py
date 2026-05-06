"""Genetic algorithm trajectory planner."""

from __future__ import annotations

from typing import Any

from apex_lab.algorithms.base import BasePlanner, CarState, Trajectory


class GAPlanner(BasePlanner):
    """Evolves a population of candidate trajectories using a genetic algorithm.

    Each individual encodes a sequence of apex control points.  Fitness is
    evaluated by the simulator (lap time, track clearance, smoothness).
    """

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 200,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.8,
        seed: int | None = None,
    ) -> None:
        self._population_size = population_size
        self._generations = generations
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._seed = seed

    @property
    def name(self) -> str:
        return "genetic_algorithm"

    def plan(self, track: Any, car_state: CarState) -> Trajectory:
        raise NotImplementedError("GAPlanner.plan() is not implemented yet.")

    def reset(self) -> None:
        pass
