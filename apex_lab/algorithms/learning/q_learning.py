"""Tabular Q-Learning agent."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex_lab.algorithms.base import BaseAgent, Batch, Observation


class QLearningAgent(BaseAgent):
    """Tabular Q-Learning with epsilon-greedy exploration.

    Suitable for discretised state/action spaces on small tracks.
    State and action spaces must be discretised before use.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ) -> None:
        self._n_states = n_states
        self._n_actions = n_actions
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._q_table: NDArray[np.float64] = np.zeros((n_states, n_actions))

    @property
    def name(self) -> str:
        return "q_learning"

    def act(self, obs: Observation) -> NDArray[Any]:
        raise NotImplementedError("QLearningAgent.act() is not implemented yet.")

    def learn(self, batch: Batch) -> dict[str, float]:
        raise NotImplementedError("QLearningAgent.learn() is not implemented yet.")

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError

    def reset_episode(self) -> None:
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
