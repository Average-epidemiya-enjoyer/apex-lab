"""Deep Q-Network (DQN) agent."""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

from apex_lab.algorithms.base import BaseAgent, Batch, Observation


class DQNAgent(BaseAgent):
    """DQN with experience replay and target network.

    Action space must be discrete.  Observation can be a raw pixel frame
    (CNN backbone) or a feature vector (MLP backbone).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 1_000,
    ) -> None:
        self._obs_dim = obs_dim
        self._n_actions = n_actions
        self._hidden_dim = hidden_dim
        self._lr = lr
        self._gamma = gamma
        self._batch_size = batch_size
        self._buffer_capacity = buffer_capacity
        self._target_update_freq = target_update_freq

    @property
    def name(self) -> str:
        return "dqn"

    def act(self, obs: Observation) -> NDArray[Any]:
        raise NotImplementedError("DQNAgent.act() is not implemented yet.")

    def learn(self, batch: Batch) -> dict[str, float]:
        raise NotImplementedError("DQNAgent.learn() is not implemented yet.")

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError

    def reset_episode(self) -> None:
        pass
