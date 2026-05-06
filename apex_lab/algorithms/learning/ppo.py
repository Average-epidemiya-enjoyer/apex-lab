"""Proximal Policy Optimisation (PPO) agent."""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

from apex_lab.algorithms.base import BaseAgent, Batch, Observation


class PPOAgent(BaseAgent):
    """PPO with clipped surrogate objective.

    Supports continuous action spaces, making it suitable for steering-angle
    and throttle control tasks.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
    ) -> None:
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._hidden_dim = hidden_dim
        self._lr_actor = lr_actor
        self._lr_critic = lr_critic
        self._gamma = gamma
        self._lam = lam
        self._clip_eps = clip_eps
        self._n_epochs = n_epochs
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return "ppo"

    def act(self, obs: Observation) -> NDArray[Any]:
        raise NotImplementedError("PPOAgent.act() is not implemented yet.")

    def learn(self, batch: Batch) -> dict[str, float]:
        raise NotImplementedError("PPOAgent.learn() is not implemented yet.")

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError

    def reset_episode(self) -> None:
        pass
