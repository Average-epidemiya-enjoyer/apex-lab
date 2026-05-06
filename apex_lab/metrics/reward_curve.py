"""Episode reward curve metric (for RL agents)."""

from __future__ import annotations

from typing import Any

import numpy as np

from apex_lab.metrics.base import BaseMetric


class RewardCurveMetric(BaseMetric):
    """Tracks cumulative reward per episode for RL training curves."""

    def __init__(self) -> None:
        self._episode_rewards: list[float] = []

    @property
    def name(self) -> str:
        return "reward_curve"

    def update(self, **kwargs: Any) -> None:
        """Expects keyword argument ``episode_reward`` (float)."""
        self._episode_rewards.append(float(kwargs["episode_reward"]))

    def compute(self) -> dict[str, float]:
        if not self._episode_rewards:
            return {"mean": float("nan"), "max": float("nan")}
        arr = np.asarray(self._episode_rewards)
        return {"mean": float(arr.mean()), "max": float(arr.max())}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "episode_rewards": self._episode_rewards,
            **self.compute(),
        }

    def reset(self) -> None:
        self._episode_rewards = []
