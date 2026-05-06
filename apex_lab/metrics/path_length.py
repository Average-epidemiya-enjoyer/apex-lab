"""Path-length metric."""

from __future__ import annotations

from typing import Any

import numpy as np

from apex_lab.metrics.base import BaseMetric


class PathLengthMetric(BaseMetric):
    """Total arc length of the driven trajectory in metres."""

    def __init__(self) -> None:
        self._lengths: list[float] = []

    @property
    def name(self) -> str:
        return "path_length"

    def update(self, **kwargs: Any) -> None:
        """Expects keyword argument ``trajectory`` — an ``(N, 2)`` array."""
        trajectory = np.asarray(kwargs["trajectory"])
        deltas = np.diff(trajectory, axis=0)
        length = float(np.linalg.norm(deltas, axis=1).sum())
        self._lengths.append(length)

    def compute(self) -> float:
        return float(np.mean(self._lengths)) if self._lengths else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "mean_length_m": self.compute()}

    def reset(self) -> None:
        self._lengths = []
