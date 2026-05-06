"""Trajectory smoothness metric (mean absolute curvature)."""

from __future__ import annotations

from typing import Any

import numpy as np

from apex_lab.metrics.base import BaseMetric


class SmoothnessMetric(BaseMetric):
    """Mean absolute curvature of the trajectory (lower = smoother)."""

    def __init__(self) -> None:
        self._scores: list[float] = []

    @property
    def name(self) -> str:
        return "smoothness"

    def update(self, **kwargs: Any) -> None:
        """Expects keyword argument ``trajectory`` — an ``(N, 2)`` array."""
        traj = np.asarray(kwargs["trajectory"], dtype=float)
        if len(traj) < 3:
            return
        d1 = np.diff(traj, axis=0)
        d2 = np.diff(d1, axis=0)
        cross = d1[:-1, 0] * d2[:, 1] - d1[:-1, 1] * d2[:, 0]
        norm_cubed = np.linalg.norm(d1[:-1], axis=1) ** 3
        curvature = np.abs(cross) / np.where(norm_cubed > 1e-9, norm_cubed, 1e-9)
        self._scores.append(float(curvature.mean()))

    def compute(self) -> float:
        return float(np.mean(self._scores)) if self._scores else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "mean_curvature": self.compute()}

    def reset(self) -> None:
        self._scores = []
