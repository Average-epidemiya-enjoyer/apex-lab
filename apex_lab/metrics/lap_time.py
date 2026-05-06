"""Lap-time metric."""

from __future__ import annotations

from typing import Any

from apex_lab.metrics.base import BaseMetric


class LapTimeMetric(BaseMetric):
    """Records lap times across episodes and computes mean / best."""

    def __init__(self) -> None:
        self._times: list[float] = []

    @property
    def name(self) -> str:
        return "lap_time"

    def update(self, **kwargs: Any) -> None:
        """Expects keyword argument ``lap_time`` (float, seconds)."""
        lap_time: float = kwargs["lap_time"]
        self._times.append(lap_time)

    def compute(self) -> dict[str, float]:
        if not self._times:
            return {"mean": float("nan"), "best": float("nan")}
        import statistics
        return {
            "mean": statistics.mean(self._times),
            "best": min(self._times),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "times": self._times, **self.compute()}

    def reset(self) -> None:
        self._times = []
