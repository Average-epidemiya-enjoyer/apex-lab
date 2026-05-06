"""Abstract interface for experiment metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """Accumulates per-step or per-episode data and produces a summary.

    Concrete metrics (lap time, path length, smoothness, reward curve, …)
    inherit from this class and are registered with the experiment runner.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, used as dict key and plot label."""

    @abstractmethod
    def update(self, **kwargs: Any) -> None:
        """Ingest one data point (step or episode result).

        Keyword arguments are metric-specific (e.g. ``lap_time=12.3``).
        """

    @abstractmethod
    def compute(self) -> float | dict[str, float]:
        """Return the aggregated metric value(s).

        Returns either a single scalar or a named dict for multi-valued
        metrics (e.g. mean ± std).
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialise all accumulated state to a JSON-compatible dictionary."""

    def reset(self) -> None:
        """Clear accumulated state (called between experiments)."""
