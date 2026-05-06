"""Smoke tests: verify abstract interfaces cannot be instantiated directly."""

import pytest

from apex_lab.algorithms.base import BasePlanner, BaseAgent
from apex_lab.metrics.base import BaseMetric


def test_base_planner_is_abstract() -> None:
    with pytest.raises(TypeError):
        BasePlanner()  # type: ignore[abstract]


def test_base_agent_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseAgent()  # type: ignore[abstract]


def test_base_metric_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseMetric()  # type: ignore[abstract]
