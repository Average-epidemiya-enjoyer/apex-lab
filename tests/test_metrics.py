"""Unit tests for concrete metric implementations."""

import numpy as np
import pytest

from apex_lab.metrics.lap_time import LapTimeMetric
from apex_lab.metrics.path_length import PathLengthMetric
from apex_lab.metrics.smoothness import SmoothnessMetric
from apex_lab.metrics.reward_curve import RewardCurveMetric


def test_lap_time_mean_and_best() -> None:
    m = LapTimeMetric()
    m.update(lap_time=10.0)
    m.update(lap_time=8.0)
    result = m.compute()
    assert result["mean"] == pytest.approx(9.0)
    assert result["best"] == pytest.approx(8.0)


def test_lap_time_reset() -> None:
    m = LapTimeMetric()
    m.update(lap_time=5.0)
    m.reset()
    result = m.compute()
    assert result["mean"] != result["mean"]  # NaN


def test_path_length_straight_line() -> None:
    m = PathLengthMetric()
    traj = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    m.update(trajectory=traj)
    assert m.compute() == pytest.approx(3.0)


def test_smoothness_straight_line_is_zero() -> None:
    m = SmoothnessMetric()
    traj = np.column_stack([np.linspace(0, 10, 50), np.zeros(50)])
    m.update(trajectory=traj)
    assert m.compute() == pytest.approx(0.0, abs=1e-9)


def test_reward_curve_accumulates() -> None:
    m = RewardCurveMetric()
    for r in [1.0, 2.0, 3.0]:
        m.update(episode_reward=r)
    result = m.compute()
    assert result["mean"] == pytest.approx(2.0)
    assert result["max"] == pytest.approx(3.0)
