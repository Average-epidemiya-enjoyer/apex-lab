"""Matplotlib helpers for training and comparison plots."""

from __future__ import annotations

from typing import Any

import numpy as np


def plot_reward_curve(
    episode_rewards: list[float],
    window: int = 20,
    title: str = "Episode Reward",
    save_path: str | None = None,
) -> None:
    """Plot a smoothed reward curve with a rolling mean overlay.

    Args:
        episode_rewards: Reward per episode in order.
        window: Rolling-average window size.
        title: Plot title.
        save_path: If given, save the figure to this path instead of showing it.
    """
    raise NotImplementedError("plot_reward_curve() is not implemented yet.")


def plot_metric_comparison(
    results: dict[str, dict[str, Any]],
    metric: str,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """Bar chart comparing *metric* across multiple algorithms.

    Args:
        results: Mapping from algorithm name to its ``to_dict()`` output.
        metric: Key to extract from each result dict.
        title: Plot title; defaults to the metric name.
        save_path: If given, save the figure to this path instead of showing it.
    """
    raise NotImplementedError("plot_metric_comparison() is not implemented yet.")
