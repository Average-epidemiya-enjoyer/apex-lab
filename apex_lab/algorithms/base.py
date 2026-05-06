"""Abstract interfaces shared by all planners and learning agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A trajectory is an (N, 2) array of (x, y) waypoints in metres.
Trajectory = NDArray[np.float64]

# A car state vector: [x, y, heading_rad, speed_m_s, ...] — exact layout
# is defined by the environment, but the first four fields are mandatory.
CarState = NDArray[np.float64]

# An observation fed to a learning agent (flattened or structured array).
Observation = NDArray[np.float32]

# A mini-batch of (obs, action, reward, next_obs, done) tuples.
Batch = dict[str, NDArray[Any]]


# ---------------------------------------------------------------------------
# BasePlanner
# ---------------------------------------------------------------------------

class BasePlanner(ABC):
    """Offline / online trajectory planner.

    A planner receives the full track description and the current car state
    and returns a trajectory (sequence of waypoints).  Classical algorithms
    (A*, RRT, MPC, …) implement this interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs and comparison tables."""

    @abstractmethod
    def plan(self, track: Any, car_state: CarState) -> Trajectory:
        """Compute a trajectory from *car_state* to the finish line.

        Args:
            track: Track object exposing geometry and constraints.
            car_state: Current state vector ``[x, y, heading, speed, ...]``.

        Returns:
            Trajectory array of shape ``(N, 2)`` with (x, y) waypoints.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state (called at the start of each episode)."""


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Online reinforcement-learning agent.

    An agent interacts with an environment step by step: it receives an
    observation, returns an action, and optionally learns from batches of
    experience.  RL algorithms (Q-Learning, DQN, PPO, …) implement this
    interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs and comparison tables."""

    @abstractmethod
    def act(self, obs: Observation) -> NDArray[Any]:
        """Select an action given the current observation.

        Args:
            obs: Observation vector from the environment.

        Returns:
            Action array whose shape matches the environment's action space.
        """

    @abstractmethod
    def learn(self, batch: Batch) -> dict[str, float]:
        """Update internal parameters from a batch of experience.

        Args:
            batch: Dictionary with keys ``obs``, ``actions``, ``rewards``,
                   ``next_obs``, ``dones`` — each an array of the same
                   leading batch dimension.

        Returns:
            Dictionary of scalar training metrics (e.g. ``{"loss": 0.42}``).
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model weights / Q-table to *path*."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore model weights / Q-table from *path*."""

    @abstractmethod
    def reset_episode(self) -> None:
        """Reset per-episode state (e.g. hidden states, eligibility traces)."""
