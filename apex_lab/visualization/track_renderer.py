"""Pygame-based real-time track and trajectory renderer."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class TrackRenderer:
    """Renders the track and one or more trajectories using Pygame.

    Usage::

        renderer = TrackRenderer(track, scale=4.0)
        renderer.open()
        renderer.draw(trajectories={"a_star": traj_array})
        renderer.close()
    """

    def __init__(
        self,
        track: Any,
        scale: float = 4.0,
        window_title: str = "apex-lab",
        fps: int = 60,
    ) -> None:
        self._track = track
        self._scale = scale
        self._window_title = window_title
        self._fps = fps
        self._surface: Any = None

    def open(self) -> None:
        """Initialise Pygame and open the render window."""
        raise NotImplementedError("TrackRenderer.open() is not implemented yet.")

    def draw(
        self,
        car_state: Any | None = None,
        trajectories: dict[str, NDArray[np.float64]] | None = None,
    ) -> None:
        """Redraw the frame.

        Args:
            car_state: Current car state (draws a marker if provided).
            trajectories: Named trajectories to overlay on the track.
        """
        raise NotImplementedError("TrackRenderer.draw() is not implemented yet.")

    def close(self) -> None:
        """Destroy the window and shut down Pygame."""
        raise NotImplementedError("TrackRenderer.close() is not implemented yet.")
