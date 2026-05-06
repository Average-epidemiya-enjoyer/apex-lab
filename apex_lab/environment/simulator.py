"""Тонкая обёртка над CarModel для пошагового запуска без gymnasium."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apex_lab.environment.car import CarModel, CarParams, CarState
from apex_lab.environment.track import Track


@dataclass
class StepResult:
    """Результат одного шага симулятора.

    Attributes:
        next_state: Состояние машины после шага.
        reward: Числовой сигнал вознаграждения (опционально).
        done: True, если эпизод завершён (авария или финиш).
        info: Вспомогательная диагностическая информация.
    """

    next_state: CarState
    reward: float
    done: bool
    info: dict[str, Any]


class Simulator:
    """Детерминированный шаговый симулятор.

    Обёртка над ``CarModel`` с дополнительной логикой:
    — проверка нахождения на трассе,
    — подсчёт прогресса,
    — завершение эпизода при выезде.

    Для полноценного RL используйте ``RaceEnvironment`` (gymnasium.Env).
    """

    def __init__(
        self,
        track: Track,
        car_params: CarParams | None = None,
        dt: float = 0.05,
    ) -> None:
        self._track = track
        self._car = CarModel(car_params or CarParams())
        self._dt = dt
        self._prev_progress: float = 0.0

    def reset(self) -> CarState:
        """Поставить машину на старт и вернуть начальное состояние."""
        cl = self._track.centre_line
        start = cl[0]
        nxt = cl[1]
        import math
        heading = math.atan2(float(nxt[1] - start[1]), float(nxt[0] - start[0]))
        state = CarState(x=float(start[0]), y=float(start[1]), heading=heading)
        self._prev_progress = self._track.progress(state.x, state.y)
        return state

    def step(
        self,
        state: CarState,
        action: tuple[float, float],
    ) -> StepResult:
        """Выполнить один шаг симуляции.

        Args:
            state: Текущее состояние (не изменяется).
            action: ``(throttle, steer)`` ∈ [-1, 1].

        Returns:
            ``StepResult`` с новым состоянием и метриками.
        """
        next_state = self._car.step(state, action, self._dt)
        on_track = self._track.is_inside(next_state.x, next_state.y)
        curr_progress = self._track.progress(next_state.x, next_state.y)
        progress_delta = curr_progress - self._prev_progress
        if progress_delta < -0.5:
            progress_delta += 1.0
        self._prev_progress = curr_progress

        return StepResult(
            next_state=next_state,
            reward=float(progress_delta * 10.0),
            done=not on_track,
            info={
                "on_track": on_track,
                "progress": curr_progress,
                "speed": next_state.speed,
            },
        )
